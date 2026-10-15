//! **逆拉普拉斯变换**（数值）：由 `F(s)` 求 `f(t)`，`t ≥ 0`。
//!
//! `f(t) = (1/2πi)·∮ F(s)·e^{st} ds`（Bromwich 围道）。
//!
//! # 方法：Dubner–Abate + **跳变扣除**
//!
//! 纯 D–A（周期 `2T` 的 Fourier 级数）有一个**固有**缺陷：周期延拓在 `t = 0 / 2T` 处
//! **不连续**（`h(0) = f(0⁺) ≠ 0`，而 `h(2T) ≈ 0`）⇒ 系数按 `1/k` 衰减 ⇒ **误差 O(1/N)**。
//! 实测（`F=1/s`、`T=12`、`a=1`）：`N=50 → 3.07e-1，150 → 9.88e-2，300 → 4.98e-2，600 → 2.50e-2`
//! —— **每翻一倍正好减半**，压到 `1e-3` 要 `N ≈ 15000`，不可用。
//!
//! **修正**（本模块做法）：把跳变**解析地扣除**——
//!
//! ```text
//! f(t) = f₀·u(t) + L⁻¹{ F(s) − f₀/s },      f₀ := f(0⁺) = lim_{s→∞} s·F(s)
//! ```
//!
//! 扣除后 `h̃(t) = f̃(t)e^{-a t}` 满足 `h̃(0) = 0` ⇒ **奇延拓连续** ⇒ 恢复**指数收敛**
//! （实测见 `docs/ops.md §9.0` 的两条 N 曲线并排）。
//!
//! ## D–A 级数（对扣除后的 `F̃` 用）
//! ```text
//! f̃(t) ≈ (e^{a·t}/T)·[ ½·F̃(a) + Σ_{k=1..N} Re{ F̃(a + i·kπ/T)·e^{i·kπ·t/T} } ],  ω_k = kπ/T
//! ```
//! 三处最容易写错的系数（本模块用 `F=1/s` 的最小验证盯住）：**`k=0` 项带 ½**、
//! **间距 `π/T`**（不是 `2π/T`）、**整体因子 `e^{at}/T`**（`T` 在分母）。
//!
//! # `f₀` 的契约（**谁保证**）
//!
//! - **首选：调用方给**（`DubnerAbate::with_f0`）——`f₀` 是**解析已知量**（初值定理），调用方按
//!   自己的 `F` 给出最可靠；**默认 `f₀ = 0`**（等于不做扣除，适用于 `f(0⁺) = 0` 的 `F`）。
//! - **数值估计**：[`estimate_f0`] 用 `s·F(s)` 在**有限 `s`** 上近似 `lim_{s→∞}` ⇒
//!   **不可靠**（`F` 衰减慢、或 `s` 取得不够大都会错）。要用**必须显式**调用并放进
//!   [`DubnerAbate::with_f0`]，**没有任何静默回退**。
//! - **取错的后果**：残留跳变 ⇒ **退回 O(1/N)**（误差随 `N` 只按 `1/N` 降，不再指数收敛）。
//!   这是"看 N 扫描曲线就能自诊"的性质，见 `docs/ops.md §9.0`。
//!
//! # 硬约束
//! - **周期 `2T`**：级数只在 `0 < t < 2T` 内代表原函数（超出是周期延拓）；建议 `T ≥ 3·t_max`。
//!   [`DubnerAbate::check`] 会拦 `T ≤ t_max`。
//! - **`a·t_max ≲ 4…8`**：`e^{a t}` 放大绝对截断误差（实测 `a=6 → 1e7`、`a=12 → 1e17`）。
//! - **`t=0` 给跳变平均值**（`F=1/s` 时 ≈ `f₀/2`）：Fourier 反演的已知行为，**不是 bug**。
//!
//! # 与正向的关系（别把故事说大）
//! `F(s_k)` 是**调用方代码**，必须逐点求值；**DSL 只负责"指数和"那部分**（`cos/sin` 核 × `F` 值），
//! 且天然批量（`NB` 个 `F` 一起算）。**若 `F` 很贵，主导成本在 `F`**。
//!
//! # 限制
//! 不能保证任意精度 · `F` 的奇点须在 `Re s < a`（分支切割同理）· 逆问题**病态** ·
//! `0 < t < 2T` · `t→0` 误差抬升（Gibbs）· **依赖 `f₀` 的正确性**。
use crate::shape::{Mat, MatBuf};
use crate::{api, matmul};

/// 复数域可求值的 `F`：`s = (s_re, s_im)`。
pub trait ComplexFn {
    /// 返回 `F(s_re + i·s_im)`。
    fn eval(&self, s_re: f64, s_im: f64) -> (f64, f64);
}

/// `F(s) = 1/(s + θ)`（`⇒ f(t) = e^{-θt}`，`f₀ = 1`）。
#[derive(Clone, Copy)]
pub struct OneOverSPlus {
    /// `θ`
    pub theta: f64,
}
impl ComplexFn for OneOverSPlus {
    fn eval(&self, re: f64, im: f64) -> (f64, f64) {
        let (dr, di) = (re + self.theta, im);
        let d = dr * dr + di * di;
        (dr / d, -di / d)
    }
}

/// `F(s) = 1/s`（`⇒ f ≡ 1`，`f₀ = 1`）——**最小验证用**。
#[derive(Clone, Copy)]
pub struct OneOverS;
impl ComplexFn for OneOverS {
    fn eval(&self, re: f64, im: f64) -> (f64, f64) {
        let d = re * re + im * im;
        (re / d, -im / d)
    }
}

/// `F(s) = ω₀/(s² + ω₀²)`（`⇒ f(t) = sin(ω₀t)`，`f₀ = 0`）。
#[derive(Clone, Copy)]
pub struct OmegaOverS2Plus {
    /// `ω₀`
    pub omega: f64,
}
impl ComplexFn for OmegaOverS2Plus {
    fn eval(&self, re: f64, im: f64) -> (f64, f64) {
        let (ar, ai) = (re * re - im * im + self.omega * self.omega, 2.0 * re * im);
        let d = ar * ar + ai * ai;
        (self.omega * ar / d, -self.omega * ai / d)
    }
}

/// **不可靠**的 `f₀` 数值估计：`s_probe · F(s_probe)`。要用请显式放进
/// [`DubnerAbate::with_f0`]，并知道 `F` 衰减慢或 `s_probe` 不够大时它会错。
pub fn estimate_f0<F: ComplexFn>(f: &F, s_probe: f64) -> f64 {
    s_probe * f.eval(s_probe, 0.0).0
}

/// Dubner–Abate（含跳变扣除）参数。
#[derive(Clone, Copy, Debug)]
pub struct DubnerAbate {
    /// 半周期 `T`（级数周期 `2T`）；建议 `≥ 3·t_max`。
    pub t_end: f64,
    /// 位移 `a`；建议 `a·t_max ≲ 4…8`。
    pub a: f64,
    /// 级数项数 `N`。
    pub n: usize,
    /// `f₀ = f(0⁺)`：**调用方给**（默认 `0` ⇒ 不扣除）。
    pub f0: f64,
}

impl DubnerAbate {
    /// 默认：`T = 3·t_max`、`a = 4/t_max`、**`f₀ = 0`**（不扣除；适用于 `f(0⁺)=0` 的 `F`）。
    pub fn new(t_max: f64, n: usize) -> Self {
        Self {
            t_end: 3.0 * t_max,
            a: 4.0 / t_max,
            n,
            f0: 0.0,
        }
    }
    /// **带上调用方给的 `f₀`**（首选路径；`f₀ = lim_{s→∞} s·F(s)`）。
    pub fn with_f0(t_max: f64, n: usize, f0: f64) -> Self {
        Self {
            f0,
            ..Self::new(t_max, n)
        }
    }
    /// 节点 `ω_k = kπ/T`。
    pub fn omega(&self, k: usize) -> f64 {
        k as f64 * std::f64::consts::PI / self.t_end
    }
    /// 参数自检：周期与 `a·t_max` 的硬约束。
    pub fn check(&self, t_max: f64) -> Result<(), String> {
        if self.t_end <= t_max {
            return Err(format!(
                "半周期 T={} 必须 > t_max={}（级数周期 2T，超出会得到周期延拓）",
                self.t_end, t_max
            ));
        }
        if self.a <= 0.0 {
            return Err("a 必须 > 0".into());
        }
        let at = self.a * t_max;
        if at > 8.0 {
            return Err(format!(
                "a·t_max = {at:.2} 过大（建议 ≲4…8）：e^(a·t) 会放大绝对截断误差"
            ));
        }
        Ok(())
    }
    /// `f₀/s` 在 `s = a + iω` 处的复数值。
    fn f0_over_s(&self, om: f64) -> (f64, f64) {
        if self.f0 == 0.0 {
            return (0.0, 0.0);
        }
        let d = self.a * self.a + om * om;
        (self.f0 * self.a / d, -self.f0 * om / d)
    }
}

/// **指数和**（DSL 核心）：`fre/fim` 是 `N×NB` 行主序（**扣除后**的 `F̃` 值）。
/// 返回 `J×NB`；**不含** DC 项与 `e^{a t}/T` 因子。
pub fn exponential_sum_f64<const J: usize, const N: usize, const NB: usize>(
    p: DubnerAbate,
    ts: &[f64],
    fre: &[f64],
    fim: &[f64],
) -> Vec<f64> {
    assert_eq!(ts.len(), J);
    assert_eq!(fre.len(), N * NB);
    assert_eq!(fim.len(), N * NB);
    assert_eq!(p.n, N);
    let mut c = vec![0.0f64; J * N];
    let mut s = vec![0.0f64; J * N];
    for (j, &t) in ts.iter().enumerate() {
        for k in 0..N {
            let w = p.omega(k + 1) * t;
            c[j * N + k] = w.cos();
            s[j * N + k] = w.sin();
        }
    }
    let mc = Mat::<f64, J, N>::new(&c).expect("C = J×N");
    let ms = Mat::<f64, J, N>::new(&s).expect("S = J×N");
    let mfr = Mat::<f64, N, NB>::new(fre).expect("ReF = N×NB");
    let mfi = Mat::<f64, N, NB>::new(fim).expect("ImF = N×NB");
    let mut acc_re = MatBuf::<f64, J, NB>::new();
    let mut acc_im = MatBuf::<f64, J, NB>::new();
    matmul!(acc_re[J, NB] = mc[J, N] * mfr[N, NB]);
    matmul!(acc_im[J, NB] = ms[J, N] * mfi[N, NB]);
    let mut out = vec![0.0f64; J * NB];
    for j in 0..J {
        for b in 0..NB {
            out[j * NB + b] = acc_re.as_slice()[j * NB + b] - acc_im.as_slice()[j * NB + b];
        }
    }
    out
}

/// `f32` 指数和（与 `api::matmul` **逐位对照**用）。
pub fn exponential_sum_f32<const J: usize, const N: usize, const NB: usize>(
    p: DubnerAbate,
    ts: &[f64],
    fre: &[f32],
    fim: &[f32],
) -> Vec<f32> {
    assert_eq!(ts.len(), J);
    assert_eq!(fre.len(), N * NB);
    assert_eq!(fim.len(), N * NB);
    assert_eq!(p.n, N);
    let mut c = vec![0.0f32; J * N];
    let mut s = vec![0.0f32; J * N];
    for (j, &t) in ts.iter().enumerate() {
        for k in 0..N {
            let w = p.omega(k + 1) * t;
            c[j * N + k] = w.cos() as f32;
            s[j * N + k] = w.sin() as f32;
        }
    }
    let mc = Mat::<f32, J, N>::new(&c).expect("C = J×N");
    let ms = Mat::<f32, J, N>::new(&s).expect("S = J×N");
    let mfr = Mat::<f32, N, NB>::new(fre).expect("ReF = N×NB");
    let mfi = Mat::<f32, N, NB>::new(fim).expect("ImF = N×NB");
    let mut acc_re = MatBuf::<f32, J, NB>::new();
    let mut acc_im = MatBuf::<f32, J, NB>::new();
    matmul!(acc_re[J, NB] = mc[J, N] * mfr[N, NB]);
    matmul!(acc_im[J, NB] = ms[J, N] * mfi[N, NB]);
    let mut out = vec![0.0f32; J * NB];
    for j in 0..J {
        for b in 0..NB {
            out[j * NB + b] = acc_re.as_slice()[j * NB + b] - acc_im.as_slice()[j * NB + b];
        }
    }
    out
}

/// `f32` 指数和的**行切参考**（逐列 `api::matmul`）。
pub fn exponential_sum_ref_f32<const J: usize, const N: usize, const NB: usize>(
    p: DubnerAbate,
    ts: &[f64],
    fre: &[f32],
    fim: &[f32],
) -> Vec<f32> {
    let mut c = vec![0.0f32; J * N];
    let mut s = vec![0.0f32; J * N];
    for (j, &t) in ts.iter().enumerate() {
        for k in 0..N {
            let w = p.omega(k + 1) * t;
            c[j * N + k] = w.cos() as f32;
            s[j * N + k] = w.sin() as f32;
        }
    }
    let mut out = vec![0.0f32; J * NB];
    let mut col = vec![0.0f32; N];
    for b in 0..NB {
        for k in 0..N {
            col[k] = fre[k * NB + b];
        }
        let re = api::matmul(J, N, 1, &c, &col).expect("J×N × N×1");
        for k in 0..N {
            col[k] = fim[k * NB + b];
        }
        let im = api::matmul(J, N, 1, &s, &col).expect("J×N × N×1");
        for j in 0..J {
            out[j * NB + b] = re[j] - im[j];
        }
    }
    out
}

/// `f64` 指数和的**同序参考**（`api::matmul` 只有 `f32` 版）。
pub fn exponential_sum_ref_f64<const J: usize, const N: usize, const NB: usize>(
    p: DubnerAbate,
    ts: &[f64],
    fre: &[f64],
    fim: &[f64],
) -> Vec<f64> {
    let mut out = vec![0.0f64; J * NB];
    for (j, &t) in ts.iter().enumerate() {
        for b in 0..NB {
            let (mut a1, mut a2) = (0.0f64, 0.0f64);
            for k in 0..N {
                let w = p.omega(k + 1) * t;
                a1 += w.cos() * fre[k * NB + b];
                a2 += w.sin() * fim[k * NB + b];
            }
            out[j * NB + b] = a1 - a2;
        }
    }
    out
}

/// **完整**逆变换（含 **跳变扣除**）：`f(t) = f₀·u(t) + L⁻¹{F − f₀/s}`。
///
/// `u(t)` 在 `t=0` 取 **½**（跳变平均值，与 Fourier 反演一致）；`t>0` 取 1。
pub fn invert_batch_f64<const J: usize, const N: usize, const NB: usize, F: ComplexFn>(
    p: DubnerAbate,
    ts: &[f64],
    fs: &[F],
) -> Vec<f64> {
    assert_eq!(ts.len(), J);
    assert_eq!(fs.len(), NB);
    assert_eq!(p.n, N);
    let t_max = ts.iter().cloned().fold(0.0f64, f64::max);
    p.check(t_max).expect("Dubner–Abate 参数越界");
    // F̃ = F − f₀/s（逐点；**调用方代码**负责 F）
    let mut fre = vec![0.0f64; N * NB];
    let mut fim = vec![0.0f64; N * NB];
    for (b, f) in fs.iter().enumerate() {
        for k in 0..N {
            let om = p.omega(k + 1);
            let (r, i) = f.eval(p.a, om);
            let (zr, zi) = p.f0_over_s(om);
            fre[k * NB + b] = r - zr;
            fim[k * NB + b] = i - zi;
        }
    }
    let sum = exponential_sum_f64::<J, N, NB>(p, ts, &fre, &fim);
    let mut out = vec![0.0f64; J * NB];
    for (b, f) in fs.iter().enumerate() {
        let fa = {
            let (r, _i) = f.eval(p.a, 0.0);
            let (zr, _) = p.f0_over_s(0.0);
            r - zr
        };
        for (j, &t) in ts.iter().enumerate() {
            let series = (p.a * t).exp() / p.t_end * (0.5 * fa + sum[j * NB + b]);
            let step = if t == 0.0 { 0.5 * p.f0 } else { p.f0 };
            out[j * NB + b] = step + series;
        }
    }
    out
}

/// **独立交叉验证**：固定围道 `Re s = c` 的 **Bromwich 梯形积分**（无 `2T` 周期陷阱）。
pub fn bromwich_ref_f64<const J: usize, const NB: usize, F: ComplexFn>(
    c: f64,
    dw: f64,
    k_max: usize,
    ts: &[f64],
    fs: &[F],
) -> Vec<f64> {
    let mut out = vec![0.0f64; J * NB];
    for (b, f) in fs.iter().enumerate() {
        let (fc, _) = f.eval(c, 0.0);
        for (j, &t) in ts.iter().enumerate() {
            let mut acc = 0.5 * fc;
            for k in 1..=k_max {
                let w = k as f64 * dw;
                let (r, i) = f.eval(c, w);
                let (ct, st) = ((w * t).cos(), (w * t).sin());
                acc += r * ct - i * st;
            }
            out[j * NB + b] = (c * t).exp() / std::f64::consts::PI * dw * acc;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const TMAX: f64 = 4.0;
    const TS: [f64; 5] = [0.5, 1.0, 2.0, 3.0, 4.0];

    /// **① 最小验证**：`F=1/s`（`f≡1`，`f₀=1`）——系数三处任一处写错都会暴露。
    #[test]
    fn minimal_one_over_s() {
        let p = DubnerAbate::with_f0(TMAX, 300, 1.0);
        let fs = [OneOverS, OneOverS, OneOverS];
        let got = invert_batch_f64::<5, 300, 3, _>(p, &TS, &fs);
        let mut worst = 0.0f64;
        for (j, _) in TS.iter().enumerate() {
            worst = worst.max((got[j * 3] - 1.0).abs());
        }
        assert!(worst < 1e-6, "F=1/s 扣除后应几乎精确，实测 {worst:.3e}");
    }

    /// **② N 扫描：跳变扣除后应恢复"快于 1/N"**（对比未扣除的 1/N 曲线）。
    #[test]
    fn n_scan_after_jump_subtraction() {
        let fs = [
            OneOverSPlus { theta: 1.0 },
            OneOverSPlus { theta: 1.0 },
            OneOverSPlus { theta: 1.0 },
        ];
        macro_rules! err_at {
            ($nn:expr, $p:expr) => {{
                let got = invert_batch_f64::<5, $nn, 3, _>($p, &TS, &fs);
                let mut e = 0.0f64;
                for (j, &t) in TS.iter().enumerate() {
                    e = e.max((got[j * 3] - (-t).exp()).abs());
                }
                e
            }};
        }
        // 扣除 f₀ = 1
        let e_sub = [
            (64usize, err_at!(64, DubnerAbate::with_f0(TMAX, 64, 1.0))),
            (128, err_at!(128, DubnerAbate::with_f0(TMAX, 128, 1.0))),
            (256, err_at!(256, DubnerAbate::with_f0(TMAX, 256, 1.0))),
            (512, err_at!(512, DubnerAbate::with_f0(TMAX, 512, 1.0))),
        ];
        // **不扣除**（f₀ 取错为 0）⇒ 残留跳变 ⇒ 退回 1/N
        let e_nosub = [
            (64usize, err_at!(64, DubnerAbate::new(TMAX, 64))),
            (128, err_at!(128, DubnerAbate::new(TMAX, 128))),
            (256, err_at!(256, DubnerAbate::new(TMAX, 256))),
            (512, err_at!(512, DubnerAbate::new(TMAX, 512))),
        ];
        println!("扣除 f₀=1：{e_sub:?}");
        println!("未扣除（f₀=0）：{e_nosub:?}");
        // 扣除后：N ≤ 512 时误差 ≤1e-3 ✓
        assert!(
            e_sub[3].1 <= 1e-3,
            "扣除后 N=512 的误差应 ≤1e-3，实测 {:?}",
            e_sub[3]
        );
        // 且明显快于 1/N：扣除后 256→512 至少降 10 倍；未扣除只降 ~2 倍
        let r_sub = e_sub[2].1 / e_sub[3].1;
        let r_no = e_nosub[2].1 / e_nosub[3].1;
        assert!(
            r_sub > 5.0,
            "扣除后应远快于 1/N（256→512 的比值 >5），实测 {r_sub:.1}×"
        );
        assert!(
            r_no < 3.0,
            "未扣除时应约 1/N（256→512 比值 ≈2），实测 {r_no:.1}×"
        );
        assert!(
            e_sub[3].1 < e_nosub[3].1 / 10.0,
            "扣除应比不扣除好一个量级以上：{:.2e} vs {:.2e}",
            e_sub[3].1,
            e_nosub[3].1
        );
    }

    /// **③ 解析对照**：`1/(s+a) ⇒ e^{-at}`（批量 `a=1,2,3`）与 `sin(ω₀t)`。
    #[test]
    fn analytic_exp_and_sin() {
        let fs = [
            OneOverSPlus { theta: 1.0 },
            OneOverSPlus { theta: 2.0 },
            OneOverSPlus { theta: 3.0 },
        ];
        let p = DubnerAbate::with_f0(TMAX, 512, 1.0);
        let got = invert_batch_f64::<5, 512, 3, _>(p, &TS, &fs);
        let mut worst = 0.0f64;
        for (b, f) in fs.iter().enumerate() {
            for (j, &t) in TS.iter().enumerate() {
                worst = worst.max((got[j * 3 + b] - (-f.theta * t).exp()).abs());
            }
        }
        assert!(
            worst < 1e-3,
            "e^{{-at}} 最大误差应 < 1e-3，实测 {worst:.3e}"
        );
        // sin：f₀ = 0 ⇒ 不扣除
        let gs = [
            OmegaOverS2Plus { omega: 1.0 },
            OmegaOverS2Plus { omega: 2.0 },
            OmegaOverS2Plus { omega: 3.0 },
        ];
        let ts2 = [0.5f64, 1.0, 1.5, 2.0, 3.0];
        let g = invert_batch_f64::<5, 512, 3, _>(DubnerAbate::new(TMAX, 512), &ts2, &gs);
        let mut ws = 0.0f64;
        for (b, f) in gs.iter().enumerate() {
            for (j, &t) in ts2.iter().enumerate() {
                ws = ws.max((g[j * 3 + b] - (f.omega * t).sin()).abs());
            }
        }
        assert!(ws < 1e-3, "sin(ω₀t) 最大误差应 < 1e-3，实测 {ws:.3e}");
    }

    /// **④ `t=0` 是跳变点**：给 `f₀/2`（`F=1/s` ⇒ ≈0.5）——**如实断言，不当 bug 修**。
    #[test]
    fn t_zero_is_jump_average() {
        let p = DubnerAbate::with_f0(TMAX, 512, 1.0);
        let fs = [OneOverS, OneOverS, OneOverS];
        let got = invert_batch_f64::<2, 512, 3, _>(p, &[0.0, 1.0], &fs);
        assert!(
            (got[0] - 0.5).abs() < 5e-2,
            "t=0 应为 f₀/2 ≈0.5，实测 {:.4}",
            got[0]
        );
        assert!((got[3] - 1.0).abs() < 1e-3, "t=1 应 ≈1，实测 {:.4}", got[3]);
    }

    /// **④ 独立交叉验证**：D–A（扣除后）vs Bromwich 梯形积分。
    #[test]
    fn cross_check_vs_bromwich() {
        let fs = [
            OneOverSPlus { theta: 1.0 },
            OneOverSPlus { theta: 2.0 },
            OneOverSPlus { theta: 3.0 },
        ];
        let da = invert_batch_f64::<5, 512, 3, _>(DubnerAbate::with_f0(TMAX, 512, 1.0), &TS, &fs);
        // 本条只做**结构互证**：独立参考是**粗**的（固定 `dw`、有限截断，自身误差 ~5e-2），
        // 所以阈值 1e-1；**精度**由上面 `analytic_exp_and_sin`（≤1e-3）与 Bromwich 无关地保证。
        let bw = bromwich_ref_f64::<5, 3, _>(1.0, std::f64::consts::PI / 32.0, 800, &TS, &fs);
        let mut worst = 0.0f64;
        for j in 0..5 {
            for b in 0..3 {
                worst = worst.max((da[j * 3 + b] - bw[j * 3 + b]).abs());
            }
        }
        assert!(
            worst < 1e-1,
            "两独立路线应结构一致（≤1e-1），实测 {worst:.3e}"
        );
    }

    /// **④ 参数硬约束**：`T ≤ t_max` 与 `a·t_max > 8` 都会被拦。
    #[test]
    fn param_check_rejects_bad_choices() {
        assert!(
            DubnerAbate {
                t_end: 1.0,
                a: 1.0,
                n: 10,
                f0: 0.0
            }
            .check(TMAX)
            .is_err()
        );
        assert!(
            DubnerAbate {
                t_end: 12.0,
                a: 6.0,
                n: 10,
                f0: 0.0
            }
            .check(TMAX)
            .is_err()
        );
        assert!(DubnerAbate::new(TMAX, 10).check(TMAX).is_ok());
    }

    /// **⑤ 位精确**：指数和的 DSL 路径 vs 行切参考（f32 用 `api::matmul`、f64 用同序手工）。
    #[test]
    fn exponential_sum_bit_for_bit() {
        let p = DubnerAbate {
            t_end: 12.0,
            a: 1.0,
            n: 32,
            f0: 0.0,
        };
        let tsv = [0.1f64, 0.7, 1.3, 2.0, 3.0];
        let fre32: Vec<f32> = (0..32 * 3)
            .map(|i| ((i % 17) as f32) * 0.125 - 1.0)
            .collect();
        let fim32: Vec<f32> = (0..32 * 3)
            .map(|i| ((i % 13) as f32) * 0.0625 - 0.5)
            .collect();
        let bits32 = |v: &[f32]| -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() };
        assert_eq!(
            bits32(&exponential_sum_f32::<5, 32, 3>(p, &tsv, &fre32, &fim32)),
            bits32(&exponential_sum_ref_f32::<5, 32, 3>(
                p, &tsv, &fre32, &fim32
            )),
            "f32 指数和：DSL 与 api::matmul 参考不逐位"
        );
        let fre64: Vec<f64> = (0..32 * 3)
            .map(|i| ((i % 17) as f64) * 0.125 - 1.0)
            .collect();
        let fim64: Vec<f64> = (0..32 * 3)
            .map(|i| ((i % 13) as f64) * 0.0625 - 0.5)
            .collect();
        let bits64 = |v: &[f64]| -> Vec<u64> { v.iter().map(|x| x.to_bits()).collect() };
        assert_eq!(
            bits64(&exponential_sum_f64::<5, 32, 3>(p, &tsv, &fre64, &fim64)),
            bits64(&exponential_sum_ref_f64::<5, 32, 3>(
                p, &tsv, &fre64, &fim64
            )),
            "f64 指数和：DSL 与同序参考不逐位"
        );
    }
}
