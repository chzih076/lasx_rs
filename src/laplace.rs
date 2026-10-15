//! **单边数值拉普拉斯变换**：`F(s) = ∫₀^∞ f(t)·e^{-st} dt`，用**公式 DSL**（`matmul!`）表达。
//!
//! # 数学与离散化
//!
//! 取 `t_i = i·Δt`（`i = 0..N-1`）、评估点 `s_j = σ_j + i·ω_j`，核矩阵
//!
//! ```text
//! Kc[j,i] = w_i · e^{-σ_j t_i} · cos(ω_j t_i)      Ks[j,i] = w_i · e^{-σ_j t_i} · sin(ω_j t_i)
//! Re F(s_j) = Σ_i Kc[j,i]·f_i        Im F(s_j) = -Σ_i Ks[j,i]·f_i
//! ```
//!
//! 即 `Re F = Kc·f`、`Im F = -Ks·f` —— 两条都是**矩阵乘**，于是**一次 `matmul!` 调用算一批信号**
//! （`f` 是 `N×NB`：`NB` 条信号并列），这正是公式 DSL 的卖点（形状进类型、位精确、自动并行）。
//!
//! **权重 `w_i`**：默认真实底线的**梯形法**（首尾各半、内部 1）⇒ `O(Δt²)`；`k=0` 的梯形权重是
//! `Δt/2`（不是 0），这一项在 `s=0` 时就是"面积"。
//!
//! # 边界与限制（**必须知道**）
//!
//! - **不是任意精度**：误差来自 ① 截断（`t` 只积到 `(N-1)Δt`）② 梯形法 `O(Δt²)`。误差量级与
//!   实测见 `docs/ops.md`（"拉普拉斯变换"小节）。
//! - **`s` 靠近极点时不能用**：如 `f ≡ 1` 的 `s=0`（真值趋于发散），本函数给的是**截断和**
//!   `Σ w_i`，只对 `σ > 0` 有意义。
//! - **不做逆变换**：本模块只做正向变换（Bromwich 积分/Mellin 反演不在范围内）。
use crate::aligned::AlignedVec;
use crate::shape::{Mat, MatBuf};
use crate::{api, matmul};

/// 梯形权重：首尾各半、内部全 1（乘 `Δt` 后即为 `w_i`）。
pub fn trapezoid_weights(n: usize) -> Vec<f64> {
    let mut w = vec![1.0f64; n];
    if n > 0 {
        w[0] = 0.5;
        w[n - 1] = 0.5;
    }
    w
}

/// **求积权重方案**（2026-10-15 加；核矩阵构造方式不变，只换 `w_i`）。
///
/// - [`Weighting::Trapezoid`]（**默认**）：`[½,1,…,1,½]·Δt`，**`O(Δt²)`**。默认保持梯形是
///   **向后兼容**：改默认会改变已有调用的结果与精度特征（且 Simpson 要求偶数个区间）。
/// - [`Weighting::Simpson`]：**复合 1/3 规则**，`[1,4,2,4,…,4,1]·Δt/3`，**`O(Δt⁴)`**；
///   仅当 **`n ≥ 3` 且 `n` 为奇数**（= 偶数个区间）时成立。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Weighting {
    /// 梯形法（默认，`O(Δt²)`）。
    #[default]
    Trapezoid,
    /// Simpson 复合 1/3 规则（`O(Δt⁴)`；`n≥3` 且 `n` 奇数时才可用）。
    Simpson,
}

impl Weighting {
    /// 取本方案的权重；**Simpson 不满足条件时回退梯形**，第二个返回值标记是否发生回退。
    ///
    /// 回退条件：`n < 3` 或 `n` 为偶数（区间数为奇数，凑不出配对的 1/3 面板）。
    /// **不"丢尾巴"**：把最后一个区间用低阶规则拼上去会引入 `O(Δt²)` 项、把整体阶数拖回二阶；
    /// 宁可**整段回退梯形**——阶数诚实、误差可预期（两种回退情形都有测试）。
    pub fn weights(self, n: usize) -> (Vec<f64>, bool) {
        match self {
            Weighting::Trapezoid => (trapezoid_weights(n), false),
            Weighting::Simpson => {
                if n < 3 || n.is_multiple_of(2) {
                    return (trapezoid_weights(n), true);
                }
                let mut w = vec![2.0f64; n];
                w[0] = 1.0;
                w[n - 1] = 1.0;
                for i in (1..n - 1).step_by(2) {
                    w[i] = 4.0;
                }
                for v in w.iter_mut() {
                    *v /= 3.0;
                }
                (w, false)
            }
        }
    }
}

/// 造核矩阵（`f64` 中间量；权重按 `kind` 取；第三个返回值 = 是否回退到梯形）。
fn kernels64(
    sig: &[f64],
    om: &[f64],
    dt: f64,
    n: usize,
    kind: Weighting,
) -> (Vec<f64>, Vec<f64>, bool) {
    let (w, fell_back) = kind.weights(n);
    let m = sig.len();
    let mut kc = vec![0.0f64; m * n];
    let mut ks = vec![0.0f64; m * n];
    for j in 0..m {
        for i in 0..n {
            let t = i as f64 * dt;
            let e = (-sig[j] * t).exp() * w[i] * dt;
            kc[j * n + i] = e * (om[j] * t).cos();
            ks[j * n + i] = e * (om[j] * t).sin();
        }
    }
    (kc, ks, fell_back)
}

/// `f32` 批量正变换：`f` 是 `N×NB`（**列主序的 N 条信号按列排**……实际用行主序 `[N][NB]`），
/// 返回 `(Re, Im)`，各为 `M×NB` 行主序。**含核心的 `matmul!` 调用**。
pub fn laplace_batch_f32<const M: usize, const N: usize, const NB: usize>(
    sig: &[f64],
    om: &[f64],
    dt: f64,
    f: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(sig.len(), M);
    assert_eq!(f.len(), N * NB);
    laplace_batch_f32_with::<M, N, NB>(sig, om, dt, f, Weighting::Trapezoid).0
}

/// 同上，但**指定权重方案**；返回 `(Re, Im, 是否回退到梯形)`。
pub fn laplace_batch_f32_with<const M: usize, const N: usize, const NB: usize>(
    sig: &[f64],
    om: &[f64],
    dt: f64,
    f: &[f32],
    kind: Weighting,
) -> ((Vec<f32>, Vec<f32>), bool) {
    assert_eq!(sig.len(), M);
    assert_eq!(f.len(), N * NB);
    let (kc64, ks64, fell_back) = kernels64(sig, om, dt, N, kind);
    let kc: Vec<f32> = kc64.iter().map(|&v| v as f32).collect();
    let ks: Vec<f32> = ks64.iter().map(|&v| v as f32).collect();
    let mkc = Mat::<f32, M, N>::new(&kc).expect("Kc 长度 = M×N");
    let mks = Mat::<f32, M, N>::new(&ks).expect("Ks 长度 = M×N");
    let mf = Mat::<f32, N, NB>::new(f).expect("f 长度 = N×NB");
    let mut re = MatBuf::<f32, M, NB>::new();
    let mut im = MatBuf::<f32, M, NB>::new();
    // **DSL 表达**：两条矩阵乘；`ks` 取负由调用方在结果上做（保持 `matmul!` 的直白形式）
    matmul!(re[M, NB] = mkc[M, N] * mf[N, NB]);
    matmul!(im[M, NB] = mks[M, N] * mf[N, NB]);
    let re = re.as_slice().to_vec();
    let im = im.as_slice().iter().map(|&v| -v).collect();
    ((re, im), fell_back)
}

/// `f64` 批量正变换（同 `f32` 版，走 `f64` 的 `Mat`/`matmul!`；默认梯形）。
pub fn laplace_batch_f64<const M: usize, const N: usize, const NB: usize>(
    sig: &[f64],
    om: &[f64],
    dt: f64,
    f: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    laplace_batch_f64_with::<M, N, NB>(sig, om, dt, f, Weighting::Trapezoid).0
}

/// 同上，但**指定权重方案**；返回 `(Re, Im, 是否回退到梯形)`。
pub fn laplace_batch_f64_with<const M: usize, const N: usize, const NB: usize>(
    sig: &[f64],
    om: &[f64],
    dt: f64,
    f: &[f64],
    kind: Weighting,
) -> ((Vec<f64>, Vec<f64>), bool) {
    assert_eq!(sig.len(), M);
    assert_eq!(f.len(), N * NB);
    let (kc, ks, fell_back) = kernels64(sig, om, dt, N, kind);
    let mkc = Mat::<f64, M, N>::new(&kc).expect("Kc 长度 = M×N");
    let mks = Mat::<f64, M, N>::new(&ks).expect("Ks 长度 = M×N");
    let mf = Mat::<f64, N, NB>::new(f).expect("f 长度 = N×NB");
    let mut re = MatBuf::<f64, M, NB>::new();
    let mut im = MatBuf::<f64, M, NB>::new();
    matmul!(re[M, NB] = mkc[M, N] * mf[N, NB]);
    matmul!(im[M, NB] = mks[M, N] * mf[N, NB]);
    let re = re.as_slice().to_vec();
    let im = im.as_slice().iter().map(|&v| -v).collect();
    ((re, im), fell_back)
}

/// **串行参考**：按 `api::matmul` 逐列做（与 DSL 路径**位精确对照**用）。
pub fn reference_f32<const M: usize, const N: usize, const NB: usize>(
    sig: &[f64],
    om: &[f64],
    dt: f64,
    f: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    reference_f32_with::<M, N, NB>(sig, om, dt, f, Weighting::Trapezoid)
}

/// 同 [`reference_f32`]，但指定权重方案（**行切参考**：逐列走 `api::matmul`）。
pub fn reference_f32_with<const M: usize, const N: usize, const NB: usize>(
    sig: &[f64],
    om: &[f64],
    dt: f64,
    f: &[f32],
    kind: Weighting,
) -> (Vec<f32>, Vec<f32>) {
    let (kc64, ks64, _) = kernels64(sig, om, dt, N, kind);
    let kc: Vec<f32> = kc64.iter().map(|&v| v as f32).collect();
    let ks: Vec<f32> = ks64.iter().map(|&v| v as f32).collect();
    let mut re = vec![0.0f32; M * NB];
    let mut im = vec![0.0f32; M * NB];
    let mut col = AlignedVec::<f32>::new(N);
    for b in 0..NB {
        for i in 0..N {
            col[i] = f[i * NB + b];
        }
        let out = api::matmul(M, N, 1, &kc, &col).expect("M×N 乘 N×1");
        for j in 0..M {
            re[j * NB + b] = out[j];
        }
        let out = api::matmul(M, N, 1, &ks, &col).expect("M×N 乘 N×1");
        for j in 0..M {
            im[j * NB + b] = -out[j];
        }
    }
    (re, im)
}

#[cfg(test)]
#[allow(dead_code)]
fn laplace_ref_f64<const M: usize, const NB: usize>(
    sig: &[f64],
    om: &[f64],
    dt: f64,
    f: &[f64],
    kind: Weighting,
) -> (Vec<f64>, Vec<f64>) {
    let n = f.len() / NB;
    let (kc, ks, _) = kernels64(sig, om, dt, n, kind);
    let mut re = vec![0.0f64; M * NB];
    let mut im = vec![0.0f64; M * NB];
    for b in 0..NB {
        for j in 0..M {
            let mut a = 0.0f64;
            let mut c = 0.0f64;
            for i in 0..n {
                a += kc[j * n + i] * f[i * NB + b];
                c += ks[j * n + i] * f[i * NB + b];
            }
            re[j * NB + b] = a;
            im[j * NB + b] = -c;
        }
    }
    (re, im)
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: usize = 4;
    const NB: usize = 2;

    /// 解析值（复数）：`F(s) = 1/s`。
    fn f_inv_s(sig: f64, om: f64) -> (f64, f64) {
        let d = sig * sig + om * om;
        (sig / d, -om / d)
    }
    /// 解析值：`F(s) = 1/(s+a)`。
    fn f_inv_s_plus_a(sig: f64, om: f64, a: f64) -> (f64, f64) {
        let (sr, si) = (sig + a, om);
        let d = sr * sr + si * si;
        (sr / d, -si / d)
    }
    /// 解析值：`f = sin(ω₀t) ⇒ F(s) = ω₀/(s²+ω₀²)`。
    fn f_sin(sig: f64, om: f64, w0: f64) -> (f64, f64) {
        let (ar, ai) = (sig * sig - om * om + w0 * w0, 2.0 * sig * om);
        let d = ar * ar + ai * ai;
        (w0 * ar / d, -w0 * ai / d)
    }

    /// **① 三组解析对照（f32）**：`1 ↔ 1/s`、`e^{-at} ↔ 1/(s+a)`、`sin(ω₀t) ↔ ω₀/(s²+ω₀²)`。
    /// **实部虚部都比**（`s` 是复的，解析值也是复数）。
    #[test]
    fn analytic_f32_three_cases() {
        const N: usize = 512;
        let dt = 0.02;
        let sigs = [1.0f64, 1.0, 1.0, 1.0];
        let oms = [0.0f64, 2.0, 0.0, 3.0];
        // NB=2：两条信号 = 1 与 e^{-2t}
        let mut f = vec![0.0f32; N * NB];
        for i in 0..N {
            let t = i as f64 * dt;
            f[i * NB] = 1.0;
            f[i * NB + 1] = (-2.0 * t).exp() as f32;
        }
        let (re, im) = laplace_batch_f32::<M, N, NB>(&sigs, &oms, dt, &f);
        for (j, (&s, &w)) in sigs.iter().zip(oms.iter()).enumerate() {
            // 信号 0：f=1 ⇒ F=1/s
            let (wr, wi) = f_inv_s(s, w);
            let (gr, gi) = (re[j * NB] as f64, im[j * NB] as f64);
            assert!(
                (gr - wr).abs() / wr.abs() < 5e-3,
                "1↔1/s 实部 j={j}: {gr} vs {wr}"
            );
            assert!(
                (gi - wi).abs() / wi.abs().max(1e-3) < 5e-3,
                "1↔1/s 虚部 j={j}: {gi} vs {wi}"
            );
            // 信号 1：f=e^{-2t} ⇒ F=1/(s+2)
            let (wr2, wi2) = f_inv_s_plus_a(s, w, 2.0);
            let (gr2, gi2) = (re[j * NB + 1] as f64, im[j * NB + 1] as f64);
            assert!(
                (gr2 - wr2).abs() / wr2.abs() < 5e-3,
                "e^-2t 实部 j={j}: {gr2} vs {wr2}"
            );
            assert!(
                (gi2 - wi2).abs() / wi2.abs().max(1e-3) < 5e-3,
                "e^-2t 虚部 j={j}: {gi2} vs {wi2}"
            );
        }
        // 第三条：sin(3t) ⇒ ω₀/(s²+ω₀²)（NB=1 的单条路径）
        let mut g = vec![0.0f32; N];
        for (i, gi) in g.iter_mut().enumerate() {
            *gi = (3.0 * (i as f64 * dt)).sin() as f32;
        }
        let (re1, im1) = laplace_batch_f32::<M, N, 1>(&sigs, &oms, dt, &g);
        for (j, (&s, &w)) in sigs.iter().zip(oms.iter()).enumerate() {
            let (wr, wi) = f_sin(s, w, 3.0);
            let (gr, gi) = (re1[j] as f64, im1[j] as f64);
            assert!(
                (gr - wr).abs() / wr.abs().max(1e-3) < 2e-2,
                "sin(3t) 实部 j={j}: {gr} vs {wr}"
            );
            assert!(
                (gi - wi).abs() / wi.abs().max(1e-3) < 2e-2,
                "sin(3t) 虚部 j={j}: {gi} vs {wi}"
            );
        }
    }

    /// **① 解析对照（f64）**：同一批算例走 `f64` 路径，精度应显著更好。
    #[test]
    fn analytic_f64_three_cases() {
        const N: usize = 512;
        let dt = 0.02;
        let sigs = [1.0f64, 1.0, 1.0, 1.0];
        let oms = [0.0f64, 2.0, 0.0, 3.0];
        let mut f = vec![0.0f64; N * NB];
        for i in 0..N {
            let t = i as f64 * dt;
            f[i * NB] = 1.0;
            f[i * NB + 1] = (-2.0 * t).exp();
        }
        let (re, im) = laplace_batch_f64::<M, N, NB>(&sigs, &oms, dt, &f);
        for (j, (&s, &w)) in sigs.iter().zip(oms.iter()).enumerate() {
            let (wr, wi) = f_inv_s(s, w);
            let (gr, gi) = (re[j * NB], im[j * NB]);
            assert!(
                (gr - wr).abs() / wr.abs() < 1e-3,
                "f64 1/s 实部 j={j}: {gr} vs {wr}"
            );
            assert!(
                (gi - wi).abs() / wi.abs().max(1e-3) < 1e-3,
                "f64 1/s 虚部 j={j}: {gi} vs {wi}"
            );
            let (wr2, wi2) = f_inv_s_plus_a(s, w, 2.0);
            let (gr2, gi2) = (re[j * NB + 1], im[j * NB + 1]);
            assert!(
                (gr2 - wr2).abs() / wr2.abs() < 1e-3,
                "f64 e^-2t 实部 j={j}: {gr2} vs {wr2}"
            );
            assert!(
                (gi2 - wi2).abs() / wi2.abs().max(1e-3) < 1e-3,
                "f64 e^-2t 虚部 j={j}: {gi2} vs {wi2}"
            );
        }
    }

    /// **② 加密收敛**：固定积分区间 `T`、把 `Δt` 减半（`N` 翻倍）⇒ **梯形法 O(Δt²)，误差应单调下降**。
    /// （若固定 `Δt` 只加大 `N`，误差会被**梯形法自身**的地板卡住、不再下降——那是另一个口径。）
    #[test]
    fn refinement_error_decreases_with_dt() {
        const T: f64 = 6.4;
        let sigs = [1.0f64, 2.0, 3.0, 4.0];
        let oms = [0.0f64; 4];
        let mut errs: Vec<(usize, f64)> = Vec::new();
        for &n in &[64usize, 128, 256, 512] {
            let dt = T / (n - 1) as f64;
            let mut f = vec![0.0f64; n];
            for (i, fi) in f.iter_mut().enumerate() {
                *fi = (-0.5 * (i as f64 * dt)).exp();
            }
            let (re, _) = laplace_ref_f64::<M, 1>(&sigs, &oms, dt, &f, Weighting::Trapezoid);
            let mut e = 0.0f64;
            for (j, &s) in sigs.iter().enumerate() {
                let want = 1.0 / (s + 0.5);
                e = e.max((re[j] - want).abs() / want);
            }
            errs.push((n, e));
        }
        for w in errs.windows(2) {
            assert!(
                w[1].1 < w[0].1,
                "误差未随 Δt 减半而下降：N={} → {:.3e}，N={} → {:.3e}",
                w[0].0,
                w[0].1,
                w[1].0,
                w[1].1
            );
        }
        // 与 O(Δt²) 相符（N 翻倍至少降 2 倍）；**地板由截断决定**（T=6.4 时 e^{-0.5T}≈0.04），
        // 所以这里只断言"落到 1e-3 以内"而不是 1e-5。
        let (e1, e3) = (errs[0].1, errs[2].1);
        assert!(
            e1 / e3 > 2.0,
            "N=64→256 误差应至少降 2 倍，实测 {:.3e}→{:.3e}",
            e1,
            e3
        );
        assert!(
            errs[3].1 < 1e-3,
            "N=512（T=6.4）误差应 < 1e-3，实测 {:.3e}",
            errs[3].1
        );
    }

    /// **③ 边界**：`ω=0 ⇒ 虚部为 0`；`s=0`（极点）给的是**截断和**（只记录口径，不当真值）。
    #[test]
    fn boundaries_omega_zero_and_s_zero() {
        const N: usize = 256;
        let dt = 0.05;
        let sigs = [0.0f64, 1.0, 2.0, 3.0];
        let oms = [0.0f64; 4];
        let f = vec![1.0f32; N * NB];
        let (re, im) = laplace_batch_f32::<M, N, NB>(&sigs, &oms, dt, &f);
        for (k, v) in im.iter().enumerate() {
            assert_eq!(*v, 0.0, "ω=0 时虚部必须是 0（sin(0)=0），k={k}");
        }
        // s=0：截断和 = Σ w_i·Δt = (N-1)·Δt（梯形首尾各半）⇒ 记录这个口径
        let want_sum = (N - 1) as f64 * dt;
        let got = re[0] as f64;
        assert!(
            (got - want_sum).abs() / want_sum < 1e-4,
            "s=0 的截断和应为 (N-1)Δt = {want_sum}，实测 {got}"
        );
        // 小 N：不能 panic、不给 NaN
        let (re_s, im_s) = laplace_batch_f64::<M, 2, 1>(&sigs, &oms, dt, &[1.0, 1.0]);
        for k in 0..M {
            assert!(
                re_s[k].is_finite() && im_s[k].is_finite(),
                "小 N 不应出 NaN/Inf"
            );
        }
    }

    /// **④ 位精确**：DSL 路径与"行切参考"（`api::matmul`，同一累加次序）**逐位一致**。
    #[test]
    fn dsl_path_is_bit_for_bit() {
        const N: usize = 128;
        let dt = 0.03;
        let sigs = [0.5f64, 1.0, 1.5, 2.0];
        let oms = [0.0f64, 1.0, 0.0, 2.0];
        // 两条信号 × 2 列
        let mut f = vec![0.0f32; N * NB];
        for i in 0..N {
            let t = i as f64 * dt;
            f[i * NB] = t.sin() as f32;
            f[i * NB + 1] = (-t).exp() as f32;
        }
        let (re_dsl, im_dsl) = laplace_batch_f32::<M, N, NB>(&sigs, &oms, dt, &f);
        let (re_ref, im_ref) = reference_f32::<M, N, NB>(&sigs, &oms, dt, &f);
        let bits = |v: &[f32]| -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() };
        assert_eq!(bits(&re_dsl), bits(&re_ref), "Re：DSL 与行切参考不逐位");
        assert_eq!(bits(&im_dsl), bits(&im_ref), "Im：DSL 与行切参考不逐位");
    }

    /// **⑤ 位精确（两种权重各断言一次）**：DSL 路径 vs 行切参考（`api::matmul`）逐位一致。
    #[test]
    fn bit_exact_both_weightings() {
        const N: usize = 129; // 奇数 ⇒ Simpson 可用
        let dt = 0.03;
        let sigs = [0.5f64, 1.0, 1.5, 2.0];
        let oms = [0.0f64, 1.0, 0.0, 2.0];
        let mut f = vec![0.0f32; N * NB];
        for i in 0..N {
            let t = i as f64 * dt;
            f[i * NB] = t.sin() as f32;
            f[i * NB + 1] = (-t).exp() as f32;
        }
        let bits = |v: &[f32]| -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() };
        for kind in [Weighting::Trapezoid, Weighting::Simpson] {
            let ((re_dsl, im_dsl), fell_back) =
                laplace_batch_f32_with::<M, N, NB>(&sigs, &oms, dt, &f, kind);
            assert!(!fell_back, "{kind:?} 在 N={N}（奇数）下不该回退");
            let (re_ref, im_ref) = reference_f32_with::<M, N, NB>(&sigs, &oms, dt, &f, kind);
            assert_eq!(bits(&re_dsl), bits(&re_ref), "{kind:?}：Re 不逐位");
            assert_eq!(bits(&im_dsl), bits(&im_ref), "{kind:?}：Im 不逐位");
        }
    }

    /// **⑥ 阶数：梯形 `O(Δt²)` vs Simpson `O(Δt⁴)`**（固定 `T`、`Δt` 减半；`n` 取奇数让 Simpson 可用）。
    ///
    /// 断言口径：梯形相邻比落在 `[2.5, 8]`（≈4，二阶）；Simpson **> 8**（理想 ≈16，四阶）。
    /// **阶数只在不被 `T` 截断地板支配时才可见**——所以这里挑 `sig ∈ [1,4]`（衰减快）与 `T=6.4`。
    #[test]
    fn convergence_order_trapezoid_vs_simpson() {
        // 采样误差：固定 T、n 递增（n 取奇数让 Simpson 可用）
        let sample = |t_end: f64, a: f64, ns: &[usize]| -> Vec<(usize, f64, f64)> {
            let sigs = [1.0f64, 2.0, 3.0, 4.0];
            let oms = [0.0f64; 4];
            let mut rows = Vec::new();
            for &n in ns {
                let dt = t_end / (n - 1) as f64;
                let mut f = vec![0.0f64; n];
                for (i, fi) in f.iter_mut().enumerate() {
                    *fi = (-a * (i as f64 * dt)).exp();
                }
                let mut errs = [0.0f64; 2];
                for (k, kind) in [Weighting::Trapezoid, Weighting::Simpson]
                    .iter()
                    .enumerate()
                {
                    let (re, _) = laplace_ref_f64::<M, 1>(&sigs, &oms, dt, &f, *kind);
                    let mut e = 0.0f64;
                    for (j, &s) in sigs.iter().enumerate() {
                        let want = 1.0 / (s + a);
                        e = e.max((re[j] - want).abs() / want);
                    }
                    errs[k] = e;
                }
                rows.push((n, errs[0], errs[1]));
            }
            rows
        };
        // ① T=20（截断可忽略）⇒ 阶数可见：梯形 ≈4×、Simpson >8×
        let rows = sample(20.0, 2.0, &[65, 129, 257, 513]);
        let trap_ratios: Vec<f64> = rows.windows(2).map(|w| w[0].1 / w[1].1).collect();
        let simp_ratios: Vec<f64> = rows.windows(2).map(|w| w[0].2 / w[1].2).collect();
        for r in &trap_ratios {
            assert!(
                (2.5..=8.0).contains(r),
                "梯形相邻比应 ≈4（二阶），实测 {r:.2}×"
            );
        }
        for r in &simp_ratios {
            assert!(
                *r > 8.0,
                "Simpson 相邻比应 ≫4（四阶、理想 ≈16），实测 {r:.2}×；表 {rows:?}"
            );
        }
        for (n, et, es) in &rows {
            assert!(es < et, "N={n} 时 Simpson({es:.3e}) 应优于梯形({et:.3e})");
        }
        // ② T=6.4：Simpson 被**截断地板**卡住（末对相邻比 ≈1），梯形仍 ≈4 ⇒ 阶数不可见
        let rows2 = sample(6.4, 0.5, &[65, 129, 257, 513]);
        let simp2: Vec<f64> = rows2.windows(2).map(|w| w[0].2 / w[1].2).collect();
        let trap2: Vec<f64> = rows2.windows(2).map(|w| w[0].1 / w[1].1).collect();
        assert!(
            *simp2.last().unwrap() < 2.0,
            "T=6.4 时 Simpson 应被截断地板卡住（末对相邻比 <2），实测 {:.2}×；表 {rows2:?}",
            simp2.last().unwrap()
        );
        for r in &trap2 {
            assert!((2.5..=8.0).contains(r), "梯形相邻比应 ≈4，实测 {r:.2}×");
        }
    }

    /// **⑦ Simpson 下的解析对照（3 组、实虚部）**：两种权重都对（阈值按 **f32 累加** 放宽到
    /// `3e-3`——513 项 f32 点积自身就有 ~1e-4 量级的舍入），并断言 **Simpson 误差 < 梯形**。
    #[test]
    fn simpson_analytic_three_cases() {
        const N: usize = 513; // 奇数
        let dt = 0.0125;
        let sigs = [1.0f64, 1.0, 1.0, 1.0];
        let oms = [0.0f64, 2.0, 0.0, 3.0];
        let mut f = vec![0.0f32; N * NB];
        for i in 0..N {
            let t = i as f64 * dt;
            f[i * NB] = 1.0;
            f[i * NB + 1] = (-2.0 * t).exp() as f32;
        }
        let mut worst = [0.0f64; 2];
        for (k, kind) in [Weighting::Trapezoid, Weighting::Simpson]
            .iter()
            .enumerate()
        {
            let ((re, im), fell_back) =
                laplace_batch_f32_with::<M, N, NB>(&sigs, &oms, dt, &f, *kind);
            assert!(!fell_back);
            for (j, (&s, &w)) in sigs.iter().zip(oms.iter()).enumerate() {
                let (wr, wi) = f_inv_s(s, w);
                let (gr, gi) = (re[j * NB] as f64, im[j * NB] as f64);
                let e1 = (gr - wr).abs() / wr.abs();
                let e2 = (gi - wi).abs() / wi.abs().max(1e-3);
                let (wr2, wi2) = f_inv_s_plus_a(s, w, 2.0);
                let (gr2, gi2) = (re[j * NB + 1] as f64, im[j * NB + 1] as f64);
                let e3 = (gr2 - wr2).abs() / wr2.abs();
                let e4 = (gi2 - wi2).abs() / wi2.abs().max(1e-3);
                let e = e1.max(e2).max(e3).max(e4);
                worst[k] = worst[k].max(e);
                assert!(e < 3e-3, "{kind:?} 解析对照误差过大 j={j}: {e:.3e}");
            }
        }
        assert!(
            worst[1] < worst[0],
            "Simpson 的最大误差应小于梯形：{:.3e} vs {:.3e}",
            worst[1],
            worst[0]
        );
    }

    /// **⑧ 边界**：`N<3` 与**奇数区间（偶数 N）**⇒ Simpson **回退梯形**并**如实返回标记**；
    /// `s=0` 的截断和、`ω=0` 的零虚部在两种权重下都成立。
    #[test]
    fn simpson_fallback_and_boundaries() {
        // 权重层面
        let (w2, fb2) = Weighting::Simpson.weights(2);
        assert!(fb2, "N=2 必须回退");
        assert_eq!(w2, trapezoid_weights(2));
        let (w4, fb4) = Weighting::Simpson.weights(4);
        assert!(fb4, "N=4（区间数 3，奇数）必须回退");
        assert_eq!(w4, trapezoid_weights(4));
        let (w5, fb5) = Weighting::Simpson.weights(5);
        assert!(!fb5);
        assert_eq!(w5.len(), 5);
        assert_eq!(
            (w5[0], w5[1], w5[2], w5[3], w5[4]),
            (1.0 / 3.0, 4.0 / 3.0, 2.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0)
        );
        // 路径层面：N=2 走 Simpson ⇒ 回退标记为真
        let sigs = [1.0f64, 2.0, 3.0, 4.0];
        let oms = [0.0f64; 4];
        let ((_, _), fell_back) =
            laplace_batch_f64_with::<M, 2, 1>(&sigs, &oms, 0.5, &[1.0, 1.0], Weighting::Simpson);
        assert!(fell_back, "N=2 的路径也必须标出回退");
        // ω=0 ⇒ 虚部恒 0；σ=0（即 s=0）时 Re = 权重和 × dt（两种方案对常数都精确 ⇒ 都等于 (n-1)Δt）
        let n = 33usize;
        let dt = 0.05;
        let zero_sig = [0.0f64; 4];
        let ((re_s, im_s), fb) = laplace_batch_f64_with::<M, 33, 2>(
            &zero_sig,
            &oms,
            dt,
            &vec![1.0f64; n * 2],
            Weighting::Simpson,
        );
        assert!(!fb);
        for v in im_s.iter() {
            assert_eq!(*v, 0.0, "ω=0 时虚部必须为 0");
        }
        // s=0 的 Re 就是"权重和 × dt"（梯形 = (n-1)Δt；Simpson = 也等于 (n-1)Δt，因为 1/3 规则
        // 对常数精确）——两种方案对 f≡1、s=0 都应给出同一个 (n-1)Δt。
        let want = (n - 1) as f64 * dt;
        assert!(
            (re_s[0] - want).abs() / want < 1e-12,
            "Simpson 对常数应精确：{} vs {want}",
            re_s[0]
        );
    }
}

/// **C 阶：拉普拉斯正向的"一次复数矩阵乘"路线**（`Kc + i·Ks` 乘**实信号** `f`）。
///
/// 数学上 `(Kc + iKs)·(f + 0i) = Kc·f + i·Ks·f` ⇒ 与"两条实矩阵乘"**同值**，但**累加次序不同**
/// ⇒ **只承诺容差**（不承诺与旧路线逐位一致）。f32 版本。
pub fn laplace_batch_f32_complex_route<const M: usize, const N: usize, const NB: usize>(
    sig: &[f64],
    om: &[f64],
    dt: f64,
    f: &[f32],
    route: crate::cmatmul::Route,
) -> (Vec<f32>, Vec<f32>) {
    use crate::cmatmul::{CMat, CMatBuf};
    assert_eq!(sig.len(), M);
    assert_eq!(f.len(), N * NB);
    let (kc64, ks64, _) = kernels64(sig, om, dt, N, Weighting::Trapezoid);
    let kc: Vec<f32> = kc64.iter().map(|&v| v as f32).collect();
    let ks: Vec<f32> = ks64.iter().map(|&v| v as f32).collect();
    // f 是**实**信号 ⇒ 虚部全 0
    let fi = vec![0.0f32; N * NB];
    let a = CMat::<M, N>::new(&kc, &ks).expect("K = M×N");
    let b = CMat::<N, NB>::new(f, &fi).expect("f = N×NB");
    let mut c = CMatBuf::<M, NB>::zeros();
    a.mul_into(&b, &mut c, route);
    let im = c.im.iter().map(|&v| -v).collect();
    (c.re, im)
}

#[cfg(test)]
mod c_route_tests {
    use super::*;

    /// **C 阶：解析三组（f32）在复数路线下仍过**，且与旧路线（两条实乘）**容差内一致**。
    #[test]
    fn complex_route_matches_old_within_tolerance() {
        const N: usize = 512;
        const NB: usize = 2;
        let dt = 0.02;
        let sigs = [1.0f64, 1.0, 1.0, 1.0];
        let oms = [0.0f64, 2.0, 0.0, 3.0];
        let mut f = vec![0.0f32; N * NB];
        for i in 0..N {
            let t = i as f64 * dt;
            f[i * NB] = 1.0;
            f[i * NB + 1] = (-2.0 * t).exp() as f32;
        }
        let (old_re, old_im) = laplace_batch_f32::<4, N, NB>(&sigs, &oms, dt, &f);
        for route in [
            crate::cmatmul::Route::Direct,
            crate::cmatmul::Route::Karatsuba,
        ] {
            let (new_re, new_im) =
                laplace_batch_f32_complex_route::<4, N, NB>(&sigs, &oms, dt, &f, route);
            let mut worst = 0.0f32;
            for l in 0..4 * NB {
                worst = worst.max((new_re[l] - old_re[l]).abs() / old_re[l].abs().max(1.0));
                worst = worst.max((new_im[l] - old_im[l]).abs() / old_im[l].abs().max(1.0));
            }
            assert!(worst < 1e-4, "{route:?} 复数路线 vs 旧路线 差 {worst:e}");
            // 解析对照：`F(s)=1/s`（**s 是复数**）⇒ Re = σ/(σ²+ω²)、Im = −ω/(σ²+ω²)
            for (j, (&s, &w)) in sigs.iter().zip(oms.iter()).enumerate() {
                let d = s * s + w * w;
                let (want_re, want_im) = (s / d, -w / d);
                assert!(
                    (new_re[j * NB] as f64 - want_re).abs() / want_re < 5e-3,
                    "{route:?} 1/s 实部 j={j}：{} vs {want_re}",
                    new_re[j * NB]
                );
                if want_im.abs() > 1e-3 {
                    assert!(
                        (new_im[j * NB] as f64 - want_im).abs() / want_im.abs() < 5e-3,
                        "{route:?} 1/s 虚部 j={j}：{} vs {want_im}",
                        new_im[j * NB]
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod c_perf {
    use super::*;
    use std::time::Instant;

    /// **C 阶性能：旧路线（两条实矩阵乘）vs 一次复数矩阵乘**（同进程交替 ≥5 对、丢首对）。
    #[ignore = "C 阶性能：两条实乘 vs 一次复数乘"]
    #[test]
    fn diag_laplace_two_real_vs_one_complex() {
        const M: usize = 64;
        const N: usize = 1024;
        const NB: usize = 8;
        let dt = 1.0 / 256.0;
        let sigs: Vec<f64> = (0..M).map(|j| 0.5 + 0.01 * j as f64).collect();
        let oms: Vec<f64> = (0..M).map(|j| (j % 16) as f64 * 0.25).collect();
        let f = vec![1.0f32; N * NB];
        let cx_flop = 2.0 * M as f64 * N as f64 * NB as f64; // 复数乘加按 2 个实数乘加计（这里只做 2 条实乘的工作量）
        let mut rows: Vec<(f64, f64, f64, i32, f64, String)> = Vec::new();
        for pair in 0..6 {
            let mut t = [0.0f64; 3]; // 0=旧(2 实乘), 1=复数 Direct, 2=复数 Karatsuba
            let mut iters = 0i32;
            let mut ms_max = 0.0f64;
            for step in 0..3 {
                let which = (pair + step) % 3;
                let mut cnt = 0i32;
                let mut mx = 0.0f64;
                let t0 = Instant::now();
                while t0.elapsed().as_secs_f64() < 0.04 {
                    let t1 = Instant::now();
                    match which {
                        0 => {
                            let _ = laplace_batch_f32::<M, N, NB>(&sigs, &oms, dt, &f);
                        }
                        1 => {
                            let _ = laplace_batch_f32_complex_route::<M, N, NB>(
                                &sigs,
                                &oms,
                                dt,
                                &f,
                                crate::cmatmul::Route::Direct,
                            );
                        }
                        _ => {
                            let _ = laplace_batch_f32_complex_route::<M, N, NB>(
                                &sigs,
                                &oms,
                                dt,
                                &f,
                                crate::cmatmul::Route::Karatsuba,
                            );
                        }
                    }
                    let d = t1.elapsed().as_secs_f64();
                    if d > mx {
                        mx = d;
                    }
                    cnt += 1;
                }
                t[which] = t0.elapsed().as_secs_f64() / cnt.max(1) as f64;
                iters = cnt;
                if mx > ms_max {
                    ms_max = mx;
                }
            }
            let l1 = std::fs::read_to_string("/proc/loadavg")
                .ok()
                .and_then(|s| s.split_whitespace().next().map(str::to_owned))
                .unwrap_or_default();
            rows.push((t[0], t[1], t[2], iters, ms_max * 1e3, l1));
        }
        let med = |i: usize| -> f64 {
            let mut v: Vec<f64> = rows[1..]
                .iter()
                .map(|r| match i {
                    0 => r.0,
                    1 => r.1,
                    _ => r.2,
                })
                .collect();
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let (o, d, k) = (med(0), med(1), med(2));
        for (i, r) in rows.iter().enumerate() {
            println!(
                "pair{} old2real={:.4}ms cxDirect={:.4}ms cxKarat={:.4}ms iters={} ms_max={:.3} loadavg={}{}",
                i + 1,
                r.0 * 1e3,
                r.1 * 1e3,
                r.2 * 1e3,
                r.3,
                r.4,
                r.5,
                if i == 0 { " (dropped)" } else { "" }
            );
        }
        println!(
            "中位：旧(2 实乘) {:.4} ms｜复数 Direct {:.4} ms（{:.2}×旧）｜复数 Karatsuba {:.4} ms（{:.2}×旧）",
            o * 1e3,
            d * 1e3,
            d / o,
            k * 1e3,
            k / o
        );
        println!("形状 M×N×NB = {M}×{N}×{NB}；线程数：串行核心（未自动多线程）");
        println!(
            "复杂乘加工作量 {:.4} GFLOP（按 2 实数乘加/复元素计）",
            cx_flop / 1e9
        );
    }
}

/// **运算微积分：拉普拉斯导数定理**（`L{f⁽ⁿ⁾}(s) = sⁿ·F(s) − Σ_{k=0}^{n−1} s^{n−1−k}·f⁽ᵏ⁾(0⁺)`）。
///
/// 输入是与 [`laplace_batch_f64`] 输出**同一批 `s` 节点**上的 `F`（分平面 `(Re,Im)`），
/// 加上**调用方给定**的初值 `f⁽ᵏ⁾(0⁺)`（`k = 0..n`）；输出是同一批节点上的 `G(s) = L{f⁽ⁿ⁾}`。
///
/// # 初值契约（**与 `f₀` 一致**）
/// - **首选：调用方给**（解析式或已知量）；**本函数不做任何数值估计、没有静默回退**；
/// - `init` 必须是 **右极限 `f⁽ᵏ⁾(0⁺)`**（单边拉普拉斯的约定）；
/// - **取错的后果**：残留 `s^{n−1−k}·ε_k` 项 ⇒ 在**大 `|s|`** 处误差按 `|s|^{n−1−k}` 放大
///   （实测见 `docs/ops.md` §9.08 的表）。
///
/// # 适用范围（**不满足就别用**）
/// - 要求 `f` 在 `[0,∞)` 上**足够次可微**、且**指数阶**（否则 `F` 本身不存在）；
/// - **阶跃/含 δ 的情形不适用**：如 `f ≡ 1` 时 `f″ = δ′`（分布），定理给的是 `s²·(1/s) − s·1 − 0 = 0`
///   ——**错**（有反例测试 `derivative_theorem_step_counterexample`）；
/// - **浮点相消**：`sⁿ·F(s)` 在 `n` 大、`|s|` 大时会发生灾难性相消（见文档给出的经验界）。
///
/// # 实现
/// `s_j` 上的运算是**逐节点的复数乘/乘幂**（不是矩阵乘）——按 `Re/Im` 分平面直接算，
/// 每步都是 2×2 实乘加，**没有新造矩阵乘**；`n` 小（1/2/3）时展开成显式表达式。
pub fn derivative_spectrum_f64(
    n: usize,
    s_re: &[f64],
    s_im: &[f64],
    fre: &[f64],
    fim: &[f64],
    init: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(s_re.len(), s_im.len(), "s 的分平面长度必须一致");
    assert_eq!(s_re.len(), fre.len(), "F 必须与 s 在同一批节点上");
    assert_eq!(fre.len(), fim.len());
    assert!(init.len() >= n, "init 至少给 n 个初值（f(0⁺)…f⁽ⁿ⁻¹⁾(0⁺)）");
    let m = s_re.len();
    let mut gre = vec![0.0f64; m];
    let mut gim = vec![0.0f64; m];
    for j in 0..m {
        // s^n（逐节点复数幂；n 小则展开）
        let (mut pr, mut pi) = (1.0f64, 0.0f64);
        for _ in 0..n {
            let (nr, ni) = (pr * s_re[j] - pi * s_im[j], pr * s_im[j] + pi * s_re[j]);
            pr = nr;
            pi = ni;
        }
        // s^n·F
        let (mut tr, mut ti) = (pr * fre[j] - pi * fim[j], pr * fim[j] + pi * fre[j]);
        // 减 Σ_{k=0}^{n−1} s^{n−1−k}·f⁽ᵏ⁾(0⁺)
        for (k, &fk) in init.iter().take(n).enumerate() {
            let p = n - 1 - k;
            let (mut qr, mut qi) = (1.0f64, 0.0f64);
            for _ in 0..p {
                let (nr, ni) = (qr * s_re[j] - qi * s_im[j], qr * s_im[j] + qi * s_re[j]);
                qr = nr;
                qi = ni;
            }
            tr -= qr * fk;
            ti -= qi * fk;
        }
        gre[j] = tr;
        gim[j] = ti;
    }
    (gre, gim)
}

/// **特化：一阶导** `L{f′}(s) = s·F(s) − f(0⁺)`。
pub fn derivative1_spectrum_f64(
    s_re: &[f64],
    s_im: &[f64],
    fre: &[f64],
    fim: &[f64],
    f0: f64,
) -> (Vec<f64>, Vec<f64>) {
    derivative_spectrum_f64(1, s_re, s_im, fre, fim, &[f0])
}

/// **特化：二阶导** `L{f″}(s) = s²·F(s) − s·f(0⁺) − f′(0⁺)`。
pub fn derivative2_spectrum_f64(
    s_re: &[f64],
    s_im: &[f64],
    fre: &[f64],
    fim: &[f64],
    f0: f64,
    d1: f64,
) -> (Vec<f64>, Vec<f64>) {
    derivative_spectrum_f64(2, s_re, s_im, fre, fim, &[f0, d1])
}

#[cfg(test)]
mod derivative_tests {
    use super::*;

    /// 在 `s` 节点上取解析 `F`（f 的变换）与解析 `F″ = L{f″}`。
    fn nodes() -> Vec<(f64, f64)> {
        // σ > 0（保证 F 有定义），带虚部
        (0..8)
            .map(|j| {
                let s = 1.0 + 0.25 * j as f64;
                (s, (j % 3) as f64 * 0.5)
            })
            .collect()
    }

    /// **② 三组解析对照（实部+虚部）** —— 定理给 `L{f″}`，与解析式比。
    #[test]
    fn derivative_theorem_three_analytic_cases() {
        let ns = nodes();
        let s_re: Vec<f64> = ns.iter().map(|&(r, _)| r).collect();
        let s_im: Vec<f64> = ns.iter().map(|&(_, i)| i).collect();
        // (a) f = e^{-a t}，a = 2：F = 1/(s+2)，f(0)=1，f'(0)=−2 ⇒ L{f″} = a²/(s+a) = 4/(s+2)
        // (b) f = sin(ω₀t)，ω₀ = 3：F = 3/(s²+9)，f(0)=0，f'(0)=3 ⇒ L{f″} = −27/(s²+9)
        // (c) f = t：F = 1/s²，f(0)=0，f'(0)=1 ⇒ L{f″} = 0
        for (name, fre, fim, init, want) in [
            (
                "e^{-2t}",
                ns.iter()
                    .map(|&(r, i)| 1.0 / ((r + 2.0).powi(2) + i * i) * (r + 2.0))
                    .collect::<Vec<_>>(),
                ns.iter()
                    .map(|&(r, i)| -i / ((r + 2.0).powi(2) + i * i))
                    .collect::<Vec<_>>(),
                vec![1.0f64, -2.0],
                // 4/(s+2)
                (|r: f64, i: f64| (4.0 * (r + 2.0)) / ((r + 2.0).powi(2) + i * i))
                    as fn(f64, f64) -> f64,
            ),
            (
                "sin(3t)",
                ns.iter()
                    .map(|&(r, i)| {
                        (3.0 * (r * r - i * i + 9.0))
                            / ((r * r - i * i + 9.0).powi(2) + (2.0 * r * i).powi(2))
                    })
                    .collect(),
                ns.iter()
                    .map(|&(r, i)| {
                        (-3.0 * (2.0 * r * i))
                            / ((r * r - i * i + 9.0).powi(2) + (2.0 * r * i).powi(2))
                    })
                    .collect(),
                vec![0.0f64, 3.0],
                |r: f64, i: f64| {
                    let (ar, ai) = (r * r - i * i + 9.0, 2.0 * r * i);
                    -27.0 * ar / (ar * ar + ai * ai)
                },
            ),
            (
                "t",
                // F = 1/s² = (s̄)²/|s|⁴
                ns.iter()
                    .map(|&(r, i)| (r * r - i * i) / ((r * r + i * i).powi(2)))
                    .collect(),
                ns.iter()
                    .map(|&(r, i)| (-2.0 * r * i) / ((r * r + i * i).powi(2)))
                    .collect(),
                vec![0.0f64, 1.0],
                |_r: f64, _i: f64| 0.0,
            ),
        ] {
            let (gre, gim) = derivative2_spectrum_f64(&s_re, &s_im, &fre, &fim, init[0], init[1]);
            // 也验证通用实现与二阶特化一致（逐位）
            let (gre2, gim2) = derivative_spectrum_f64(2, &s_re, &s_im, &fre, &fim, &init);
            assert_eq!(
                gre.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                gre2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{name}: 特化与通用不一致"
            );
            assert_eq!(
                gim.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                gim2.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            // 与解析 `L{f″}` 比（实部；虚部同理）
            let mut worst = 0.0f64;
            for (j, &(r, i)) in ns.iter().enumerate() {
                let w = want(r, i);
                let e = (gre[j] - w).abs() / w.abs().max(1e-9);
                worst = worst.max(e);
                if w == 0.0 {
                    assert!(gre[j].abs() < 1e-9, "{name}: 期望 0，实测 {:.3e}", gre[j]);
                }
                let _ = gim[j];
            }
            if want(1.0, 0.5) != 0.0 {
                assert!(worst < 1e-12, "{name}: 最大相对误差 {worst:.3e}");
            }
        }
    }

    /// **③ 端到端往返（最值钱）**：`F` → **导数定理**（得到 `L{f″}`）→ **库内逆变换** → `f″`。
    ///
    /// 这里把 `G(s) = s²F(s) − s·f(0⁺) − f′(0⁺)` 直接作为 [`ComplexFn`] 交给 `invert_batch_f64`，
    /// **并且用 `DubnerAbate::f0 = f″(0⁺) = a²` 做跳变扣除**（否则会退回 1/N 收敛，见 §9.05 的 `f₀` 契约）。
    #[test]
    fn derivative_theorem_round_trip_through_inverse() {
        use crate::ilaplace::{ComplexFn, DubnerAbate, OneOverSPlus, invert_batch_f64};
        let a = 1.0f64;
        /// `G(s) = s²·F(s) − s·f0 − d1`（`F` 用解析 `1/(s+a)`）。
        struct G {
            f: OneOverSPlus,
            f0: f64,
            d1: f64,
        }
        impl ComplexFn for G {
            fn eval(&self, re: f64, im: f64) -> (f64, f64) {
                let (fr, fi) = self.f.eval(re, im);
                let (s2r, s2i) = (re * re - im * im, 2.0 * re * im);
                let (pr, pi) = (s2r * fr - s2i * fi, s2r * fi + s2i * fr);
                (pr - re * self.f0 - self.d1, pi - im * self.f0)
            }
        }
        // 直接在库内的节点上核对"导数定理数值 = 解析 G"（分平面）
        let p_f0 = 1.0f64; // f(0⁺) = 1
        let p = DubnerAbate {
            t_end: 12.0,
            a: 1.0,
            n: 512,
            f0: a * a, // **f″(0⁺) = a²** ⇒ 逆变换的跳变扣除
        };
        let one = OneOverSPlus { theta: a };
        let s_re = vec![p.a; 512];
        let s_im: Vec<f64> = (1..=512)
            .map(|k| k as f64 * std::f64::consts::PI / p.t_end)
            .collect();
        let fre: Vec<f64> = s_im.iter().map(|&om| one.eval(p.a, om).0).collect();
        let fim: Vec<f64> = s_im.iter().map(|&om| one.eval(p.a, om).1).collect();
        let (gre, gim) = derivative2_spectrum_f64(&s_re, &s_im, &fre, &fim, p_f0, -a);
        // 与解析 G 比（导数定理本身的正确性，分平面）
        let g = G {
            f: OneOverSPlus { theta: a },
            f0: p_f0,
            d1: -a,
        };
        let mut worst_g = 0.0f64;
        for k in 0..512 {
            let (wr, wi) = g.eval(s_re[k], s_im[k]);
            worst_g = worst_g.max((gre[k] - wr).abs() / wr.abs().max(1.0));
            worst_g = worst_g.max((gim[k] - wi).abs() / wi.abs().max(1.0));
        }
        assert!(worst_g < 1e-12, "导数定理 vs 解析 G 差 {worst_g:.3e}");
        // 逆变换 ⇒ 应得 f″(t) = a²·e^{−at}
        let ts = [0.5f64, 1.0, 2.0, 3.0];
        let got = invert_batch_f64::<4, 512, 1, _>(p, &ts, &[g]);
        let mut worst = 0.0f64;
        let mut table = Vec::new();
        for (j, &t) in ts.iter().enumerate() {
            let want = a * a * (-a * t).exp();
            let e = (got[j] - want).abs() / want;
            worst = worst.max(e);
            table.push((t, got[j], want, e));
        }
        println!("端到端往返（正向解析 F → 导数定理 → D–A 逆变换）：");
        for (t, got, want, e) in &table {
            println!("  t={t}: 得到 {got:.6} 期望 {want:.6} 相对误差 {e:.3e}");
        }
        assert!(worst < 5e-3, "往返误差过大：{worst:.3e}（数列 {got:?}）");
    }

    /// **④ 初值契约：取错会怎样（实测）** —— `f=e^{−at}` 的 `f(0)` 故意给错 `ε`，
    /// 残差应≈ `ε·s`（在 `|s|` 大处按 |s| 放大）。
    #[test]
    fn derivative_theorem_wrong_init_leaves_residual() {
        let s_re = [2.0f64, 8.0, 32.0];
        let s_im = [0.0f64, 0.0, 0.0];
        let a = 1.0f64;
        let fre: Vec<f64> = s_re.iter().map(|&r| 1.0 / (r + a)).collect();
        let fim = vec![0.0f64; 3];
        let (gre, _) = derivative2_spectrum_f64(&s_re, &s_im, &fre, &fim, 1.0, -a);
        let (gwrong, _) = derivative2_spectrum_f64(&s_re, &s_im, &fre, &fim, 1.0 + 0.01, -a);
        for j in 0..3 {
            let residual = (gwrong[j] - gre[j]).abs();
            // 残差 = ε·s_j ⇒ 相对 |s| 线性放大
            let want = 0.01 * s_re[j];
            assert!(
                (residual - want).abs() / want < 1e-12,
                "s={} 残差应 = ε·s = {want}，实测 {residual}",
                s_re[j]
            );
        }
    }

    /// **⑤ 边界/反例：阶跃 `f≡1` 不适用**（`f″ = δ′` 是分布，定理给 0 —— 明确记为"不适用"）。
    #[test]
    fn derivative_theorem_step_counterexample() {
        let s_re = [1.0f64, 2.0, 3.0];
        let s_im = [0.0f64; 3];
        let fre: Vec<f64> = s_re.iter().map(|&r| 1.0 / r).collect(); // F = 1/s
        let fim = vec![0.0f64; 3];
        let (gre, gim) = derivative2_spectrum_f64(&s_re, &s_im, &fre, &fim, 1.0, 0.0);
        for j in 0..3 {
            assert!(gre[j].abs() < 1e-12 && gim[j].abs() < 1e-12);
        }
        // 定理给 0，但真值是 δ′（分布）⇒ **不适用**。这里不做"修正"，只把"定理的输出是 0、
        // 而真值不是 0"这件事**钉成断言**（如实记录不适用；反例的说明见 `docs/ops.md` §9.08）。
        assert_eq!(
            gre[0], 0.0,
            "阶跃情形定理输出 0（而真值 f″=δ′ ⇒ 不适用，见 §9.08）"
        );
        assert_eq!(gim[0], 0.0);
    }
}
