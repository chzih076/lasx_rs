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

/// 造核矩阵（`f64` 中间量、按 `to_f32`/原样给两条路径用）。
fn kernels64(sig: &[f64], om: &[f64], dt: f64, n: usize, w: &[f64]) -> (Vec<f64>, Vec<f64>) {
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
    (kc, ks)
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
    let (kc64, ks64) = kernels64(sig, om, dt, N, &trapezoid_weights(N));
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
    (re, im)
}

/// `f64` 批量正变换（同 `f32` 版，走 `f64` 的 `Mat`/`matmul!`）。
pub fn laplace_batch_f64<const M: usize, const N: usize, const NB: usize>(
    sig: &[f64],
    om: &[f64],
    dt: f64,
    f: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(sig.len(), M);
    assert_eq!(f.len(), N * NB);
    let (kc, ks) = kernels64(sig, om, dt, N, &trapezoid_weights(N));
    let mkc = Mat::<f64, M, N>::new(&kc).expect("Kc 长度 = M×N");
    let mks = Mat::<f64, M, N>::new(&ks).expect("Ks 长度 = M×N");
    let mf = Mat::<f64, N, NB>::new(f).expect("f 长度 = N×NB");
    let mut re = MatBuf::<f64, M, NB>::new();
    let mut im = MatBuf::<f64, M, NB>::new();
    matmul!(re[M, NB] = mkc[M, N] * mf[N, NB]);
    matmul!(im[M, NB] = mks[M, N] * mf[N, NB]);
    let re = re.as_slice().to_vec();
    let im = im.as_slice().iter().map(|&v| -v).collect();
    (re, im)
}

/// **串行参考**：按 `api::matmul` 逐列做（与 DSL 路径**位精确对照**用）。
pub fn reference_f32<const M: usize, const N: usize, const NB: usize>(
    sig: &[f64],
    om: &[f64],
    dt: f64,
    f: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let (kc64, ks64) = kernels64(sig, om, dt, N, &trapezoid_weights(N));
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
) -> (Vec<f64>, Vec<f64>) {
    let n = f.len() / NB;
    let (kc, ks) = kernels64(sig, om, dt, n, &trapezoid_weights(n));
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
            let (re, _) = laplace_ref_f64::<M, 1>(&sigs, &oms, dt, &f);
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
}
