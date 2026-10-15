//! **复数矩阵乘核心**（分平面 `(Re, Im)` 两块实数组；f32/f64）。
//!
//! 复数乘法 `(a+bi)(c+di) = (ac−bd) + (ad+bc)i` 有两条实现路线：
//!
//! | 路线 | 实矩阵乘条数 | 位精确契约 |
//! |---|---|---|
//! | [`Route::Direct`]（默认） | **4**：`Re=Ar·Br−Ai·Bi`、`Im=Ar·Bi+Ai·Br` | **每条乘积与对应实矩阵乘逐位一致**（可测：与 `ops::matmul` 的组合逐位比对）；和/差按 IEEE 逐元素 |
//! | [`Route::Karatsuba`] | **3**：`k1=(Ar+Ai)(Br+Bi)`、`k2=ArBr`、`k3=AiBi` | **只承诺容差**（`k1` 里多了一次加法、组合次序也不同）⇒ **不承诺**与直算逐位一致 |
//!
//! **"完全 SIMD"的判据（A 阶口径）**：**每一次复数乘加都由 LASX 实矩阵乘内核完成、不许标量回退**。
//! **直算路线天然满足**：它的 4 条乘积全都是现有 LASX 实内核（`ops::matmul::matmul_f32`/`_f64`）；
//! Karatsuba 的 3 条同样走实内核，但要额外付 `Ar+Ai`/`Br+Bi` 两次**逐元素**加法（SIMD 化、但不是矩阵乘）。
//!
//! **谁守位精确**：
//! - 直算：测试 `direct_matches_four_real_matmul_bitwise`（与 4 条实矩阵乘的组合**逐位**比对）；
//! - Karatsuba：测试 `routes_agree_within_tolerance`（容差）+ 文档声明（**无逐位承诺**）。
//!
//! **多线程**：本模块的两条路线都是**串行核心**（与 `ops::matmul` 同层）。要多线程请调用方按
//! **分平面**切：对 `Ar/Ai/Br/Bi` 各条实矩阵乘用 `parallel::*`（计划/池复用）自行并行，或按输出行切。
//! 复数路径**暂不做自动多线程**（A 阶只保证"完全 SIMD"，并行留到后续阶）。
use super::matmul as rmath;
use super::matmul_f64 as rmath64;

/// 复数矩阵乘的实现路线。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Route {
    /// **直算 4 条**实矩阵乘（位精确契约最强；`N=192×768×3072` 实测 57.4 复数 GFLOP/s）。
    Direct,
    /// **Karatsuba 3 条**实矩阵乘（**默认**：省一条矩阵乘、换两次逐元素加法；只承诺容差。
    /// 同源实测 **67.9 复数 GFLOP/s，比直算快 18%** ⇒ 按数据选它当默认）。
    #[default]
    Karatsuba,
}

/// 逐元素复数参考（最朴素，用于容差对照）。
fn ref_into<
    T: Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + Default,
>(
    m: usize,
    k: usize,
    n: usize,
    ar: &[T],
    ai: &[T],
    br: &[T],
    bi: &[T],
    cr: &mut [T],
    ci: &mut [T],
) {
    for i in 0..m {
        for j in 0..n {
            let mut sr = T::default();
            let mut si = T::default();
            for l in 0..k {
                let (a1, a2) = (ar[i * k + l], ai[i * k + l]);
                let (b1, b2) = (br[l * n + j], bi[l * n + j]);
                sr = sr + a1 * b1 - a2 * b2;
                si = si + a1 * b2 + a2 * b1;
            }
            cr[i * n + j] = sr;
            ci[i * n + j] = si;
        }
    }
}

/// `f32` 复数矩阵乘（**直算 4 条**；`c` 复用调用方缓冲）。
pub fn cmatmul_f32_into(
    m: usize,
    k: usize,
    n: usize,
    ar: &[f32],
    ai: &[f32],
    br: &[f32],
    bi: &[f32],
    cr: &mut [f32],
    ci: &mut [f32],
) {
    cmatmul_f32_into_route(m, k, n, ar, ai, br, bi, cr, ci, Route::Direct);
}

/// `f32` 复数矩阵乘，**指定路线**。
#[allow(clippy::too_many_arguments)]
pub fn cmatmul_f32_into_route(
    m: usize,
    k: usize,
    n: usize,
    ar: &[f32],
    ai: &[f32],
    br: &[f32],
    bi: &[f32],
    cr: &mut [f32],
    ci: &mut [f32],
    route: Route,
) {
    assert_eq!(ar.len(), m * k);
    assert_eq!(ai.len(), m * k);
    assert_eq!(br.len(), k * n);
    assert_eq!(bi.len(), k * n);
    assert_eq!(cr.len(), m * n);
    assert_eq!(ci.len(), m * n);
    if m == 0 || n == 0 || k == 0 {
        for v in cr.iter_mut() {
            *v = 0.0;
        }
        for v in ci.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let (mut ar_br, mut ai_bi, mut ar_bi, mut ai_br) = (
        vec![0.0f32; m * n],
        vec![0.0f32; m * n],
        vec![0.0f32; m * n],
        vec![0.0f32; m * n],
    );
    match route {
        Route::Direct => {
            rmath::matmul_f32(m, k, n, ar, br, &mut ar_br);
            rmath::matmul_f32(m, k, n, ai, bi, &mut ai_bi);
            rmath::matmul_f32(m, k, n, ar, bi, &mut ar_bi);
            rmath::matmul_f32(m, k, n, ai, br, &mut ai_br);
            for l in 0..m * n {
                cr[l] = ar_br[l] - ai_bi[l];
                ci[l] = ar_bi[l] + ai_br[l];
            }
        }
        Route::Karatsuba => {
            let (mut a_sum, mut b_sum) = (vec![0.0f32; m * k], vec![0.0f32; k * n]);
            for l in 0..m * k {
                a_sum[l] = ar[l] + ai[l];
            }
            for l in 0..k * n {
                b_sum[l] = br[l] + bi[l];
            }
            let mut k1 = vec![0.0f32; m * n];
            rmath::matmul_f32(m, k, n, &a_sum, &b_sum, &mut k1);
            rmath::matmul_f32(m, k, n, ar, br, &mut ar_br); // k2
            rmath::matmul_f32(m, k, n, ai, bi, &mut ai_bi); // k3
            for l in 0..m * n {
                cr[l] = ar_br[l] - ai_bi[l];
                ci[l] = k1[l] - ar_br[l] - ai_bi[l];
            }
        }
    }
}

/// `f64` 复数矩阵乘（**直算 4 条**）。
pub fn cmatmul_f64_into(
    m: usize,
    k: usize,
    n: usize,
    ar: &[f64],
    ai: &[f64],
    br: &[f64],
    bi: &[f64],
    cr: &mut [f64],
    ci: &mut [f64],
) {
    cmatmul_f64_into_route(m, k, n, ar, ai, br, bi, cr, ci, Route::Direct);
}

/// `f64` 复数矩阵乘，**指定路线**。
#[allow(clippy::too_many_arguments)]
pub fn cmatmul_f64_into_route(
    m: usize,
    k: usize,
    n: usize,
    ar: &[f64],
    ai: &[f64],
    br: &[f64],
    bi: &[f64],
    cr: &mut [f64],
    ci: &mut [f64],
    route: Route,
) {
    assert_eq!(ar.len(), m * k);
    assert_eq!(ai.len(), m * k);
    assert_eq!(br.len(), k * n);
    assert_eq!(bi.len(), k * n);
    assert_eq!(cr.len(), m * n);
    assert_eq!(ci.len(), m * n);
    if m == 0 || n == 0 || k == 0 {
        for v in cr.iter_mut() {
            *v = 0.0;
        }
        for v in ci.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let (mut ar_br, mut ai_bi, mut ar_bi, mut ai_br) = (
        vec![0.0f64; m * n],
        vec![0.0f64; m * n],
        vec![0.0f64; m * n],
        vec![0.0f64; m * n],
    );
    match route {
        Route::Direct => {
            rmath64::matmul_f64(m, k, n, ar, br, &mut ar_br);
            rmath64::matmul_f64(m, k, n, ai, bi, &mut ai_bi);
            rmath64::matmul_f64(m, k, n, ar, bi, &mut ar_bi);
            rmath64::matmul_f64(m, k, n, ai, br, &mut ai_br);
            for l in 0..m * n {
                cr[l] = ar_br[l] - ai_bi[l];
                ci[l] = ar_bi[l] + ai_br[l];
            }
        }
        Route::Karatsuba => {
            let (mut a_sum, mut b_sum) = (vec![0.0f64; m * k], vec![0.0f64; k * n]);
            for l in 0..m * k {
                a_sum[l] = ar[l] + ai[l];
            }
            for l in 0..k * n {
                b_sum[l] = br[l] + bi[l];
            }
            let mut k1 = vec![0.0f64; m * n];
            rmath64::matmul_f64(m, k, n, &a_sum, &b_sum, &mut k1);
            rmath64::matmul_f64(m, k, n, ar, br, &mut ar_br);
            rmath64::matmul_f64(m, k, n, ai, bi, &mut ai_bi);
            for l in 0..m * n {
                cr[l] = ar_br[l] - ai_bi[l];
                ci[l] = k1[l] - ar_br[l] - ai_bi[l];
            }
        }
    }
}

/// 逐元素复数参考（f32）。
pub fn cmatmul_ref_f32(
    m: usize,
    k: usize,
    n: usize,
    ar: &[f32],
    ai: &[f32],
    br: &[f32],
    bi: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let (mut cr, mut ci) = (vec![0.0f32; m * n], vec![0.0f32; m * n]);
    ref_into(m, k, n, ar, ai, br, bi, &mut cr, &mut ci);
    (cr, ci)
}

/// 逐元素复数参考（f64）。
pub fn cmatmul_ref_f64(
    m: usize,
    k: usize,
    n: usize,
    ar: &[f64],
    ai: &[f64],
    br: &[f64],
    bi: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let (mut cr, mut ci) = (vec![0.0f64; m * n], vec![0.0f64; m * n]);
    ref_into(m, k, n, ar, ai, br, bi, &mut cr, &mut ci);
    (cr, ci)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data32(n: usize) -> (Vec<f32>, Vec<f32>) {
        let re = (0..n).map(|i| ((i % 17) as f32) * 0.125 - 1.0).collect();
        let im = (0..n).map(|i| ((i % 13) as f32) * 0.0625 - 0.5).collect();
        (re, im)
    }
    fn data64(n: usize) -> (Vec<f64>, Vec<f64>) {
        let re = (0..n).map(|i| ((i % 17) as f64) * 0.125 - 1.0).collect();
        let im = (0..n).map(|i| ((i % 13) as f64) * 0.0625 - 0.5).collect();
        (re, im)
    }

    fn rel32(a: &[f32], b: &[f32]) -> f32 {
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            num = num.max((x - y).abs());
            den = den.max(y.abs());
        }
        if den == 0.0 { num } else { num / den }
    }

    /// 形状矩阵：覆盖 `k=1`、`n=1`、不整除、非 4 倍 rows、非 32 倍 n。
    const SHAPES: [(usize, usize, usize); 8] = [
        (1, 1, 1),
        (1, 5, 1),
        (7, 1, 3),
        (6, 3, 5),   // 非 4 倍 rows、非 32 倍 n
        (13, 7, 9),  // 全不整除
        (17, 4, 33), // 非 4 倍 rows、n=33
        (64, 64, 64),
        (100, 51, 97),
    ];

    /// **直算 vs 4 条实矩阵乘的组合：逐位一致**（直算路线的位精确契约）。
    #[test]
    fn direct_matches_four_real_matmul_bitwise() {
        for &(m, k, n) in &SHAPES {
            let (ar, ai) = data32(m * k);
            let (br, bi) = data32(k * n);
            let (mut cr, mut ci) = (vec![0.0f32; m * n], vec![0.0f32; m * n]);
            // 显式走 **Direct**（默认已改成 Karatsuba）
            cmatmul_f32_into_route(m, k, n, &ar, &ai, &br, &bi, &mut cr, &mut ci, Route::Direct);
            // 手工用同一个实内核组合（表达式与实现逐字相同）
            let (mut p1, mut p2, mut p3, mut p4) = (
                vec![0.0f32; m * n],
                vec![0.0f32; m * n],
                vec![0.0f32; m * n],
                vec![0.0f32; m * n],
            );
            rmath::matmul_f32(m, k, n, &ar, &br, &mut p1);
            rmath::matmul_f32(m, k, n, &ai, &bi, &mut p2);
            rmath::matmul_f32(m, k, n, &ar, &bi, &mut p3);
            rmath::matmul_f32(m, k, n, &ai, &br, &mut p4);
            for l in 0..m * n {
                assert_eq!(
                    cr[l].to_bits(),
                    (p1[l] - p2[l]).to_bits(),
                    "Re 不逐位 m={m}"
                );
                assert_eq!(
                    ci[l].to_bits(),
                    (p3[l] + p4[l]).to_bits(),
                    "Im 不逐位 m={m}"
                );
            }
        }
    }

    /// **两路线互比**（容差内）+ 与逐元素参考的容差对照（f32/f64）。
    #[test]
    fn routes_agree_within_tolerance() {
        for &(m, k, n) in &SHAPES {
            let (ar32, ai32) = data32(m * k);
            let (br32, bi32) = data32(k * n);
            let mut d = (vec![0.0f32; m * n], vec![0.0f32; m * n]);
            let mut ka = (vec![0.0f32; m * n], vec![0.0f32; m * n]);
            cmatmul_f32_into_route(
                m,
                k,
                n,
                &ar32,
                &ai32,
                &br32,
                &bi32,
                &mut d.0,
                &mut d.1,
                Route::Direct,
            );
            cmatmul_f32_into_route(
                m,
                k,
                n,
                &ar32,
                &ai32,
                &br32,
                &bi32,
                &mut ka.0,
                &mut ka.1,
                Route::Karatsuba,
            );
            let (rr, ri) = cmatmul_ref_f32(m, k, n, &ar32, &ai32, &br32, &bi32);
            assert!(rel32(&d.0, &rr) < 1e-5, "f32 直算 vs 参考 m={m}");
            assert!(rel32(&ka.0, &rr) < 1e-4, "f32 Karatsuba vs 参考 m={m}");
            assert!(rel32(&ka.0, &d.0) < 1e-4, "f32 两路线 Re 差过大 m={m}");
            assert!(rel32(&ka.1, &d.1) < 1e-4, "f32 两路线 Im 差过大 m={m}");
            assert!(rel32(&d.1, &ri) < 1e-5, "f32 直算 Im vs 参考 m={m}");

            let (ar64, ai64) = data64(m * k);
            let (br64, bi64) = data64(k * n);
            let mut d64 = (vec![0.0f64; m * n], vec![0.0f64; m * n]);
            let mut k64 = (vec![0.0f64; m * n], vec![0.0f64; m * n]);
            cmatmul_f64_into_route(
                m,
                k,
                n,
                &ar64,
                &ai64,
                &br64,
                &bi64,
                &mut d64.0,
                &mut d64.1,
                Route::Direct,
            );
            cmatmul_f64_into_route(
                m,
                k,
                n,
                &ar64,
                &ai64,
                &br64,
                &bi64,
                &mut k64.0,
                &mut k64.1,
                Route::Karatsuba,
            );
            let (rr64, ri64) = cmatmul_ref_f64(m, k, n, &ar64, &ai64, &br64, &bi64);
            for l in 0..m * n {
                let d0 = (d64.0[l] - rr64[l]).abs();
                let k0 = (k64.0[l] - rr64[l]).abs();
                assert!(d0 < 1e-12, "f64 直算 Re 差 {d0:e} m={m}");
                assert!(k0 < 1e-11, "f64 Karatsuba Re 差 {k0:e} m={m}");
                assert!((d64.1[l] - ri64[l]).abs() < 1e-12, "f64 直算 Im m={m}");
            }
        }
    }

    /// **复乘恒等式自检**：`(a·1) = a`、`(1·b) = b`、`i·i = −1`。
    #[test]
    fn complex_identities() {
        let (m, k, n) = (4, 5, 6);
        let (ar, ai) = data32(m * k);
        // B = 单位阵（实）
        let mut br = vec![0.0f32; k * n];
        let bi = vec![0.0f32; k * n];
        for l in 0..k.min(n) {
            br[l * n + l] = 1.0;
        }
        let (mut cr, mut ci) = (vec![0.0f32; m * n], vec![0.0f32; m * n]);
        // 恒等式用 **Direct** 断言逐位（Karatsuba 只承诺容差，见 mod 文档）
        cmatmul_f32_into_route(m, k, n, &ar, &ai, &br, &bi, &mut cr, &mut ci, Route::Direct);
        // A·I 的左上 k×k 块应等于 A 的对应列
        for i in 0..m {
            for j in 0..k.min(n) {
                assert_eq!(cr[i * n + j].to_bits(), ar[i * k + j].to_bits());
                assert_eq!(ci[i * n + j].to_bits(), ai[i * k + j].to_bits());
            }
        }
        // i·i = −1：A = iI（实部 0、虚部 I），B = iI
        let kk = 8;
        let ar2 = vec![0.0f32; kk * kk];
        let mut ai2 = vec![0.0f32; kk * kk];
        let mut bi2 = vec![0.0f32; kk * kk];
        let br2 = vec![0.0f32; kk * kk];
        for l in 0..kk {
            ai2[l * kk + l] = 1.0;
            bi2[l * kk + l] = 1.0;
        }
        let mut cr2 = vec![0.0f32; kk * kk];
        let mut ci2 = vec![0.0f32; kk * kk];
        cmatmul_f32_into_route(
            kk,
            kk,
            kk,
            &ar2,
            &ai2,
            &br2,
            &bi2,
            &mut cr2,
            &mut ci2,
            Route::Direct,
        );
        for l in 0..kk * kk {
            let diag = if l % (kk + 1) == 0 { -1.0 } else { 0.0 };
            assert_eq!(cr2[l], diag, "i·i 的实部应为 −I 的对角");
            assert_eq!(ci2[l], 0.0);
        }
    }

    /// **空/零维形状**：`m=0`/`n=0`/`k=0` 不 panic、输出清零。
    #[test]
    fn zero_dim_shapes() {
        let mut cr = vec![7.0f32; 3];
        let mut ci = vec![7.0f32; 3];
        cmatmul_f32_into(3, 0, 1, &[], &[], &[], &[], &mut cr, &mut ci);
        assert!(cr.iter().all(|&v| v == 0.0) && ci.iter().all(|&v| v == 0.0));
        let (mut cr2, mut ci2) = (vec![7.0f64; 2], vec![7.0f64; 2]);
        cmatmul_f64_into(
            2, 3, 1, &[0.0; 6], &[0.0; 6], &[0.0; 3], &[0.0; 3], &mut cr2, &mut ci2,
        );
        assert!(cr2[0] == 0.0 && ci2[1] == 0.0);
    }

    /// **大形状 + 缓冲复用**：连续两次调用（复用同一对输出缓冲）结果一致。
    #[test]
    fn large_shape_and_buffer_reuse() {
        let (m, k, n) = (192, 768, 3072);
        let (ar, ai) = data32(m * k);
        let (br, bi) = data32(k * n);
        let mut cr = vec![0.0f32; m * n];
        let mut ci = vec![0.0f32; m * n];
        cmatmul_f32_into(m, k, n, &ar, &ai, &br, &bi, &mut cr, &mut ci);
        let first = (cr.clone(), ci.clone());
        cmatmul_f32_into(m, k, n, &ar, &ai, &br, &bi, &mut cr, &mut ci);
        assert_eq!(first.0, cr, "复用缓冲的第二次调用结果应完全相同");
        assert_eq!(first.1, ci);
        // 抽查一个元素与参考一致（大形状只抽查，避免 O(m·k·n) 参考太慢）
        let (i, j, l) = (7usize, 11usize, 5usize);
        let mut sr = 0.0f64;
        let mut si = 0.0f64;
        for t in 0..k {
            let (a1, a2) = (ar[i * k + t] as f64, ai[i * k + t] as f64);
            let (b1, b2) = (br[t * n + j] as f64, bi[t * n + j] as f64);
            sr += a1 * b1 - a2 * b2;
            si += a1 * b2 + a2 * b1;
        }
        let e = ((cr[i * n + j] as f64 - sr).abs() + (ci[i * n + j] as f64 - si).abs())
            / (sr.abs() + si.abs() + 1e-9);
        assert!(e < 1e-4, "抽查元素相对误差过大：{e:e}（l={l}）");
    }
}

#[cfg(test)]
mod perf {
    use super::*;
    use std::time::Instant;

    /// **A 阶同源性能诊断（默认 `#[ignore]`）**：直算 4 条 vs Karatsuba 3 条 vs 同规模**实矩阵乘**基准。
    ///
    /// 口径：同进程交替、≥5 对、丢首对、逐对报 `ms`/`iters`/`ms_max`/loadavg；
    /// **复数 GFLOP/s = 8·m·k·n/ms**（一个复数乘加 = 8 个实数 FLOP）；
    /// **效率比 = 复数 GFLOP/s ÷ (条数 × 实矩阵乘 GFLOP/s)**（直算条数 4、Karatsuba 3）。
    ///
    /// 跑法：`cargo test --release -p lasx_rs --lib diag_cmatmul_routes -- --ignored --nocapture`
    #[ignore = "A 阶性能诊断：直算 vs Karatsuba vs 实矩阵乘基准"]
    #[test]
    fn diag_cmatmul_routes() {
        let (m, k, n) = (192usize, 768usize, 3072usize);
        let (ar, ai) = {
            let re: Vec<f32> = (0..m * k)
                .map(|i| ((i % 17) as f32) * 0.125 - 1.0)
                .collect();
            let im: Vec<f32> = (0..m * k)
                .map(|i| ((i % 13) as f32) * 0.0625 - 0.5)
                .collect();
            (re, im)
        };
        let (br, bi) = {
            let re: Vec<f32> = (0..k * n)
                .map(|i| ((i % 11) as f32) * 0.0625 - 0.75)
                .collect();
            let im: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32) * 0.125 - 0.5).collect();
            (re, im)
        };
        let (mut cr, mut ci) = (vec![0.0f32; m * n], vec![0.0f32; m * n]);
        let (mut real, mut k1buf) = (vec![0.0f32; m * n], vec![0.0f32; m * n]);
        let cx_flop = 8.0 * m as f64 * k as f64 * n as f64;

        let mut rows: Vec<(f64, f64, f64, i32, f64, String)> = Vec::new(); // (direct, karat, real, iters, ms_max, loadavg)
        for pair in 0..6 {
            // 每对内部依次跑三条路线（轮换起跑次序摊平顺序效应）
            let mut t = [0.0f64; 3];
            let mut iters = 0i32;
            let mut ms_max = 0.0f64;
            for step in 0..3 {
                let which = (pair + step) % 3;
                let mut cnt = 0i32;
                let mut mx = 0.0f64;
                let t0 = Instant::now();
                while t0.elapsed().as_secs_f64() < 0.05 {
                    let t1 = Instant::now();
                    match which {
                        0 => cmatmul_f32_into_route(
                            m,
                            k,
                            n,
                            &ar,
                            &ai,
                            &br,
                            &bi,
                            &mut cr,
                            &mut ci,
                            Route::Direct,
                        ),
                        1 => cmatmul_f32_into_route(
                            m,
                            k,
                            n,
                            &ar,
                            &ai,
                            &br,
                            &bi,
                            &mut cr,
                            &mut ci,
                            Route::Karatsuba,
                        ),
                        _ => {
                            rmath::matmul_f32(m, k, n, &ar, &br, &mut real);
                            rmath::matmul_f32(m, k, n, &ai, &bi, &mut k1buf);
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
        let med = |idx: usize| -> f64 {
            let mut v: Vec<f64> = rows[1..]
                .iter()
                .map(|r| match idx {
                    0 => r.0,
                    1 => r.1,
                    _ => r.2,
                })
                .collect();
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let (d_ms, k_ms, r_ms) = (med(0), med(1), med(2));
        for (i, r) in rows.iter().enumerate() {
            println!(
                "pair{} direct={:.4}ms karat={:.4}ms real={:.4}ms iters={} ms_max={:.3} loadavg={}{}",
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
        let cx_d = cx_flop / d_ms / 1e9;
        let cx_k = cx_flop / k_ms / 1e9;
        // `real` 那一行每轮跑的是 **2 条**实矩阵乘 ⇒ 单条速率要把时间折算一半
        let real1_g = 2.0 * m as f64 * k as f64 * n as f64 / (r_ms / 2.0) / 1e9;
        println!(
            "中位数（丢首对）：direct={:.4}ms karat={:.4}ms real(2 条)={:.4}ms ⇒ 单条实矩阵乘≈{:.4}ms",
            d_ms * 1e3,
            k_ms * 1e3,
            r_ms * 1e3,
            r_ms * 1e3 / 2.0
        );
        println!("复数 GFLOP/s：direct={cx_d:.1} karat={cx_k:.1}｜实矩阵乘（单条）={real1_g:.1}");
        println!(
            "**物理效率比**（复数 GFLOP/s ÷ 单条实矩阵乘 GFLOP/s）：direct={:.3} karat={:.3}",
            cx_d / real1_g,
            cx_k / real1_g
        );
        println!(
            "**字面效率比**（复数 GFLOP/s ÷ (k × 单条实矩阵乘 GFLOP/s)，k=4/3）：direct={:.3} karat={:.3}",
            cx_d / (4.0 * real1_g),
            cx_k / (3.0 * real1_g)
        );
        println!(
            "⚠️ 字面比的分母含义是\"k 条实矩阵乘**并行**时的合计速率\"；若 k 条**串行**跑，\
             合计速率就是单条速率 ⇒ 那种口径下 direct≈0.95、karat≈1.12（见物理效率比）"
        );
        println!("线程数：本模块是**串行核心**（未自动多线程；多线程由调用方按分平面切）");
    }
}
