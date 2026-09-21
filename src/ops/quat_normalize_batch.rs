//! `lasx_quat_normalize_batch` —— 批量单位化四元数（f64 SOA，**原地**）。
//!
//! 约定与 loong-sci `attitude::Quat` 一致：**标量在前** `q = w + xi + yj + zk`，
//! 范数按左结合 `((w² + x²) + y²) + z²` 求，且
//!
//! - `|q| < 1e-15` ⇒ 该样本输出**单位四元数** `(1, 0, 0, 0)`（不是 NaN）；
//! - 否则逐分量**相除** `q / |q|`（不是乘倒数——与参考实现逐位一致）。
//!
//! LASX 缺失时走 LSX 128 位路径。

use crate::arch::SimdPath;
use crate::arch::{lasx, lsx};
use std::arch::loongarch64::*;

/// 判定"退化四元数"的阈值，与 loong-sci `quat_normalize` 相同。
pub(super) const DEGENERATE: f64 = 1e-15;

/// 算子入口：解析当前线程的向量路径后分派。
#[inline]
pub(crate) fn quat_normalize_batch(qw: &mut [f64], qx: &mut [f64], qy: &mut [f64], qz: &mut [f64]) {
    match SimdPath::detect() {
        SimdPath::Lasx => quat_normalize_batch_lasx(qw, qx, qy, qz),
        SimdPath::Lsx => quat_normalize_batch_lsx(qw, qx, qy, qz),
    }
}

/// LASX：把 4 个分量向量单位化（供本文件与 [`super::quat_to_dcm_batch`] 复用）。
#[inline]
pub(super) fn normalize_lasx(
    w: m256d,
    x: m256d,
    y: m256d,
    z: m256d,
) -> (m256d, m256d, m256d, m256d) {
    unsafe {
        // 左结合，与 `Quat::norm` 逐位一致
        let sq = lasx_xvfadd_d(
            lasx_xvfadd_d(
                lasx_xvfadd_d(lasx_xvfmul_d(w, w), lasx_xvfmul_d(x, x)),
                lasx_xvfmul_d(y, y),
            ),
            lasx_xvfmul_d(z, z),
        );
        let n = lasx_xvfsqrt_d(sq);
        let tiny = lasx::splat_f64(DEGENERATE);
        let mask = lasx_xvfcmp_clt_d(n, tiny);
        let bits = |v: m256d| std::mem::transmute::<m256d, m256i>(v);
        let unbits = |v: m256i| std::mem::transmute::<m256i, m256d>(v);
        // 退化 lane：andn 清成 +0.0；w 再 or 上 1.0 的位型 ⇒ (1,0,0,0)
        let wo = unbits(lasx_xvor_v(
            lasx_xvandn_v(mask, bits(lasx_xvfdiv_d(w, n))),
            lasx_xvand_v(mask, one_bits()),
        ));
        let xo = unbits(lasx_xvandn_v(mask, bits(lasx_xvfdiv_d(x, n))));
        let yo = unbits(lasx_xvandn_v(mask, bits(lasx_xvfdiv_d(y, n))));
        let zo = unbits(lasx_xvandn_v(mask, bits(lasx_xvfdiv_d(z, n))));
        (wo, xo, yo, zo)
    }
}

/// `1.0f64` 的位型向量。
#[inline]
fn one_bits() -> m256i {
    unsafe { std::mem::transmute::<m256d, m256i>(lasx::splat_f64(1.0)) }
}

/// LASX 256 位实现（4 样本/向量）。
#[inline]
fn quat_normalize_batch_lasx(qw: &mut [f64], qx: &mut [f64], qy: &mut [f64], qz: &mut [f64]) {
    let n = qw.len();
    let mut i = 0;
    while i + 4 <= n {
        unsafe {
            let w = lasx::load_f64x4(qw.as_ptr().add(i));
            let x = lasx::load_f64x4(qx.as_ptr().add(i));
            let y = lasx::load_f64x4(qy.as_ptr().add(i));
            let z = lasx::load_f64x4(qz.as_ptr().add(i));
            let (wo, xo, yo, zo) = normalize_lasx(w, x, y, z);
            lasx::store_f64x4(qw.as_mut_ptr().add(i), wo);
            lasx::store_f64x4(qx.as_mut_ptr().add(i), xo);
            lasx::store_f64x4(qy.as_mut_ptr().add(i), yo);
            lasx::store_f64x4(qz.as_mut_ptr().add(i), zo);
        }
        i += 4;
    }
    tail(qw, qx, qy, qz, i);
}

/// LSX 128 位实现（2 样本/向量）。
#[inline]
fn quat_normalize_batch_lsx(qw: &mut [f64], qx: &mut [f64], qy: &mut [f64], qz: &mut [f64]) {
    let n = qw.len();
    let mut i = 0;
    while i + 2 <= n {
        unsafe {
            let w = lsx::load_f64x2(qw.as_ptr().add(i));
            let x = lsx::load_f64x2(qx.as_ptr().add(i));
            let y = lsx::load_f64x2(qy.as_ptr().add(i));
            let z = lsx::load_f64x2(qz.as_ptr().add(i));
            let sq = lsx_vfadd_d(
                lsx_vfadd_d(
                    lsx_vfadd_d(lsx_vfmul_d(w, w), lsx_vfmul_d(x, x)),
                    lsx_vfmul_d(y, y),
                ),
                lsx_vfmul_d(z, z),
            );
            let mag = lsx_vfsqrt_d(sq);
            let tiny = lsx::splat_f64(DEGENERATE);
            let mask = lsx_vfcmp_clt_d(mag, tiny);
            let one = lsx::splat_f64(1.0);
            let bits = |v: m128d| std::mem::transmute::<m128d, m128i>(v);
            let unbits = |v: m128i| std::mem::transmute::<m128i, m128d>(v);
            let wo = unbits(lsx_vor_v(
                lsx_vandn_v(mask, bits(lsx_vfdiv_d(w, mag))),
                lsx_vand_v(mask, std::mem::transmute::<m128d, m128i>(one)),
            ));
            let xo = unbits(lsx_vandn_v(mask, bits(lsx_vfdiv_d(x, mag))));
            let yo = unbits(lsx_vandn_v(mask, bits(lsx_vfdiv_d(y, mag))));
            let zo = unbits(lsx_vandn_v(mask, bits(lsx_vfdiv_d(z, mag))));
            lsx::store_f64x2(qw.as_mut_ptr().add(i), wo);
            lsx::store_f64x2(qx.as_mut_ptr().add(i), xo);
            lsx::store_f64x2(qy.as_mut_ptr().add(i), yo);
            lsx::store_f64x2(qz.as_mut_ptr().add(i), zo);
        }
        i += 2;
    }
    tail(qw, qx, qy, qz, i);
}

/// 标量尾（与向量路径同结合、同退化规则、同为"相除"）。
#[inline]
fn tail(qw: &mut [f64], qx: &mut [f64], qy: &mut [f64], qz: &mut [f64], from: usize) {
    for i in from..qw.len() {
        let n = ((qw[i] * qw[i] + qx[i] * qx[i]) + qy[i] * qy[i]) + qz[i] * qz[i];
        let n = n.sqrt();
        if n < DEGENERATE {
            qw[i] = 1.0;
            qx[i] = 0.0;
            qy[i] = 0.0;
            qz[i] = 0.0;
        } else {
            qw[i] /= n;
            qx[i] /= n;
            qy[i] /= n;
            qz[i] /= n;
        }
    }
}

/// 数值回归测试。
#[cfg(test)]
mod tests {
    use crate::ffi::attitude::lasx_quat_normalize_batch;
    use crate::ops::testutil::Lcg;

    #[test]
    fn test_quat_normalize_batch_matches_reference() {
        let mut rng = Lcg(0x4a11_0c05);
        for n in [0usize, 1, 2, 3, 4, 5, 8, 9, 33] {
            let (mut w, mut x, mut y, mut z) = (
                (0..n).map(|_| rng.f64() * 2.0).collect::<Vec<f64>>(),
                (0..n).map(|_| rng.f64()).collect::<Vec<f64>>(),
                (0..n).map(|_| rng.f64()).collect::<Vec<f64>>(),
                (0..n).map(|_| rng.f64()).collect::<Vec<f64>>(),
            );
            // 参考值（与 loong-sci `quat_normalize` 同公式、同顺序、同为相除）
            let want: Vec<(f64, f64, f64, f64)> = (0..n)
                .map(|i| {
                    let nn = ((w[i] * w[i] + x[i] * x[i]) + y[i] * y[i]) + z[i] * z[i];
                    let nn = nn.sqrt();
                    if nn < 1e-15 {
                        (1.0, 0.0, 0.0, 0.0)
                    } else {
                        (w[i] / nn, x[i] / nn, y[i] / nn, z[i] / nn)
                    }
                })
                .collect();
            lasx_quat_normalize_batch(
                w.as_mut_ptr(),
                x.as_mut_ptr(),
                y.as_mut_ptr(),
                z.as_mut_ptr(),
                n as i32,
            );
            for i in 0..n {
                assert_eq!(w[i].to_bits(), want[i].0.to_bits(), "n={n} i={i} w");
                assert_eq!(x[i].to_bits(), want[i].1.to_bits(), "n={n} i={i} x");
                assert_eq!(y[i].to_bits(), want[i].2.to_bits(), "n={n} i={i} y");
                assert_eq!(z[i].to_bits(), want[i].3.to_bits(), "n={n} i={i} z");
            }
        }
    }

    /// 退化四元数（范数 < 1e-15）必须变成单位四元数，而不是 NaN。
    #[test]
    fn test_quat_normalize_degenerate_is_identity() {
        let mut w = vec![0.0f64, 1e-20, 3.0, 0.0];
        let mut x = vec![0.0f64, 0.0, 4.0, 0.0];
        let mut y = vec![0.0f64, 1e-20, 0.0, 0.0];
        let mut z = vec![0.0f64, 0.0, 0.0, -0.0];
        lasx_quat_normalize_batch(
            w.as_mut_ptr(),
            x.as_mut_ptr(),
            y.as_mut_ptr(),
            z.as_mut_ptr(),
            4,
        );
        for i in [0usize, 1] {
            assert_eq!(
                (w[i], x[i], y[i], z[i]),
                (1.0, 0.0, 0.0, 0.0),
                "退化样本 {i} 应为单位四元数"
            );
        }
        // 归一化后模长为 1
        for i in 2..4 {
            let m = ((w[i] * w[i] + x[i] * x[i]) + y[i] * y[i]) + z[i] * z[i];
            assert!((m.sqrt() - 1.0).abs() < 1e-15, "i={i} 模长 {}", m.sqrt());
        }
    }
}
