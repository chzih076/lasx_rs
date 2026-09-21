//! `lasx_unitize3_batch` —— 批量三维单位化 `o = v / |v|`（f64 SOA）。
//!
//! 模长与 [`crate::ops::norm3_batch`] **同结合次序**（`x² + (y² + z²)`），
//! 因此 `unitize3_batch` 与 "`norm3_batch` 再逐分量除" 逐位一致。
//!
//! **零向量约定**：`|v| == 0` 时输出 `(0, 0, 0)`（而不是 `0 · ∞ = NaN`）。
//! 这一条与"朴素标量公式"不同，是与调用方约定的安全行为；三条路径同规则。
//!
//! LASX 缺失时走 LSX 128 位路径。

use crate::arch::SimdPath;
use crate::arch::{lasx, lsx};
use std::arch::loongarch64::*;

/// 算子入口：解析当前线程的向量路径后分派。
#[inline]
pub(crate) fn unitize3_batch(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    ox: &mut [f64],
    oy: &mut [f64],
    oz: &mut [f64],
) {
    match SimdPath::detect() {
        SimdPath::Lasx => unitize3_batch_lasx(x, y, z, ox, oy, oz),
        SimdPath::Lsx => unitize3_batch_lsx(x, y, z, ox, oy, oz),
    }
}

/// 模长平方（与 `norm3_batch` 同结合）。
#[inline]
fn sq_lasx(vx: m256d, vy: m256d, vz: m256d) -> m256d {
    unsafe {
        lasx_xvfadd_d(
            lasx_xvfmul_d(vx, vx),
            lasx_xvfadd_d(lasx_xvfmul_d(vy, vy), lasx_xvfmul_d(vz, vz)),
        )
    }
}

/// `1/n`，且 `n == 0` 的 lane 精确置 0（用 `andn(mask, inv)`，见 perf-report 的探针）。
#[inline]
fn recip_masked_lasx(n: m256d) -> m256d {
    unsafe {
        let zero = lasx::zero_f64x4();
        let one = lasx::splat_f64(1.0);
        let inv = lasx_xvfdiv_d(one, n);
        let mask = lasx_xvfcmp_ceq_d(n, zero);
        std::mem::transmute(lasx_xvandn_v(
            mask,
            std::mem::transmute::<m256d, m256i>(inv),
        ))
    }
}

/// LASX 256 位实现（4 样本/向量）。
#[inline]
fn unitize3_batch_lasx(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    ox: &mut [f64],
    oy: &mut [f64],
    oz: &mut [f64],
) {
    let n = x.len();
    let mut i = 0;
    while i + 4 <= n {
        unsafe {
            let vx = lasx::load_f64x4(x.as_ptr().add(i));
            let vy = lasx::load_f64x4(y.as_ptr().add(i));
            let vz = lasx::load_f64x4(z.as_ptr().add(i));
            let inv = recip_masked_lasx(lasx_xvfsqrt_d(sq_lasx(vx, vy, vz)));
            lasx::store_f64x4(ox.as_mut_ptr().add(i), lasx_xvfmul_d(vx, inv));
            lasx::store_f64x4(oy.as_mut_ptr().add(i), lasx_xvfmul_d(vy, inv));
            lasx::store_f64x4(oz.as_mut_ptr().add(i), lasx_xvfmul_d(vz, inv));
        }
        i += 4;
    }
    tail(x, y, z, ox, oy, oz, i);
}

/// LSX 128 位实现（2 样本/向量）。
#[inline]
fn unitize3_batch_lsx(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    ox: &mut [f64],
    oy: &mut [f64],
    oz: &mut [f64],
) {
    let n = x.len();
    let mut i = 0;
    while i + 2 <= n {
        unsafe {
            let vx = lsx::load_f64x2(x.as_ptr().add(i));
            let vy = lsx::load_f64x2(y.as_ptr().add(i));
            let vz = lsx::load_f64x2(z.as_ptr().add(i));
            let sq = lsx_vfadd_d(
                lsx_vfmul_d(vx, vx),
                lsx_vfadd_d(lsx_vfmul_d(vy, vy), lsx_vfmul_d(vz, vz)),
            );
            let mag = lsx_vfsqrt_d(sq);
            let zero: m128d = std::mem::transmute(lsx_vldi(0));
            let one = lsx::splat_f64(1.0);
            let inv = lsx_vfdiv_d(one, mag);
            let mask = lsx_vfcmp_ceq_d(mag, zero);
            let inv: m128d =
                std::mem::transmute(lsx_vandn_v(mask, std::mem::transmute::<m128d, m128i>(inv)));
            lsx::store_f64x2(ox.as_mut_ptr().add(i), lsx_vfmul_d(vx, inv));
            lsx::store_f64x2(oy.as_mut_ptr().add(i), lsx_vfmul_d(vy, inv));
            lsx::store_f64x2(oz.as_mut_ptr().add(i), lsx_vfmul_d(vz, inv));
        }
        i += 2;
    }
    tail(x, y, z, ox, oy, oz, i);
}

/// 标量尾（与向量路径同结合、同零向量规则）。
#[inline]
fn tail(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    ox: &mut [f64],
    oy: &mut [f64],
    oz: &mut [f64],
    from: usize,
) {
    for j in from..x.len() {
        // 与 norm3_batch 逐位一致：x² + (y² + z²)
        let mag = (x[j] * x[j] + (y[j] * y[j] + z[j] * z[j])).sqrt();
        let inv = if mag == 0.0 { 0.0 } else { 1.0 / mag };
        ox[j] = x[j] * inv;
        oy[j] = y[j] * inv;
        oz[j] = z[j] * inv;
    }
}

/// 数值回归测试。
#[cfg(test)]
mod tests {
    use crate::ffi::attitude::lasx_unitize3_batch;
    use crate::ffi::batch::lasx_norm3_batch;
    use crate::ops::testutil::{rel_err, states};

    #[test]
    fn test_unitize3_batch_bit_exact() {
        for n in [0usize, 1, 2, 3, 4, 5, 8, 9, 33] {
            let (x, y, z) = states(n);
            let (mut ox, mut oy, mut oz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            lasx_unitize3_batch(
                x.as_ptr(),
                y.as_ptr(),
                z.as_ptr(),
                ox.as_mut_ptr(),
                oy.as_mut_ptr(),
                oz.as_mut_ptr(),
                n as i32,
            );
            // 与 "norm3_batch 再逐分量除" 逐位一致（两者结合次序相同）
            let mut mag = vec![0.0; n];
            lasx_norm3_batch(
                x.as_ptr(),
                y.as_ptr(),
                z.as_ptr(),
                mag.as_mut_ptr(),
                n as i32,
            );
            for i in 0..n {
                if mag[i] == 0.0 {
                    assert_eq!((ox[i], oy[i], oz[i]), (0.0, 0.0, 0.0), "零向量约定");
                    continue;
                }
                let inv = 1.0 / mag[i];
                assert_eq!(ox[i].to_bits(), (x[i] * inv).to_bits(), "n={n} i={i} ox");
                assert_eq!(oy[i].to_bits(), (y[i] * inv).to_bits(), "n={n} i={i} oy");
                assert_eq!(oz[i].to_bits(), (z[i] * inv).to_bits(), "n={n} i={i} oz");
                // 单位化后模长为 1（相对误差量级 1e-15）
                let m = (ox[i] * ox[i] + (oy[i] * oy[i] + oz[i] * oz[i])).sqrt();
                assert!(rel_err(m, 1.0) < 1e-9, "n={n} i={i}: |o| = {m}");
            }
        }
    }

    /// 零向量必须给出 (0,0,0) 而不是 NaN。
    #[test]
    fn test_unitize3_zero_vector() {
        let x = [0.0f64, 3.0, 0.0];
        let y = [0.0f64, 4.0, 0.0];
        let z = [0.0f64, 0.0, 0.0];
        let (mut ox, mut oy, mut oz) = ([9.9f64; 3], [9.9f64; 3], [9.9f64; 3]);
        lasx_unitize3_batch(
            x.as_ptr(),
            y.as_ptr(),
            z.as_ptr(),
            ox.as_mut_ptr(),
            oy.as_mut_ptr(),
            oz.as_mut_ptr(),
            3,
        );
        assert_eq!((ox[0], oy[0], oz[0]), (0.0, 0.0, 0.0), "零向量应为 (0,0,0)");
        // 内核按 v·(1/|v|) 算，期望值也这么写（3·(1/5) 并非字面量 0.6）
        let expect = (3.0f64 * (1.0f64 / 5.0), 4.0f64 * (1.0f64 / 5.0), 0.0);
        assert_eq!((ox[1], oy[1], oz[1]), expect, "3-4-5 应单位化");
        assert_eq!((ox[2], oy[2], oz[2]), (0.0, 0.0, 0.0), "零向量应为 (0,0,0)");
    }
}
