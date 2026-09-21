//! `lasx_mat3_mul_vec3_batch` —— 批量 3×3 矩阵乘三维向量 `o = M·v`（f64 SOA）。
//!
//! `m0..m8` 是**行主序**的 3×3 矩阵（`M[r][c] = m[3r + c]`），与之配套的
//! [`crate::ops::quat_to_dcm_batch`] 输出同样是行主序，两者可直接串起来用。
//!
//! 分量的结合次序固定为左结合 `((m0·x + m1·y) + m2·z)`，三条路径一致 ⇒ 逐位一致。
//!
//! LASX 缺失时走 LSX 128 位路径。

use crate::arch::SimdPath;
use crate::arch::{lasx, lsx};
use std::arch::loongarch64::*;

/// 算子入口：解析当前线程的向量路径后分派。
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn mat3_mul_vec3_batch(
    m: [&[f64]; 9],
    x: &[f64],
    y: &[f64],
    z: &[f64],
    ox: &mut [f64],
    oy: &mut [f64],
    oz: &mut [f64],
) {
    match SimdPath::detect() {
        SimdPath::Lasx => mat3_mul_vec3_batch_lasx(m, x, y, z, ox, oy, oz),
        SimdPath::Lsx => mat3_mul_vec3_batch_lsx(m, x, y, z, ox, oy, oz),
    }
}

/// LASX 256 位实现（4 样本/向量）。
#[inline]
#[allow(clippy::too_many_arguments)]
fn mat3_mul_vec3_batch_lasx(
    m: [&[f64]; 9],
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
            let mut rows: [m256d; 3] = [lasx::zero_f64x4(); 3];
            for (r, out) in rows.iter_mut().enumerate() {
                let mut acc = lasx_xvfmul_d(lasx::load_f64x4(m[r * 3].as_ptr().add(i)), vx);
                acc = lasx_xvfadd_d(
                    acc,
                    lasx_xvfmul_d(lasx::load_f64x4(m[r * 3 + 1].as_ptr().add(i)), vy),
                );
                acc = lasx_xvfadd_d(
                    acc,
                    lasx_xvfmul_d(lasx::load_f64x4(m[r * 3 + 2].as_ptr().add(i)), vz),
                );
                *out = acc;
            }
            lasx::store_f64x4(ox.as_mut_ptr().add(i), rows[0]);
            lasx::store_f64x4(oy.as_mut_ptr().add(i), rows[1]);
            lasx::store_f64x4(oz.as_mut_ptr().add(i), rows[2]);
        }
        i += 4;
    }
    tail(m, x, y, z, ox, oy, oz, i);
}

/// LSX 128 位实现（2 样本/向量）。
#[inline]
#[allow(clippy::too_many_arguments)]
fn mat3_mul_vec3_batch_lsx(
    m: [&[f64]; 9],
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
            let mut rows: [m128d; 3] = [std::mem::transmute(lsx_vldi(0)); 3];
            for (r, out) in rows.iter_mut().enumerate() {
                let mut acc = lsx_vfmul_d(lsx::load_f64x2(m[r * 3].as_ptr().add(i)), vx);
                acc = lsx_vfadd_d(
                    acc,
                    lsx_vfmul_d(lsx::load_f64x2(m[r * 3 + 1].as_ptr().add(i)), vy),
                );
                acc = lsx_vfadd_d(
                    acc,
                    lsx_vfmul_d(lsx::load_f64x2(m[r * 3 + 2].as_ptr().add(i)), vz),
                );
                *out = acc;
            }
            lsx::store_f64x2(ox.as_mut_ptr().add(i), rows[0]);
            lsx::store_f64x2(oy.as_mut_ptr().add(i), rows[1]);
            lsx::store_f64x2(oz.as_mut_ptr().add(i), rows[2]);
        }
        i += 2;
    }
    tail(m, x, y, z, ox, oy, oz, i);
}

/// 标量尾（与向量路径同结合次序）。
#[inline]
#[allow(clippy::too_many_arguments)]
fn tail(
    m: [&[f64]; 9],
    x: &[f64],
    y: &[f64],
    z: &[f64],
    ox: &mut [f64],
    oy: &mut [f64],
    oz: &mut [f64],
    from: usize,
) {
    for j in from..x.len() {
        for (r, out) in [&mut *ox, &mut *oy, &mut *oz].into_iter().enumerate() {
            out[j] = (m[r * 3][j] * x[j] + m[r * 3 + 1][j] * y[j]) + m[r * 3 + 2][j] * z[j];
        }
    }
}

/// 数值回归测试。
#[cfg(test)]
mod tests {
    use crate::ffi::attitude::lasx_mat3_mul_vec3_batch;
    use crate::ops::testutil::Lcg;

    #[test]
    fn test_mat3_mul_vec3_batch_bit_exact() {
        let mut rng = Lcg(0x3a3a_1234);
        for n in [0usize, 1, 2, 3, 5, 8, 9, 33] {
            let m: Vec<Vec<f64>> = (0..9)
                .map(|_| (0..n).map(|_| rng.f64() * 3.0).collect())
                .collect();
            let x: Vec<f64> = (0..n).map(|_| rng.f64()).collect();
            let y: Vec<f64> = (0..n).map(|_| rng.f64()).collect();
            let z: Vec<f64> = (0..n).map(|_| rng.f64()).collect();
            let (mut ox, mut oy, mut oz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            lasx_mat3_mul_vec3_batch(
                m[0].as_ptr(),
                m[1].as_ptr(),
                m[2].as_ptr(),
                m[3].as_ptr(),
                m[4].as_ptr(),
                m[5].as_ptr(),
                m[6].as_ptr(),
                m[7].as_ptr(),
                m[8].as_ptr(),
                x.as_ptr(),
                y.as_ptr(),
                z.as_ptr(),
                ox.as_mut_ptr(),
                oy.as_mut_ptr(),
                oz.as_mut_ptr(),
                n as i32,
            );
            for i in 0..n {
                for (r, got) in [&ox, &oy, &oz].into_iter().enumerate() {
                    let want =
                        (m[r * 3][i] * x[i] + m[r * 3 + 1][i] * y[i]) + m[r * 3 + 2][i] * z[i];
                    assert_eq!(got[i].to_bits(), want.to_bits(), "n={n} i={i} r={r}");
                }
            }
        }
    }

    /// 单位阵是恒等变换。
    #[test]
    fn test_mat3_identity() {
        let n = 7;
        let mut rng = Lcg(0x99);
        let x: Vec<f64> = (0..n).map(|_| rng.f64()).collect();
        let y: Vec<f64> = (0..n).map(|_| rng.f64()).collect();
        let z: Vec<f64> = (0..n).map(|_| rng.f64()).collect();
        let eye: [Vec<f64>; 9] = [
            vec![1.0; n],
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n],
            vec![1.0; n],
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n],
            vec![1.0; n],
        ];
        let (mut ox, mut oy, mut oz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        lasx_mat3_mul_vec3_batch(
            eye[0].as_ptr(),
            eye[1].as_ptr(),
            eye[2].as_ptr(),
            eye[3].as_ptr(),
            eye[4].as_ptr(),
            eye[5].as_ptr(),
            eye[6].as_ptr(),
            eye[7].as_ptr(),
            eye[8].as_ptr(),
            x.as_ptr(),
            y.as_ptr(),
            z.as_ptr(),
            ox.as_mut_ptr(),
            oy.as_mut_ptr(),
            oz.as_mut_ptr(),
            n as i32,
        );
        assert_eq!(ox, x);
        assert_eq!(oy, y);
        assert_eq!(oz, z);
    }
}
