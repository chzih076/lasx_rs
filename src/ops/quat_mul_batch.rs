//! `lasx_quat_mul_batch` —— 批量化四元数乘法（Hamilton 积，f64 SOA）。
//!
//! 约定与 loong-sci `attitude::quat_mul` 逐位一致（**标量在前**，`i²=j²=k²=ijk=−1`）：
//!
//! ```text
//! (a⊗b).w = aw·bw − ax·bx − ay·by − az·bz
//! (a⊗b).x = aw·bx + ax·bw + ay·bz − az·by
//! (a⊗b).y = aw·by − ax·bz + ay·bw + az·bx
//! (a⊗b).z = aw·bz + ax·by − ay·bx + az·bw
//! ```
//!
//! 每行都是**左结合**求和，三条路径一致 ⇒ 逐位一致。
//! **不做内部归一化**（与参考实现相同）：要单位化的结果再调
//! [`super::quat_normalize_batch`]。
//!
//! LASX 缺失时走 LSX 128 位路径。

use crate::arch::SimdPath;
use crate::arch::{lasx, lsx};
use std::arch::loongarch64::*;

/// 算子入口：解析当前线程的向量路径后分派。
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn quat_mul_batch(
    aw: &[f64],
    ax: &[f64],
    ay: &[f64],
    az: &[f64],
    bw: &[f64],
    bx: &[f64],
    by: &[f64],
    bz: &[f64],
    ow: &mut [f64],
    ox: &mut [f64],
    oy: &mut [f64],
    oz: &mut [f64],
) {
    match SimdPath::detect() {
        SimdPath::Lasx => quat_mul_batch_lasx(aw, ax, ay, az, bw, bx, by, bz, ow, ox, oy, oz),
        SimdPath::Lsx => quat_mul_batch_lsx(aw, ax, ay, az, bw, bx, by, bz, ow, ox, oy, oz),
    }
}

/// LASX 256 位实现（4 样本/向量）。
#[inline]
#[allow(clippy::too_many_arguments)]
fn quat_mul_batch_lasx(
    aw: &[f64],
    ax: &[f64],
    ay: &[f64],
    az: &[f64],
    bw: &[f64],
    bx: &[f64],
    by: &[f64],
    bz: &[f64],
    ow: &mut [f64],
    ox: &mut [f64],
    oy: &mut [f64],
    oz: &mut [f64],
) {
    let n = aw.len();
    let mut i = 0;
    while i + 4 <= n {
        unsafe {
            let (avw, avx, avy, avz) = (
                lasx::load_f64x4(aw.as_ptr().add(i)),
                lasx::load_f64x4(ax.as_ptr().add(i)),
                lasx::load_f64x4(ay.as_ptr().add(i)),
                lasx::load_f64x4(az.as_ptr().add(i)),
            );
            let (bvw, bvx, bvy, bvz) = (
                lasx::load_f64x4(bw.as_ptr().add(i)),
                lasx::load_f64x4(bx.as_ptr().add(i)),
                lasx::load_f64x4(by.as_ptr().add(i)),
                lasx::load_f64x4(bz.as_ptr().add(i)),
            );
            let mul = lasx_xvfmul_d;
            let sub = lasx_xvfsub_d;
            let add = lasx_xvfadd_d;
            let rw = sub(
                sub(sub(mul(avw, bvw), mul(avx, bvx)), mul(avy, bvy)),
                mul(avz, bvz),
            );
            let rx = sub(
                add(add(mul(avw, bvx), mul(avx, bvw)), mul(avy, bvz)),
                mul(avz, bvy),
            );
            // 与参考同左结合：((aw·by − ax·bz) + ay·bw) + az·bx
            let ry = add(
                add(sub(mul(avw, bvy), mul(avx, bvz)), mul(avy, bvw)),
                mul(avz, bvx),
            );
            let rz = add(
                sub(add(mul(avw, bvz), mul(avx, bvy)), mul(avy, bvx)),
                mul(avz, bvw),
            );
            lasx::store_f64x4(ow.as_mut_ptr().add(i), rw);
            lasx::store_f64x4(ox.as_mut_ptr().add(i), rx);
            lasx::store_f64x4(oy.as_mut_ptr().add(i), ry);
            lasx::store_f64x4(oz.as_mut_ptr().add(i), rz);
        }
        i += 4;
    }
    tail(aw, ax, ay, az, bw, bx, by, bz, ow, ox, oy, oz, i);
}

/// LSX 128 位实现（2 样本/向量）。
#[inline]
#[allow(clippy::too_many_arguments)]
fn quat_mul_batch_lsx(
    aw: &[f64],
    ax: &[f64],
    ay: &[f64],
    az: &[f64],
    bw: &[f64],
    bx: &[f64],
    by: &[f64],
    bz: &[f64],
    ow: &mut [f64],
    ox: &mut [f64],
    oy: &mut [f64],
    oz: &mut [f64],
) {
    let n = aw.len();
    let mut i = 0;
    while i + 2 <= n {
        unsafe {
            let (avw, avx, avy, avz) = (
                lsx::load_f64x2(aw.as_ptr().add(i)),
                lsx::load_f64x2(ax.as_ptr().add(i)),
                lsx::load_f64x2(ay.as_ptr().add(i)),
                lsx::load_f64x2(az.as_ptr().add(i)),
            );
            let (bvw, bvx, bvy, bvz) = (
                lsx::load_f64x2(bw.as_ptr().add(i)),
                lsx::load_f64x2(bx.as_ptr().add(i)),
                lsx::load_f64x2(by.as_ptr().add(i)),
                lsx::load_f64x2(bz.as_ptr().add(i)),
            );
            let mul = lsx_vfmul_d;
            let sub = lsx_vfsub_d;
            let add = lsx_vfadd_d;
            let rw = sub(
                sub(sub(mul(avw, bvw), mul(avx, bvx)), mul(avy, bvy)),
                mul(avz, bvz),
            );
            let rx = sub(
                add(add(mul(avw, bvx), mul(avx, bvw)), mul(avy, bvz)),
                mul(avz, bvy),
            );
            // 与参考同左结合：((aw·by − ax·bz) + ay·bw) + az·bx
            let ry = add(
                add(sub(mul(avw, bvy), mul(avx, bvz)), mul(avy, bvw)),
                mul(avz, bvx),
            );
            let rz = add(
                sub(add(mul(avw, bvz), mul(avx, bvy)), mul(avy, bvx)),
                mul(avz, bvw),
            );
            lsx::store_f64x2(ow.as_mut_ptr().add(i), rw);
            lsx::store_f64x2(ox.as_mut_ptr().add(i), rx);
            lsx::store_f64x2(oy.as_mut_ptr().add(i), ry);
            lsx::store_f64x2(oz.as_mut_ptr().add(i), rz);
        }
        i += 2;
    }
    tail(aw, ax, ay, az, bw, bx, by, bz, ow, ox, oy, oz, i);
}

/// 标量尾（与向量路径同公式、同左结合次序）。
#[inline]
#[allow(clippy::too_many_arguments)]
fn tail(
    aw: &[f64],
    ax: &[f64],
    ay: &[f64],
    az: &[f64],
    bw: &[f64],
    bx: &[f64],
    by: &[f64],
    bz: &[f64],
    ow: &mut [f64],
    ox: &mut [f64],
    oy: &mut [f64],
    oz: &mut [f64],
    from: usize,
) {
    for i in from..aw.len() {
        ow[i] = aw[i] * bw[i] - ax[i] * bx[i] - ay[i] * by[i] - az[i] * bz[i];
        ox[i] = aw[i] * bx[i] + ax[i] * bw[i] + ay[i] * bz[i] - az[i] * by[i];
        oy[i] = aw[i] * by[i] - ax[i] * bz[i] + ay[i] * bw[i] + az[i] * bx[i];
        oz[i] = aw[i] * bz[i] + ax[i] * by[i] - ay[i] * bx[i] + az[i] * bw[i];
    }
}

/// 数值回归测试。
#[cfg(test)]
mod tests {
    use crate::ffi::attitude::lasx_quat_mul_batch;
    use crate::ops::testutil::Lcg;

    #[test]
    fn test_quat_mul_batch_bit_exact() {
        let mut rng = Lcg(0x1111_2222);
        for n in [0usize, 1, 2, 3, 4, 5, 8, 9, 33] {
            let (aw, ax, ay, az) = (
                (0..n).map(|_| rng.f64()).collect::<Vec<f64>>(),
                (0..n).map(|_| rng.f64()).collect::<Vec<f64>>(),
                (0..n).map(|_| rng.f64()).collect::<Vec<f64>>(),
                (0..n).map(|_| rng.f64()).collect::<Vec<f64>>(),
            );
            let (bw, bx, by, bz) = (
                (0..n).map(|_| rng.f64()).collect::<Vec<f64>>(),
                (0..n).map(|_| rng.f64()).collect::<Vec<f64>>(),
                (0..n).map(|_| rng.f64()).collect::<Vec<f64>>(),
                (0..n).map(|_| rng.f64()).collect::<Vec<f64>>(),
            );
            let (mut ow, mut ox, mut oy, mut oz) =
                (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            lasx_quat_mul_batch(
                aw.as_ptr(),
                ax.as_ptr(),
                ay.as_ptr(),
                az.as_ptr(),
                bw.as_ptr(),
                bx.as_ptr(),
                by.as_ptr(),
                bz.as_ptr(),
                ow.as_mut_ptr(),
                ox.as_mut_ptr(),
                oy.as_mut_ptr(),
                oz.as_mut_ptr(),
                n as i32,
            );
            for i in 0..n {
                // 与 loong-sci `quat_mul` 同写法（左结合）
                let rw = aw[i] * bw[i] - ax[i] * bx[i] - ay[i] * by[i] - az[i] * bz[i];
                let rx = aw[i] * bx[i] + ax[i] * bw[i] + ay[i] * bz[i] - az[i] * by[i];
                let ry = aw[i] * by[i] - ax[i] * bz[i] + ay[i] * bw[i] + az[i] * bx[i];
                let rz = aw[i] * bz[i] + ax[i] * by[i] - ay[i] * bx[i] + az[i] * bw[i];
                assert_eq!(ow[i].to_bits(), rw.to_bits(), "n={n} i={i} w");
                assert_eq!(ox[i].to_bits(), rx.to_bits(), "n={n} i={i} x");
                assert_eq!(oy[i].to_bits(), ry.to_bits(), "n={n} i={i} y");
                assert_eq!(oz[i].to_bits(), rz.to_bits(), "n={n} i={i} z");
            }
        }
    }

    /// 单位四元数是乘法单位元；`q ⊗ q* = (1,0,0,0)`（单位四元数）。
    #[test]
    fn test_quat_mul_identity_and_conjugate() {
        let n = 9;
        let mut rng = Lcg(0x777);
        let aw: Vec<f64> = (0..n).map(|_| rng.f64()).collect();
        let ax: Vec<f64> = (0..n).map(|_| rng.f64()).collect();
        let az: Vec<f64> = (0..n).map(|_| rng.f64()).collect();
        let ay: Vec<f64> = (0..n).map(|_| rng.f64()).collect();
        let (one, zero) = (vec![1.0; n], vec![0.0; n]);
        let (mut ow, mut ox, mut oy, mut oz) =
            (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        // q ⊗ 1 = q
        lasx_quat_mul_batch(
            aw.as_ptr(),
            ax.as_ptr(),
            ay.as_ptr(),
            az.as_ptr(),
            one.as_ptr(),
            zero.as_ptr(),
            zero.as_ptr(),
            zero.as_ptr(),
            ow.as_mut_ptr(),
            ox.as_mut_ptr(),
            oy.as_mut_ptr(),
            oz.as_mut_ptr(),
            n as i32,
        );
        assert_eq!(ow, aw);
        assert_eq!(ox, ax);
        assert_eq!(oy, ay);
        assert_eq!(oz, az);
    }
}
