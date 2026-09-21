//! `lasx_j2_accel_batch` —— 批量中心引力 + J2 摄动加速度（f64 SOA）。
//!
//! LASX 缺失时走 LSX 128 位路径。
use crate::arch::SimdPath;
use crate::arch::{lasx, lsx};
use std::arch::loongarch64::*;

/// 算子入口：解析当前线程的向量路径后分派。
#[inline]
pub(crate) fn j2_accel_batch(
    rx: &[f64],
    ry: &[f64],
    rz: &[f64],
    mu: f64,
    j2: f64,
    re: f64,
    ax: &mut [f64],
    ay: &mut [f64],
    az: &mut [f64],
) {
    match SimdPath::detect() {
        SimdPath::Lasx => j2_accel_batch_lasx(rx, ry, rz, mu, j2, re, ax, ay, az),
        SimdPath::Lsx => j2_accel_batch_lsx(rx, ry, rz, mu, j2, re, ax, ay, az),
    }
}

/// LASX 256 位实现。
#[inline]
fn j2_accel_batch_lasx(
    rx: &[f64],
    ry: &[f64],
    rz: &[f64],
    mu: f64,
    j2: f64,
    re: f64,
    ax: &mut [f64],
    ay: &mut [f64],
    az: &mut [f64],
) {
    let n = rx.len();
    // k 取正值（−k 即此值）：1.5·J2·μ·Re²
    let j2k = 1.5 * j2 * mu * re * re;
    let vmu = lasx::splat_f64(-mu); // 中心项 = −μ·r/|r|³
    let vj2k = lasx::splat_f64(j2k);
    let v5 = lasx::splat_f64(5.0);
    let v1 = lasx::splat_f64(1.0);
    let v3 = lasx::splat_f64(3.0);
    let mut i = 0;
    while i + 4 <= n {
        let vx = unsafe { lasx::load_f64x4(rx.as_ptr().add(i)) };
        let vy = unsafe { lasx::load_f64x4(ry.as_ptr().add(i)) };
        let vz = unsafe { lasx::load_f64x4(rz.as_ptr().add(i)) };
        // |r|²、|r|、|r|³、|r|⁵ = |r|³·|r|²
        let vrm2 = unsafe {
            lasx_xvfadd_d(
                lasx_xvfmul_d(vx, vx),
                lasx_xvfadd_d(lasx_xvfmul_d(vy, vy), lasx_xvfmul_d(vz, vz)),
            )
        };
        let vrm = unsafe { lasx_xvfsqrt_d(vrm2) };
        let vrm3 = unsafe { lasx_xvfmul_d(vrm, vrm2) };
        let vrm5 = unsafe { lasx_xvfmul_d(vrm3, vrm2) };
        // 中心项 −μ·r/|r|³
        let vcen = unsafe { lasx_xvfdiv_d(vmu, vrm3) };
        // J2 项系数（正）：1.5·J2·μ·Re²/rm⁵
        let vk = unsafe { lasx_xvfdiv_d(vj2k, vrm5) };
        // zr2 = (z/|r|)²
        let vzr2 = unsafe { lasx_xvfdiv_d(lasx_xvfmul_d(vz, vz), vrm2) };
        let m1 = unsafe { lasx_xvfsub_d(lasx_xvfmul_d(v5, vzr2), v1) };
        let m3 = unsafe { lasx_xvfsub_d(lasx_xvfmul_d(v5, vzr2), v3) };
        // a_x = −μ·x/rm³ + k·x·(5·zr2−1)（k 正 = −k_j2）
        unsafe {
            lasx::store_f64x4(
                ax.as_mut_ptr().add(i),
                lasx_xvfmadd_d(lasx_xvfmul_d(vk, vx), m1, lasx_xvfmul_d(vcen, vx)),
            );
            lasx::store_f64x4(
                ay.as_mut_ptr().add(i),
                lasx_xvfmadd_d(lasx_xvfmul_d(vk, vy), m1, lasx_xvfmul_d(vcen, vy)),
            );
            lasx::store_f64x4(
                az.as_mut_ptr().add(i),
                lasx_xvfmadd_d(lasx_xvfmul_d(vk, vz), m3, lasx_xvfmul_d(vcen, vz)),
            );
        }
        i += 4;
    }
    for j in i..n {
        // 与向量路径逐位同式（结合律/除式/末项 FMA 完全一致），保证任一分块逐位一致
        let rm2 = rx[j] * rx[j] + (ry[j] * ry[j] + rz[j] * rz[j]);
        let rm = rm2.sqrt();
        let rm3 = rm * rm2;
        let rm5 = rm3 * rm2;
        let zr2 = (rz[j] * rz[j]) / rm2;
        let k = j2k / rm5;
        let vcen = -mu / rm3;
        ax[j] = f64::mul_add(k * rx[j], 5.0 * zr2 - 1.0, vcen * rx[j]);
        ay[j] = f64::mul_add(k * ry[j], 5.0 * zr2 - 1.0, vcen * ry[j]);
        az[j] = f64::mul_add(k * rz[j], 5.0 * zr2 - 3.0, vcen * rz[j]);
    }
}

/// LSX 128 位实现（2 样本/向量）。
#[inline]
fn j2_accel_batch_lsx(
    rx: &[f64],
    ry: &[f64],
    rz: &[f64],
    mu: f64,
    j2: f64,
    re: f64,
    ax: &mut [f64],
    ay: &mut [f64],
    az: &mut [f64],
) {
    let n = rx.len();
    // k 取正值（−k 即此值）：1.5·J2·μ·Re²
    let j2k = 1.5 * j2 * mu * re * re;
    let vs = lsx::splat_f64(j2k);
    let vmu = lsx::splat_f64(-mu); // 中心项 = −μ·r/|r|³
    let v5 = lsx::splat_f64(5.0);
    let v1 = lsx::splat_f64(1.0);
    let v3 = lsx::splat_f64(3.0);
    let mut i = 0;
    while i + 2 <= n {
        let vx = unsafe { lsx::load_f64x2(rx.as_ptr().add(i)) };
        let vy = unsafe { lsx::load_f64x2(ry.as_ptr().add(i)) };
        let vz = unsafe { lsx::load_f64x2(rz.as_ptr().add(i)) };
        let vrm2 = unsafe {
            lsx_vfadd_d(
                lsx_vfmul_d(vx, vx),
                lsx_vfadd_d(lsx_vfmul_d(vy, vy), lsx_vfmul_d(vz, vz)),
            )
        };
        let vrm = unsafe { lsx_vfsqrt_d(vrm2) };
        let vrm3 = unsafe { lsx_vfmul_d(vrm, vrm2) };
        let vrm5 = unsafe { lsx_vfmul_d(vrm3, vrm2) };
        let vcen = unsafe { lsx_vfdiv_d(vmu, vrm3) };
        let vk = unsafe { lsx_vfdiv_d(vs, vrm5) };
        let vzr2 = unsafe { lsx_vfdiv_d(lsx_vfmul_d(vz, vz), vrm2) };
        let m1 = unsafe { lsx_vfsub_d(lsx_vfmul_d(v5, vzr2), v1) };
        let m3 = unsafe { lsx_vfsub_d(lsx_vfmul_d(v5, vzr2), v3) };
        unsafe {
            lsx::store_f64x2(
                ax.as_mut_ptr().add(i),
                lsx_vfmadd_d(lsx_vfmul_d(vk, vx), m1, lsx_vfmul_d(vcen, vx)),
            );
            lsx::store_f64x2(
                ay.as_mut_ptr().add(i),
                lsx_vfmadd_d(lsx_vfmul_d(vk, vy), m1, lsx_vfmul_d(vcen, vy)),
            );
            lsx::store_f64x2(
                az.as_mut_ptr().add(i),
                lsx_vfmadd_d(lsx_vfmul_d(vk, vz), m3, lsx_vfmul_d(vcen, vz)),
            );
        }
        i += 2;
    }
    for j in i..n {
        // 与 LSX 向量路径逐位同式（同结合/除式/末项 FMA）
        let rm2 = rx[j] * rx[j] + (ry[j] * ry[j] + rz[j] * rz[j]);
        let rm = rm2.sqrt();
        let rm3 = rm * rm2;
        let rm5 = rm3 * rm2;
        let zr2 = (rz[j] * rz[j]) / rm2;
        let k = j2k / rm5;
        let vcen = -mu / rm3;
        ax[j] = f64::mul_add(k * rx[j], 5.0 * zr2 - 1.0, vcen * rx[j]);
        ay[j] = f64::mul_add(k * ry[j], 5.0 * zr2 - 1.0, vcen * ry[j]);
        az[j] = f64::mul_add(k * rz[j], 5.0 * zr2 - 3.0, vcen * rz[j]);
    }
}

/// 数值回归测试：对照独立参考实现。
#[cfg(test)]
mod tests {
    use crate::ffi::physics::lasx_j2_accel_batch;
    use crate::ops::testutil::{rel_err, states};

    fn scalar_j2(
        rx: &[f64],
        ry: &[f64],
        rz: &[f64],
        mu: f64,
        j2: f64,
        re: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = rx.len();
        let (mut ax, mut ay, mut az) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let j2k = 1.5 * j2 * mu * re * re;
        for i in 0..n {
            let rm = (rx[i] * rx[i] + ry[i] * ry[i] + rz[i] * rz[i]).sqrt();
            let rm3 = rm * rm * rm;
            let rm5 = rm3 * rm * rm;
            let zr2 = (rz[i] / rm) * (rz[i] / rm);
            let k = j2k / rm5;
            ax[i] = -mu * rx[i] / rm3 + k * rx[i] * (5.0 * zr2 - 1.0);
            ay[i] = -mu * ry[i] / rm3 + k * ry[i] * (5.0 * zr2 - 1.0);
            az[i] = -mu * rz[i] / rm3 + k * rz[i] * (5.0 * zr2 - 3.0);
        }
        (ax, ay, az)
    }
    #[test]
    fn test_j2_accel_batch_matches_scalar() {
        for n in [0usize, 1, 2, 4, 9, 20] {
            let (x, y, z) = states(n);
            let (mut ax, mut ay, mut az) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            lasx_j2_accel_batch(
                x.as_ptr(),
                y.as_ptr(),
                z.as_ptr(),
                3.986004418e14,
                1.08262668e-3,
                6.378137e6,
                ax.as_mut_ptr(),
                ay.as_mut_ptr(),
                az.as_mut_ptr(),
                n as i32,
            );
            let (wx, wy, wz) = scalar_j2(&x, &y, &z, 3.986004418e14, 1.08262668e-3, 6.378137e6);
            for i in 0..n {
                assert!(
                    rel_err(ax[i], wx[i]) < 1e-9
                        && rel_err(ay[i], wy[i]) < 1e-9
                        && rel_err(az[i], wz[i]) < 1e-9,
                    "n={n} i={i}: ({},{},{}) vs ({},{},{})",
                    ax[i],
                    ay[i],
                    az[i],
                    wx[i],
                    wy[i],
                    wz[i]
                );
            }
        }
    }
}
