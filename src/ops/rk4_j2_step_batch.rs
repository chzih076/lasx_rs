//! `lasx_rk4_j2_step_batch` —— 批量 RK4 J2 轨道步（f64 SOA，4 样本/向量，原地更新）。
//!
//! LASX 缺失时整体退化为纯标量。
use crate::arch::lasx;
use crate::arch::SimdPath;
use std::arch::loongarch64::*;

/// 算子入口：解析当前线程的向量路径后分派。
#[inline]
pub(crate) fn rk4_j2_step_batch(
    rx: &mut [f64],
    ry: &mut [f64],
    rz: &mut [f64],
    vx: &mut [f64],
    vy: &mut [f64],
    vz: &mut [f64],
    mu: f64,
    j2: f64,
    re: f64,
    dt: f64,
) {
    match SimdPath::detect() {
        SimdPath::Lasx => rk4_j2_step_batch_lasx(rx, ry, rz, vx, vy, vz, mu, j2, re, dt),
        SimdPath::Lsx => rk4_j2_step_batch_scalar(rx, ry, rz, vx, vy, vz, mu, j2, re, dt),
    }
}

/// LASX 256 位实现。
#[inline]
fn rk4_j2_step_batch_lasx(
    rx: &mut [f64],
    ry: &mut [f64],
    rz: &mut [f64],
    vx: &mut [f64],
    vy: &mut [f64],
    vz: &mut [f64],
    mu: f64,
    j2: f64,
    re: f64,
    dt: f64,
) {
    let n = rx.len();
    let vmu = lasx::splat_f64(-mu);
    let vj2k = lasx::splat_f64(1.5 * j2 * mu * re * re);
    let v5 = lasx::splat_f64(5.0);
    let v1 = lasx::splat_f64(1.0);
    let v3s = lasx::splat_f64(3.0);
    let vh = lasx::splat_f64(dt);
    let vh6 = lasx::splat_f64(dt / 6.0);
    let vh2 = lasx::splat_f64(0.5 * dt);
    let v2s = lasx::splat_f64(2.0);
    let mut i = 0;
    while i + 4 <= n {
        let lrx = unsafe { lasx::load_f64x4(rx.as_ptr().add(i)) };
        let lry = unsafe { lasx::load_f64x4(ry.as_ptr().add(i)) };
        let lrz = unsafe { lasx::load_f64x4(rz.as_ptr().add(i)) };
        let lvx = unsafe { lasx::load_f64x4(vx.as_ptr().add(i)) };
        let lvy = unsafe { lasx::load_f64x4(vy.as_ptr().add(i)) };
        let lvz = unsafe { lasx::load_f64x4(vz.as_ptr().add(i)) };
        let (k1x, k1y, k1z) = unsafe { j2_accel_vec(lrx, lry, lrz, vmu, vj2k, v5, v1, v3s) };
        let r2 = (
            unsafe { lasx_xvfmadd_d(vh2, lvx, lrx) },
            unsafe { lasx_xvfmadd_d(vh2, lvy, lry) },
            unsafe { lasx_xvfmadd_d(vh2, lvz, lrz) },
        );
        let v2 = (
            unsafe { lasx_xvfmadd_d(vh2, k1x, lvx) },
            unsafe { lasx_xvfmadd_d(vh2, k1y, lvy) },
            unsafe { lasx_xvfmadd_d(vh2, k1z, lvz) },
        );
        let (k2x, k2y, k2z) = unsafe { j2_accel_vec(r2.0, r2.1, r2.2, vmu, vj2k, v5, v1, v3s) };
        let r3 = (
            unsafe { lasx_xvfmadd_d(vh2, v2.0, lrx) },
            unsafe { lasx_xvfmadd_d(vh2, v2.1, lry) },
            unsafe { lasx_xvfmadd_d(vh2, v2.2, lrz) },
        );
        let v3 = (
            unsafe { lasx_xvfmadd_d(vh2, k2x, lvx) },
            unsafe { lasx_xvfmadd_d(vh2, k2y, lvy) },
            unsafe { lasx_xvfmadd_d(vh2, k2z, lvz) },
        );
        let (k3x, k3y, k3z) = unsafe { j2_accel_vec(r3.0, r3.1, r3.2, vmu, vj2k, v5, v1, v3s) };
        let r4 = (
            unsafe { lasx_xvfmadd_d(vh, v3.0, lrx) },
            unsafe { lasx_xvfmadd_d(vh, v3.1, lry) },
            unsafe { lasx_xvfmadd_d(vh, v3.2, lrz) },
        );
        let v4 = (
            unsafe { lasx_xvfmadd_d(vh, k3x, lvx) },
            unsafe { lasx_xvfmadd_d(vh, k3y, lvy) },
            unsafe { lasx_xvfmadd_d(vh, k3z, lvz) },
        );
        let (k4x, k4y, k4z) = unsafe { j2_accel_vec(r4.0, r4.1, r4.2, vmu, vj2k, v5, v1, v3s) };
        // lsum = v + 2·v2 + 2·v3 + v4；ksum 同理
        let l = (
            unsafe {
                lasx_xvfadd_d(
                    lasx_xvfmadd_d(v2s, v2.0, lvx),
                    lasx_xvfmadd_d(v2s, v3.0, v4.0),
                )
            },
            unsafe {
                lasx_xvfadd_d(
                    lasx_xvfmadd_d(v2s, v2.1, lvy),
                    lasx_xvfmadd_d(v2s, v3.1, v4.1),
                )
            },
            unsafe {
                lasx_xvfadd_d(
                    lasx_xvfmadd_d(v2s, v2.2, lvz),
                    lasx_xvfmadd_d(v2s, v3.2, v4.2),
                )
            },
        );
        let k = (
            unsafe { lasx_xvfadd_d(lasx_xvfmadd_d(v2s, k2x, k1x), lasx_xvfmadd_d(v2s, k3x, k4x)) },
            unsafe { lasx_xvfadd_d(lasx_xvfmadd_d(v2s, k2y, k1y), lasx_xvfmadd_d(v2s, k3y, k4y)) },
            unsafe { lasx_xvfadd_d(lasx_xvfmadd_d(v2s, k2z, k1z), lasx_xvfmadd_d(v2s, k3z, k4z)) },
        );
        let nr = (
            unsafe { lasx_xvfmadd_d(vh6, l.0, lrx) },
            unsafe { lasx_xvfmadd_d(vh6, l.1, lry) },
            unsafe { lasx_xvfmadd_d(vh6, l.2, lrz) },
        );
        let nv = (
            unsafe { lasx_xvfmadd_d(vh6, k.0, lvx) },
            unsafe { lasx_xvfmadd_d(vh6, k.1, lvy) },
            unsafe { lasx_xvfmadd_d(vh6, k.2, lvz) },
        );
        unsafe {
            lasx::store_f64x4(rx.as_mut_ptr().add(i), nr.0);
            lasx::store_f64x4(ry.as_mut_ptr().add(i), nr.1);
            lasx::store_f64x4(rz.as_mut_ptr().add(i), nr.2);
            lasx::store_f64x4(vx.as_mut_ptr().add(i), nv.0);
            lasx::store_f64x4(vy.as_mut_ptr().add(i), nv.1);
            lasx::store_f64x4(vz.as_mut_ptr().add(i), nv.2);
        }
        i += 4;
    }
    for j in i..n {
        rk4_j2_step_scalar(
            &mut rx[j], &mut ry[j], &mut rz[j], &mut vx[j], &mut vy[j], &mut vz[j], mu, j2, re, dt,
        );
    }
}

/// 纯标量降级（逐样本 `rk4_j2_step_scalar`，本内核**没有** LSX 实现）。
#[inline]
fn rk4_j2_step_batch_scalar(
    rx: &mut [f64],
    ry: &mut [f64],
    rz: &mut [f64],
    vx: &mut [f64],
    vy: &mut [f64],
    vz: &mut [f64],
    mu: f64,
    j2: f64,
    re: f64,
    dt: f64,
) {
    let n = rx.len();
    for j in 0..n {
        rk4_j2_step_scalar(
            &mut rx[j], &mut ry[j], &mut rz[j], &mut vx[j], &mut vy[j], &mut vz[j], mu, j2, re, dt,
        );
    }
}

unsafe fn j2_accel_vec(
    vx: m256d,
    vy: m256d,
    vz: m256d,
    vmu: m256d,
    vj2k: m256d,
    v5: m256d,
    v1: m256d,
    v3: m256d,
) -> (m256d, m256d, m256d) {
    let vrm2 = lasx_xvfadd_d(
        lasx_xvfmul_d(vx, vx),
        lasx_xvfadd_d(lasx_xvfmul_d(vy, vy), lasx_xvfmul_d(vz, vz)),
    );
    let vrm = lasx_xvfsqrt_d(vrm2);
    // 1 次除法 + 乘法推导（与 j2_accel_batch 的 LASX/标量路径同式，见 docs/dev.md §13）
    let vone = lasx::splat_f64(1.0);
    let vinv2 = lasx_xvfdiv_d(vone, vrm2);
    let vinvrm = lasx_xvfmul_d(vinv2, vrm);
    let vinv3 = lasx_xvfmul_d(vinv2, vinvrm);
    let vinv5 = lasx_xvfmul_d(vinv3, vinv2);
    let vcen = lasx_xvfmul_d(vmu, vinv3); // −μ/|r|³
    let vk = lasx_xvfmul_d(vj2k, vinv5); // +1.5·J2·μ·Re²/|r|⁵
    let vzr2 = lasx_xvfmul_d(lasx_xvfmul_d(vz, vz), vinv2);
    let m1 = lasx_xvfsub_d(lasx_xvfmul_d(v5, vzr2), v1); // 5·zr2−1
    let m3 = lasx_xvfsub_d(lasx_xvfmul_d(v5, vzr2), v3); // 5·zr2−3
    let ax = lasx_xvfmadd_d(lasx_xvfmul_d(vk, vx), m1, lasx_xvfmul_d(vcen, vx));
    let ay = lasx_xvfmadd_d(lasx_xvfmul_d(vk, vy), m1, lasx_xvfmul_d(vcen, vy));
    let az = lasx_xvfmadd_d(lasx_xvfmul_d(vk, vz), m3, lasx_xvfmul_d(vcen, vz));
    (ax, ay, az)
}

pub(crate) fn rk4_j2_step_scalar(
    rx: &mut f64,
    ry: &mut f64,
    rz: &mut f64,
    vx: &mut f64,
    vy: &mut f64,
    vz: &mut f64,
    mu: f64,
    j2: f64,
    re: f64,
    h: f64,
) {
    let j2k = 1.5 * j2 * mu * re * re;
    let accel = |x: f64, y: f64, z: f64| -> [f64; 3] {
        // 结合序与向量路径 `lasx_xvfadd_d(x·x, xvfadd_d(y·y, z·z))` 一致
        let rm2 = x * x + (y * y + z * z);
        let rm = rm2.sqrt();
        // 与向量路径同式：1 次除法 + 乘法推导
        let inv2 = 1.0 / rm2;
        let invrm = inv2 * rm;
        let inv3 = inv2 * invrm;
        let inv5 = inv3 * inv2;
        let zr2 = (z * z) * inv2;
        let k = j2k * inv5;
        let vcen = -mu * inv3;
        // a = k·x·(5·zr2−1) + vcen·x（FMA 融合末加，与 `j2_accel_vec` 同式）
        [
            f64::mul_add(k * x, 5.0 * zr2 - 1.0, vcen * x),
            f64::mul_add(k * y, 5.0 * zr2 - 1.0, vcen * y),
            f64::mul_add(k * z, 5.0 * zr2 - 3.0, vcen * z),
        ]
    };
    let (x0, y0, z0, vx0, vy0, vz0) = (*rx, *ry, *rz, *vx, *vy, *vz);
    let hh = 0.5 * h;
    let h6 = h / 6.0;
    let k1 = accel(x0, y0, z0);
    let r2 = (
        hh.mul_add(vx0, x0),
        hh.mul_add(vy0, y0),
        hh.mul_add(vz0, z0),
    );
    let v2 = (
        hh.mul_add(k1[0], vx0),
        hh.mul_add(k1[1], vy0),
        hh.mul_add(k1[2], vz0),
    );
    let k2 = accel(r2.0, r2.1, r2.2);
    let r3 = (
        hh.mul_add(v2.0, x0),
        hh.mul_add(v2.1, y0),
        hh.mul_add(v2.2, z0),
    );
    let v3 = (
        hh.mul_add(k2[0], vx0),
        hh.mul_add(k2[1], vy0),
        hh.mul_add(k2[2], vz0),
    );
    let k3 = accel(r3.0, r3.1, r3.2);
    let r4 = (
        h.mul_add(v3.0, x0),
        h.mul_add(v3.1, y0),
        h.mul_add(v3.2, z0),
    );
    let v4 = (
        h.mul_add(k3[0], vx0),
        h.mul_add(k3[1], vy0),
        h.mul_add(k3[2], vz0),
    );
    let k4 = accel(r4.0, r4.1, r4.2);
    let l = (
        2.0_f64.mul_add(v2.0, vx0) + 2.0_f64.mul_add(v3.0, v4.0),
        2.0_f64.mul_add(v2.1, vy0) + 2.0_f64.mul_add(v3.1, v4.1),
        2.0_f64.mul_add(v2.2, vz0) + 2.0_f64.mul_add(v3.2, v4.2),
    );
    let k = (
        2.0_f64.mul_add(k2[0], k1[0]) + 2.0_f64.mul_add(k3[0], k4[0]),
        2.0_f64.mul_add(k2[1], k1[1]) + 2.0_f64.mul_add(k3[1], k4[1]),
        2.0_f64.mul_add(k2[2], k1[2]) + 2.0_f64.mul_add(k3[2], k4[2]),
    );
    *rx = h6.mul_add(l.0, x0);
    *ry = h6.mul_add(l.1, y0);
    *rz = h6.mul_add(l.2, z0);
    *vx = h6.mul_add(k.0, vx0);
    *vy = h6.mul_add(k.1, vy0);
    *vz = h6.mul_add(k.2, vz0);
}

/// 数值回归测试：对照独立参考实现。
#[cfg(test)]
mod tests {
    use super::rk4_j2_step_scalar;
    use crate::ffi::physics::lasx_rk4_j2_step_batch;
    use crate::ops::testutil::{rel_err, states};

    #[test]
    fn test_rk4_j2_step_batch_matches_scalar() {
        // 批量单步 RK4 vs 逐星标量（同公式），多步累积后仍 <1e-9
        let mu = 3.986004418e14;
        let j2 = 1.08262668e-3;
        let re = 6.378137e6;
        for n in [1usize, 2, 3, 4, 5, 7, 8, 17] {
            let (x, y, z) = states(n);
            // 速度
            let vx: Vec<f64> = (0..n).map(|i| 100.0 * (i as f64) + 500.0).collect();
            let vy: Vec<f64> = (0..n).map(|i| 7700.0 * (1.0 + 0.01 * (i as f64))).collect();
            let vz: Vec<f64> = (0..n).map(|i| 20.0 * (i as f64)).collect();
            let (mut bx, mut by, mut bz) = (x.clone(), y.clone(), z.clone());
            let (mut bvx, mut bvy, mut bvz) = (vx.clone(), vy.clone(), vz.clone());
            // 参考：逐星标量 50 步
            let (mut sx, mut sy, mut sz) = (x.clone(), y.clone(), z.clone());
            let (mut svx, mut svy, mut svz) = (vx.clone(), vy.clone(), vz.clone());
            let dt = 10.0;
            for _ in 0..50 {
                lasx_rk4_j2_step_batch(
                    bx.as_mut_ptr(),
                    by.as_mut_ptr(),
                    bz.as_mut_ptr(),
                    bvx.as_mut_ptr(),
                    bvy.as_mut_ptr(),
                    bvz.as_mut_ptr(),
                    mu,
                    j2,
                    re,
                    dt,
                    n as i32,
                );
                for i in 0..n {
                    rk4_j2_step_scalar(
                        &mut sx[i],
                        &mut sy[i],
                        &mut sz[i],
                        &mut svx[i],
                        &mut svy[i],
                        &mut svz[i],
                        mu,
                        j2,
                        re,
                        dt,
                    );
                }
            }
            for i in 0..n {
                assert!(
                    rel_err(bx[i], sx[i]) < 1e-9 && rel_err(bvx[i], svx[i]) < 1e-9,
                    "n={n} i={i}: batch r {} vs scalar {}",
                    bx[i],
                    sx[i]
                );
                assert!(rel_err(by[i], sy[i]) < 1e-9 && rel_err(bvy[i], svy[i]) < 1e-9);
                assert!(rel_err(bz[i], sz[i]) < 1e-9 && rel_err(bvz[i], svz[i]) < 1e-9);
            }
        }
    }
    #[test]
    fn test_rk4_j2_step_scalar_fma_vs_plain_reference() {
        // 标量尾循环 FMA 版（f64::mul_add）vs 分离 mul+add 参考（原版公式）：
        // 单步 FMA 融合舍入差 ≤1 ulp，50 步累积后相对差仍 <1e-9（物理等价）。
        let mu = 3.986004418e14;
        let j2 = 1.08262668e-3;
        let re = 6.378137e6;
        let (x, y, z) = states(9);
        // 分离 mul+add 参考实现（与旧版 rk4_j2_step_scalar 同式）
        #[allow(clippy::too_many_arguments)]
        fn step_plain(
            rx: &mut f64,
            ry: &mut f64,
            rz: &mut f64,
            vx: &mut f64,
            vy: &mut f64,
            vz: &mut f64,
            mu: f64,
            j2: f64,
            re: f64,
            h: f64,
        ) {
            let j2k = 1.5 * j2 * mu * re * re;
            let accel = |x: f64, y: f64, z: f64| -> [f64; 3] {
                let rm = (x * x + y * y + z * z).sqrt();
                let rm3 = rm * rm * rm;
                let rm5 = rm3 * rm * rm;
                let zr2 = (z / rm) * (z / rm);
                let k = j2k / rm5;
                [
                    -mu * x / rm3 + k * x * (5.0 * zr2 - 1.0),
                    -mu * y / rm3 + k * y * (5.0 * zr2 - 1.0),
                    -mu * z / rm3 + k * z * (5.0 * zr2 - 3.0),
                ]
            };
            let (x0, y0, z0, vx0, vy0, vz0) = (*rx, *ry, *rz, *vx, *vy, *vz);
            let k1 = accel(x0, y0, z0);
            let r2 = (x0 + 0.5 * h * vx0, y0 + 0.5 * h * vy0, z0 + 0.5 * h * vz0);
            let v2 = (
                vx0 + 0.5 * h * k1[0],
                vy0 + 0.5 * h * k1[1],
                vz0 + 0.5 * h * k1[2],
            );
            let k2 = accel(r2.0, r2.1, r2.2);
            let r3 = (
                x0 + 0.5 * h * v2.0,
                y0 + 0.5 * h * v2.1,
                z0 + 0.5 * h * v2.2,
            );
            let v3 = (
                vx0 + 0.5 * h * k2[0],
                vy0 + 0.5 * h * k2[1],
                vz0 + 0.5 * h * k2[2],
            );
            let k3 = accel(r3.0, r3.1, r3.2);
            let r4 = (x0 + h * v3.0, y0 + h * v3.1, z0 + h * v3.2);
            let v4 = (vx0 + h * k3[0], vy0 + h * k3[1], vz0 + h * k3[2]);
            let k4 = accel(r4.0, r4.1, r4.2);
            let l = (
                vx0 + 2.0 * v2.0 + 2.0 * v3.0 + v4.0,
                vy0 + 2.0 * v2.1 + 2.0 * v3.1 + v4.1,
                vz0 + 2.0 * v2.2 + 2.0 * v3.2 + v4.2,
            );
            let k = (
                k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0],
                k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1],
                k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2],
            );
            *rx = x0 + (h / 6.0) * l.0;
            *ry = y0 + (h / 6.0) * l.1;
            *rz = z0 + (h / 6.0) * l.2;
            *vx = vx0 + (h / 6.0) * k.0;
            *vy = vy0 + (h / 6.0) * k.1;
            *vz = vz0 + (h / 6.0) * k.2;
        }
        let (mut fx, mut fy, mut fz) = (x.clone(), y.clone(), z.clone());
        let (mut fvx, mut fvy, mut fvz) = (vec![0.0; 9], vec![0.0; 9], vec![0.0; 9]);
        for i in 0..9 {
            fvx[i] = 500.0 + 100.0 * i as f64;
            fvy[i] = 7700.0 * (1.0 + 0.01 * i as f64);
            fvz[i] = 20.0 * i as f64;
        }
        let (mut px, mut py, mut pz) = (x.clone(), y.clone(), z.clone());
        let (mut pvx, mut pvy, mut pvz) = (fvx.clone(), fvy.clone(), fvz.clone());
        let dt = 10.0;
        for _ in 0..50 {
            for i in 0..9 {
                rk4_j2_step_scalar(
                    &mut fx[i],
                    &mut fy[i],
                    &mut fz[i],
                    &mut fvx[i],
                    &mut fvy[i],
                    &mut fvz[i],
                    mu,
                    j2,
                    re,
                    dt,
                );
                step_plain(
                    &mut px[i],
                    &mut py[i],
                    &mut pz[i],
                    &mut pvx[i],
                    &mut pvy[i],
                    &mut pvz[i],
                    mu,
                    j2,
                    re,
                    dt,
                );
            }
        }
        for i in 0..9 {
            assert!(
                rel_err(fx[i], px[i]) < 1e-9 && rel_err(fvx[i], pvx[i]) < 1e-9,
                "i={i}: FMA r {} vs plain {}",
                fx[i],
                px[i]
            );
            assert!(rel_err(fy[i], py[i]) < 1e-9 && rel_err(fvy[i], pvy[i]) < 1e-9);
            assert!(rel_err(fz[i], pz[i]) < 1e-9 && rel_err(fvz[i], pvz[i]) < 1e-9);
        }
    }
}
