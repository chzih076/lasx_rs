//! `lasx_j2_accel_batch` —— 批量中心引力 + J2 摄动加速度（f64 SOA）。
//!
//! LASX 缺失时走 LSX 128 位路径。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（策略见 `docs/dev.md` §17）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX/LSX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]
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
        // **只做一次除法**：有了 1/|r|²，其余倒数用乘法推出来
        //   invrm = (1/|r|²)·|r| = 1/|r|，inv3 = (1/|r|²)·(1/|r|) = 1/|r|³，
        //   inv5 = inv3·(1/|r|²) = 1/|r|⁵，zr2 = z²·(1/|r|²)
        // 比"三次除法"多 ~2 ulp 舍入（有意，见 docs/dev.md §13）；三条路径同式 ⇒ 仍逐位一致。
        let vone = lasx::splat_f64(1.0);
        let vinv2 = unsafe { lasx_xvfdiv_d(vone, vrm2) };
        let vinvrm = unsafe { lasx_xvfmul_d(vinv2, vrm) };
        let vinv3 = unsafe { lasx_xvfmul_d(vinv2, vinvrm) };
        let vinv5 = unsafe { lasx_xvfmul_d(vinv3, vinv2) };
        // 中心项 −μ·r/|r|³
        let vcen = unsafe { lasx_xvfmul_d(vmu, vinv3) };
        // J2 项系数（正）：1.5·J2·μ·Re²/rm⁵
        let vk = unsafe { lasx_xvfmul_d(vj2k, vinv5) };
        // zr2 = (z/|r|)²
        let vzr2 = unsafe { lasx_xvfmul_d(lasx_xvfmul_d(vz, vz), vinv2) };
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
        // 与向量路径同式：1 次除法 + 乘法推导
        let inv2 = 1.0 / rm2;
        let invrm = inv2 * rm;
        let inv3 = inv2 * invrm;
        let inv5 = inv3 * inv2;
        let zr2 = (rz[j] * rz[j]) * inv2;
        let k = j2k * inv5;
        let vcen = -mu * inv3;
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
        // 与 LASX / 标量路径**同式**：1 次除法 + 乘法推导（docs/dev.md §13）
        let vone = lsx::splat_f64(1.0);
        let vinv2 = unsafe { lsx_vfdiv_d(vone, vrm2) };
        let vinvrm = unsafe { lsx_vfmul_d(vinv2, vrm) };
        let vinv3 = unsafe { lsx_vfmul_d(vinv2, vinvrm) };
        let vinv5 = unsafe { lsx_vfmul_d(vinv3, vinv2) };
        let vcen = unsafe { lsx_vfmul_d(vmu, vinv3) };
        let vk = unsafe { lsx_vfmul_d(vs, vinv5) };
        let vzr2 = unsafe { lsx_vfmul_d(lsx_vfmul_d(vz, vz), vinv2) };
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
        // 与向量路径同式：1 次除法 + 乘法推导
        let inv2 = 1.0 / rm2;
        let invrm = inv2 * rm;
        let inv3 = inv2 * invrm;
        let inv5 = inv3 * inv2;
        let zr2 = (rz[j] * rz[j]) * inv2;
        let k = j2k * inv5;
        let vcen = -mu * inv3;
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
    /// 精度回归：`1/|r|³`、`1/|r|⁵`、`zr2` 由 `1/|r|²` 用乘法推导（省掉两次除法），
    /// 比直接除法多几次舍入。上界取 8 ulp，并打印实测最坏值供文档引用。
    #[test]
    fn reciprocal_derivation_precision() {
        let mut worst3 = 0f64;
        let mut worst5 = 0f64;
        let mut worst_zr2 = 0f64;
        for i in 0..4096 {
            // |r|² ∈ [1, 10)：覆盖典型近地轨道量级
            let rm2 = 1.0 + 9.0 * (i as f64) / 4096.0;
            let rm = rm2.sqrt();
            let inv2 = 1.0 / rm2;
            let invrm = inv2 * rm;
            let inv3 = inv2 * invrm;
            let inv5 = inv3 * inv2;
            let e3 = 1.0 / (rm2 * rm);
            let e5 = 1.0 / (rm2 * rm2 * rm);
            worst3 = worst3.max(((inv3 - e3) / e3).abs());
            worst5 = worst5.max(((inv5 - e5) / e5).abs());
            let z = 0.37 * rm;
            let zr2 = (z * z) * inv2;
            let exact = (z * z) / rm2;
            worst_zr2 = worst_zr2.max(((zr2 - exact) / exact).abs());
        }
        println!("最坏相对偏差：1/|r|³={worst3:e} 1/|r|⁵={worst5:e} zr2={worst_zr2:e}");
        let bound = 8.0 * f64::EPSILON;
        assert!(worst3 < bound, "1/|r|³ {worst3:e}");
        assert!(worst5 < bound, "1/|r|⁵ {worst5:e}");
        assert!(worst_zr2 < bound, "zr2 {worst_zr2:e}");
    }

    /// **逐元素内核的分块无关性**：同一个元素不管落在向量体里还是标量尾里，
    /// 都必须给出逐位相同的结果（把除法换成乘法的推导差点破坏这条）。
    #[test]
    fn vector_and_tail_bit_exact() {
        const MU: f64 = 3.986004418e14;
        const J2: f64 = 1.08262668e-3;
        const RE: f64 = 6.378137e6;
        for k in [1usize, 8, 64] {
            let nv = 4 * k; // 向量体覆盖的元素数（LASX）
            let (x, y, z) = states(nv + 3);
            let (mut ax1, mut ay1, mut az1) = (vec![0.0; nv], vec![0.0; nv], vec![0.0; nv]);
            let (mut ax2, mut ay2, mut az2) =
                (vec![0.0; nv + 3], vec![0.0; nv + 3], vec![0.0; nv + 3]);
            lasx_j2_accel_batch(
                x.as_ptr(),
                y.as_ptr(),
                z.as_ptr(),
                MU,
                J2,
                RE,
                ax1.as_mut_ptr(),
                ay1.as_mut_ptr(),
                az1.as_mut_ptr(),
                nv as i32,
            );
            lasx_j2_accel_batch(
                x.as_ptr(),
                y.as_ptr(),
                z.as_ptr(),
                MU,
                J2,
                RE,
                ax2.as_mut_ptr(),
                ay2.as_mut_ptr(),
                az2.as_mut_ptr(),
                (nv + 3) as i32,
            );
            for i in 0..nv {
                assert_eq!(ax1[i].to_bits(), ax2[i].to_bits(), "ax @ {i} (k={k})");
                assert_eq!(ay1[i].to_bits(), ay2[i].to_bits(), "ay @ {i} (k={k})");
                assert_eq!(az1[i].to_bits(), az2[i].to_bits(), "az @ {i} (k={k})");
            }
        }
    }

    /// LASX 与 LSX 两条向量路径必须**逐位一致**（每阶段 3 次除法降成 1 次后仍成立，
    /// 因为两条路径用的是完全同式的算式）。
    #[test]
    fn lasx_vs_lsx_bit_exact() {
        const MU: f64 = 3.986004418e14;
        const J2: f64 = 1.08262668e-3;
        const RE: f64 = 6.378137e6;
        let n = 257usize; // 覆盖 2 元素向量的尾部
        let (x, y, z) = states(n);
        let (mut ax1, mut ay1, mut az1) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let (mut ax2, mut ay2, mut az2) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        lasx_j2_accel_batch(
            x.as_ptr(),
            y.as_ptr(),
            z.as_ptr(),
            MU,
            J2,
            RE,
            ax1.as_mut_ptr(),
            ay1.as_mut_ptr(),
            az1.as_mut_ptr(),
            n as i32,
        );
        crate::arch::lasx_force_lsx_thread(true);
        lasx_j2_accel_batch(
            x.as_ptr(),
            y.as_ptr(),
            z.as_ptr(),
            MU,
            J2,
            RE,
            ax2.as_mut_ptr(),
            ay2.as_mut_ptr(),
            az2.as_mut_ptr(),
            n as i32,
        );
        crate::arch::lasx_force_lsx_thread(false);
        for i in 0..n {
            assert_eq!(ax1[i].to_bits(), ax2[i].to_bits(), "ax @ {i}");
            assert_eq!(ay1[i].to_bits(), ay2[i].to_bits(), "ay @ {i}");
            assert_eq!(az1[i].to_bits(), az2[i].to_bits(), "az @ {i}");
        }
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
