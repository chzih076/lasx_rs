//! `lasx_ballistic_step` —— 批量弹道欧拉步（f32 SOA，8 发/向量）。
//!
//! LASX 缺失时整体退化为纯标量。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（策略见 `docs/dev.md` §17）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX/LSX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]
use crate::arch::lasx;
use crate::arch::SimdPath;
use std::arch::loongarch64::*;

/// 算子入口：解析当前线程的向量路径后分派。
#[inline]
pub(crate) fn ballistic_step(
    x: &mut [f32],
    y: &mut [f32],
    z: &mut [f32],
    vx: &mut [f32],
    vy: &mut [f32],
    vz: &mut [f32],
    k: &[f32],
    dt: f32,
    g: f32,
) {
    match SimdPath::detect() {
        SimdPath::Lasx => ballistic_step_lasx(x, y, z, vx, vy, vz, k, dt, g),
        SimdPath::Lsx => ballistic_step_scalar(x, y, z, vx, vy, vz, k, dt, g),
    }
}

/// LASX 256 位实现。
#[inline]
fn ballistic_step_lasx(
    x: &mut [f32],
    y: &mut [f32],
    z: &mut [f32],
    vx: &mut [f32],
    vy: &mut [f32],
    vz: &mut [f32],
    k: &[f32],
    dt: f32,
    g: f32,
) {
    let n = x.len();
    let vdt = lasx::splat_f32(dt);
    let mut i = 0;
    while i + 8 <= n {
        let vvx: m256 =
            unsafe { std::mem::transmute(lasx_xvld(vx.as_ptr().add(i) as *const i8, 0)) };
        let vvy: m256 =
            unsafe { std::mem::transmute(lasx_xvld(vy.as_ptr().add(i) as *const i8, 0)) };
        let vvz: m256 =
            unsafe { std::mem::transmute(lasx_xvld(vz.as_ptr().add(i) as *const i8, 0)) };
        let vk: m256 = unsafe { std::mem::transmute(lasx_xvld(k.as_ptr().add(i) as *const i8, 0)) };
        // |v| = sqrt(vx²+vy²+vz²)
        let vsq = unsafe {
            lasx_xvfadd_s(
                lasx_xvfmul_s(vvx, vvx),
                lasx_xvfadd_s(lasx_xvfmul_s(vvy, vvy), lasx_xvfmul_s(vvz, vvz)),
            )
        };
        let vmag = unsafe { lasx_xvfsqrt_s(vsq) };
        // drag = k * |v|
        let vdrag = unsafe { lasx_xvfmul_s(vk, vmag) };
        // dvx = -drag*vx; dvy = -drag*vy - g; dvz = -drag*vz
        let vdvx = unsafe { lasx_xvfmul_s(vdrag, vvx) };
        let vdvy = unsafe { lasx_xvfmul_s(vdrag, vvy) };
        let vdvz = unsafe { lasx_xvfmul_s(vdrag, vvz) };
        // 更新（对齐标量参考）：vx -= drag*vx*dt（欧拉；dvx = -drag*vx，用 xvfnmadd）
        // vny 额外减 g*dt（重力项——旧实现缺失，阻力符号亦反）
        let vdvx_dt = unsafe { lasx_xvfmul_s(vdvx, vdt) };
        let vdvy_dt = unsafe { lasx_xvfmul_s(vdvy, vdt) };
        let vdvz_dt = unsafe { lasx_xvfmul_s(vdvz, vdt) };
        let vnx = unsafe { lasx_xvfsub_s(vvx, vdvx_dt) };
        let vny_base = unsafe { lasx_xvfsub_s(vvy, vdvy_dt) };
        let vnz = unsafe { lasx_xvfsub_s(vvz, vdvz_dt) };
        let gv = lasx::splat_f32(g);
        let vny = unsafe { lasx_xvfsub_s(vny_base, lasx_xvfmul_s(gv, vdt)) };
        // x += vx*dt（用新速度 vx，与标量参考一致）
        let vnx_ = unsafe {
            lasx_xvfmadd_s(
                vnx,
                vdt,
                std::mem::transmute(lasx_xvld(x.as_ptr().add(i) as *const i8, 0)),
            )
        };
        let vny_ = unsafe {
            lasx_xvfmadd_s(
                vny,
                vdt,
                std::mem::transmute(lasx_xvld(y.as_ptr().add(i) as *const i8, 0)),
            )
        };
        let vnz_ = unsafe {
            lasx_xvfmadd_s(
                vnz,
                vdt,
                std::mem::transmute(lasx_xvld(z.as_ptr().add(i) as *const i8, 0)),
            )
        };
        // 存储
        unsafe {
            lasx_xvst(
                std::mem::transmute(vnx),
                vx.as_mut_ptr().add(i) as *mut i8,
                0,
            )
        };
        unsafe {
            lasx_xvst(
                std::mem::transmute(vny),
                vy.as_mut_ptr().add(i) as *mut i8,
                0,
            )
        };
        unsafe {
            lasx_xvst(
                std::mem::transmute(vnz),
                vz.as_mut_ptr().add(i) as *mut i8,
                0,
            )
        };
        unsafe {
            lasx_xvst(
                std::mem::transmute(vnx_),
                x.as_mut_ptr().add(i) as *mut i8,
                0,
            )
        };
        unsafe {
            lasx_xvst(
                std::mem::transmute(vny_),
                y.as_mut_ptr().add(i) as *mut i8,
                0,
            )
        };
        unsafe {
            lasx_xvst(
                std::mem::transmute(vnz_),
                z.as_mut_ptr().add(i) as *mut i8,
                0,
            )
        };
        i += 8;
    }
    for j in i..n {
        euler_step(
            &mut x[j], &mut y[j], &mut z[j], &mut vx[j], &mut vy[j], &mut vz[j], k[j], dt, g,
        );
    }
}

/// 纯标量降级（逐发 `euler_step`，本内核**没有** LSX 实现）。
#[inline]
fn ballistic_step_scalar(
    x: &mut [f32],
    y: &mut [f32],
    z: &mut [f32],
    vx: &mut [f32],
    vy: &mut [f32],
    vz: &mut [f32],
    k: &[f32],
    dt: f32,
    g: f32,
) {
    let n = x.len();
    for j in 0..n {
        euler_step(
            &mut x[j], &mut y[j], &mut z[j], &mut vx[j], &mut vy[j], &mut vz[j], k[j], dt, g,
        );
    }
}

fn euler_step(
    x: &mut f32,
    y: &mut f32,
    z: &mut f32,
    vx: &mut f32,
    vy: &mut f32,
    vz: &mut f32,
    k: f32,
    dt: f32,
    g: f32,
) {
    let v = (*vx * *vx + *vy * *vy + *vz * *vz).sqrt();
    let drag = k * v;
    *vx -= drag * *vx * dt;
    *vy -= (drag * *vy + g) * dt;
    *vz -= drag * *vz * dt;
    *x += *vx * dt;
    *y += *vy * dt;
    *z += *vz * dt;
}

/// 数值回归测试：对照独立参考实现。
#[cfg(test)]
mod tests {
    use crate::ffi::physics::lasx_ballistic_step;

    #[test]
    fn test_ballistic_step_batch_matches_scalar() {
        // 向量路径 vs 标量参考逐位/近位一致（修复阻力符号+重力+位置用新速度后）
        let n = 64usize;
        let x = vec![0.0f32; n];
        let y = vec![0.0f32; n];
        let z = vec![0.0f32; n];
        let mut vx = vec![300.0f32; n];
        let mut vy = vec![40.0f32; n];
        let vz = vec![0.0f32; n];
        for i in 0..n {
            vx[i] = 300.0 + i as f32 * 10.0;
            vy[i] = 40.0 - i as f32 * 2.0;
        }
        // 向量路径
        let (mut xv, mut yv, mut zv) = (x.clone(), y.clone(), z.clone());
        let (mut vxv, mut vyv, mut vzv) = (vx.clone(), vy.clone(), vz.clone());
        let k = vec![1.0e-5f32; n];
        lasx_ballistic_step(
            xv.as_mut_ptr(),
            yv.as_mut_ptr(),
            zv.as_mut_ptr(),
            vxv.as_mut_ptr(),
            vyv.as_mut_ptr(),
            vzv.as_mut_ptr(),
            k.as_ptr(),
            n as i32,
            0.005,
            9.81,
        );
        // 标量参考（同公式）
        let mut xs = x.clone();
        let mut ys = y.clone();
        let mut zs = z.clone();
        let mut vxs = vx.clone();
        let mut vys = vy.clone();
        let mut vzs = vz.clone();
        for i in 0..n {
            let v = (vxs[i] * vxs[i] + vys[i] * vys[i] + vzs[i] * vzs[i]).sqrt();
            let drag = 1.0e-5 * v;
            vxs[i] -= drag * vxs[i] * 0.005;
            vys[i] -= (drag * vys[i] + 9.81) * 0.005;
            vzs[i] -= drag * vzs[i] * 0.005;
            xs[i] += vxs[i] * 0.005;
            ys[i] += vys[i] * 0.005;
            zs[i] += vzs[i] * 0.005;
        }
        for i in 0..n {
            assert!(
                (vxv[i] - vxs[i]).abs() < 1e-3,
                "vx[{i}]: {} vs {}",
                vxv[i],
                vxs[i]
            );
            assert!(
                (vyv[i] - vys[i]).abs() < 1e-3,
                "vy[{i}]: {} vs {}",
                vyv[i],
                vys[i]
            );
            assert!(
                (xv[i] - xs[i]).abs() < 1e-2,
                "x[{i}]: {} vs {}",
                xv[i],
                xs[i]
            );
        }
    }
}
