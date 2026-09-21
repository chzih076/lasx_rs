//! `lasx_batch_distance2d` —— 批量 2D 距离 `d = √(dx²+dy²)`（f32 SOA）。
//!
//! LASX 缺失时整体退化为纯标量。
use crate::arch::lasx;
use crate::arch::SimdPath;
use std::arch::loongarch64::*;

/// 算子入口：解析当前线程的向量路径后分派。
#[inline]
pub(crate) fn batch_distance2d(px: f32, py: f32, xs: &[f32], ys: &[f32], out: &mut [f32]) {
    match SimdPath::detect() {
        SimdPath::Lasx => batch_distance2d_lasx(px, py, xs, ys, out),
        SimdPath::Lsx => batch_distance2d_scalar(px, py, xs, ys, out),
    }
}

/// LASX 256 位实现。
#[inline]
fn batch_distance2d_lasx(px: f32, py: f32, xs: &[f32], ys: &[f32], out: &mut [f32]) {
    let n = xs.len();
    let vpx = lasx::splat_f32(px);
    let vpy = lasx::splat_f32(py);
    let mut i = 0;
    while i + 8 <= n {
        let vx: m256 =
            unsafe { std::mem::transmute(lasx_xvld(xs.as_ptr().add(i) as *const i8, 0)) };
        let vy: m256 =
            unsafe { std::mem::transmute(lasx_xvld(ys.as_ptr().add(i) as *const i8, 0)) };
        let dx = unsafe { lasx_xvfsub_s(vx, vpx) };
        let dy = unsafe { lasx_xvfsub_s(vy, vpy) };
        // dist = sqrt(dx*dx + dy*dy)
        let sq = unsafe { lasx_xvfadd_s(lasx_xvfmul_s(dx, dx), lasx_xvfmul_s(dy, dy)) };
        let d = unsafe { lasx_xvfsqrt_s(sq) };
        unsafe {
            lasx_xvst(
                std::mem::transmute(d),
                out.as_mut_ptr().add(i) as *mut i8,
                0,
            )
        };
        i += 8;
    }
    for j in i..n {
        let dx = xs[j] - px;
        let dy = ys[j] - py;
        out[j] = (dx * dx + dy * dy).sqrt();
    }
}

/// 纯标量降级（本内核**没有** LSX 向量实现）。
#[inline]
fn batch_distance2d_scalar(px: f32, py: f32, xs: &[f32], ys: &[f32], out: &mut [f32]) {
    let n = xs.len();
    for j in 0..n {
        let dx = xs[j] - px;
        let dy = ys[j] - py;
        out[j] = (dx * dx + dy * dy).sqrt();
    }
}
