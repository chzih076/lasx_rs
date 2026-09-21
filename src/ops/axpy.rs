//! `lasx_axpy` —— `y += alpha·x`（单条 FMA 写回）。
//!
//!
use crate::arch::lasx;
use std::arch::loongarch64::*;

/// LASX-only：本内核没有 `has_lasx()` 降级分支，直接执行 256 位实现。
/// 在无 LASX 的 CPU 上会执行 LASX 指令（见手册 Caveats）。
#[inline]
pub(crate) fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    let n = x.len();
    let va = lasx::splat_f32(alpha);
    let mut i = 0;
    while i + 8 <= n {
        let vx = unsafe { lasx::load_f32x8(x.as_ptr().add(i)) };
        let vy = unsafe { lasx::load_f32x8(y.as_ptr().add(i)) };
        let vf = unsafe { lasx_xvfmadd_s(vx, va, vy) };
        unsafe { lasx::store_f32x8(y.as_mut_ptr().add(i), vf) };
        i += 8;
    }
    for j in i..n {
        y[j] += alpha * x[j];
    }
}
