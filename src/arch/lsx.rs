//! LSX（128 位）样板：类型别名与载入/存储/广播/置零。
//!
//! 供 LASX 缺失时的降级路径使用（3A5000/3A6000 等 LSX-only 机器），
//! 也是测试里 `lasx_force_lsx_thread(true)` 强制执行的路径。

use std::arch::loongarch64::*;

/// 4×`f32` 向量（128 位）。
pub type F32x4 = m128;
/// 2×`f64` 向量（128 位）。
pub type F64x2 = m128d;

/// 载入 4 个连续 `f32`。
///
/// # Safety
/// `p` 必须指向至少 4 个可读的 `f32`。
#[inline]
pub unsafe fn load_f32x4(p: *const f32) -> F32x4 {
    std::mem::transmute(lsx_vld(p as *const i8, 0))
}

/// 写回 4 个连续 `f32`。
///
/// # Safety
/// `p` 必须指向至少 4 个可写的 `f32`。
#[inline]
pub unsafe fn store_f32x4(p: *mut f32, v: F32x4) {
    lsx_vst(std::mem::transmute(v), p as *mut i8, 0);
}

/// 全零 `f32` 向量。
#[inline]
pub fn zero_f32x4() -> F32x4 {
    unsafe { std::mem::transmute(lsx_vldi(0)) }
}

/// 载入 2 个连续 `f64`。
///
/// # Safety
/// `p` 必须指向至少 2 个可读的 `f64`。
#[inline]
pub unsafe fn load_f64x2(p: *const f64) -> F64x2 {
    std::mem::transmute(lsx_vld(p as *const i8, 0))
}

/// 写回 2 个连续 `f64`。
///
/// # Safety
/// `p` 必须指向至少 2 个可写的 `f64`。
#[inline]
pub unsafe fn store_f64x2(p: *mut f64, v: F64x2) {
    lsx_vst(std::mem::transmute(v), p as *mut i8, 0);
}

/// 将标量广播到 2 个 `f64` 通道。
#[inline]
pub fn splat_f64(x: f64) -> F64x2 {
    let bits = x.to_bits() as i64;
    unsafe { std::mem::transmute(lsx_vreplgr2vr_d(bits)) }
}
