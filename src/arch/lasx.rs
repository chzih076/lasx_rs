//! LASX（256 位）样板：类型别名与对齐无关的载入/存储/广播/置零。
//!
//! 这些函数把 `stdarch` 的裸 `m256i` 与具名浮点向量之间的 `transmute` 收敛到一处，
//! 内核里只出现 `lasx::load_f32x8(..)` 这类语义化调用。

use std::arch::loongarch64::*;

/// 8×`f32` 向量（256 位）。
pub type F32x8 = m256;
/// 4×`f64` 向量（256 位）。
pub type F64x4 = m256d;

/// 载入 8 个连续 `f32`（无需 32 字节对齐）。
///
/// # Safety
/// `p` 必须指向至少 8 个可读的 `f32`。
#[inline]
pub unsafe fn load_f32x8(p: *const f32) -> F32x8 {
    std::mem::transmute(lasx_xvld(p as *const i8, 0))
}

/// 写回 8 个连续 `f32`。
///
/// # Safety
/// `p` 必须指向至少 8 个可写的 `f32`。
#[inline]
pub unsafe fn store_f32x8(p: *mut f32, v: F32x8) {
    lasx_xvst(std::mem::transmute(v), p as *mut i8, 0);
}

/// 载入 4 个连续 `f64`。
///
/// # Safety
/// `p` 必须指向至少 4 个可读的 `f64`。
#[inline]
pub unsafe fn load_f64x4(p: *const f64) -> F64x4 {
    std::mem::transmute(lasx_xvld(p as *const i8, 0))
}

/// 写回 4 个连续 `f64`。
///
/// # Safety
/// `p` 必须指向至少 4 个可写的 `f64`。
#[inline]
pub unsafe fn store_f64x4(p: *mut f64, v: F64x4) {
    lasx_xvst(std::mem::transmute(v), p as *mut i8, 0);
}

/// 全零 `f32` 向量。
#[inline]
pub fn zero_f32x8() -> F32x8 {
    // SAFETY: `xvldi 0` 是全零向量，transmute 只在 m256i/m256 之间换名字，无前提。
    unsafe { std::mem::transmute(lasx_xvldi(0)) }
}

/// 全零 `f64` 向量。
#[inline]
pub fn zero_f64x4() -> F64x4 {
    // SAFETY: `xvldi 0` 是全零向量，transmute 只在 m256i/m256 之间换名字，无前提。
    unsafe { std::mem::transmute(lasx_xvldi(0)) }
}

/// 全零 8×`i32` 向量（整数内核的累加器初值）。
#[inline]
pub fn zero_i32x8() -> m256i {
    // SAFETY: 纯寄存器操作（生成全零向量），不碰内存，无前提。
    unsafe { lasx_xvldi(0) }
}

/// 将标量广播到 8 个 `f32` 通道。
#[inline]
pub fn splat_f32(x: f32) -> F32x8 {
    let bits = x.to_bits() as i32;
    // SAFETY: 纯寄存器操作（GPR 位型 → 向量广播），不碰内存，无前提。
    unsafe { std::mem::transmute(lasx_xvreplgr2vr_w(bits)) }
}

/// 将标量广播到 4 个 `f64` 通道。
#[inline]
pub fn splat_f64(x: f64) -> F64x4 {
    let bits = x.to_bits() as i64;
    // SAFETY: 纯寄存器操作（GPR 位型 → 向量广播），不碰内存，无前提。
    unsafe { std::mem::transmute(lasx_xvreplgr2vr_d(bits)) }
}
