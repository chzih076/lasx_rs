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

/// lane-wise `max`（8 个 `f32` 通道各自取较大者）。
///
/// 用于 softmax 的行内最大值：`max` 精确且满足结合律，所以水平归约的次序不影响结果——
/// 行内归约里只有这一步是"次序无关"的（求和不是，见 `ops::softmax_rows` 的设计说明）。
#[inline]
pub fn max_f32x8(a: F32x8, b: F32x8) -> F32x8 {
    // SAFETY: 纯寄存器操作，不碰内存，无前提。
    unsafe { lasx_xvfmax_s(a, b) }
}

/// lane-wise 浮点 → 整数**截断**（向零取整）。
///
/// # Safety
/// 纯寄存器操作、无内存前提；但语义上要求输入是可表示的整数值（否则结果未定义，与 C 的
/// 浮点转整型一致）。`ops::softmax_rows` 的 `exp` 里，输入由 magic 数技巧保证是整数值。
#[inline]
pub unsafe fn trunc_i32(v: F32x8) -> m256i {
    lasx_xvftintrz_w_s(v)
}

/// 由 8 个**偏置指数**构造 `2^n`（整数域左移 23 位后按位重解释成 `f32`）。
///
/// 放在 `arch` 是因为这是内核里唯一需要 `m256i ↔ m256` 转换的地方——§1 的分层约定是
/// "transmute 只在 `arch` 出现"，内核里只出现这种语义化调用。
#[inline]
pub fn pow2_from_exponent(n: m256i) -> F32x8 {
    // SAFETY: 纯寄存器操作（整数移位 + 位型重解释），不碰内存，无前提。
    unsafe { std::mem::transmute(lasx_xvslli_w(n, 23)) }
}
