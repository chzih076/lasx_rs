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

/// 按位型广播到 8 个 32 位 lane（`splat_f32` 接受的是 `f32` 值，这里要的是**位型**，
/// 返回整数向量——给按位 `and`/`xor` 当掩码用）。
#[inline]
fn splat_bits_i32(bits: u32) -> m256i {
    // SAFETY: 纯寄存器操作（GPR → 向量广播），不碰内存，无前提。
    unsafe { lasx_xvreplgr2vr_w(bits as i32) }
}

/// lane-wise 取负（**翻转符号位**：`0x80000000`）。
///
/// 这是精确的按位取负（`±0`、`±inf`、NaN 都保留/翻转正确），一条指令。
///
/// **更正**（2026-09-24，写 `rope` 时发现）：这里原来写的是 `0.0 − x`，理由写成"本机 stdarch
/// 快照没有 LASX 的按位 `xor`"。那个理由是**错的**——`lasx_xvxor_v`/`lasx_xvand_v` 是
/// `portable.rs` 里**宏生成**的，当时只 grep 了 `generated.rs` 所以没找到。现在改成位翻转，
/// 顺带把 `abs` 也改成一条 `and`（见下）。
#[inline]
pub fn neg_f32x8(x: F32x8) -> F32x8 {
    // SAFETY: 纯寄存器操作，不碰内存；transmute 只在 m256/m256i 之间换名字（`docs/dev.md` §1
    // 的分层约定：位型转换只出现在 arch）。
    unsafe {
        let xi: m256i = std::mem::transmute(x);
        std::mem::transmute(lasx_xvxor_v(xi, splat_bits_i32(0x8000_0000)))
    }
}

/// lane-wise 绝对值（清符号位：`and 0x7FFFFFFF`，一条指令）。
#[inline]
pub fn abs_f32x8(x: F32x8) -> F32x8 {
    // SAFETY: 纯寄存器操作，不碰内存；transmute 只在 m256/m256i 之间换名字。
    unsafe {
        let xi: m256i = std::mem::transmute(x);
        std::mem::transmute(lasx_xvand_v(xi, splat_bits_i32(0x7fff_ffff)))
    }
}

/// 载入 16 个 `f16`（256 位）并转成两组 8×`f32`，**按自然序**返回 `(前 8 个, 后 8 个)`。
///
/// **转换是精确的**：f16 的每个值（含次正规、`±inf`）都能被 f32 精确表示，所以这一步不引入
/// 误差；`NaN` 的 payload 由硬件决定（不在逐位契约内）。`ops::dot_f16` 的测试**穷举全部
/// 65536 个 f16 位型**来钉死这一点。
///
/// 内部为什么要两条 `xvpermi_q`：`xvfcvtl.s.h`/`xvfcvth.s.h` 是**128 位 lane 内**操作
/// ——`fcvtl` 转每个 128 位 lane 的**低 4 个** f16，`fcvth` 转**高 4 个**。所以对一次
/// 256 位载入（16 个 f16 = 4 组）拿到的是：
///
/// ```text
/// fcvtl → 元素 {0,1,2,3, 8,9,10,11}
/// fcvth → 元素 {4,5,6,7, 12,13,14,15}
/// ```
///
/// 两条 `xvpermi_q`（128 位 lane 选择）把 `lane0` / `lane1` 分别拼回去，于是调用方看到的是
/// 自然序。**这是实测出来的**（探针打印了真实排布），不是照手册推的——手册只说"低半/高半"，
/// 在 LASX 上"半"是**每个 128 位 lane 的半**。
///
/// # Safety
/// `p` 必须指向至少 16 个可读的 `u16`。
#[inline]
pub unsafe fn load_f16x16_as_f32x8x2(p: *const u16) -> (F32x8, F32x8) {
    let h: m256i = lasx_xvld(p as *const i8, 0);
    let lo: m256i = std::mem::transmute(lasx_xvfcvtl_s_h(h));
    let hi: m256i = std::mem::transmute(lasx_xvfcvth_s_h(h));
    // imm 的 2 位/lane：0 = b.lane0、1 = b.lane1、2 = a.lane0、3 = a.lane1；
    // 低 2 位选输出的 lane0，第 5:4 位选输出的 lane1（实测）。
    // 0x02 → [lo.lane0, hi.lane0] = 元素 0..8；0x13 → [lo.lane1, hi.lane1] = 元素 8..16
    let l: F32x8 = std::mem::transmute(lasx_xvpermi_q::<0x02>(lo, hi));
    let u: F32x8 = std::mem::transmute(lasx_xvpermi_q::<0x13>(lo, hi));
    (l, u)
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
/// 放在 `arch` 是因为这是内核里唯一需要 `m256i ↔ m256` 转换的地方——`docs/dev.md` §1 的分层
/// 约定是"transmute 只在 `arch` 出现"，内核里只出现这种语义化调用。
#[inline]
pub fn pow2_from_exponent(n: m256i) -> F32x8 {
    // SAFETY: 纯寄存器操作（整数移位 + 位型重解释），不碰内存，无前提。
    unsafe { std::mem::transmute(lasx_xvslli_w(n, 23)) }
}
