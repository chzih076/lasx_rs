//! 算子层：一个 FFI 内核一个模块。
//!
//! 约定：
//!
//! - 算子函数一律接收**切片**（`&[T]` / `&mut [T]`），裸指针解引用与长度约定属于
//!   [`crate::ffi`] 的职责；这样算子本身是安全的纯函数，便于测试与复用。
//! - 支持降级的算子在入口用 `match SimdPath::detect()` 分派到 `<name>_lasx` 与
//!   `<name>_lsx` / `<name>_scalar`，每个实现是一个独立函数（而不是嵌套分支）。
//! - **LASX-only** 的算子没有降级分支，在模块文档里显式标注。
//! - 每个模块自带 `#[cfg(test)] mod tests`，对照**独立**参考实现校验数值
//!   （故意不复用算子内部的标量兜底，避免自证）。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（策略见 `docs/dev.md` §17）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX/LSX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]
// SOA 内核的签名刻意与 C ABI 一一对应（如 6 个状态数组 + 3 个模型常数），
// 参数个数偏多是这一数据布局的固有形状，不做拆分。
#![allow(clippy::too_many_arguments)]

pub mod axpy;
pub mod ballistic_step;
pub mod batch_distance2d;
pub mod cross3_batch;
pub mod dot;
pub mod dot_f64;
pub mod dot_i8;
pub mod dot_q4;
pub mod j2_accel_batch;
pub mod mat3_mul_vec3_batch;
pub mod matmul;
pub mod matmul_f64;
pub mod norm3_batch;
pub mod quat_mul_batch;
pub mod quat_normalize_batch;
pub mod quat_rotate_batch;
pub mod quat_to_dcm_batch;
pub mod rk4_j2_step_batch;
pub mod sum;
pub mod unitize3_batch;
pub mod vec3_add_scaled_batch;

#[cfg(test)]
pub(crate) mod testutil;
