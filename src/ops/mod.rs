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
//!
//! **纯化说明**：本 crate 不依赖任何 DSL/simd 类型；原先 `dsl-*` 命名的内核模块已改为中性名
//! （[`map_ops`]/[`fft_strided`]），对外 `pub` 供外部适配器调用；值枚举/结构见 [`types`]。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（分层策略：内核模块在函数级统一写 SAFETY，
// 不逐块重复；策略全文见 `docs/dev.md` §17）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX/LSX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]
// SOA 内核的签名刻意与 C ABI 一一对应（如 6 个状态数组 + 3 个模型常数），
// 参数个数偏多是这一数据布局的固有形状，不做拆分。
#![allow(clippy::too_many_arguments)]

pub mod axpy;
pub mod ballistic_step;
pub mod batch_distance2d;
/// **广播/标量二元内核**（`map_binary_{rows,cols,scalar}_f32`）。
pub mod broadcast;
pub mod cmatmul;
pub mod cross3_batch;
pub mod dot;
pub mod dot_f16;
pub mod dot_f64;
pub mod dot_i8;
pub mod dot_q4;
/// **f64 逐元素内核**（`map_binary_f64`/`map_affine_f64` 的 LASX 向量版）。
pub mod f64_map;
/// **原地按列 radix-2 复数 FFT**（R19-H）：沿列 8 路向量化、旋转因子 f64 预算降 f32。
pub mod fft_strided;
/// **P4-b 融合内核**（`energy_dc` 单遍去均值能量、`psd_pair` 成对 epilogue）。
pub mod fused;
pub mod gather;
/// **gather / scatter 内核**（`stride` 显式、`ids: &[u32]`、越界 ⇒ `Err`）。
pub mod gather_scatter;
pub mod gelu_erf;
pub mod gelu_quick;
pub mod gemv_i8;
pub mod j2_accel_batch;
pub mod layer_norm;
/// **逐元素/轴归约内核**（`map_binary`(Add/Sub/Mul/Div)、`map_affine`、
/// `reduce_sum_cols`/`reduce_weighted_sum_cols`、`reduce_sum`、轴 max/min）。
pub mod map_ops;
/// **单趟 map + 行和**（`exp(·)` 与行和一次扫描）。
pub mod map_rows_sum;
pub mod mat3_mul_vec3_batch;
pub mod matmul;
/// **NT-GEMM 内核**（`out[i*n+j] = Σ_t a[i*k+t]·b[j*k+t]`；`b` 行主序 ⇒ 不需要物化转置）。
pub mod matmul_bt;
pub mod matmul_f64;
pub mod matmul_i8;
pub mod nn_math;
pub mod norm3_batch;
/// **组合原语内核**（`map_unary_exp`/`map_unary_sqrt` + `reduce_rows_map`）。
pub mod prims;
pub mod quant_i8;
pub mod quat_mul_batch;
pub mod quat_normalize_batch;
pub mod quat_rotate_batch;
pub mod quat_to_dcm_batch;
pub mod rk4_j2_step_batch;
pub mod rms_norm;
pub mod rope;
pub mod silu;
pub mod softmax_rows;
/// **跨步原语**（`copy_strided_f32` / `map_binary_strided_f32`，统一 `Strides` 视图）。
pub mod strided;
pub mod sum;
/// **中性内核类型**（运算种类/融合规格；无外部 crate 依赖）。
pub mod types;
pub mod unitize3_batch;
pub mod vec3_add_scaled_batch;

#[cfg(test)]
pub(crate) mod testutil;
