//! FFI 导出层：C ABI 边界。
//!
//! 这一层只做三件事——把裸指针 + 长度转成切片、调用 [`crate::ops`] 里的算子、
//! 把结果原样返回。**不含任何计算逻辑**，这样 ABI 约定与算法演进互不干扰。
//!
//! 导出符号（15 个，名称与语义与历史版本保持兼容）：
//!
//! | 模块 | 符号 |
//! |---|---|
//! | [`reduce`] | `lasx_dot`、`lasx_sum`、`lasx_dot_f64`、`lasx_axpy` |
//! | [`matmul`] | `lasx_matmul`、`lasx_matmul_f64` |
//! | [`quant`] | `lasx_dot_i8`、`lasx_dot_q4` |
//! | [`batch`] | `lasx_norm3_batch`、`lasx_vec3_add_scaled_batch`、`lasx_batch_distance2d` |
//! | [`physics`] | `lasx_ballistic_step`、`lasx_j2_accel_batch`、`lasx_rk4_j2_step_batch` |
//! | [`memory`] | `lasx_alloc` |
//!
//! # 安全约定
//!
//! 所有导出函数都是**安全函数**（非 `unsafe fn`），遵循 C ABI 惯例：调用方保证
//! 传入的指针对声明的长度有效、可读/可写且不重叠。长度参数为负或为 0 时不做额外
//! 防护（见手册 Caveats）。

pub mod batch;
pub mod matmul;
pub mod memory;
pub mod physics;
pub mod quant;
pub mod reduce;
