//! FFI 导出层：C ABI 边界。
//!
//! 这一层只做三件事——把裸指针 + 长度转成切片、调用 `ops` 里的算子、
//! 把结果原样返回。**不含任何计算逻辑**，这样 ABI 约定与算法演进互不干扰。
//!
//! 导出符号（59 个：30 个原始符号 + 29 个 `lasx_*_checked` 变体，名称与语义与历史版本保持兼容）：
//!
//! | 模块 | 符号 |
//! |---|---|
//! | [`reduce`] | `lasx_dot`、`lasx_sum`、`lasx_dot_f64`、`lasx_axpy` |
//! | [`matmul`] | `lasx_matmul`、`lasx_matmul_f64` |
//! | [`quant`] | `lasx_dot_i8`、`lasx_dot_q4` |
//! | [`batch`] | `lasx_norm3_batch`、`lasx_vec3_add_scaled_batch`、`lasx_batch_distance2d` |
//! | [`physics`] | `lasx_ballistic_step`、`lasx_j2_accel_batch`、`lasx_rk4_j2_step_batch` |
//! | [`memory`] | `lasx_alloc` |
//! | [`attitude`] | 批量姿态/几何 7 个：叉积、单位化、3×3·向量、四元数四则/旋转/DCM |
//! | [`nn`] | NN 侧 8 个：`lasx_softmax_rows`、`lasx_rms_norm`、`lasx_silu`、`lasx_gelu_quick`、`lasx_gelu_erf`、`lasx_rope`、`lasx_dot_f16`、`lasx_gemv_f16` |
//! | [`checked`] | 29 个 `lasx_*_checked`：带 `int *status` 出参的错误通道 |
//! | [`status`] | [`status::LasxStatus`]：错误码与校验辅助 |
//!
//! # 安全约定
//!
//! 所有导出函数都是**安全函数**（非 `unsafe fn`），遵循 C ABI 惯例：调用方保证
//! 传入的指针对声明的长度有效、可读/可写且不重叠。
//!
//! # 两种用法
//!
//! - **原始 30 个符号**（本模块各子模块）：不做任何校验，误用即 UB——适合已经
//!   自己保证前置条件的调用方，零开销；
//! - **[`checked`] 的 `lasx_*_checked` 变体**：多一个 `int *status` 出参，先校验
//!   指针/长度/形状/物理常数，失败时写入 [`status::LasxStatus`] 并返回安全中性值。
//!   需要把错误**上抛**给上层（如脚本语言）时用它。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（策略见 `docs/dev.md` §17）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX/LSX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]

pub mod attitude;
pub mod batch;
pub mod checked;
pub mod matmul;
pub mod memory;
pub mod nn;
pub mod physics;
pub mod quant;
pub mod reduce;
pub mod status;
