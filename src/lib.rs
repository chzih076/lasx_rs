//! lasx_rs —— 龙芯 LoongArch 平台的 LASX（256 位）/ LSX（128 位）批量数值内核库。
//!
//! 无 LASX 的 CPU（3A5000/3A6000 等 LSX-only）自动降级到 LSX 路径，逐位确定。
//!
//! # 代码组织
//!
//! | 模块 | 职责 |
//! |---|---|
//! | [`aligned`] | [`aligned::AlignedVec`]：保证 32 字节以上对齐的缓冲区（LASX 性能前提） |
//! | [`arch`] | 指令集能力探测（[`arch::SimdPath`]）与 LASX/LSX 的 load/store/splat 样板 |
//! | `ops` | **算子层**：一个内核一个模块，入口按 `match SimdPath::detect()` 分派；接收切片 |
//! | [`ffi`] | **C ABI 导出层**：15 个 `lasx_*` 符号，只做裸指针 → 切片，不含计算 |
//!
//! 分层的目的是让"算法"与"ABI 约定"互不干扰：算子只处理安全的切片，指针有效性与
//! 长度约定收敛在 [`ffi`] 一层；新增内核时只需在 `ops` 下加一个文件、在 [`ffi`]
//! 下加一个薄包装。
//!
//! # 快速上手
//!
//! ```no_run
//! # let (a, b) = (vec![1.0f32, 2.0, 3.0, 4.0], vec![1.0f32; 4]);
//! # let n = a.len() as i32;
//! // 走 C ABI 导出层
//! let d = lasx_rs::ffi::reduce::lasx_dot(a.as_ptr(), b.as_ptr(), n);
//! // crate 根的重导出与历史 API 完全等价
//! let d2 = lasx_rs::lasx_dot(a.as_ptr(), b.as_ptr(), n);
//! assert_eq!(d, d2);
//! ```
//!
//! 测试/验证用的线程级降级钩子见 [`lasx_force_lsx_thread`]。
#![feature(stdarch_loongarch)]
// SIMD intrinsics 固有：裸 i8 向量 ↔ 类型化向量（F32x8/F64x4 等）的 transmute，
// 目标类型由辅助函数签名约束，显式标注为样板噪音。
#![allow(clippy::missing_transmute_annotations)]
// extern "C" FFI 函数按约定解引用传入裸指针（调用方保证有效），非 unsafe fn 语义
#![allow(clippy::not_unsafe_ptr_arg_deref)]

pub mod aligned;
pub mod arch;
pub mod ffi;
mod ops;

pub use arch::lasx_force_lsx_thread;

// 历史 API 兼容：`lasx_*` 符号原先位于 crate 根，继续在此重导出，
// 使既有的 `lasx_rs::lasx_dot(..)` 调用与文档无需改动。
pub use ffi::batch::{lasx_batch_distance2d, lasx_norm3_batch, lasx_vec3_add_scaled_batch};
pub use ffi::matmul::{lasx_matmul, lasx_matmul_f64};
pub use ffi::memory::lasx_alloc;
pub use ffi::physics::{lasx_ballistic_step, lasx_j2_accel_batch, lasx_rk4_j2_step_batch};
pub use ffi::quant::{lasx_dot_i8, lasx_dot_q4};
pub use ffi::reduce::{lasx_axpy, lasx_dot, lasx_dot_f64, lasx_sum};
