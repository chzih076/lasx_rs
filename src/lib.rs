//! lasx_rs —— 龙芯 LoongArch 平台的 LASX（256 位）/ LSX（128 位）批量数值内核库。
//!
//! 无 LASX 的 CPU（3A5000/3A6000 等 LSX-only）自动降级到 LSX 路径，逐位确定。
//!
//! # 代码组织
//!
//! | 模块 | 职责 |
//! |---|---|
//! | [`api`] | **Rust 原生 API**：切片进出、`Result` 上抛、输出是 [`aligned::AlignedVec`] |
//! | [`aligned`] | [`aligned::AlignedVec`]：保证 32 字节以上对齐的缓冲区（LASX 性能前提） |
//! | [`view`] | [`view::MatRef`] / [`view::MatMut`]：带形状与跨距的矩阵视图，构造时校验一次 |
//! | [`plan`] | [`plan::MatmulPlan`]：把 `B` 打包一次、反复算 `A·B`（`B` 固定时用它替代 [`api::matmul`]） |
//! | [`shape`] | [`shape::Mat`] / [`shape::MatDyn`]：**形状进类型**（编译期排版 + 形状错误编译期报 + 声明式调用） |
//! | [`pool`] | [`pool::WorkerPool`]：可选常驻线程池，把批量内核铺到多核且跨调用复用线程 |
//! | [`parallel`] | 把内核铺到池上的多核调用策略层（`matmul_f32/f64`、`rk4_j2_step_batch`） |
//! | [`arch`] | 指令集能力探测（[`arch::SimdPath`]）与 LASX/LSX 的 load/store/splat 样板 |
//! | `ops` | **算子层**：一个内核一个模块，入口按 `match SimdPath::detect()` 分派；接收切片 |
//! | [`ffi`] | **C ABI 导出层**：22 个 `lasx_*` 符号（历史 15 + 姿态/几何 7），只做裸指针 → 切片 |
//!
//! 分层的目的是让"算法"与"ABI 约定"互不干扰：算子只处理安全的切片，指针有效性与
//! 长度约定收敛在 [`ffi`] 一层；新增内核时只需在 `ops` 下加一个文件、在 [`ffi`]
//! 下加一个薄包装。
//!
//! # 快速上手
//!
//! Rust 调用方用 [`api`]（切片 + `Result`）：
//!
//! ```
//! let a = [1.0f32, 2.0, 3.0, 4.0];
//! let b = [1.0f32; 4];
//! assert_eq!(lasx_rs::api::dot(&a, &b).unwrap(), 10.0);
//! // 要输出的内核直接返回 32 字节对齐的缓冲
//! let c = lasx_rs::api::matmul(2, 2, 2, &[1.0, 0.0, 0.0, 1.0], &[1.0, 2.0, 3.0, 4.0]).unwrap();
//! assert_eq!(&c[..], &[1.0, 2.0, 3.0, 4.0]);
//! ```
//!
//! C / Dart 等 FFI 调用方用裸指针版本（[`ffi`]，或 crate 根的历史重导出）：
//!
//! ```no_run
//! # let (a, b) = (vec![1.0f32, 2.0, 3.0, 4.0], vec![1.0f32; 4]);
//! # let n = a.len() as i32;
//! let d = lasx_rs::ffi::reduce::lasx_dot(a.as_ptr(), b.as_ptr(), n);
//! // crate 根的重导出与历史 API 完全等价
//! let d2 = lasx_rs::lasx_dot(a.as_ptr(), b.as_ptr(), n);
//! assert_eq!(d, d2);
//! ```
//!
//! 测试/验证用的线程级降级钩子见 [`lasx_force_lsx_thread`]。
#![feature(stdarch_loongarch)]
// **unsafe 审查**：`undocumented_unsafe_blocks` 要求每个 `unsafe` 块都紧邻一条 SAFETY 说明。
// 策略（见 docs/dev.md §17）：
// - `src/pool`、`src/arch`、`src/aligned`、`src/api`、`src/parallel`、`src/scalar_ref.rs`
//   这些**基础设施**里逐块写清楚（那里才是指针/生命周期/并发不变量的所在）；
// - `src/ops/*`、`src/ffi/*` 里同一组前提下的 intrinsic 调用是重复的（406 个 unsafe 块里
//   约 300 个是这种），逐块抄注释只会变噪声 —— 这些文件显式豁免，不变量在各自函数级 SAFETY
//   段里统一给出（模块头也写了豁免理由）。
#![warn(clippy::undocumented_unsafe_blocks)]
// SIMD intrinsics 固有：裸 i8 向量 ↔ 类型化向量（F32x8/F64x4 等）的 transmute，
// 目标类型由辅助函数签名约束，显式标注为样板噪音。
#![allow(clippy::missing_transmute_annotations)]
// extern "C" FFI 函数按约定解引用传入裸指针（调用方保证有效），非 unsafe fn 语义
#![allow(clippy::not_unsafe_ptr_arg_deref)]

pub mod aligned;
pub mod api;
pub mod arch;
pub mod ffi;
mod ops;
pub mod parallel;
pub mod plan;
pub mod pool;
pub mod shape;
pub mod view;

pub use arch::lasx_force_lsx_thread;

/// 公式 DSL：`matmul!(y[M, N] = x[M, K] * w[K, N])`。
///
/// 实现在内部的 `lasx_rs_macros`（`publish = false`，零第三方依赖），这里重导出，
/// 于是用户视角只有一个依赖 `lasx_rs`。规则与诊断见 [`shape`] 模块文档。
pub use lasx_rs_macros::matmul;

/// 内部基准用的转发（不面向使用者；生产路径由 `lasx_matmul` 的按形状分派选择）。
#[doc(hidden)]
pub fn ops_bench_packed(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    ops::matmul::matmul_f32_packed(m, k, n, a, b, c);
}

/// **实验性**：`C = alpha·(A·B) + beta·C`（原地），只在 k 不分块、`m%4==0`、`n%32==0`
/// 时成立。用于量"融合内核能不能保持非融合内核的速度"（`docs/dev.md` §19.11）。
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn ops_bench_scaled(
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    beta: f32,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    ops::matmul::matmul_f32_scaled(m, k, n, alpha, beta, a, b, c);
}

/// 同上，f64 侧（`docs/dev.md` §8.5 的 A/B 用）。
#[doc(hidden)]
pub fn ops_bench_packed_f64(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    ops::matmul_f64::matmul_f64_packed(m, k, n, a, b, c);
}

/// 强制 f32 走"列块在外"次序（A/B 基准；`docs/dev.md` §8.4）。
#[doc(hidden)]
pub fn ops_bench_cols_f32(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    ops::matmul::matmul_f32_cols_block(m, k, n, 0, n, a, b, c, ops::matmul::COL_BLOCK);
}

/// 强制 f64 走流式次序（A/B 基准）。
#[doc(hidden)]
pub fn ops_bench_stream_f64(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    ops::matmul_f64::matmul_f64_stream(m, k, n, a, b, c);
}

// 历史 API 兼容：`lasx_*` 符号原先位于 crate 根，继续在此重导出，
// 使既有的 `lasx_rs::lasx_dot(..)` 调用与文档无需改动。
pub use ffi::attitude::{
    lasx_cross3_batch, lasx_mat3_mul_vec3_batch, lasx_quat_mul_batch, lasx_quat_normalize_batch,
    lasx_quat_rotate_batch, lasx_quat_to_dcm_batch, lasx_unitize3_batch,
};
pub use ffi::batch::{lasx_batch_distance2d, lasx_norm3_batch, lasx_vec3_add_scaled_batch};
pub use ffi::matmul::{lasx_matmul, lasx_matmul_f64};
pub use ffi::memory::lasx_alloc;
pub use ffi::physics::{lasx_ballistic_step, lasx_j2_accel_batch, lasx_rk4_j2_step_batch};
pub use ffi::quant::{lasx_dot_i8, lasx_dot_q4};
pub use ffi::reduce::{lasx_axpy, lasx_dot, lasx_dot_f64, lasx_sum};
// NN 侧 N1 批次（`docs/dev.md` §20）：与上面的导出保持一致，Rust 侧可以直接 `lasx_rs::lasx_silu`。
// 上一轮加 `softmax_rows`/`rms_norm` 时漏了这行重新导出——只有 `lasx_rs::ffi::nn::*` 能用，
// 这一轮补齐（`docs/ops.md` §4 的符号表列的是 C ABI 名，Rust 路径不该和它不一致）。
pub use ffi::nn::{
    lasx_gelu_erf, lasx_gelu_quick, lasx_rms_norm, lasx_rope, lasx_silu, lasx_softmax_rows,
};
