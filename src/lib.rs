//! lasx_rs —— 龙芯 LoongArch 平台的 LASX（256 位）/ LSX（128 位）批量数值内核库。
//!
//! # 平台要求（分发前必读）
//!
//! | 要求 | 说明 |
//! |---|---|
//! | **只在 `loongarch64` 上编译** | `src/` 里 **0 处** `target_arch` 门控、66 个文件直接用 LASX/LSX intrinsic ⇒ 其它架构**编译期**就过不去（这是**有意**的：不给 x86 加标量兜底，避免把"用错平台"从编译期错误退化成运行期错结果）。依赖方必须写成 `[target.'cfg(target_arch = "loongarch64")'.dependencies]`，否则他们的 x86 CI 会直接红（Cargo 为所有目标解析依赖、不按平台跳过） |
//! | **需要 nightly 工具链** | 唯一的 feature 门是 `#![feature(stdarch_loongarch)]`（跟踪 **issue [#117427]**）——stable 上 `std::arch::loongarch64` 仍是 unstable（实测：stable **1.98.1** 与 nightly **1.100.0** 都仍要求该门）。**复查触发条件**：某个 stable 的 release note 宣布 `stdarch_loongarch` 稳定时，删掉 `src/lib.rs`/`cli/src/main.rs` 里的那两行 `#![feature(...)]` 并跑全套测试，然后本 crate 才能真正在 stable 上编译 |
//! | **`-C target-feature=+lasx` 只影响性能** | 仓库的 `.cargo/config.toml` 里有它，但**依赖方的 cargo 不会读依赖自带的 config**。实测：**不带它也能编、数值也位精确**（intrinsic 自带 `#[target_feature(enable = "lasx")]`）；它对性能有意义（没有全局特性 ⇒ 那些 LASX 函数不能跨函数内联），所以想拿满性能请自行加 |
//! | **LA664 勘误** | 3A6000/3C6000/3B6000 的原子操作丢失更新（`amadd` 等无屏障变体）会**静默错结果**；本库的生产路径已全部使用带屏障的序（`AcqRel`），并有"同输入跑两次逐位一致"的回归测试守着。详见 [`docs/platform.md`](https://github.com/chzih076/lasx_rs/blob/master/docs/platform.md) §1 |
//!
//! [#117427]: https://github.com/rust-lang/rust/issues/117427
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
//! | [`parallel`] | 把内核铺到池上的多核调用策略层（`matmul_f32/f64`、`gemv_f16`、`rk4_j2_step_batch`） |
//! | [`arch`] | 指令集能力探测（[`arch::SimdPath`]）与 LASX/LSX 的 load/store/splat 样板 |
//! | `ops` | **算子层**：一个内核一个模块，入口按 `match SimdPath::detect()` 分派；接收切片 |
//! | [`ffi`] | **C ABI 导出层**：41 个原始 `lasx_*` 符号 + 40 个 `_checked` 变体，只做裸指针 → 切片 |
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
// 分层策略（全文见 docs/dev.md §17 —— 本仓库保留的唯一一份策略正本）：
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

// ---------------------------------------------------------------------------
// 架构防呆：错架构要"错得明显"
//
// 本 crate 是 LoongArch LASX/LSX 的**指令集绑定**实现——源码里直接用
// `std::arch::loongarch64::*` 内在函数与内联汇编，**没有任何标量兜底**。不在编译期拦住的话，
// 在 x86/arm 上构建会得到几十条"找不到 loongarch64 intrinsic / 目标不支持该指令"的噪音，
// 真正的原因（用错平台了）反而被淹没。
//
// 这里给出**一条说清楚的话**。口径：这不是"提升了可移植性"，而是**让错误显式**——本库只在
// LoongArch64 上构建/运行；依赖方请写成
// `[target.'cfg(target_arch = "loongarch64")'.dependencies]`，否则他们的 x86 CI 会停在这一行
// （**这是有意的**：把"用错平台"从运行期静默错结果收敛成编译期硬错误）。
//
// 与之**不同**的另一件事：LSX-only 机型（无 LASX，如 2K3000/3B6000M）**能构建也能运行**，
// 只是不能调 `[LASX-only]` 符号——那是运行期问题，用 `lasx_has_lasx()`（C 侧）/
// `SimdPath::detect()`（Rust 侧）防呆。两件事别混。
// ---------------------------------------------------------------------------
#[cfg(not(target_arch = "loongarch64"))]
compile_error!(
    "lasx_rs 是 LoongArch LASX/LSX 指令集绑定实现（手写内联汇编/内在函数），不是可移植标量库。\n\
     请在本机 LoongArch64 上构建，或改用其它后端。\n\
     依赖方请写成 `[target.'cfg(target_arch = \"loongarch64\")'.dependencies]`，\
     否则 x86/arm 上的 CI 会停在这一行（有意为之）。"
);

// **crate 自身的名字别名**：`lasx_rs_macros` 生成的代码一律走**绝对路径**
// （`lasx_rs::shape::{Mat, MatBuf, MatDyn, MatBufDyn}`、`lasx_rs::shape::vec::dot_of`），
// 这样调用方不必 `use` 任何类型。但 Rust 里 crate **不能用自己的名字当路径根**
// （实测 `selfpath::…` ⇒ E0433），而本 crate 自己就是这些宏的消费者
// （`src/laplace.rs`、`src/ilaplace.rs` 里有 `matmul!`）⇒ 需要这条**私有**别名，
// 让生成代码里的 `lasx_rs::…` 在 crate 内部也可解析。
//
// 私有（无 `pub`）⇒ **对外不可见**，不改任何公开签名/行为，也不进产物符号表。
extern crate self as lasx_rs;

pub mod aligned;
pub mod api;
pub mod arch;
/// **`simd_dsl::Backend` 的 `lasx_rs` 实现**（**feature 门控** `dsl-backend`；默认关 ⇒ 无依赖、无编译）
/// ——见 [`dsl_backend`] 模块文档（③a 只接必需方法 `matmul_f32`/`dot_f32`）。
#[cfg(feature = "dsl-backend")]
pub mod dsl_backend;
/// **超越函数 recon 原型**（opt-in；默认仍走 `libm`）——见 [`fast_trig`] 与 `docs/ops.md` §9.10。
pub mod fast_trig;
pub mod ffi;
/// **逆拉普拉斯变换**（Dubner–Abate + 跳变扣除；指数和仍用公式 DSL 表达）——见 [`ilaplace`]。
pub mod ilaplace;
/// **单边数值拉普拉斯变换**（用公式 DSL 表达的批量矩阵乘）——见 [`laplace`] 模块文档。
pub mod laplace;
/// **表达式级代数重写：模式识别 + 安全区分析**（原型；**默认路径行为不变**）——见 [`rewrite`]。
pub mod rewrite;
/// **复数矩阵乘核心**（A 阶：分平面 `(Re, Im)`、f32/f64、直算 4 条 / Karatsuba 3 条）——
/// 见 [`cmatmul`] 模块文档（默认路线与位精确契约差异）。
pub use ops::cmatmul;
mod ops;
pub mod parallel;
pub mod plan;
pub mod pool;
pub mod shape;
pub mod view;

pub use arch::lasx_force_lsx_thread;

/// 公式 DSL：`matmul!(y[M, N] = x[M, K] * w[K, N])`。
///
/// 实现在 `lasx_rs_macros`（**随本 crate 一起发布到 crates.io**，零第三方依赖），这里重导出，
/// 于是用户视角只有一个依赖 `lasx_rs`。规则与诊断见 [`shape`] 模块文档。
pub use lasx_rs_macros::{dot, gemv, matmul};

/// 内部基准用的转发（不面向使用者；生产路径由 `lasx_matmul` 的按形状分派选择）。
#[doc(hidden)]
pub fn ops_bench_packed(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    ops::matmul::matmul_f32_packed(m, k, n, a, b, c, ops::matmul::Epi::Plain);
}

/// `C = alpha·(A·B) + beta·C`（原地）；正式实现见 [`api::matmul_scaled_into`]。
///
/// 保留这个 `#[doc(hidden)]` 入口只为让基准/例子能直接量内核层（`examples/fusion_kernel.rs`
/// 用它做"融合 vs 两遍"的对照），语义与 `api` 层一致：`beta == 0` 不读 `C`。
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

/// 同上，f64 侧（打包路径的 A/B 基准用）。
#[doc(hidden)]
pub fn ops_bench_packed_f64(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    ops::matmul_f64::matmul_f64_packed(m, k, n, a, b, c);
}

/// 强制 f32 走"列块在外"次序（A/B 基准）。
#[doc(hidden)]
pub fn ops_bench_cols_f32(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    ops::matmul::matmul_f32_cols_block(
        m,
        k,
        n,
        0,
        n,
        a,
        b,
        c,
        ops::matmul::COL_BLOCK,
        ops::matmul::Epi::Plain,
    );
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
pub use ffi::matmul::{lasx_matmul, lasx_matmul_f64, lasx_matmul_scaled};
pub use ffi::memory::lasx_alloc;
pub use ffi::physics::{lasx_ballistic_step, lasx_j2_accel_batch, lasx_rk4_j2_step_batch};
pub use ffi::quant::{lasx_dot_i8, lasx_dot_q4};
// N3 批次（int8 推理）：新增符号按 C ABI"只追加"的约定挂在这里，
// `lasx_*_checked` 变体在 `lasx_rs::ffi::checked`。
pub use ffi::quant::{
    lasx_absmax_rows, lasx_amax, lasx_dequantize_i8, lasx_dequantize_i8_rows, lasx_gemv_i8,
    lasx_matmul_i8, lasx_quantize_i8_per_row, lasx_quantize_i8_per_tensor,
};
pub use ffi::reduce::{lasx_axpy, lasx_dot, lasx_dot_f64, lasx_sum};
// NN 侧 N1 批次：与上面的导出保持一致，Rust 侧可以直接 `lasx_rs::lasx_silu`。
// 上一轮加 `softmax_rows`/`rms_norm` 时漏了这行重新导出——只有 `lasx_rs::ffi::nn::*` 能用，
// 这一轮补齐（符号表列的是 C ABI 名，Rust 路径不该和它不一致）。
pub use ffi::nn::{
    lasx_dot_f16, lasx_gather_rows, lasx_gather_rows_i8, lasx_gelu_erf, lasx_gelu_quick,
    lasx_gemv_f16, lasx_layer_norm, lasx_rms_norm, lasx_rope, lasx_silu, lasx_softmax_rows,
};
