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
pub mod cmatmul;
pub mod cross3_batch;
pub mod dot;
pub mod dot_f16;
pub mod dot_f64;
pub mod dot_i8;
pub mod dot_q4;
/// **DSL 后端的 LASX 原地按列 radix-2 复数 FFT**（R19-H）：沿列 8 路向量化、旋转因子 f64 预算降 f32。
/// 同样挂在 `dsl-backend` 下（唯一调用方是 `src/dsl_backend.rs` 的 `fft_strided_f32`）。
#[cfg(feature = "dsl-backend")]
pub mod dsl_fft;
/// **DSL 后端的逐元素/轴归约 LASX 内核**（R19-E）：`map_binary`(Add/Sub/Mul/Div)、`map_affine`、
/// `reduce_sum_cols`/`reduce_weighted_sum_cols`、`reduce_sum`。**与标量参照逐位**（`map_affine`
/// 故意不融 FMA；轴归约每列一条 lane、按行序累加）——理由见文件头。
///
/// **挂在 `dsl-backend` feature 下**：唯一的调用方就是 `src/dsl_backend.rs`（同样受该 feature 门控）；
/// 不门控的话默认构建会把它编成"无人使用的模块"⇒ `clippy -D warnings` 的 `dead_code` 直接红
/// （CI 实测过）。**不要用 `#[allow(dead_code)]` 掩盖**——那会同时掩盖未来真的死代码。
#[cfg(feature = "dsl-backend")]
pub mod dsl_map;

/// **DSL 后端的广播/标量二元内核**（`map_binary_{rows,cols,scalar}_f32` 的 override 落点）：
/// rows 广播每行一次标量 + SIMD 扫 `cols`、cols 广播每行 SIMD 连续 `w[c]`、标量整段 SIMD；
/// `Add/Sub/Mul/Div` 与标量参照**逐位**，`Max/Min` 委托标量参照（NaN 语义）。
///
/// **同样挂在 `dsl-backend` 下**（唯一调用方 `src/dsl_backend.rs`）——理由同 [`dsl_map`]：
/// 默认构建不能把它编成无人使用的模块（`clippy -D warnings` 的 `dead_code`）。
#[cfg(feature = "dsl-backend")]
pub mod broadcast;

/// **f64 逐元素内核**（P1-4：`map_binary_f64`/`map_affine_f64` 的 LASX 向量版）。
///
/// **4 f64/lane（LASX 256 位）+ 4 路展开 + 尾部标量兜底**；`Add`/`Sub`/`Mul`/`Div`、
/// `a·x+b`（**故意不融 FMA**）全部与 `simd_dsl::kernels::scalar` 参照 **逐位**
/// （`Max`/`Min` 的 NaN 语义不同 ⇒ 仍委托标量，见 `dsl_backend.rs`）。
/// **挂在 `dsl-backend` feature 下**：唯一调用方是 `src/dsl_backend.rs`（同 feature 门控）；
/// 不门控会被 `dead_code` 判死（AGENTS #49 的教训）——**不要**用 `#[allow(dead_code)]` 掩盖。
#[cfg(feature = "dsl-backend")]
pub mod f64_map;

/// **B1b（M5 续）：组合原语 LASX 内核**——`map_unary_f32`（`Exp`/`Sqrt` 向量化、其余委托标量）
/// 与 `reduce_rows_map_f32`（`Sum`/`SumSq` 行内 8-lane + epilogue + map；未验证组合委托标量）。
///
/// **挂 `dsl-backend`**：唯一调用方是 `src/dsl_backend.rs`（同 feature 门控）；不门控会被
/// `clippy -D warnings` 的 `dead_code` 判死（AGENTS.md #49）——**不要**用 `#[allow(dead_code)]` 掩盖。
#[cfg(feature = "dsl-backend")]
pub mod prims;

/// **P4-b 融合内核**（`energy_dc` 单遍去均值能量、`psd_pair` 成对 epilogue）——
/// **纯标量、与便携参照逐位**（`to_bits()` 相等）；LASX FMA 版**未做**——FMA 一次舍入会破逐位，
/// 若要做必须给容差界与实测（当前不做，保持逐位）。
///
/// **门控已转为生产档**：`simd_dsl` 的 trait 已声明这两个原语、`dsl_backend.rs` 已有 LASX override
/// ⇒ 有生产调用方 ⇒ 只挂 `#[cfg(feature = "dsl-backend")]`（不再需要 `test` 门控，见 AGENTS.md #49）。
#[cfg(feature = "dsl-backend")]
pub mod fused;

/// **P5-b：跨步原语**（`copy_strided_f32` / `map_binary_strided_f32`，
/// 统一 `Strides{row,col}`）。**内层连续（`col==1`）⇒ memcpy 行拷贝**（P5-a 实测手写 8-lane 更慢）；
/// 跨步 ⇒ 标量按 `(r,c)`。空形状空操作、越界 ⇒ `Err`。
///
/// **门控**：`simd_dsl` 的 trait 已声明这 2 个方法、`dsl_backend.rs` 已有 2 个 override 调用本模块
/// ⇒ 有生产调用方 ⇒ 只挂 `#[cfg(feature = "dsl-backend")]`（不再需要 `test` 门控；否则触发
/// `clippy -D warnings` 的 `dead_code`，见 AGENTS.md #49）。
#[cfg(feature = "dsl-backend")]
pub mod strided;

pub mod gather;
/// **P5-a：gather / scatter 的 LASX 内核**（签名按 `simd_dsl/docs/structure.md §7` 的**解冻最终契约**：
/// `stride` 显式、`ids: &[u32]`、越界 ⇒ `Err`）。行拷贝：**memcpy 上限**（手写 8-lane 实测更慢，见 `dev.md §7.23`）。
///
/// **门控已转生产档**：`simd_dsl` trait 已声明这 3 个方法、`dsl_backend.rs` 已有 override ⇒ 有调用方 ⇒
/// 只挂 `#[cfg(feature = "dsl-backend")]`（不再需要 `test`，见 AGENTS.md #49）。
#[cfg(feature = "dsl-backend")]
pub mod gather_scatter;
pub mod gelu_erf;
pub mod gelu_quick;
pub mod gemv_i8;
pub mod j2_accel_batch;
pub mod layer_norm;
/// **P0-2 单趟 map+行和**（`Backend::map_rows_sum_f32` 的 LASX 向量实现）。
/// 唯一调用方是 `src/dsl_backend.rs`（同 feature 门控）⇒ 挂在 `dsl-backend` 下，避免默认构建
/// 编成"无人使用的模块"（AGENTS.md #49 的 `dead_code` 教训）。
#[cfg(feature = "dsl-backend")]
pub mod map_rows_sum;
pub mod mat3_mul_vec3_batch;
pub mod matmul;
/// **P5-b：NT-GEMM 内核**（`out[i*n+j] = Σ_t a[i*k+t]·b[j*k+t]`；`b` 行主序 ⇒ **不需要物化转置**）。
/// **逐位类**：严格按 `t` 顺序累加、无 FMA/无多累加器/无重排。
///
/// **门控已转生产档（阶段3 批1 / IS-1）**：`dsl_backend.rs` 已有 `matmul_bt_f32` override 调本模块
/// ⇒ **有生产调用方** ⇒ 只挂 `#[cfg(feature = "dsl-backend")]`（不再需要 `test`，见 AGENTS.md #49）。
#[cfg(feature = "dsl-backend")]
pub mod matmul_bt;
pub mod matmul_f64;
pub mod matmul_i8;
pub mod nn_math;
pub mod norm3_batch;
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
pub mod sum;
pub mod unitize3_batch;
pub mod vec3_add_scaled_batch;

#[cfg(test)]
pub(crate) mod testutil;
