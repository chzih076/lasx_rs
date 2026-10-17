//! **融合原语 override 的"就绪骨架"**（阶段 3 批 2 收尾 / 等 `simd_dsl` 契约合入）。
//!
//! 状态：**契约分支 `fused-contract` 尚未领先于 master**（`master..fused-contract` 为空）⇒ 方法签名
//! 未定；本文件的 `impl` 块用 **`#[cfg(any())]` 关闭**（否则当前 pin `2214b57` 上编译不过）。
//! **启用步骤**（拿到 Lead 给的新 SHA 后）：
//! 1) `Cargo.toml` 两行 rev 用 `edit` + `replace_all`：`2214b57…` → 新 SHA；`cargo update -p simd_dsl -p ops_dsl`；`Cargo.lock` 同提交；
//! 2) 去掉本文件 `#[cfg(any())]` 一行；按契约**核对签名**（若与骨架不同，以契约为准并改这里）；
//! 3) 在 `examples/fused_ab.rs` 去掉 `#[cfg(any())]` 并加 `[[example]] required-features=["dsl-backend"]`；
//! 4) 跑 `fused_ab`（判据 ≥0.98）+ 门禁。
//!
//! # 映射表（DSL 方法 → LASX 内核）+ 等价类（界来自批 2 A/B 实测）
//!
//! | `Backend` 方法 | LASX 实现 | 等价类 | 实测界 / 备注 |
//! |---|---|---|---|
//! | `softmax_rows_f32(x, rows, cols, out)` | `api::softmax_rows_into`（若无则 `api::softmax_rows(…, scale=1.0)` 后拷贝） | **容差** | rel ≤ 2.79e-9（批2 A/B） |
//! | `rms_norm_rows_f32(x, rows, cols, eps, out)` | `api::rms_norm_into` | **容差** | rel ≤ 1.83e-7 |
//! | `layer_norm_rows_f32(x, rows, cols, eps, out)` | `api::layer_norm_into` | **容差** | rel ≤ 2.08e-7 |
//! | `gelu_quick_f32(x, out)` | `api::gelu_quick_into` | **容差** | rel ≤ **5.26e-3**（LASX tanh 近似 vs σ(2z) 恒等式，界显著大于 1e-5，登记时别写错） |
//! | `rope_apply_f32(x, rows, cols, n_dims, cos, sin, out)` | `api::rope_into(…, RopeMode::NeoX, out)` | **容差** | rel ≤ 6.92e-8；`NeoX` = half-split（组合版同布局）；`GptJ` 不在本原语范围 |
//! | `axpy_f32(alpha, x, y)`（`y: &mut`） | `api::axpy(alpha, x, y)` | **容差** | rel ≤ 8.27e-8；**别名**：`x`/`y` 重叠未定义（借用检查器 + `debug_assert` 基地址） |
//! | `UnaryOp::Abs`（`map_unary_f32` 变体） | 无专用内核 ⇒ 走 `affine!(−1·x)`+`max(x,−x)` 组合或标量默认 | 逐位 | 若走组合 = 2 趟；建议先如实报"无内核"（能力探测不声明 Abs） |
//!
//! **判据**：override 后 DSL 路径 vs `api::*` 应 **≥0.98**（融合核同源，预期 ≈1.0）；
//! 各 override 必须按上表在 `precision_claim`/文档里声明**容差类 + 界**（不能默认"同一原语"同类）。

// 与下面的 `impl` 一起被 `#[cfg(any())]` 关掉（否则默认/feature 两档都会报 unused import）。
#[cfg(any())]
use crate::dsl_backend::LasxBackend;
#[cfg(any())]
use simd_dsl::{Backend, BackendError};

// ⚠️ 契约合入后删掉下面这一行（`#[cfg(any())]` 使整个 impl 不参与编译）。
#[cfg(any())]
impl Backend for LasxBackend {
    fn softmax_rows_f32(
        &self,
        x: &[f32],
        rows: usize,
        cols: usize,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        // 契约核对点：参数顺序/是否带 `scale`/`mask`。LASX 的行 softmax 无 mask 时走快档。
        crate::api::softmax_rows_into(x, None, rows, cols, 1.0, out).map_err(|e| {
            BackendError::Unsupported {
                op: "softmax_rows_f32",
                cap: simd_dsl::Capability::Reduction,
            }
        })
    }

    fn rms_norm_rows_f32(
        &self,
        x: &[f32],
        rows: usize,
        cols: usize,
        eps: f32,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        crate::api::rms_norm_into(x, None, rows, cols, eps, out).map_err(|_| {
            BackendError::Unsupported {
                op: "rms_norm_rows_f32",
                cap: simd_dsl::Capability::Reduction,
            }
        })
    }

    fn layer_norm_rows_f32(
        &self,
        x: &[f32],
        rows: usize,
        cols: usize,
        eps: f32,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        crate::api::layer_norm_into(x, None, None, rows, cols, eps, out).map_err(|_| {
            BackendError::Unsupported {
                op: "layer_norm_rows_f32",
                cap: simd_dsl::Capability::Reduction,
            }
        })
    }

    fn gelu_quick_f32(&self, x: &[f32], out: &mut [f32]) -> Result<(), BackendError> {
        // LASX 是融合 tanh 近似核 ⇒ 与"σ(2z) 恒等式"组合版**容差 5.26e-3**（见映射表）。
        crate::api::gelu_quick_into(x, out).map_err(|_| BackendError::Unsupported {
            op: "gelu_quick_f32",
            cap: simd_dsl::Capability::Reduction,
        })
    }

    fn rope_apply_f32(
        &self,
        x: &[f32],
        rows: usize,
        cols: usize,
        n_dims: usize,
        cos: &[f32],
        sin: &[f32],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        // `NeoX` = half-split（与 ops_dsl 组合版同布局）；`GptJ` 留作后续。
        crate::api::rope_into(
            x,
            cos,
            sin,
            rows,
            cols,
            n_dims,
            crate::api::RopeMode::NeoX,
            out,
        )
        .map_err(|_| BackendError::Unsupported {
            op: "rope_apply_f32",
            cap: simd_dsl::Capability::Reduction,
        })
    }

    fn axpy_f32(&self, alpha: f32, x: &[f32], y: &mut [f32]) -> Result<(), BackendError> {
        // **别名规则**：`x` 与 `y` 重叠 ⇒ 未定义（安全 Rust 由借用检查器守；这里只 `debug_assert` 基地址）。
        debug_assert_ne!(x.as_ptr(), y.as_ptr(), "axpy_f32：x/y 不得重叠");
        crate::api::axpy(alpha, x, y).map_err(|_| BackendError::Unsupported {
            op: "axpy_f32",
            cap: simd_dsl::Capability::Reduction,
        })
    }
}
