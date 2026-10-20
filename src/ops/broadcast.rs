//! **广播/标量二元的 LASX 内核**（`Backend::map_binary_{rows,cols,scalar}_f32` 的 override 落点）。
//!
//! 语义与 `simd_dsl::kernels::scalar` 的默认实现**逐位一致**（同一组 `Add/Sub/Mul/Div` IEEE 运算，
//! 逐元素、无重排：行广播每行一次标量、SIMD 扫 `cols`；列广播每行 SIMD 连续 `w[c]`；标量整段 SIMD）：
//!
//! * [`map_binary_rows`]：`out[r·cols+c] = op(x[r·cols+c], w[r])`（`w.len() == rows`）；
//! * [`map_binary_cols`]：`out[r·cols+c] = op(x[r·cols+c], w[c])`（`w.len() == cols`）；
//! * [`map_binary_scalar`]：`out[i] = op(x[i], scalar)`。
//!
//! **op 覆盖**只有 [`dsl_map::BinOp`] 的 `Add/Sub/Mul/Div`；`Max/Min`（以及 `#[non_exhaustive]`
//! 的未来变体）由 `dsl_backend` 的 override **委托标量参照**——Rust `f32::max/min` 的 NaN 语义
//! （另一侧是数就返回它）≠ LASX `xvfmax/xvfmin` 的 IEEE 语义，不能向量化还声称逐位。
//!
//! **校验在调用方**：`dsl_backend` 的 override 先按默认实现同口径校验 `rows×cols` 溢出、
//! `x/out/w` 长度与空形状，再进本模块；这里只做"已校验切片"上的 SIMD 运算。

use crate::ops::dsl_map::{self, BinOp};

/// `out[r·cols+c] = op(x[r·cols+c], w[r])`（`w.len() == rows`；调用方已校验）。
#[inline]
pub(crate) fn map_binary_rows(
    x: &[f32],
    rows: usize,
    cols: usize,
    w: &[f32],
    out: &mut [f32],
    op: BinOp,
) {
    if rows == 0 || cols == 0 {
        return; // 空视图（调用方已校验长度/溢出）⇒ 无事可做
    }
    // 每行：`w[r]` 只读一次、splat 进 SIMD；行内连续扫 cols（`dsl_map` 的 4 路展开 + 8-lane 尾）。
    for (r, (x_row, out_row)) in x
        .chunks_exact(cols)
        .zip(out.chunks_exact_mut(cols))
        .enumerate()
    {
        dsl_map::map_binary_scalar(x_row, w[r], out_row, op);
    }
}

/// `out[r·cols+c] = op(x[r·cols+c], w[c])`（`w.len() == cols`；调用方已校验）。
#[inline]
pub(crate) fn map_binary_cols(
    x: &[f32],
    rows: usize,
    cols: usize,
    w: &[f32],
    out: &mut [f32],
    op: BinOp,
) {
    if rows == 0 || cols == 0 {
        return;
    }
    // 每行：`w` 是连续向量 ⇒ 直接走逐元素 `map_binary` 的 SIMD 核（无需重复实现）。
    x.chunks_exact(cols)
        .zip(out.chunks_exact_mut(cols))
        .for_each(|(x_row, out_row)| dsl_map::map_binary(x_row, w, out_row, op));
}

/// `out[i] = op(x[i], scalar)`（调用方已校验等长）。
#[inline]
pub(crate) fn map_binary_scalar(x: &[f32], scalar: f32, out: &mut [f32], op: BinOp) {
    dsl_map::map_binary_scalar(x, scalar, out, op);
}
