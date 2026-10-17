//! `dsl_map` —— 给 `simd_dsl::Backend` 的**逐元素与轴归约族**提供 LASX 内核（R19-E）。
//!
//! **为什么这些能逐位**（与 `simd_dsl::kernels::scalar` 的参照实现对比）：
//! * `map_binary_*`：逐元素 `f32` 运算、**每个元素独立** ⇒ 向量化不改变任何数；
//! * `map_affine`：契约是 `a·x + b`（**两步**）⇒ 这里**故意不用 FMA**（`xvfmul` 再 `xvfadd`），
//!   否则舍入与标量参照不同、就不是逐位类了；
//! * `reduce_sum_cols` / `reduce_weighted_sum_cols`：参照是"每列一个 `f32` 累加器、**按行序**累加"
//!   ⇒ 这里**每条 lane 只负责一列**、也按行序遍历 ⇒ **每列累加次序与参照完全相同** ⇒ 逐位。
//!   （对照：`reduce_sum_f32` 的参照本身是多累加器 ⇒ 那条**只能**容差类，见 `dsl_backend` 的说明。）
//!
//! **本文件不依赖 `simd_dsl`**（本仓内核保持独立）⇒ 用一个本地 `BinOp` 枚举做 op 分派。
//! **注意**：`Max`/`Min` 的 NaN 语义（Rust `f32::max`：另一侧是数就返回它）与 LASX `xvfmax` 的
//! IEEE 语义**不同** ⇒ 这两条**不在本文件实现**，由 `LasxBackend` 委托标量参照（见 `dsl_backend.rs`）。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（分层策略：内核模块在函数级统一写 SAFETY，不逐块重复）。
#![allow(clippy::undocumented_unsafe_blocks)]
use crate::arch::lasx;
use std::arch::loongarch64::*;

/// 本地产物：逐元素二元运算种类（与 `simd_dsl::BinaryOp` 的 `Add/Sub/Mul/Div` 一一对应）。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum BinOp {
    /// `x + y`
    Add,
    /// `x − y`
    Sub,
    /// `x · y`
    Mul,
    /// `x / y`
    Div,
}

#[inline]
fn bin_scalar(op: BinOp, a: f32, b: f32) -> f32 {
    match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => a / b,
    }
}

/// `outᵢ = xᵢ op yᵢ`（三切片等长；调用方已校验）。
///
/// # Safety
/// 调用方保证 `x.len() == y.len() == out.len()` 且 `out` 不与 `x`/`y` 重叠。
#[inline]
pub(crate) fn map_binary(x: &[f32], y: &[f32], out: &mut [f32], op: BinOp) {
    let n = x.len();
    let mut i = 0;
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n` ⇒ 三次 8 元素访存都在切片内；非别名由调用方保证。
        unsafe {
            let a = lasx::load_f32x8(x.as_ptr().add(i));
            let b = lasx::load_f32x8(y.as_ptr().add(i));
            let r = match op {
                BinOp::Add => lasx_xvfadd_s(a, b),
                BinOp::Sub => lasx_xvfsub_s(a, b),
                BinOp::Mul => lasx_xvfmul_s(a, b),
                BinOp::Div => lasx_xvfdiv_s(a, b),
            };
            lasx::store_f32x8(out.as_mut_ptr().add(i), r);
        }
        i += 8;
    }
    for j in i..n {
        out[j] = bin_scalar(op, x[j], y[j]);
    }
}

/// `outᵢ = a·xᵢ + b`（**两步运算**，不融 FMA ⇒ 与标量参照逐位）。
///
/// # Safety
/// 调用方保证 `x.len() == out.len()` 且 `out` 不与 `x` 重叠。
#[inline]
pub(crate) fn map_affine(a: f32, b: f32, x: &[f32], out: &mut [f32]) {
    let n = x.len();
    let va = lasx::splat_f32(a);
    let vb = lasx::splat_f32(b);
    let mut i = 0;
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n` ⇒ 访存落在切片内；非别名由调用方保证。
        unsafe {
            let v = lasx::load_f32x8(x.as_ptr().add(i));
            let r = lasx_xvfadd_s(lasx_xvfmul_s(va, v), vb);
            lasx::store_f32x8(out.as_mut_ptr().add(i), r);
        }
        i += 8;
    }
    for j in i..n {
        out[j] = a * x[j] + b;
    }
}

/// `out[c] = Σ_r x[r·cols + c]`（**每列一条 lane、按行序累加** ⇒ 与标量参照逐位）。
///
/// # Safety
/// 调用方保证 `x.len() == rows·cols`、`out.len() == cols`、`out` 不与 `x` 重叠。
#[inline]
pub(crate) fn reduce_sum_cols(x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    let mut c = 0;
    while c + 8 <= cols {
        let mut acc = lasx::zero_f32x8();
        for r in 0..rows {
            let off = r * cols + c;
            // SAFETY: `c + 8 <= cols` 且 `r < rows` ⇒ `off + 8 <= rows*cols`，落在 `x` 内。
            unsafe {
                acc = lasx_xvfadd_s(acc, lasx::load_f32x8(x.as_ptr().add(off)));
            }
        }
        // SAFETY: `c + 8 <= cols == out.len()`。
        unsafe { lasx::store_f32x8(out.as_mut_ptr().add(c), acc) };
        c += 8;
    }
    // 尾部列：与参照同形（f32 单累加器、按行序）
    for cc in c..cols {
        let mut acc = 0.0f32;
        for r in 0..rows {
            acc += x[r * cols + cc];
        }
        out[cc] = acc;
    }
}

/// `out[c] = Σ_r x[r·cols + c]·w[r]`（`out + (x·w)`：**先乘后加**，与标量参照逐位）。
///
/// # Safety
/// 调用方保证 `x.len() == rows·cols`、`w.len() >= rows`、`out.len() == cols`、`out` 不与 `x` 重叠。
#[inline]
pub(crate) fn reduce_weighted_sum_cols(
    x: &[f32],
    w: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    let mut c = 0;
    while c + 8 <= cols {
        let mut acc = lasx::zero_f32x8();
        for (r, wv) in w.iter().enumerate().take(rows) {
            let off = r * cols + c;
            // SAFETY: 同 `reduce_sum_cols`；`*wv` 是标量广播。
            unsafe {
                let v = lasx::load_f32x8(x.as_ptr().add(off));
                acc = lasx_xvfadd_s(acc, lasx_xvfmul_s(v, lasx::splat_f32(*wv)));
            }
        }
        // SAFETY: `c + 8 <= cols == out.len()`。
        unsafe { lasx::store_f32x8(out.as_mut_ptr().add(c), acc) };
        c += 8;
    }
    for cc in c..cols {
        let mut acc = 0.0f32;
        for (r, wv) in w.iter().enumerate().take(rows) {
            acc += x[r * cols + cc] * *wv;
        }
        out[cc] = acc;
    }
}

/// `Σ x[i]`：直接复用本仓既有的 LASX 归约（`ops::sum`）。
#[inline]
pub(crate) fn reduce_sum(x: &[f32]) -> f32 {
    crate::ops::sum::sum(x)
}
