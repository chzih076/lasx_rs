//! **组合原语的内核实现（中性 API，无 DSL 依赖）**。
//!
//! 两个入口：
//!
//! * [`map_unary_exp`] / [`map_unary_sqrt`]：逐元素一元，**只向量化这两个**。
//!   `Sqrt` 走 `lasx_xvfsqrt_s`（IEEE 正确舍入 ⇒ 逐位）；`Exp` 走本仓 `nn_math::exp_vec`
//!   （容差类，实测界登记在 `docs/dev.md`）。其余一元变体**不在本仓实现**——由 DSL 适配器
//!   回退到 simd 侧标量（本库不猜语义）。
//! * [`reduce_rows_map`]：窄三元组融合（[`RowFuseSpec`]）。**`Sum`/`SumSq` + `Add/Sub/Mul/Div`**
//!   走向量路径；**不支持的组合返回 [`Error::BadValue`]**，由适配器回退（本库不静默换语义）。
//!
//! # 等价类（与纯化前的契约口径一致）
//!
//! * `map_unary_sqrt` **逐位**（同一 IEEE sqrt）；`map_unary_exp` **容差类**（LASX 多项式 vs libm）。
//! * `reduce_rows_map` 的**受支持组合**：行内 8-lane 累加 + 固定次序 hsum、epilogue/map 逐字照旧
//!   标量次序 ⇒ 与 DSL 侧标量参照 **`to_bits` 全等**（纯化前由 `tests/prims_override.rs` 钉住；
//!   该等价性现在归 DSL 侧 conformance/适配器测试）。
#![allow(clippy::undocumented_unsafe_blocks)]

use std::arch::loongarch64::*;

use crate::api::Error;
use crate::arch::lasx;
use crate::ops::nn_math::{exp_seq, exp_vec};
use crate::ops::types::{RowEpilogueKind, RowFuseSpec, RowMapKind, RowReduceKind};

/// 长度不符 ⇒ `Error::Shape`（字段口径同 `api` 层）。
#[inline]
fn len_err(op: &'static str, what: &'static str, expected: usize, got: usize) -> Error {
    Error::Shape {
        op,
        what,
        expected,
        got,
    }
}

/// 逐元素 `out = exp(x)`（LASX 向量近似；容差类）。
///
/// # Errors
/// `x.len() != out.len()` ⇒ [`Error::Shape`]。
pub fn map_unary_exp(x: &[f32], out: &mut [f32]) -> Result<(), Error> {
    expect_same_len("map_unary_exp", x, out)?;
    debug_assert_ne!(x.as_ptr(), out.as_ptr(), "map_unary_exp：out 不得与 x 重叠");
    // 主循环 8 lane；尾部用 `exp_seq`（与 `exp_vec` 同一串操作的标量版 ⇒ 尾块与向量块同一近似）。
    let (xc, xt) = x.as_chunks::<8>();
    let (oc, ot) = out.as_chunks_mut::<8>();
    for (xi, oi) in xc.iter().zip(oc.iter_mut()) {
        // SAFETY: `xi`/`oi` 都是 8 个 f32 的引用，load/store 的 8 lane 前提成立。
        unsafe {
            let v = lasx::load_f32x8(xi.as_ptr());
            lasx::store_f32x8(oi.as_mut_ptr(), exp_vec(v));
        }
    }
    for (o, v) in ot.iter_mut().zip(xt) {
        *o = exp_seq(*v);
    }
    Ok(())
}

/// 逐元素 `out = sqrt(x)`（LASX `xvfsqrt_s`，IEEE 正确舍入 ⇒ 逐位）。
///
/// # Errors
/// `x.len() != out.len()` ⇒ [`Error::Shape`]。
pub fn map_unary_sqrt(x: &[f32], out: &mut [f32]) -> Result<(), Error> {
    expect_same_len("map_unary_sqrt", x, out)?;
    debug_assert_ne!(
        x.as_ptr(),
        out.as_ptr(),
        "map_unary_sqrt：out 不得与 x 重叠"
    );
    let (xc, xt) = x.as_chunks::<8>();
    let (oc, ot) = out.as_chunks_mut::<8>();
    for (xi, oi) in xc.iter().zip(oc.iter_mut()) {
        // SAFETY: 同 `map_unary_exp`（8 lane 读写）。
        unsafe {
            let v = lasx::load_f32x8(xi.as_ptr());
            // `xvfsqrt_s` 为 IEEE 正确舍入平方根 ⇒ 与标量 `f32::sqrt` 逐位。
            lasx::store_f32x8(oi.as_mut_ptr(), lasx_xvfsqrt_s(v));
        }
    }
    for (o, v) in ot.iter_mut().zip(xt) {
        *o = v.sqrt();
    }
    Ok(())
}

#[inline]
fn expect_same_len(op: &'static str, x: &[f32], out: &[f32]) -> Result<(), Error> {
    if x.len() != out.len() {
        return Err(len_err(op, "out", x.len(), out.len()));
    }
    Ok(())
}

/// 窄三元组融合：行统计 + epilogue + 行内 map（[`RowFuseSpec`]）。
///
/// 支持 `reduce ∈ {Sum, SumSq}` 且 `map ∈ {Add, Sub, Mul, Div}` 的向量路径；
/// 其余组合（含任何未来变体）返回 [`Error::BadValue`]（**不静默降级**，由适配器回退）。
///
/// # Errors
/// * 长度/溢出不符 ⇒ [`Error::Shape`] / [`Error::Overflow`]；
/// * 不受支持的 `plan` ⇒ [`Error::BadValue`]（`what = "plan"`）。
pub fn reduce_rows_map(
    x: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    plan: RowFuseSpec,
) -> Result<(), Error> {
    let op = "reduce_rows_map_f32";
    let total = rows.checked_mul(cols).ok_or(Error::Overflow {
        op,
        what: "rows×cols",
    })?;
    if x.len() != total {
        return Err(len_err(op, "x", total, x.len()));
    }
    if out.len() != total {
        return Err(len_err(op, "out", total, out.len()));
    }
    debug_assert_ne!(
        x.as_ptr(),
        out.as_ptr(),
        "reduce_rows_map_f32：out 不得与 x 重叠"
    );
    if rows == 0 || cols == 0 {
        return Ok(());
    }

    let vectorizable = matches!(plan.reduce, RowReduceKind::Sum | RowReduceKind::SumSq)
        && matches!(
            plan.map,
            RowMapKind::Add | RowMapKind::Sub | RowMapKind::Mul | RowMapKind::Div
        );
    if !vectorizable {
        return Err(Error::BadValue {
            op,
            what: "plan",
            requirement: "受支持的组合（reduce ∈ {Sum,SumSq} 且 map ∈ {Add,Sub,Mul,Div}）；其余由适配器回退",
        });
    }

    // **自动并行分派（P0-3 推广）**：`reduce_rows_map` 的每一行都是**独立**的
    // （行内统计量 → epilogue → 行内 map），并行只改"哪一行先算"，**任何元素的运算次序不变**
    // ⇒ **逐位类**（与 `rms/layer/softmax` 的行块切分同一论证）。
    // 阈值由纯化前 `examples/row_par_ab.rs` 实测定；阈值以下原地串行。
    let work = (rows as u128) * (cols as u128);
    let pool = crate::pool::global();
    if work >= PAR_MIN_ROWS_MAP_ELEMS && pool.threads() > 1 && rows >= 2 {
        crate::parallel::reduce_rows_map(pool, x, rows, cols, out, plan);
        return Ok(());
    }
    reduce_rows_map_rows(x, out, cols, rows, plan);
    Ok(())
}

/// `reduce_rows_map_f32` 的**行区间主体**：对 `rows_local` 行（`x`/`out` 都是该区间的行主序切片）
/// 执行行内统计 + epilogue + 行内 map。
///
/// 串行主体（整段）与并行分块（`parallel::reduce_rows_map` 的每块）**共用本函数** ⇒ 两条路径
/// 对同一行执行完全相同的指令序列（**逐位**）。
pub fn reduce_rows_map_rows(
    x: &[f32],
    out: &mut [f32],
    cols: usize,
    rows_local: usize,
    plan: RowFuseSpec,
) {
    let square = matches!(plan.reduce, RowReduceKind::SumSq);
    let inv_cols = 1.0f32 / cols as f32;
    let map = plan.map;
    for r in 0..rows_local {
        let row = &x[r * cols..r * cols + cols];
        let stat = row_stat_vec(row, square);
        let k = epilogue(stat, inv_cols, plan.eps, plan.epilogue);
        let dst = &mut out[r * cols..r * cols + cols];
        // map 主体 8 lane + 尾部标量（与标量参照逐位：都是 IEEE 单条运算）。
        let (rc, rt) = row.as_chunks::<8>();
        let (dc, dt) = dst.as_chunks_mut::<8>();
        let vk = lasx::splat_f32(k);
        for (ri, di) in rc.iter().zip(dc.iter_mut()) {
            // SAFETY: 两侧都是 8 个 f32 的引用。
            unsafe {
                let v = lasx::load_f32x8(ri.as_ptr());
                let res = match map {
                    RowMapKind::Add => lasx_xvfadd_s(v, vk),
                    RowMapKind::Sub => lasx_xvfsub_s(v, vk),
                    RowMapKind::Mul => lasx_xvfmul_s(v, vk),
                    RowMapKind::Div => lasx_xvfdiv_s(v, vk),
                };
                lasx::store_f32x8(di.as_mut_ptr(), res);
            }
        }
        for (d, v) in dt.iter_mut().zip(rt) {
            *d = match map {
                RowMapKind::Add => v + k,
                RowMapKind::Sub => v - k,
                RowMapKind::Mul => v * k,
                RowMapKind::Div => v / k,
            };
        }
    }
}

/// `reduce_rows_map_f32` 的自动并行阈值（元素数；按纯化前 `examples/row_par_ab.rs` 校准）。
///
/// **实测**（12 线程、loadavg≈5、同进程 1t vs auto、6 对丢首对取中位；判据 1t/auto ≥1.10）：
/// 8K **2.72×**、16K 4.62×、32K 5.73×、64K 8.27×、128K 7.54×、256K 8.32×、512K 7.71×、
/// 786K 7.49×、4.2M 2.11×、16.7M 2.29×；**8K 以下走串行**（不触发分派 ⇒ 与 1t 同一路径）。
/// 阈值取 8K 与 `rms_norm`/`softmax`/`layer_norm`（P0-3 同口径）一致；等价类 = **逐位**
/// （并行只改行序，行内次序不变）。
pub const PAR_MIN_ROWS_MAP_ELEMS: u128 = 1 << 13; // 8K 元素

/// 行统计量：**与标量多累加器版同序**——每个 8-lane chunk 加到对应 lane 的向量累加器，
/// 末尾按 `acc[0]+acc[1]+…+acc[7]`（自 0 起）顺序归约，再按 c 递增补尾。
fn row_stat_vec(row: &[f32], square: bool) -> f32 {
    let (chunks, tail) = row.as_chunks::<8>();
    let mut acc = lasx::zero_f32x8();
    // SAFETY: 每个 chunk 恰有 8 个 f32；全程寄存器操作。
    unsafe {
        for c in chunks {
            let v = lasx::load_f32x8(c.as_ptr());
            let term = if square { lasx_xvfmul_s(v, v) } else { v };
            acc = lasx_xvfadd_s(acc, term);
        }
    }
    let mut buf = [0.0f32; 8];
    // SAFETY: `buf` 有 8 个元素，store 8 lane 合法。
    unsafe { lasx::store_f32x8(buf.as_mut_ptr(), acc) };
    let mut s = 0.0f32;
    for &a in &buf {
        s += a; // 与标量 `fold(0.0, |s, v| s + v)` 同序
    }
    for &v in tail {
        s += if square { v * v } else { v };
    }
    s
}

/// 每行一次的标量收尾——**逐字复制标量参照的运算次序**（含乘法口径的 `Mean`）。
#[inline]
fn epilogue(stat: f32, inv_cols: f32, eps: f32, kind: RowEpilogueKind) -> f32 {
    match kind {
        RowEpilogueKind::Raw => stat,
        RowEpilogueKind::Mean => stat * inv_cols,
        RowEpilogueKind::MeanPlusEps => stat * inv_cols + eps,
        RowEpilogueKind::SqrtMeanPlusEps => (stat * inv_cols + eps).sqrt(),
        RowEpilogueKind::InvSqrtMeanPlusEps => 1.0f32 / (stat * inv_cols + eps).sqrt(),
        RowEpilogueKind::Inv => 1.0f32 / stat,
    }
}
