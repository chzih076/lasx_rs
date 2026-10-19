//! **B1b（M5 续）：组合原语 override 的 LASX 向量内核**。
//!
//! 两个入口（`dsl-backend` feature 下唯一调用方是 `src/dsl_backend.rs`）：
//!
//! * [`map_unary`]：逐元素一元。**`Exp` 走 `nn_math::exp_vec`（8 lane 多项式）、`Sqrt` 走
//!   `lasx_xvfsqrt_s`（IEEE 正确舍入 ⇒ 逐位）；其余变体整片委托
//!   `simd_dsl::kernels::scalar::map_unary_f32`**（逐位不变，不做未经验证的向量化）。
//! * [`reduce_rows_map`]：窄三元组融合（`RowFuse`）。**`Sum`/`SumSq` 行内 8-lane 累加 + 同序
//!   hsum，`Add/Sub/Mul/Div` 的 map 用向量写回**；`Max` 归约与 `Max/Min` map 等**未验证的组合
//!   整片委托标量参照**（契约完整性优先，`NaN`/`±0` 语义不为优化让路）。
//!
//! # 等价类（实测，见 `tests/prims_override.rs`）
//!
//! * `map_unary`：`Sqrt` **逐位**（同一 IEEE sqrt）；`Exp` **容差类**（LASX 多项式 vs libm，
//!   实测 rel 界写在测试消息里/`docs/dev.md §7.52`）；其余变体委托标量 ⇒ **逐位**。
//! * `reduce_rows_map`：**`Sum`/`SumSq` 的累加次序与标量多累加器版逐 lane 同序**（每个 chunk 加到
//!   对应 lane，末尾按 `acc[0]+…+acc[7]` 顺序 hsum、再逐元素补尾）⇒ 与
//!   `kernels::scalar::reduce_rows_map_f32` **逐位**；`epilogue`/`map` 的标量运算次序逐字复制 ⇒
//!   整个支持组合 **`to_bits` 全等**。委托组合自然逐位。
//!
//! **门控**：挂 `#[cfg(feature = "dsl-backend")]`（无 feature 时 `dsl_backend.rs` 不编译 ⇒ 本模块
//! 无人使用会被 `clippy -D warnings` 的 `dead_code` 判死，见 AGENTS.md #49；**不要**用
//! `#[allow(dead_code)]` 掩盖）。
#![allow(clippy::undocumented_unsafe_blocks)]

use std::arch::loongarch64::*;

use simd_dsl::kernels::scalar;
use simd_dsl::{BackendError, BinaryOp, RowEpilogue, RowFuse, RowReduce, UnaryOp};

use crate::arch::lasx;
use crate::ops::nn_math::{exp_seq, exp_vec};

/// 长度不符 ⇒ 与 portable 默认同错误码同字段。
#[inline]
fn len_err(op: &'static str, what: &'static str, expected: usize, got: usize) -> BackendError {
    BackendError::LengthMismatch {
        op,
        what,
        expected,
        got,
    }
}

/// 逐元素一元运算（`Backend::map_unary_f32` 的 LASX 实现）。
///
/// `Exp`/`Sqrt` 向量化；其余变体（`Sin/Cos/Ln/Erf/Sigmoid/Rcp/Abs`）**整片委托**
/// `scalar::map_unary_f32`——这是**有意**的：它们的 libm/近似语义没有 LASX 逐位对应物，
/// 自造近似会改变等价类，而本函数的第一职责是"让组合路径不再走标量"时**不破坏契约**。
pub(crate) fn map_unary(x: &[f32], out: &mut [f32], op: UnaryOp) -> Result<(), BackendError> {
    match op {
        UnaryOp::Exp => {
            expect_same_len(x, out)?;
            debug_assert_ne!(x.as_ptr(), out.as_ptr(), "map_unary_f32：out 不得与 x 重叠");
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
        UnaryOp::Sqrt => {
            expect_same_len(x, out)?;
            debug_assert_ne!(x.as_ptr(), out.as_ptr(), "map_unary_f32：out 不得与 x 重叠");
            let (xc, xt) = x.as_chunks::<8>();
            let (oc, ot) = out.as_chunks_mut::<8>();
            for (xi, oi) in xc.iter().zip(oc.iter_mut()) {
                // SAFETY: 同 `Exp` 分支（8 lane 读写）。
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
        _ => scalar::map_unary_f32(x, out, op),
    }
}

#[inline]
fn expect_same_len(x: &[f32], out: &[f32]) -> Result<(), BackendError> {
    if x.len() != out.len() {
        return Err(len_err("map_unary_f32", "out", x.len(), out.len()));
    }
    Ok(())
}

/// 窄三元组融合（`Backend::reduce_rows_map_f32` 的 LASX 实现）。
///
/// 支持 `reduce ∈ {Sum, SumSq}` 且 `map ∈ {Add, Sub, Mul, Div}` 的向量路径；
/// **`Max` 归约与 `Max/Min` map 整片委托标量**（见模块头"等价类"）。
pub(crate) fn reduce_rows_map(
    x: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    plan: RowFuse,
) -> Result<(), BackendError> {
    let op = "reduce_rows_map_f32";
    let total = rows.checked_mul(cols).ok_or(BackendError::ShapeOverflow {
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

    let vectorizable = matches!(plan.reduce, RowReduce::Sum | RowReduce::SumSq)
        && matches!(
            plan.map,
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div
        );
    if !vectorizable {
        // 未验证的组合（`Max` 的 `f32::max` NaN/±0 语义、`Max/Min` map）⇒ 契约优先，整片委托。
        return scalar::reduce_rows_map_f32(x, rows, cols, out, plan);
    }

    let square = matches!(plan.reduce, RowReduce::SumSq);
    let inv_cols = 1.0f32 / cols as f32;
    let map = plan.map;
    for r in 0..rows {
        let row = &x[r * cols..r * cols + cols];
        let stat = row_stat_vec(row, square);
        let k = epilogue(stat, inv_cols, plan.eps, plan.epilogue);
        let dst = &mut out[r * cols..r * cols + cols];
        // map 主体 8 lane + 尾部标量（与标量 `apply_scalar_op` 逐位：都是 IEEE 单条运算）。
        let (rc, rt) = row.as_chunks::<8>();
        let (dc, dt) = dst.as_chunks_mut::<8>();
        let vk = lasx::splat_f32(k);
        for (ri, di) in rc.iter().zip(dc.iter_mut()) {
            // SAFETY: 两侧都是 8 个 f32 的引用。
            unsafe {
                let v = lasx::load_f32x8(ri.as_ptr());
                let res = match map {
                    BinaryOp::Add => lasx_xvfadd_s(v, vk),
                    BinaryOp::Sub => lasx_xvfsub_s(v, vk),
                    BinaryOp::Mul => lasx_xvfmul_s(v, vk),
                    BinaryOp::Div => lasx_xvfdiv_s(v, vk),
                    // 上面 `vectorizable` 已排除；这里给出保持 `match` 穷尽的安全分支。
                    _ => unreachable!("vectorizable 已限制 map 集合"),
                };
                lasx::store_f32x8(di.as_mut_ptr(), res);
            }
        }
        for (d, v) in dt.iter_mut().zip(rt) {
            *d = match map {
                BinaryOp::Add => v + k,
                BinaryOp::Sub => v - k,
                BinaryOp::Mul => v * k,
                BinaryOp::Div => v / k,
                _ => unreachable!("vectorizable 已限制 map 集合"),
            };
        }
    }
    Ok(())
}

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
fn epilogue(stat: f32, inv_cols: f32, eps: f32, kind: RowEpilogue) -> f32 {
    match kind {
        RowEpilogue::Raw => stat,
        RowEpilogue::Mean => stat * inv_cols,
        RowEpilogue::MeanPlusEps => stat * inv_cols + eps,
        RowEpilogue::SqrtMeanPlusEps => (stat * inv_cols + eps).sqrt(),
        RowEpilogue::InvSqrtMeanPlusEps => 1.0f32 / (stat * inv_cols + eps).sqrt(),
        RowEpilogue::Inv => 1.0f32 / stat,
    }
}
