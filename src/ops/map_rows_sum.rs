//! **P0-2：单趟 map + 行和**（`Backend::map_rows_sum_f32` 的 LASX 实现）。
//!
//! 一次物理扫描同时做两件事：把 `exp(·)`（或 `exp(· − shift[r])`）写进 `out_mapped`，
//! 并用 **per-lane 累加器** 求行和写进 `out_rowsum`。相对 portable 的"map_unary + reduce_sum_rows"
//! 两遍组合，省一遍全长读写（契约里的 "单趟融合归约"）。
//!
//! # 等价类（**如实**，`docs/dev.md` 登记用）
//! **容差**，两个来源：
//! 1. **映射值**：这里用本仓 `exp_vec`（与 `lasx_softmax_rows` 同一串多项式，LASX 内部一致），
//!    portable 用 `f32::exp`（libm）⇒ 两者**不是逐位**（实测界见 `examples/map_rowsum_ab.rs`）；
//! 2. **行和**：per-lane 累加 + 固定次序水平求和，portable 是**单累加器顺序累加** ⇒ 次序不同。
//!
//! 契约方法文档本身也声明"**语义/等价类：容差**"，两者一致。

#![allow(clippy::undocumented_unsafe_blocks)]

use crate::arch::lasx;
use crate::ops::nn_math::exp_vec;
use crate::ops::types::ReduceMapKind;
use std::arch::loongarch64::*;

/// 8 lane 水平求和（与 `ops::softmax_rows::hsum` **同一固定折叠次序**，保证 LASX 内部一致）。
#[inline]
fn hsum(v: lasx::F32x8) -> f32 {
    let mut buf = [0f32; 8];
    // SAFETY: `buf` 有 8 个元素。
    unsafe { lasx::store_f32x8(buf.as_mut_ptr(), v) };
    ((buf[0] + buf[1]) + (buf[2] + buf[3])) + ((buf[4] + buf[5]) + (buf[6] + buf[7]))
}

/// `out_mapped[r·cols+c] = f(x[r·cols+c], shift[r])`、`out_rowsum[r] = Σ_c out_mapped[…]`。
///
/// # Panics（调用方适配器已校验，这里是 debug 断言）
/// `x.len() == rows·cols`、`out_mapped.len() == rows·cols`、`out_rowsum.len() == rows`；
/// `ExpShiftedRow` 要求 `shift.len() == rows`；`Exp` 要求 `shift` 为空。
pub fn map_rows_sum_f32(
    x: &[f32],
    rows: usize,
    cols: usize,
    shift: &[f32],
    map: ReduceMapKind,
    out_mapped: &mut [f32],
    out_rowsum: &mut [f32],
) {
    debug_assert_eq!(x.len(), rows * cols);
    debug_assert_eq!(out_mapped.len(), rows * cols);
    debug_assert_eq!(out_rowsum.len(), rows);
    debug_assert!(shift.is_empty() || shift.len() == rows);

    let shifted = matches!(map, ReduceMapKind::ExpShiftedRow);
    for r in 0..rows {
        let x_row = &x[r * cols..(r + 1) * cols];
        let out_row = &mut out_mapped[r * cols..(r + 1) * cols];
        let vs = if shifted {
            lasx::splat_f32(shift[r])
        } else {
            lasx::zero_f32x8()
        };
        let mut acc = lasx::zero_f32x8();
        let mut j = 0;
        while j + 8 <= cols {
            // SAFETY: `j + 8 ≤ cols`，落在本行内。
            let e = unsafe {
                let vx = lasx::load_f32x8(x_row.as_ptr().add(j));
                let d = if shifted { lasx_xvfsub_s(vx, vs) } else { vx };
                exp_vec(d)
            };
            acc = unsafe { lasx_xvfadd_s(acc, e) };
            // SAFETY: 同上，写回本行 `[j, j+8)`。
            unsafe { lasx::store_f32x8(out_row.as_mut_ptr().add(j), e) };
            j += 8;
        }
        if j < cols {
            // 尾部：装进 8 宽向量，填充 lane 用 `−1e30` ⇒ `exp` 夹到下界给出**精确 0**，
            // 因此不会污染行和；映射值只拷回有效 lane。
            let mut buf = [-1e30f32; 8];
            buf[..cols - j].copy_from_slice(&x_row[j..cols]);
            // SAFETY: `buf` 有 8 个元素。
            let e = unsafe {
                let vx = lasx::load_f32x8(buf.as_ptr());
                let d = if shifted { lasx_xvfsub_s(vx, vs) } else { vx };
                exp_vec(d)
            };
            acc = unsafe { lasx_xvfadd_s(acc, e) };
            let mut ob = [0f32; 8];
            // SAFETY: `ob` 有 8 个元素；只拷回有效 lane。
            unsafe { lasx::store_f32x8(ob.as_mut_ptr(), e) };
            out_row[j..cols].copy_from_slice(&ob[..cols - j]);
        }
        out_rowsum[r] = hsum(acc);
    }
}
