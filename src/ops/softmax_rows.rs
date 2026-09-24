//! `lasx_softmax_rows` —— 行内 softmax（`scale·x + mask` → 减 max → exp → 归一）。
//!
//! **LASX-only**：没有降级分支，无 LASX 的 CPU 上会执行 LASX 指令（见手册 Caveats）。
//!
//! # 位精确构造（设计见 `docs/dev.md` §20.1）
//!
//! `exp` 没有硬件指令，只能用多项式展开；而"向量主体 + 标量列尾"天然会出现两种结合次序
//! （向量水平归约 vs 逐个累加）。本实现把它拆成"**全部 lane-wise 操作 + 最后一次固定
//! 次序的水平归约**"，于是下面三条都逐位成立：
//!
//! 1. **列尾与整列同序**：不足 8 个元素时，把尾元素装进一个 8 宽向量、**其余 lane 填
//!    `−1e30`**（经下面的下界夹取后落到 `exp` 的极小值），走的还是同一串 lane-wise 操作
//!    ——列尾不是一条单独的标量路径，因此不可能与主体不一致。
//! 2. **行内求和是同序的**：保留 **8 个 per-lane 累加器**（lane `l` 累加下标 `j ≡ l (mod 8)`
//!    的元素），最后按**固定次序的两两归约**折叠——这与标量模拟逐位一致（测试里验）。
//! 3. **`exp` 的 op 序列标量/向量共用同一串**（现在独立成 [`crate::ops::nn_math`]，
//!    `silu`/`gelu` 用的是同一份代码，不可能"各算子精度不同"）。
//!
//! # 数值行为
//!
//! - 行内先减 max（数值必需，否则 `exp` 溢出），再 `exp`；
//! - `exp` 是 [`crate::ops::nn_math`] 里那一串（`exp2` 的 6 次多项式，端到端实测相对误差
//!   `|x| ≤ 20` 时 ≲ 1.1e-6、`|x| ≤ 87` 时 ≲ 3.9e-6，见 `docs/ops.md` §2.6）；
//! - 归一化是 `e · (1/Σ)`（**一次除法 + 乘法**），不是逐元素除法——契约记在
//!   `docs/ops.md`：参考实现必须同样写 `e * (1.0 / sum)` 才逐位一致；
//! - 输入夹取（`≥ −104.0`）由 `exp` **自己在入口做**：这既让"填充 lane"得到精确的 0
//!   而不是 NaN，也挡住 `|x|` 极大时 magic 数技巧失效、浮点转整型越界的情形。
//!
//! # 可选加性 mask
//!
//! `mask` 与 `x` 同形状、**加性**（attention 里那种 `+ mask`）。传空切片表示无 mask：
//! 此时仍然走 `fma(x, scale, 0.0)`，与 `x · scale` 的差别只可能在 `±0.0`
//! （`-0.0 + 0.0 = +0.0`），而 `exp` 对两者取值相同，故不影响任何输出。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（策略见 `docs/dev.md` §17）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX/LSX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]
use crate::arch::lasx;
use crate::ops::nn_math::exp_vec;
use std::arch::loongarch64::*;

/// 取 8 lane 的最大值（固定次序的多路折叠；`max` 精确且次序无关）。
#[inline]
fn hmax(v: lasx::F32x8) -> f32 {
    let mut buf = [0f32; 8];
    // SAFETY: `buf` 有 8 个元素。
    unsafe { lasx::store_f32x8(buf.as_mut_ptr(), v) };
    let (a, b) = ((buf[0].max(buf[1])), (buf[2].max(buf[3])));
    let (c, d) = ((buf[4].max(buf[5])), (buf[6].max(buf[7])));
    (a.max(b)).max(c.max(d))
}

/// 把 8 个 per-lane 累加器按**固定次序的两两归约**求和。
///
/// 次序是契约的一部分：标量模拟（测试）必须用同一棵树，否则"同序"无从谈起。
#[inline]
fn hsum(v: lasx::F32x8) -> f32 {
    let mut buf = [0f32; 8];
    // SAFETY: `buf` 有 8 个元素。
    unsafe { lasx::store_f32x8(buf.as_mut_ptr(), v) };
    ((buf[0] + buf[1]) + (buf[2] + buf[3])) + ((buf[4] + buf[5]) + (buf[6] + buf[7]))
}

/// 把 `x[i][j0..cols]`（不足 8 个）装进一个 8 宽向量：其余 lane 填 `−1e30`。
///
/// `−1e30` 在 `exp` 入口被夹到下界，于是那些 lane 给出**精确 0**——
/// 加到 `≥ 1` 的归一化分母上不改变任何一位（见模块文档第 1 条）。
#[inline]
fn tail_vector(x_row: &[f32], mask_row: &[f32], j0: usize, cols: usize, scale: f32) -> lasx::F32x8 {
    let mut buf = [-1e30f32; 8];
    for (k, slot) in buf.iter_mut().enumerate().take(cols - j0) {
        let m = if mask_row.is_empty() {
            0.0
        } else {
            mask_row[j0 + k]
        };
        *slot = x_row[j0 + k].mul_add(scale, m);
    }
    // SAFETY: `buf` 有 8 个元素。
    unsafe { lasx::load_f32x8(buf.as_ptr()) }
}

/// 行内 softmax（LASX）。
///
/// `x`/`mask`/`out` 都是 `rows × cols` 行主序；`mask` 空切片表示无 mask。
///
/// # Panics
/// 调用方（`ffi`/`api`）已保证长度一致、`cols > 0`；这里只做 debug 断言。
pub(crate) fn softmax_rows(
    x: &[f32],
    mask: &[f32],
    scale: f32,
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), rows * cols);
    debug_assert_eq!(out.len(), rows * cols);
    debug_assert!(mask.is_empty() || mask.len() == rows * cols);
    debug_assert!(cols > 0);

    let vs = lasx::splat_f32(scale);
    let vscale = vs;
    for i in 0..rows {
        let x_row = &x[i * cols..(i + 1) * cols];
        let mask_row = if mask.is_empty() {
            &mask[..0]
        } else {
            &mask[i * cols..(i + 1) * cols]
        };
        let out_row = &mut out[i * cols..(i + 1) * cols];

        // ---- 第 1 遍：行内 max（先算 scale·x + mask）----
        let mut vmax = lasx::splat_f32(f32::NEG_INFINITY);
        let mut j = 0;
        while j + 8 <= cols {
            // SAFETY: `j + 8 ≤ cols`，落在本行内。
            let y = unsafe {
                let vx = lasx::load_f32x8(x_row.as_ptr().add(j));
                let vm = if mask_row.is_empty() {
                    lasx::zero_f32x8()
                } else {
                    lasx::load_f32x8(mask_row.as_ptr().add(j))
                };
                lasx_xvfmadd_s(vx, vscale, vm)
            };
            vmax = lasx::max_f32x8(vmax, y);
            j += 8;
        }
        if j < cols {
            let y = tail_vector(x_row, mask_row, j, cols, scale);
            vmax = lasx::max_f32x8(vmax, y);
        }
        let vmax = lasx::splat_f32(hmax(vmax));

        // ---- 第 2 遍：exp(d − max) 写进 out，同时按 per-lane 累加器求和 ----
        let mut acc = lasx::zero_f32x8();
        let mut j = 0;
        while j + 8 <= cols {
            // SAFETY: `j + 8 ≤ cols`。
            let e = unsafe {
                let vx = lasx::load_f32x8(x_row.as_ptr().add(j));
                let vm = if mask_row.is_empty() {
                    lasx::zero_f32x8()
                } else {
                    lasx::load_f32x8(mask_row.as_ptr().add(j))
                };
                let y = lasx_xvfmadd_s(vx, vscale, vm);
                let d = lasx_xvfsub_s(y, vmax);
                // 夹下界由 `exp` 入口负责：填充 lane 与极负输入都落到精确 0（有限、非 NaN）
                exp_vec(d)
            };
            acc = unsafe { lasx_xvfadd_s(acc, e) };
            // SAFETY: `j + 8 ≤ cols`。
            unsafe { lasx::store_f32x8(out_row.as_mut_ptr().add(j), e) };
            j += 8;
        }
        if j < cols {
            let y = tail_vector(x_row, mask_row, j, cols, scale);
            // SAFETY: 寄存器操作。
            let e = unsafe {
                let d = lasx_xvfsub_s(y, vmax);
                exp_vec(d)
            };
            acc = unsafe { lasx_xvfadd_s(acc, e) };
            let mut buf = [0f32; 8];
            // SAFETY: `buf` 有 8 个元素；只写本行的有效 lane（`[j, cols)`）。
            unsafe {
                lasx::store_f32x8(buf.as_mut_ptr(), e);
                for (slot, src) in out_row[j..cols].iter_mut().zip(buf.iter()) {
                    *slot = *src;
                }
            }
        }

        // ---- 第 3 遍：用 `1/Σ` 归一（一次除法 + 乘法，见模块文档）----
        let inv = 1.0 / hsum(acc);
        let vinv = lasx::splat_f32(inv);
        let mut j = 0;
        while j + 8 <= cols {
            // SAFETY: `j + 8 ≤ cols`。
            unsafe {
                let v = lasx::load_f32x8(out_row.as_ptr().add(j));
                lasx::store_f32x8(out_row.as_mut_ptr().add(j), lasx_xvfmul_s(v, vinv));
            }
            j += 8;
        }
        for slot in &mut out_row[j..cols] {
            *slot *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::nn_math::exp_seq;

    /// 独立参考：**标量模拟同一串 op**（不调算子内部的任何函数，避免自证）。
    ///
    /// 关键是与向量版同样的三个"同序"要素：per-lane 累加器（`acc[j % 8]`）、固定次序的
    /// 两两归约、以及 `exp` 的同一条 op 序列。所以它会与向量路径**逐位相同**。
    fn scalar_emulation(x: &[f32], mask: &[f32], scale: f32, rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0f32; rows * cols];
        for i in 0..rows {
            let (xr, mr) = (
                &x[i * cols..(i + 1) * cols],
                if mask.is_empty() {
                    &mask[..0]
                } else {
                    &mask[i * cols..(i + 1) * cols]
                },
            );
            let y = |j: usize| -> f32 {
                let m = if mr.is_empty() { 0.0 } else { mr[j] };
                xr[j].mul_add(scale, m)
            };
            // max：向量是"8 lane 各自 max 后固定次序折叠"，标量直接顺序 max（max 次序无关）
            let m = (0..cols).map(y).fold(f32::NEG_INFINITY, f32::max);
            let mut acc = [0f32; 8];
            for j in 0..cols {
                let e = exp_seq(y(j) - m);
                out[i * cols + j] = e;
                acc[j % 8] += e;
            }
            let sum =
                ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
            let inv = 1.0 / sum;
            for j in 0..cols {
                out[i * cols + j] *= inv;
            }
        }
        out
    }

    /// f64 参考：只用来量**精度**（不是逐位对照）。
    fn reference_f64(x: &[f32], mask: &[f32], scale: f32, rows: usize, cols: usize) -> Vec<f64> {
        let mut out = vec![0f64; rows * cols];
        for i in 0..rows {
            let mut row = vec![0f64; cols];
            for j in 0..cols {
                let m = if mask.is_empty() {
                    0.0
                } else {
                    mask[i * cols + j] as f64
                };
                row[j] = (x[i * cols + j] as f64) * (scale as f64) + m;
            }
            let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0f64;
            for v in row.iter_mut() {
                *v = (*v - m).exp();
                sum += *v;
            }
            for j in 0..cols {
                out[i * cols + j] = row[j] / sum;
            }
        }
        out
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// **逐位一致**：向量路径与标量模拟（同一串 op、同一累加次序）。
    #[test]
    fn test_matches_scalar_emulation_bit_for_bit() {
        let mut rng = crate::ops::testutil::Lcg(0x5eed);
        for &(rows, cols) in &[(1usize, 1usize), (2, 5), (3, 8), (4, 13), (5, 32), (7, 129)] {
            let x: Vec<f32> = (0..rows * cols)
                .map(|_| rng.f64() as f32 * 4.0 - 2.0)
                .collect();
            let mask: Vec<f32> = (0..rows * cols).map(|_| rng.f64() as f32).collect();
            for scale in [1.0f32, 0.125, 0.0, 8.0] {
                for use_mask in [false, true] {
                    let m: &[f32] = if use_mask { &mask } else { &[] };
                    let mut got = vec![0f32; rows * cols];
                    softmax_rows(&x, m, scale, rows, cols, &mut got);
                    let want = scalar_emulation(&x, m, scale, rows, cols);
                    assert_eq!(
                        bits(&got),
                        bits(&want),
                        "{rows}×{cols} scale={scale} mask={use_mask}"
                    );
                }
            }
        }
    }

    /// **列尾就是"填充 lane"的情形**：同一行按 `cols` 算，与把它显式补到 8 的倍数
    /// （补 `−1e30`，即模块文档里那条填充规则）再按补齐后的宽度算，**前 `cols` 个输出
    /// 必须逐位相同**。
    ///
    /// 这条直接检验设计：列尾不是单独的标量路径，而是"少几个有效 lane 的同一条向量路径"，
    /// 所以补出来的那次运行在真元素上必须给出同样的位。
    #[test]
    fn test_explicit_padding_matches_tail_path() {
        let mut rng = crate::ops::testutil::Lcg(0xabcd);
        for cols in [1usize, 5, 7, 8, 9, 13, 15, 16] {
            let row: Vec<f32> = (0..cols).map(|_| rng.f64() as f32 * 6.0 - 3.0).collect();
            let mut want = vec![0f32; cols];
            softmax_rows(&row, &[], 1.0, 1, cols, &mut want);

            // 显式补齐到 8 的倍数：多出来的 lane 填 −1e30（与算子内部同一条规则）
            let padded_len = cols.div_ceil(8) * 8;
            let mut padded = row.clone();
            padded.resize(padded_len, -1e30);
            let mut got = vec![0f32; padded_len];
            softmax_rows(&padded, &[], 1.0, 1, padded_len, &mut got);

            assert_eq!(
                bits(&got[..cols]),
                bits(&want),
                "cols={cols}：显式补齐与列尾路径不一致"
            );
        }
    }

    /// 数值：与 f64 参考的相对误差在 `exp` 多项式的量级上。
    #[test]
    fn test_accuracy_vs_f64_reference() {
        let mut rng = crate::ops::testutil::Lcg(0x1234);
        let (rows, cols) = (4usize, 100usize);
        let x: Vec<f32> = (0..rows * cols)
            .map(|_| rng.f64() as f32 * 20.0 - 10.0)
            .collect();
        let mask: Vec<f32> = (0..rows * cols)
            .map(|_| rng.f64() as f32 * 2.0 - 1.0)
            .collect();
        let mut got = vec![0f32; rows * cols];
        softmax_rows(&x, &mask, 0.5, rows, cols, &mut got);
        let want = reference_f64(&x, &mask, 0.5, rows, cols);
        for k in 0..rows * cols {
            let err = (got[k] as f64 - want[k]).abs();
            assert!(
                err < 1e-6,
                "idx={k}: got {} want {} err {err}",
                got[k],
                want[k]
            );
        }
        // 每行和应为 1（f32 精度内）
        for i in 0..rows {
            let s: f32 = got[i * cols..(i + 1) * cols].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "第 {i} 行和 = {s}");
        }
    }

    /// 边界：各行都退化的情形。
    #[test]
    fn test_degenerate_inputs() {
        // 行内全相等 → 均匀分布
        let x = vec![3.0f32; 12];
        let mut got = vec![0f32; 12];
        softmax_rows(&x, &[], 1.0, 2, 6, &mut got);
        for (k, v) in got.iter().enumerate() {
            assert!((v - 1.0 / 6.0).abs() < 1e-7, "idx={k}: {v}");
        }
        // scale = 0 → 均匀（因为 y 全为 0 或 mask）
        let mut got0 = vec![0f32; 8];
        softmax_rows(
            &[1.0, -5.0, 3.0, 0.0, 2.0, 2.0, 2.0, 2.0],
            &[],
            0.0,
            1,
            8,
            &mut got0,
        );
        for v in &got0 {
            assert!((v - 0.125).abs() < 1e-7, "{v}");
        }
        // 含 −inf（mask 常见形态）：−inf 的权重必须是 0
        let x = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mask = [0.0f32, f32::NEG_INFINITY, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut got = vec![0f32; 9];
        softmax_rows(&x, &mask, 1.0, 1, 9, &mut got);
        assert_eq!(got[1], 0.0, "−inf 的权重必须是 0");
        let s: f32 = got.iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "和 = {s}");
        // 极负输入：不得出现 NaN
        let big = [0.0f32, -1e30, -1e6];
        let mut gotb = vec![0f32; 3];
        softmax_rows(&big, &[], 1.0, 1, 3, &mut gotb);
        assert!(gotb.iter().all(|v| v.is_finite()), "{gotb:?}");
        assert!((gotb[0] - 1.0).abs() < 1e-7, "{gotb:?}");
    }
}
