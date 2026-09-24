//! `lasx_rms_norm` —— 行内 RMSNorm（`out = x / √(mean(x²)+eps) · w`）。
//!
//! **LASX-only**：没有降级分支，无 LASX 的 CPU 上会执行 LASX 指令（见手册 Caveats）。
//!
//! # 位精确构造（与 `softmax_rows` 同一套骨架）
//!
//! 行归约是位精确的唯一难点（向量水平归约 vs 逐个累加是两种结合次序），这里沿用
//! `softmax_rows` 的两条做法：
//!
//! 1. **8 个 per-lane 累加器**（lane `l` 累加 `j ≡ l (mod 8)`），末尾按**固定次序两两归约**；
//! 2. 列尾不足 8 个元素时**用 `0.0` 填充**——`0² = 0`，对平方和恰好无贡献，所以填充是
//!    完全精确的（比 softmax 那边用 `−1e30` + 夹取更干净，不需要任何夹取）。
//!
//! 另一个决定：**行级数学（除法、开方、倒数）在标量域算完再 splat**。理由是两条：
//!
//! - 省指令：每行只做一次 `s/cols`、一次 `sqrt`、一次 `1/x`，而不是每 8 个元素一次；
//! - 更重要的是**天然位精确**：标量模拟与向量路径用的是同一串标量运算，不存在"向量
//!   `sqrt` 与标量 `sqrt` 是否逐位相同"这个问题。
//!
//! 开方与倒数用**精确指令**（`f32::sqrt` + `1.0/x`），不用 `frsqrte`/`frecipe` 估算 +
//! 牛顿迭代——`docs/dev.md` §13.1 已实测本机估算指令比精确除法**更慢**（6.8 周期 vs 4.0），
//! 这条结论在这里直接复用，不重测。
//!
//! # 数值契约
//!
//! ```text
//! s   = Σ_j x[j]²                    // 8 lane 累加器 + 固定次序两两归约（列尾补 0）
//! ms  = s / cols                     // 一次除法
//! inv = 1 / sqrt(ms + eps)           // 一次 sqrt + 一次除法（都是标量）
//! out = (x · inv) · w                // 先乘 inv 再乘权重（次序是契约的一部分）
//! ```
//!
//! **不在契约内**：`eps = 0` 且整行全零 ⇒ `ms + eps = 0` ⇒ `inv = +inf` ⇒ `out = 0·inf = NaN`。
//! 这是定义域问题（RMSNorm 的 `eps` 就是为此存在的），`api` 层要求 `eps > 0`。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（策略见 `docs/dev.md` §17）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX/LSX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]
use crate::arch::lasx;
use std::arch::loongarch64::*;

/// 把 8 个 per-lane 累加器按**固定次序的两两归约**求和（与 `softmax_rows` 同序）。
#[inline]
fn hsum(v: lasx::F32x8) -> f32 {
    let mut buf = [0f32; 8];
    // SAFETY: `buf` 有 8 个元素。
    unsafe { lasx::store_f32x8(buf.as_mut_ptr(), v) };
    ((buf[0] + buf[1]) + (buf[2] + buf[3])) + ((buf[4] + buf[5]) + (buf[6] + buf[7]))
}

/// 行内 RMSNorm（LASX）。
///
/// `x`/`out` 是 `rows × cols` 行主序；`w` 是**按列**的权重（长度 `cols`，空切片表示无权重）。
///
/// # Panics
/// 调用方（`ffi`/`api`）已保证长度一致、`cols > 0`；这里只做 debug 断言。
pub(crate) fn rms_norm(x: &[f32], w: &[f32], eps: f32, rows: usize, cols: usize, out: &mut [f32]) {
    debug_assert_eq!(x.len(), rows * cols);
    debug_assert_eq!(out.len(), rows * cols);
    debug_assert!(w.is_empty() || w.len() == cols);
    debug_assert!(cols > 0);

    for i in 0..rows {
        let x_row = &x[i * cols..(i + 1) * cols];
        let out_row = &mut out[i * cols..(i + 1) * cols];

        // ---- 第 1 遍：平方和（列尾补 0，填充精确无贡献）----
        let mut acc = lasx::zero_f32x8();
        let mut j = 0;
        while j + 8 <= cols {
            // SAFETY: `j + 8 ≤ cols`，落在本行内。
            let vx = unsafe { lasx::load_f32x8(x_row.as_ptr().add(j)) };
            // SAFETY: 寄存器操作。
            acc = unsafe { lasx_xvfmadd_s(vx, vx, acc) };
            j += 8;
        }
        if j < cols {
            // 尾部：装进 8 宽向量，其余 lane 为 0.0
            let mut buf = [0f32; 8];
            buf[..cols - j].copy_from_slice(&x_row[j..cols]);
            // SAFETY: `buf` 有 8 个元素。
            let vx = unsafe { lasx::load_f32x8(buf.as_ptr()) };
            // SAFETY: 寄存器操作。
            acc = unsafe { lasx_xvfmadd_s(vx, vx, acc) };
        }
        let s = hsum(acc);

        // ---- 行级数学：标量域（见模块文档：省指令 + 天然位精确）----
        let inv = 1.0 / (s / cols as f32 + eps).sqrt();
        let vinv = lasx::splat_f32(inv);

        // ---- 第 2 遍：`out = (x · inv) · w` ----
        let mut j = 0;
        while j + 8 <= cols {
            // SAFETY: `j + 8 ≤ cols`。
            let vw = unsafe {
                if w.is_empty() {
                    lasx::splat_f32(1.0)
                } else {
                    lasx::load_f32x8(w.as_ptr().add(j))
                }
            };
            // SAFETY: `j + 8 ≤ cols`。
            unsafe {
                let vx = lasx::load_f32x8(x_row.as_ptr().add(j));
                let v = lasx_xvfmul_s(lasx_xvfmul_s(vx, vinv), vw);
                lasx::store_f32x8(out_row.as_mut_ptr().add(j), v);
            }
            j += 8;
        }
        for k in j..cols {
            let weight = if w.is_empty() { 1.0 } else { w[k] };
            out_row[k] = x_row[k] * inv * weight;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 独立参考：**标量模拟同一串 op**（per-lane 累加 + 固定次序两两归约 + 标量域行级数学）。
    /// 与向量路径必须**逐位相同**。
    fn scalar_emulation(x: &[f32], w: &[f32], eps: f32, rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0f32; rows * cols];
        for i in 0..rows {
            let xr = &x[i * cols..(i + 1) * cols];
            let mut acc = [0f32; 8];
            for j in 0..cols {
                // 与向量路径同一条 op：`fma(x, x, acc)`（一次舍入），不是 `acc + x*x`（两次）
                acc[j % 8] = xr[j].mul_add(xr[j], acc[j % 8]);
            }
            let s =
                ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
            let inv = 1.0 / (s / cols as f32 + eps).sqrt();
            for j in 0..cols {
                let weight = if w.is_empty() { 1.0 } else { w[j] };
                out[i * cols + j] = xr[j] * inv * weight;
            }
        }
        out
    }

    /// f64 参考：只量精度。
    fn reference_f64(x: &[f32], w: &[f32], eps: f32, rows: usize, cols: usize) -> Vec<f64> {
        let mut out = vec![0f64; rows * cols];
        for i in 0..rows {
            let mut s = 0f64;
            for j in 0..cols {
                let v = x[i * cols + j] as f64;
                s += v * v;
            }
            let inv = 1.0 / (s / cols as f64 + eps as f64).sqrt();
            for j in 0..cols {
                let weight = if w.is_empty() { 1.0 } else { w[j] as f64 };
                out[i * cols + j] = (x[i * cols + j] as f64) * inv * weight;
            }
        }
        out
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// **逐位一致**：向量路径 vs 标量模拟。
    #[test]
    fn test_matches_scalar_emulation_bit_for_bit() {
        let mut rng = crate::ops::testutil::Lcg(0x11ee);
        for &(rows, cols) in &[(1usize, 1usize), (2, 7), (3, 8), (4, 13), (5, 64), (6, 257)] {
            let x: Vec<f32> = (0..rows * cols)
                .map(|_| rng.f64() as f32 * 8.0 - 4.0)
                .collect();
            let w: Vec<f32> = (0..cols).map(|_| rng.f64() as f32 * 2.0).collect();
            for eps in [1e-5f32, 1e-6, 0.5] {
                for use_w in [false, true] {
                    let ww: &[f32] = if use_w { &w } else { &[] };
                    let mut got = vec![0f32; rows * cols];
                    rms_norm(&x, ww, eps, rows, cols, &mut got);
                    let want = scalar_emulation(&x, ww, eps, rows, cols);
                    assert_eq!(bits(&got), bits(&want), "{rows}×{cols} eps={eps} w={use_w}");
                }
            }
        }
    }

    /// **列尾边界扫描**：`cols` 从 1 到 17（跨过 8 的两侧）逐位对照标量模拟。
    ///
    /// 这里与 `softmax_rows` 的那条测试**不同**，值得记一笔：softmax 可以"把行显式补齐再算"
    /// 来验列尾（因为它的分母是 `exp` 的和，填充 lane 经夹取后贡献**精确的 0**，且 `cols`
    /// 本身不进分母）；rms_norm **不能**这么做——`mean = s / cols`，把行补齐会同时改掉除数，
    /// 那是另一个输入。所以这里改成"扫过边界、逐位对照独立模拟"。
    #[test]
    fn test_column_tail_boundary_sweep() {
        let mut rng = crate::ops::testutil::Lcg(0x22ff);
        for cols in 1..=17usize {
            let row: Vec<f32> = (0..cols).map(|_| rng.f64() as f32 * 4.0 - 2.0).collect();
            let w: Vec<f32> = (0..cols).map(|_| rng.f64() as f32 + 0.5).collect();
            for use_w in [false, true] {
                let ww: &[f32] = if use_w { &w } else { &[] };
                let mut got = vec![0f32; cols];
                rms_norm(&row, ww, 1e-5, 1, cols, &mut got);
                let want = scalar_emulation(&row, ww, 1e-5, 1, cols);
                assert_eq!(bits(&got), bits(&want), "cols={cols} w={use_w}");
            }
        }
    }

    /// 数值：与 f64 参考的相对误差在 f32 精度内。
    #[test]
    fn test_accuracy_vs_f64_reference() {
        let mut rng = crate::ops::testutil::Lcg(0x3344);
        let (rows, cols) = (4usize, 200usize);
        let x: Vec<f32> = (0..rows * cols)
            .map(|_| rng.f64() as f32 * 10.0 - 5.0)
            .collect();
        let w: Vec<f32> = (0..cols).map(|_| rng.f64() as f32 + 0.5).collect();
        let mut got = vec![0f32; rows * cols];
        rms_norm(&x, &w, 1e-5, rows, cols, &mut got);
        let want = reference_f64(&x, &w, 1e-5, rows, cols);
        for k in 0..rows * cols {
            let err = (got[k] as f64 - want[k]).abs() / want[k].abs().max(1e-12);
            assert!(
                err < 1e-5,
                "idx={k}: got {} want {} rel {err}",
                got[k],
                want[k]
            );
        }
    }

    /// 边界：单列、全零行（eps 保护）、权重为 1（等价于无权重）。
    #[test]
    fn test_degenerate_inputs() {
        // cols = 1：单个元素的 RMSNorm 是 ±1（x/|x|，符号保留）
        let x = [3.0f32, -2.0, 0.5];
        let mut got = vec![0f32; 3];
        rms_norm(&x, &[], 1e-5, 3, 1, &mut got);
        for (k, v) in got.iter().enumerate() {
            let sign = if x[k] < 0.0 { -1.0 } else { 1.0 };
            assert!((v - sign).abs() < 1e-4, "idx={k}: {v}");
        }
        // 全零行 + eps > 0：不得出 NaN/Inf
        let z = vec![0f32; 12];
        let mut gotz = vec![0f32; 12];
        rms_norm(&z, &[], 1e-5, 3, 4, &mut gotz);
        assert!(gotz.iter().all(|v| *v == 0.0), "{gotz:?}");
        // 权重全 1 == 无权重（逐位）
        let w1 = vec![1.0f32; 4];
        let mut g1 = vec![0f32; 4];
        let mut g2 = vec![0f32; 4];
        rms_norm(&[1.0, 2.0, 3.0, 4.0], &w1, 1e-6, 1, 4, &mut g1);
        rms_norm(&[1.0, 2.0, 3.0, 4.0], &[], 1e-6, 1, 4, &mut g2);
        assert_eq!(bits(&g1), bits(&g2), "权重全 1 应与无权重逐位相同");
    }
}
