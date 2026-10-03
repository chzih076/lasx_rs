//! `lasx_layer_norm` —— 行内 LayerNorm（`out = (x − mean)/√(var+eps) · w + b`）。
//!
//! **LASX-only**：没有降级分支，无 LASX 的 CPU 上会执行 LASX 指令（见手册 Caveats）。
//!
//! 与 [`crate::ops::rms_norm`] 的关系：**同一套行归约骨架**（8 个 per-lane 累加器 +
//! 固定次序两两归约 + 列尾补齐），差别只有两处——LayerNorm 要**减均值**、并且有**偏置 `b`**。
//! 它存在的理由是外部契约：ONNX 图上 `LayerNormalization` 17 处（`docs/platform.md` §4）。
//!
//! # 位精确构造（沿用 `softmax_rows`/`rms_norm` 的两条做法）
//!
//! 1. **8 个 per-lane 累加器**（lane `l` 累加 `j ≡ l (mod 8)`），末尾按**固定次序两两归约**；
//! 2. 列尾不足 8 个元素时**补齐**，且补的东西必须对累加**恰好无贡献**：
//!    - 第 1 遍（求和）：补 `0.0` ⇒ `fma(0,0,acc) = acc`，精确；
//!    - 第 2 遍（偏差平方和）：**补的是偏差 `0.0`**（不是补 `x=0`——那会得到 `−mean`，
//!      凭空加上 `mean²`）⇒ 尾块先把 `x−mean` 算进缓冲、其余 lane 填 `0.0`。
//!
//! **行级数学在标量域算完再 splat**（`mean = s/cols`、`ms = ss/cols`、`inv = 1/√(ms+eps)`）：
//! 省指令，而且天然位精确——标量模拟与向量路径用的是同一串标量运算。
//!
//! # 数值契约
//!
//! ```text
//! s   = Σ_j x[j]                     // 第 1 遍：8 lane 累加 + 固定次序归约（列尾补 0）
//! mean= s / cols                     // 一次除法（标量域）
//! ss  = (x[j] − mean)² 之和          // 第 2 遍：先减均值再平方（**不是** E[x²]−E[x]²）
//! ms  = ss / cols                    // 一次除法（标量域）
//! inv = 1 / sqrt(ms + eps)           // 一次 sqrt + 一次除法（标量域）
//! out = (x[j] − mean) · inv · w[j] + b[j]   // 次序是契约：先减、再乘 inv、再乘 w、最后加 b
//! ```
//!
//! 三条必须照抄的细节：
//!
//! - **方差用两遍法**（先算 mean、再算 Σ(x−mean)²），不用 `E[x²] − mean²`——后者在
//!   `mean` 远大于标准差时会灾难性相消；
//! - **`w`/`b` 传空切片表示"无权重/无偏置"**：无权重等价于乘 `1.0`（精确），
//!   **无偏置必须整个跳过加法**（不是加 `0.0`——那会把 `−0.0` 变成 `+0.0`，破坏逐位一致）；
//! - **行级除法/开方只在标量域做一次**，向量路径与标量模拟共用同一串。
//!
//! **不在契约内**：`eps = 0` 且整行同值（`ss = 0`）⇒ `inv = +inf` ⇒ `0·inf = NaN`
//! （与 `rms_norm` 同理，`api` 层要求 `eps > 0`）。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（策略见 `docs/dev.md` §17）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX/LSX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]
use crate::arch::lasx;
use std::arch::loongarch64::*;

/// 8 个 per-lane 累加器按**固定次序两两归约**求和（与 `softmax_rows`/`rms_norm` 同序）。
#[inline]
fn hsum(v: lasx::F32x8) -> f32 {
    let mut buf = [0f32; 8];
    // SAFETY: `buf` 有 8 个元素。
    unsafe { lasx::store_f32x8(buf.as_mut_ptr(), v) };
    ((buf[0] + buf[1]) + (buf[2] + buf[3])) + ((buf[4] + buf[5]) + (buf[6] + buf[7]))
}

/// 行内 LayerNorm（LASX）。
///
/// `x`/`out` 是 `rows × cols` 行主序；`w`/`b` 是**按列**的权重与偏置（长度 `cols`，
/// 空切片表示无）。`eps` 由调用方保证为正（`api`/`ffi` 层校验）。
///
/// # Panics
/// 调用方（`ffi`/`api`）已保证长度一致、`cols > 0`；这里只做 debug 断言。
pub(crate) fn layer_norm(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    eps: f32,
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), rows * cols);
    debug_assert_eq!(out.len(), rows * cols);
    debug_assert!(w.is_empty() || w.len() == cols);
    debug_assert!(b.is_empty() || b.len() == cols);
    debug_assert!(cols > 0);

    for i in 0..rows {
        let x_row = &x[i * cols..(i + 1) * cols];
        let out_row = &mut out[i * cols..(i + 1) * cols];

        // ---- 第 1 遍：Σx（列尾补 0.0，对累加恰好无贡献）----
        let mut acc = lasx::zero_f32x8();
        let mut j = 0;
        while j + 8 <= cols {
            // SAFETY: `j + 8 ≤ cols`，落在本行内。
            let vx = unsafe { lasx::load_f32x8(x_row.as_ptr().add(j)) };
            acc = unsafe { lasx_xvfadd_s(acc, vx) };
            j += 8;
        }
        if j < cols {
            let mut buf = [0f32; 8];
            buf[..cols - j].copy_from_slice(&x_row[j..cols]);
            // SAFETY: `buf` 有 8 个元素。
            let vx = unsafe { lasx::load_f32x8(buf.as_ptr()) };
            acc = unsafe { lasx_xvfadd_s(acc, vx) };
        }
        let mean = hsum(acc) / cols as f32;
        let vmean = lasx::splat_f32(mean);

        // ---- 第 2 遍：Σ(x−mean)²（**补偏差 0.0**，不是补 x=0）----
        let mut acc2 = lasx::zero_f32x8();
        let mut j = 0;
        while j + 8 <= cols {
            // SAFETY: `j + 8 ≤ cols`。
            let vx = unsafe { lasx::load_f32x8(x_row.as_ptr().add(j)) };
            let d = unsafe { lasx_xvfsub_s(vx, vmean) };
            acc2 = unsafe { lasx_xvfmadd_s(d, d, acc2) };
            j += 8;
        }
        if j < cols {
            let mut buf = [0f32; 8];
            for k in 0..cols - j {
                buf[k] = x_row[j + k] - mean;
            }
            // SAFETY: `buf` 有 8 个元素。
            let d = unsafe { lasx::load_f32x8(buf.as_ptr()) };
            acc2 = unsafe { lasx_xvfmadd_s(d, d, acc2) };
        }
        let inv = 1.0 / (hsum(acc2) / cols as f32 + eps).sqrt();
        let vinv = lasx::splat_f32(inv);

        // ---- 第 3 遍：out = ((x − mean) · inv) · w (+ b) ----
        let mut j = 0;
        while j + 8 <= cols {
            // SAFETY: `j + 8 ≤ cols`。
            let vx = unsafe { lasx::load_f32x8(x_row.as_ptr().add(j)) };
            let d = unsafe { lasx_xvfsub_s(vx, vmean) };
            let mut v = unsafe {
                lasx_xvfmul_s(lasx_xvfmul_s(d, vinv), {
                    if w.is_empty() {
                        lasx::splat_f32(1.0)
                    } else {
                        lasx::load_f32x8(w.as_ptr().add(j))
                    }
                })
            };
            if !b.is_empty() {
                // SAFETY: `j + 8 ≤ cols`。
                let vb = unsafe { lasx::load_f32x8(b.as_ptr().add(j)) };
                v = unsafe { lasx_xvfadd_s(v, vb) };
            }
            // SAFETY: `j + 8 ≤ cols`。
            unsafe { lasx::store_f32x8(out_row.as_mut_ptr().add(j), v) };
            j += 8;
        }
        for k in j..cols {
            let weight = if w.is_empty() { 1.0 } else { w[k] };
            let v = (x_row[k] - mean) * inv * weight;
            out_row[k] = if b.is_empty() { v } else { v + b[k] };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 独立参考：**标量模拟同一串 op**（per-lane 累加 + 固定次序归约 + 标量域行级数学）。
    /// 与向量路径必须**逐位相同**。
    fn scalar_emulation(
        x: &[f32],
        w: &[f32],
        b: &[f32],
        eps: f32,
        rows: usize,
        cols: usize,
    ) -> Vec<f32> {
        let mut out = vec![0f32; rows * cols];
        for i in 0..rows {
            let xr = &x[i * cols..(i + 1) * cols];
            let mut acc = [0f32; 8];
            for j in 0..cols {
                acc[j % 8] += xr[j];
            }
            let s =
                ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
            let mean = s / cols as f32;
            let mut acc2 = [0f32; 8];
            for j in 0..cols {
                let d = xr[j] - mean;
                acc2[j % 8] = d.mul_add(d, acc2[j % 8]);
            }
            let ss = ((acc2[0] + acc2[1]) + (acc2[2] + acc2[3]))
                + ((acc2[4] + acc2[5]) + (acc2[6] + acc2[7]));
            let inv = 1.0 / (ss / cols as f32 + eps).sqrt();
            for j in 0..cols {
                let weight = if w.is_empty() { 1.0 } else { w[j] };
                let v = (xr[j] - mean) * inv * weight;
                out[i * cols + j] = if b.is_empty() { v } else { v + b[j] };
            }
        }
        out
    }

    /// 与标量模拟**逐位一致**，覆盖列尾（cols 非 8 的倍数）、无权重/无偏置的组合、单行单列。
    #[test]
    fn test_layer_norm_matches_scalar_emulation_bit_for_bit() {
        for &(rows, cols) in &[
            (1usize, 1usize),
            (1, 7),
            (3, 8),
            (5, 17),
            (2, 33),
            (7, 64),
            (4, 129),
            (1, 1000),
        ] {
            let mut rng = crate::ops::testutil::Lcg((rows * 1000 + cols) as u64 ^ 0x1a2b);
            let x: Vec<f32> = (0..rows * cols)
                .map(|_| 8.0 * rng.f64() as f32 - 4.0)
                .collect();
            let w: Vec<f32> = (0..cols).map(|_| 2.0 * rng.f64() as f32).collect();
            let b: Vec<f32> = (0..cols).map(|_| 4.0 * rng.f64() as f32 - 2.0).collect();
            for (wname, ww) in [("有 w", &w[..]), ("无 w", &[][..])] {
                for (bname, bb) in [("有 b", &b[..]), ("无 b", &[][..])] {
                    let want = scalar_emulation(&x, ww, bb, 1e-5, rows, cols);
                    let mut got = vec![0f32; rows * cols];
                    layer_norm(&x, ww, bb, 1e-5, rows, cols, &mut got);
                    for i in 0..rows * cols {
                        assert_eq!(
                            got[i].to_bits(),
                            want[i].to_bits(),
                            "{rows}×{cols} {wname} {bname} @ {i}"
                        );
                    }
                }
            }
        }
    }

    /// **减均值确实在做**：整行平移常数 ⇒ LayerNorm 输出不变（RMSNorm 会变）。
    /// 这条是"LayerNorm ≠ RMSNorm"的行为级断言，防止实现退化成 RMSNorm。
    #[test]
    fn test_layer_norm_is_shift_invariant_unlike_rms_norm() {
        let (rows, cols) = (3usize, 40usize);
        let mut rng = crate::ops::testutil::Lcg(0x5151);
        let x: Vec<f32> = (0..rows * cols).map(|_| 3.0 * rng.f64() as f32).collect();
        let shifted: Vec<f32> = x.iter().map(|&v| v + 100.0).collect();
        let mut a = vec![0f32; rows * cols];
        let mut b = vec![0f32; rows * cols];
        layer_norm(&x, &[], &[], 1e-5, rows, cols, &mut a);
        layer_norm(&shifted, &[], &[], 1e-5, rows, cols, &mut b);
        for i in 0..rows * cols {
            // 平移 100 后容差内不变（浮点上不是严格逐位：`(x+100)-mean'` 会掉低位）
            assert!((a[i] - b[i]).abs() < 1e-5, "@{i}: {} vs {}", a[i], b[i]);
        }
        // 对照：同一输入下 RMSNorm 会因平移而变（且变化很大）
        let mut c = vec![0f32; rows * cols];
        crate::ops::rms_norm::rms_norm(&shifted, &[], 1e-5, rows, cols, &mut c);
        let mut d = vec![0f32; rows * cols];
        crate::ops::rms_norm::rms_norm(&x, &[], 1e-5, rows, cols, &mut d);
        let moved = c
            .iter()
            .zip(&d)
            .fold(0f32, |m, (&u, &v)| m.max((u - v).abs()));
        assert!(moved > 0.5, "RMSNorm 应当被平移显著影响：{moved}");
    }

    /// `w`/`b` 的"空切片"语义：**无偏置不是加 `0.0`**——那会把 `−0.0` 变成 `+0.0`。
    /// 所以空 `b` 必须**整个跳过加法**；显式传全 `0.0` 的 `b` 才会翻符号。
    #[test]
    fn test_layer_norm_empty_bias_keeps_signed_zero() {
        let x = [-0.0f32, -0.0, 0.0, 0.0];
        let zb = [0.0f32; 4];
        let mut no_bias = [9f32; 4];
        let mut with_bias = [9f32; 4];
        layer_norm(&x, &[], &[], 1e-5, 1, 4, &mut no_bias);
        layer_norm(&x, &[], &zb, 1e-5, 1, 4, &mut with_bias);
        assert_eq!(
            no_bias[0].to_bits(),
            (-0.0f32).to_bits(),
            "无偏置：`−0.0` 应保持为 `−0.0`"
        );
        assert_eq!(
            with_bias[0].to_bits(),
            0.0f32.to_bits(),
            "显式 `+0.0` 偏置会把 `−0.0` 变成 `+0.0`（这正是空 `b` 必须跳过加法的原因）"
        );
    }
}
