//! `matmul_i8` —— 批量 int8 矩阵乘（N3 的"prefill 形态"：权重读一遍、token 复用）。
//!
//! 语义：`X` 是 `m × k` 行主序 int8 激活（每 **token** 一个 scale），`W` 是 `n × k` 行主序
//! int8 权重（每 **输出通道** 一个 scale）：
//!
//! ```text
//! Y[t, o] = (Σ_i X[t,i]·W[o,i]) · (scale_w[o] · scale_x[t])
//! ```
//!
//! 这正是"权重矩阵 × 激活矩阵转置"（`Y = X·Wᵀ`）——**两侧的行都连续**，所以每个输出元素
//! 就是一次点积，直接复用 [`crate::ops::dot_i8`]。**位精确因此是可证明的**：整数累加精确、
//! 与顺序无关（i32 回绕只在 `k > 133 000` 时可达，与 `dot_i8` 同一语义），出口一次乘法。
//!
//! **LASX-only**（与 N1/N3 一致，见 `docs/ops.md` §3.2）。
//!
//! # 与 `gemv_i8` 的关系（为什么两个都要）
//!
//! `gemv_i8` 是 `m = 1` 的特例，但它**每算一个 token 就把权重从内存里重读一遍**（带宽受限，
//! 实测 10.7–12.4 GB/s ≈ 单流锚点）。批量把限制**从带宽换成算力**：权重在 L2 里被 `m` 个
//! token 复用，`m = 192` 时同一个前向的权重流量从 192×118 MB 降到 118 MB。
//! 依据与账见 `docs/dev.md` §21.6/§21.7、数据见 §7.9。
//!
//! # 循环顺序（有意写死）
//!
//! `t` 外层、`o` 内层：固定一个 token 时把 `W` 整片扫一遍（`n×k` 字节），**全部 token 扫的是
//! 同一片 `W`** ⇒ `W` 驻 L2 被复用（`docs/dev.md` §21.6 的账按这个前提算）。
//! 交换顺序会让 `X` 那一侧变成重读，`X` 小、`W` 大，所以不换。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（策略见 `docs/dev.md` §17）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]

/// `Y[t,o] = dot_i8(X[t,·], W[o,·]) · (scale_w[o] · scale_x[t])`。
///
/// `x` 是 `m × k`、`w` 是 `n × k`、`scale_x` 长 `m`、`scale_w` 长 `n`、`y` 是 `m × n`。
pub(crate) fn matmul_i8(
    x: &[i8],
    w: &[i8],
    scale_w: &[f32],
    scale_x: &[f32],
    m: usize,
    k: usize,
    n: usize,
    y: &mut [f32],
) {
    debug_assert!(x.len() >= m * k && w.len() >= n * k && y.len() >= m * n);
    debug_assert!(scale_w.len() >= n && scale_x.len() >= m);
    for t in 0..m {
        let x_row = &x[t * k..t * k + k];
        let sx = scale_x[t];
        let y_row = &mut y[t * n..t * n + n];
        for o in 0..n {
            let acc = crate::ops::dot_i8::dot_i8(&w[o * k..o * k + k], x_row);
            // 两个 scale 先乘成一个（一次舍入）再乘 —— 与 `gemv_i8` 同一出口契约（§2.13/§2.14）
            y_row[o] = (acc as f32) * (scale_w[o] * sx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::dot_i8::dot_i8;

    fn lcg_i8(seed: u64) -> impl FnMut() -> i8 {
        let mut s = seed;
        move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as i32 as i8
        }
    }

    /// **逐位对照"逐元素调用 `dot_i8`"**：这是"复用而非重写"的可证明形式。
    #[test]
    fn test_matmul_i8_is_bitwise_equal_to_per_element_dot_i8() {
        let mut rnd = lcg_i8(0x5a5a_1234);
        for &(m, k, n) in &[
            (1usize, 1usize, 1usize),
            (1, 7, 3),
            (2, 8, 5),
            (4, 33, 7),
            (17, 100, 13),
            (33, 257, 9),
            (64, 1055, 4),
        ] {
            let x: Vec<i8> = (0..m * k).map(|_| rnd()).collect();
            let w: Vec<i8> = (0..n * k).map(|_| rnd()).collect();
            let sw: Vec<f32> = (0..n).map(|o| 0.001 * (o as f32 + 1.0)).collect();
            let sx: Vec<f32> = (0..m).map(|t| 0.002 * (t as f32 + 1.0)).collect();
            let mut y = vec![0f32; m * n];
            matmul_i8(&x, &w, &sw, &sx, m, k, n, &mut y);
            for t in 0..m {
                for o in 0..n {
                    let acc = dot_i8(&w[o * k..o * k + k], &x[t * k..t * k + k]);
                    let want = (acc as f32) * (sw[o] * sx[t]);
                    assert_eq!(
                        y[t * n + o].to_bits(),
                        want.to_bits(),
                        "m={m} k={k} n={n} t={t} o={o}"
                    );
                }
            }
        }
    }

    /// **与 i64 精确参考一致**（k 在 N3 形状范围内：`127²·k < 2³¹` 不回绕）。
    #[test]
    fn test_matmul_i8_matches_exact_i64_reference_in_range() {
        let mut rnd = lcg_i8(0xfeed_face);
        for &(m, k, n) in &[(192usize, 768usize, 768usize), (192, 3072, 768)] {
            let x: Vec<i8> = (0..m * k).map(|_| rnd()).collect();
            let w: Vec<i8> = (0..n * k).map(|_| rnd()).collect();
            let sw: Vec<f32> = (0..n).map(|o| 0.0005 * (o as f32 % 7.0 + 1.0)).collect();
            let sx: Vec<f32> = (0..m).map(|t| 0.0007 * (t as f32 % 5.0 + 1.0)).collect();
            let mut y = vec![0f32; m * n];
            matmul_i8(&x, &w, &sw, &sx, m, k, n, &mut y);
            // 抽查若干元素（全量 192×768 用 i64 参考也可，但没必要跑满）
            for &(t, o) in &[(0usize, 0usize), (1, 7), (m / 2, n / 3), (m - 1, n - 1)] {
                let acc: i64 = (0..k)
                    .map(|i| x[t * k + i] as i64 * w[o * k + i] as i64)
                    .sum();
                let want = (acc as i32 as f32) * (sw[o] * sx[t]);
                assert_eq!(y[t * n + o].to_bits(), want.to_bits(), "t={t} o={o}");
            }
        }
    }

    /// `m = 1` 时与 `gemv_i8` **逐位一致**（同一个算子族的两个入口不能有两套数值）。
    #[test]
    fn test_matmul_i8_single_token_matches_gemv_i8() {
        let mut rnd = lcg_i8(0x1234_abcd);
        let (k, n) = (768usize, 96usize);
        let x: Vec<i8> = (0..k).map(|_| rnd()).collect();
        let w: Vec<i8> = (0..n * k).map(|_| rnd()).collect();
        let sw: Vec<f32> = (0..n).map(|o| 0.001 + 0.0003 * o as f32).collect();
        let sx = 0.0025f32;
        let mut y_mm = vec![0f32; n];
        matmul_i8(&x, &w, &sw, &[sx], 1, k, n, &mut y_mm);
        let mut y_gv = vec![0f32; n];
        crate::ops::gemv_i8::gemv_i8(&w, &sw, &x, sx, n, k, &mut y_gv);
        for o in 0..n {
            assert_eq!(y_mm[o].to_bits(), y_gv[o].to_bits(), "o={o}");
        }
    }
}
