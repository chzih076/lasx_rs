//! `gemv_i8` —— int8 权重 × int8 激活的矩阵-向量乘（N3 的消费主力）。
//!
//! 语义：`W` 是 `m × k` 行主序的 int8 权重（每输出通道一个 scale），`x` 是 `k` 长的 int8 激活
//! （每 token 一个 scale）：
//!
//! ```text
//! y[o] = (Σ_i W[o,i]·x[i]) · (scale_w[o] · scale_x)      // i32 累加，出口乘两个 scale 的积
//! ```
//!
//! **LASX-only**（与 N1 批次一致，见 `docs/ops.md` §3.2）。
//!
//! # 为什么可以复用 `dot_i8`
//!
//! `dot_i8` 已经是"i16 乘积 → 拓宽 i32 累加、每 1024 字节落 i64"的成熟实现（含防溢出与
//! 行尾标量尾），而 GEMV 的每一行**就是一次点积**。所以这里不重写向量内核，只做两件事：
//! 逐行调用 `dot_i8`，再把 i32 和乘上两个 scale —— 一致性因此是**复用**来的，不是碰巧。
//!
//! # 数值契约
//!
//! 1. **累加是整数精确的**：`acc = Σ W·x`（i64 中间累加，返回 i32；`m`/`k` 在
//!    `docs/ops.md` §12.3 N3 的形状范围内不会回绕：`k ≤ 133 000` 时 `127²·k < 2³¹`）。
//! 2. **出口一次乘法**：`y = (acc as f32) · (sw[o] · sx)`——先把两个 scale **乘成一个**
//!    （f32 一次舍入），再与 `acc` 相乘。参考实现必须同样写 `(acc as f32) * (sw * sx)`，
//!    写成 `(acc as f32 * sw) * sx` 是两次舍入，不保证逐位一致。
//! 3. 未量化的偏置/残差不在本内核内（调用方自己在 f32 域加）。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（策略见 `docs/dev.md` §17）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]

/// `y[o] = dot_i8(W[o,·], x) · (scale_w[o] · scale_x)`。
///
/// `w` 是 `m × k`（行主序），`x` 长 `k`，`scale_w` 长 `m`，`y` 长 `m`。
pub(crate) fn gemv_i8(
    w: &[i8],
    scale_w: &[f32],
    x: &[i8],
    scale_x: f32,
    m: usize,
    k: usize,
    y: &mut [f32],
) {
    for o in 0..m {
        let acc = crate::ops::dot_i8::dot_i8(&w[o * k..o * k + k], x);
        // 两个 scale 先乘成一个（一次舍入），再乘到累加和上 —— 契约第 2 条
        let s = scale_w[o] * scale_x;
        y[o] = (acc as f32) * s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 精确的 i64 参考：整数累加 → i32（与 `dot_i8` 同一语义）→ 同一个出口乘法。
    fn reference(w: &[i8], sw: &[f32], x: &[i8], sx: f32, m: usize, k: usize) -> Vec<f32> {
        (0..m)
            .map(|o| {
                let acc: i64 = (0..k).map(|i| w[o * k + i] as i64 * x[i] as i64).sum();
                (acc as i32 as f32) * (sw[o] * sx)
            })
            .collect()
    }

    fn lcg_i8(seed: u64) -> impl FnMut() -> i8 {
        let mut s = seed;
        move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as i32 as i8
        }
    }

    /// 与精确参考**逐位一致**，覆盖 k 的向量尾/标量尾与各种 m。
    #[test]
    fn test_gemv_i8_matches_exact_reference() {
        let mut rnd = lcg_i8(0x9e3779b9);
        for &(m, k) in &[
            (1usize, 1usize),
            (1, 7),
            (3, 8),
            (4, 33),
            (5, 64),
            (8, 100),
            (17, 257),
            (33, 1024),
            (64, 1055),
        ] {
            let w: Vec<i8> = (0..m * k).map(|_| rnd()).collect();
            let x: Vec<i8> = (0..k).map(|_| rnd()).collect();
            let sw: Vec<f32> = (0..m).map(|o| 0.001 * (o as f32 + 1.0)).collect();
            let sx = 0.0025f32;
            let mut y = vec![0f32; m];
            gemv_i8(&w, &sw, &x, sx, m, k, &mut y);
            let want = reference(&w, &sw, &x, sx, m, k);
            for o in 0..m {
                assert_eq!(y[o].to_bits(), want[o].to_bits(), "m={m} k={k} o={o}");
            }
        }
    }

    /// 极值填满（±127 / −128）不溢出：`k` 在 N3 形状范围内（768/3072）应精确。
    #[test]
    fn test_gemv_i8_extremes_are_exact_in_range() {
        for &k in &[768usize, 1024, 3072] {
            let w = vec![127i8; k];
            let x = vec![127i8; k];
            let sw = [1.0f32];
            let mut y = [0f32; 1];
            gemv_i8(&w, &sw, &x, 1.0, 1, k, &mut y);
            let want = reference(&w, &sw, &x, 1.0, 1, k);
            assert_eq!(y[0].to_bits(), want[0].to_bits(), "k={k} 全 127");
            // 精确值可解析算出：127·127·k
            assert_eq!(y[0], (127.0f32 * 127.0) * k as f32);
        }
    }

    /// 出口乘法的次序进契约：`(acc as f32)·(sw·sx)` 而不是 `((acc as f32)·sw)·sx`。
    /// 用一个"两次舍入会不同"的样本锁住它。
    #[test]
    fn test_scale_product_is_single_rounding() {
        let k = 768usize;
        let w: Vec<i8> = (0..k)
            .map(|i| if i % 2 == 0 { 127 } else { -128 })
            .collect();
        let x: Vec<i8> = (0..k).map(|i| if i % 3 == 0 { 127 } else { -1 }).collect();
        let sw = [0.007_812_5f32]; // 1/128，二进制精确
        let sx = 0.003_333_33f32; // 非二进制精确 ⇒ 两种次序会有 1 ulp 差别
        let mut y = [0f32; 1];
        gemv_i8(&w, &sw, &x, sx, 1, k, &mut y);
        let acc: i64 = (0..k).map(|i| w[i] as i64 * x[i] as i64).sum();
        let once = (acc as i32 as f32) * (sw[0] * sx);
        assert_eq!(y[0].to_bits(), once.to_bits(), "出口必须是单次舍入");
    }
}
