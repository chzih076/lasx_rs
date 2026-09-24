//! `lasx_silu` —— SiLU（swish）：`y = x / (1 + exp(−x))`。
//!
//! **LASX-only**：没有降级分支，无 LASX 的 CPU 上会执行 LASX 指令（见手册 Caveats）。
//!
//! # 位精确构造（设计见 `docs/dev.md` §20.4）
//!
//! 逐元素、**无归约**，所以不存在 softmax 那种"两种结合次序"的问题：只要标量与向量跑的是
//! 同一串 op，就逐位相同。本实现靠三件事保证这点：
//!
//! 1. `exp` 与门控分母都用 [`crate::ops::nn_math`] 里那一份（标量/向量相邻放置）；
//! 2. 除法写**直接除法** `x / den`，不是 `x · (1/den)`（后者多一次舍入；两者数学等价）；
//! 3. 列尾是**标量路径**，用与向量同式的 [`silu_seq`]，不是"另一套近似"。
//!
//! # 契约里的公式顺序（参考实现必须照抄才算逐位一致）
//!
//! ```text
//! den = 1 + exp(−x)     // exp 自带输入夹取 [−104, +88]
//! y   = x / den         // 单次舍入
//! ```
//!
//! `x < −88` 时 `exp` 的上界夹取让分母饱和在 `≈ e^88`，输出是 `x/e^88`（绝对值 ≤ 1e-35，
//! 对 f32 无实际影响）；`±inf` 不在契约内（见 `docs/ops.md` §2.8）。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（策略见 `docs/dev.md` §17）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]
use crate::arch::lasx;
use crate::ops::nn_math::{sigmoid_den_seq, sigmoid_den_vec};
use std::arch::loongarch64::*;

/// `silu` 的标量式（规范版；标量尾与测试里的"标量模拟"都照抄这一串）。
#[inline]
pub(crate) fn silu_seq(x: f32) -> f32 {
    x / sigmoid_den_seq(x)
}

/// 就地逐元素 `silu`：`out[..n] = silu(x[..n])`。
///
/// `x` 与 `out` 长度相同（`n = x.len()`；调用方负责相等与对齐）。
///
/// # Safety
/// 无额外前提：`x`/`out` 长度相等，循环只在 `i + 8 <= n` 时载入 8 个 `f32`，
/// 标量尾只在 `i < n` 时下标访问。
pub(crate) fn silu_f32(x: &[f32], out: &mut [f32]) {
    let n = x.len();
    debug_assert_eq!(n, out.len(), "silu: 输入输出长度必须相等");
    let mut i = 0;
    while i + 8 <= n {
        let vx = unsafe { lasx::load_f32x8(x.as_ptr().add(i)) };
        let den = sigmoid_den_vec(vx);
        // y = x / den（直接除法，单次舍入）
        let vy = unsafe { lasx_xvfdiv_s(vx, den) };
        unsafe { lasx::store_f32x8(out.as_mut_ptr().add(i), vy) };
        i += 8;
    }
    for j in i..n {
        // 与向量路径同式，保证任一分块逐位一致
        out[j] = silu_seq(x[j]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::nn_math::exp_seq;

    /// 标量模拟：**独立**写出同一串 op（不调用 `silu_seq`），用来交叉验证"规范版"本身。
    fn emul(x: f32) -> f32 {
        let den = 1.0 + exp_seq(-x);
        x / den
    }

    /// f64 参考（绝对误差有意义：`x → −∞` 时输出是"小于 f32 可分辨"的量级）。
    fn reference(x: f64) -> f64 {
        x / (1.0 + (-x).exp())
    }

    fn run(x: &[f32]) -> Vec<f32> {
        let mut out = vec![0f32; x.len()];
        silu_f32(x, &mut out);
        out
    }

    /// 逐位一致：向量主体 + 标量尾 vs 逐元素标量模拟（覆盖 8 的边界宽度）。
    #[test]
    fn test_matches_scalar_emulation_bit_for_bit() {
        for n in 0..=17usize {
            let x: Vec<f32> = (0..n).map(|i| (i as f32 - 4.0) * 0.75).collect();
            let out = run(&x);
            for (j, (&xj, &oj)) in x.iter().zip(out.iter()).enumerate() {
                assert_eq!(oj.to_bits(), emul(xj).to_bits(), "n={n} j={j} x={xj}");
            }
        }
        // 更宽的向量，跨多轮主体循环
        for n in [64usize, 65, 127, 129] {
            let x: Vec<f32> = (0..n).map(|i| (i as f32).mul_add(0.37, -12.0)).collect();
            let out = run(&x);
            for (j, (&xj, &oj)) in x.iter().zip(out.iter()).enumerate() {
                assert_eq!(oj.to_bits(), emul(xj).to_bits(), "n={n} j={j} x={xj}");
            }
        }
    }

    /// 边界值：数值各自**指定**期望，而不是靠"应该差不多"。
    #[test]
    fn test_exact_cases() {
        assert_eq!(silu_seq(0.0).to_bits(), 0.0f32.to_bits());
        // (−0)/(1+1) = −0：符号位保留（所以要比位型，比 `==` 是假通过）
        assert_eq!(silu_seq(-0.0).to_bits(), (-0.0f32).to_bits());
        // 大正侧：den = 1 + exp(−x) 夹到 exp 下界 → 恰为 1 → y = x 本身
        for x in [90.0f32, 120.0, 1e30] {
            assert_eq!(silu_seq(x), x, "x={x}");
        }
        // 大负侧：den 饱和在 e^88 附近，输出是 x/e^88（有限、非 NaN、极小）
        let y = silu_seq(-100.0);
        assert!(
            y.is_finite() && y < 0.0 && y.abs() < 1e-35,
            "silu(-100) = {y}"
        );
        // x = +inf：den = 1 → +inf（数学极限正确）
        assert_eq!(silu_seq(f32::INFINITY), f32::INFINITY);
        // x = −inf：不在契约内，但行为必须**确定**（不是 NaN 漂移）
        assert!(silu_seq(f32::NEG_INFINITY).is_infinite());
    }

    /// 数值：与 f64 参考的误差。`silu` 的输出在负侧会小到"相对误差无意义"，所以
    /// 断言用**绝对**误差，并且按区间分别卡（实测包络，见 `docs/ops.md` §2.8）。
    #[test]
    fn test_accuracy_vs_f64_reference() {
        let mut worst_abs = 0f64;
        let mut worst_rel = 0f64;
        let mut x = -120.0f64;
        while x <= 120.0 {
            let want = reference(x);
            let got = silu_seq(x as f32) as f64;
            worst_abs = worst_abs.max((got - want).abs());
            if want.abs() > 1.0 {
                worst_rel = worst_rel.max(((got - want) / want).abs());
            }
            x += 0.001;
        }
        // 绝对误差：负侧输出本身就 ≤ 1e-35，误差全在正侧被 exp 的相对误差放大
        assert!(worst_abs < 1e-5, "最差绝对误差 {worst_abs}");
        // 相对误差：|y| > 1 的区间（x ≳ 1.3），此时 y ≈ x，误差来自 exp 的相对误差
        assert!(worst_rel < 1e-5, "最差相对误差 {worst_rel}");
    }
}
