//! `lasx_gelu_quick` —— GELU 的 sigmoid 近似：`y = x / (1 + exp(−1.702·x))`。
//!
//! **LASX-only**：没有降级分支，无 LASX 的 CPU 上会执行 LASX 指令（见手册 Caveats）。
//!
//! # 与 `gelu`（erf 形式）的关系
//!
//! 生态里 GELU 有两个常用形式，命名按**对照实现**取，方便调用方对齐：
//!
//! | 本库符号 | 公式 | 对照 |
//! |---|---|---|
//! | `lasx_gelu_quick` | `x·σ(1.702x)` | ggml `GELU_QUICK` |
//! | `lasx_gelu_erf` | `0.5x(1+erf(x/√2))` | PyTorch `gelu`（默认） |
//!
//! 本文件只有 quick 形式，erf 形式在 `ops::gelu_erf`（设计见 `docs/dev.md` §20.4 第 3 条）。
//!
//! # 位精确构造（设计见 `docs/dev.md` §20.4）
//!
//! 与 `silu` 共用骨架：逐元素、无归约，公式写**直接除法**而不是 `x·(1/den)`，
//! 门控分母用 [`crate::ops::nn_math`] 那一份。差别只有常系数：先把 `c·x` 乘出来再进 `exp`。
//!
//! ```text
//! t   = 1.702 · x      // f32 常量，先乘
//! den = 1 + exp(−t)
//! y   = x / den        // 单次舍入
//! ```

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（策略见 `docs/dev.md` §17）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]
use crate::arch::lasx;
use crate::ops::nn_math::{sigmoid_den_seq, sigmoid_den_vec};
use std::arch::loongarch64::*;

/// quick 形式的常数（ggml `GELU_QUICK` 用的就是 `1.702`）。
pub(crate) const GELU_QUICK_C: f32 = 1.702;

/// `gelu_quick` 的标量式（规范版；标量尾与测试里的"标量模拟"都照抄这一串）。
#[inline]
pub(crate) fn gelu_quick_seq(x: f32) -> f32 {
    x / sigmoid_den_seq(GELU_QUICK_C * x)
}

/// 就地逐元素 `gelu_quick`：`out[..n] = gelu_quick(x[..n])`。
///
/// `x` 与 `out` 长度相同（`n = x.len()`；调用方负责相等与对齐）。
///
/// # Safety
/// 无额外前提：`x`/`out` 长度相等，循环只在 `i + 8 <= n` 时载入 8 个 `f32`，
/// 标量尾只在 `i < n` 时下标访问。
pub(crate) fn gelu_quick_f32(x: &[f32], out: &mut [f32]) {
    let n = x.len();
    debug_assert_eq!(n, out.len(), "gelu_quick: 输入输出长度必须相等");
    let vc = lasx::splat_f32(GELU_QUICK_C);
    let mut i = 0;
    while i + 8 <= n {
        let vx = unsafe { lasx::load_f32x8(x.as_ptr().add(i)) };
        // t = c·x，再进门控分母
        let vt = unsafe { lasx_xvfmul_s(vx, vc) };
        let den = sigmoid_den_vec(vt);
        let vy = unsafe { lasx_xvfdiv_s(vx, den) };
        unsafe { lasx::store_f32x8(out.as_mut_ptr().add(i), vy) };
        i += 8;
    }
    for j in i..n {
        // 与向量路径同式，保证任一分块逐位一致
        out[j] = gelu_quick_seq(x[j]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::nn_math::exp_seq;

    /// 标量模拟：**独立**写出同一串 op（不调用 `gelu_quick_seq`）。
    fn emul(x: f32) -> f32 {
        let t = GELU_QUICK_C * x;
        let den = 1.0 + exp_seq(-t);
        x / den
    }

    /// f64 参考。`1.702` 在 f64 侧用同一个十进制常量（不是更精确的常数——那就是另一个式子了）。
    fn reference(x: f64) -> f64 {
        x / (1.0 + (-(1.702f64 * x)).exp())
    }

    fn run(x: &[f32]) -> Vec<f32> {
        let mut out = vec![0f32; x.len()];
        gelu_quick_f32(x, &mut out);
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
        for n in [64usize, 65, 127, 129] {
            let x: Vec<f32> = (0..n).map(|i| (i as f32).mul_add(0.37, -12.0)).collect();
            let out = run(&x);
            for (j, (&xj, &oj)) in x.iter().zip(out.iter()).enumerate() {
                assert_eq!(oj.to_bits(), emul(xj).to_bits(), "n={n} j={j} x={xj}");
            }
        }
    }

    /// 边界值：数值各自**指定**期望。
    #[test]
    fn test_exact_cases() {
        assert_eq!(gelu_quick_seq(0.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(gelu_quick_seq(-0.0).to_bits(), (-0.0f32).to_bits());
        // 大正侧：t 很大 ⇒ exp(−t) 夹到下界 = 0 ⇒ den 恰为 1 ⇒ y = x 本身
        for x in [60.0f32, 120.0, 1e30] {
            assert_eq!(gelu_quick_seq(x), x, "x={x}");
        }
        // 大负侧：den 饱和 ⇒ 输出极小但有限
        let y = gelu_quick_seq(-100.0);
        assert!(y.is_finite() && y < 0.0 && y.abs() < 1e-35, "y = {y}");
        assert_eq!(gelu_quick_seq(f32::INFINITY), f32::INFINITY);
        assert!(gelu_quick_seq(f32::NEG_INFINITY).is_infinite());
    }

    /// 数值：与 f64 参考的误差（实测包络，见 `docs/ops.md` §2.8）。
    #[test]
    fn test_accuracy_vs_f64_reference() {
        let (mut worst_abs, mut worst_rel) = (0f64, 0f64);
        let mut x = -150.0f64;
        while x <= 150.0 {
            let want = reference(x);
            let got = gelu_quick_seq(x as f32) as f64;
            worst_abs = worst_abs.max((got - want).abs());
            if want.abs() > 1.0 {
                worst_rel = worst_rel.max(((got - want) / want).abs());
            }
            x += 0.0005;
        }
        assert!(worst_abs < 1e-5, "最差绝对误差 {worst_abs:.3e}");
        assert!(worst_rel < 1e-6, "最差相对误差 {worst_rel:.3e}");
    }
}
