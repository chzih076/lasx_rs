//! `lasx_gelu_erf` —— GELU 的 erf 形式：`y = 0.5·x·(1 + erf(x/√2))`。
//!
//! **LASX-only**：没有降级分支，无 LASX 的 CPU 上会执行 LASX 指令（见手册 Caveats）。
//!
//! # 与 `gelu_quick` 的关系
//!
//! `gelu_quick`（`x·σ(1.702x)`）对的是 ggml 的 `GELU_QUICK`；本文件对的是 PyTorch `gelu`
//! 的默认形式（真 erf）。两者不是同一个函数的两种精度，而是**两个不同的近似**，调用方要按
//! 自己对齐的生态选。
//!
//! # `erf` 的近似（这是契约的核心，见 `docs/ops.md` §2.9）
//!
//! `erf` 没有硬件指令，用 **Abramowitz & Stegun 7.1.26**（5 项 + 指数）：
//!
//! ```text
//! p = 0.3275911;  t = 1/(1 + p·|z|)
//! erf(|z|) = 1 − ((((a5 t + a4)t + a3)t + a2)t + a1)·t·exp(−z²)
//! ```
//!
//! 这个式子**本身**的最大绝对误差是 1.5e-7（A&S 给出的界，本轮用 glibc 的 `erf` 复测：
//! 折算到 gelu 输出上是 2.1e-7，出现在 `|x| ≈ 3.08`）。
//!
//! # 位精确构造
//!
//! 逐元素、无归约。两个"写成哪种等价形式"的决定：
//!
//! 1. **不带 `copysign` 的写法**：`y = 0.5·x + 0.5·|x|·erf(|x|/√2)`。它与
//!    `0.5x(1 + sign(x)·erf(|x|/√2))` 数学等价，但省掉了"按符号选 lane"（本机 stdarch 既没有
//!    浮点 `copysign` 也没有按位 `and`/`xor`，只能靠比较 + 混合，更贵）；代价是一次 `|x|`
//!    ——`abs` 用 `max(x, −x)` 实现（见 `arch::lasx::abs_f32x8`）。
//! 2. **`exp` 与门控分母复用 [`crate::ops::nn_math`]**，与 `silu`/`gelu_quick`/`softmax` 同一串。
//!
//! 列尾是标量路径，用与向量同式的 [`gelu_erf_seq`]。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（策略见 `docs/dev.md` §17）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]
use crate::arch::lasx;
use crate::ops::nn_math::{exp_seq, exp_vec};
use std::arch::loongarch64::*;

/// `1/(1+p·|z|)` 里的 `p`。
pub(crate) const ERF_P: f32 = 0.327_591_1;
/// A&S 7.1.26 的五个系数 `a1..a5`（`a2`/`a4` 是负的）。
///
/// 写的是**最短往返十进制**（`clippy::excessive_precision` 要求），不是 A&S 原文的 9 位：
/// `0.254829592 / −0.284496736 / 1.421413741 / −1.453152027 / 1.061405429`。
/// 两者转成 f32 后**位型相同**（逐条核对过 bit pattern），精度没有任何损失。
/// （我第一次凭感觉缩短，`a2`/`a3` 各差 1 ulp——所以这里不是"看着差不多"就写上了。）
pub(crate) const ERF_COEF: [f32; 5] = [
    0.254_829_6,
    -0.284_496_72,
    1.421_413_8,
    -1.453_152_1,
    1.061_405_4,
];

/// `1/√2`（`z = |x|/√2`；用库常量，避免自己写十进制导致 f32 位型不同）。
const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// `gelu_erf` 的标量式（规范版；标量尾与测试里的"标量模拟"都照抄这一串）。
#[inline]
pub(crate) fn gelu_erf_seq(x: f32) -> f32 {
    let ax = x.abs();
    let z = ax * INV_SQRT2;
    let t = 1.0 / ERF_P.mul_add(z, 1.0);
    let mut p = ERF_COEF[4];
    for k in (0..4).rev() {
        p = p.mul_add(t, ERF_COEF[k]);
    }
    let poly = p * t;
    let erf_a = 1.0 - poly * exp_seq(-(z * z));
    // 0.5x + 0.5|x|·erf(|x|/√2)：单次 FMA 收尾
    (0.5 * ax).mul_add(erf_a, 0.5 * x)
}

/// erf 的近似的向量版（与 [`gelu_erf_seq`] 同一串 op）。
#[inline]
fn gelu_erf_vec(vx: lasx::F32x8) -> lasx::F32x8 {
    // SAFETY: 全为寄存器操作，不碰内存。
    unsafe {
        let vax = lasx::abs_f32x8(vx);
        let z = lasx_xvfmul_s(vax, lasx::splat_f32(INV_SQRT2));
        // t = 1/(1 + p·z)
        let den = lasx_xvfmadd_s(lasx::splat_f32(ERF_P), z, lasx::splat_f32(1.0));
        let t = lasx_xvfdiv_s(lasx::splat_f32(1.0), den);
        // Horner：a5 → a1
        let mut p = lasx::splat_f32(ERF_COEF[4]);
        for k in (0..4).rev() {
            p = lasx_xvfmadd_s(p, t, lasx::splat_f32(ERF_COEF[k]));
        }
        let poly = lasx_xvfmul_s(p, t);
        // exp(−z²)：z ≥ 0，先乘再取负（取负是精确的）
        let e = exp_vec(lasx::neg_f32x8(lasx_xvfmul_s(z, z)));
        let erf_a = lasx_xvfsub_s(lasx::splat_f32(1.0), lasx_xvfmul_s(poly, e));
        let half_x = lasx_xvfmul_s(vx, lasx::splat_f32(0.5));
        let half_ax = lasx_xvfmul_s(vax, lasx::splat_f32(0.5));
        lasx_xvfmadd_s(half_ax, erf_a, half_x)
    }
}

/// 就地逐元素 `gelu_erf`：`out[..n] = gelu_erf(x[..n])`。
///
/// `x` 与 `out` 长度相同（`n = x.len()`；调用方负责相等与对齐）。
///
/// # Safety
/// 无额外前提：`x`/`out` 长度相等，循环只在 `i + 8 <= n` 时载入 8 个 `f32`，
/// 标量尾只在 `i < n` 时下标访问。
pub(crate) fn gelu_erf_f32(x: &[f32], out: &mut [f32]) {
    let n = x.len();
    debug_assert_eq!(n, out.len(), "gelu_erf: 输入输出长度必须相等");
    let mut i = 0;
    while i + 8 <= n {
        let vx = unsafe { lasx::load_f32x8(x.as_ptr().add(i)) };
        let vy = gelu_erf_vec(vx);
        unsafe { lasx::store_f32x8(out.as_mut_ptr().add(i), vy) };
        i += 8;
    }
    for j in i..n {
        // 与向量路径同式，保证任一分块逐位一致
        out[j] = gelu_erf_seq(x[j]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::nn_math::exp_seq;

    /// 标量模拟：**独立**写出同一串 op（不调用 `gelu_erf_seq`）。
    ///
    /// 这里**必须**显式写 `mul_add`：FMA 与"先乘后加"差一次舍入，写错就是 1 ulp 的分歧
    /// （第一版忘了，`x = −3` 立刻被这条测试抓到——这正是它存在的意义）。
    fn emul(x: f32) -> f32 {
        let ax = if x < 0.0 { -x } else { x };
        let z = ax * INV_SQRT2;
        let t = 1.0 / ERF_P.mul_add(z, 1.0);
        let mut p = ERF_COEF[4];
        for k in (0..4).rev() {
            p = p.mul_add(t, ERF_COEF[k]);
        }
        let poly = p * t;
        let erf_a = 1.0 - poly * exp_seq(-(z * z));
        (0.5 * ax).mul_add(erf_a, 0.5 * x)
    }

    /// f64 参考：`0.5x(1+erf(x/√2))`。`erf` 用**高精度**实现（不能用同一个 A&S 式子，
    /// 否则量到的是 f32 舍入而不是近似误差）：
    /// `erf(z) = sign(z)·(1 − erfc(|z|))`，`erfc` 用连分式展开。
    fn erf_f64(z: f64) -> f64 {
        if z == 0.0 {
            return 0.0;
        }
        let a = z.abs();
        // 小 |z| 用泰勒级数（收敛快），大 |z| 用连分式（避免相消）
        let v = if a < 2.0 {
            let mut term = a;
            let mut sum = a;
            let mut n = 1.0f64;
            let a2 = a * a;
            loop {
                term *= -a2 / n;
                let add = term / (2.0 * n + 1.0);
                sum += add;
                if add.abs() < 1e-18 * sum.abs() {
                    break;
                }
                n += 1.0;
            }
            2.0 / std::f64::consts::PI.sqrt() * sum
        } else {
            // erfc(a) = exp(−a²)/(a√π) · 1/(1 + 1/(2a²)/(1 + 2/(2a²)/(1 + …)))
            let mut cf = 0.0f64;
            for k in (1..=60).rev() {
                cf = (k as f64 / 2.0) / (a * a) / (1.0 + cf);
            }
            let erfc = (-a * a).exp() / (a * std::f64::consts::PI.sqrt()) / (1.0 + cf);
            1.0 - erfc
        };
        if z < 0.0 {
            -v
        } else {
            v
        }
    }

    fn reference(x: f64) -> f64 {
        0.5 * x * (1.0 + erf_f64(x * std::f64::consts::FRAC_1_SQRT_2))
    }

    fn run(x: &[f32]) -> Vec<f32> {
        let mut out = vec![0f32; x.len()];
        gelu_erf_f32(x, &mut out);
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

    /// 边界与饱和：这些是**实测出来的确定值**，不是"应该差不多"。
    #[test]
    fn test_exact_cases() {
        assert_eq!(gelu_erf_seq(0.0).to_bits(), 0.0f32.to_bits());
        // 大负侧：erf_a 恰好舍入成 1.0 ⇒ 1 + (−1) = 0 ⇒ 输出**精确 0**
        assert_eq!(gelu_erf_seq(-6.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(gelu_erf_seq(-10.0).to_bits(), 0.0f32.to_bits());
        // `−inf` 是**未定式**（`−inf·0`），本实现给出 NaN——与 PyTorch 的 `gelu(−inf)`
        // 行为一致（它也是同一个代数式）。契约只覆盖有限输入，所以这里只钉住"不发生漂移"。
        assert!(gelu_erf_seq(f32::NEG_INFINITY).is_nan());
        // 大正侧：erf_a = 1 ⇒ y = 0.5x + 0.5|x| = x（位精确的恒等）
        for x in [6.0f32, 7.5, 10.0, 1e30] {
            assert_eq!(gelu_erf_seq(x), x, "x={x}");
        }
        assert_eq!(gelu_erf_seq(f32::INFINITY), f32::INFINITY);
        // 负侧尾部：真值从 `0⁻` 一侧趋于 0，而 `erf_a` 在 x ≈ −5.6 处**舍入到 1.0**
        // 给出精确 0，再往右一点又给出小负数（x = −5 时真值就是 −1.5e-6）⇒ f32 下这里
        // **不单调**。这不是 bug，是"绝对误差 1.5e-7 的 erf 近似"能给出的全部；
        // 关键在于每一点与 f64 参考的差仍在近似界内。
        assert_eq!(
            gelu_erf_seq(-6.0),
            0.0,
            "x ≤ −5.6 时 erf_a 舍入到 1 ⇒ 精确 0"
        );
        assert!(
            gelu_erf_seq(-5.5) < 0.0,
            "再往右一点又变成小负数 ⇒ 尾部不单调"
        );
        for x in [-20.0f32, -10.0, -6.0, -5.5, -5.0, -4.0, -2.0, -1.0] {
            let e = (gelu_erf_seq(x) as f64 - reference(x as f64)).abs();
            assert!(e < 2.5e-7, "x={x} 与 f64 参考差 {e:.3e}");
        }
        // 正侧（x ≥ 0）单调增：erf_a 与 ax 同时增大，y 必增
        let mut prev = 0.0f32;
        let mut x = 0.0f32;
        while x <= 8.0 {
            let y = gelu_erf_seq(x);
            assert!(y >= prev, "x={x} 处不单调：{prev} → {y}");
            prev = y;
            x += 0.01;
        }
    }

    /// 数值：与**高精度** f64 参考的误差（实测包络，见 `docs/ops.md` §2.9）。
    ///
    /// 误差由两部分构成：A&S 7.1.26 本身的 ~2.1e-7（折算到 gelu 输出）+ f32 舍入。
    /// `|x|` 很大时则是**输入量化**主导（`x` 只有 24 位有效位），所以按区间分别卡。
    #[test]
    fn test_accuracy_vs_f64_reference() {
        let worst = |lo: f32, hi: f32, bound: f64| {
            let (mut w, mut at) = (0f64, 0f32);
            let mut x = lo;
            while x <= hi {
                let want = reference(x as f64);
                let got = gelu_erf_seq(x) as f64;
                let e = (got - want).abs();
                if e > w {
                    (w, at) = (e, x);
                }
                x += 0.0005;
            }
            assert!(
                w < bound,
                "|x| ≤ {} 最差绝对误差 {w:.3e}（x = {at}）",
                hi.abs()
            );
        };
        worst(-1.0, 1.0, 6e-7);
        worst(-4.0, 4.0, 8e-7);
        worst(-16.0, 16.0, 1e-6);
        worst(-40.0, 40.0, 6e-6);
    }
}
