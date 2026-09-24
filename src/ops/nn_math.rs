//! NN 侧共用的数值件：**`exp` 的 op 序列**（契约见 `docs/ops.md` §2.6）。
//!
//! 为什么单独一个模块：`softmax_rows`、`silu`、`gelu`（以及后续 `rope`）都要用同一个 `exp`，
//! 而"近似哪个多项式、怎么取整、下溢落到哪"这些决定必须**只有一份**——否则"各算子精度不同"
//! 会成为一笔说不清的账。标量与向量两条实现放在一起，改动时不可能只改一条。
//!
//! ```text
//! t = x·LOG2E                      // mul
//! n = (t + MAGIC) − MAGIC          // floor(t)，magic 数技巧（|t| < 2²² 精确）
//! f = t − n                        // 精确（n 是整数值，Sterbenz）
//! p = exp2 的 6 次 Horner 多项式(f) // 全程 FMA，|f| ≤ 0.5
//! kk = clamp(trunc(n) + 127, 0, 254)
//! exp(x) = p · bitcast_f32(kk << 23)
//! ```
//!
//! `kk` 的下界取 **0**（不是 1）：`x < −87.6831` 时指数位为 0 ⇒ `from_bits(0) = +0.0`，
//! 于是下溢恰好落到**精确 0**（这条是实测定的：夹到 −87 时 `exp(−87) ≈ 1e-38` 仍是正规数，
//! 分母不会归零；见 `docs/dev.md` §20.1）。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（策略见 `docs/dev.md` §17）：
// 这里的 unsafe 都是"纯寄存器操作 / 在刚校验过长度的切片上调用 intrinsic"，
// 同一组前提在**函数级 SAFETY 段**里统一说明。
#![allow(clippy::undocumented_unsafe_blocks)]
use crate::arch::lasx;
use std::arch::loongarch64::*;

/// `log2(e)`。
pub(crate) const LOG2E: f32 = std::f32::consts::LOG2_E;
/// magic 数（`1.5·2²³`）：`floor(t) = (t + MAGIC) − MAGIC`。
pub(crate) const MAGIC: f32 = 12_582_912.0;
/// `exp` 的输入下界夹取（−104 落在"下溢到精确 0"的阈值 −87.6831 之内，见模块文档）。
pub(crate) const EXP_CLAMP_LO: f32 = -104.0;
/// `exp` 的输入上界夹取（`exp(88) ≈ 1.65e38`，仍是有限值）。
///
/// 上界不是"精度"需要，而是**安全性**需要：`|t| = |x·log2e|` 一旦接近 `2²²`，magic 数
/// 取整技巧失效，`n as i32` 就可能落到未定义区间。夹到 88 之后 `|t| ≤ 127`，序列全程
/// 有界。（下界同理，见模块文档；夹取本身是 lane-wise 精确操作，不破坏位精确。）
pub(crate) const EXP_CLAMP_HI: f32 = 88.0;
/// `exp2` 的多项式系数 `(ln2)^k / k!`（`k = 0..=6`）。
pub(crate) const EXP2_COEF: [f32; 7] = [
    1.0,
    std::f32::consts::LN_2, // (ln2)^1/1!
    0.240_226_5,            // (ln2)^2/2!
    0.055_504_11,           // (ln2)^3/3!
    0.009_618_129,          // (ln2)^4/4!
    0.001_333_355_8,        // (ln2)^5/5!
    0.000_154_035_3,        // (ln2)^6/6!
];

/// `exp` 的**标量**实现（规范版；也是各算子"标量模拟"测试必须照抄的那一串）。
///
/// 入口**自带夹取**：调用方不需要（也不该）再夹一次，`±inf` 因此不会走进 magic 数技巧。
/// `NaN` 进 `NaN` 出（不在契约内，但保证不产生 UB）。
#[inline]
pub(crate) fn exp_seq(x: f32) -> f32 {
    let x = x.clamp(EXP_CLAMP_LO, EXP_CLAMP_HI);
    let t = x * LOG2E;
    let n = (t + MAGIC) - MAGIC;
    let f = t - n;
    let mut p = EXP2_COEF[6];
    for k in (0..6).rev() {
        p = f.mul_add(p, EXP2_COEF[k]);
    }
    let kk = ((n as i32) + 127).clamp(0, 254) as u32;
    p * f32::from_bits(kk << 23)
}

/// `exp` 的 8 lane 向量实现（与 [`exp_seq`] 同一串操作，故逐 lane 逐位相同）。
#[inline]
pub(crate) fn exp_vec(x: lasx::F32x8) -> lasx::F32x8 {
    let c = |k: usize| lasx::splat_f32(EXP2_COEF[k]);
    // SAFETY: 全为寄存器操作，不碰内存。
    unsafe {
        let x = lasx_xvfmin_s(
            lasx_xvfmax_s(x, lasx::splat_f32(EXP_CLAMP_LO)),
            lasx::splat_f32(EXP_CLAMP_HI),
        );
        let t = lasx_xvfmul_s(x, lasx::splat_f32(LOG2E));
        let magic = lasx::splat_f32(MAGIC);
        let n = lasx_xvfsub_s(lasx_xvfadd_s(t, magic), magic);
        let f = lasx_xvfsub_s(t, n);
        let mut p = c(6);
        for k in (0..6).rev() {
            p = lasx_xvfmadd_s(f, p, c(k));
        }
        let ki = lasx::trunc_i32(n);
        let c127 = lasx_xvreplgr2vr_w(127);
        let kk = lasx_xvmax_w(
            lasx_xvmin_w(lasx_xvadd_w(ki, c127), lasx_xvreplgr2vr_w(254)),
            lasx_xvreplgr2vr_w(0),
        );
        lasx_xvfmul_s(p, lasx::pow2_from_exponent(kk))
    }
}

/// `σ(x)` 的**分母** `1 + exp(−x)`（标量）。
///
/// 为什么返回分母而不是 `σ(x)`：`silu`/`gelu` 用的是 `x / (1 + exp(−c·x))` 这个**直接除法**
/// 形式（一次舍入），把 `σ` 算完再乘回去是两次。返回分母让两个算子共用同一串 op，
/// 又不必多一次舍入——`c` 由调用方先乘进自变量（`silu` 时 `c = 1.0`，`1.0·x` 精确）。
#[inline]
pub(crate) fn sigmoid_den_seq(x: f32) -> f32 {
    1.0 + exp_seq(-x)
}

/// 同 [`sigmoid_den_seq`] 的 8 lane 向量版。
#[inline]
pub(crate) fn sigmoid_den_vec(x: lasx::F32x8) -> lasx::F32x8 {
    // SAFETY: 全为寄存器操作。
    unsafe {
        let e = exp_vec(lasx::neg_f32x8(x));
        lasx_xvfadd_s(lasx::splat_f32(1.0), e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 向量 `exp` 与标量 `exp_seq` 逐 lane 一致（这是所有算子"逐位一致"的地基）。
    ///
    /// 取样点刻意**跨出契约范围**（`±inf`、−200、+200）：夹取在两侧都生效，于是"越界输入"
    /// 也落在同一串 lane-wise 操作上，不会因为走标量/向量不同路径而分叉。
    #[test]
    fn test_exp_vec_matches_scalar_lane_by_lane() {
        for x in [
            f32::NEG_INFINITY,
            f32::INFINITY,
            -200.0,
            -120.0,
            -104.0,
            -88.1,
            -88.0,
            -10.0,
            -0.5,
            0.0,
            0.5,
            1.0,
            5.0,
            80.0,
            88.0,
            200.0,
        ] {
            let buf = [x; 8];
            // SAFETY: `buf` 有 8 个元素。
            let v = unsafe { lasx::load_f32x8(buf.as_ptr()) };
            let mut out = [0f32; 8];
            // SAFETY: `out` 有 8 个元素。
            unsafe { lasx::store_f32x8(out.as_mut_ptr(), exp_vec(v)) };
            for (lane, got) in out.iter().enumerate() {
                assert_eq!(got.to_bits(), exp_seq(x).to_bits(), "x={x} lane={lane}");
            }
        }
    }

    /// 下溢到**精确 0**（不是 1e-38）——`mask = −inf` 那类用法的正确性依据；
    /// 上溢侧被夹到有限值（`exp(88)`），于是 magic 数技巧与浮点转整型永远有界。
    ///
    /// 阈值是**实测**的：`n = rint(x·log2e)`，`kk = n + 127` 被夹到 0 ⟺ `n ≤ −127`
    /// ⟺ `x·log2e ≤ −126.5`（rint 的取整边界）⟺ `x < −126.5/log2e = −87.6831…`。
    /// 注意这**不是** `ln(2^−127) = −88.03`：那个值算的是"数学上 2^−127 何时不可表示"，
    /// 而本序列在 `kk = 0` 时给的是 `from_bits(0) = +0.0`，比可表示性提前了半个指数。
    #[test]
    fn test_underflow_is_exactly_zero_and_overflow_is_finite() {
        assert_eq!(exp_seq(-200.0), 0.0);
        assert_eq!(exp_seq(f32::NEG_INFINITY), 0.0);
        assert_eq!(exp_seq(-100.0), 0.0);
        assert_eq!(exp_seq(-87.7), 0.0);
        assert!(exp_seq(-87.68) > 0.0, "边界内侧应仍是次正规数");
        assert!(exp_seq(-87.0) > 0.0);
        // 上溢：夹到 88 之后是 exp(88) ≈ 1.65e38，有限且可复现
        let hi = exp_seq(1e30);
        assert_eq!(hi, exp_seq(f32::INFINITY));
        assert!(hi.is_finite() && hi > 1e38, "exp(88) = {hi}");
    }

    /// 数值：与 `f64::exp` 的**端到端**相对误差（实测包络，见 `docs/ops.md` §2.6）。
    ///
    /// 误差上限 ≈ `|x|·4e-8`（`x·log2e` 那一次 f32 乘法的舍入被指数放大）+ `2e-7`
    /// （Horner 链自身的 f32 舍入下限）。这不是靠"理论上应该很准"，是探针扫出来的：
    /// 在 `|x| ≤ 1 / 4 / 8 / 20 / 87` 上分别是 `2.1e-7 / 4.1e-7 / 6.2e-7 / 1.1e-6 / 3.9e-6`。
    /// 顺带被否掉的方案：Cody–Waite 两段归约（`x·HI + x·LO`）实测**更差**（`|x| ≤ 87` 时
    /// 6.4e-6），因为多出来的那次加法自己有一次舍入——所以这里保留单次乘法。
    #[test]
    fn test_accuracy_vs_libm() {
        let worst = |lo: f32, hi: f32, bound: f64| {
            let (mut w, mut at) = (0f64, 0f32);
            let mut x = lo;
            while x <= hi {
                let want = (x as f64).exp();
                if want.is_normal() {
                    let rel = ((exp_seq(x) as f64 - want) / want).abs();
                    if rel > w {
                        (w, at) = (rel, x);
                    }
                }
                x += 0.0001;
            }
            assert!(
                w < bound,
                "|x| ≤ {} 最差相对误差 {w:.3e}（x = {at}）",
                hi.abs()
            );
        };
        worst(-1.0, 1.0, 3e-7);
        worst(-8.0, 8.0, 8e-7);
        worst(-20.0, 20.0, 1.5e-6);
        worst(-87.0, 87.0, 4.5e-6);
    }

    /// 门控分母的向量/标量一致性（`silu`/`gelu` 的位精确就靠这条）。
    #[test]
    fn test_sigmoid_den_vec_matches_scalar() {
        for x in [
            f32::NEG_INFINITY,
            f32::INFINITY,
            -104.0,
            -88.0,
            -80.0,
            -1.0,
            -0.0,
            0.0,
            1.0,
            80.0,
            88.0,
            1e30,
        ] {
            let buf = [x; 8];
            // SAFETY: `buf` 有 8 个元素。
            let v = unsafe { lasx::load_f32x8(buf.as_ptr()) };
            let mut out = [0f32; 8];
            // SAFETY: `out` 有 8 个元素。
            unsafe { lasx::store_f32x8(out.as_mut_ptr(), sigmoid_den_vec(v)) };
            for (lane, got) in out.iter().enumerate() {
                assert_eq!(
                    got.to_bits(),
                    sigmoid_den_seq(x).to_bits(),
                    "x={x} lane={lane}"
                );
            }
        }
    }
}
