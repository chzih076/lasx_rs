//! `lasx_norm2_batch` —— 批量 2 分量模长 `out[i] = √(x² + y²)`（f64 SOA，task-180）。
//!
//! 语义契约（`simd_dsl` 的 `FusedPlan::Norm2BatchF64`，见 `src/contract/patterns.tsv`）：
//! `out[i] = (x[i]·x[i] + y[i]·y[i]).sqrt()`，**逐元素、无归约**。
//!
//! **等价类 = 逐位**（与 `simd_dsl` portable 解释的同一表达式/同一绑定）：
//! 硬件 `xvfsqrt_d` 与标量 `f64::sqrt` 都是 IEEE 正确舍入；`xvfmul_d`/`xvfadd_d` 与
//! 标量的乘/加同序 ⇒ 有限/±0/±inf 上 `to_bits()` 全等；NaN 的 payload 由指令决定，
//! 测试里按 `is_nan` 对拍（不进逐位声明）。
//!
//! LASX 缺失时走 LSX 128 位路径（2 f64/向量），再缺走标量尾同式。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（分层策略：内核模块在函数级统一写 SAFETY，
// 不逐块重复；策略全文见 `docs/dev.md` §17）：
#![allow(clippy::undocumented_unsafe_blocks)]
use crate::arch::SimdPath;
use crate::arch::{lasx, lsx};
use std::arch::loongarch64::*;

/// 算子入口：解析当前线程的向量路径后分派。
///
/// **调用方保证**：`xs/ys/out` 等长（适配器/adapter 侧已由 `FusedPlan::validate` 校验）；
/// 这里只做 debug 断言（不产生错误码）。
#[inline]
pub fn norm2_batch(xs: &[f64], ys: &[f64], out: &mut [f64]) {
    debug_assert_eq!(xs.len(), ys.len(), "norm2_batch：x/y 长度必须相等");
    debug_assert_eq!(xs.len(), out.len(), "norm2_batch：out 长度必须等于 n");
    match SimdPath::detect() {
        SimdPath::Lasx => norm2_batch_lasx(xs, ys, out),
        SimdPath::Lsx => norm2_batch_lsx(xs, ys, out),
    }
}

/// LASX 256 位实现（4 f64/lane，2 路展开 = 8 样本/迭代）。
#[inline]
fn norm2_batch_lasx(xs: &[f64], ys: &[f64], out: &mut [f64]) {
    let n = xs.len();
    let mut i = 0;
    // **2 路展开不改变任何数值**：每样本仍是 `x² + y²` 再开方，展开只为摊薄循环开销。
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n` ⇒ 两组访存分别落在 [i,i+4)、[i+4,i+8)，全在 `n` 之内；
        // 三个切片等长（入口 debug 断言 + 调用方校验），`out` 与输入不重叠（适配器借用检查器守）。
        unsafe {
            let (p, q) = (i, i + 4);
            let x0 = lasx::load_f64x4(xs.as_ptr().add(p));
            let y0 = lasx::load_f64x4(ys.as_ptr().add(p));
            let x1 = lasx::load_f64x4(xs.as_ptr().add(q));
            let y1 = lasx::load_f64x4(ys.as_ptr().add(q));
            let s0 = lasx_xvfadd_d(lasx_xvfmul_d(x0, x0), lasx_xvfmul_d(y0, y0));
            let s1 = lasx_xvfadd_d(lasx_xvfmul_d(x1, x1), lasx_xvfmul_d(y1, y1));
            lasx::store_f64x4(out.as_mut_ptr().add(p), lasx_xvfsqrt_d(s0));
            lasx::store_f64x4(out.as_mut_ptr().add(q), lasx_xvfsqrt_d(s1));
        }
        i += 8;
    }
    while i + 4 <= n {
        // SAFETY: `i + 4 <= n`；同上。
        unsafe {
            let vx = lasx::load_f64x4(xs.as_ptr().add(i));
            let vy = lasx::load_f64x4(ys.as_ptr().add(i));
            let sq = lasx_xvfadd_d(lasx_xvfmul_d(vx, vx), lasx_xvfmul_d(vy, vy));
            lasx::store_f64x4(out.as_mut_ptr().add(i), lasx_xvfsqrt_d(sq));
        }
        i += 4;
    }
    for j in i..n {
        // 与向量路径同式同序（x² + y²）⇒ 任一分块逐位一致
        out[j] = (xs[j] * xs[j] + ys[j] * ys[j]).sqrt();
    }
}

/// LSX 128 位实现（2 样本/向量）。
#[inline]
fn norm2_batch_lsx(xs: &[f64], ys: &[f64], out: &mut [f64]) {
    let n = xs.len();
    let mut i = 0;
    while i + 2 <= n {
        // SAFETY: `i + 2 <= n`；三个切片等长、`out` 不重叠；纯寄存器 + 已校验访存。
        unsafe {
            let vx = lsx::load_f64x2(xs.as_ptr().add(i));
            let vy = lsx::load_f64x2(ys.as_ptr().add(i));
            let sq = lsx_vfadd_d(lsx_vfmul_d(vx, vx), lsx_vfmul_d(vy, vy));
            lsx::store_f64x2(out.as_mut_ptr().add(i), lsx_vfsqrt_d(sq));
        }
        i += 2;
    }
    for j in i..n {
        out[j] = (xs[j] * xs[j] + ys[j] * ys[j]).sqrt();
    }
}

/// 数值回归：**逐位**对照独立标量参考（有限/±0/±inf；NaN 按 `is_nan` 对拍）。
#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_norm2(x: &[f64], y: &[f64]) -> Vec<f64> {
        (0..x.len())
            .map(|i| (x[i] * x[i] + y[i] * y[i]).sqrt())
            .collect()
    }

    fn check(x: &[f64], y: &[f64], label: &str) {
        let mut out = vec![0.0f64; x.len()];
        norm2_batch(x, y, &mut out);
        let want = scalar_norm2(x, y);
        for i in 0..x.len() {
            if want[i].is_nan() {
                assert!(
                    out[i].is_nan(),
                    "{label}: n={} i={i} 期望 NaN，得到 {}",
                    x.len(),
                    out[i]
                );
            } else {
                assert_eq!(
                    out[i].to_bits(),
                    want[i].to_bits(),
                    "{label}: n={} i={i} 不逐位（{} vs {}）",
                    x.len(),
                    out[i],
                    want[i]
                );
            }
        }
    }

    #[test]
    fn test_norm2_batch_bitwise_vs_scalar() {
        // 覆盖 LASX 2 路展开/单向量/标量尾与 LSX 尾：长度 0..=35
        for n in 0..=35usize {
            let x: Vec<f64> = (0..n).map(|i| (i as f64) * 0.37 - 3.0).collect();
            let y: Vec<f64> = (0..n).map(|i| (i as f64) * -0.23 + 1.5).collect();
            check(&x, &y, "regular");
        }
        // 边界值：±0、±inf、次正规、极大/极小、NaN
        let x = [
            f64::NAN,
            1.0,
            0.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MIN_POSITIVE / 2.0,
            1e-320,
            1e300,
            -1e300,
            3.0,
        ];
        let y = [
            1.0,
            f64::NAN,
            -0.0,
            0.0,
            f64::NEG_INFINITY,
            f64::INFINITY,
            1e-320,
            f64::MIN_POSITIVE / 2.0,
            1e300,
            1e300,
            4.0,
        ];
        check(&x, &y, "edge");
    }
}
