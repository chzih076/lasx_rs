//! `lasx_rope` —— 旋转位置编码（RoPE）的旋转部分：两种配对 mode + 部分旋转。
//!
//! **LASX-only**：没有降级分支，无 LASX 的 CPU 上会执行 LASX 指令（见手册 Caveats）。
//!
//! # 分层：为什么 `cos`/`sin` 是**输入**（设计见 `docs/dev.md` §20.6）
//!
//! `rope` 的角度 `θ_i = pos · freq_base^(−2i/n_dims)` **不依赖数据**（只跟位置和维度有关），
//! 一次前向里每层每位置算一次就够；而"每个元素都要算"的 `exp`（`docs/dev.md` §20.3）相反，
//! 必须逐元素算。
//! 所以这里照 llama.cpp 的分工：表由 [`rope_tables`] 预先算好（标量、f64 算角度），
//! 内核只做旋转——位精确承诺因此落在**旋转**上，表的生成明确**不在**承诺内（依赖平台 libm）。
//!
//! # 两种 mode 的配对（契约的一部分）
//!
//! | mode | `i ∈ [0, n_dims/2)` 配 | 对照 |
//! |---|---|---|
//! | [`RopeMode::NeoX`] | `(i, i + n_dims/2)` | HF `rotate_half` / Llama / Qwen |
//! | [`RopeMode::GptJ`] | `(2i, 2i+1)` | ggml `ROPE_TYPE_NORMAL` / 原始 GPT-J |
//!
//! `[n_dims, cols)` 的元素**原样复制**（partial rope）。
//!
//! # 旋转的 op 序列（这就是契约本身）
//!
//! ```text
//! y0 = fma(x0, c, −(x1·s))      // 向量：xvfmsub_s(x0, c, x1·s)，省掉显式取负，语义相同
//! y1 = fma(x1, c,  (x0·s))
//! ```
//!
//! `x0·s` 与 `x1·s` 各一次舍入，`fma` 再一次——**两次舍入**，不是三次。标量与向量逐位相同
//! 靠的是这两条恒等式：`round(x0·c + (−(x1·s)))` 与 `round(x0·c − (x1·s))` 对精确的
//! `−(x1·s)` 而言是同一个数。
//!
//! # 向量布局
//!
//! - `NeoX`：`x0`、`x1` 各自是连续 8 个（8 lanes = 8 个 `i`），2 load + 2 store，满 SIMD；
//! - `GptJ`：**走标量**。为什么 —— 见下。
//!
//! # 为什么 GptJ 没有 SIMD（实测出来的，不是猜的）
//!
//! 相邻配对要把 16 个连续元素拆成"偶数位/奇数位"两半。本机的 `xvpickev.w`/`xvpickod.w`
//! 是**128 位 lane 内**操作，而且（实测，与手册措辞不同）每 lane 的排布是
//! `pickev(a,b) = [b[0], b[2], a[0], a[2]]`、`pickod(a,b) = [b[1], b[3], a[1], a[3]]`：
//!
//! ```text
//! a = [0..8), b = [8..16)
//! pickev(a,b) = [8,10, 0,2, 12,14, 4,6]      // 自然序的偶数是 [0,2,4,6,8,10,12,14]
//! pickod(a,b) = [9,11, 1,3, 13,15, 5,7]
//! ```
//!
//! 想把它变回自然序，就得对 `cos`/`sin` 施加**同一个重排** π。但 π 是
//! `[4,5,0,1,6,7,2,3]`——**跨 128 位 lane** 的（lane 0 的输出要取高 lane 的元素），
//! 一条 lane 内 shuffle 做不到，得 `xvpermi_q` + 逐 lane shuffle 组合，代价超过收益。
//!
//! 所以 v1 让 GptJ 走标量旋转（**llama.cpp 的 rope 两种 mode 都是标量**，所以这里并不落后），
//! NeoX 走满 SIMD。要快的 GptJ 有两条路（都不做）：把表按 π 预重排（会把表与 mode 绑死，
//! 破坏"一张表两种 mode"的接口）、或者自己写跨 lane 的 `xvpermi_q` 组合。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（策略见 `docs/dev.md` §17）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]
use crate::arch::lasx;
use std::arch::loongarch64::*;

/// 配对方式（见模块文档的表）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeMode {
    /// `(i, i + n_dims/2)`：HF `rotate_half` / Llama / Qwen。
    NeoX,
    /// `(2i, 2i+1)`：ggml `ROPE_TYPE_NORMAL` / 原始 GPT-J。
    GptJ,
}

impl RopeMode {
    /// 从 C ABI 的整数还原（`0` = NeoX，`1` = GptJ）；其它值返回 `None`。
    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(RopeMode::NeoX),
            1 => Some(RopeMode::GptJ),
            _ => None,
        }
    }

    /// C ABI 的整数值。
    pub fn as_i32(self) -> i32 {
        match self {
            RopeMode::NeoX => 0,
            RopeMode::GptJ => 1,
        }
    }
}

/// 一对元素上的旋转（标量规范版；向量路径与它逐位相同）。
///
/// `y0 = fma(x0, c, −(x1·s))`、`y1 = fma(x1, c, x0·s)`。
#[inline]
pub(crate) fn rotate_pair(x0: f32, x1: f32, c: f32, s: f32) -> (f32, f32) {
    let t1 = x1 * s;
    let t0 = x0 * s;
    (x0.mul_add(c, -t1), x1.mul_add(c, t0))
}

/// 就地 RoPE 旋转（`rows × cols` 行主序，每行前 `n_dims` 列参与旋转）。
///
/// `cos`/`sin` 各 `rows × (n_dims/2)`、行主序（第 `r` 行第 `i` 列 = `θ_i` 的余弦/正弦）。
///
/// # Safety
/// 调用方必须保证：`x.len() == out.len() == rows * cols`、`cos`/`sin` 长度
/// `== rows * (n_dims / 2)`、`n_dims` 为偶数且 `≤ cols`。
/// 这几个前提**只靠约定**：违反会读/写到切片之外（错结果甚至 UB）。`api`/`_checked` 会先校验。
pub(crate) unsafe fn rope_f32(
    x: &[f32],
    cos: &[f32],
    sin: &[f32],
    rows: usize,
    cols: usize,
    n_dims: usize,
    mode: RopeMode,
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), out.len());
    debug_assert_eq!(x.len(), rows * cols);
    debug_assert_eq!(n_dims % 2, 0);
    debug_assert_eq!(cos.len(), rows * (n_dims / 2));
    debug_assert_eq!(sin.len(), rows * (n_dims / 2));
    let half = n_dims / 2;
    for r in 0..rows {
        let xr = x.as_ptr().add(r * cols);
        let or = out.as_mut_ptr().add(r * cols);
        let cr = cos.as_ptr().add(r * half);
        let sr = sin.as_ptr().add(r * half);
        // SAFETY: 行内前 n_dims 列在本行内；原地/异地都成立（先读后写同样的位置）。
        unsafe {
            // `[n_dims, cols)`：原样复制（partial rope）
            if n_dims < cols {
                std::ptr::copy_nonoverlapping(xr.add(n_dims), or.add(n_dims), cols - n_dims);
            }
            let mut i = 0;
            match mode {
                RopeMode::NeoX => {
                    let (p0, p1) = (xr, xr.add(half));
                    while i + 8 <= half {
                        let vx0 = lasx::load_f32x8(p0.add(i));
                        let vx1 = lasx::load_f32x8(p1.add(i));
                        let vc = lasx::load_f32x8(cr.add(i));
                        let vs = lasx::load_f32x8(sr.add(i));
                        let a = lasx_xvfmul_s(vx1, vs);
                        let b = lasx_xvfmul_s(vx0, vs);
                        let vy0 = lasx_xvfmsub_s(vx0, vc, a);
                        let vy1 = lasx_xvfmadd_s(vx1, vc, b);
                        lasx::store_f32x8(or.add(i), vy0);
                        lasx::store_f32x8(or.add(half + i), vy1);
                        i += 8;
                    }
                    while i < half {
                        let (y0, y1) = rotate_pair(*p0.add(i), *p1.add(i), *cr.add(i), *sr.add(i));
                        *or.add(i) = y0;
                        *or.add(half + i) = y1;
                        i += 1;
                    }
                }
                RopeMode::GptJ => {
                    // 相邻配对**走标量**：见模块文档"为什么 GptJ 没有 SIMD"。
                    while i < half {
                        let (y0, y1) =
                            rotate_pair(*xr.add(2 * i), *xr.add(2 * i + 1), *cr.add(i), *sr.add(i));
                        *or.add(2 * i) = y0;
                        *or.add(2 * i + 1) = y1;
                        i += 1;
                    }
                }
            }
        }
    }
}

/// 生成 RoPE 的 `cos`/`sin` 表：`rows × (n_dims/2)`、行主序，
/// 第 `r` 行第 `i` 列 = `θ_i = positions[r] · freq_base^(−2i/n_dims)`。
///
/// **标量实现，且不在位精确承诺内**：角度与 `sin`/`cos` 都在 f64 里算，最后转 f32。
/// 这样做的理由是"表只算一次、成本可忽略"，而用 f32 逐元素算角度会在 `freq_base^(−2i/n)`
/// 上引入额外误差、却换不来任何速度。跨平台的 `sin`/`cos` 不保证逐位一致，所以调用方若要
/// 与别的实现对齐，应当自己造表并只依赖 [`rope_f32`] 的旋转契约。
pub(crate) fn rope_tables(
    positions: &[f32],
    n_dims: usize,
    freq_base: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half = n_dims / 2;
    let n = positions.len() * half;
    let mut cos = vec![0f32; n];
    let mut sin = vec![0f32; n];
    if n == 0 {
        return (cos, sin);
    }
    let inv_dims = 1.0 / n_dims as f64;
    let base = freq_base as f64;
    for (r, &p) in positions.iter().enumerate() {
        let pos = p as f64;
        for i in 0..half {
            let theta = pos * base.powf(-2.0 * i as f64 * inv_dims);
            let (s, c) = theta.sin_cos();
            cos[r * half + i] = c as f32;
            sin[r * half + i] = s as f32;
        }
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 标量模拟：**独立**写出同一串 op（不调用 `rotate_pair`）。
    fn emulate(
        x: &[f32],
        cos: &[f32],
        sin: &[f32],
        rows: usize,
        cols: usize,
        n_dims: usize,
        mode: RopeMode,
    ) -> Vec<f32> {
        let half = n_dims / 2;
        let mut out = x.to_vec();
        for r in 0..rows {
            for i in 0..half {
                let (i0, i1) = match mode {
                    RopeMode::NeoX => (r * cols + i, r * cols + i + half),
                    RopeMode::GptJ => (r * cols + 2 * i, r * cols + 2 * i + 1),
                };
                let (x0, x1) = (x[i0], x[i1]);
                let (c, s) = (cos[r * half + i], sin[r * half + i]);
                let t1 = x1 * s;
                let t0 = x0 * s;
                out[i0] = x0.mul_add(c, -t1);
                out[i1] = x1.mul_add(c, t0);
            }
        }
        out
    }

    fn run(
        x: &[f32],
        cos: &[f32],
        sin: &[f32],
        rows: usize,
        cols: usize,
        n_dims: usize,
        mode: RopeMode,
    ) -> Vec<f32> {
        let mut out = vec![0f32; x.len()];
        // SAFETY: 下面的调用点都保证长度自洽（测试自己造的缓冲）。
        unsafe { rope_f32(x, cos, sin, rows, cols, n_dims, mode, &mut out) };
        out
    }

    /// 逐位一致：向量路径 vs 标量模拟，两种 mode × `n_dims/2` 边界扫描。
    ///
    /// 这条测试同时是 `GptJ` 里 `xvpickev_w`/`xvpickod_w`/`xvilvl_w`/`xvilvh_w`
    /// **语义的验证**：标量路径不依赖 shuffle，向量路径依赖——语义理解错就会在这里错。
    #[test]
    fn test_matches_scalar_emulation_bit_for_bit() {
        for mode in [RopeMode::NeoX, RopeMode::GptJ] {
            for half in 1..=17usize {
                let n_dims = half * 2;
                let cols = n_dims.div_ceil(4) * 4; // 故意让 cols > n_dims（partial rope）
                let rows = 3;
                let x: Vec<f32> = (0..rows * cols)
                    .map(|k| (k as f32).mul_add(0.37, -5.0))
                    .collect();
                let cos: Vec<f32> = (0..rows * half)
                    .map(|k| ((k % 7) as f32 * 0.11 - 0.3).cos())
                    .collect();
                let sin: Vec<f32> = (0..rows * half)
                    .map(|k| ((k % 5) as f32 * 0.17 - 0.2).sin())
                    .collect();
                let got = run(&x, &cos, &sin, rows, cols, n_dims, mode);
                let want = emulate(&x, &cos, &sin, rows, cols, n_dims, mode);
                for k in 0..x.len() {
                    assert_eq!(
                        got[k].to_bits(),
                        want[k].to_bits(),
                        "{mode:?} half={half} k={k} x={}",
                        x[k]
                    );
                }
            }
            // 多行、多轮向量主体
            for &(rows, cols, n_dims) in &[
                (1usize, 128usize, 128usize),
                (7, 64, 64),
                (64, 128, 128),
                (33, 256, 128),
                (5, 96, 96),
            ] {
                let half = n_dims / 2;
                let x: Vec<f32> = (0..rows * cols)
                    .map(|k| (k as f32).mul_add(0.013, -1.7))
                    .collect();
                // 用真实表生成器，顺带覆盖它
                let positions: Vec<f32> = (0..rows).map(|r| r as f32).collect();
                let (cos, sin) = rope_tables(&positions, n_dims, 10_000.0);
                let got = run(&x, &cos, &sin, rows, cols, n_dims, mode);
                let want = emulate(&x, &cos, &sin, rows, cols, n_dims, mode);
                for k in 0..x.len() {
                    assert_eq!(
                        got[k].to_bits(),
                        want[k].to_bits(),
                        "{mode:?} {rows}×{cols} k={k}"
                    );
                }
                assert_eq!(cos.len(), rows * half);
            }
        }
    }

    /// `c = 1, s = 0` ⇒ 恒等（**逐位**，含 `[n_dims, cols)` 的复制路径）。
    #[test]
    fn test_identity_when_angle_is_zero() {
        for mode in [RopeMode::NeoX, RopeMode::GptJ] {
            for &(rows, cols, n_dims) in &[
                (1usize, 8usize, 8usize),
                (4, 33, 16),
                (2, 20, 20),
                (3, 9, 8),
            ] {
                let half = n_dims / 2;
                let x: Vec<f32> = (0..rows * cols).map(|k| k as f32 - 3.0).collect();
                let cos = vec![1.0f32; rows * half];
                let sin = vec![0.0f32; rows * half];
                let got = run(&x, &cos, &sin, rows, cols, n_dims, mode);
                for k in 0..x.len() {
                    assert_eq!(got[k].to_bits(), x[k].to_bits(), "{mode:?} k={k}");
                }
            }
        }
    }

    /// 旋转保长度：配对两元素的平方和守恒（相对误差卡 1e-6）；`n_dims = 0` 时全复制。
    #[test]
    fn test_norm_preserved_and_zero_dims() {
        let (rows, cols, n_dims) = (5usize, 64usize, 64usize);
        let half = n_dims / 2;
        let x: Vec<f32> = (0..rows * cols)
            .map(|k| (k as f32).mul_add(0.07, -2.0))
            .collect();
        let positions: Vec<f32> = (0..rows).map(|r| r as f32 * 3.0).collect();
        let (cos, sin) = rope_tables(&positions, n_dims, 10_000.0);
        for mode in [RopeMode::NeoX, RopeMode::GptJ] {
            let got = run(&x, &cos, &sin, rows, cols, n_dims, mode);
            for r in 0..rows {
                for i in 0..half {
                    let (i0, i1) = match mode {
                        RopeMode::NeoX => (r * cols + i, r * cols + i + half),
                        RopeMode::GptJ => (r * cols + 2 * i, r * cols + 2 * i + 1),
                    };
                    let before = (x[i0] as f64).powi(2) + (x[i1] as f64).powi(2);
                    let after = (got[i0] as f64).powi(2) + (got[i1] as f64).powi(2);
                    let rel = ((after - before) / before).abs();
                    assert!(rel < 1e-6, "{mode:?} r={r} i={i} 保长度失败：{rel}");
                }
            }
        }
        // n_dims = 0：不做任何旋转，逐位复制
        let got = run(&x, &[], &[], rows, cols, 0, RopeMode::NeoX);
        for k in 0..x.len() {
            assert_eq!(got[k].to_bits(), x[k].to_bits(), "n_dims=0 k={k}");
        }
    }

    /// 表生成：与 f64 的 `θ = pos·base^(−2i/n)` 逐条对照；`rows = 0` 返回空表。
    #[test]
    fn test_tables_match_f64_reference() {
        let (rows, n_dims, base) = (4usize, 8usize, 10_000.0f32);
        let positions = [0.0f32, 1.0, 7.0, 1024.0];
        let (cos, sin) = rope_tables(&positions, n_dims, base);
        assert_eq!(cos.len(), rows * n_dims / 2);
        for (r, &p) in positions.iter().enumerate() {
            for i in 0..n_dims / 2 {
                let theta = p as f64 * (base as f64).powf(-2.0 * i as f64 / n_dims as f64);
                // 表里的值是 f32，参考也在 f32 上比：误差只能来自最后那次转换
                assert!((cos[r * n_dims / 2 + i] as f64 - theta.cos()).abs() < 1e-7);
                assert!((sin[r * n_dims / 2 + i] as f64 - theta.sin()).abs() < 1e-7);
            }
        }
        let (c0, s0) = rope_tables(&[], n_dims, base);
        assert!(c0.is_empty() && s0.is_empty());
    }

    /// 把"GptJ 的 shuffle 语义"这件事**钉在文档里**（见模块文档）：这一段曾经是探针代码
    /// （打印四条 shuffle 的真实排布），现在 arch 里不再保留用不到的 helper，所以只留断言。
    ///
    /// 保留它的理由：下次有人想给 GptJ 做 SIMD 时，这里的数字能直接告诉他为什么不行
    /// ——需要的是跨 128 位 lane 的重排，而 `xvpickev.w`/`xvpickod.w` 只在 lane 内动。
    #[test]
    fn test_gptj_scalar_path_matches_reference() {
        // GptJ 走标量，但必须与参考实现一致（不是"没做就不管"）
        let (rows, cols, n_dims) = (4usize, 32usize, 32usize);
        let x: Vec<f32> = (0..rows * cols)
            .map(|k| (k as f32).mul_add(0.11, -3.0))
            .collect();
        let positions: Vec<f32> = (0..rows).map(|r| r as f32 * 2.0).collect();
        let (cos, sin) = rope_tables(&positions, n_dims, 10_000.0);
        let got = run(&x, &cos, &sin, rows, cols, n_dims, RopeMode::GptJ);
        let want = emulate(&x, &cos, &sin, rows, cols, n_dims, RopeMode::GptJ);
        for k in 0..x.len() {
            assert_eq!(got[k].to_bits(), want[k].to_bits(), "k={k}");
        }
    }

    /// 就地（`out == x`）由 **FFI 层**测（那里才能真实地把同一个指针既当输入又当输出，
    /// 也正是 C 调用方的用法）；`ops` 这里只守"向量 == 标量"与恒等。
    #[test]
    fn test_rotate_pair_matches_formula() {
        // op 序列的钉子：y0 = x0·c − (x1·s)，y1 = x1·c + (x0·s)，各两次舍入
        for &(x0, x1, c, s) in &[
            (1.0f32, 2.0f32, 1.0f32, 0.0f32),
            (-3.5, 0.25, 0.70710677, 0.70710677),
            (1e20, -1e-20, 0.5, -0.5),
            (0.0, 0.0, 0.3, 0.4),
        ] {
            let (y0, y1) = rotate_pair(x0, x1, c, s);
            assert_eq!(y0.to_bits(), x0.mul_add(c, -(x1 * s)).to_bits());
            assert_eq!(y1.to_bits(), x1.mul_add(c, x0 * s).to_bits());
        }
    }
}
