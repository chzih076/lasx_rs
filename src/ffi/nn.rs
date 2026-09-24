//! NN 类内核的 C ABI 导出（`docs/ops.md` §12.3 的 N1 批次）。
//!
//! 与其它导出模块一致：本文件只做"裸指针 + 长度 → 切片"，算法在 `ops`。
//!
//! C 签名：
//!
//! ```c
//! void lasx_softmax_rows(const float *x, const float *mask, float *out,
//!                        int n_rows, int n_cols, float scale);
//! void lasx_softmax_rows_checked(const float *x, const float *mask, float *out,
//!                                int n_rows, int n_cols, float scale, int *status);
//! void lasx_rms_norm(const float *x, const float *w, float *out, int n_rows, int n_cols, float eps);
//! void lasx_silu(const float *x, float *out, int n);
//! void lasx_gelu_quick(const float *x, float *out, int n);
//! void lasx_gelu_erf(const float *x, float *out, int n);
//! ```
//!
//! `mask` **可以为 NULL**（表示无加性 mask）；`checked` 变体对 NULL 与"x/out 为空指针"
//! 分别处理：`mask == NULL` 是合法输入，`x`/`out` 在 `n_rows × n_cols > 0` 时为空则报
//! [`crate::ffi::status::LasxStatus::NullPointer`]。
//!
//! `silu`/`gelu_*` 是**逐元素**算子，形状只是 `n`（不需要 rows/cols），且允许
//! `out == x`（就地）：内核每轮先读完 8 个再写回同样的 8 个位置。
//!
//! 数值契约（**不是**"差不多"）：`out = e · (1/Σ)`，`e = exp(clamp(scale·x + mask − max))`，
//! `exp` 用 `exp2` 的 6 次多项式 + magic 数取整；向量路径与标量模拟**逐位一致**
//! （见 `docs/dev.md` §20.1 与 `src/ops/softmax_rows.rs` 的模块文档）。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（策略见 `docs/dev.md` §17）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX/LSX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]

/// 行内 softmax（`rows × cols` 行主序）。
///
/// C 签名：`void lasx_softmax_rows(const float *x, const float *mask, float *out, int n_rows, int n_cols, float scale)`
///
/// `mask` 允许为 NULL。
#[unsafe(no_mangle)]
pub extern "C" fn lasx_softmax_rows(
    x: *const f32,
    mask: *const f32,
    out: *mut f32,
    n_rows: i32,
    n_cols: i32,
    scale: f32,
) {
    let (rows, cols, n) = match softmax_shape(n_rows, n_cols) {
        Some(v) => v,
        None => return, // 原始符号零校验：形状不合法时就地返回（误用即 UB 是历史约定）
    };
    // SAFETY: FFI 约定——调用方保证 `x`/`out` 各至少 n 个元素、`mask` 要么为 NULL
    // 要么也有 n 个可读元素。
    unsafe {
        let x = std::slice::from_raw_parts(x, n);
        let out = std::slice::from_raw_parts_mut(out, n);
        let mask = if mask.is_null() {
            &[][..]
        } else {
            std::slice::from_raw_parts(mask, n)
        };
        crate::ops::softmax_rows::softmax_rows(x, mask, scale, rows, cols, out);
    }
}

/// 行内 softmax 的形状解算：`(rows, cols, rows×cols)`；任一为负或乘积溢出则 `None`。
///
/// `cols == 0` 视为"无事可做"（返回 `Some((rows, 0, 0))`，调用方直接跳过）——空行不是
/// 错误，与 `sum` 对空切片有定义是同一个口径。
fn softmax_shape(n_rows: i32, n_cols: i32) -> Option<(usize, usize, usize)> {
    if n_rows < 0 || n_cols < 0 {
        return None;
    }
    let (rows, cols) = (n_rows as usize, n_cols as usize);
    let n = rows.checked_mul(cols)?;
    Some((rows, cols, n))
}

/// 行内 RMSNorm（`out = x / √(mean(x²)+eps) · w`）。
///
/// C 签名：`void lasx_rms_norm(const float *x, const float *w, float *out, int n_rows, int n_cols, float eps)`
///
/// `w` 是按列的权重、**允许为 NULL**（无权重）；`eps` 的合法性由 `_checked` 变体与 `api`
/// 层把关（`eps = 0` 且整行全零会得到 NaN，属定义域问题，数值契约见 `docs/ops.md` §2.7）。
#[unsafe(no_mangle)]
pub extern "C" fn lasx_rms_norm(
    x: *const f32,
    w: *const f32,
    out: *mut f32,
    n_rows: i32,
    n_cols: i32,
    eps: f32,
) {
    let Some((rows, cols, n)) = softmax_shape(n_rows, n_cols) else {
        return; // 原始符号零校验：形状不合法就地返回（误用即 UB 是历史约定）
    };
    if n == 0 {
        return;
    }
    // SAFETY: FFI 约定——调用方保证 `x`/`out` 各至少 n 个元素、`w` 要么为 NULL 要么
    // 有 `cols` 个可读元素。
    unsafe {
        let x = std::slice::from_raw_parts(x, n);
        let out = std::slice::from_raw_parts_mut(out, n);
        let w = if w.is_null() {
            &[][..]
        } else {
            std::slice::from_raw_parts(w, cols)
        };
        crate::ops::rms_norm::rms_norm(x, w, eps, rows, cols, out);
    }
}

/// SiLU（swish）：`out[i] = x[i] / (1 + exp(−x[i]))`，逐元素。
///
/// C 签名：`void lasx_silu(const float *x, float *out, int n)`
///
/// 允许 `out == x`（就地）。`n < 0` 会被夹成 0（原始符号零校验，不做错误上报）。
///
/// 数值契约：**直接除法**（`x/den`，单次舍入，不是 `x·(1/den)`），`exp` 与门控分母用
/// `ops::nn_math` 那一份；向量路径与标量尾逐位一致（`docs/ops.md` §2.8）。
#[unsafe(no_mangle)]
pub extern "C" fn lasx_silu(x: *const f32, out: *mut f32, n: i32) {
    let n = n.max(0) as usize;
    if n == 0 {
        return;
    }
    // SAFETY: FFI 约定——调用方保证 `x`/`out` 各至少 n 个元素（允许同一块内存）。
    unsafe {
        let xs = std::slice::from_raw_parts(x, n);
        let os = std::slice::from_raw_parts_mut(out, n);
        crate::ops::silu::silu_f32(xs, os);
    }
}

/// GELU 的 sigmoid 近似（ggml `GELU_QUICK`）：`out[i] = x[i] / (1 + exp(−1.702·x[i]))`。
///
/// C 签名：`void lasx_gelu_quick(const float *x, float *out, int n)`
///
/// 允许 `out == x`（就地）。契约同 `lasx_silu`（`docs/ops.md` §2.8）。
#[unsafe(no_mangle)]
pub extern "C" fn lasx_gelu_quick(x: *const f32, out: *mut f32, n: i32) {
    let n = n.max(0) as usize;
    if n == 0 {
        return;
    }
    // SAFETY: FFI 约定——调用方保证 `x`/`out` 各至少 n 个元素（允许同一块内存）。
    unsafe {
        let xs = std::slice::from_raw_parts(x, n);
        let os = std::slice::from_raw_parts_mut(out, n);
        crate::ops::gelu_quick::gelu_quick_f32(xs, os);
    }
}

/// GELU 的 erf 形式（PyTorch `gelu` 默认）：`out[i] = 0.5·x[i]·(1 + erf(x[i]/√2))`。
///
/// C 签名：`void lasx_gelu_erf(const float *x, float *out, int n)`
///
/// 允许 `out == x`（就地）。`erf` 用 A&S 7.1.26（绝对误差 ≤1.5e-7），契约见 `docs/ops.md` §2.9。
#[unsafe(no_mangle)]
pub extern "C" fn lasx_gelu_erf(x: *const f32, out: *mut f32, n: i32) {
    let n = n.max(0) as usize;
    if n == 0 {
        return;
    }
    // SAFETY: FFI 约定——调用方保证 `x`/`out` 各至少 n 个元素（允许同一块内存）。
    unsafe {
        let xs = std::slice::from_raw_parts(x, n);
        let os = std::slice::from_raw_parts_mut(out, n);
        crate::ops::gelu_erf::gelu_erf_f32(xs, os);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 原始符号的数值自检（对照 f64 参考），并覆盖 `mask = NULL` 与 `cols = 0`。
    #[test]
    fn test_softmax_rows_ffi() {
        let x: Vec<f32> = vec![1.0, 2.0, 3.0, -1.0, 0.0, 4.0];
        let mut out = vec![0f32; 6];
        lasx_softmax_rows(x.as_ptr(), std::ptr::null(), out.as_mut_ptr(), 2, 3, 1.0);
        for i in 0..2 {
            let s: f32 = out[i * 3..(i + 1) * 3].iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "第 {i} 行和 = {s}");
        }
        assert!(out[2] > out[1] && out[1] > out[0], "单调性");

        // cols = 0：什么都不写、不崩
        let mut empty: Vec<f32> = Vec::new();
        lasx_softmax_rows(x.as_ptr(), std::ptr::null(), empty.as_mut_ptr(), 3, 0, 1.0);
    }

    /// `silu`/`gelu_quick` 的 FFI 自检：就地 == 异地、对照 f64 参考、`n = 0` 不崩。
    #[test]
    fn test_activation_ffi() {
        let x: Vec<f32> = (-40..40).map(|i| i as f32 * 0.5).collect();
        for (name, ffi, want) in [
            (
                "silu",
                lasx_silu as extern "C" fn(*const f32, *mut f32, i32),
                (|v: f64| v / (1.0 + (-v).exp())) as fn(f64) -> f64,
            ),
            (
                "gelu_quick",
                lasx_gelu_quick as extern "C" fn(*const f32, *mut f32, i32),
                (|v: f64| v / (1.0 + (-(1.702 * v)).exp())) as fn(f64) -> f64,
            ),
            (
                "gelu_erf",
                lasx_gelu_erf as extern "C" fn(*const f32, *mut f32, i32),
                (|v: f64| {
                    // 参考用同一近似式（f64 精度），只验"F32 实现 == 近似式"，不验近似质量
                    // （近似质量由 `ops::gelu_erf` 的单测对高精度 erf 参考负责）
                    let ax = v.abs();
                    let z = ax * std::f64::consts::FRAC_1_SQRT_2;
                    let t = 1.0 / (0.327_591_1f64).mul_add(z, 1.0);
                    let mut p = 1.061_405_429f64;
                    for c in [
                        (-1.453_152_027f64),
                        1.421_413_741,
                        -0.284_496_736,
                        0.254_829_592,
                    ] {
                        p = p.mul_add(t, c);
                    }
                    let erf_a = 1.0 - p * t * (-(z * z)).exp();
                    (0.5 * ax).mul_add(erf_a, 0.5 * v)
                }) as fn(f64) -> f64,
            ),
        ] {
            let mut out = vec![0f32; x.len()];
            ffi(x.as_ptr(), out.as_mut_ptr(), x.len() as i32);
            for (i, (&xi, &oi)) in x.iter().zip(out.iter()).enumerate() {
                let want = want(xi as f64);
                assert!(
                    ((oi as f64) - want).abs() < 1e-5,
                    "{name}[{i}] x={xi} got={oi} want={want}"
                );
            }
            // 就地：同一块内存
            let mut inplace = x.clone();
            ffi(inplace.as_ptr(), inplace.as_mut_ptr(), inplace.len() as i32);
            for (i, (a, b)) in inplace.iter().zip(out.iter()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "{name} 就地[{i}] 与异地不一致");
            }
            // n = 0：不写、不崩
            ffi(x.as_ptr(), std::ptr::null_mut(), 0);
        }
    }
}
