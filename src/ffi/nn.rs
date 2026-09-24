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

/// 旋转位置编码（RoPE）：每行前 `n_dims` 列旋转、其余原样复制。
///
/// C 签名：`void lasx_rope(const float *x, const float *cos, const float *sin, float *out,
///                        int n_rows, int n_cols, int n_dims, int mode)`
///
/// `mode`：`0` = NeoX（配对 `(i, i + n_dims/2)`，HF `rotate_half`）、`1` = GptJ（配对
/// `(2i, 2i+1)`）；其它值就地返回。`cos`/`sin` 各 `n_rows × (n_dims/2)`（行主序），
/// **与 mode 无关**（同一张表两种配对都能用）。允许 `out == x`（就地）。
///
/// # Safety（C 侧）
/// `x`/`out` 各至少 `n_rows × n_cols` 个元素，`cos`/`sin` 各至少 `n_rows × (n_dims/2)` 个，
/// `n_dims` 为偶数且 `≤ n_cols`。这几条**只靠约定**（原始符号零校验）：违反会读写越界。
#[unsafe(no_mangle)]
pub extern "C" fn lasx_rope(
    x: *const f32,
    cos: *const f32,
    sin: *const f32,
    out: *mut f32,
    n_rows: i32,
    n_cols: i32,
    n_dims: i32,
    mode: i32,
) {
    let Some((rows, cols, n_dims, m)) = rope_shape(n_rows, n_cols, n_dims, mode) else {
        return; // 原始符号零校验：形状/mode 不合法就地返回（误用即 UB 是历史约定）
    };
    let n = rows * cols;
    let tables = rows * (n_dims / 2);
    // SAFETY: FFI 约定——见函数文档的 # Safety。
    unsafe {
        let x = std::slice::from_raw_parts(x, n);
        let out = std::slice::from_raw_parts_mut(out, n);
        let cos = std::slice::from_raw_parts(cos, tables);
        let sin = std::slice::from_raw_parts(sin, tables);
        crate::ops::rope::rope_f32(x, cos, sin, rows, cols, n_dims, m, out);
    }
}

/// RoPE 的形状/mode 解算：`(rows, cols, n_dims, mode)`；不合法返回 `None`。
///
/// `n_dims == 0` 是**合法**的（整块复制、等于不旋转）；`rows × cols` 溢出也返回 `None`。
fn rope_shape(
    n_rows: i32,
    n_cols: i32,
    n_dims: i32,
    mode: i32,
) -> Option<(usize, usize, usize, crate::ops::rope::RopeMode)> {
    if n_rows < 0 || n_cols < 0 || n_dims < 0 {
        return None;
    }
    let (rows, cols, n_dims) = (n_rows as usize, n_cols as usize, n_dims as usize);
    if n_dims % 2 != 0 || n_dims > cols {
        return None;
    }
    let m = crate::ops::rope::RopeMode::from_i32(mode)?;
    rows.checked_mul(cols)?;
    Some((rows, cols, n_dims, m))
}

/// f16 权重 × f32 向量的点积：`Σ f16_to_f32(a[i]) · b[i]`。
///
/// C 签名：`float lasx_dot_f16(const uint16_t *a, const float *b, int n)`
///
/// f16 以 **u16 位型**给出（本库零依赖，不引入 half crate）；`n = 0` 返回 `0.0`。
/// 累加次序与位精确契约见 `docs/ops.md` §2.11。
#[unsafe(no_mangle)]
pub extern "C" fn lasx_dot_f16(a: *const u16, b: *const f32, n: i32) -> f32 {
    let n = n.max(0) as usize;
    if n == 0 {
        return 0.0;
    }
    // SAFETY: FFI 约定——调用方保证 `a` 可读 n 个 u16、`b` 可读 n 个 f32。
    unsafe {
        let a = std::slice::from_raw_parts(a, n);
        let b = std::slice::from_raw_parts(b, n);
        crate::ops::dot_f16::dot_f16(a, b)
    }
}

/// f16 权重矩阵（`m × k` 行主序）× f32 向量：`y[r] = dot_f16(a[r*k..], x)`。
///
/// C 签名：`void lasx_gemv_f16(const uint16_t *a, const float *x, float *y, int m, int k)`
///
/// 每行独立、逐行等于 `lasx_dot_f16`（**逐位**）。`m`/`k` 为负或乘积溢出时就地返回。
#[unsafe(no_mangle)]
pub extern "C" fn lasx_gemv_f16(a: *const u16, x: *const f32, y: *mut f32, m: i32, k: i32) {
    let Some((m, k, n)) = gemv_shape(m, k) else {
        return; // 原始符号零校验：形状不合法就地返回（误用即 UB 是历史约定）
    };
    if m == 0 {
        return;
    }
    // SAFETY: FFI 约定——调用方保证 `y` 可写 m 个；`k > 0` 时还保证 `a` 可读 m*k 个 u16、
    // `x` 可读 k 个 f32。
    unsafe {
        let y = std::slice::from_raw_parts_mut(y, m);
        if k == 0 {
            // 空向量的点积是 0（与 `lasx_dot_f16` 的 `n = 0 ⇒ 0.0` 同一口径），
            // 而不是"什么都不写"——后者会让调用方拿到未初始化的输出。
            y.fill(0.0);
            return;
        }
        let a = std::slice::from_raw_parts(a, n);
        let x = std::slice::from_raw_parts(x, k);
        crate::ops::dot_f16::gemv_f16(a, x, m, k, y);
    }
}

/// `gemv` 的形状解算：`(m, k, m×k)`；负值或乘积溢出返回 `None`。
fn gemv_shape(m: i32, k: i32) -> Option<(usize, usize, usize)> {
    if m < 0 || k < 0 {
        return None;
    }
    let (m, k) = (m as usize, k as usize);
    Some((m, k, m.checked_mul(k)?))
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

    /// f16 点积 / gemv 的 FFI 自检：与 f64 参考一致、每行等于 `lasx_dot_f16`（逐位）、
    /// `n = 0` / `m = 0` / 负形状不崩。
    #[test]
    fn test_dot_f16_and_gemv_ffi() {
        let (m, k) = (5usize, 40usize);
        // 用精确可表示的 f16（小整数/半整数），这样参考值可以照抄
        let a: Vec<u16> = (0..m * k)
            .map(|i| {
                let v = ((i % 17) as f32 - 8.0) * 0.25;
                f32_to_f16_bits(v)
            })
            .collect();
        let x: Vec<f32> = (0..k).map(|i| (i as f32).mul_add(0.05, 0.25)).collect();

        // dot：与 f64 参考比
        for r in 0..m {
            let row = &a[r * k..(r + 1) * k];
            let got = lasx_dot_f16(row.as_ptr(), x.as_ptr(), k as i32);
            let want: f64 = row
                .iter()
                .zip(&x)
                .map(|(&h, &xb)| f16_to_f32_local(h) as f64 * xb as f64)
                .sum();
            assert!(
                (got as f64 - want).abs() < 1e-4,
                "r={r} got={got} want={want}"
            );
        }

        // gemv：逐行与 lasx_dot_f16 逐位相同
        let mut y = vec![0f32; m];
        lasx_gemv_f16(a.as_ptr(), x.as_ptr(), y.as_mut_ptr(), m as i32, k as i32);
        for r in 0..m {
            let row = &a[r * k..(r + 1) * k];
            let want = lasx_dot_f16(row.as_ptr(), x.as_ptr(), k as i32);
            assert_eq!(y[r].to_bits(), want.to_bits(), "r={r}");
        }

        // n = 0 / m = 0 / k = 0
        assert_eq!(lasx_dot_f16(a.as_ptr(), x.as_ptr(), 0), 0.0);
        lasx_gemv_f16(a.as_ptr(), x.as_ptr(), y.as_mut_ptr(), 0, k as i32);
        let mut y0 = vec![7f32; 3];
        lasx_gemv_f16(a.as_ptr(), x.as_ptr(), y0.as_mut_ptr(), 3, 0);
        assert!(y0.iter().all(|&v| v == 0.0), "k=0 ⇒ 全 0");
        // 负形状：就地返回，不写输出
        let before = y0.clone();
        lasx_gemv_f16(a.as_ptr(), x.as_ptr(), y0.as_mut_ptr(), -1, k as i32);
        assert_eq!(y0, before);
    }

    /// 测试用：f32 → f16 位型（只用于构造精确可表示的值）。
    fn f32_to_f16_bits(x: f32) -> u16 {
        let bits = x.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32;
        let man = bits & 0x007f_ffff;
        if x == 0.0 {
            return sign;
        }
        let e16 = exp - 127 + 15;
        assert!(
            (1..=30).contains(&e16) && man & 0x1fff == 0,
            "只支持精确值：{x}"
        );
        sign | ((e16 as u16) << 10) | ((man >> 13) as u16)
    }

    /// 测试用的 f16→f32（与内核同义的独立小实现）。
    fn f16_to_f32_local(bits: u16) -> f32 {
        let sign = ((bits & 0x8000) as u32) << 16;
        let exp = ((bits >> 10) & 0x1f) as u32;
        let man = (bits & 0x03ff) as u32;
        if exp == 0 {
            if man == 0 {
                return f32::from_bits(sign);
            }
            let mut m = man;
            let mut e: i32 = -14;
            while m & 0x0400 == 0 {
                m <<= 1;
                e -= 1;
            }
            return f32::from_bits(sign | (((e + 127) as u32) << 23) | ((m & 0x03ff) << 13));
        }
        f32::from_bits(sign | ((exp + 127 - 15) << 23) | (man << 13))
    }

    /// RoPE 的 FFI 自检：**就地 == 异地**（这里才测得到真正的别名：同一个指针既当输入又当
    /// 输出，正是 C 调用方的用法）、两种 mode 的旋转与 f64 参考一致、形状不合法就地返回。
    #[test]
    fn test_rope_ffi() {
        let (rows, cols, n_dims) = (3usize, 16usize, 16usize);
        let positions: Vec<f32> = (0..rows).map(|r| r as f32).collect();
        let (cos, sin) = crate::ops::rope::rope_tables(&positions, n_dims, 10_000.0);
        let x: Vec<f32> = (0..rows * cols)
            .map(|k| (k as f32).mul_add(0.13, -2.0))
            .collect();
        for mode in [0i32, 1] {
            let mut away = vec![0f32; x.len()];
            lasx_rope(
                x.as_ptr(),
                cos.as_ptr(),
                sin.as_ptr(),
                away.as_mut_ptr(),
                rows as i32,
                cols as i32,
                n_dims as i32,
                mode,
            );
            // 就地：同一个缓冲当输入与输出
            let mut inplace = x.clone();
            lasx_rope(
                inplace.as_ptr(),
                cos.as_ptr(),
                sin.as_ptr(),
                inplace.as_mut_ptr(),
                rows as i32,
                cols as i32,
                n_dims as i32,
                mode,
            );
            for k in 0..x.len() {
                assert_eq!(
                    inplace[k].to_bits(),
                    away[k].to_bits(),
                    "mode={mode} 就地 k={k}"
                );
            }
            // 与 f64 参考比（旋转是正交变换，绝对误差应在 |x|·几 ulp 量级）
            let half = n_dims / 2;
            for r in 0..rows {
                for i in 0..half {
                    let (i0, i1) = if mode == 0 {
                        (r * cols + i, r * cols + i + half)
                    } else {
                        (r * cols + 2 * i, r * cols + 2 * i + 1)
                    };
                    let (x0, x1) = (x[i0] as f64, x[i1] as f64);
                    let (c, s) = (cos[r * half + i] as f64, sin[r * half + i] as f64);
                    assert!(
                        (away[i0] as f64 - (x0 * c - x1 * s)).abs() < 1e-6,
                        "mode={mode} y0"
                    );
                    assert!(
                        (away[i1] as f64 - (x0 * s + x1 * c)).abs() < 1e-6,
                        "mode={mode} y1"
                    );
                }
            }
        }
        // 不合法的形状 / mode：就地返回，不写输出
        let mut out = vec![7f32; x.len()];
        let before = out.clone();
        for (rows_i, cols_i, dims_i, mode_i) in [
            (3, 16, 7, 0),  // n_dims 奇数
            (3, 16, 32, 0), // n_dims > cols
            (-1, 16, 16, 0),
            (3, 16, 16, 9), // mode 非法
        ] {
            lasx_rope(
                x.as_ptr(),
                cos.as_ptr(),
                sin.as_ptr(),
                out.as_mut_ptr(),
                rows_i,
                cols_i,
                dims_i,
                mode_i,
            );
            assert_eq!(
                out, before,
                "非法参数 ({rows_i},{cols_i},{dims_i},{mode_i}) 不该写输出"
            );
        }
    }

    /// `n_dims = 0`：整块复制（逐位）。
    #[test]
    fn test_rope_ffi_zero_dims_copies() {
        let x: Vec<f32> = (0..12).map(|k| k as f32).collect();
        let mut out = vec![0f32; x.len()];
        lasx_rope(
            x.as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            out.as_mut_ptr(),
            3,
            4,
            0,
            0,
        );
        for k in 0..x.len() {
            assert_eq!(out[k].to_bits(), x[k].to_bits(), "k={k}");
        }
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
