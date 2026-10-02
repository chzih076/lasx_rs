//! 量化内核的 C ABI 导出（LASX-only）。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（策略见 `docs/dev.md` §17）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX/LSX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]

/// int8 量化点积。
///
/// C 签名：`int lasx_dot_i8(const int8_t *a, const int8_t *b, int n)`
#[unsafe(no_mangle)]
pub extern "C" fn lasx_dot_i8(a: *const i8, b: *const i8, n: i32) -> i32 {
    let n = n as usize;
    // SAFETY: FFI 约定——调用方保证 `a`/`b` 至少 n 个可读元素。
    let a = unsafe { std::slice::from_raw_parts(a, n) };
    let b = unsafe { std::slice::from_raw_parts(b, n) };
    crate::ops::dot_i8::dot_i8(a, b)
}

/// Q4 量化点积（每字节 2 个无符号 nibble，scale 每 32 字节一组）。
///
/// C 签名：
/// `double lasx_dot_q4(const uint8_t *qa, const float *sa, const uint8_t *qb, const float *sb, int n_bytes)`
#[unsafe(no_mangle)]
pub extern "C" fn lasx_dot_q4(
    qa: *const u8,
    sa: *const f32,
    qb: *const u8,
    sb: *const f32,
    n_bytes: i32,
) -> f64 {
    let n = n_bytes as usize;
    let n_groups = n.div_ceil(32);
    // SAFETY: FFI 约定——`qa`/`qb` 至少 n 字节，`sa`/`sb` 至少 ceil(n/32) 个 f32。
    let qa = unsafe { std::slice::from_raw_parts(qa, n) };
    let qb = unsafe { std::slice::from_raw_parts(qb, n) };
    let sa = unsafe { std::slice::from_raw_parts(sa, n_groups) };
    let sb = unsafe { std::slice::from_raw_parts(sb, n_groups) };
    crate::ops::dot_q4::dot_q4(qa, sa, qb, sb)
}

/* ==================== N3：int8 推理（量化生产端 + GEMV） ==================== */
//
// 契约（数值与边界）在 `docs/ops.md` §2.12/§2.13；设计记录在 `docs/dev.md` §21。
// 这些是**追加**的新符号，原始 15 个（§4.1）的签名与语义不受影响。

/// 最大绝对值：`max_i |x_i|`。空输入返回 `0.0`（NaN 被忽略，见 §2.12）。
///
/// C 签名：`float lasx_amax(const float *x, int n)`
#[unsafe(no_mangle)]
pub extern "C" fn lasx_amax(x: *const f32, n: i32) -> f32 {
    let n = n as usize;
    // SAFETY: FFI 约定——`x` 至少 n 个可读元素。
    let x = unsafe { std::slice::from_raw_parts(x, n) };
    crate::ops::quant_i8::amax(x)
}

/// 逐行最大绝对值：`out[r] = max_j |x[r·cols + j]|`。
///
/// C 签名：`void lasx_absmax_rows(const float *x, int rows, int cols, float *out)`
#[unsafe(no_mangle)]
pub extern "C" fn lasx_absmax_rows(x: *const f32, rows: i32, cols: i32, out: *mut f32) {
    let (rows, cols) = (rows as usize, cols as usize);
    // SAFETY: FFI 约定——`x` 至少 rows×cols 个可读元素，`out` 至少 rows 个可写元素。
    let x = unsafe { std::slice::from_raw_parts(x, rows * cols) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, rows) };
    crate::ops::quant_i8::absmax_rows(x, rows, cols, out);
}

/// 逐张量对称量化：`q = round_ties_even(x · (1/scale))`，**返回 scale**。
///
/// C 签名：`float lasx_quantize_i8_per_tensor(const float *x, int8_t *q, int n)`
#[unsafe(no_mangle)]
pub extern "C" fn lasx_quantize_i8_per_tensor(x: *const f32, q: *mut i8, n: i32) -> f32 {
    let n = n as usize;
    // SAFETY: FFI 约定——`x` 至少 n 个可读元素，`q` 至少 n 个可写元素。
    let x = unsafe { std::slice::from_raw_parts(x, n) };
    let q = unsafe { std::slice::from_raw_parts_mut(q, n) };
    crate::ops::quant_i8::quantize_i8_per_tensor(x, q)
}

/// 逐行对称量化：每行一个 scale（per-token 激活 / per-channel 权重共用）。
///
/// C 签名：`void lasx_quantize_i8_per_row(const float *x, int8_t *q, float *scales, int rows, int cols)`
#[unsafe(no_mangle)]
pub extern "C" fn lasx_quantize_i8_per_row(
    x: *const f32,
    q: *mut i8,
    scales: *mut f32,
    rows: i32,
    cols: i32,
) {
    let (rows, cols) = (rows as usize, cols as usize);
    let n = rows * cols;
    // SAFETY: FFI 约定——`x` 至少 n 个可读元素，`q` 至少 n 个可写元素，`scales` 至少 rows 个。
    let x = unsafe { std::slice::from_raw_parts(x, n) };
    let q = unsafe { std::slice::from_raw_parts_mut(q, n) };
    let scales = unsafe { std::slice::from_raw_parts_mut(scales, rows) };
    crate::ops::quant_i8::quantize_i8_per_row(x, rows, cols, q, scales);
}

/// 逐张量反量化：`out[i] = (q[i] as f32) · scale`。
///
/// C 签名：`void lasx_dequantize_i8(const int8_t *q, float scale, float *out, int n)`
#[unsafe(no_mangle)]
pub extern "C" fn lasx_dequantize_i8(q: *const i8, scale: f32, out: *mut f32, n: i32) {
    let n = n as usize;
    // SAFETY: FFI 约定——`q` 至少 n 个可读元素，`out` 至少 n 个可写元素。
    let q = unsafe { std::slice::from_raw_parts(q, n) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, n) };
    crate::ops::quant_i8::dequantize_i8(q, scale, out);
}

/// 逐行反量化：每行用自己的 `scales[r]`。
///
/// C 签名：`void lasx_dequantize_i8_rows(const int8_t *q, const float *scales, float *out, int rows, int cols)`
#[unsafe(no_mangle)]
pub extern "C" fn lasx_dequantize_i8_rows(
    q: *const i8,
    scales: *const f32,
    out: *mut f32,
    rows: i32,
    cols: i32,
) {
    let (rows, cols) = (rows as usize, cols as usize);
    let n = rows * cols;
    // SAFETY: FFI 约定——`q` 至少 n 个可读元素，`scales` 至少 rows 个，`out` 至少 n 个可写元素。
    let q = unsafe { std::slice::from_raw_parts(q, n) };
    let scales = unsafe { std::slice::from_raw_parts(scales, rows) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, n) };
    crate::ops::quant_i8::dequantize_i8_per_row(q, rows, cols, scales, out);
}

/// int8 权重 × int8 激活的矩阵-向量乘：`y[o] = (Σ W[o,i]·x[i]) · (scale_w[o]·scale_x)`。
///
/// C 签名：
/// `void lasx_gemv_i8(const int8_t *w, const float *scale_w, const int8_t *x, float scale_x, float *y, int m, int k)`
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn lasx_gemv_i8(
    w: *const i8,
    scale_w: *const f32,
    x: *const i8,
    scale_x: f32,
    y: *mut f32,
    m: i32,
    k: i32,
) {
    let (m, k) = (m as usize, k as usize);
    // SAFETY: FFI 约定——`w` 至少 m×k 个可读元素，`x` 至少 k 个，`scale_w` 至少 m 个，
    // `y` 至少 m 个可写元素。
    let w = unsafe { std::slice::from_raw_parts(w, m * k) };
    let x = unsafe { std::slice::from_raw_parts(x, k) };
    let scale_w = unsafe { std::slice::from_raw_parts(scale_w, m) };
    let y = unsafe { std::slice::from_raw_parts_mut(y, m) };
    crate::ops::gemv_i8::gemv_i8(w, scale_w, x, scale_x, m, k, y);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::checked::{
        lasx_amax_checked, lasx_gemv_i8_checked, lasx_quantize_i8_per_tensor_checked,
    };
    use crate::ffi::status::LasxStatus;

    /// C ABI 路径与安全 Rust API **逐位一致**（"追加符号"不能引入第二套数值）。
    #[test]
    fn test_n3_c_abi_matches_api_bitwise() {
        let mut rnd = crate::ops::testutil::Lcg(0x0bad_c0de);
        let x: Vec<f32> = (0..64).map(|_| 3.0 * rnd.f64() as f32).collect();
        let mut q_c = vec![0i8; x.len()];
        let scale_c = lasx_quantize_i8_per_tensor(x.as_ptr(), q_c.as_mut_ptr(), x.len() as i32);
        let (q_api, scale_api) = crate::api::quantize_i8_per_tensor(&x).unwrap();
        assert_eq!(q_c, q_api.as_slice(), "量化逐位一致");
        assert_eq!(scale_c.to_bits(), scale_api.to_bits(), "scale 逐位一致");
        assert_eq!(
            lasx_amax(x.as_ptr(), x.len() as i32).to_bits(),
            crate::api::amax(&x).unwrap().to_bits()
        );

        let mut back = vec![0f32; x.len()];
        lasx_dequantize_i8(q_c.as_ptr(), scale_c, back.as_mut_ptr(), x.len() as i32);
        let back_api = crate::api::dequantize_i8(&q_api, scale_api).unwrap();
        assert_eq!(back, back_api.as_slice(), "反量化逐位一致");

        // GEMV：C 路径 vs api 路径
        let (m, k) = (4usize, 64usize);
        let w: Vec<i8> = (0..m * k).map(|i| (i % 255) as i32 as i8).collect();
        let sw: Vec<f32> = (0..m).map(|o| 0.01 * (o as f32 + 1.0)).collect();
        let mut y_c = vec![0f32; m];
        lasx_gemv_i8(
            w.as_ptr(),
            sw.as_ptr(),
            q_c.as_ptr(),
            0.02,
            y_c.as_mut_ptr(),
            m as i32,
            k as i32,
        );
        let y_api = crate::api::gemv_i8(&w, &sw, &q_c, 0.02, m, k).unwrap();
        for o in 0..m {
            assert_eq!(y_c[o].to_bits(), y_api[o].to_bits(), "o={o}");
        }
    }

    /// `_checked` 变体：成功写 Ok，负长度/空指针写对应状态并返回中性值。
    #[test]
    fn test_n3_checked_status_channel() {
        let x = [1.0f32, -2.0, 3.0, -4.0];
        let mut status = -1i32;
        let mut q = [0i8; 4];
        let s = lasx_quantize_i8_per_tensor_checked(
            x.as_ptr(),
            q.as_mut_ptr(),
            4,
            &mut status as *mut i32,
        );
        assert_eq!(status, LasxStatus::Ok as i32);
        assert!(s > 0.0);

        // 负长度
        let mut st = -1i32;
        let s = lasx_quantize_i8_per_tensor_checked(
            x.as_ptr(),
            q.as_mut_ptr(),
            -1,
            &mut st as *mut i32,
        );
        assert_eq!(st, LasxStatus::NegativeLength as i32);
        assert_eq!(s, 0.0, "失败时返回中性值");

        // 空指针（长度非 0）
        let mut st = -1i32;
        let a = lasx_amax_checked(std::ptr::null(), 4, &mut st as *mut i32);
        assert_eq!(st, LasxStatus::NullPointer as i32);
        assert_eq!(a, 0.0);

        // GEMV：长度为 i32 ⇒ 64 位下 `m×k` 永远不会溢出 `usize`，所以这里能触发的只有
        // 空指针分支（`SizeOverflow` 只在 Rust `api` 层用 usize 参数时可达，见 api::tests）。
        let mut st = -1i32;
        let mut y = [0f32; 1];
        lasx_gemv_i8_checked(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            1.0,
            y.as_mut_ptr(),
            i32::MAX,
            3,
            &mut st as *mut i32,
        );
        assert_eq!(st, LasxStatus::NullPointer as i32);
    }
}
