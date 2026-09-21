//! 量化内核的 C ABI 导出（LASX-only）。

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
