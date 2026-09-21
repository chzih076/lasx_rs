//! 归约类内核的 C ABI 导出：点积、求和、axpy。

/// f32 点积。
///
/// C 签名：`float lasx_dot(const float *a, const float *b, int n)`
#[unsafe(no_mangle)]
pub extern "C" fn lasx_dot(a: *const f32, b: *const f32, n: i32) -> f32 {
    let n = n as usize;
    // SAFETY: FFI 约定——调用方保证 `a`/`b` 至少 n 个可读元素。
    let a = unsafe { std::slice::from_raw_parts(a, n) };
    let b = unsafe { std::slice::from_raw_parts(b, n) };
    crate::ops::dot::dot(a, b)
}

/// f32 向量归约。
///
/// C 签名：`float lasx_sum(const float *x, int n)`
#[unsafe(no_mangle)]
pub extern "C" fn lasx_sum(x: *const f32, n: i32) -> f32 {
    let n = n as usize;
    // SAFETY: FFI 约定——调用方保证 `x` 至少 n 个可读元素。
    let x = unsafe { std::slice::from_raw_parts(x, n) };
    crate::ops::sum::sum(x)
}

/// f64 点积。
///
/// C 签名：`double lasx_dot_f64(const double *a, const double *b, int n)`
#[unsafe(no_mangle)]
pub extern "C" fn lasx_dot_f64(a: *const f64, b: *const f64, n: i32) -> f64 {
    let n = n as usize;
    // SAFETY: FFI 约定——调用方保证 `a`/`b` 至少 n 个可读元素。
    let a = unsafe { std::slice::from_raw_parts(a, n) };
    let b = unsafe { std::slice::from_raw_parts(b, n) };
    crate::ops::dot_f64::dot_f64(a, b)
}

/// `y += alpha·x`。
///
/// C 签名：`void lasx_axpy(float alpha, const float *x, float *y, int n)`
#[unsafe(no_mangle)]
pub extern "C" fn lasx_axpy(alpha: f32, x: *const f32, y: *mut f32, n: i32) {
    let n = n as usize;
    // SAFETY: FFI 约定——调用方保证 `x` 可读、`y` 可写且各至少 n 个元素。
    let x = unsafe { std::slice::from_raw_parts(x, n) };
    let y = unsafe { std::slice::from_raw_parts_mut(y, n) };
    crate::ops::axpy::axpy(alpha, x, y)
}
