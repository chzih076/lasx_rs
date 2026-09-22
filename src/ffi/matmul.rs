//! 矩阵乘内核的 C ABI 导出（行主序，B 不转置）。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（策略见 `docs/dev.md` §17）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX/LSX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]

/// f32 矩阵乘 `C[m×n] = A[m×k]·B[k×n]`。
///
/// C 签名：`void lasx_matmul(int m, int k, int n, const float *a, const float *b, float *c)`
#[unsafe(no_mangle)]
pub extern "C" fn lasx_matmul(m: i32, k: i32, n: i32, a: *const f32, b: *const f32, c: *mut f32) {
    let (m, k, n) = (m as usize, k as usize, n as usize);
    // SAFETY: FFI 约定——`a` 为 m×k、`b` 为 k×n 可读，`c` 为 m×n 可写。
    let a = unsafe { std::slice::from_raw_parts(a, m * k) };
    let b = unsafe { std::slice::from_raw_parts(b, k * n) };
    let c = unsafe { std::slice::from_raw_parts_mut(c, m * n) };
    crate::ops::matmul::matmul_f32(m, k, n, a, b, c);
}

/// f64 矩阵乘 `C[m×n] = A[m×k]·B[k×n]`。
///
/// C 签名：`void lasx_matmul_f64(int m, int k, int n, const double *a, const double *b, double *c)`
#[unsafe(no_mangle)]
pub extern "C" fn lasx_matmul_f64(
    m: i32,
    k: i32,
    n: i32,
    a: *const f64,
    b: *const f64,
    c: *mut f64,
) {
    let (m, k, n) = (m as usize, k as usize, n as usize);
    // SAFETY: FFI 约定——`a` 为 m×k、`b` 为 k×n 可读，`c` 为 m×n 可写。
    let a = unsafe { std::slice::from_raw_parts(a, m * k) };
    let b = unsafe { std::slice::from_raw_parts(b, k * n) };
    let c = unsafe { std::slice::from_raw_parts_mut(c, m * n) };
    crate::ops::matmul_f64::matmul_f64(m, k, n, a, b, c);
}
