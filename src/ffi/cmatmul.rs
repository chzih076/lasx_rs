//! **复数矩阵乘的 C ABI 导出**（分平面 `(Re, Im)`，**紧凑行主序**）。
//!
//! 契约、命名与错误约定全部沿用现有符号（`docs/c-abi.md` §9.1）：
//! - 命名：`lasx_cmatmul_{f32,f64}` + `_checked` 孪生（4 个新符号）；
//! - **错误通道照抄现有 `_checked`**：`m/k/n < 0` ⇒ `NegativeLength`；尺寸相乘溢出 ⇒
//!   `SizeOverflow`；指针空且长度非 0 ⇒ `NullPointer`；**`route ∉ {0,1}` ⇒ `NegativeLength`**
//!   （复用现有"非法整型参数"码，**不新增状态码**、不动 `LasxStatus` 这个 ABI 面）；
//!   失败时**不改输出**。
//! - **紧凑行主序**（不引 leading dimension；与库内 `ops::matmul` 和既有
//!   `lasx_matmul(m,k,n,a,b,c)` 的 ABI 一致）。strided 版本将来作为**独立追加符号**。
//! - **别名规则**：`cr`/`ci` 不得与 `ar`/`ai`/`br`/`bi` 任一重叠；`ar ≠ ai`、`br ≠ bi`
//!   （分平面两块独立缓冲）。违反是 UB（同一块内存既读又写）。
//! - **无内部分配**：全部复用调用方缓冲。
//!
//! **`route` 语义与契约**（见 `docs/ops.md` §9.06）：
//! `0` = `Route::Direct`（4 条实矩阵乘，**各乘积与对应实矩阵乘逐位一致**）；
//! `1` = `Route::Karatsuba`（3 条实矩阵乘，**仅承诺容差**；Rust 侧默认）。
//!
//! **成本（真话）**：复数 ≈ **1.22×** 调 3 条实矩阵乘的成本；复数 API 的价值是**一次调用 /
//! 正确的复数语义 / 可选逐位**，**不是速度**。
use crate::ops::cmatmul::{Route, cmatmul_f32_into_route, cmatmul_f64_into_route};

use super::status::{
    LasxStatus, checked_len, checked_mul, checked_slice, checked_slice_mut, or_fail,
};

/// C 侧 `int route` → [`Route`]；越界用现有"非法整型参数"码（不新增状态码）。
fn route_of(route: i32) -> Result<Route, LasxStatus> {
    match route {
        0 => Ok(Route::Direct),
        1 => Ok(Route::Karatsuba),
        _ => Err(LasxStatus::NegativeLength),
    }
}

/// 复数 f32 矩阵乘 `C[m×n] = A[m×k]·B[k×n]`（分平面、紧凑行主序）。
///
/// C 签名：`void lasx_cmatmul_f32(int m, int k, int n, const float *ar, const float *ai,
/// const float *br, const float *bi, float *cr, float *ci, int route)`
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn lasx_cmatmul_f32(
    m: i32,
    k: i32,
    n: i32,
    ar: *const f32,
    ai: *const f32,
    br: *const f32,
    bi: *const f32,
    cr: *mut f32,
    ci: *mut f32,
    route: i32,
) {
    let (m, k, n) = (m as usize, k as usize, n as usize);
    let rt = match route {
        1 => Route::Karatsuba,
        _ => Route::Direct,
    };
    // SAFETY: FFI 约定——`ar/ai` 为 m×k、`br/bi` 为 k×n 可读，`cr/ci` 为 m×n 可写且不与输入重叠
    // （别名规则见模块文档；裸符号沿用既有风格：不校验、直接建视图）。
    unsafe {
        let (ar, ai) = (
            std::slice::from_raw_parts(ar, m * k),
            std::slice::from_raw_parts(ai, m * k),
        );
        let (br, bi) = (
            std::slice::from_raw_parts(br, k * n),
            std::slice::from_raw_parts(bi, k * n),
        );
        let (cr, ci) = (
            std::slice::from_raw_parts_mut(cr, m * n),
            std::slice::from_raw_parts_mut(ci, m * n),
        );
        cmatmul_f32_into_route(m, k, n, ar, ai, br, bi, cr, ci, rt);
    }
}

/// 带错误通道的 [`lasx_cmatmul_f32`]。C 签名：
/// `int lasx_cmatmul_f32_checked(int m, int k, int n, const float *ar, const float *ai,
/// const float *br, const float *bi, float *cr, float *ci, int route, int *status)`
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn lasx_cmatmul_f32_checked(
    m: i32,
    k: i32,
    n: i32,
    ar: *const f32,
    ai: *const f32,
    br: *const f32,
    bi: *const f32,
    cr: *mut f32,
    ci: *mut f32,
    route: i32,
    status: *mut i32,
) -> i32 {
    let rt = or_fail!(route_of(route), status, 1);
    let m = or_fail!(checked_len(m), status, 1);
    let k = or_fail!(checked_len(k), status, 1);
    let n = or_fail!(checked_len(n), status, 1);
    let (mk, kn, mn) = (
        or_fail!(checked_mul(m, k), status, 1),
        or_fail!(checked_mul(k, n), status, 1),
        or_fail!(checked_mul(m, n), status, 1),
    );
    // SAFETY: 四个 `checked_slice*` 各自校验指针与长度（空且 0 长度是合法空切片）。
    let (ar, ai, br, bi) = unsafe {
        (
            or_fail!(checked_slice(ar, mk), status, 1),
            or_fail!(checked_slice(ai, mk), status, 1),
            or_fail!(checked_slice(br, kn), status, 1),
            or_fail!(checked_slice(bi, kn), status, 1),
        )
    };
    // SAFETY: 同上一组；`cr/ci` 是可写切片，契约要求它们不与输入重叠。
    let (cr, ci) = unsafe {
        (
            or_fail!(checked_slice_mut(cr, mn), status, 1),
            or_fail!(checked_slice_mut(ci, mn), status, 1),
        )
    };
    cmatmul_f32_into_route(m, k, n, ar, ai, br, bi, cr, ci, rt);
    LasxStatus::Ok.write(status);
    LasxStatus::Ok as i32
}

/// 复数 f64 矩阵乘（同 [`lasx_cmatmul_f32`] 的语义与布局）。C 签名：
/// `void lasx_cmatmul_f64(int m, int k, int n, const double *ar, const double *ai,
/// const double *br, const double *bi, double *cr, double *ci, int route)`
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn lasx_cmatmul_f64(
    m: i32,
    k: i32,
    n: i32,
    ar: *const f64,
    ai: *const f64,
    br: *const f64,
    bi: *const f64,
    cr: *mut f64,
    ci: *mut f64,
    route: i32,
) {
    let (m, k, n) = (m as usize, k as usize, n as usize);
    let rt = match route {
        1 => Route::Karatsuba,
        _ => Route::Direct,
    };
    // SAFETY: 同 `lasx_cmatmul_f32`。
    unsafe {
        let (ar, ai) = (
            std::slice::from_raw_parts(ar, m * k),
            std::slice::from_raw_parts(ai, m * k),
        );
        let (br, bi) = (
            std::slice::from_raw_parts(br, k * n),
            std::slice::from_raw_parts(bi, k * n),
        );
        let (cr, ci) = (
            std::slice::from_raw_parts_mut(cr, m * n),
            std::slice::from_raw_parts_mut(ci, m * n),
        );
        cmatmul_f64_into_route(m, k, n, ar, ai, br, bi, cr, ci, rt);
    }
}

/// 带错误通道的 [`lasx_cmatmul_f64`]（签名同 f32 版，`float`→`double`）。
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn lasx_cmatmul_f64_checked(
    m: i32,
    k: i32,
    n: i32,
    ar: *const f64,
    ai: *const f64,
    br: *const f64,
    bi: *const f64,
    cr: *mut f64,
    ci: *mut f64,
    route: i32,
    status: *mut i32,
) -> i32 {
    let rt = or_fail!(route_of(route), status, 1);
    let m = or_fail!(checked_len(m), status, 1);
    let k = or_fail!(checked_len(k), status, 1);
    let n = or_fail!(checked_len(n), status, 1);
    let (mk, kn, mn) = (
        or_fail!(checked_mul(m, k), status, 1),
        or_fail!(checked_mul(k, n), status, 1),
        or_fail!(checked_mul(m, n), status, 1),
    );
    // SAFETY: 同 f32 版的 `_checked`。
    let (ar, ai, br, bi) = unsafe {
        (
            or_fail!(checked_slice(ar, mk), status, 1),
            or_fail!(checked_slice(ai, mk), status, 1),
            or_fail!(checked_slice(br, kn), status, 1),
            or_fail!(checked_slice(bi, kn), status, 1),
        )
    };
    // SAFETY: 同上（可写切片，且契约要求不与输入重叠）。
    let (cr, ci) = unsafe {
        (
            or_fail!(checked_slice_mut(cr, mn), status, 1),
            or_fail!(checked_slice_mut(ci, mn), status, 1),
        )
    };
    cmatmul_f64_into_route(m, k, n, ar, ai, br, bi, cr, ci, rt);
    LasxStatus::Ok.write(status);
    LasxStatus::Ok as i32
}
