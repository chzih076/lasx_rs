//! `lasx_*_checked`：带错误通道的 C ABI 变体。
//!
//! 与 [`crate::ffi::reduce`] 等模块里的原始符号一一对应，区别只有一个末尾的
//! `int *status` 出参：先校验，失败时写入 [`LasxStatus`] 并返回安全中性值
//! （数值型 `0.0`/`0`，`void` 型只写状态）。原始 15 个符号的语义与签名完全不变。
//!
//! C 层能校验的是**结构**：指针非空、长度非负、尺寸相乘不溢出、物理常数合法。
//! "数组真实长度是否与声明的形状一致"只有知道长度的上层（如 YouLiLong 原生扩展）
//! 才判得了，那里用 [`LasxStatus::BadShape`]。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（策略见 `docs/dev.md` §17）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX/LSX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]

use super::status::{
    checked_finite, checked_len, checked_mul, checked_positive, checked_slice, checked_slice_mut,
    LasxStatus,
};

/// 校验失败即写状态并提前返回。把各包装里重复的样板收敛到一处。
macro_rules! or_fail {
    // `void` 版：提前 `return`（写成 `return ();` 会被 clippy 判为多余）
    ($expr:expr, $status:expr, ()) => {
        match $expr {
            Ok(v) => v,
            Err(e) => {
                e.write($status);
                return;
            }
        }
    };
    ($expr:expr, $status:expr, $ret:expr) => {
        match $expr {
            Ok(v) => v,
            Err(e) => {
                e.write($status);
                return $ret;
            }
        }
    };
}

/* ==================== 归约类 ==================== */

/// 带错误通道的 f32 点积。C 签名：`float lasx_dot_checked(const float*, const float*, int, int*)`
#[unsafe(no_mangle)]
pub extern "C" fn lasx_dot_checked(a: *const f32, b: *const f32, n: i32, status: *mut i32) -> f32 {
    let n = or_fail!(checked_len(n), status, 0.0);
    // SAFETY: 由 checked_slice 校验非空；长度由调用方保证。
    let (a, b) = unsafe {
        (
            or_fail!(checked_slice(a, n), status, 0.0),
            or_fail!(checked_slice(b, n), status, 0.0),
        )
    };
    LasxStatus::Ok.write(status);
    crate::ops::dot::dot(a, b)
}

/// 带错误通道的 f32 归约。C 签名：`float lasx_sum_checked(const float*, int, int*)`
#[unsafe(no_mangle)]
pub extern "C" fn lasx_sum_checked(x: *const f32, n: i32, status: *mut i32) -> f32 {
    let n = or_fail!(checked_len(n), status, 0.0);
    // SAFETY: 见上。
    let x = unsafe { or_fail!(checked_slice(x, n), status, 0.0) };
    LasxStatus::Ok.write(status);
    crate::ops::sum::sum(x)
}

/// 带错误通道的 f64 点积。C 签名：`double lasx_dot_f64_checked(const double*, const double*, int, int*)`
#[unsafe(no_mangle)]
pub extern "C" fn lasx_dot_f64_checked(
    a: *const f64,
    b: *const f64,
    n: i32,
    status: *mut i32,
) -> f64 {
    let n = or_fail!(checked_len(n), status, 0.0);
    // SAFETY: 见上。
    let (a, b) = unsafe {
        (
            or_fail!(checked_slice(a, n), status, 0.0),
            or_fail!(checked_slice(b, n), status, 0.0),
        )
    };
    LasxStatus::Ok.write(status);
    crate::ops::dot_f64::dot_f64(a, b)
}

/// 带错误通道的 `y += alpha·x`。C 签名：`void lasx_axpy_checked(float, const float*, float*, int, int*)`
#[unsafe(no_mangle)]
pub extern "C" fn lasx_axpy_checked(
    alpha: f32,
    x: *const f32,
    y: *mut f32,
    n: i32,
    status: *mut i32,
) {
    let n = or_fail!(checked_len(n), status, ());
    // SAFETY: 见上。
    let (x, y) = unsafe {
        (
            or_fail!(checked_slice(x, n), status, ()),
            or_fail!(checked_slice_mut(y, n), status, ()),
        )
    };
    LasxStatus::Ok.write(status);
    crate::ops::axpy::axpy(alpha, x, y);
}

/* ==================== 矩阵乘 ==================== */

/// 带错误通道的 f32 矩阵乘。
/// C 签名：`void lasx_matmul_checked(int m, int k, int n, const float*, const float*, float*, int*)`
#[unsafe(no_mangle)]
pub extern "C" fn lasx_matmul_checked(
    m: i32,
    k: i32,
    n: i32,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    status: *mut i32,
) {
    let (m, k, n) = (
        or_fail!(checked_len(m), status, ()),
        or_fail!(checked_len(k), status, ()),
        or_fail!(checked_len(n), status, ()),
    );
    let (la, lb, lc) = (
        or_fail!(checked_mul(m, k), status, ()),
        or_fail!(checked_mul(k, n), status, ()),
        or_fail!(checked_mul(m, n), status, ()),
    );
    // SAFETY: 见上；各缓冲长度由调用方按 m×k / k×n / m×n 保证。
    let (a, b, c) = unsafe {
        (
            or_fail!(checked_slice(a, la), status, ()),
            or_fail!(checked_slice(b, lb), status, ()),
            or_fail!(checked_slice_mut(c, lc), status, ()),
        )
    };
    LasxStatus::Ok.write(status);
    crate::ops::matmul::matmul_f32(m, k, n, a, b, c);
}

/// 带错误通道的 f64 矩阵乘。
/// C 签名：`void lasx_matmul_f64_checked(int m, int k, int n, const double*, const double*, double*, int*)`
#[unsafe(no_mangle)]
pub extern "C" fn lasx_matmul_f64_checked(
    m: i32,
    k: i32,
    n: i32,
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    status: *mut i32,
) {
    let (m, k, n) = (
        or_fail!(checked_len(m), status, ()),
        or_fail!(checked_len(k), status, ()),
        or_fail!(checked_len(n), status, ()),
    );
    let (la, lb, lc) = (
        or_fail!(checked_mul(m, k), status, ()),
        or_fail!(checked_mul(k, n), status, ()),
        or_fail!(checked_mul(m, n), status, ()),
    );
    // SAFETY: 见上。
    let (a, b, c) = unsafe {
        (
            or_fail!(checked_slice(a, la), status, ()),
            or_fail!(checked_slice(b, lb), status, ()),
            or_fail!(checked_slice_mut(c, lc), status, ()),
        )
    };
    LasxStatus::Ok.write(status);
    crate::ops::matmul_f64::matmul_f64(m, k, n, a, b, c);
}

/* ==================== 量化 ==================== */

/// 带错误通道的 int8 点积。C 签名：`int lasx_dot_i8_checked(const int8_t*, const int8_t*, int, int*)`
#[unsafe(no_mangle)]
pub extern "C" fn lasx_dot_i8_checked(a: *const i8, b: *const i8, n: i32, status: *mut i32) -> i32 {
    let n = or_fail!(checked_len(n), status, 0);
    // SAFETY: 见上。
    let (a, b) = unsafe {
        (
            or_fail!(checked_slice(a, n), status, 0),
            or_fail!(checked_slice(b, n), status, 0),
        )
    };
    LasxStatus::Ok.write(status);
    crate::ops::dot_i8::dot_i8(a, b)
}

/// 带错误通道的 Q4 量化点积。
/// C 签名：`double lasx_dot_q4_checked(const uint8_t*, const float*, const uint8_t*, const float*, int, int*)`
#[unsafe(no_mangle)]
pub extern "C" fn lasx_dot_q4_checked(
    qa: *const u8,
    sa: *const f32,
    qb: *const u8,
    sb: *const f32,
    n_bytes: i32,
    status: *mut i32,
) -> f64 {
    let n = or_fail!(checked_len(n_bytes), status, 0.0);
    let groups = n.div_ceil(32);
    // SAFETY: 见上；scale 数组按 ceil(n/32) 个元素。
    let (qa, qb, sa, sb) = unsafe {
        (
            or_fail!(checked_slice(qa, n), status, 0.0),
            or_fail!(checked_slice(qb, n), status, 0.0),
            or_fail!(checked_slice(sa, groups), status, 0.0),
            or_fail!(checked_slice(sb, groups), status, 0.0),
        )
    };
    LasxStatus::Ok.write(status);
    crate::ops::dot_q4::dot_q4(qa, sa, qb, sb)
}

/* ==================== 批量几何 ==================== */

/// 带错误通道的批量 3 分量模长。
/// C 签名：`void lasx_norm3_batch_checked(const double*, const double*, const double*, double*, int, int*)`
#[unsafe(no_mangle)]
pub extern "C" fn lasx_norm3_batch_checked(
    xs: *const f64,
    ys: *const f64,
    zs: *const f64,
    out: *mut f64,
    n: i32,
    status: *mut i32,
) {
    let n = or_fail!(checked_len(n), status, ());
    // SAFETY: 见上。
    let (xs, ys, zs, out) = unsafe {
        (
            or_fail!(checked_slice(xs, n), status, ()),
            or_fail!(checked_slice(ys, n), status, ()),
            or_fail!(checked_slice(zs, n), status, ()),
            or_fail!(checked_slice_mut(out, n), status, ()),
        )
    };
    LasxStatus::Ok.write(status);
    crate::ops::norm3_batch::norm3_batch(xs, ys, zs, out);
}

/// 带错误通道的批量缩放加。
/// C 签名：`void lasx_vec3_add_scaled_batch_checked(const double*, ..., double s, double*, ..., int, int*)`
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn lasx_vec3_add_scaled_batch_checked(
    ax: *const f64,
    ay: *const f64,
    az: *const f64,
    bx: *const f64,
    by: *const f64,
    bz: *const f64,
    s: f64,
    ox: *mut f64,
    oy: *mut f64,
    oz: *mut f64,
    n: i32,
    status: *mut i32,
) {
    let n = or_fail!(checked_len(n), status, ());
    // SAFETY: 见上。
    let (ax, ay, az, bx, by, bz, ox, oy, oz) = unsafe {
        (
            or_fail!(checked_slice(ax, n), status, ()),
            or_fail!(checked_slice(ay, n), status, ()),
            or_fail!(checked_slice(az, n), status, ()),
            or_fail!(checked_slice(bx, n), status, ()),
            or_fail!(checked_slice(by, n), status, ()),
            or_fail!(checked_slice(bz, n), status, ()),
            or_fail!(checked_slice_mut(ox, n), status, ()),
            or_fail!(checked_slice_mut(oy, n), status, ()),
            or_fail!(checked_slice_mut(oz, n), status, ()),
        )
    };
    LasxStatus::Ok.write(status);
    crate::ops::vec3_add_scaled_batch::vec3_add_scaled_batch(ax, ay, az, bx, by, bz, s, ox, oy, oz);
}

/// 带错误通道的批量 2D 距离。
/// C 签名：`void lasx_batch_distance2d_checked(float, float, const float*, const float*, float*, int, int*)`
#[unsafe(no_mangle)]
pub extern "C" fn lasx_batch_distance2d_checked(
    px: f32,
    py: f32,
    xs: *const f32,
    ys: *const f32,
    out: *mut f32,
    n: i32,
    status: *mut i32,
) {
    let n = or_fail!(checked_len(n), status, ());
    // SAFETY: 见上。
    let (xs, ys, out) = unsafe {
        (
            or_fail!(checked_slice(xs, n), status, ()),
            or_fail!(checked_slice(ys, n), status, ()),
            or_fail!(checked_slice_mut(out, n), status, ()),
        )
    };
    LasxStatus::Ok.write(status);
    crate::ops::batch_distance2d::batch_distance2d(px, py, xs, ys, out);
}

/* ==================== 物理 ==================== */

/// 带错误通道的批量弹道欧拉步。
/// C 签名：`void lasx_ballistic_step_checked(float*, ..., const float* k, int, float dt, float g, int*)`
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn lasx_ballistic_step_checked(
    x: *mut f32,
    y: *mut f32,
    z: *mut f32,
    vx: *mut f32,
    vy: *mut f32,
    vz: *mut f32,
    k: *const f32,
    n: i32,
    dt: f32,
    g: f32,
    status: *mut i32,
) {
    let n = or_fail!(checked_len(n), status, ());
    // SAFETY: 见上。
    let (x, y, z, vx, vy, vz, k) = unsafe {
        (
            or_fail!(checked_slice_mut(x, n), status, ()),
            or_fail!(checked_slice_mut(y, n), status, ()),
            or_fail!(checked_slice_mut(z, n), status, ()),
            or_fail!(checked_slice_mut(vx, n), status, ()),
            or_fail!(checked_slice_mut(vy, n), status, ()),
            or_fail!(checked_slice_mut(vz, n), status, ()),
            or_fail!(checked_slice(k, n), status, ()),
        )
    };
    LasxStatus::Ok.write(status);
    crate::ops::ballistic_step::ballistic_step(x, y, z, vx, vy, vz, k, dt, g);
}

/// 带错误通道的批量 J2 加速度。会额外校验 `mu > 0`、`re > 0`、`j2` 有限。
/// C 签名：`void lasx_j2_accel_batch_checked(const double*, ..., double mu, double j2, double re, double*, ..., int, int*)`
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn lasx_j2_accel_batch_checked(
    rx: *const f64,
    ry: *const f64,
    rz: *const f64,
    mu: f64,
    j2: f64,
    re: f64,
    ax: *mut f64,
    ay: *mut f64,
    az: *mut f64,
    n: i32,
    status: *mut i32,
) {
    let n = or_fail!(checked_len(n), status, ());
    or_fail!(checked_positive(mu), status, ());
    or_fail!(checked_positive(re), status, ());
    or_fail!(checked_finite(j2), status, ());
    // SAFETY: 见上。
    let (rx, ry, rz, ax, ay, az) = unsafe {
        (
            or_fail!(checked_slice(rx, n), status, ()),
            or_fail!(checked_slice(ry, n), status, ()),
            or_fail!(checked_slice(rz, n), status, ()),
            or_fail!(checked_slice_mut(ax, n), status, ()),
            or_fail!(checked_slice_mut(ay, n), status, ()),
            or_fail!(checked_slice_mut(az, n), status, ()),
        )
    };
    LasxStatus::Ok.write(status);
    crate::ops::j2_accel_batch::j2_accel_batch(rx, ry, rz, mu, j2, re, ax, ay, az);
}

/// 带错误通道的批量 RK4 J2 步。会额外校验 `mu > 0`、`re > 0`、`j2`/`dt` 有限。
/// C 签名：`void lasx_rk4_j2_step_batch_checked(double*, ..., double mu, double j2, double re, double dt, int, int*)`
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn lasx_rk4_j2_step_batch_checked(
    rx: *mut f64,
    ry: *mut f64,
    rz: *mut f64,
    vx: *mut f64,
    vy: *mut f64,
    vz: *mut f64,
    mu: f64,
    j2: f64,
    re: f64,
    dt: f64,
    n: i32,
    status: *mut i32,
) {
    let n = or_fail!(checked_len(n), status, ());
    or_fail!(checked_positive(mu), status, ());
    or_fail!(checked_positive(re), status, ());
    or_fail!(checked_finite(j2), status, ());
    or_fail!(checked_finite(dt), status, ());
    // SAFETY: 见上。
    let (rx, ry, rz, vx, vy, vz) = unsafe {
        (
            or_fail!(checked_slice_mut(rx, n), status, ()),
            or_fail!(checked_slice_mut(ry, n), status, ()),
            or_fail!(checked_slice_mut(rz, n), status, ()),
            or_fail!(checked_slice_mut(vx, n), status, ()),
            or_fail!(checked_slice_mut(vy, n), status, ()),
            or_fail!(checked_slice_mut(vz, n), status, ()),
        )
    };
    LasxStatus::Ok.write(status);
    crate::ops::rk4_j2_step_batch::rk4_j2_step_batch(rx, ry, rz, vx, vy, vz, mu, j2, re, dt);
}

/* ==================== 姿态/几何批处理 ==================== */

/// 带错误通道的批量叉积。
/// C 签名：`void lasx_cross3_batch_checked(const double*, ..., double*, ..., int, int*)`
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn lasx_cross3_batch_checked(
    ax: *const f64,
    ay: *const f64,
    az: *const f64,
    bx: *const f64,
    by: *const f64,
    bz: *const f64,
    ox: *mut f64,
    oy: *mut f64,
    oz: *mut f64,
    n: i32,
    status: *mut i32,
) {
    let n = or_fail!(checked_len(n), status, ());
    let (ax, ay, az, bx, by, bz) = unsafe {
        (
            or_fail!(checked_slice(ax, n), status, ()),
            or_fail!(checked_slice(ay, n), status, ()),
            or_fail!(checked_slice(az, n), status, ()),
            or_fail!(checked_slice(bx, n), status, ()),
            or_fail!(checked_slice(by, n), status, ()),
            or_fail!(checked_slice(bz, n), status, ()),
        )
    };
    let (ox, oy, oz) = unsafe {
        (
            or_fail!(checked_slice_mut(ox, n), status, ()),
            or_fail!(checked_slice_mut(oy, n), status, ()),
            or_fail!(checked_slice_mut(oz, n), status, ()),
        )
    };
    LasxStatus::Ok.write(status);
    crate::ops::cross3_batch::cross3_batch(ax, ay, az, bx, by, bz, ox, oy, oz);
}

/// 带错误通道的批量单位化。
/// C 签名：`void lasx_unitize3_batch_checked(const double*, const double*, const double*, double*, double*, double*, int, int*)`
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn lasx_unitize3_batch_checked(
    x: *const f64,
    y: *const f64,
    z: *const f64,
    ox: *mut f64,
    oy: *mut f64,
    oz: *mut f64,
    n: i32,
    status: *mut i32,
) {
    let n = or_fail!(checked_len(n), status, ());
    let (x, y, z) = unsafe {
        (
            or_fail!(checked_slice(x, n), status, ()),
            or_fail!(checked_slice(y, n), status, ()),
            or_fail!(checked_slice(z, n), status, ()),
        )
    };
    let (ox, oy, oz) = unsafe {
        (
            or_fail!(checked_slice_mut(ox, n), status, ()),
            or_fail!(checked_slice_mut(oy, n), status, ()),
            or_fail!(checked_slice_mut(oz, n), status, ()),
        )
    };
    LasxStatus::Ok.write(status);
    crate::ops::unitize3_batch::unitize3_batch(x, y, z, ox, oy, oz);
}

/// 带错误通道的批量 `o = M·v`。
/// C 签名：`void lasx_mat3_mul_vec3_batch_checked(const double *m0..m8, const double*, ..., double*, ..., int, int*)`
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn lasx_mat3_mul_vec3_batch_checked(
    m0: *const f64,
    m1: *const f64,
    m2: *const f64,
    m3: *const f64,
    m4: *const f64,
    m5: *const f64,
    m6: *const f64,
    m7: *const f64,
    m8: *const f64,
    x: *const f64,
    y: *const f64,
    z: *const f64,
    ox: *mut f64,
    oy: *mut f64,
    oz: *mut f64,
    n: i32,
    status: *mut i32,
) {
    let n = or_fail!(checked_len(n), status, ());
    // 先逐个校验 9 个矩阵数组，再组装成切片数组（`or_fail!` 会直接 return）
    let mp = [m0, m1, m2, m3, m4, m5, m6, m7, m8];
    let mut m: [&[f64]; 9] = [&[]; 9];
    for (k, p) in mp.iter().enumerate() {
        // SAFETY: FFI 约定——每个矩阵数组各 n 个可读元素。
        m[k] = unsafe { or_fail!(checked_slice(*p, n), status, ()) };
    }
    let (x, y, z) = unsafe {
        (
            or_fail!(checked_slice(x, n), status, ()),
            or_fail!(checked_slice(y, n), status, ()),
            or_fail!(checked_slice(z, n), status, ()),
        )
    };
    let (ox, oy, oz) = unsafe {
        (
            or_fail!(checked_slice_mut(ox, n), status, ()),
            or_fail!(checked_slice_mut(oy, n), status, ()),
            or_fail!(checked_slice_mut(oz, n), status, ()),
        )
    };
    LasxStatus::Ok.write(status);
    crate::ops::mat3_mul_vec3_batch::mat3_mul_vec3_batch(m, x, y, z, ox, oy, oz);
}

/// 带错误通道的批量四元数单位化（原地）。
/// C 签名：`void lasx_quat_normalize_batch_checked(double*, double*, double*, double*, int, int*)`
#[unsafe(no_mangle)]
pub extern "C" fn lasx_quat_normalize_batch_checked(
    qw: *mut f64,
    qx: *mut f64,
    qy: *mut f64,
    qz: *mut f64,
    n: i32,
    status: *mut i32,
) {
    let n = or_fail!(checked_len(n), status, ());
    // SAFETY: 见上。
    let (qw, qx, qy, qz) = unsafe {
        (
            or_fail!(checked_slice_mut(qw, n), status, ()),
            or_fail!(checked_slice_mut(qx, n), status, ()),
            or_fail!(checked_slice_mut(qy, n), status, ()),
            or_fail!(checked_slice_mut(qz, n), status, ()),
        )
    };
    LasxStatus::Ok.write(status);
    crate::ops::quat_normalize_batch::quat_normalize_batch(qw, qx, qy, qz);
}

/// 带错误通道的批量化四元数乘法。
/// C 签名：`void lasx_quat_mul_batch_checked(const double *aw..az, const double *bw..bz, double *ow..oz, int, int*)`
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn lasx_quat_mul_batch_checked(
    aw: *const f64,
    ax: *const f64,
    ay: *const f64,
    az: *const f64,
    bw: *const f64,
    bx: *const f64,
    by: *const f64,
    bz: *const f64,
    ow: *mut f64,
    ox: *mut f64,
    oy: *mut f64,
    oz: *mut f64,
    n: i32,
    status: *mut i32,
) {
    let n = or_fail!(checked_len(n), status, ());
    // SAFETY: 见上。
    let (aw, ax, ay, az, bw, bx, by, bz) = unsafe {
        (
            or_fail!(checked_slice(aw, n), status, ()),
            or_fail!(checked_slice(ax, n), status, ()),
            or_fail!(checked_slice(ay, n), status, ()),
            or_fail!(checked_slice(az, n), status, ()),
            or_fail!(checked_slice(bw, n), status, ()),
            or_fail!(checked_slice(bx, n), status, ()),
            or_fail!(checked_slice(by, n), status, ()),
            or_fail!(checked_slice(bz, n), status, ()),
        )
    };
    let (ow, ox, oy, oz) = unsafe {
        (
            or_fail!(checked_slice_mut(ow, n), status, ()),
            or_fail!(checked_slice_mut(ox, n), status, ()),
            or_fail!(checked_slice_mut(oy, n), status, ()),
            or_fail!(checked_slice_mut(oz, n), status, ()),
        )
    };
    LasxStatus::Ok.write(status);
    crate::ops::quat_mul_batch::quat_mul_batch(aw, ax, ay, az, bw, bx, by, bz, ow, ox, oy, oz);
}

/// 带错误通道的批量四元数旋转。
/// C 签名：`void lasx_quat_rotate_batch_checked(const double *qw..qz, const double *vx..vz, double *ox..oz, int, int*)`
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn lasx_quat_rotate_batch_checked(
    qw: *const f64,
    qx: *const f64,
    qy: *const f64,
    qz: *const f64,
    vx: *const f64,
    vy: *const f64,
    vz: *const f64,
    ox: *mut f64,
    oy: *mut f64,
    oz: *mut f64,
    n: i32,
    status: *mut i32,
) {
    let n = or_fail!(checked_len(n), status, ());
    // SAFETY: 见上。
    let (qw, qx, qy, qz, vx, vy, vz) = unsafe {
        (
            or_fail!(checked_slice(qw, n), status, ()),
            or_fail!(checked_slice(qx, n), status, ()),
            or_fail!(checked_slice(qy, n), status, ()),
            or_fail!(checked_slice(qz, n), status, ()),
            or_fail!(checked_slice(vx, n), status, ()),
            or_fail!(checked_slice(vy, n), status, ()),
            or_fail!(checked_slice(vz, n), status, ()),
        )
    };
    let (ox, oy, oz) = unsafe {
        (
            or_fail!(checked_slice_mut(ox, n), status, ()),
            or_fail!(checked_slice_mut(oy, n), status, ()),
            or_fail!(checked_slice_mut(oz, n), status, ()),
        )
    };
    LasxStatus::Ok.write(status);
    crate::ops::quat_rotate_batch::quat_rotate_batch(qw, qx, qy, qz, vx, vy, vz, ox, oy, oz);
}

/// 带错误通道的批量四元数 → DCM。
/// C 签名：`void lasx_quat_to_dcm_batch_checked(const double *qw..qz, double *m0..m8, int, int*)`
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn lasx_quat_to_dcm_batch_checked(
    qw: *const f64,
    qx: *const f64,
    qy: *const f64,
    qz: *const f64,
    m0: *mut f64,
    m1: *mut f64,
    m2: *mut f64,
    m3: *mut f64,
    m4: *mut f64,
    m5: *mut f64,
    m6: *mut f64,
    m7: *mut f64,
    m8: *mut f64,
    n: i32,
    status: *mut i32,
) {
    let n = or_fail!(checked_len(n), status, ());
    // SAFETY: 见上。
    let (qw, qx, qy, qz) = unsafe {
        (
            or_fail!(checked_slice(qw, n), status, ()),
            or_fail!(checked_slice(qx, n), status, ()),
            or_fail!(checked_slice(qy, n), status, ()),
            or_fail!(checked_slice(qz, n), status, ()),
        )
    };
    // 9 个输出数组**必须逐个走 checked_slice_mut**：`_checked` 变体的契约就是
    // "NULL + n > 0 报 NullPointer 而不是 UB"，这里曾经漏掉（其余 checked 包装都做了）。
    // 注意不能把 `or_fail!` 放进闭包：它的失败分支是裸 `return`，在闭包里会退化成 `()`。
    let mp = [m0, m1, m2, m3, m4, m5, m6, m7, m8];
    let mut tmp: Vec<&mut [f64]> = Vec::with_capacity(9);
    for &p in &mp {
        tmp.push(or_fail!(unsafe { checked_slice_mut(p, n) }, status, ()));
    }
    let m: [&mut [f64]; 9] = match tmp.try_into() {
        Ok(a) => a,
        // 容量恒为 9，不可能走到这里；不写状态是安全的（宁可报错也不要 UB）
        Err(_) => return,
    };
    LasxStatus::Ok.write(status);
    crate::ops::quat_to_dcm_batch::quat_to_dcm_batch(qw, qx, qy, qz, m);
}

/// 带错误通道的行内 softmax。
///
/// C 签名：`void lasx_softmax_rows_checked(const float*, const float*, float*, int, int, float, int*)`
///
/// `mask` 为 NULL 是**合法**输入（无 mask），与 `x`/`out` 的空指针区别对待：
/// `n_rows × n_cols > 0` 时 `x`/`out` 为空报 `NullPointer`。
#[unsafe(no_mangle)]
pub extern "C" fn lasx_softmax_rows_checked(
    x: *const f32,
    mask: *const f32,
    out: *mut f32,
    n_rows: i32,
    n_cols: i32,
    scale: f32,
    status: *mut i32,
) {
    if n_rows < 0 || n_cols < 0 {
        LasxStatus::BadShape.write(status);
        return;
    }
    let (rows, cols) = (n_rows as usize, n_cols as usize);
    let n = or_fail!(checked_mul(rows, cols), status, ());
    // SAFETY: 见上；`mask` 允许 NULL（下面单独判）。
    let (x, out) = unsafe {
        (
            or_fail!(checked_slice(x, n), status, ()),
            or_fail!(checked_slice_mut(out, n), status, ()),
        )
    };
    let mask = if mask.is_null() {
        &[][..]
    } else {
        // SAFETY: 调用方声明 `mask` 与 `x` 同形状。
        or_fail!(unsafe { checked_slice(mask, n) }, status, ())
    };
    LasxStatus::Ok.write(status);
    crate::ops::softmax_rows::softmax_rows(x, mask, scale, rows, cols, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 校验失败时：状态被写入、返回值是安全中性值、且**不触碰**输出缓冲。
    #[test]
    fn test_null_pointer_is_reported() {
        let mut st = 999;
        let d = lasx_dot_checked(std::ptr::null(), std::ptr::null(), 4, &mut st);
        assert_eq!(st, LasxStatus::NullPointer as i32);
        assert_eq!(d, 0.0);

        st = 999;
        let s = lasx_sum_checked(std::ptr::null(), 4, &mut st);
        assert_eq!(st, LasxStatus::NullPointer as i32);
        assert_eq!(s, 0.0);

        // 长度为 0 时空指针合法
        st = 999;
        let s = lasx_sum_checked(std::ptr::null(), 0, &mut st);
        assert_eq!(st, LasxStatus::Ok as i32);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn test_negative_length_is_reported() {
        let a = [1.0f32; 4];
        let mut st = 999;
        let d = lasx_dot_checked(a.as_ptr(), a.as_ptr(), -1, &mut st);
        assert_eq!(st, LasxStatus::NegativeLength as i32);
        assert_eq!(d, 0.0);
    }

    /// `lasx_quat_to_dcm_batch_checked` 的 9 个输出也必须做空指针校验（曾经漏掉）。
    #[test]
    fn test_quat_to_dcm_null_output_is_reported() {
        let q = [1.0f64, 0.0, 0.0, 0.0];
        let mut out = [0.0f64; 9];
        let p = out.as_mut_ptr();
        // 其余 8 个有效、只有第 9 个是 NULL ⇒ 仍须报 NullPointer 且不写任何输出
        let mut st = 999;
        lasx_quat_to_dcm_batch_checked(
            q.as_ptr(),
            q.as_ptr(),
            q.as_ptr(),
            q.as_ptr(),
            p,
            unsafe { p.add(1) },
            unsafe { p.add(2) },
            unsafe { p.add(3) },
            unsafe { p.add(4) },
            unsafe { p.add(5) },
            unsafe { p.add(6) },
            unsafe { p.add(7) },
            std::ptr::null_mut(),
            1,
            &mut st,
        );
        assert_eq!(st, LasxStatus::NullPointer as i32);
        assert!(out.iter().all(|&v| v == 0.0), "校验失败时不得触碰输出");
    }

    #[test]
    fn test_status_out_may_be_null() {
        let a = [1.0f32; 4];
        // 不关心状态：不崩、返回中性值
        let d = lasx_dot_checked(std::ptr::null(), a.as_ptr(), 4, std::ptr::null_mut());
        assert_eq!(d, 0.0);
    }

    #[test]
    fn test_valid_input_reports_ok_and_computes() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [1.0f32, 1.0, 1.0, 1.0];
        let mut st = 999;
        let d = lasx_dot_checked(a.as_ptr(), b.as_ptr(), 4, &mut st);
        assert_eq!(st, LasxStatus::Ok as i32);
        assert!((d - 10.0).abs() < 1e-6, "期望 10.0，得到 {d}");
    }

    #[test]
    fn test_matmul_checked_shape_errors() {
        let a = [1.0f32; 16];
        let b = [1.0f32; 16];
        let mut c = [0.0f32; 16];
        // 负维度
        let mut st = 999;
        lasx_matmul_checked(-1, 4, 4, a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), &mut st);
        assert_eq!(st, LasxStatus::NegativeLength as i32);
        // 合法 4×4×4
        st = 999;
        lasx_matmul_checked(4, 4, 4, a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), &mut st);
        assert_eq!(st, LasxStatus::Ok as i32);
        // 单位阵 × 全 1 阵 → 每行 4.0
        for v in c.iter() {
            assert!((v - 4.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_physics_constant_validation() {
        let n = 4usize;
        let r = vec![7.0e6f64; n];
        let (mut ax, mut ay, mut az) = (vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]);
        let mut st = 999;
        // mu <= 0
        lasx_j2_accel_batch_checked(
            r.as_ptr(),
            r.as_ptr(),
            r.as_ptr(),
            0.0,
            1e-3,
            6.378e6,
            ax.as_mut_ptr(),
            ay.as_mut_ptr(),
            az.as_mut_ptr(),
            n as i32,
            &mut st,
        );
        assert_eq!(st, LasxStatus::NonPositiveConstant as i32);
        // j2 = NaN
        st = 999;
        lasx_j2_accel_batch_checked(
            r.as_ptr(),
            r.as_ptr(),
            r.as_ptr(),
            3.986e14,
            f64::NAN,
            6.378e6,
            ax.as_mut_ptr(),
            ay.as_mut_ptr(),
            az.as_mut_ptr(),
            n as i32,
            &mut st,
        );
        assert_eq!(st, LasxStatus::NonFiniteConstant as i32);
        // 合法
        st = 999;
        lasx_j2_accel_batch_checked(
            r.as_ptr(),
            r.as_ptr(),
            r.as_ptr(),
            3.986e14,
            1.082_626_68e-3,
            6.378_137e6,
            ax.as_mut_ptr(),
            ay.as_mut_ptr(),
            az.as_mut_ptr(),
            n as i32,
            &mut st,
        );
        assert_eq!(st, LasxStatus::Ok as i32);
        assert!(ax.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_checked_matches_unchecked_on_valid_input() {
        let n = 33usize;
        let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).cos()).collect();
        let want = crate::ffi::reduce::lasx_dot(a.as_ptr(), b.as_ptr(), n as i32);
        let mut st = 999;
        let got = lasx_dot_checked(a.as_ptr(), b.as_ptr(), n as i32, &mut st);
        assert_eq!(st, LasxStatus::Ok as i32);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "checked 与 unchecked 必须逐位一致"
        );
    }
}
