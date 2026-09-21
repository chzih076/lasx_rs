//! 批量姿态/几何内核的 C ABI 导出（f64 SOA，全部 LASX + LSX 双路径）。
//!
//! 约定与 loong-sci `attitude` / `orbit` 的标量实现逐位一致（四元数**标量在前**、
//! 旋转为**体→惯**、`|q| < 1e-15` 视为单位四元数）。
//!
//! 全部内核**允许输出与输入别名**（向量路径先取完本轮的输入再写回）。

use crate::ops;

/// 批量三维叉积 `o = a × b`。
///
/// C 签名：`void lasx_cross3_batch(const double *ax, ..., const double *bz, double *ox, ..., double *oz, int n)`
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn lasx_cross3_batch(
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
) {
    let n = n as usize;
    // SAFETY: FFI 约定——a/b 各 3 个数组可读，o 各 3 个数组可写，长度均为 n。
    let (ax, ay, az, bx, by, bz) = unsafe {
        (
            std::slice::from_raw_parts(ax, n),
            std::slice::from_raw_parts(ay, n),
            std::slice::from_raw_parts(az, n),
            std::slice::from_raw_parts(bx, n),
            std::slice::from_raw_parts(by, n),
            std::slice::from_raw_parts(bz, n),
        )
    };
    let (ox, oy, oz) = unsafe {
        (
            std::slice::from_raw_parts_mut(ox, n),
            std::slice::from_raw_parts_mut(oy, n),
            std::slice::from_raw_parts_mut(oz, n),
        )
    };
    ops::cross3_batch::cross3_batch(ax, ay, az, bx, by, bz, ox, oy, oz);
}

/// 批量三维单位化 `o = v/|v|`（零向量 → `(0,0,0)`）。
///
/// C 签名：`void lasx_unitize3_batch(const double *x, const double *y, const double *z, double *ox, double *oy, double *oz, int n)`
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn lasx_unitize3_batch(
    x: *const f64,
    y: *const f64,
    z: *const f64,
    ox: *mut f64,
    oy: *mut f64,
    oz: *mut f64,
    n: i32,
) {
    let n = n as usize;
    // SAFETY: FFI 约定——x/y/z 可读、ox/oy/oz 可写，长度均为 n。
    let (x, y, z) = unsafe {
        (
            std::slice::from_raw_parts(x, n),
            std::slice::from_raw_parts(y, n),
            std::slice::from_raw_parts(z, n),
        )
    };
    let (ox, oy, oz) = unsafe {
        (
            std::slice::from_raw_parts_mut(ox, n),
            std::slice::from_raw_parts_mut(oy, n),
            std::slice::from_raw_parts_mut(oz, n),
        )
    };
    ops::unitize3_batch::unitize3_batch(x, y, z, ox, oy, oz);
}

/// 批量 `o = M·v`（`m0..m8` 行主序 3×3）。
///
/// C 签名：`void lasx_mat3_mul_vec3_batch(const double *m0, ..., const double *m8, const double *x, const double *y, const double *z, double *ox, double *oy, double *oz, int n)`
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn lasx_mat3_mul_vec3_batch(
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
) {
    let n = n as usize;
    let mp = [m0, m1, m2, m3, m4, m5, m6, m7, m8];
    // SAFETY: FFI 约定——9 个矩阵数组与 x/y/z 可读、ox/oy/oz 可写，长度均为 n。
    let m: [&[f64]; 9] = std::array::from_fn(|k| unsafe { std::slice::from_raw_parts(mp[k], n) });
    let (x, y, z) = unsafe {
        (
            std::slice::from_raw_parts(x, n),
            std::slice::from_raw_parts(y, n),
            std::slice::from_raw_parts(z, n),
        )
    };
    let (ox, oy, oz) = unsafe {
        (
            std::slice::from_raw_parts_mut(ox, n),
            std::slice::from_raw_parts_mut(oy, n),
            std::slice::from_raw_parts_mut(oz, n),
        )
    };
    ops::mat3_mul_vec3_batch::mat3_mul_vec3_batch(m, x, y, z, ox, oy, oz);
}

/// 批量四元数单位化（**原地**；`|q| < 1e-15` → 单位四元数）。
///
/// C 签名：`void lasx_quat_normalize_batch(double *qw, double *qx, double *qy, double *qz, int n)`
#[unsafe(no_mangle)]
pub extern "C" fn lasx_quat_normalize_batch(
    qw: *mut f64,
    qx: *mut f64,
    qy: *mut f64,
    qz: *mut f64,
    n: i32,
) {
    let n = n as usize;
    // SAFETY: FFI 约定——4 个分量数组各 n 个可读写元素。
    let (qw, qx, qy, qz) = unsafe {
        (
            std::slice::from_raw_parts_mut(qw, n),
            std::slice::from_raw_parts_mut(qx, n),
            std::slice::from_raw_parts_mut(qy, n),
            std::slice::from_raw_parts_mut(qz, n),
        )
    };
    ops::quat_normalize_batch::quat_normalize_batch(qw, qx, qy, qz);
}

/// 批量化四元数乘法（Hamilton 积，标量在前）。
///
/// C 签名：`void lasx_quat_mul_batch(const double *aw, ..., const double *bz, double *ow, ..., double *oz, int n)`
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn lasx_quat_mul_batch(
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
) {
    let n = n as usize;
    // SAFETY: FFI 约定——a/b 各 4 个数组可读、o 各 4 个数组可写，长度均为 n。
    let (aw, ax, ay, az, bw, bx, by, bz) = unsafe {
        (
            std::slice::from_raw_parts(aw, n),
            std::slice::from_raw_parts(ax, n),
            std::slice::from_raw_parts(ay, n),
            std::slice::from_raw_parts(az, n),
            std::slice::from_raw_parts(bw, n),
            std::slice::from_raw_parts(bx, n),
            std::slice::from_raw_parts(by, n),
            std::slice::from_raw_parts(bz, n),
        )
    };
    let (ow, ox, oy, oz) = unsafe {
        (
            std::slice::from_raw_parts_mut(ow, n),
            std::slice::from_raw_parts_mut(ox, n),
            std::slice::from_raw_parts_mut(oy, n),
            std::slice::from_raw_parts_mut(oz, n),
        )
    };
    ops::quat_mul_batch::quat_mul_batch(aw, ax, ay, az, bw, bx, by, bz, ow, ox, oy, oz);
}

/// 批量用四元数旋转向量（先单位化，再 `o = R(q)·v`，体→惯）。
///
/// C 签名：`void lasx_quat_rotate_batch(const double *qw, ..., const double *qz, const double *vx, const double *vy, const double *vz, double *ox, double *oy, double *oz, int n)`
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn lasx_quat_rotate_batch(
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
) {
    let n = n as usize;
    // SAFETY: FFI 约定——q 的 4 个与 v 的 3 个数组可读、o 的 3 个可写，长度均为 n。
    let (qw, qx, qy, qz, vx, vy, vz) = unsafe {
        (
            std::slice::from_raw_parts(qw, n),
            std::slice::from_raw_parts(qx, n),
            std::slice::from_raw_parts(qy, n),
            std::slice::from_raw_parts(qz, n),
            std::slice::from_raw_parts(vx, n),
            std::slice::from_raw_parts(vy, n),
            std::slice::from_raw_parts(vz, n),
        )
    };
    let (ox, oy, oz) = unsafe {
        (
            std::slice::from_raw_parts_mut(ox, n),
            std::slice::from_raw_parts_mut(oy, n),
            std::slice::from_raw_parts_mut(oz, n),
        )
    };
    ops::quat_rotate_batch::quat_rotate_batch(qw, qx, qy, qz, vx, vy, vz, ox, oy, oz);
}

/// 批量四元数 → 3×3 方向余弦阵（行主序，体→惯）。
///
/// C 签名：`void lasx_quat_to_dcm_batch(const double *qw, ..., const double *qz, double *m0, ..., double *m8, int n)`
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn lasx_quat_to_dcm_batch(
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
) {
    let n = n as usize;
    // SAFETY: FFI 约定——4 个分量数组可读、9 个输出数组可写，长度均为 n。
    let (qw, qx, qy, qz) = unsafe {
        (
            std::slice::from_raw_parts(qw, n),
            std::slice::from_raw_parts(qx, n),
            std::slice::from_raw_parts(qy, n),
            std::slice::from_raw_parts(qz, n),
        )
    };
    let mp = [m0, m1, m2, m3, m4, m5, m6, m7, m8];
    let m: [&mut [f64]; 9] =
        std::array::from_fn(|k| unsafe { std::slice::from_raw_parts_mut(mp[k], n) });
    ops::quat_to_dcm_batch::quat_to_dcm_batch(qw, qx, qy, qz, m);
}
