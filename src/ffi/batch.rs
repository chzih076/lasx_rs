//! SOA 批量几何内核的 C ABI 导出。

/// 批量 3 分量模长 `out[i] = √(xs[i]²+ys[i]²+zs[i]²)`。
///
/// C 签名：`void lasx_norm3_batch(const double *xs, const double *ys, const double *zs, double *out, int n)`
#[unsafe(no_mangle)]
pub extern "C" fn lasx_norm3_batch(
    xs: *const f64,
    ys: *const f64,
    zs: *const f64,
    out: *mut f64,
    n: i32,
) {
    let n = n as usize;
    // SAFETY: FFI 约定——三个输入各 n 个可读元素，`out` n 个可写元素。
    let xs = unsafe { std::slice::from_raw_parts(xs, n) };
    let ys = unsafe { std::slice::from_raw_parts(ys, n) };
    let zs = unsafe { std::slice::from_raw_parts(zs, n) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, n) };
    crate::ops::norm3_batch::norm3_batch(xs, ys, zs, out);
}

/// 批量缩放加 `o[i] = a[i] + s·b[i]`（x/y/z 三分量独立）。
///
/// C 签名：
/// `void lasx_vec3_add_scaled_batch(const double *ax, ..., const double *bz, double s, double *ox, ..., double *oz, int n)`
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn lasx_vec3_add_scaled_batch(
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
) {
    let n = n as usize;
    // SAFETY: FFI 约定——六个输入各 n 个可读元素，三个输出各 n 个可写元素。
    let ax = unsafe { std::slice::from_raw_parts(ax, n) };
    let ay = unsafe { std::slice::from_raw_parts(ay, n) };
    let az = unsafe { std::slice::from_raw_parts(az, n) };
    let bx = unsafe { std::slice::from_raw_parts(bx, n) };
    let by = unsafe { std::slice::from_raw_parts(by, n) };
    let bz = unsafe { std::slice::from_raw_parts(bz, n) };
    let ox = unsafe { std::slice::from_raw_parts_mut(ox, n) };
    let oy = unsafe { std::slice::from_raw_parts_mut(oy, n) };
    let oz = unsafe { std::slice::from_raw_parts_mut(oz, n) };
    crate::ops::vec3_add_scaled_batch::vec3_add_scaled_batch(ax, ay, az, bx, by, bz, s, ox, oy, oz);
}

/// 批量 2D 距离 `d[i] = √(dx²+dy²)`。
///
/// C 签名：
/// `void lasx_batch_distance2d(float px, float py, const float *xs, const float *ys, float *out, int n)`
#[unsafe(no_mangle)]
pub extern "C" fn lasx_batch_distance2d(
    px: f32,
    py: f32,
    xs: *const f32,
    ys: *const f32,
    out: *mut f32,
    n: i32,
) {
    let n = n as usize;
    // SAFETY: FFI 约定——`xs`/`ys` 各 n 个可读元素，`out` n 个可写元素。
    let xs = unsafe { std::slice::from_raw_parts(xs, n) };
    let ys = unsafe { std::slice::from_raw_parts(ys, n) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, n) };
    crate::ops::batch_distance2d::batch_distance2d(px, py, xs, ys, out);
}
