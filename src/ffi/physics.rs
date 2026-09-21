//! 物理内核（弹道 / J2 引力 / RK4 轨道传播）的 C ABI 导出。

/// 批量弹道欧拉步（f32 SOA）。
///
/// C 签名：
/// `void lasx_ballistic_step(float *x, ..., float *vz, const float *k, int n, float dt, float g)`
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn lasx_ballistic_step(
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
) {
    let n = n as usize;
    // SAFETY: FFI 约定——六个状态数组各 n 个可写元素，`k` n 个可读元素。
    let x = unsafe { std::slice::from_raw_parts_mut(x, n) };
    let y = unsafe { std::slice::from_raw_parts_mut(y, n) };
    let z = unsafe { std::slice::from_raw_parts_mut(z, n) };
    let vx = unsafe { std::slice::from_raw_parts_mut(vx, n) };
    let vy = unsafe { std::slice::from_raw_parts_mut(vy, n) };
    let vz = unsafe { std::slice::from_raw_parts_mut(vz, n) };
    let k = unsafe { std::slice::from_raw_parts(k, n) };
    crate::ops::ballistic_step::ballistic_step(x, y, z, vx, vy, vz, k, dt, g);
}

/// 批量中心引力 + J2 摄动加速度（f64 SOA）。
///
/// C 签名：
/// `void lasx_j2_accel_batch(const double *rx, ..., const double *rz, double mu, double j2, double re, double *ax, ..., double *az, int n)`
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn lasx_j2_accel_batch(
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
) {
    let n = n as usize;
    // SAFETY: FFI 约定——三个位置数组各 n 个可读元素，三个输出各 n 个可写元素。
    let rx = unsafe { std::slice::from_raw_parts(rx, n) };
    let ry = unsafe { std::slice::from_raw_parts(ry, n) };
    let rz = unsafe { std::slice::from_raw_parts(rz, n) };
    let ax = unsafe { std::slice::from_raw_parts_mut(ax, n) };
    let ay = unsafe { std::slice::from_raw_parts_mut(ay, n) };
    let az = unsafe { std::slice::from_raw_parts_mut(az, n) };
    crate::ops::j2_accel_batch::j2_accel_batch(rx, ry, rz, mu, j2, re, ax, ay, az);
}

/// 批量 RK4 J2 轨道步（f64 SOA，原地更新）。
///
/// C 签名：
/// `void lasx_rk4_j2_step_batch(double *rx, ..., double *vz, double mu, double j2, double re, double dt, int n)`
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn lasx_rk4_j2_step_batch(
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
) {
    let n = n as usize;
    // SAFETY: FFI 约定——六个状态数组各 n 个可读可写元素（原地更新）。
    let rx = unsafe { std::slice::from_raw_parts_mut(rx, n) };
    let ry = unsafe { std::slice::from_raw_parts_mut(ry, n) };
    let rz = unsafe { std::slice::from_raw_parts_mut(rz, n) };
    let vx = unsafe { std::slice::from_raw_parts_mut(vx, n) };
    let vy = unsafe { std::slice::from_raw_parts_mut(vy, n) };
    let vz = unsafe { std::slice::from_raw_parts_mut(vz, n) };
    crate::ops::rk4_j2_step_batch::rk4_j2_step_batch(rx, ry, rz, vx, vy, vz, mu, j2, re, dt);
}
