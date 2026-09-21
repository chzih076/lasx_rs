//! 标量基线实现（作为"加速比"的分母）。
//!
//! SOA 内核的签名与 C ABI 一一对应，参数偏多不做拆分。
#![allow(clippy::too_many_arguments)]
//!
//! 注意：`target-feature=+lasx` 下 LLVM 会把其中大部分自动向量化成 LASX，
//! 因此这一列的含义是"手写 intrinsic vs 编译器自动向量化"，只有串行累加依赖强
//! 的 `scalar_dot` / `scalar_sum` / `scalar_dot_f64` / `scalar_dot_q4` 是真标量。

#[inline(never)]
pub fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[inline(never)]
pub fn scalar_sum(x: &[f32]) -> f32 {
    let mut s = 0f32;
    for &v in x {
        s += v;
    }
    s
}

#[inline(never)]
pub fn scalar_axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    for i in 0..x.len() {
        y[i] += alpha * x[i];
    }
}

#[inline(never)]
pub fn scalar_dot_f64(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0f64;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[inline(never)]
pub fn scalar_dot_i8(a: &[i8], b: &[i8]) -> i32 {
    let mut s = 0i64;
    for i in 0..a.len() {
        s += (a[i] as i64) * (b[i] as i64);
    }
    s as i32
}

#[inline(never)]
pub fn scalar_dot_q4(qa: &[u8], sa: &[f32], qb: &[u8], sb: &[f32]) -> f64 {
    let n = qa.len();
    let mut acc = 0f64;
    for j in 0..n {
        let la = (qa[j] & 0x0f) as i64;
        let ha = ((qa[j] >> 4) & 0x0f) as i64;
        let lb = (qb[j] & 0x0f) as i64;
        let hb = ((qb[j] >> 4) & 0x0f) as i64;
        let sc = sa[j / 32] * sb[j / 32];
        acc += (sc as f64) * ((la * lb + ha * hb) as f64);
    }
    acc
}

/// 朴素 i-k-j 三重循环（B 不转置，逐行顺序访问）
#[inline(never)]
pub fn scalar_matmul(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for v in c[..m * n].iter_mut() {
        *v = 0.0;
    }
    for i in 0..m {
        for p in 0..k {
            let aip = a[i * k + p];
            let brow = &b[p * n..p * n + n];
            let crow = &mut c[i * n..i * n + n];
            for j in 0..n {
                crow[j] += aip * brow[j];
            }
        }
    }
}

#[inline(never)]
pub fn scalar_matmul_f64(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    for v in c[..m * n].iter_mut() {
        *v = 0.0;
    }
    for i in 0..m {
        for p in 0..k {
            let aip = a[i * k + p];
            let brow = &b[p * n..p * n + n];
            let crow = &mut c[i * n..i * n + n];
            for j in 0..n {
                crow[j] += aip * brow[j];
            }
        }
    }
}

#[inline(never)]
pub fn scalar_norm3(xs: &[f64], ys: &[f64], zs: &[f64], out: &mut [f64]) {
    for i in 0..xs.len() {
        out[i] = (xs[i] * xs[i] + (ys[i] * ys[i] + zs[i] * zs[i])).sqrt();
    }
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub fn scalar_vec3_add_scaled(
    ax: &[f64],
    ay: &[f64],
    az: &[f64],
    bx: &[f64],
    by: &[f64],
    bz: &[f64],
    s: f64,
    ox: &mut [f64],
    oy: &mut [f64],
    oz: &mut [f64],
) {
    for i in 0..ax.len() {
        ox[i] = f64::mul_add(s, bx[i], ax[i]);
        oy[i] = f64::mul_add(s, by[i], ay[i]);
        oz[i] = f64::mul_add(s, bz[i], az[i]);
    }
}

#[inline(never)]
pub fn scalar_j2_accel(
    rx: &[f64],
    ry: &[f64],
    rz: &[f64],
    mu: f64,
    j2: f64,
    re: f64,
    ax: &mut [f64],
    ay: &mut [f64],
    az: &mut [f64],
) {
    let j2k = 1.5 * j2 * mu * re * re;
    for j in 0..rx.len() {
        let rm2 = rx[j] * rx[j] + (ry[j] * ry[j] + rz[j] * rz[j]);
        let rm = rm2.sqrt();
        let rm3 = rm * rm2;
        let rm5 = rm3 * rm2;
        let zr2 = (rz[j] * rz[j]) / rm2;
        let k = j2k / rm5;
        let vcen = -mu / rm3;
        ax[j] = f64::mul_add(k * rx[j], 5.0 * zr2 - 1.0, vcen * rx[j]);
        ay[j] = f64::mul_add(k * ry[j], 5.0 * zr2 - 1.0, vcen * ry[j]);
        az[j] = f64::mul_add(k * rz[j], 5.0 * zr2 - 3.0, vcen * rz[j]);
    }
}

/// 逐发弹道欧拉步的**循环体**（切片版，等价于库内 `euler_step` 被内联进循环）
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub fn scalar_ballistic(
    x: &mut [f32],
    y: &mut [f32],
    z: &mut [f32],
    vx: &mut [f32],
    vy: &mut [f32],
    vz: &mut [f32],
    k: &[f32],
    dt: f32,
    g: f32,
) {
    for i in 0..x.len() {
        let v = (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]).sqrt();
        let drag = k[i] * v;
        vx[i] -= drag * vx[i] * dt;
        vy[i] -= (drag * vy[i] + g) * dt;
        vz[i] -= drag * vz[i] * dt;
        x[i] += vx[i] * dt;
        y[i] += vy[i] * dt;
        z[i] += vz[i] * dt;
    }
}

#[inline(never)]
pub fn scalar_batch_distance2d(px: f32, py: f32, xs: &[f32], ys: &[f32], out: &mut [f32]) {
    for j in 0..xs.len() {
        let dx = xs[j] - px;
        let dy = ys[j] - py;
        out[j] = (dx * dx + dy * dy).sqrt();
    }
}

/// 与 `src/lib.rs` 的 `rk4_j2_step_scalar` 同式（FMA 版）
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub fn scalar_rk4_batch(
    rx: &mut [f64],
    ry: &mut [f64],
    rz: &mut [f64],
    vx: &mut [f64],
    vy: &mut [f64],
    vz: &mut [f64],
    mu: f64,
    j2: f64,
    re: f64,
    h: f64,
) {
    let j2k = 1.5 * j2 * mu * re * re;
    let accel = |x: f64, y: f64, z: f64| -> [f64; 3] {
        let rm2 = x * x + (y * y + z * z);
        let rm = rm2.sqrt();
        let rm3 = rm * rm2;
        let rm5 = rm3 * rm2;
        let zr2 = (z * z) / rm2;
        let k = j2k / rm5;
        let vcen = -mu / rm3;
        [
            f64::mul_add(k * x, 5.0 * zr2 - 1.0, vcen * x),
            f64::mul_add(k * y, 5.0 * zr2 - 1.0, vcen * y),
            f64::mul_add(k * z, 5.0 * zr2 - 3.0, vcen * z),
        ]
    };
    for i in 0..rx.len() {
        let (x0, y0, z0, vx0, vy0, vz0) = (rx[i], ry[i], rz[i], vx[i], vy[i], vz[i]);
        let hh = 0.5 * h;
        let h6 = h / 6.0;
        let k1 = accel(x0, y0, z0);
        let r2 = (
            hh.mul_add(vx0, x0),
            hh.mul_add(vy0, y0),
            hh.mul_add(vz0, z0),
        );
        let v2 = (
            hh.mul_add(k1[0], vx0),
            hh.mul_add(k1[1], vy0),
            hh.mul_add(k1[2], vz0),
        );
        let k2 = accel(r2.0, r2.1, r2.2);
        let r3 = (
            hh.mul_add(v2.0, x0),
            hh.mul_add(v2.1, y0),
            hh.mul_add(v2.2, z0),
        );
        let v3 = (
            hh.mul_add(k2[0], vx0),
            hh.mul_add(k2[1], vy0),
            hh.mul_add(k2[2], vz0),
        );
        let k3 = accel(r3.0, r3.1, r3.2);
        let r4 = (
            h.mul_add(v3.0, x0),
            h.mul_add(v3.1, y0),
            h.mul_add(v3.2, z0),
        );
        let v4 = (
            h.mul_add(k3[0], vx0),
            h.mul_add(k3[1], vy0),
            h.mul_add(k3[2], vz0),
        );
        let k4 = accel(r4.0, r4.1, r4.2);
        let l = (
            2.0_f64.mul_add(v2.0, vx0) + 2.0_f64.mul_add(v3.0, v4.0),
            2.0_f64.mul_add(v2.1, vy0) + 2.0_f64.mul_add(v3.1, v4.1),
            2.0_f64.mul_add(v2.2, vz0) + 2.0_f64.mul_add(v3.2, v4.2),
        );
        let kk = (
            2.0_f64.mul_add(k2[0], k1[0]) + 2.0_f64.mul_add(k3[0], k4[0]),
            2.0_f64.mul_add(k2[1], k1[1]) + 2.0_f64.mul_add(k3[1], k4[1]),
            2.0_f64.mul_add(k2[2], k1[2]) + 2.0_f64.mul_add(k3[2], k4[2]),
        );
        rx[i] = h6.mul_add(l.0, x0);
        ry[i] = h6.mul_add(l.1, y0);
        rz[i] = h6.mul_add(l.2, z0);
        vx[i] = h6.mul_add(kk.0, vx0);
        vy[i] = h6.mul_add(kk.1, vy0);
        vz[i] = h6.mul_add(kk.2, vz0);
    }
}
