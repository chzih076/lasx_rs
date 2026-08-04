//! LASX 256 位向量引擎（Rust 版，llama.cpp 风格内核）
//! 通过 Dart FFI 调用，提供 8×float32 / 4×float64 FMA 加速。
#![feature(stdarch_loongarch)]

use std::arch::loongarch64::*;

type F32x8 = m256;
type F64x4 = m256d;

#[inline]
unsafe fn ld_f32(p: *const f32) -> F32x8 {
    std::mem::transmute(lasx_xvld(p as *const i8, 0))
}
#[inline]
unsafe fn st_f32(p: *mut f32, v: F32x8) {
    lasx_xvst(std::mem::transmute(v), p as *mut i8, 0);
}
#[inline]
unsafe fn ld_f64(p: *const f64) -> F64x4 {
    std::mem::transmute(lasx_xvld(p as *const i8, 0))
}
#[inline]
unsafe fn st_f64(p: *mut f64, v: F64x4) {
    lasx_xvst(std::mem::transmute(v), p as *mut i8, 0);
}
#[inline]
unsafe fn zero_f32() -> F32x8 {
    std::mem::transmute(lasx_xvldi(0))
}
#[inline]
unsafe fn zero_f64() -> F64x4 {
    std::mem::transmute(lasx_xvldi(0))
}
#[inline]
fn splat_f32(x: f32) -> F32x8 {
    let bits = x.to_bits() as i32;
    unsafe { std::mem::transmute(lasx_xvreplgr2vr_w(bits)) }
}

// ===== LSX 128 位路径（兼容无 LASX 的龙芯 CPU）=====
type F32x4 = m128;
type F64x2 = m128d;

/// 运行时检测 LASX（cpucfg 读 CPUCFG2 bit7）
static HAS_LASX: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
fn has_lasx() -> bool {
    *HAS_LASX.get_or_init(|| unsafe {
        let mut cfg2: u32;
        std::arch::asm!("cpucfg {}, {}", out(reg) cfg2, in(reg) 2u32);
        (cfg2 & (1 << 7)) != 0
    })
}

#[inline]
unsafe fn ld4_f32(p: *const f32) -> F32x4 {
    std::mem::transmute(lsx_vld(p as *const i8, 0))
}
#[inline]
unsafe fn st4_f32(p: *mut f32, v: F32x4) {
    lsx_vst(std::mem::transmute(v), p as *mut i8, 0);
}
#[inline]
unsafe fn zero4_f32() -> F32x4 {
    std::mem::transmute(lsx_vldi(0))
}
#[inline]
unsafe fn ld2_f64(p: *const f64) -> F64x2 {
    std::mem::transmute(lsx_vld(p as *const i8, 0))
}
#[inline]
unsafe fn st2_f64(p: *mut f64, v: F64x2) {
    lsx_vst(std::mem::transmute(v), p as *mut i8, 0);
}
#[inline]
unsafe fn zero2_f64() -> F64x2 {
    std::mem::transmute(lsx_vldi(0))
}
#[inline]
fn splat4_f32(x: f32) -> F32x4 {
    let bits = x.to_bits() as i32;
    unsafe { std::mem::transmute(lsx_vreplgr2vr_w(bits)) }
}

/// 点积（8×f32 FMA）
#[unsafe(no_mangle)]
pub extern "C" fn lasx_dot(a: *const f32, b: *const f32, n: i32) -> f32 {
    let n = n as usize;
    let a = unsafe { std::slice::from_raw_parts(a, n) };
    let b = unsafe { std::slice::from_raw_parts(b, n) };
    if !has_lasx() {
        // LSX 128 位路径
        let mut acc = unsafe { zero4_f32() };
        let mut acc_d = 0f64;
        let mut i = 0;
        while i + 64 <= n {
            for _ in 0..16 {
                let va = unsafe { ld4_f32(a.as_ptr().add(i)) };
                let vb = unsafe { ld4_f32(b.as_ptr().add(i)) };
                acc = unsafe { lsx_vfmadd_s(va, vb, acc) };
                i += 4;
            }
            let mut tmp = [0f32; 4];
            unsafe { st4_f32(tmp.as_mut_ptr(), acc) };
            for k in 0..4 {
                acc_d += tmp[k] as f64;
            }
            acc = unsafe { zero4_f32() };
        }
        while i + 4 <= n {
            let va = unsafe { ld4_f32(a.as_ptr().add(i)) };
            let vb = unsafe { ld4_f32(b.as_ptr().add(i)) };
            acc = unsafe { lsx_vfmadd_s(va, vb, acc) };
            i += 4;
        }
        let mut tmp = [0f32; 4];
        unsafe { st4_f32(tmp.as_mut_ptr(), acc) };
        for k in 0..4 {
            acc_d += tmp[k] as f64;
        }
        for j in i..n {
            acc_d += (a[j] * b[j]) as f64;
        }
        return acc_d as f32;
    }
    let mut acc = unsafe { zero_f32() };
    let mut acc_d = 0f64;
    let mut i = 0;
    // 每 64 元素（8 个向量）把 f32 向量累加落盘到 double，抑制累加误差
    while i + 64 <= n {
        for _ in 0..8 {
            let va = unsafe { ld_f32(a.as_ptr().add(i)) };
            let vb = unsafe { ld_f32(b.as_ptr().add(i)) };
            acc = unsafe { lasx_xvfmadd_s(va, vb, acc) };
            i += 8;
        }
        let mut tmp = [0f32; 8];
        unsafe { st_f32(tmp.as_mut_ptr(), acc) };
        for k in 0..8 {
            acc_d += tmp[k] as f64;
        }
        acc = unsafe { zero_f32() };
    }
    while i + 8 <= n {
        let va = unsafe { ld_f32(a.as_ptr().add(i)) };
        let vb = unsafe { ld_f32(b.as_ptr().add(i)) };
        acc = unsafe { lasx_xvfmadd_s(va, vb, acc) };
        i += 8;
    }
    let mut tmp = [0f32; 8];
    unsafe { st_f32(tmp.as_mut_ptr(), acc) };
    for k in 0..8 {
        acc_d += tmp[k] as f64;
    }
    for j in i..n {
        acc_d += (a[j] * b[j]) as f64;
    }
    acc_d as f32
}

/// 矩阵乘 C[m×n] = A[m×k] × B[k×n]（B 转置，8×f32 FMA）
#[unsafe(no_mangle)]
pub extern "C" fn lasx_matmul(m: i32, k: i32, n: i32, a: *const f32, b: *const f32, c: *mut f32) {
    let (m, k, n) = (m as usize, k as usize, n as usize);
    let a = unsafe { std::slice::from_raw_parts(a, m * k) };
    let b = unsafe { std::slice::from_raw_parts(b, k * n) };
    let c = unsafe { std::slice::from_raw_parts_mut(c, m * n) };
    let mut b_t = vec![0f32; n * k];
    for j in 0..n {
        for p in 0..k {
            b_t[j * k + p] = b[p * n + j];
        }
    }
    for i in 0..m {
        for j in 0..n {
            let mut acc = unsafe { zero_f32() };
            let a_row = &a[i * k..(i + 1) * k];
            let b_row = &b_t[j * k..(j + 1) * k];
            let mut p = 0;
            while p + 8 <= k {
                let va = unsafe { ld_f32(a_row.as_ptr().add(p)) };
                let vb = unsafe { ld_f32(b_row.as_ptr().add(p)) };
                acc = unsafe { lasx_xvfmadd_s(va, vb, acc) };
                p += 8;
            }
            let mut tmp = [0f32; 8];
            unsafe { st_f32(tmp.as_mut_ptr(), acc) };
            let mut s = 0f32;
            for q in 0..8 {
                s += tmp[q];
            }
            for q in p..k {
                s += a_row[q] * b_row[q];
            }
            c[i * n + j] = s;
        }
    }
}

/// axpy: y += a*x
#[unsafe(no_mangle)]
pub extern "C" fn lasx_axpy(alpha: f32, x: *const f32, y: *mut f32, n: i32) {
    let n = n as usize;
    let x = unsafe { std::slice::from_raw_parts(x, n) };
    let y = unsafe { std::slice::from_raw_parts_mut(y, n) };
    let va = splat_f32(alpha);
    let mut i = 0;
    while i + 8 <= n {
        let vx = unsafe { ld_f32(x.as_ptr().add(i)) };
        let vy = unsafe { ld_f32(y.as_ptr().add(i)) };
        let vf = unsafe { lasx_xvfmadd_s(vx, va, vy) };
        unsafe { st_f32(y.as_mut_ptr().add(i), vf) };
        i += 8;
    }
    for j in i..n {
        y[j] += alpha * x[j];
    }
}

/// sum: 向量归约
#[unsafe(no_mangle)]
pub extern "C" fn lasx_sum(x: *const f32, n: i32) -> f32 {
    let n = n as usize;
    let x = unsafe { std::slice::from_raw_parts(x, n) };
    let mut acc = unsafe { zero_f32() };
    let mut acc_d = 0f64;
    let mut i = 0;
    while i + 64 <= n {
        for _ in 0..8 {
            acc = unsafe { lasx_xvfadd_s(acc, ld_f32(x.as_ptr().add(i))) };
            i += 8;
        }
        let mut tmp = [0f32; 8];
        unsafe { st_f32(tmp.as_mut_ptr(), acc) };
        for k in 0..8 {
            acc_d += tmp[k] as f64;
        }
        acc = unsafe { zero_f32() };
    }
    while i + 8 <= n {
        acc = unsafe { lasx_xvfadd_s(acc, ld_f32(x.as_ptr().add(i))) };
        i += 8;
    }
    let mut tmp = [0f32; 8];
    unsafe { st_f32(tmp.as_mut_ptr(), acc) };
    for k in 0..8 {
        acc_d += tmp[k] as f64;
    }
    for j in i..n {
        acc_d += x[j] as f64;
    }
    acc_d as f32
}

/// F64 点积（4×f64 FMA）
#[unsafe(no_mangle)]
pub extern "C" fn lasx_dot_f64(a: *const f64, b: *const f64, n: i32) -> f64 {
    let n = n as usize;
    let a = unsafe { std::slice::from_raw_parts(a, n) };
    let b = unsafe { std::slice::from_raw_parts(b, n) };
    let mut acc = unsafe { zero_f64() };
    let mut i = 0;
    while i + 4 <= n {
        let va = unsafe { ld_f64(a.as_ptr().add(i)) };
        let vb = unsafe { ld_f64(b.as_ptr().add(i)) };
        acc = unsafe { lasx_xvfmadd_d(va, vb, acc) };
        i += 4;
    }
    let mut tmp = [0f64; 4];
    unsafe { st_f64(tmp.as_mut_ptr(), acc) };
    let mut s = 0f64;
    for k in 0..4 {
        s += tmp[k];
    }
    for j in i..n {
        s += a[j] * b[j];
    }
    s
}

/// 内存分配（Dart FFI 侧）
#[unsafe(no_mangle)]
pub extern "C" fn lasx_alloc(n: i32) -> *mut f32 {
    let mut v = Vec::with_capacity(n as usize);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// int8 量化点积（32×int8/向量，vmulwev/vmulwod 模式）
#[unsafe(no_mangle)]
pub extern "C" fn lasx_dot_i8(a: *const i8, b: *const i8, n: i32) -> i32 {
    let n = n as usize;
    let a = unsafe { std::slice::from_raw_parts(a, n) };
    let b = unsafe { std::slice::from_raw_parts(b, n) };
    let mut acc: m256i = unsafe { lasx_xvldi(0) };
    let mut s_acc: i64 = 0;
    let mut i = 0;
    while i + 32 <= n {
        let va = unsafe { lasx_xvld(a.as_ptr().add(i) as *const i8, 0) };
        let vb = unsafe { lasx_xvld(b.as_ptr().add(i) as *const i8, 0) };
        let lo16 = unsafe { lasx_xvmulwev_h_b(va, vb) };
        let hi16 = unsafe { lasx_xvmulwod_h_b(va, vb) };
        let sum16 = unsafe { lasx_xvadd_h(lo16, hi16) };
        // 每 32 字节落内存，用 int64 累加避免溢出
        let mut tmp = [0i16; 16];
        unsafe { lasx_xvst(sum16, tmp.as_mut_ptr() as *mut i8, 0) };
        for k in 0..16 {
            s_acc += tmp[k] as i64;
        }
        i += 32;
    }
    for j in i..n {
        s_acc += (a[j] as i64) * (b[j] as i64);
    }
    s_acc as i32
}

/// Q4 量化点积：每字节 2 个无符号 nibble，scale 每 32 字节一组
/// dot = Σ_g scale_a[g] * scale_b[g] * Σ_{64 nibble} (qa*qb)
#[unsafe(no_mangle)]
pub extern "C" fn lasx_dot_q4(
    qa: *const u8, sa: *const f32, qb: *const u8, sb: *const f32, n_bytes: i32,
) -> f64 {
    let n = n_bytes as usize;
    let qa = unsafe { std::slice::from_raw_parts(qa, n) };
    let qb = unsafe { std::slice::from_raw_parts(qb, n) };
    let n_groups = n.div_ceil(32);
    let sa = unsafe { std::slice::from_raw_parts(sa, n_groups) };
    let sb = unsafe { std::slice::from_raw_parts(sb, n_groups) };
    let mut acc = 0f64;
    let mut b = 0;
    while b + 32 <= n {
        let va = unsafe { lasx_xvld(qa.as_ptr().add(b) as *const i8, 0) };
        let vb = unsafe { lasx_xvld(qb.as_ptr().add(b) as *const i8, 0) };
        let alo = unsafe { lasx_xvandi_b(va, 0x0f) };
        let ahi = unsafe { lasx_xvsrli_b(va, 4) };
        let blo = unsafe { lasx_xvandi_b(vb, 0x0f) };
        let bhi = unsafe { lasx_xvsrli_b(vb, 4) };
        let s0 = unsafe {
            lasx_xvadd_h(
                lasx_xvmulwev_h_b(alo, blo),
                lasx_xvmulwod_h_b(alo, blo),
            )
        };
        let s1 = unsafe {
            lasx_xvadd_h(
                lasx_xvmulwev_h_b(ahi, bhi),
                lasx_xvmulwod_h_b(ahi, bhi),
            )
        };
        let sall = unsafe { lasx_xvadd_h(s0, s1) };
        let mut tmp = [0i16; 16];
        unsafe { lasx_xvst(sall, tmp.as_mut_ptr() as *mut i8, 0) };
        let mut dot: i64 = 0;
        for k in 0..16 {
            dot += tmp[k] as i64;
        }
        let sc = sa[b / 32] * sb[b / 32];
        acc += (sc as f64) * (dot as f64);
        b += 32;
    }
    for j in b..n {
        let la = (qa[j] & 0x0f) as i64;
        let ha = ((qa[j] >> 4) & 0x0f) as i64;
        let lb = (qb[j] & 0x0f) as i64;
        let hb = ((qb[j] >> 4) & 0x0f) as i64;
        let sc = sa[j / 32] * sb[j / 32];
        acc += (sc as f64) * ((la * lb + ha * hb) as f64);
    }
    acc
}

/// F64 矩阵乘 C[m×n] = A[m×k] × B[k×n]（B 转置，4×f64 FMA）
#[unsafe(no_mangle)]
pub extern "C" fn lasx_matmul_f64(
    m: i32, k: i32, n: i32, a: *const f64, b: *const f64, c: *mut f64,
) {
    let (m, k, n) = (m as usize, k as usize, n as usize);
    let a = unsafe { std::slice::from_raw_parts(a, m * k) };
    let b = unsafe { std::slice::from_raw_parts(b, k * n) };
    let c = unsafe { std::slice::from_raw_parts_mut(c, m * n) };
    let mut b_t = vec![0f64; n * k];
    for j in 0..n {
        for p in 0..k {
            b_t[j * k + p] = b[p * n + j];
        }
    }
    for i in 0..m {
        for j in 0..n {
            let mut acc = unsafe { zero_f64() };
            let a_row = &a[i * k..(i + 1) * k];
            let b_row = &b_t[j * k..(j + 1) * k];
            let mut p = 0;
            while p + 4 <= k {
                let va = unsafe { ld_f64(a_row.as_ptr().add(p)) };
                let vb = unsafe { ld_f64(b_row.as_ptr().add(p)) };
                acc = unsafe { lasx_xvfmadd_d(va, vb, acc) };
                p += 4;
            }
            let mut tmp = [0f64; 4];
            unsafe { st_f64(tmp.as_mut_ptr(), acc) };
            let mut s = 0f64;
            for q in 0..4 {
                s += tmp[q];
            }
            for q in p..k {
                s += a_row[q] * b_row[q];
            }
            c[i * n + j] = s;
        }
    }
}

/// 弹道步进内核：8 发弹并行，单循环向量化
/// 输入 SOA 数组（每发弹 6 状态 + 阻力系数 k），推进一步欧拉
/// 内层：|v| = sqrt(vx²+vy²+vz²)；vx -= k*|v|*vx*dt；x += vx*dt（全部向量）
#[unsafe(no_mangle)]
pub extern "C" fn lasx_ballistic_step(
    x: *mut f32, y: *mut f32, z: *mut f32,
    vx: *mut f32, vy: *mut f32, vz: *mut f32,
    k: *const f32, n: i32, dt: f32, g: f32,
) {
    let n = n as usize;
    let x = unsafe { std::slice::from_raw_parts_mut(x, n) };
    let y = unsafe { std::slice::from_raw_parts_mut(y, n) };
    let z = unsafe { std::slice::from_raw_parts_mut(z, n) };
    let vx = unsafe { std::slice::from_raw_parts_mut(vx, n) };
    let vy = unsafe { std::slice::from_raw_parts_mut(vy, n) };
    let vz = unsafe { std::slice::from_raw_parts_mut(vz, n) };
    let k = unsafe { std::slice::from_raw_parts(k, n) };

    if has_lasx() {
        let vdt = splat_f32(dt);
        let vg = splat_f32(g);
        let mut i = 0;
        while i + 8 <= n {
            let vvx: m256 = unsafe { std::mem::transmute(lasx_xvld(vx.as_ptr().add(i) as *const i8, 0)) };
            let vvy: m256 = unsafe { std::mem::transmute(lasx_xvld(vy.as_ptr().add(i) as *const i8, 0)) };
            let vvz: m256 = unsafe { std::mem::transmute(lasx_xvld(vz.as_ptr().add(i) as *const i8, 0)) };
            let vk: m256 = unsafe { std::mem::transmute(lasx_xvld(k.as_ptr().add(i) as *const i8, 0)) };
            // |v| = sqrt(vx²+vy²+vz²)
            let vsq = unsafe {
                lasx_xvfadd_s(
                    lasx_xvfmul_s(vvx, vvx),
                    lasx_xvfadd_s(lasx_xvfmul_s(vvy, vvy), lasx_xvfmul_s(vvz, vvz)),
                )
            };
            let vmag = unsafe { lasx_xvfsqrt_s(vsq) };
            // drag = k * |v|
            let vdrag = unsafe { lasx_xvfmul_s(vk, vmag) };
            // dvx = -drag*vx; dvy = -drag*vy - g; dvz = -drag*vz
            let vdvx = unsafe { lasx_xvfmul_s(vdrag, vvx) };
            let vdvy = unsafe { lasx_xvfmul_s(vdrag, vvy) };
            let vdvz = unsafe { lasx_xvfmul_s(vdrag, vvz) };
            // 更新：vx += dvx*dt（符号：dvx 为负，直接用 dvx*dt）
            // 注意：dvx = -drag*vx，更新 vx += dvx*dt（欧拉）
            let vnx = unsafe { lasx_xvfmadd_s(vdvx, vdt, vvx) };
            let vny = unsafe { lasx_xvfmadd_s(vdvy, vdt, vvy) };
            let vnz = unsafe { lasx_xvfmadd_s(vdvz, vdt, vvz) };
            // x += vx*dt（用旧 vx，欧拉）
            let vnx_old = vvx; let vny_old = vvy; let vnz_old = vvz;
            let vnx_ = unsafe { lasx_xvfmadd_s(vnx_old, vdt, std::mem::transmute(lasx_xvld(x.as_ptr().add(i) as *const i8, 0))) };
            let vny_ = unsafe { lasx_xvfmadd_s(vny_old, vdt, std::mem::transmute(lasx_xvld(y.as_ptr().add(i) as *const i8, 0))) };
            let vnz_ = unsafe { lasx_xvfmadd_s(vnz_old, vdt, std::mem::transmute(lasx_xvld(z.as_ptr().add(i) as *const i8, 0))) };
            // 存储
            unsafe { lasx_xvst(std::mem::transmute(vnx), vx.as_mut_ptr().add(i) as *mut i8, 0) };
            unsafe { lasx_xvst(std::mem::transmute(vny), vy.as_mut_ptr().add(i) as *mut i8, 0) };
            unsafe { lasx_xvst(std::mem::transmute(vnz), vz.as_mut_ptr().add(i) as *mut i8, 0) };
            unsafe { lasx_xvst(std::mem::transmute(vnx_), x.as_mut_ptr().add(i) as *mut i8, 0) };
            unsafe { lasx_xvst(std::mem::transmute(vny_), y.as_mut_ptr().add(i) as *mut i8, 0) };
            unsafe { lasx_xvst(std::mem::transmute(vnz_), z.as_mut_ptr().add(i) as *mut i8, 0) };
            i += 8;
        }
        for j in i..n {
            euler_step(&mut x[j], &mut y[j], &mut z[j], &mut vx[j], &mut vy[j], &mut vz[j], k[j], dt, g);
        }
    } else {
        for j in 0..n {
            euler_step(&mut x[j], &mut y[j], &mut z[j], &mut vx[j], &mut vy[j], &mut vz[j], k[j], dt, g);
        }
    }
}

fn euler_step(x: &mut f32, y: &mut f32, z: &mut f32, vx: &mut f32, vy: &mut f32, vz: &mut f32, k: f32, dt: f32, g: f32) {
    let v = (*vx * *vx + *vy * *vy + *vz * *vz).sqrt();
    let drag = k * v;
    *vx -= drag * *vx * dt;
    *vy -= (drag * *vy + g) * dt;
    *vz -= drag * *vz * dt;
    *x += *vx * dt;
    *y += *vy * dt;
    *z += *vz * dt;
}
