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

// 线程级强制 LSX 降级（验证/测试钩子）：仅影响调用线程的向量内核，避免进程级
// 环境变量污染并发测试（旧机制 `LOONGSCI_FORCE_LSX` 在进程级 OnceLock 缓存，
// 并发测试时会让其他依赖 LASX 黄金值的测试随机走 LSX 路径而失败）。
thread_local! {
    static FORCE_LSX: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// 强制当前线程的向量内核走 LSX 降级路径（模拟 LSX-only CPU——如 3A5000/3A6000；
/// 本机 3B6000 含 LASX，用钩子让代码真实执行 LSX intrinsic 分支验证数值/性能，
/// 诚实标注非无-LASX 真机）。仅本线程生效；测试结束时置回 false。
pub fn lasx_force_lsx_thread(force: bool) {
    FORCE_LSX.with(|c| c.set(force));
}

/// 硬件能力缓存（cpucfg 只读一次；LSX 强制为线程级，不缓存）
static HW_HAS_LASX: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
fn has_lasx() -> bool {
    let hw = *HW_HAS_LASX.get_or_init(|| unsafe {
        let mut cfg2: u32;
        std::arch::asm!("cpucfg {}, {}", out(reg) cfg2, in(reg) 2u32);
        (cfg2 & (1 << 7)) != 0
    });
    hw && !FORCE_LSX.with(|c| c.get())
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
    qa: *const u8,
    sa: *const f32,
    qb: *const u8,
    sb: *const f32,
    n_bytes: i32,
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
        let s0 = unsafe { lasx_xvadd_h(lasx_xvmulwev_h_b(alo, blo), lasx_xvmulwod_h_b(alo, blo)) };
        let s1 = unsafe { lasx_xvadd_h(lasx_xvmulwev_h_b(ahi, bhi), lasx_xvmulwod_h_b(ahi, bhi)) };
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
    m: i32,
    k: i32,
    n: i32,
    a: *const f64,
    b: *const f64,
    c: *mut f64,
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
    let x = unsafe { std::slice::from_raw_parts_mut(x, n) };
    let y = unsafe { std::slice::from_raw_parts_mut(y, n) };
    let z = unsafe { std::slice::from_raw_parts_mut(z, n) };
    let vx = unsafe { std::slice::from_raw_parts_mut(vx, n) };
    let vy = unsafe { std::slice::from_raw_parts_mut(vy, n) };
    let vz = unsafe { std::slice::from_raw_parts_mut(vz, n) };
    let k = unsafe { std::slice::from_raw_parts(k, n) };

    if has_lasx() {
        let vdt = splat_f32(dt);
        let mut i = 0;
        while i + 8 <= n {
            let vvx: m256 =
                unsafe { std::mem::transmute(lasx_xvld(vx.as_ptr().add(i) as *const i8, 0)) };
            let vvy: m256 =
                unsafe { std::mem::transmute(lasx_xvld(vy.as_ptr().add(i) as *const i8, 0)) };
            let vvz: m256 =
                unsafe { std::mem::transmute(lasx_xvld(vz.as_ptr().add(i) as *const i8, 0)) };
            let vk: m256 =
                unsafe { std::mem::transmute(lasx_xvld(k.as_ptr().add(i) as *const i8, 0)) };
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
            let vnx_old = vvx;
            let vny_old = vvy;
            let vnz_old = vvz;
            let vnx_ = unsafe {
                lasx_xvfmadd_s(
                    vnx_old,
                    vdt,
                    std::mem::transmute(lasx_xvld(x.as_ptr().add(i) as *const i8, 0)),
                )
            };
            let vny_ = unsafe {
                lasx_xvfmadd_s(
                    vny_old,
                    vdt,
                    std::mem::transmute(lasx_xvld(y.as_ptr().add(i) as *const i8, 0)),
                )
            };
            let vnz_ = unsafe {
                lasx_xvfmadd_s(
                    vnz_old,
                    vdt,
                    std::mem::transmute(lasx_xvld(z.as_ptr().add(i) as *const i8, 0)),
                )
            };
            // 存储
            unsafe {
                lasx_xvst(
                    std::mem::transmute(vnx),
                    vx.as_mut_ptr().add(i) as *mut i8,
                    0,
                )
            };
            unsafe {
                lasx_xvst(
                    std::mem::transmute(vny),
                    vy.as_mut_ptr().add(i) as *mut i8,
                    0,
                )
            };
            unsafe {
                lasx_xvst(
                    std::mem::transmute(vnz),
                    vz.as_mut_ptr().add(i) as *mut i8,
                    0,
                )
            };
            unsafe {
                lasx_xvst(
                    std::mem::transmute(vnx_),
                    x.as_mut_ptr().add(i) as *mut i8,
                    0,
                )
            };
            unsafe {
                lasx_xvst(
                    std::mem::transmute(vny_),
                    y.as_mut_ptr().add(i) as *mut i8,
                    0,
                )
            };
            unsafe {
                lasx_xvst(
                    std::mem::transmute(vnz_),
                    z.as_mut_ptr().add(i) as *mut i8,
                    0,
                )
            };
            i += 8;
        }
        for j in i..n {
            euler_step(
                &mut x[j], &mut y[j], &mut z[j], &mut vx[j], &mut vy[j], &mut vz[j], k[j], dt, g,
            );
        }
    } else {
        for j in 0..n {
            euler_step(
                &mut x[j], &mut y[j], &mut z[j], &mut vx[j], &mut vy[j], &mut vz[j], k[j], dt, g,
            );
        }
    }
}

fn euler_step(
    x: &mut f32,
    y: &mut f32,
    z: &mut f32,
    vx: &mut f32,
    vy: &mut f32,
    vz: &mut f32,
    k: f32,
    dt: f32,
    g: f32,
) {
    let v = (*vx * *vx + *vy * *vy + *vz * *vz).sqrt();
    let drag = k * v;
    *vx -= drag * *vx * dt;
    *vy -= (drag * *vy + g) * dt;
    *vz -= drag * *vz * dt;
    *x += *vx * dt;
    *y += *vy * dt;
    *z += *vz * dt;
}

/// 批量 2D 距离：n 个点 (x[i],y[i]) 到 (px,py) 的距离，8 点并行
/// LASX：dx²+dy² 用 FMA，xvfsqrt 开方
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
    let xs = unsafe { std::slice::from_raw_parts(xs, n) };
    let ys = unsafe { std::slice::from_raw_parts(ys, n) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, n) };
    if has_lasx() {
        let vpx = splat_f32(px);
        let vpy = splat_f32(py);
        let mut i = 0;
        while i + 8 <= n {
            let vx: m256 =
                unsafe { std::mem::transmute(lasx_xvld(xs.as_ptr().add(i) as *const i8, 0)) };
            let vy: m256 =
                unsafe { std::mem::transmute(lasx_xvld(ys.as_ptr().add(i) as *const i8, 0)) };
            let dx = unsafe { lasx_xvfsub_s(vx, vpx) };
            let dy = unsafe { lasx_xvfsub_s(vy, vpy) };
            // dist = sqrt(dx*dx + dy*dy)
            let sq = unsafe { lasx_xvfadd_s(lasx_xvfmul_s(dx, dx), lasx_xvfmul_s(dy, dy)) };
            let d = unsafe { lasx_xvfsqrt_s(sq) };
            unsafe {
                lasx_xvst(
                    std::mem::transmute(d),
                    out.as_mut_ptr().add(i) as *mut i8,
                    0,
                )
            };
            i += 8;
        }
        for j in i..n {
            let dx = xs[j] - px;
            let dy = ys[j] - py;
            out[j] = (dx * dx + dy * dy).sqrt();
        }
    } else {
        for j in 0..n {
            let dx = xs[j] - px;
            let dy = ys[j] - py;
            out[j] = (dx * dx + dy * dy).sqrt();
        }
    }
}

// ==================== 批量轨道向量内核（f64 SOA，LASX 4 样本/向量） ====================
// 供 loong-sci `space::propagate_orbits_batch`（多星同时传播）使用。
// SOA 布局：每个分量一个连续 f64 数组；LASX 256 位一次处理 4 个样本的同一分量，
// LSX 128 位一次 2 个（降级），无向量指令 CPU 走标量尾循环——与引擎 axpy8 模式一致。
// 支持原地（输出数组可与输入 a 同一内存，如 r = r + s·lsum）。

#[inline]
fn splat_f64(x: f64) -> F64x4 {
    let bits = x.to_bits() as i64;
    unsafe { std::mem::transmute(lasx_xvreplgr2vr_d(bits)) }
}
#[inline]
fn splat2_f64(x: f64) -> F64x2 {
    let bits = x.to_bits() as i64;
    unsafe { std::mem::transmute(lsx_vreplgr2vr_d(bits)) }
}

/// 批量 3 分量模长：out[i] = √(x[i]²+y[i]²+z[i]²)
/// SOA（xs/ys/zs 各为 n 长连续数组）；LASX 4 样本/向量，LSX 2 样本，标量兜底。
#[unsafe(no_mangle)]
pub extern "C" fn lasx_norm3_batch(
    xs: *const f64,
    ys: *const f64,
    zs: *const f64,
    out: *mut f64,
    n: i32,
) {
    let n = n as usize;
    let xs = unsafe { std::slice::from_raw_parts(xs, n) };
    let ys = unsafe { std::slice::from_raw_parts(ys, n) };
    let zs = unsafe { std::slice::from_raw_parts(zs, n) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, n) };
    if has_lasx() {
        let mut i = 0;
        while i + 4 <= n {
            let vx = unsafe { ld_f64(xs.as_ptr().add(i)) };
            let vy = unsafe { ld_f64(ys.as_ptr().add(i)) };
            let vz = unsafe { ld_f64(zs.as_ptr().add(i)) };
            let sq = unsafe {
                lasx_xvfadd_d(
                    lasx_xvfmul_d(vx, vx),
                    lasx_xvfadd_d(lasx_xvfmul_d(vy, vy), lasx_xvfmul_d(vz, vz)),
                )
            };
            let mag = unsafe { lasx_xvfsqrt_d(sq) };
            unsafe { st_f64(out.as_mut_ptr().add(i), mag) };
            i += 4;
        }
        for j in i..n {
            // 与向量路径同结合（x² + (y²+z²)），保证任一分块逐位一致
            out[j] = (xs[j] * xs[j] + (ys[j] * ys[j] + zs[j] * zs[j])).sqrt();
        }
    } else {
        let mut i = 0;
        while i + 2 <= n {
            let vx = unsafe { ld2_f64(xs.as_ptr().add(i)) };
            let vy = unsafe { ld2_f64(ys.as_ptr().add(i)) };
            let vz = unsafe { ld2_f64(zs.as_ptr().add(i)) };
            let sq = unsafe {
                lsx_vfadd_d(
                    lsx_vfmul_d(vx, vx),
                    lsx_vfadd_d(lsx_vfmul_d(vy, vy), lsx_vfmul_d(vz, vz)),
                )
            };
            let mag = unsafe { lsx_vfsqrt_d(sq) };
            unsafe { st2_f64(out.as_mut_ptr().add(i), mag) };
            i += 2;
        }
        for j in i..n {
            out[j] = (xs[j] * xs[j] + (ys[j] * ys[j] + zs[j] * zs[j])).sqrt();
        }
    }
}

/// 批量 3 分量“缩放加”：o[i] = a[i] + s·b[i]（x/y/z 各分量独立，SOA）
/// 覆盖加（s=1）与缩放（a=0）两个退化情形；LASX 4 样本/向量。
#[unsafe(no_mangle)]
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
    let ax = unsafe { std::slice::from_raw_parts(ax, n) };
    let ay = unsafe { std::slice::from_raw_parts(ay, n) };
    let az = unsafe { std::slice::from_raw_parts(az, n) };
    let bx = unsafe { std::slice::from_raw_parts(bx, n) };
    let by = unsafe { std::slice::from_raw_parts(by, n) };
    let bz = unsafe { std::slice::from_raw_parts(bz, n) };
    let ox = unsafe { std::slice::from_raw_parts_mut(ox, n) };
    let oy = unsafe { std::slice::from_raw_parts_mut(oy, n) };
    let oz = unsafe { std::slice::from_raw_parts_mut(oz, n) };
    if has_lasx() {
        let vs = splat_f64(s);
        let mut i = 0;
        while i + 4 <= n {
            let vax = unsafe { ld_f64(ax.as_ptr().add(i)) };
            let vay = unsafe { ld_f64(ay.as_ptr().add(i)) };
            let vaz = unsafe { ld_f64(az.as_ptr().add(i)) };
            let vbx = unsafe { ld_f64(bx.as_ptr().add(i)) };
            let vby = unsafe { ld_f64(by.as_ptr().add(i)) };
            let vbz = unsafe { ld_f64(bz.as_ptr().add(i)) };
            // o = a + s·b（单次 FMA，误差 ≤ 标量 mul+add 两舍入）
            unsafe {
                st_f64(ox.as_mut_ptr().add(i), lasx_xvfmadd_d(vs, vbx, vax));
                st_f64(oy.as_mut_ptr().add(i), lasx_xvfmadd_d(vs, vby, vay));
                st_f64(oz.as_mut_ptr().add(i), lasx_xvfmadd_d(vs, vbz, vaz));
            }
            i += 4;
        }
        for j in i..n {
            // 与向量路径同式：单次 FMA（s·b + a），保证任一分块逐位一致
            ox[j] = f64::mul_add(s, bx[j], ax[j]);
            oy[j] = f64::mul_add(s, by[j], ay[j]);
            oz[j] = f64::mul_add(s, bz[j], az[j]);
        }
    } else {
        let vs = splat2_f64(s);
        let mut i = 0;
        while i + 2 <= n {
            let vax = unsafe { ld2_f64(ax.as_ptr().add(i)) };
            let vay = unsafe { ld2_f64(ay.as_ptr().add(i)) };
            let vaz = unsafe { ld2_f64(az.as_ptr().add(i)) };
            let vbx = unsafe { ld2_f64(bx.as_ptr().add(i)) };
            let vby = unsafe { ld2_f64(by.as_ptr().add(i)) };
            let vbz = unsafe { ld2_f64(bz.as_ptr().add(i)) };
            unsafe {
                st2_f64(ox.as_mut_ptr().add(i), lsx_vfmadd_d(vs, vbx, vax));
                st2_f64(oy.as_mut_ptr().add(i), lsx_vfmadd_d(vs, vby, vay));
                st2_f64(oz.as_mut_ptr().add(i), lsx_vfmadd_d(vs, vbz, vaz));
            }
            i += 2;
        }
        for j in i..n {
            ox[j] = f64::mul_add(s, bx[j], ax[j]);
            oy[j] = f64::mul_add(s, by[j], ay[j]);
            oz[j] = f64::mul_add(s, bz[j], az[j]);
        }
    }
}

/// 批量“中心点质量 + J2”摄动加速度（SOA）：
///   a_i = −μ·r_i/|r_i|³  +  J2 项（与 loong-sci `EarthModel::j2_acceleration` 同公式）
///     k = −1.5·J2·μ·Re²/|r_i|⁵；zr2 = (z_i/|r_i|)²
///     a_x += −k·x·(5·zr2−1)，a_y += −k·y·(5·zr2−1)，a_z += −k·z·(5·zr2−3)
/// 全部为逐样本（lane）独立运算 → 天然 4 路并行。
#[unsafe(no_mangle)]
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
    let rx = unsafe { std::slice::from_raw_parts(rx, n) };
    let ry = unsafe { std::slice::from_raw_parts(ry, n) };
    let rz = unsafe { std::slice::from_raw_parts(rz, n) };
    let ax = unsafe { std::slice::from_raw_parts_mut(ax, n) };
    let ay = unsafe { std::slice::from_raw_parts_mut(ay, n) };
    let az = unsafe { std::slice::from_raw_parts_mut(az, n) };
    // k 取正值（−k 即此值）：1.5·J2·μ·Re²
    let j2k = 1.5 * j2 * mu * re * re;
    if has_lasx() {
        let vmu = splat_f64(-mu); // 中心项 = −μ·r/|r|³
        let vj2k = splat_f64(j2k);
        let v5 = splat_f64(5.0);
        let v1 = splat_f64(1.0);
        let v3 = splat_f64(3.0);
        let mut i = 0;
        while i + 4 <= n {
            let vx = unsafe { ld_f64(rx.as_ptr().add(i)) };
            let vy = unsafe { ld_f64(ry.as_ptr().add(i)) };
            let vz = unsafe { ld_f64(rz.as_ptr().add(i)) };
            // |r|²、|r|、|r|³、|r|⁵ = |r|³·|r|²
            let vrm2 = unsafe {
                lasx_xvfadd_d(
                    lasx_xvfmul_d(vx, vx),
                    lasx_xvfadd_d(lasx_xvfmul_d(vy, vy), lasx_xvfmul_d(vz, vz)),
                )
            };
            let vrm = unsafe { lasx_xvfsqrt_d(vrm2) };
            let vrm3 = unsafe { lasx_xvfmul_d(vrm, vrm2) };
            let vrm5 = unsafe { lasx_xvfmul_d(vrm3, vrm2) };
            // 中心项 −μ·r/|r|³
            let vcen = unsafe { lasx_xvfdiv_d(vmu, vrm3) };
            // J2 项系数（正）：1.5·J2·μ·Re²/rm⁵
            let vk = unsafe { lasx_xvfdiv_d(vj2k, vrm5) };
            // zr2 = (z/|r|)²
            let vzr2 = unsafe { lasx_xvfdiv_d(lasx_xvfmul_d(vz, vz), vrm2) };
            let m1 = unsafe { lasx_xvfsub_d(lasx_xvfmul_d(v5, vzr2), v1) };
            let m3 = unsafe { lasx_xvfsub_d(lasx_xvfmul_d(v5, vzr2), v3) };
            // a_x = −μ·x/rm³ + k·x·(5·zr2−1)（k 正 = −k_j2）
            unsafe {
                st_f64(
                    ax.as_mut_ptr().add(i),
                    lasx_xvfmadd_d(lasx_xvfmul_d(vk, vx), m1, lasx_xvfmul_d(vcen, vx)),
                );
                st_f64(
                    ay.as_mut_ptr().add(i),
                    lasx_xvfmadd_d(lasx_xvfmul_d(vk, vy), m1, lasx_xvfmul_d(vcen, vy)),
                );
                st_f64(
                    az.as_mut_ptr().add(i),
                    lasx_xvfmadd_d(lasx_xvfmul_d(vk, vz), m3, lasx_xvfmul_d(vcen, vz)),
                );
            }
            i += 4;
        }
        for j in i..n {
            // 与向量路径逐位同式（结合律/除式/末项 FMA 完全一致），保证任一分块逐位一致
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
    } else {
        let vs = splat2_f64(j2k);
        let vmu = splat2_f64(-mu); // 中心项 = −μ·r/|r|³
        let v5 = splat2_f64(5.0);
        let v1 = splat2_f64(1.0);
        let v3 = splat2_f64(3.0);
        let mut i = 0;
        while i + 2 <= n {
            let vx = unsafe { ld2_f64(rx.as_ptr().add(i)) };
            let vy = unsafe { ld2_f64(ry.as_ptr().add(i)) };
            let vz = unsafe { ld2_f64(rz.as_ptr().add(i)) };
            let vrm2 = unsafe {
                lsx_vfadd_d(
                    lsx_vfmul_d(vx, vx),
                    lsx_vfadd_d(lsx_vfmul_d(vy, vy), lsx_vfmul_d(vz, vz)),
                )
            };
            let vrm = unsafe { lsx_vfsqrt_d(vrm2) };
            let vrm3 = unsafe { lsx_vfmul_d(vrm, vrm2) };
            let vrm5 = unsafe { lsx_vfmul_d(vrm3, vrm2) };
            let vcen = unsafe { lsx_vfdiv_d(vmu, vrm3) };
            let vk = unsafe { lsx_vfdiv_d(vs, vrm5) };
            let vzr2 = unsafe { lsx_vfdiv_d(lsx_vfmul_d(vz, vz), vrm2) };
            let m1 = unsafe { lsx_vfsub_d(lsx_vfmul_d(v5, vzr2), v1) };
            let m3 = unsafe { lsx_vfsub_d(lsx_vfmul_d(v5, vzr2), v3) };
            unsafe {
                st2_f64(
                    ax.as_mut_ptr().add(i),
                    lsx_vfmadd_d(lsx_vfmul_d(vk, vx), m1, lsx_vfmul_d(vcen, vx)),
                );
                st2_f64(
                    ay.as_mut_ptr().add(i),
                    lsx_vfmadd_d(lsx_vfmul_d(vk, vy), m1, lsx_vfmul_d(vcen, vy)),
                );
                st2_f64(
                    az.as_mut_ptr().add(i),
                    lsx_vfmadd_d(lsx_vfmul_d(vk, vz), m3, lsx_vfmul_d(vcen, vz)),
                );
            }
            i += 2;
        }
        for j in i..n {
            // 与 LSX 向量路径逐位同式（同结合/除式/末项 FMA）
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
}

// ==================== 批量内核数值测试（标量对照，<1e-9） ====================
#[cfg(test)]
mod batch_tests {
    use super::*;

    fn scalar_norm3(x: &[f64], y: &[f64], z: &[f64]) -> Vec<f64> {
        (0..x.len())
            .map(|i| (x[i] * x[i] + y[i] * y[i] + z[i] * z[i]).sqrt())
            .collect()
    }
    fn scalar_add_scaled(a: &[f64], b: &[f64], s: f64) -> Vec<f64> {
        a.iter().zip(b).map(|(&x, &y)| x + s * y).collect()
    }
    fn scalar_j2(
        rx: &[f64],
        ry: &[f64],
        rz: &[f64],
        mu: f64,
        j2: f64,
        re: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = rx.len();
        let (mut ax, mut ay, mut az) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let j2k = 1.5 * j2 * mu * re * re;
        for i in 0..n {
            let rm = (rx[i] * rx[i] + ry[i] * ry[i] + rz[i] * rz[i]).sqrt();
            let rm3 = rm * rm * rm;
            let rm5 = rm3 * rm * rm;
            let zr2 = (rz[i] / rm) * (rz[i] / rm);
            let k = j2k / rm5;
            ax[i] = -mu * rx[i] / rm3 + k * rx[i] * (5.0 * zr2 - 1.0);
            ay[i] = -mu * ry[i] / rm3 + k * ry[i] * (5.0 * zr2 - 1.0);
            az[i] = -mu * rz[i] / rm3 + k * rz[i] * (5.0 * zr2 - 3.0);
        }
        (ax, ay, az)
    }

    // 伪随机轨道状态（LEO/GEO/椭圆/高轨），避免巧合
    fn states(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut z = Vec::with_capacity(n);
        for i in 0..n {
            let a = 6.8e6 + (i as f64) * 1.7e6; // 500km .. 12万 km
            let e = 0.05 + 0.3 * (i as f64) / n as f64;
            let th = (i as f64) * 2.39996;
            let r = a * (1.0 - e * e) / (1.0 + e * (th * 1.7).cos());
            let ph = (i as f64) * 1.131;
            x.push(r * th.cos() * ph.cos());
            y.push(r * th.sin() * ph.cos());
            z.push(r * ph.sin());
        }
        (x, y, z)
    }

    /// 相对误差（|a−b| / max(1,|b|)），对 ~1e7 量级轨道状态用相对度量
    fn rel_err(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs().max(1.0)
    }

    #[test]
    fn test_norm3_batch_matches_scalar() {
        for n in [0usize, 1, 2, 3, 4, 5, 7, 8, 16, 33] {
            let (x, y, z) = states(n);
            let mut out = vec![0.0; n];
            unsafe {
                lasx_norm3_batch(
                    x.as_ptr(),
                    y.as_ptr(),
                    z.as_ptr(),
                    out.as_mut_ptr(),
                    n as i32,
                )
            };
            let want = scalar_norm3(&x, &y, &z);
            for i in 0..n {
                assert!(
                    rel_err(out[i], want[i]) < 1e-9,
                    "n={n} i={i}: {} vs {}",
                    out[i],
                    want[i]
                );
            }
        }
    }

    #[test]
    fn test_vec3_add_scaled_batch_matches_scalar() {
        let (x, y, z) = states(37);
        let (b, c, d) = states(37);
        for s in [0.0, 0.5, 2.0, -1.3, 1.0] {
            let (mut ox, mut oy, mut oz) = (vec![0.0; 37], vec![0.0; 37], vec![0.0; 37]);
            unsafe {
                lasx_vec3_add_scaled_batch(
                    x.as_ptr(),
                    y.as_ptr(),
                    z.as_ptr(),
                    b.as_ptr(),
                    c.as_ptr(),
                    d.as_ptr(),
                    s,
                    ox.as_mut_ptr(),
                    oy.as_mut_ptr(),
                    oz.as_mut_ptr(),
                    37,
                );
            }
            let wx = scalar_add_scaled(&x, &b, s);
            let wy = scalar_add_scaled(&y, &c, s);
            let wz = scalar_add_scaled(&z, &d, s);
            for i in 0..37 {
                assert!(
                    rel_err(ox[i], wx[i]) < 1e-9
                        && rel_err(oy[i], wy[i]) < 1e-9
                        && rel_err(oz[i], wz[i]) < 1e-9,
                    "s={s} i={i}"
                );
            }
        }
    }

    #[test]
    fn test_rk4_j2_step_batch_matches_scalar() {
        // 批量单步 RK4 vs 逐星标量（同公式），多步累积后仍 <1e-9
        let mu = 3.986004418e14;
        let j2 = 1.08262668e-3;
        let re = 6.378137e6;
        for n in [1usize, 2, 3, 4, 5, 7, 8, 17] {
            let (x, y, z) = states(n);
            // 速度
            let vx: Vec<f64> = (0..n).map(|i| 100.0 * (i as f64) + 500.0).collect();
            let vy: Vec<f64> = (0..n).map(|i| 7700.0 * (1.0 + 0.01 * (i as f64))).collect();
            let vz: Vec<f64> = (0..n).map(|i| 20.0 * (i as f64) as f64).collect();
            let (mut bx, mut by, mut bz) = (x.clone(), y.clone(), z.clone());
            let (mut bvx, mut bvy, mut bvz) = (vx.clone(), vy.clone(), vz.clone());
            // 参考：逐星标量 50 步
            let (mut sx, mut sy, mut sz) = (x.clone(), y.clone(), z.clone());
            let (mut svx, mut svy, mut svz) = (vx.clone(), vy.clone(), vz.clone());
            let dt = 10.0;
            for _ in 0..50 {
                unsafe {
                    lasx_rk4_j2_step_batch(
                        bx.as_mut_ptr(),
                        by.as_mut_ptr(),
                        bz.as_mut_ptr(),
                        bvx.as_mut_ptr(),
                        bvy.as_mut_ptr(),
                        bvz.as_mut_ptr(),
                        mu,
                        j2,
                        re,
                        dt,
                        n as i32,
                    );
                }
                for i in 0..n {
                    rk4_j2_step_scalar(
                        &mut sx[i],
                        &mut sy[i],
                        &mut sz[i],
                        &mut svx[i],
                        &mut svy[i],
                        &mut svz[i],
                        mu,
                        j2,
                        re,
                        dt,
                    );
                }
            }
            for i in 0..n {
                assert!(
                    rel_err(bx[i], sx[i]) < 1e-9 && rel_err(bvx[i], svx[i]) < 1e-9,
                    "n={n} i={i}: batch r {} vs scalar {}",
                    bx[i],
                    sx[i]
                );
                assert!(rel_err(by[i], sy[i]) < 1e-9 && rel_err(bvy[i], svy[i]) < 1e-9);
                assert!(rel_err(bz[i], sz[i]) < 1e-9 && rel_err(bvz[i], svz[i]) < 1e-9);
            }
        }
    }

    #[test]
    fn test_j2_accel_batch_matches_scalar() {
        for n in [0usize, 1, 2, 4, 9, 20] {
            let (x, y, z) = states(n);
            let (mut ax, mut ay, mut az) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            unsafe {
                lasx_j2_accel_batch(
                    x.as_ptr(),
                    y.as_ptr(),
                    z.as_ptr(),
                    3.986004418e14,
                    1.08262668e-3,
                    6.378137e6,
                    ax.as_mut_ptr(),
                    ay.as_mut_ptr(),
                    az.as_mut_ptr(),
                    n as i32,
                );
            }
            let (wx, wy, wz) = scalar_j2(&x, &y, &z, 3.986004418e14, 1.08262668e-3, 6.378137e6);
            for i in 0..n {
                assert!(
                    rel_err(ax[i], wx[i]) < 1e-9
                        && rel_err(ay[i], wy[i]) < 1e-9
                        && rel_err(az[i], wz[i]) < 1e-9,
                    "n={n} i={i}: ({},{},{}) vs ({},{},{})",
                    ax[i],
                    ay[i],
                    az[i],
                    wx[i],
                    wy[i],
                    wz[i]
                );
            }
        }
    }

    #[test]
    fn test_rk4_j2_step_scalar_fma_vs_plain_reference() {
        // 标量尾循环 FMA 版（f64::mul_add）vs 分离 mul+add 参考（原版公式）：
        // 单步 FMA 融合舍入差 ≤1 ulp，50 步累积后相对差仍 <1e-9（物理等价）。
        let mu = 3.986004418e14;
        let j2 = 1.08262668e-3;
        let re = 6.378137e6;
        let (x, y, z) = states(9);
        // 分离 mul+add 参考实现（与旧版 rk4_j2_step_scalar 同式）
        fn step_plain(
            rx: &mut f64,
            ry: &mut f64,
            rz: &mut f64,
            vx: &mut f64,
            vy: &mut f64,
            vz: &mut f64,
            mu: f64,
            j2: f64,
            re: f64,
            h: f64,
        ) {
            let j2k = 1.5 * j2 * mu * re * re;
            let accel = |x: f64, y: f64, z: f64| -> [f64; 3] {
                let rm = (x * x + y * y + z * z).sqrt();
                let rm3 = rm * rm * rm;
                let rm5 = rm3 * rm * rm;
                let zr2 = (z / rm) * (z / rm);
                let k = j2k / rm5;
                [
                    -mu * x / rm3 + k * x * (5.0 * zr2 - 1.0),
                    -mu * y / rm3 + k * y * (5.0 * zr2 - 1.0),
                    -mu * z / rm3 + k * z * (5.0 * zr2 - 3.0),
                ]
            };
            let (x0, y0, z0, vx0, vy0, vz0) = (*rx, *ry, *rz, *vx, *vy, *vz);
            let k1 = accel(x0, y0, z0);
            let r2 = (x0 + 0.5 * h * vx0, y0 + 0.5 * h * vy0, z0 + 0.5 * h * vz0);
            let v2 = (
                vx0 + 0.5 * h * k1[0],
                vy0 + 0.5 * h * k1[1],
                vz0 + 0.5 * h * k1[2],
            );
            let k2 = accel(r2.0, r2.1, r2.2);
            let r3 = (
                x0 + 0.5 * h * v2.0,
                y0 + 0.5 * h * v2.1,
                z0 + 0.5 * h * v2.2,
            );
            let v3 = (
                vx0 + 0.5 * h * k2[0],
                vy0 + 0.5 * h * k2[1],
                vz0 + 0.5 * h * k2[2],
            );
            let k3 = accel(r3.0, r3.1, r3.2);
            let r4 = (x0 + h * v3.0, y0 + h * v3.1, z0 + h * v3.2);
            let v4 = (vx0 + h * k3[0], vy0 + h * k3[1], vz0 + h * k3[2]);
            let k4 = accel(r4.0, r4.1, r4.2);
            let l = (
                vx0 + 2.0 * v2.0 + 2.0 * v3.0 + v4.0,
                vy0 + 2.0 * v2.1 + 2.0 * v3.1 + v4.1,
                vz0 + 2.0 * v2.2 + 2.0 * v3.2 + v4.2,
            );
            let k = (
                k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0],
                k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1],
                k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2],
            );
            *rx = x0 + (h / 6.0) * l.0;
            *ry = y0 + (h / 6.0) * l.1;
            *rz = z0 + (h / 6.0) * l.2;
            *vx = vx0 + (h / 6.0) * k.0;
            *vy = vy0 + (h / 6.0) * k.1;
            *vz = vz0 + (h / 6.0) * k.2;
        }
        let (mut fx, mut fy, mut fz) = (x.clone(), y.clone(), z.clone());
        let (mut fvx, mut fvy, mut fvz) = (vec![0.0; 9], vec![0.0; 9], vec![0.0; 9]);
        for i in 0..9 {
            fvx[i] = 500.0 + 100.0 * i as f64;
            fvy[i] = 7700.0 * (1.0 + 0.01 * i as f64);
            fvz[i] = 20.0 * i as f64;
        }
        let (mut px, mut py, mut pz) = (x.clone(), y.clone(), z.clone());
        let (mut pvx, mut pvy, mut pvz) = (fvx.clone(), fvy.clone(), fvz.clone());
        let dt = 10.0;
        for _ in 0..50 {
            for i in 0..9 {
                rk4_j2_step_scalar(
                    &mut fx[i],
                    &mut fy[i],
                    &mut fz[i],
                    &mut fvx[i],
                    &mut fvy[i],
                    &mut fvz[i],
                    mu,
                    j2,
                    re,
                    dt,
                );
                step_plain(
                    &mut px[i],
                    &mut py[i],
                    &mut pz[i],
                    &mut pvx[i],
                    &mut pvy[i],
                    &mut pvz[i],
                    mu,
                    j2,
                    re,
                    dt,
                );
            }
        }
        for i in 0..9 {
            assert!(
                rel_err(fx[i], px[i]) < 1e-9 && rel_err(fvx[i], pvx[i]) < 1e-9,
                "i={i}: FMA r {} vs plain {}",
                fx[i],
                px[i]
            );
            assert!(rel_err(fy[i], py[i]) < 1e-9 && rel_err(fvy[i], pvy[i]) < 1e-9);
            assert!(rel_err(fz[i], pz[i]) < 1e-9 && rel_err(fvz[i], pvz[i]) < 1e-9);
        }
    }
}

/// 单步 RK4 的 J2 加速度（向量 lane 版本，4 样本并行）：
/// a = −μ·r/|r|³ + J2 项（公式同 `lasx_j2_accel_batch`；vmu 已取负）
#[inline]
unsafe fn j2_accel_vec(
    vx: m256d,
    vy: m256d,
    vz: m256d,
    vmu: m256d,
    vj2k: m256d,
    v5: m256d,
    v1: m256d,
    v3: m256d,
) -> (m256d, m256d, m256d) {
    let vrm2 = lasx_xvfadd_d(
        lasx_xvfmul_d(vx, vx),
        lasx_xvfadd_d(lasx_xvfmul_d(vy, vy), lasx_xvfmul_d(vz, vz)),
    );
    let vrm = lasx_xvfsqrt_d(vrm2);
    let vrm3 = lasx_xvfmul_d(vrm, vrm2);
    let vrm5 = lasx_xvfmul_d(vrm3, vrm2);
    let vcen = lasx_xvfdiv_d(vmu, vrm3); // −μ/|r|³
    let vk = lasx_xvfdiv_d(vj2k, vrm5); // +1.5·J2·μ·Re²/|r|⁵
    let vzr2 = lasx_xvfdiv_d(lasx_xvfmul_d(vz, vz), vrm2);
    let m1 = lasx_xvfsub_d(lasx_xvfmul_d(v5, vzr2), v1); // 5·zr2−1
    let m3 = lasx_xvfsub_d(lasx_xvfmul_d(v5, vzr2), v3); // 5·zr2−3
    let ax = lasx_xvfmadd_d(lasx_xvfmul_d(vk, vx), m1, lasx_xvfmul_d(vcen, vx));
    let ay = lasx_xvfmadd_d(lasx_xvfmul_d(vk, vy), m1, lasx_xvfmul_d(vcen, vy));
    let az = lasx_xvfmadd_d(lasx_xvfmul_d(vk, vz), m3, lasx_xvfmul_d(vcen, vz));
    (ax, ay, az)
}

/// 标量单步 RK4（J2 力模型，供非 LASX 兜底/尾数；与 loong-sci `propagate_orbit` 同式）。
/// FMA 版：组合算术与加速度末项用 `f64::mul_add`（单指令 `fmadd.d`，LA664），与
/// LASX 向量路径 `j2_accel_vec`/内核组合算术逐位一致（物理等价 <1e-9）。
fn rk4_j2_step_scalar(
    rx: &mut f64,
    ry: &mut f64,
    rz: &mut f64,
    vx: &mut f64,
    vy: &mut f64,
    vz: &mut f64,
    mu: f64,
    j2: f64,
    re: f64,
    h: f64,
) {
    let j2k = 1.5 * j2 * mu * re * re;
    let accel = |x: f64, y: f64, z: f64| -> [f64; 3] {
        // 结合序与向量路径 `lasx_xvfadd_d(x·x, xvfadd_d(y·y, z·z))` 一致
        let rm2 = x * x + (y * y + z * z);
        let rm = rm2.sqrt();
        let rm3 = rm * rm2;
        let rm5 = rm3 * rm2;
        let zr2 = (z * z) / rm2; // 与向量路径 `lasx_xvfdiv_d(xvfmul(z,z), rm2)` 一致
        let k = j2k / rm5;
        let vcen = -mu / rm3;
        // a = k·x·(5·zr2−1) + vcen·x（FMA 融合末加，与 `j2_accel_vec` 同式）
        [
            f64::mul_add(k * x, 5.0 * zr2 - 1.0, vcen * x),
            f64::mul_add(k * y, 5.0 * zr2 - 1.0, vcen * y),
            f64::mul_add(k * z, 5.0 * zr2 - 3.0, vcen * z),
        ]
    };
    let (x0, y0, z0, vx0, vy0, vz0) = (*rx, *ry, *rz, *vx, *vy, *vz);
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
    let k = (
        2.0_f64.mul_add(k2[0], k1[0]) + 2.0_f64.mul_add(k3[0], k4[0]),
        2.0_f64.mul_add(k2[1], k1[1]) + 2.0_f64.mul_add(k3[1], k4[1]),
        2.0_f64.mul_add(k2[2], k1[2]) + 2.0_f64.mul_add(k3[2], k4[2]),
    );
    *rx = h6.mul_add(l.0, x0);
    *ry = h6.mul_add(l.1, y0);
    *rz = h6.mul_add(l.2, z0);
    *vx = h6.mul_add(k.0, vx0);
    *vy = h6.mul_add(k.1, vy0);
    *vz = h6.mul_add(k.2, vz0);
}

/// 批量单步 RK4（中心项 + J2 力模型）：N 个轨道状态 [r(3),v(3)] 同时推进一步。
/// SOA 布局（rx/ry/rz/vx/vy/vz 各为 n 长连续数组），LASX 一次处理 4 个样本；
/// 全部 k1..k4 / 组合算术在寄存器内完成（少一次内存往返即省 30+ 次内核调用开销）。
/// 原地更新（每块先读后写，块间不重叠）；非 LASX CPU 走全标量。
#[unsafe(no_mangle)]
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
    let rx = unsafe { std::slice::from_raw_parts_mut(rx, n) };
    let ry = unsafe { std::slice::from_raw_parts_mut(ry, n) };
    let rz = unsafe { std::slice::from_raw_parts_mut(rz, n) };
    let vx = unsafe { std::slice::from_raw_parts_mut(vx, n) };
    let vy = unsafe { std::slice::from_raw_parts_mut(vy, n) };
    let vz = unsafe { std::slice::from_raw_parts_mut(vz, n) };
    if has_lasx() {
        let vmu = splat_f64(-mu);
        let vj2k = splat_f64(1.5 * j2 * mu * re * re);
        let v5 = splat_f64(5.0);
        let v1 = splat_f64(1.0);
        let v3s = splat_f64(3.0);
        let vh = splat_f64(dt);
        let vh6 = splat_f64(dt / 6.0);
        let vh2 = splat_f64(0.5 * dt);
        let v2s = splat_f64(2.0);
        let mut i = 0;
        while i + 4 <= n {
            let lrx = unsafe { ld_f64(rx.as_ptr().add(i)) };
            let lry = unsafe { ld_f64(ry.as_ptr().add(i)) };
            let lrz = unsafe { ld_f64(rz.as_ptr().add(i)) };
            let lvx = unsafe { ld_f64(vx.as_ptr().add(i)) };
            let lvy = unsafe { ld_f64(vy.as_ptr().add(i)) };
            let lvz = unsafe { ld_f64(vz.as_ptr().add(i)) };
            let (k1x, k1y, k1z) = unsafe { j2_accel_vec(lrx, lry, lrz, vmu, vj2k, v5, v1, v3s) };
            let r2 = (
                unsafe { lasx_xvfmadd_d(vh2, lvx, lrx) },
                unsafe { lasx_xvfmadd_d(vh2, lvy, lry) },
                unsafe { lasx_xvfmadd_d(vh2, lvz, lrz) },
            );
            let v2 = (
                unsafe { lasx_xvfmadd_d(vh2, k1x, lvx) },
                unsafe { lasx_xvfmadd_d(vh2, k1y, lvy) },
                unsafe { lasx_xvfmadd_d(vh2, k1z, lvz) },
            );
            let (k2x, k2y, k2z) = unsafe { j2_accel_vec(r2.0, r2.1, r2.2, vmu, vj2k, v5, v1, v3s) };
            let r3 = (
                unsafe { lasx_xvfmadd_d(vh2, v2.0, lrx) },
                unsafe { lasx_xvfmadd_d(vh2, v2.1, lry) },
                unsafe { lasx_xvfmadd_d(vh2, v2.2, lrz) },
            );
            let v3 = (
                unsafe { lasx_xvfmadd_d(vh2, k2x, lvx) },
                unsafe { lasx_xvfmadd_d(vh2, k2y, lvy) },
                unsafe { lasx_xvfmadd_d(vh2, k2z, lvz) },
            );
            let (k3x, k3y, k3z) = unsafe { j2_accel_vec(r3.0, r3.1, r3.2, vmu, vj2k, v5, v1, v3s) };
            let r4 = (
                unsafe { lasx_xvfmadd_d(vh, v3.0, lrx) },
                unsafe { lasx_xvfmadd_d(vh, v3.1, lry) },
                unsafe { lasx_xvfmadd_d(vh, v3.2, lrz) },
            );
            let v4 = (
                unsafe { lasx_xvfmadd_d(vh, k3x, lvx) },
                unsafe { lasx_xvfmadd_d(vh, k3y, lvy) },
                unsafe { lasx_xvfmadd_d(vh, k3z, lvz) },
            );
            let (k4x, k4y, k4z) = unsafe { j2_accel_vec(r4.0, r4.1, r4.2, vmu, vj2k, v5, v1, v3s) };
            // lsum = v + 2·v2 + 2·v3 + v4；ksum 同理
            let l = (
                unsafe {
                    lasx_xvfadd_d(
                        lasx_xvfmadd_d(v2s, v2.0, lvx),
                        lasx_xvfmadd_d(v2s, v3.0, v4.0),
                    )
                },
                unsafe {
                    lasx_xvfadd_d(
                        lasx_xvfmadd_d(v2s, v2.1, lvy),
                        lasx_xvfmadd_d(v2s, v3.1, v4.1),
                    )
                },
                unsafe {
                    lasx_xvfadd_d(
                        lasx_xvfmadd_d(v2s, v2.2, lvz),
                        lasx_xvfmadd_d(v2s, v3.2, v4.2),
                    )
                },
            );
            let k = (
                unsafe {
                    lasx_xvfadd_d(lasx_xvfmadd_d(v2s, k2x, k1x), lasx_xvfmadd_d(v2s, k3x, k4x))
                },
                unsafe {
                    lasx_xvfadd_d(lasx_xvfmadd_d(v2s, k2y, k1y), lasx_xvfmadd_d(v2s, k3y, k4y))
                },
                unsafe {
                    lasx_xvfadd_d(lasx_xvfmadd_d(v2s, k2z, k1z), lasx_xvfmadd_d(v2s, k3z, k4z))
                },
            );
            let nr = (
                unsafe { lasx_xvfmadd_d(vh6, l.0, lrx) },
                unsafe { lasx_xvfmadd_d(vh6, l.1, lry) },
                unsafe { lasx_xvfmadd_d(vh6, l.2, lrz) },
            );
            let nv = (
                unsafe { lasx_xvfmadd_d(vh6, k.0, lvx) },
                unsafe { lasx_xvfmadd_d(vh6, k.1, lvy) },
                unsafe { lasx_xvfmadd_d(vh6, k.2, lvz) },
            );
            unsafe {
                st_f64(rx.as_mut_ptr().add(i), nr.0);
                st_f64(ry.as_mut_ptr().add(i), nr.1);
                st_f64(rz.as_mut_ptr().add(i), nr.2);
                st_f64(vx.as_mut_ptr().add(i), nv.0);
                st_f64(vy.as_mut_ptr().add(i), nv.1);
                st_f64(vz.as_mut_ptr().add(i), nv.2);
            }
            i += 4;
        }
        for j in i..n {
            rk4_j2_step_scalar(
                &mut rx[j], &mut ry[j], &mut rz[j], &mut vx[j], &mut vy[j], &mut vz[j], mu, j2, re,
                dt,
            );
        }
    } else {
        for j in 0..n {
            rk4_j2_step_scalar(
                &mut rx[j], &mut ry[j], &mut rz[j], &mut vx[j], &mut vy[j], &mut vz[j], mu, j2, re,
                dt,
            );
        }
    }
}
