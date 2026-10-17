//! `dsl_fft` —— 给 `simd_dsl::Backend::fft_strided_f32` 提供 **LASX 原地按列 radix-2 复数 FFT**（R19-H）。
//!
//! **为什么这样切**：谱检测帧里 FFT 是"按列"的（`[n, b]` 行主序、每列一条信号），而 b=256 是
//! **8 的倍数** ⇒ 沿**列**切 8 条 lane 正好是 LASX 的 256 位；旋转因子在同一蝶形里对 8 列是**同一个标量**
//! ⇒ 广播即可，不需要 gather/shuffle。**这样避免转置**（§14.11/§14.13 已实测转置是瓶颈）。
//!
//! **等价类**：**容差类**（与 `simd_dsl::kernels::scalar::fft_radix2_strided_f32` 的 f64 内部实现比）：
//! 旋转因子在 f64 里算好再降 f32、蝶形在 f32 里做 ⇒ 每级都有舍入。实测界由 `dsl_backend.rs` 的测试钉住。
//! **语义**：正变换、`out[k] = Σ x[j]e^{−2πijk/n}`、原地、`n` 必须是 2 的幂（调用方校验）。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（分层策略：内核模块在函数级统一写 SAFETY，不逐块重复）。
#![allow(clippy::undocumented_unsafe_blocks)]
use crate::arch::lasx;
use std::arch::loongarch64::*;

/// 预计算旋转因子：`e^{−2πik/n}`（`k in 0..n/2`）。f64 里算、降 f32。
fn twiddle_table(n: usize) -> Vec<(f32, f32)> {
    let half = n / 2;
    let mut t = Vec::with_capacity(half);
    let scale = -2.0 * std::f64::consts::PI / n as f64;
    for k in 0..half {
        let a = scale * k as f64;
        t.push((a.cos() as f32, a.sin() as f32));
    }
    t
}

/// 交换两**行**（每行 `b` 个元素；8 路向量化 + 标量尾）。
#[inline]
fn swap_rows(re: &mut [f32], im: &mut [f32], i: usize, j: usize, b: usize) {
    let mut c = 0;
    while c + 8 <= b {
        let (oi, oj) = (i * b + c, j * b + c);
        // SAFETY: `c + 8 <= b` ⇒ 两条 8 元素访问都在各自行内；`i != j`（i<j 且位反序不等）⇒ 不重叠。
        unsafe {
            for arr in [&mut *re, &mut *im] {
                let p = arr.as_mut_ptr();
                let a = lasx::load_f32x8(p.add(oi));
                let bb = lasx::load_f32x8(p.add(oj));
                lasx::store_f32x8(p.add(oi), bb);
                lasx::store_f32x8(p.add(oj), a);
            }
        }
        c += 8;
    }
    for c in c..b {
        let (oi, oj) = (i * b + c, j * b + c);
        re.swap(oi, oj);
        im.swap(oi, oj);
    }
}

/// 单个蝶形：`u = z[i0]`、`v = z[i1]·w`；`z[i0] = u+v`、`z[i1] = u−v`（沿列 8 路）。
#[inline]
fn butterfly(re: &mut [f32], im: &mut [f32], b: usize, i0: usize, i1: usize, wr: f32, wi: f32) {
    let (vr_w, vi_w) = (lasx::splat_f32(wr), lasx::splat_f32(wi));
    let (p0, p1) = (i0 * b, i1 * b);
    let mut c = 0;
    while c + 8 <= b {
        // SAFETY: `c + 8 <= b` ⇒ 4 条 8 元素访问都在各自行内；`i0 != i1` ⇒ 无重叠。
        unsafe {
            let rp = re.as_mut_ptr();
            let ip = im.as_mut_ptr();
            let vr = lasx::load_f32x8(rp.add(p0 + c));
            let vi = lasx::load_f32x8(ip.add(p0 + c));
            let xr = lasx::load_f32x8(rp.add(p1 + c));
            let xi = lasx::load_f32x8(ip.add(p1 + c));
            // t = x · w（复数乘；FMA：(-xi)·wi + xr·wr、xr·wi + xi·wr）
            let tr = lasx_xvfmadd_s(lasx::neg_f32x8(xi), vi_w, lasx_xvfmul_s(xr, vr_w));
            let ti = lasx_xvfmadd_s(xr, vi_w, lasx_xvfmul_s(xi, vr_w));
            lasx::store_f32x8(rp.add(p0 + c), lasx_xvfadd_s(vr, tr));
            lasx::store_f32x8(ip.add(p0 + c), lasx_xvfadd_s(vi, ti));
            lasx::store_f32x8(rp.add(p1 + c), lasx_xvfsub_s(vr, tr));
            lasx::store_f32x8(ip.add(p1 + c), lasx_xvfsub_s(vi, ti));
        }
        c += 8;
    }
    for c in c..b {
        let (ur, ui) = (re[p0 + c], im[p0 + c]);
        let (xr, xi) = (re[p1 + c], im[p1 + c]);
        let tr = xr * wr - xi * wi;
        let ti = xr * wi + xi * wr;
        re[p0 + c] = ur + tr;
        im[p0 + c] = ui + ti;
        re[p1 + c] = ur - tr;
        im[p1 + c] = ui - ti;
    }
}

/// 原地按列 radix-2 复数 FFT（正变换）。**前提**：`n` 是 2 的幂、`re/im` 长度都是 `n*b`。
pub(crate) fn fft_strided(re: &mut [f32], im: &mut [f32], n: usize, b: usize) {
    debug_assert!(n.is_power_of_two() && n >= 2);
    debug_assert!(re.len() == n * b && im.len() == n * b);
    let levels = n.trailing_zeros() as usize;
    let tw = twiddle_table(n);
    // 1) 位反序置换（按行交换）
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS as usize - levels);
        if i < j {
            swap_rows(re, im, i, j, b);
        }
    }
    // 2) 逐级蝶形
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let step = n / len;
        for start in (0..n).step_by(len) {
            for j in 0..half {
                let (wr, wi) = tw[(j * step).min(tw.len() - 1)];
                butterfly(re, im, b, start + j, start + j + half, wr, wi);
            }
        }
        len *= 2;
    }
}

/// Σ|v|² 之类的便利：本模块暂不需要，留待后续（不导出）。
#[allow(dead_code)]
fn _unused() {}
