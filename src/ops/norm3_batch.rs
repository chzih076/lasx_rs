//! `lasx_norm3_batch` —— 批量 3 分量模长 `out[i] = √(x²+y²+z²)`（f64 SOA）。
//!
//! LASX 缺失时走 LSX 128 位路径。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（分层策略：内核模块在函数级统一写 SAFETY，不逐块重复）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX/LSX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]
use crate::arch::SimdPath;
use crate::arch::{lasx, lsx};
use std::arch::loongarch64::*;

/// 算子入口：解析当前线程的向量路径后分派。
#[inline]
pub(crate) fn norm3_batch(xs: &[f64], ys: &[f64], zs: &[f64], out: &mut [f64]) {
    match SimdPath::detect() {
        SimdPath::Lasx => norm3_batch_lasx(xs, ys, zs, out),
        SimdPath::Lsx => norm3_batch_lsx(xs, ys, zs, out),
    }
}

/// LASX 256 位实现。
#[inline]
fn norm3_batch_lasx(xs: &[f64], ys: &[f64], zs: &[f64], out: &mut [f64]) {
    let n = xs.len();
    let mut i = 0;
    // **2 路展开（8 样本/迭代）**：每样本的结合次序是**逐元素固定**的（`x² + (y²+z²)` 再开方）
    // ⇒ 展开**不改变任何数值**（按构造与改前逐位相同），只为摊薄循环开销、提高访存并行度。
    // 动机：单路时 1M 档比 LLVM 自动向量化略慢（0.98×）；同型循环在 `axpy` / `vec3_add_scaled_batch`
    // 上展开后分别拿回 23% / 3.5%。
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n` ⇒ 两组访存分别落在 [i,i+4)、[i+4,i+8)，全在 `n` 之内；
        // 四个切片等长（`api` 入口已校验），`out` 是新分配、与输入不重叠。
        unsafe {
            let (p, q) = (i, i + 4);
            let x0 = lasx::load_f64x4(xs.as_ptr().add(p));
            let y0 = lasx::load_f64x4(ys.as_ptr().add(p));
            let z0 = lasx::load_f64x4(zs.as_ptr().add(p));
            let x1 = lasx::load_f64x4(xs.as_ptr().add(q));
            let y1 = lasx::load_f64x4(ys.as_ptr().add(q));
            let z1 = lasx::load_f64x4(zs.as_ptr().add(q));
            let s0 = lasx_xvfadd_d(
                lasx_xvfmul_d(x0, x0),
                lasx_xvfadd_d(lasx_xvfmul_d(y0, y0), lasx_xvfmul_d(z0, z0)),
            );
            let s1 = lasx_xvfadd_d(
                lasx_xvfmul_d(x1, x1),
                lasx_xvfadd_d(lasx_xvfmul_d(y1, y1), lasx_xvfmul_d(z1, z1)),
            );
            lasx::store_f64x4(out.as_mut_ptr().add(p), lasx_xvfsqrt_d(s0));
            lasx::store_f64x4(out.as_mut_ptr().add(q), lasx_xvfsqrt_d(s1));
        }
        i += 8;
    }
    while i + 4 <= n {
        let vx = unsafe { lasx::load_f64x4(xs.as_ptr().add(i)) };
        let vy = unsafe { lasx::load_f64x4(ys.as_ptr().add(i)) };
        let vz = unsafe { lasx::load_f64x4(zs.as_ptr().add(i)) };
        let sq = unsafe {
            lasx_xvfadd_d(
                lasx_xvfmul_d(vx, vx),
                lasx_xvfadd_d(lasx_xvfmul_d(vy, vy), lasx_xvfmul_d(vz, vz)),
            )
        };
        let mag = unsafe { lasx_xvfsqrt_d(sq) };
        unsafe { lasx::store_f64x4(out.as_mut_ptr().add(i), mag) };
        i += 4;
    }
    for j in i..n {
        // 与向量路径同结合（x² + (y²+z²)），保证任一分块逐位一致
        out[j] = (xs[j] * xs[j] + (ys[j] * ys[j] + zs[j] * zs[j])).sqrt();
    }
}

/// LSX 128 位实现（2 样本/向量）。
#[inline]
fn norm3_batch_lsx(xs: &[f64], ys: &[f64], zs: &[f64], out: &mut [f64]) {
    let n = xs.len();
    let mut i = 0;
    while i + 2 <= n {
        let vx = unsafe { lsx::load_f64x2(xs.as_ptr().add(i)) };
        let vy = unsafe { lsx::load_f64x2(ys.as_ptr().add(i)) };
        let vz = unsafe { lsx::load_f64x2(zs.as_ptr().add(i)) };
        let sq = unsafe {
            lsx_vfadd_d(
                lsx_vfmul_d(vx, vx),
                lsx_vfadd_d(lsx_vfmul_d(vy, vy), lsx_vfmul_d(vz, vz)),
            )
        };
        let mag = unsafe { lsx_vfsqrt_d(sq) };
        unsafe { lsx::store_f64x2(out.as_mut_ptr().add(i), mag) };
        i += 2;
    }
    for j in i..n {
        out[j] = (xs[j] * xs[j] + (ys[j] * ys[j] + zs[j] * zs[j])).sqrt();
    }
}

/// 数值回归测试：对照独立参考实现。
#[cfg(test)]
mod tests {
    use crate::ffi::batch::lasx_norm3_batch;
    use crate::ops::testutil::{rel_err, states};

    fn scalar_norm3(x: &[f64], y: &[f64], z: &[f64]) -> Vec<f64> {
        (0..x.len())
            .map(|i| (x[i] * x[i] + y[i] * y[i] + z[i] * z[i]).sqrt())
            .collect()
    }
    #[test]
    fn test_norm3_batch_matches_scalar() {
        for n in [0usize, 1, 2, 3, 4, 5, 7, 8, 16, 33] {
            let (x, y, z) = states(n);
            let mut out = vec![0.0; n];
            lasx_norm3_batch(
                x.as_ptr(),
                y.as_ptr(),
                z.as_ptr(),
                out.as_mut_ptr(),
                n as i32,
            );
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
}
