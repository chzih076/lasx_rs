//! `lasx_matmul_f64` —— f64 矩阵乘 `C[m×n] = A[m×k]·B[k×n]`（行主序，列方向向量化）。
//!
//! 与 [`crate::ops::matmul`] 同构，只是向量宽度为 4×f64、列块取 16：
//! 一次算 4 行 × 16 列（16 个 4 通道累加器），同一段 B 被 4 行复用。
//! 不做 B 转置、不做跨 lane 水平归约。
//!
//! 数值行为不变：每个输出元素仍是沿 k 的单个 f64 累加器，累加次序与 v2 一致。
//!
//! **LASX-only**：没有降级分支，无 LASX 的 CPU 上会执行 LASX 指令（见手册 Caveats）。

use crate::arch::lasx;
use std::arch::loongarch64::*;

/// f64 矩阵乘 `C[m×n] = A[m×k]·B[k×n]`（行主序）。
#[inline]
pub(crate) fn matmul_f64(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    if m == 0 || n == 0 {
        return;
    }
    let mut i = 0;
    while i + 4 <= m {
        rows4_f64(i, k, n, a, b, c);
        i += 4;
    }
    while i < m {
        let a_row = &a[i * k..(i + 1) * k];
        let c_row = &mut c[i * n..(i + 1) * n];
        cols16_f64(a_row, c_row, b, n);
        row_tail_f64(a_row, c_row, b, k, n, n / 16 * 16);
        i += 1;
    }
}

/// 一次算 4 行 × 16 列：16 个累加器，B 的 4 个向量被 4 行共享。
///
/// `p` 既用于索引 4 行 A 又用于计算 B 的地址，故保留下标循环。
#[inline]
#[allow(clippy::needless_range_loop)]
fn rows4_f64(i0: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    let a0 = &a[i0 * k..(i0 + 1) * k];
    let a1 = &a[(i0 + 1) * k..(i0 + 2) * k];
    let a2 = &a[(i0 + 2) * k..(i0 + 3) * k];
    let a3 = &a[(i0 + 3) * k..(i0 + 4) * k];

    let mut rows = c[i0 * n..(i0 + 4) * n].chunks_mut(n);
    let c0 = rows.next().expect("4 行");
    let c1 = rows.next().expect("4 行");
    let c2 = rows.next().expect("4 行");
    let c3 = rows.next().expect("4 行");

    let nb = n / 16 * 16;
    let mut j = 0;
    while j + 16 <= n {
        let (mut r0a, mut r0b, mut r0c, mut r0d) = z4();
        let (mut r1a, mut r1b, mut r1c, mut r1d) = z4();
        let (mut r2a, mut r2b, mut r2c, mut r2d) = z4();
        let (mut r3a, mut r3b, mut r3c, mut r3d) = z4();
        for p in 0..k {
            let base = p * n + j;
            let vb0 = unsafe { lasx::load_f64x4(b.as_ptr().add(base)) };
            let vb1 = unsafe { lasx::load_f64x4(b.as_ptr().add(base + 4)) };
            let vb2 = unsafe { lasx::load_f64x4(b.as_ptr().add(base + 8)) };
            let vb3 = unsafe { lasx::load_f64x4(b.as_ptr().add(base + 12)) };
            let s0 = lasx::splat_f64(a0[p]);
            let s1 = lasx::splat_f64(a1[p]);
            let s2 = lasx::splat_f64(a2[p]);
            let s3 = lasx::splat_f64(a3[p]);
            unsafe {
                r0a = lasx_xvfmadd_d(vb0, s0, r0a);
                r0b = lasx_xvfmadd_d(vb1, s0, r0b);
                r0c = lasx_xvfmadd_d(vb2, s0, r0c);
                r0d = lasx_xvfmadd_d(vb3, s0, r0d);
                r1a = lasx_xvfmadd_d(vb0, s1, r1a);
                r1b = lasx_xvfmadd_d(vb1, s1, r1b);
                r1c = lasx_xvfmadd_d(vb2, s1, r1c);
                r1d = lasx_xvfmadd_d(vb3, s1, r1d);
                r2a = lasx_xvfmadd_d(vb0, s2, r2a);
                r2b = lasx_xvfmadd_d(vb1, s2, r2b);
                r2c = lasx_xvfmadd_d(vb2, s2, r2c);
                r2d = lasx_xvfmadd_d(vb3, s2, r2d);
                r3a = lasx_xvfmadd_d(vb0, s3, r3a);
                r3b = lasx_xvfmadd_d(vb1, s3, r3b);
                r3c = lasx_xvfmadd_d(vb2, s3, r3c);
                r3d = lasx_xvfmadd_d(vb3, s3, r3d);
            }
        }
        unsafe {
            lasx::store_f64x4(c0.as_mut_ptr().add(j), r0a);
            lasx::store_f64x4(c0.as_mut_ptr().add(j + 4), r0b);
            lasx::store_f64x4(c0.as_mut_ptr().add(j + 8), r0c);
            lasx::store_f64x4(c0.as_mut_ptr().add(j + 12), r0d);
            lasx::store_f64x4(c1.as_mut_ptr().add(j), r1a);
            lasx::store_f64x4(c1.as_mut_ptr().add(j + 4), r1b);
            lasx::store_f64x4(c1.as_mut_ptr().add(j + 8), r1c);
            lasx::store_f64x4(c1.as_mut_ptr().add(j + 12), r1d);
            lasx::store_f64x4(c2.as_mut_ptr().add(j), r2a);
            lasx::store_f64x4(c2.as_mut_ptr().add(j + 4), r2b);
            lasx::store_f64x4(c2.as_mut_ptr().add(j + 8), r2c);
            lasx::store_f64x4(c2.as_mut_ptr().add(j + 12), r2d);
            lasx::store_f64x4(c3.as_mut_ptr().add(j), r3a);
            lasx::store_f64x4(c3.as_mut_ptr().add(j + 4), r3b);
            lasx::store_f64x4(c3.as_mut_ptr().add(j + 8), r3c);
            lasx::store_f64x4(c3.as_mut_ptr().add(j + 12), r3d);
        }
        j += 16;
    }
    row_tail_f64(a0, c0, b, k, n, nb);
    row_tail_f64(a1, c1, b, k, n, nb);
    row_tail_f64(a2, c2, b, k, n, nb);
    row_tail_f64(a3, c3, b, k, n, nb);
}

/// 单行主循环：16 列一块（4 个累加器）。
#[inline]
fn cols16_f64(a_row: &[f64], c_row: &mut [f64], b: &[f64], n: usize) {
    let mut j = 0;
    while j + 16 <= n {
        let (mut acc0, mut acc1, mut acc2, mut acc3) = z4();
        for (p, &a_p) in a_row.iter().enumerate() {
            let va = lasx::splat_f64(a_p);
            let base = p * n + j;
            let b0 = unsafe { lasx::load_f64x4(b.as_ptr().add(base)) };
            let b1 = unsafe { lasx::load_f64x4(b.as_ptr().add(base + 4)) };
            let b2 = unsafe { lasx::load_f64x4(b.as_ptr().add(base + 8)) };
            let b3 = unsafe { lasx::load_f64x4(b.as_ptr().add(base + 12)) };
            unsafe {
                acc0 = lasx_xvfmadd_d(b0, va, acc0);
                acc1 = lasx_xvfmadd_d(b1, va, acc1);
                acc2 = lasx_xvfmadd_d(b2, va, acc2);
                acc3 = lasx_xvfmadd_d(b3, va, acc3);
            }
        }
        unsafe {
            lasx::store_f64x4(c_row.as_mut_ptr().add(j), acc0);
            lasx::store_f64x4(c_row.as_mut_ptr().add(j + 4), acc1);
            lasx::store_f64x4(c_row.as_mut_ptr().add(j + 8), acc2);
            lasx::store_f64x4(c_row.as_mut_ptr().add(j + 12), acc3);
        }
        j += 16;
    }
}

/// 单行的列尾：`[j0, n)`，先 4 列块再标量。
#[inline]
fn row_tail_f64(a_row: &[f64], c_row: &mut [f64], b: &[f64], k: usize, n: usize, j0: usize) {
    let mut j = j0;
    while j + 4 <= n {
        let mut acc = lasx::zero_f64x4();
        for (p, &a_p) in a_row.iter().enumerate() {
            let va = lasx::splat_f64(a_p);
            let vb = unsafe { lasx::load_f64x4(b.as_ptr().add(p * n + j)) };
            acc = unsafe { lasx_xvfmadd_d(vb, va, acc) };
        }
        unsafe { lasx::store_f64x4(c_row.as_mut_ptr().add(j), acc) };
        j += 4;
    }
    for jj in j..n {
        let mut s = 0f64;
        for p in 0..k {
            s += a_row[p] * b[p * n + jj];
        }
        c_row[jj] = s;
    }
}

/// 4 个零累加器。
#[inline]
fn z4() -> (lasx::F64x4, lasx::F64x4, lasx::F64x4, lasx::F64x4) {
    (
        lasx::zero_f64x4(),
        lasx::zero_f64x4(),
        lasx::zero_f64x4(),
        lasx::zero_f64x4(),
    )
}

/// 数值回归测试：对照独立参考实现。
#[cfg(test)]
mod tests {
    use crate::ffi::matmul::lasx_matmul_f64;
    use crate::ops::testutil::{reference, rel, Lcg};

    #[test]
    fn test_matmul_f64_matches_reference() {
        let shapes = [
            (0usize, 4usize, 3usize),
            (1, 1, 1),
            (1, 0, 5),
            (1, 4, 1),
            (3, 5, 7),
            (8, 8, 8),
            (16, 17, 19),
            (32, 32, 32),
            (33, 65, 31),
            (64, 64, 64),
        ];
        for &(m, k, n) in &shapes {
            let mut rng = Lcg(0xbeef_5678 ^ ((m * 31 + k) * 17 + n) as u64);
            let a: Vec<f64> = (0..m * k).map(|_| rng.f64()).collect();
            let b: Vec<f64> = (0..k * n).map(|_| rng.f64()).collect();
            let mut c = vec![f64::NAN; m * n];
            lasx_matmul_f64(
                m as i32,
                k as i32,
                n as i32,
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
            );
            let want = reference(m, k, n, &a, &b);
            for idx in 0..m * n {
                assert!(
                    rel(c[idx], want[idx]) < 1e-12,
                    "{m}×{k}×{n} idx={idx}: got {} want {}",
                    c[idx],
                    want[idx]
                );
            }
        }
    }
}
