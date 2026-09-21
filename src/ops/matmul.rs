//! `lasx_matmul` —— f32 矩阵乘 `C[m×n] = A[m×k]·B[k×n]`（行主序，列方向向量化）。
//!
//! 微内核是 i-k-j 形式：A 的行元素广播、B 沿列方向连续流式读取，**不做 B 转置、
//! 不做跨 lane 水平归约**（这两点是 v1 慢于朴素三重循环的原因）。
//!
//! v3 再叠加**行分块**：一次算 4 行 × 32 列（16 个 8 通道累加器），同一段 B 被 4
//! 行复用，B 的重复读取量降到 1/4。256³ 时 B 为 256 KiB、放不进 L1，这一项是主要瓶颈。
//!
//! 数值行为不变：每个输出元素仍是**沿 k 的单个 f32 累加器**，累加次序与 v2 完全一致。
//!
//! **LASX-only**：没有降级分支，无 LASX 的 CPU 上会执行 LASX 指令（见手册 Caveats）。

use crate::arch::lasx;
use std::arch::loongarch64::*;

/// f32 矩阵乘 `C[m×n] = A[m×k]·B[k×n]`（行主序）。
#[inline]
pub(crate) fn matmul_f32(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    if m == 0 || n == 0 {
        return;
    }
    let mut i = 0;
    // 主路径：4 行一块（B 复用 4 次）
    while i + 4 <= m {
        rows4_f32(i, k, n, a, b, c);
        i += 4;
    }
    // 尾部不足 4 行
    while i < m {
        let a_row = &a[i * k..(i + 1) * k];
        let c_row = &mut c[i * n..(i + 1) * n];
        cols32_f32(a_row, c_row, b, n);
        row_tail_f32(a_row, c_row, b, k, n, n / 32 * 32);
        i += 1;
    }
}

/// 一次算 4 行 × 32 列：16 个累加器，B 的 4 个向量被 4 行共享。
///
/// `p` 既用于索引 4 行 A 又用于计算 B 的地址，故保留下标循环。
#[inline]
#[allow(clippy::needless_range_loop)]
fn rows4_f32(i0: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let a0 = &a[i0 * k..(i0 + 1) * k];
    let a1 = &a[(i0 + 1) * k..(i0 + 2) * k];
    let a2 = &a[(i0 + 2) * k..(i0 + 3) * k];
    let a3 = &a[(i0 + 3) * k..(i0 + 4) * k];

    // 4 行 C 互不重叠，用 chunks_mut 安全地同时取出
    let mut rows = c[i0 * n..(i0 + 4) * n].chunks_mut(n);
    let c0 = rows.next().expect("4 行");
    let c1 = rows.next().expect("4 行");
    let c2 = rows.next().expect("4 行");
    let c3 = rows.next().expect("4 行");

    let nb = n / 32 * 32;
    let mut j = 0;
    while j + 32 <= n {
        let (mut r0a, mut r0b, mut r0c, mut r0d) = z4();
        let (mut r1a, mut r1b, mut r1c, mut r1d) = z4();
        let (mut r2a, mut r2b, mut r2c, mut r2d) = z4();
        let (mut r3a, mut r3b, mut r3c, mut r3d) = z4();
        for p in 0..k {
            let base = p * n + j;
            let vb0 = unsafe { lasx::load_f32x8(b.as_ptr().add(base)) };
            let vb1 = unsafe { lasx::load_f32x8(b.as_ptr().add(base + 8)) };
            let vb2 = unsafe { lasx::load_f32x8(b.as_ptr().add(base + 16)) };
            let vb3 = unsafe { lasx::load_f32x8(b.as_ptr().add(base + 24)) };
            let s0 = lasx::splat_f32(a0[p]);
            let s1 = lasx::splat_f32(a1[p]);
            let s2 = lasx::splat_f32(a2[p]);
            let s3 = lasx::splat_f32(a3[p]);
            unsafe {
                r0a = lasx_xvfmadd_s(vb0, s0, r0a);
                r0b = lasx_xvfmadd_s(vb1, s0, r0b);
                r0c = lasx_xvfmadd_s(vb2, s0, r0c);
                r0d = lasx_xvfmadd_s(vb3, s0, r0d);
                r1a = lasx_xvfmadd_s(vb0, s1, r1a);
                r1b = lasx_xvfmadd_s(vb1, s1, r1b);
                r1c = lasx_xvfmadd_s(vb2, s1, r1c);
                r1d = lasx_xvfmadd_s(vb3, s1, r1d);
                r2a = lasx_xvfmadd_s(vb0, s2, r2a);
                r2b = lasx_xvfmadd_s(vb1, s2, r2b);
                r2c = lasx_xvfmadd_s(vb2, s2, r2c);
                r2d = lasx_xvfmadd_s(vb3, s2, r2d);
                r3a = lasx_xvfmadd_s(vb0, s3, r3a);
                r3b = lasx_xvfmadd_s(vb1, s3, r3b);
                r3c = lasx_xvfmadd_s(vb2, s3, r3c);
                r3d = lasx_xvfmadd_s(vb3, s3, r3d);
            }
        }
        unsafe {
            lasx::store_f32x8(c0.as_mut_ptr().add(j), r0a);
            lasx::store_f32x8(c0.as_mut_ptr().add(j + 8), r0b);
            lasx::store_f32x8(c0.as_mut_ptr().add(j + 16), r0c);
            lasx::store_f32x8(c0.as_mut_ptr().add(j + 24), r0d);
            lasx::store_f32x8(c1.as_mut_ptr().add(j), r1a);
            lasx::store_f32x8(c1.as_mut_ptr().add(j + 8), r1b);
            lasx::store_f32x8(c1.as_mut_ptr().add(j + 16), r1c);
            lasx::store_f32x8(c1.as_mut_ptr().add(j + 24), r1d);
            lasx::store_f32x8(c2.as_mut_ptr().add(j), r2a);
            lasx::store_f32x8(c2.as_mut_ptr().add(j + 8), r2b);
            lasx::store_f32x8(c2.as_mut_ptr().add(j + 16), r2c);
            lasx::store_f32x8(c2.as_mut_ptr().add(j + 24), r2d);
            lasx::store_f32x8(c3.as_mut_ptr().add(j), r3a);
            lasx::store_f32x8(c3.as_mut_ptr().add(j + 8), r3b);
            lasx::store_f32x8(c3.as_mut_ptr().add(j + 16), r3c);
            lasx::store_f32x8(c3.as_mut_ptr().add(j + 24), r3d);
        }
        j += 32;
    }
    // 余下列交给单行尾处理
    row_tail_f32(a0, c0, b, k, n, nb);
    row_tail_f32(a1, c1, b, k, n, nb);
    row_tail_f32(a2, c2, b, k, n, nb);
    row_tail_f32(a3, c3, b, k, n, nb);
}

/// 单行主循环：32 列一块（4 个累加器）。
#[inline]
fn cols32_f32(a_row: &[f32], c_row: &mut [f32], b: &[f32], n: usize) {
    let mut j = 0;
    while j + 32 <= n {
        let (mut acc0, mut acc1, mut acc2, mut acc3) = z4();
        for (p, &a_p) in a_row.iter().enumerate() {
            let va = lasx::splat_f32(a_p);
            let base = p * n + j;
            let b0 = unsafe { lasx::load_f32x8(b.as_ptr().add(base)) };
            let b1 = unsafe { lasx::load_f32x8(b.as_ptr().add(base + 8)) };
            let b2 = unsafe { lasx::load_f32x8(b.as_ptr().add(base + 16)) };
            let b3 = unsafe { lasx::load_f32x8(b.as_ptr().add(base + 24)) };
            unsafe {
                acc0 = lasx_xvfmadd_s(b0, va, acc0);
                acc1 = lasx_xvfmadd_s(b1, va, acc1);
                acc2 = lasx_xvfmadd_s(b2, va, acc2);
                acc3 = lasx_xvfmadd_s(b3, va, acc3);
            }
        }
        unsafe {
            lasx::store_f32x8(c_row.as_mut_ptr().add(j), acc0);
            lasx::store_f32x8(c_row.as_mut_ptr().add(j + 8), acc1);
            lasx::store_f32x8(c_row.as_mut_ptr().add(j + 16), acc2);
            lasx::store_f32x8(c_row.as_mut_ptr().add(j + 24), acc3);
        }
        j += 32;
    }
}

/// 单行的列尾：`[j0, n)`，先 8 列块再标量。
#[inline]
fn row_tail_f32(a_row: &[f32], c_row: &mut [f32], b: &[f32], k: usize, n: usize, j0: usize) {
    let mut j = j0;
    while j + 8 <= n {
        let mut acc = lasx::zero_f32x8();
        for (p, &a_p) in a_row.iter().enumerate() {
            let va = lasx::splat_f32(a_p);
            let vb = unsafe { lasx::load_f32x8(b.as_ptr().add(p * n + j)) };
            acc = unsafe { lasx_xvfmadd_s(vb, va, acc) };
        }
        unsafe { lasx::store_f32x8(c_row.as_mut_ptr().add(j), acc) };
        j += 8;
    }
    for jj in j..n {
        let mut s = 0f32;
        for p in 0..k {
            s += a_row[p] * b[p * n + jj];
        }
        c_row[jj] = s;
    }
}

/// 4 个零累加器。
#[inline]
fn z4() -> (lasx::F32x8, lasx::F32x8, lasx::F32x8, lasx::F32x8) {
    (
        lasx::zero_f32x8(),
        lasx::zero_f32x8(),
        lasx::zero_f32x8(),
        lasx::zero_f32x8(),
    )
}

/// 数值回归测试：对照独立参考实现。
#[cfg(test)]
mod tests {
    use crate::ffi::matmul::lasx_matmul;
    use crate::ops::testutil::{reference, rel, Lcg};

    #[test]
    fn test_matmul_f32_matches_reference() {
        let shapes = [
            (0usize, 4usize, 3usize),
            (1, 1, 1),
            (1, 0, 5),
            (1, 8, 1),
            (3, 5, 7),
            (4, 4, 32),
            (4, 4, 33),
            (5, 7, 40),
            (8, 8, 8),
            (16, 33, 17),
            (32, 32, 32),
            (33, 65, 31),
            (64, 64, 64),
            (128, 128, 128),
        ];
        for &(m, k, n) in &shapes {
            let mut rng = Lcg(0x5eed_1234 ^ ((m * 31 + k) * 17 + n) as u64);
            let a: Vec<f32> = (0..m * k).map(|_| rng.f64() as f32).collect();
            let b: Vec<f32> = (0..k * n).map(|_| rng.f64() as f32).collect();
            let mut c = vec![f32::NAN; m * n];
            lasx_matmul(
                m as i32,
                k as i32,
                n as i32,
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
            );
            let want = reference(m, k, n, &a, &b);
            for idx in 0..m * n {
                // f32 沿 k 累加，k≤128 时相对误差应在 1e-4 内
                assert!(
                    rel(c[idx] as f64, want[idx]) < 1e-4,
                    "{m}×{k}×{n} idx={idx}: got {} want {}",
                    c[idx],
                    want[idx]
                );
            }
        }
    }
}
