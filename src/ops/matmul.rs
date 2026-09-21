//! `lasx_matmul` —— f32 矩阵乘 `C[m×n] = A[m×k]·B[k×n]`（行主序，列方向向量化）。
//!
//!
use crate::arch::lasx;
use std::arch::loongarch64::*;

/// LASX-only：本内核没有 `has_lasx()` 降级分支，直接执行 256 位实现。
/// 在无 LASX 的 CPU 上会执行 LASX 指令（见手册 Caveats）。
#[inline]
pub(crate) fn matmul_f32(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let c_row = &mut c[i * n..(i + 1) * n];
        let mut j = 0;
        // 主块：一次 32 列 = 4×8 通道累加器
        while j + 32 <= n {
            let mut acc0 = lasx::zero_f32x8();
            let mut acc1 = lasx::zero_f32x8();
            let mut acc2 = lasx::zero_f32x8();
            let mut acc3 = lasx::zero_f32x8();
            for (p, &a_p) in a_row.iter().enumerate() {
                let va = lasx::splat_f32(a_p);
                let base = p * n + j;
                let b0 = unsafe { lasx::load_f32x8(b.as_ptr().add(base)) };
                let b1 = unsafe { lasx::load_f32x8(b.as_ptr().add(base + 8)) };
                let b2 = unsafe { lasx::load_f32x8(b.as_ptr().add(base + 16)) };
                let b3 = unsafe { lasx::load_f32x8(b.as_ptr().add(base + 24)) };
                acc0 = unsafe { lasx_xvfmadd_s(b0, va, acc0) };
                acc1 = unsafe { lasx_xvfmadd_s(b1, va, acc1) };
                acc2 = unsafe { lasx_xvfmadd_s(b2, va, acc2) };
                acc3 = unsafe { lasx_xvfmadd_s(b3, va, acc3) };
            }
            unsafe {
                lasx::store_f32x8(c_row.as_mut_ptr().add(j), acc0);
                lasx::store_f32x8(c_row.as_mut_ptr().add(j + 8), acc1);
                lasx::store_f32x8(c_row.as_mut_ptr().add(j + 16), acc2);
                lasx::store_f32x8(c_row.as_mut_ptr().add(j + 24), acc3);
            }
            j += 32;
        }

        // 8 列块
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

        // 标量尾（n 不为 8 的倍数时最多 7 列）
        for jj in j..n {
            let mut s = 0f32;
            for p in 0..k {
                s += a_row[p] * b[p * n + jj];
            }
            c_row[jj] = s;
        }
    }
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
