//! `lasx_matmul_f64` —— f64 矩阵乘（与 `matmul` 同构）。
//!
//!
use crate::arch::lasx;
use std::arch::loongarch64::*;

/// LASX-only：本内核没有 `has_lasx()` 降级分支，直接执行 256 位实现。
/// 在无 LASX 的 CPU 上会执行 LASX 指令（见手册 Caveats）。
#[inline]
pub(crate) fn matmul_f64(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let c_row = &mut c[i * n..(i + 1) * n];
        let mut j = 0;
        // 主块：一次 16 列 = 4×4 通道累加器
        while j + 16 <= n {
            let mut acc0 = lasx::zero_f64x4();
            let mut acc1 = lasx::zero_f64x4();
            let mut acc2 = lasx::zero_f64x4();
            let mut acc3 = lasx::zero_f64x4();
            for (p, &a_p) in a_row.iter().enumerate() {
                let va = lasx::splat_f64(a_p);
                let base = p * n + j;
                let b0 = unsafe { lasx::load_f64x4(b.as_ptr().add(base)) };
                let b1 = unsafe { lasx::load_f64x4(b.as_ptr().add(base + 4)) };
                let b2 = unsafe { lasx::load_f64x4(b.as_ptr().add(base + 8)) };
                let b3 = unsafe { lasx::load_f64x4(b.as_ptr().add(base + 12)) };
                acc0 = unsafe { lasx_xvfmadd_d(b0, va, acc0) };
                acc1 = unsafe { lasx_xvfmadd_d(b1, va, acc1) };
                acc2 = unsafe { lasx_xvfmadd_d(b2, va, acc2) };
                acc3 = unsafe { lasx_xvfmadd_d(b3, va, acc3) };
            }
            unsafe {
                lasx::store_f64x4(c_row.as_mut_ptr().add(j), acc0);
                lasx::store_f64x4(c_row.as_mut_ptr().add(j + 4), acc1);
                lasx::store_f64x4(c_row.as_mut_ptr().add(j + 8), acc2);
                lasx::store_f64x4(c_row.as_mut_ptr().add(j + 12), acc3);
            }
            j += 16;
        }

        // 4 列块
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

        // 标量尾
        for jj in j..n {
            let mut s = 0f64;
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
