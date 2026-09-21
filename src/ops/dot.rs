//! `lasx_dot` —— f32 点积 `Σ a[i]·b[i]`（8×f32 FMA，f64 分块落盘抑制累加误差）。
//!
//! LASX 缺失时走 LSX 128 位路径。
use crate::arch::SimdPath;
use crate::arch::{lasx, lsx};
use std::arch::loongarch64::*;

/// 算子入口：解析当前线程的向量路径后分派。
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    match SimdPath::detect() {
        SimdPath::Lasx => dot_lasx(a, b),
        SimdPath::Lsx => dot_lsx(a, b),
    }
}

/// LASX 256 位实现。
#[inline]
fn dot_lasx(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());

    // LASX 256 位路径：4 条独立累加链，每 256 元素落盘一次（向量化）
    let mut accd = lasx::zero_f64x4();
    let mut i = 0;
    while i + 256 <= n {
        let mut c0 = lasx::zero_f32x8();
        let mut c1 = lasx::zero_f32x8();
        let mut c2 = lasx::zero_f32x8();
        let mut c3 = lasx::zero_f32x8();
        for _ in 0..8 {
            unsafe {
                c0 = lasx_xvfmadd_s(lasx::load_f32x8(pa.add(i)), lasx::load_f32x8(pb.add(i)), c0);
                c1 = lasx_xvfmadd_s(
                    lasx::load_f32x8(pa.add(i + 8)),
                    lasx::load_f32x8(pb.add(i + 8)),
                    c1,
                );
                c2 = lasx_xvfmadd_s(
                    lasx::load_f32x8(pa.add(i + 16)),
                    lasx::load_f32x8(pb.add(i + 16)),
                    c2,
                );
                c3 = lasx_xvfmadd_s(
                    lasx::load_f32x8(pa.add(i + 24)),
                    lasx::load_f32x8(pb.add(i + 24)),
                    c3,
                );
            }
            i += 32;
        }
        // 8 个 f32 分量 → 2 个 f64 向量 → 4 条累加指令（旧版 17 条）
        for c in [c0, c1, c2, c3] {
            accd = unsafe { lasx_xvfadd_d(accd, lasx_xvfcvtl_d_s(c)) };
            accd = unsafe { lasx_xvfadd_d(accd, lasx_xvfcvth_d_s(c)) };
        }
    }
    // 8 元素块尾
    let mut acc = lasx::zero_f32x8();
    while i + 8 <= n {
        unsafe {
            acc = lasx_xvfmadd_s(
                lasx::load_f32x8(pa.add(i)),
                lasx::load_f32x8(pb.add(i)),
                acc,
            );
        }
        i += 8;
    }
    accd = unsafe { lasx_xvfadd_d(accd, lasx_xvfcvtl_d_s(acc)) };
    accd = unsafe { lasx_xvfadd_d(accd, lasx_xvfcvth_d_s(acc)) };

    let mut t = [0f64; 4];
    unsafe { lasx::store_f64x4(t.as_mut_ptr(), accd) };
    let mut acc_d = t[0] + t[1] + t[2] + t[3];
    for j in i..n {
        acc_d += (a[j] * b[j]) as f64;
    }
    acc_d as f32
}

/// LSX 128 位实现（每链 16 个向量，每通道 f32 累加深度 16）。
#[inline]
fn dot_lsx(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    // LSX 128 位路径：同样 4 条独立累加链（每链 16 个向量 = 每通道深度 16，与 v1 一致）
    let mut acc_d = 0f64;
    let mut i = 0;
    while i + 256 <= n {
        let mut c0 = lsx::zero_f32x4();
        let mut c1 = lsx::zero_f32x4();
        let mut c2 = lsx::zero_f32x4();
        let mut c3 = lsx::zero_f32x4();
        for _ in 0..16 {
            unsafe {
                c0 = lsx_vfmadd_s(lsx::load_f32x4(pa.add(i)), lsx::load_f32x4(pb.add(i)), c0);
                c1 = lsx_vfmadd_s(
                    lsx::load_f32x4(pa.add(i + 4)),
                    lsx::load_f32x4(pb.add(i + 4)),
                    c1,
                );
                c2 = lsx_vfmadd_s(
                    lsx::load_f32x4(pa.add(i + 8)),
                    lsx::load_f32x4(pb.add(i + 8)),
                    c2,
                );
                c3 = lsx_vfmadd_s(
                    lsx::load_f32x4(pa.add(i + 12)),
                    lsx::load_f32x4(pb.add(i + 12)),
                    c3,
                );
            }
            i += 16;
        }
        let mut tmp = [0f32; 4];
        for c in [c0, c1, c2, c3] {
            unsafe { lsx::store_f32x4(tmp.as_mut_ptr(), c) };
            for &v in &tmp {
                acc_d += v as f64;
            }
        }
    }
    let mut acc = lsx::zero_f32x4();
    while i + 4 <= n {
        unsafe {
            acc = lsx_vfmadd_s(lsx::load_f32x4(pa.add(i)), lsx::load_f32x4(pb.add(i)), acc);
        }
        i += 4;
    }
    let mut tmp = [0f32; 4];
    unsafe { lsx::store_f32x4(tmp.as_mut_ptr(), acc) };
    for &v in &tmp {
        acc_d += v as f64;
    }
    for j in i..n {
        acc_d += (a[j] * b[j]) as f64;
    }
    acc_d as f32
}

/// 数值回归测试：对照独立参考实现。
#[cfg(test)]
mod tests {
    use crate::ffi::reduce::lasx_dot;
    use crate::ops::testutil::{both_paths, data, rel, NS};

    #[test]
    fn test_dot_matches_f64_reference() {
        for &n in NS {
            let (a, b) = data(n);
            let want: f64 = a.iter().zip(&b).map(|(&x, &y)| x as f64 * y as f64).sum();
            let (lasx, lsx) = both_paths(|| lasx_dot(a.as_ptr(), b.as_ptr(), n as i32));
            assert!(rel(lasx, want) < 1e-5, "n={n} LASX: {lasx} vs {want}");
            assert!(rel(lsx, want) < 1e-5, "n={n} LSX: {lsx} vs {want}");
        }
    }
}
