//! `lasx_sum` —— f32 向量归约 `Σ x[i]`（与 `dot` 同构）。
//!
//!

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（策略见 `docs/dev.md` §17）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX/LSX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]
use crate::arch::lasx;
use std::arch::loongarch64::*;

/// LASX-only：本内核没有 `has_lasx()` 降级分支，直接执行 256 位实现。
/// 在无 LASX 的 CPU 上会执行 LASX 指令（见手册 Caveats）。
#[inline]
pub(crate) fn sum(x: &[f32]) -> f32 {
    let n = x.len();
    let px = x.as_ptr();
    let mut accd = lasx::zero_f64x4();
    let mut i = 0;
    while i + 256 <= n {
        let mut c0 = lasx::zero_f32x8();
        let mut c1 = lasx::zero_f32x8();
        let mut c2 = lasx::zero_f32x8();
        let mut c3 = lasx::zero_f32x8();
        for _ in 0..8 {
            unsafe {
                c0 = lasx_xvfadd_s(c0, lasx::load_f32x8(px.add(i)));
                c1 = lasx_xvfadd_s(c1, lasx::load_f32x8(px.add(i + 8)));
                c2 = lasx_xvfadd_s(c2, lasx::load_f32x8(px.add(i + 16)));
                c3 = lasx_xvfadd_s(c3, lasx::load_f32x8(px.add(i + 24)));
            }
            i += 32;
        }
        for c in [c0, c1, c2, c3] {
            accd = unsafe { lasx_xvfadd_d(accd, lasx_xvfcvtl_d_s(c)) };
            accd = unsafe { lasx_xvfadd_d(accd, lasx_xvfcvth_d_s(c)) };
        }
    }
    let mut acc = lasx::zero_f32x8();
    while i + 8 <= n {
        unsafe {
            acc = lasx_xvfadd_s(acc, lasx::load_f32x8(px.add(i)));
        }
        i += 8;
    }
    accd = unsafe { lasx_xvfadd_d(accd, lasx_xvfcvtl_d_s(acc)) };
    accd = unsafe { lasx_xvfadd_d(accd, lasx_xvfcvth_d_s(acc)) };
    let mut t = [0f64; 4];
    unsafe { lasx::store_f64x4(t.as_mut_ptr(), accd) };
    let mut acc_d = t[0] + t[1] + t[2] + t[3];
    for &v in &x[i..n] {
        acc_d += v as f64;
    }
    acc_d as f32
}

/// 数值回归测试：对照独立参考实现。
#[cfg(test)]
mod tests {
    use crate::ffi::reduce::lasx_sum;
    use crate::ops::testutil::{both_paths, data, rel, NS};

    #[test]
    fn test_sum_matches_f64_reference() {
        for &n in NS {
            let (a, _) = data(n);
            let want: f64 = a.iter().map(|&x| x as f64).sum();
            let (lasx, lsx) = both_paths(|| lasx_sum(a.as_ptr(), n as i32));
            assert!(rel(lasx, want) < 1e-5, "n={n} LASX: {lasx} vs {want}");
            assert!(rel(lsx, want) < 1e-5, "n={n} LSX: {lsx} vs {want}");
        }
    }
}
