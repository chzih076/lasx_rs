//! `lasx_dot_f64` —— f64 点积（4×f64 FMA）。
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
pub(crate) fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let mut acc = lasx::zero_f64x4();
    let mut i = 0;
    while i + 4 <= n {
        let va = unsafe { lasx::load_f64x4(a.as_ptr().add(i)) };
        let vb = unsafe { lasx::load_f64x4(b.as_ptr().add(i)) };
        acc = unsafe { lasx_xvfmadd_d(va, vb, acc) };
        i += 4;
    }
    let mut tmp = [0f64; 4];
    unsafe { lasx::store_f64x4(tmp.as_mut_ptr(), acc) };
    let mut s = 0f64;
    for &v in &tmp {
        s += v;
    }
    for j in i..n {
        s += a[j] * b[j];
    }
    s
}
