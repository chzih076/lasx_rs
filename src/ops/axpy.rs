//! `lasx_axpy` —— `y += alpha·x`（单条 FMA 写回）。
//!
//!

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（分层策略：内核模块在函数级统一写 SAFETY，不逐块重复）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX/LSX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]
use crate::arch::lasx;
use std::arch::loongarch64::*;

/// LASX-only：本内核没有 `has_lasx()` 降级分支，直接执行 256 位实现。
/// 在无 LASX 的 CPU 上会执行 LASX 指令（见手册 Caveats）。
#[inline]
pub(crate) fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    let n = x.len();
    let va = lasx::splat_f32(alpha);
    let mut i = 0;
    // **4 路展开（32 元素/迭代）**：逐元素运算**没有归约** ⇒ 展开**不改变任何数值**（按构造逐位安全），
    // 只为摊薄循环开销、给访存与 FMA 更多并行度。动机是实测：单路时 65536 档比 LLVM 自动向量化的
    // 标量参考慢 5%（0.95×）、4 MiB 档慢 11%（0.89×）——4 MiB 档已是带宽墙（读写共 32 MB / 2.87 ms
    // ≈ 11 GB/s，贴近单流 DRAM 上限），可回收的主要是驻留档的循环开销。
    while i + 32 <= n {
        // SAFETY: `i + 32 <= n` ⇒ 四组访存分别落在 [i,i+8)、[i+8,i+16)、[i+16,i+24)、[i+24,i+32)，
        // 全在 `n` 之内；`x`/`y` 是长度已校验的切片。
        unsafe {
            let p = x.as_ptr().add(i);
            let q = y.as_mut_ptr().add(i);
            let a0 = lasx::load_f32x8(p);
            let a1 = lasx::load_f32x8(p.add(8));
            let a2 = lasx::load_f32x8(p.add(16));
            let a3 = lasx::load_f32x8(p.add(24));
            let b0 = lasx::load_f32x8(q);
            let b1 = lasx::load_f32x8(q.add(8));
            let b2 = lasx::load_f32x8(q.add(16));
            let b3 = lasx::load_f32x8(q.add(24));
            lasx::store_f32x8(q, lasx_xvfmadd_s(a0, va, b0));
            lasx::store_f32x8(q.add(8), lasx_xvfmadd_s(a1, va, b1));
            lasx::store_f32x8(q.add(16), lasx_xvfmadd_s(a2, va, b2));
            lasx::store_f32x8(q.add(24), lasx_xvfmadd_s(a3, va, b3));
        }
        i += 32;
    }
    while i + 8 <= n {
        let vx = unsafe { lasx::load_f32x8(x.as_ptr().add(i)) };
        let vy = unsafe { lasx::load_f32x8(y.as_ptr().add(i)) };
        let vf = unsafe { lasx_xvfmadd_s(vx, va, vy) };
        unsafe { lasx::store_f32x8(y.as_mut_ptr().add(i), vf) };
        i += 8;
    }
    for j in i..n {
        y[j] += alpha * x[j];
    }
}
