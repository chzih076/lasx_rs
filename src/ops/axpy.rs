//! `lasx_axpy` —— `y += alpha·x`（**两次舍入**：先 `t = alpha·x + 0.0`、再 `y + t`）。
//!
//! **逐位契约（W3 修正）**：契约默认 `axpy_inplace_f32` 是 `t = alpha*xi + 0.0; yi += t`
//! （两次 IEEE 舍入）；旧 LASX 内核用单条 FMA（`xvfmadd`，一次舍入）⇒ 与组合版差 **1 ulp**
//! （Lead 实测 `max_abs 2.44e-4`）。本版改成 `mul → add(+0) → add` 的向量序列，与契约默认**逐位**
//! （含 `±0` 语义：`-0.0 + 0.0` 的符号处理与标量路径一致）；仍是 1 趟内存（带宽墙不变）。
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
    let zero = lasx::zero_f32x8();
    let mut i = 0;
    // **4 路展开（32 元素/迭代）**：逐元素运算**没有归约** ⇒ 展开**不改变任何数值**（按构造逐位安全），
    // 只为摊薄循环开销、给访存与算术更多并行度。动机是实测：单路时 65536 档比 LLVM 自动向量化的
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
            // 两次舍入：`t = alpha*x + 0.0`（第二项加法与标量路径同语义）、`y + t`。
            let t0 = lasx_xvfadd_s(lasx_xvfmul_s(a0, va), zero);
            let t1 = lasx_xvfadd_s(lasx_xvfmul_s(a1, va), zero);
            let t2 = lasx_xvfadd_s(lasx_xvfmul_s(a2, va), zero);
            let t3 = lasx_xvfadd_s(lasx_xvfmul_s(a3, va), zero);
            lasx::store_f32x8(q, lasx_xvfadd_s(b0, t0));
            lasx::store_f32x8(q.add(8), lasx_xvfadd_s(b1, t1));
            lasx::store_f32x8(q.add(16), lasx_xvfadd_s(b2, t2));
            lasx::store_f32x8(q.add(24), lasx_xvfadd_s(b3, t3));
        }
        i += 32;
    }
    while i + 8 <= n {
        let vx = unsafe { lasx::load_f32x8(x.as_ptr().add(i)) };
        let vy = unsafe { lasx::load_f32x8(y.as_ptr().add(i)) };
        let t = unsafe { lasx_xvfadd_s(lasx_xvfmul_s(vx, va), zero) };
        let vf = unsafe { lasx_xvfadd_s(vy, t) };
        unsafe { lasx::store_f32x8(y.as_mut_ptr().add(i), vf) };
        i += 8;
    }
    for j in i..n {
        let t = alpha * x[j] + 0.0;
        y[j] += t;
    }
}
