//! `lasx_dot` —— f32 点积 `Σ a[i]·b[i]`（8×f32 FMA，f64 分块落盘抑制累加误差）。
//!
//! LASX 缺失时走 LSX 128 位路径。
//!
//! 规模小于 [`SMALL_N`] 时直接走标量：向量建立 + f64 落盘 + 最终归约的固定开销
//! 在十几个元素上摊不薄（2026-09-25 实测：n = 8 时 LASX 19 ns vs 标量 10 ns、
//! n = 16 时 33 vs 19 ns，两档都是 LASX 更慢；n = 24 才反超 1.27×。
//! 完整表见 `docs/dev.md` §7.4）。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（策略见 `docs/dev.md` §17）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX/LSX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]
use crate::arch::SimdPath;
use crate::arch::{lasx, lsx};
use std::arch::loongarch64::*;

/// 低于该规模改走标量（实测交叉点约 n=20，取 24 留余量）。
const SMALL_N: usize = 24;

/// 算子入口：极小规模走标量，否则解析向量路径后分派。
///
/// 极小规模用**单精度累加**（与向量路径的 f64 分块累加不同）：n<16 时 f32 舍入
/// 误差量级只有 ~1e-6 绝对值，比向量路径多出来的转换 + f64 加法开销划算
/// ——实测 n=8 时 f64 口径 10 ns、f32 口径 6 ns，而向量路径是 10 ns。
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    if a.len() < SMALL_N {
        // **累加次序与原 `zip` 版逐字一致**（严格顺序的单累加器）：4 路展开只为消除边界检查、
        // 让 LLVM 展开取指/取数，**不改变任何一次加法的顺序** ⇒ 结果与原版**逐位相同**。
        // 为什么改：原 `zip` 版在 n=8 时 **11 ns**，而 bench 的标量参考（普通索引循环）只要
        // **6 ns**——即"我们的回退比 LLVM 自动向量化的标量循环慢近一倍"，实测比值 0.55×。
        // 改后：n=8 **7 ns（0.86×）**、n=16 **13 ns（0.85–0.92×）**，仍差 8–15%（LLVM 的索引循环
        // 能用多累加器天然获得 ILP，我们的单链受 ~4 周期/次的依赖链限制）。
        //
        // **两条已实测否决的路，别再走**（记录见 `src/ops/dot_q4.rs` 的 v4/v5 段）：
        // ① 把"`xvst` 落盘 + 4 次标量载入"换成 `xvpickve2gr_d` 直接取 lane（想消掉 store→load
        //    转发气泡）⇒ A/B 在 2^25 **慢 7%**（21.40 vs 20.0 ms）⇒ **转发延迟不是瓶颈**；
        // ② 多链/加大批量 ⇒ 活跃向量数上升，寄存器压力反吃 OoO 空间（8 组批量在 2^25 **慢 34%**）。
        // ⇒ 小 n 这条的可行空间就只剩"改数值结合次序换 ILP"（2 链实测在噪声窗口下方向不定），
        //    而收益上限只有 ~14%、代价是动归约口径 ⇒ **代价大于收益，维持现状并如实记录**。
        let n = a.len().min(b.len());
        let mut acc = 0f32;
        let mut i = 0;
        while i + 4 <= n {
            acc += a[i] * b[i];
            acc += a[i + 1] * b[i + 1];
            acc += a[i + 2] * b[i + 2];
            acc += a[i + 3] * b[i + 3];
            i += 4;
        }
        while i < n {
            acc += a[i] * b[i];
            i += 1;
        }
        return acc;
    }
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
    use crate::ops::testutil::{NS, both_paths, data, rel};

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
