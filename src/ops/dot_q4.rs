//! `lasx_dot_q4` —— Q4 量化点积：每字节 2 个无符号 nibble，scale 每 32 字节一组。
//!
//! 公式：`dot = Σ_g scale_a[g]·scale_b[g]·Σ_{组内 64 个 nibble} (qa·qb)`
//! （组内整数点积精确，再乘该组的 scale 积、累加进 f64）。
//!
//! v2：原先每 32 字节把 16×i16 部分和 `xvst` 到栈再 **16 次标量 i16 加法**求和，
//! 标量归约占了近一半指令。现在用两条水平加宽把 16×i16 直接压成 4×i64
//! （`xvhaddw_w_h` 相邻两两相加 → 8×i32，`xvhaddw_d_w` 再两两相加 → 4×i64），
//! 每组的标量加法从 16 次降到 4 次。
//!
//! 与 `lasx_dot_i8` 不同，**这里的组内归约无法省掉**：每 32 字节有自己的 scale，
//! 必须在组边界完成"整数点积 → 乘 scale → 进 f64"，不能把多组的整数部分先合并。
//! 好在 nibble 是 0..15 的无符号数，乘积 ≤225、组内每 lane ≤900、总和 ≤14400，
//! i16/i32/i64 全程不溢出（不像 i8 那样有 (−128)² 的边界）。
//!
//! v3：大集（工作集 ≫ L3）下掉到 L3 内的 43%（10.15 → 4.35 G nibble/s），原因是**每轮
//! 只发 2 个 32 字节载入**，DRAM 延迟下在飞请求太少。改成一"批"4 组（128 字节）：8 个
//! 载入先全部发出、4 组的组内部分和互相独立地算出来，然后**仍按组序**做水平归约、
//! 乘 scale、累加进 f64——浮点求和次序与 v2 完全一致，**结果逐位不变**。
//!
//! 受控 A/B（stash 单文件重建）：L2/L3 档 +7~25%（2^22：873.6–955.9 µs → 762.5–822.7 µs），
//! DRAM 档持平（±4%，落在噪声内）。
//!
//! v4（**已否决**）：曾把组内归约的 `xvst` + 4 次标量载入换成 4 次 `xvpickve2gr_d` 直接取
//! lane，想消掉 store→load 转发的十几周期气泡。A/B 否决了它：2^25 中位 21.40 ms vs
//! 20.0 ms（**慢 7%**），2^22 持平 ⇒ 转发延迟不是瓶颈。
//!
//! v5（**已否决**）：把一批从 4 组加到 8 组（256 字节、16 个载入），想让 DRAM 延迟下有
//! 更多在飞请求。三回合交替 A/B：2^22 中位 799.9 µs vs 797.3 µs（持平），
//! 2^25 中位 **19.96 ms vs 14.87 ms（慢 34%）**——16 个 m256i 输入同时活跃的寄存器压力
//! 反而吃掉了 OoO 的空间。⇒ 4 组是本内核的甜点。
//!
//! **瓶颈到底在哪**（实测口径）：2^16（数据全热）11.2 G nibble/s，每 32 字节一组 ⇒
//! 12.6 周期/组，而 `group_sums` + 归约约 17 条向量指令、整组约 25 µops ⇒ IPC ≈ 2，
//! 已贴着发射上限；2^22（L3 内）流量 10.5 GB/s、2^25 掉到 4.5 GB/s。纯字节双流读锚点
//! （同规模 u64 异或折叠）在 2^25 是 9.9 GB/s，**不比 f64 双流读（10.7–12.0）低**，
//! 所以"字节粒度吃亏"这个猜想也被否掉：大集差距是"每字节指令数 × DRAM 停留"的乘积，
//! 不是访存粒度。要再快只能减少每组的浮点归约次数，但组内必须先做精确整数点积、
//! 再乘该组 scale——这会改变浮点结合次序（**未授权**），故本轮止步于此。
//!
//! **LASX-only**：没有降级分支，无 LASX 的 CPU 上会执行 LASX 指令（见手册 Caveats）。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（策略见 `docs/dev.md` §17）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX/LSX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]

use std::arch::loongarch64::*;

/// 一组（32 字节 = 64 个 nibble）的组内部分和：16 个 i16 lane，每 lane 两个 nibble 乘积之和。
#[inline(always)]
unsafe fn group_sums(va: m256i, vb: m256i) -> m256i {
    let alo = lasx_xvandi_b(va, 0x0f);
    let ahi = lasx_xvsrli_b(va, 4);
    let blo = lasx_xvandi_b(vb, 0x0f);
    let bhi = lasx_xvsrli_b(vb, 4);
    let s0 = lasx_xvadd_h(lasx_xvmulwev_h_b(alo, blo), lasx_xvmulwod_h_b(alo, blo));
    let s1 = lasx_xvadd_h(lasx_xvmulwev_h_b(ahi, bhi), lasx_xvmulwod_h_b(ahi, bhi));
    lasx_xvadd_h(s0, s1)
}

/// 16×i16 → 8×i32 → 4×i64（相邻两两相加），再 4 次标量加得到组内点积（精确整数）。
#[inline(always)]
unsafe fn horizontal_i64(sall: m256i, tmp: &mut [i64; 4]) -> i64 {
    let w = lasx_xvhaddw_w_h(sall, sall);
    let wide = lasx_xvhaddw_d_w(w, w);
    lasx_xvst(wide, tmp.as_mut_ptr() as *mut i8, 0);
    tmp[0] + tmp[1] + tmp[2] + tmp[3]
}

/// LASX-only：本内核没有 `has_lasx()` 降级分支，直接执行 256 位实现。
/// 在无 LASX 的 CPU 上会执行 LASX 指令（见手册 Caveats）。
#[inline]
pub(crate) fn dot_q4(qa: &[u8], sa: &[f32], qb: &[u8], sb: &[f32]) -> f64 {
    let n = qa.len();
    let mut acc = 0f64;
    let mut b = 0;
    let (pa, pb) = (qa.as_ptr(), qb.as_ptr());
    let mut tmp = [0i64; 4];
    // 一批 4 组（128 字节）：先发 8 个载入，再算 4 个独立的组内部分和
    while b + 128 <= n {
        // SAFETY: 循环条件保证 [b, b+128) 在 qa/qb 内。
        let (va0, vb0, va1, vb1, va2, vb2, va3, vb3) = unsafe {
            (
                lasx_xvld(pa.add(b) as *const i8, 0),
                lasx_xvld(pb.add(b) as *const i8, 0),
                lasx_xvld(pa.add(b + 32) as *const i8, 0),
                lasx_xvld(pb.add(b + 32) as *const i8, 0),
                lasx_xvld(pa.add(b + 64) as *const i8, 0),
                lasx_xvld(pb.add(b + 64) as *const i8, 0),
                lasx_xvld(pa.add(b + 96) as *const i8, 0),
                lasx_xvld(pb.add(b + 96) as *const i8, 0),
            )
        };
        let s = unsafe {
            (
                group_sums(va0, vb0),
                group_sums(va1, vb1),
                group_sums(va2, vb2),
                group_sums(va3, vb3),
            )
        };
        // 归约与 f64 累加**严格按组序**（0,1,2,3），与旧的逐组循环等价
        for (k, sall) in [s.0, s.1, s.2, s.3].into_iter().enumerate() {
            let g = b / 32 + k;
            let dot = unsafe { horizontal_i64(sall, &mut tmp) };
            acc += ((sa[g] * sb[g]) as f64) * (dot as f64);
        }
        b += 128;
    }
    // 余下不足 128 字节：逐组处理（与 v2 相同）
    while b + 32 <= n {
        let (sall, sc) = unsafe {
            let va = lasx_xvld(pa.add(b) as *const i8, 0);
            let vb = lasx_xvld(pb.add(b) as *const i8, 0);
            (group_sums(va, vb), sa[b / 32] * sb[b / 32])
        };
        let dot = unsafe { horizontal_i64(sall, &mut tmp) };
        acc += (sc as f64) * (dot as f64);
        b += 32;
    }
    for j in b..n {
        let la = (qa[j] & 0x0f) as i64;
        let ha = ((qa[j] >> 4) & 0x0f) as i64;
        let lb = (qb[j] & 0x0f) as i64;
        let hb = ((qb[j] >> 4) & 0x0f) as i64;
        let sc = sa[j / 32] * sb[j / 32];
        acc += (sc as f64) * ((la * lb + ha * hb) as f64);
    }
    acc
}

#[cfg(test)]
mod tests {
    use crate::ffi::quant::lasx_dot_q4;
    use crate::ops::testutil::Lcg;

    /// f64 精确参考：逐 nibble 算整数点积，再逐组乘 scale 累加。
    fn reference(qa: &[u8], sa: &[f32], qb: &[u8], sb: &[f32]) -> f64 {
        let n = qa.len();
        let mut acc = 0f64;
        let mut b = 0;
        while b + 32 <= n {
            let mut dot = 0i64;
            for j in b..b + 32 {
                let la = (qa[j] & 0x0f) as i64;
                let ha = ((qa[j] >> 4) & 0x0f) as i64;
                let lb = (qb[j] & 0x0f) as i64;
                let hb = ((qb[j] >> 4) & 0x0f) as i64;
                dot += la * lb + ha * hb;
            }
            acc += ((sa[b / 32] * sb[b / 32]) as f64) * (dot as f64);
            b += 32;
        }
        for j in b..n {
            let la = (qa[j] & 0x0f) as i64;
            let ha = ((qa[j] >> 4) & 0x0f) as i64;
            let lb = (qb[j] & 0x0f) as i64;
            let hb = ((qb[j] >> 4) & 0x0f) as i64;
            acc += ((sa[j / 32] * sb[j / 32]) as f64) * ((la * lb + ha * hb) as f64);
        }
        acc
    }

    /// 全量 0xFF（nibble 全 15，乘积取最大）覆盖溢出边界；含长度非 32 倍数的尾巴。
    #[test]
    fn test_dot_q4_matches_exact_reference() {
        let mut rng = Lcg(0x0404_0404);
        for &n in &[0usize, 1, 31, 32, 33, 63, 64, 65, 96, 127, 128, 1024, 4096] {
            let groups = n.div_ceil(32);
            for case in 0..3 {
                let (qa, qb): (Vec<u8>, Vec<u8>) = match case {
                    0 => (
                        (0..n).map(|_| (rng.f64().abs() * 255.0) as u8).collect(),
                        (0..n).map(|_| (rng.f64().abs() * 255.0) as u8).collect(),
                    ),
                    // 全 0xFF：每组 64 个 nibble 全为 15 → 组内点积取最大值
                    1 => (vec![0xFFu8; n], vec![0xFFu8; n]),
                    _ => (vec![0u8; n], vec![0xFFu8; n]),
                };
                let sa: Vec<f32> = (0..groups).map(|i| 0.01 + i as f32 * 1e-5).collect();
                let sb: Vec<f32> = (0..groups).map(|i| 0.02 - i as f32 * 1e-6).collect();
                let want = reference(&qa, &sa, &qb, &sb);
                let got = lasx_dot_q4(qa.as_ptr(), sa.as_ptr(), qb.as_ptr(), sb.as_ptr(), n as i32);
                let err = (got - want).abs() / want.abs().max(1.0);
                assert!(
                    err < 1e-12,
                    "n={n} case={case}: got {got} want {want} rel={err}"
                );
            }
        }
    }
}
