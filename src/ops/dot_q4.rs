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
//! **LASX-only**：没有降级分支，无 LASX 的 CPU 上会执行 LASX 指令（见手册 Caveats）。

use std::arch::loongarch64::*;

/// LASX-only：本内核没有 `has_lasx()` 降级分支，直接执行 256 位实现。
/// 在无 LASX 的 CPU 上会执行 LASX 指令（见手册 Caveats）。
#[inline]
pub(crate) fn dot_q4(qa: &[u8], sa: &[f32], qb: &[u8], sb: &[f32]) -> f64 {
    let n = qa.len();
    let mut acc = 0f64;
    let mut b = 0;
    let (pa, pb) = (qa.as_ptr(), qb.as_ptr());
    let mut tmp = [0i64; 4];
    while b + 32 <= n {
        // SAFETY: 循环条件保证 [b, b+32) 在 qa/qb 内。
        let (sall, sc) = unsafe {
            let va = lasx_xvld(pa.add(b) as *const i8, 0);
            let vb = lasx_xvld(pb.add(b) as *const i8, 0);
            let alo = lasx_xvandi_b(va, 0x0f);
            let ahi = lasx_xvsrli_b(va, 4);
            let blo = lasx_xvandi_b(vb, 0x0f);
            let bhi = lasx_xvsrli_b(vb, 4);
            // 组内 16 个 i16 lane，每 lane = 两个 nibble 乘积之和
            let s0 = lasx_xvadd_h(lasx_xvmulwev_h_b(alo, blo), lasx_xvmulwod_h_b(alo, blo));
            let s1 = lasx_xvadd_h(lasx_xvmulwev_h_b(ahi, bhi), lasx_xvmulwod_h_b(ahi, bhi));
            (lasx_xvadd_h(s0, s1), sa[b / 32] * sb[b / 32])
        };
        // 16×i16 → 8×i32 → 4×i64（都是"相邻两两相加"），再 4 次标量加
        let wide = unsafe {
            let w = lasx_xvhaddw_w_h(sall, sall);
            lasx_xvhaddw_d_w(w, w)
        };
        unsafe { lasx_xvst(wide, tmp.as_mut_ptr() as *mut i8, 0) };
        let dot = tmp[0] + tmp[1] + tmp[2] + tmp[3];
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
