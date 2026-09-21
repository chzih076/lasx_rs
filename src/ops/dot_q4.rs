//! `lasx_dot_q4` —— Q4 量化点积：每字节 2 个无符号 nibble，scale 每 32 字节一组。
//!
//!
use std::arch::loongarch64::*;

/// LASX-only：本内核没有 `has_lasx()` 降级分支，直接执行 256 位实现。
/// 在无 LASX 的 CPU 上会执行 LASX 指令（见手册 Caveats）。
#[inline]
pub(crate) fn dot_q4(qa: &[u8], sa: &[f32], qb: &[u8], sb: &[f32]) -> f64 {
    let n = qa.len();
    let mut acc = 0f64;
    let mut b = 0;
    while b + 32 <= n {
        let va = unsafe { lasx_xvld(qa.as_ptr().add(b) as *const i8, 0) };
        let vb = unsafe { lasx_xvld(qb.as_ptr().add(b) as *const i8, 0) };
        let alo = unsafe { lasx_xvandi_b(va, 0x0f) };
        let ahi = unsafe { lasx_xvsrli_b(va, 4) };
        let blo = unsafe { lasx_xvandi_b(vb, 0x0f) };
        let bhi = unsafe { lasx_xvsrli_b(vb, 4) };
        let s0 = unsafe { lasx_xvadd_h(lasx_xvmulwev_h_b(alo, blo), lasx_xvmulwod_h_b(alo, blo)) };
        let s1 = unsafe { lasx_xvadd_h(lasx_xvmulwev_h_b(ahi, bhi), lasx_xvmulwod_h_b(ahi, bhi)) };
        let sall = unsafe { lasx_xvadd_h(s0, s1) };
        let mut tmp = [0i16; 16];
        unsafe { lasx_xvst(sall, tmp.as_mut_ptr() as *mut i8, 0) };
        let mut dot: i64 = 0;
        for &v in &tmp {
            dot += v as i64;
        }
        let sc = sa[b / 32] * sb[b / 32];
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
