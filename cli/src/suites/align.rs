//! 对齐影响：LASX 是 256 位访问，起始地址不落在 32 字节边界时，32 字节的
//! load/store 会跨 64 字节缓存行，代价可观（实测 `lasx_dot` f32 慢约 38%）。
//!
//! 本套件用**同一份数据**、只改起始地址（对齐 vs 偏移 1 个元素），把这一项单独量化。
//! 表格里的其余套件用的是普通 `Vec<T>`（保守：模拟未对齐的调用方缓冲区），
//! 因此 f32 内核的实测值低于其对齐后的能力。

use crate::data::{AlignedBuf, Lcg};
use crate::timing::{fmt_t, timeit};
use lasx_rs::*;
use std::hint::black_box;

/// 打印对齐 vs 未对齐的对比表。
pub fn align() {
    println!();
    println!("## 对齐影响：同一份数据，只改起始地址");
    println!();
    println!("| 内核 | n | 32B/64B 对齐 | 未对齐（偏移 1 元素） | 对齐收益 |");
    println!("|---|---|---|---|---|");

    for &n in &[1024usize, 2048, 4096, 8192, 1 << 16] {
        let mut rng = Lcg::new(0xa116 ^ n as u64);
        let a = AlignedBuf::fill_with(n, || rng.f32());
        let b = AlignedBuf::fill_with(n, || rng.f32());
        let al = a.as_slice();
        let bl = b.as_slice();
        let len = (n - 1) as i32;

        let aligned = timeit(|| {
            let _ = black_box(lasx_dot(al.as_ptr(), bl.as_ptr(), len));
        });
        let unaligned = timeit(|| {
            let _ = black_box(lasx_dot(al[1..].as_ptr(), bl[1..].as_ptr(), len));
        });
        println!(
            "| `lasx_dot` f32 | {n} | {} | {} | {:.2}× |",
            fmt_t(aligned),
            fmt_t(unaligned),
            unaligned.as_secs_f64() / aligned.as_secs_f64()
        );
    }

    for &n in &[4096usize, 1 << 16] {
        let mut rng = Lcg::new(0xb227 ^ n as u64);
        let a = AlignedBuf::fill_with(n, || rng.f64());
        let b = AlignedBuf::fill_with(n, || rng.f64());
        let al = a.as_slice();
        let bl = b.as_slice();
        let len = (n - 1) as i32;

        let aligned = timeit(|| {
            let _ = black_box(lasx_dot_f64(al.as_ptr(), bl.as_ptr(), len));
        });
        let unaligned = timeit(|| {
            let _ = black_box(lasx_dot_f64(al[1..].as_ptr(), bl[1..].as_ptr(), len));
        });
        println!(
            "| `lasx_dot_f64` | {n} | {} | {} | {:.2}× |",
            fmt_t(aligned),
            fmt_t(unaligned),
            unaligned.as_secs_f64() / aligned.as_secs_f64()
        );
    }
}
