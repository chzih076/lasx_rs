#![feature(stdarch_loongarch)]
//! 微内核探针：把 `tile4x32_chunk` 的循环体放到 **L1 热数据**上，量它自身的 FMA/周期上限。
//!
//! 结论（本机 Loongson-3B6000 / LA664，2.2 GHz，见 `docs/dev.md` §6.3）：
//!
//! | 变体 | 每轮内容 | FMA/周期 | GFLOP/s |
//! |---|---|---|---|
//! | `fma16` | 16 条 `xvfmadd.s`，无访存 | **3.96** | 139.2（f32 峰值） |
//! | `dfma16` | 16 条 `xvfmadd.d`，无访存 | **3.97** | 69.9（f64 峰值） |
//! | `norepl` | 16 FMA + 4 条 B 载入 | 2.66 | 93.8 |
//! | `kernel` | 16 FMA + 4 条 B 载入 + 4 条 A 广播 | **2.00** | 70.3 |
//! | `dkernel` | 同 `kernel`，f64（4 行 × 16 列） | **1.99** | 35.1 |
//! | `cRxC` | R 行 × C 列（R·C/8 个累加器） | 只要含 A 广播就**恒为 2.0** | — |
//!
//! 即：**FP 流水线是 4 条 256 位 FMA/周期（与元素宽度无关，f64 于是正好一半 FLOP）**，
//! 而 **A 标量广播让含访存的内核停在 2.0 FMA/周期**，与行数/列数/累加器个数无关
//! （实测每行每 k 步恰好 2 周期）。所以内层循环没有"再挤一挤"的空间，
//! 提升只能来自缓存侧（见 §21 的 k 分块、§22 的 f64 打包）。
//!
//! 用法：`cargo run --release --example kernel_probe -- <变体> [k] [轮数]`
//! 变体：`fma16`、`norepl`、`kernel`、`c<行>x<列>`（如 `c4x32`、`c2x64`、`c6x16`）。

use std::arch::loongarch64::*;
use std::hint::black_box;
use std::time::Instant;

use lasx_rs::aligned::AlignedVec;
use lasx_rs::arch::lasx::{
    load_f32x8, load_f64x4, splat_f32, splat_f64, store_f32x8, store_f64x4, zero_f32x8, zero_f64x4,
};

/// 访存型内核每样品的内层重复次数（把 L1 热的循环体跑到 ~100 ms 量级）。
const ITERS: usize = 200;
/// 纯 FMA 探针的重复次数：没有访存，一轮只有 ~0.7 µs，必须放大到毫秒级才量得准。
const FMA_ITERS: usize = 5_000_000;

/// 把累加器求和成一个标量，防止优化器把整个循环消掉。
unsafe fn sink(acc: &[m256]) -> f64 {
    let mut tmp = [0f32; 8];
    let mut s = 0f64;
    for v in acc {
        store_f32x8(tmp.as_mut_ptr(), *v);
        s += tmp.iter().map(|&x| x as f64).sum::<f64>();
    }
    black_box(s)
}

/// `dkernel`：与 f64 生产内核同形 —— 4 行 × 16 列，4 条 B 载入 + 4 条 A 广播 + 16 FMA。
#[inline(never)]
unsafe fn probe_dkernel(strip: &[f64], at: &[f64], k: usize) -> f64 {
    let mut acc = [zero_f64x4(); 16];
    for _ in 0..ITERS {
        for p in 0..k {
            let ptr = strip.as_ptr().add(p * 16);
            let b = [
                load_f64x4(ptr),
                load_f64x4(ptr.add(4)),
                load_f64x4(ptr.add(8)),
                load_f64x4(ptr.add(12)),
            ];
            for r in 0..4 {
                let a = splat_f64(*at.get_unchecked(r * k + p));
                for i in 0..4 {
                    acc[r * 4 + i] = lasx_xvfmadd_d(b[i], a, acc[r * 4 + i]);
                }
            }
        }
    }
    let mut tmp = [0f64; 4];
    let mut sink = 0f64;
    for v in acc {
        store_f64x4(tmp.as_mut_ptr(), v);
        sink += tmp.iter().sum::<f64>();
    }
    black_box(sink)
}

/// `fma16`：16 条独立 FMA，无任何访存 —— 量 f32 侧 FP 流水线峰值。
///
/// 注意：累加器必须写成 **16 个独立变量**。写成 `[m256; 16]` + `iter_mut()` 时 LLVM 能把
/// 整个循环折成闭式（实测变成 0.001 ms / 400+ FMA/周期），量出来的是假的。
#[inline(never)]
unsafe fn probe_fma16() -> f64 {
    let s = splat_f32(1.0);
    let c = splat_f32(1e-9);
    let (mut v0, mut v1, mut v2, mut v3) = (c, c, c, c);
    let (mut v4, mut v5, mut v6, mut v7) = (c, c, c, c);
    let (mut v8, mut v9, mut v10, mut v11) = (c, c, c, c);
    let (mut v12, mut v13, mut v14, mut v15) = (c, c, c, c);
    for _ in 0..FMA_ITERS {
        v0 = lasx_xvfmadd_s(s, c, v0);
        v1 = lasx_xvfmadd_s(s, c, v1);
        v2 = lasx_xvfmadd_s(s, c, v2);
        v3 = lasx_xvfmadd_s(s, c, v3);
        v4 = lasx_xvfmadd_s(s, c, v4);
        v5 = lasx_xvfmadd_s(s, c, v5);
        v6 = lasx_xvfmadd_s(s, c, v6);
        v7 = lasx_xvfmadd_s(s, c, v7);
        v8 = lasx_xvfmadd_s(s, c, v8);
        v9 = lasx_xvfmadd_s(s, c, v9);
        v10 = lasx_xvfmadd_s(s, c, v10);
        v11 = lasx_xvfmadd_s(s, c, v11);
        v12 = lasx_xvfmadd_s(s, c, v12);
        v13 = lasx_xvfmadd_s(s, c, v13);
        v14 = lasx_xvfmadd_s(s, c, v14);
        v15 = lasx_xvfmadd_s(s, c, v15);
    }
    sink(&[
        v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
    ])
}

/// `dfma16`：同上，f64（`xvfmadd.d`）—— 量 f64 侧 FP 流水线峰值。
#[inline(never)]
unsafe fn probe_dfma16() -> f64 {
    let s = splat_f64(1.0);
    let c = splat_f64(1e-9);
    let (mut v0, mut v1, mut v2, mut v3) = (c, c, c, c);
    let (mut v4, mut v5, mut v6, mut v7) = (c, c, c, c);
    let (mut v8, mut v9, mut v10, mut v11) = (c, c, c, c);
    let (mut v12, mut v13, mut v14, mut v15) = (c, c, c, c);
    for _ in 0..FMA_ITERS {
        v0 = lasx_xvfmadd_d(s, c, v0);
        v1 = lasx_xvfmadd_d(s, c, v1);
        v2 = lasx_xvfmadd_d(s, c, v2);
        v3 = lasx_xvfmadd_d(s, c, v3);
        v4 = lasx_xvfmadd_d(s, c, v4);
        v5 = lasx_xvfmadd_d(s, c, v5);
        v6 = lasx_xvfmadd_d(s, c, v6);
        v7 = lasx_xvfmadd_d(s, c, v7);
        v8 = lasx_xvfmadd_d(s, c, v8);
        v9 = lasx_xvfmadd_d(s, c, v9);
        v10 = lasx_xvfmadd_d(s, c, v10);
        v11 = lasx_xvfmadd_d(s, c, v11);
        v12 = lasx_xvfmadd_d(s, c, v12);
        v13 = lasx_xvfmadd_d(s, c, v13);
        v14 = lasx_xvfmadd_d(s, c, v14);
        v15 = lasx_xvfmadd_d(s, c, v15);
    }
    let mut tmp = [0f64; 4];
    let mut sum = 0f64;
    for v in [
        v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
    ] {
        store_f64x4(tmp.as_mut_ptr(), v);
        sum += tmp.iter().sum::<f64>();
    }
    black_box(sum)
}

/// `norepl`：16 FMA + 4 条 B 载入（A 广播用常量寄存器顶替）。
#[inline(never)]
unsafe fn probe_norepl(strip: &[f32], k: usize) -> f64 {
    let s = [
        splat_f32(1.0),
        splat_f32(0.5),
        splat_f32(0.25),
        splat_f32(0.125),
    ];
    let mut acc = [zero_f32x8(); 16];
    for _ in 0..ITERS {
        for p in 0..k {
            let ptr = strip.as_ptr().add(p * 32);
            let b = [
                load_f32x8(ptr),
                load_f32x8(ptr.add(8)),
                load_f32x8(ptr.add(16)),
                load_f32x8(ptr.add(24)),
            ];
            for i in 0..4 {
                for r in 0..4 {
                    acc[r * 4 + i] = lasx_xvfmadd_s(b[i], s[r], acc[r * 4 + i]);
                }
            }
        }
    }
    sink(&acc)
}

/// `kernel`：与生产内核同形 —— 4 行 × 32 列，4 条 B 载入 + 4 条 A 广播 + 16 FMA。
#[inline(never)]
unsafe fn probe_kernel(strip: &[f32], at: &[f32], k: usize) -> f64 {
    let mut acc = [zero_f32x8(); 16];
    for _ in 0..ITERS {
        for p in 0..k {
            let ptr = strip.as_ptr().add(p * 32);
            let b = [
                load_f32x8(ptr),
                load_f32x8(ptr.add(8)),
                load_f32x8(ptr.add(16)),
                load_f32x8(ptr.add(24)),
            ];
            for r in 0..4 {
                let a = splat_f32(*at.get_unchecked(r * k + p));
                for i in 0..4 {
                    acc[r * 4 + i] = lasx_xvfmadd_s(b[i], a, acc[r * 4 + i]);
                }
            }
        }
    }
    sink(&acc)
}

/// `c<行>x<列>`：扫描 (R, C) 形状空间（累加器 = R·C/8，必须 ≤ 16 才放得进寄存器）。
#[inline(never)]
unsafe fn probe_rc<const R: usize, const NB: usize>(strip: &[f32], at: &[f32], k: usize) -> f64 {
    let mut acc = [[zero_f32x8(); NB]; R];
    for _ in 0..ITERS {
        for p in 0..k {
            let ptr = strip.as_ptr().add(p * 8 * NB);
            let mut b = [zero_f32x8(); NB];
            for (i, bi) in b.iter_mut().enumerate() {
                *bi = load_f32x8(ptr.add(i * 8));
            }
            for (r, row) in acc.iter_mut().enumerate() {
                let a = splat_f32(*at.get_unchecked(r * k + p));
                for (i, x) in row.iter_mut().enumerate() {
                    *x = lasx_xvfmadd_s(b[i], a, *x);
                }
            }
        }
    }
    let mut tmp = [0f32; 8];
    let mut s = 0f64;
    for row in acc.iter() {
        for v in row.iter() {
            store_f32x8(tmp.as_mut_ptr(), *v);
            s += tmp.iter().map(|&x| x as f64).sum::<f64>();
        }
    }
    black_box(s)
}

fn main() {
    let arg: Vec<String> = std::env::args().collect();
    let variant = arg.get(1).cloned().unwrap_or_else(|| "kernel".into());
    let k: usize = arg.get(2).map(|s| s.parse().unwrap()).unwrap_or(256);
    let rounds: usize = arg.get(3).map(|s| s.parse().unwrap()).unwrap_or(15);

    // k=256 时条带 32 KB，连同 A 一起待在 L1（64 KB）里
    let strip = AlignedVec::<f32>::fill_with(k * 32, |i| (i % 13) as f32 * 0.25);
    let at = AlignedVec::<f32>::fill_with(4 * k, |i| (i % 7) as f32 * 0.125);
    let strip_d = AlignedVec::<f64>::fill_with(k * 16, |i| (i % 13) as f64 * 0.25);
    let at_d = AlignedVec::<f64>::fill_with(4 * k, |i| (i % 7) as f64 * 0.125);

    let run = || -> f64 {
        unsafe {
            match variant.as_str() {
                "fma16" => probe_fma16(),
                "norepl" => probe_norepl(&strip, k),
                "kernel" => probe_kernel(&strip, &at, k),
                "dfma16" => probe_dfma16(),
                "dkernel" => probe_dkernel(&strip_d, &at_d, k),
                "c2x48" => probe_rc::<2, 6>(&strip, &at, k),
                "c2x64" => probe_rc::<2, 8>(&strip, &at, k),
                "c3x40" => probe_rc::<3, 5>(&strip, &at, k),
                "c4x32" => probe_rc::<4, 4>(&strip, &at, k),
                "c4x40" => probe_rc::<4, 5>(&strip, &at, k),
                "c5x32" => probe_rc::<5, 4>(&strip, &at, k),
                "c6x16" => probe_rc::<6, 2>(&strip, &at, k),
                "c8x16" => probe_rc::<8, 2>(&strip, &at, k),
                other => panic!("未知变体 {other}（试试 fma16/norepl/kernel/c4x32）"),
            }
        }
    };

    let per_iter = match variant.as_str() {
        "fma16" | "norepl" | "kernel" | "dfma16" | "dkernel" => 16.0,
        other => {
            let (rs, cs) = other
                .trim_start_matches('c')
                .split_once('x')
                .unwrap_or_else(|| panic!("未知变体 {other}（试试 fma16/norepl/kernel/c4x32）"));
            rs.parse::<f64>().unwrap() * cs.parse::<f64>().unwrap() / 8.0
        }
    };

    run();
    let mut ts = Vec::new();
    for _ in 0..rounds {
        let t = Instant::now();
        run();
        ts.push(t.elapsed().as_secs_f64());
    }

    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let t = ts[ts.len() / 2];
    // 纯 FMA 探针不遍历 k，只按 FMA_ITERS 计
    let pure = variant == "fma16" || variant == "dfma16";
    let fmas = if pure {
        per_iter * FMA_ITERS as f64
    } else {
        per_iter * k as f64 * ITERS as f64
    };
    let flop_per_fma = if variant.starts_with('d') { 8.0 } else { 16.0 };
    println!(
        "{variant:>7} k={k}: {:8.3} ms/轮  {:.2} FMA/周期  {:6.1} GFLOP/s",
        t * 1e3,
        fmas / (t * 2.2e9),
        fmas * flop_per_fma / t / 1e9
    );
}
