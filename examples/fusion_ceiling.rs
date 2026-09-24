//! `alpha`/`beta` 融合到底值不值：先量**收益上限**，再决定要不要写新内核。
//!
//! 融合能省的只有对 `C` 的一次写与一次读——`y` 的读/写在融合版里照样要（`beta != 0` 时
//! 内核必须读 `y`）。所以：
//!
//! ```text
//! 二遍法：matmul 写 C          + elementwise(读 C、读 y、写 y)
//! 融合版：matmul 直接读写 y（读 y 一次）
//! 上限   = 二遍法的 elementwise 减去"融合内核里多读的那一遍 y" ≈ 那一遍的 2/3
//! ```
//!
//! 也就是说：**二遍法里那一遍 elementwise 占 matmul 的比例，就是融合的收益上限**。
//! 这里不需要新内核就能量出来（`lasx_matmul` + 一个自动向量化的三流循环）。
//!
//! `cargo run --release --example fusion_ceiling`

use lasx_rs::aligned::AlignedVec;
use lasx_rs::lasx_matmul;
use std::hint::black_box;
use std::time::{Duration, Instant};

/// 自适应重复 + 5 取样取中位数（与 `lasx_bench` 同口径）。
fn timeit<F: FnMut()>(mut f: F) -> Duration {
    f();
    f();
    let t = Instant::now();
    f();
    let single = t.elapsed().max(Duration::from_nanos(1));
    let reps =
        (Duration::from_millis(25).as_nanos() / single.as_nanos()).clamp(1, 10_000_000) as u32;
    let mut v = Vec::with_capacity(5);
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        v.push(t.elapsed() / reps);
    }
    v.sort_unstable();
    v[2]
}

fn us(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

/// 二遍法的第二遍：`y = alpha·C + beta·y`（自动向量化；实际内核会比这更快一点，
/// 所以这里量到的是**偏保守**的上限）。
fn scale_add(alpha: f32, c: &[f32], beta: f32, y: &mut [f32]) {
    for i in 0..y.len() {
        y[i] = alpha * c[i] + beta * y[i];
    }
}

fn main() {
    println!("### `alpha`/`beta` 融合的收益上限（单线程，`C` 复用缓冲）");
    println!();
    println!(
        "| 形状 | matmul | 第二遍 `y=a·C+b·y` | 第二遍/matmul | 融合上限(≈2/3) | 第二遍带宽 |"
    );
    println!("|---|---|---|---|---|---|");

    for &(m, k, n) in &[
        (512usize, 512usize, 512usize),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (100, 64, 19147), // 极宽 n、小 k：内存受限那一档
        (256, 64, 4096),
        (64, 4096, 64), // k 很大、m/n 很小：算力受限那一档
    ] {
        let a = AlignedVec::<f32>::fill_with(m * k, |i| (i % 23) as f32 * 0.25 - 2.0);
        let b = AlignedVec::<f32>::fill_with(k * n, |i| (i % 19) as f32 * 0.5 - 4.0);
        let mut c = AlignedVec::<f32>::new(m * n);
        let mut y = AlignedVec::<f32>::fill_with(m * n, |i| (i % 7) as f32);
        let (alpha, beta) = (1.5f32, 0.5f32);

        let t_mm = timeit(|| {
            lasx_matmul(
                m as i32,
                k as i32,
                n as i32,
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
            );
            black_box(c[0]);
        });
        let t_elem = timeit(|| {
            scale_add(alpha, c.as_slice(), beta, y.as_mut_slice());
            black_box(y[0]);
        });

        // 一遍 elementwise 的字节数：读 C + 读 y + 写 y
        let bytes = 3.0 * (m * n) as f64 * 4.0;
        let gbs = bytes / t_elem.as_secs_f64() / 1e9;
        let ratio = t_elem.as_secs_f64() / t_mm.as_secs_f64();
        println!(
            "| f32 {m}×{k}×{n} | {} | {} | {:.1}% | {:.1}% | {gbs:.1} GB/s |",
            fmt_us(t_mm),
            fmt_us(t_elem),
            ratio * 100.0,
            ratio * 200.0 / 3.0,
        );
    }
    println!();
    println!("参考锚点：本机单流只读 ~7.5–8.9 GB/s（`docs/dev.md` §7.1）；");
    println!("「融合上限」= 第二遍的 2/3 —— 融合省掉对 C 的读写，但 `y` 仍要读一遍。");
}

fn fmt_us(d: Duration) -> String {
    let v = us(d);
    if v < 1000.0 {
        format!("{v:.1} µs")
    } else {
        format!("{:.2} ms", v / 1000.0)
    }
}
