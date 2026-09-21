//! 矩阵乘的算法效率曲线：单线程 GFLOP/s 与**建模的访存量**随规模怎么变。
//!
//! 运行：`cargo run --release --example matmul_scaling`
//!
//! 现在的内核是 i-k-j + 4 行 × 32 列微块：对每个 4 行块都要把整个 B 流一遍，于是
//!
//!   B 的访存字节数 = ceil(m/4) × k×n×4   ← 规模一大就爆
//!   A 的访存字节数 = ceil(n/32) × m×k×4
//!
//! 这个例子把"建模的访存速率"打出来：如果它在某个规模后稳定在几十 GB/s，
//! 说明已经撞到内存带宽，而不是算力——那才是要削的地方。

use std::hint::black_box;
use std::time::Instant;

use lasx_rs::aligned::AlignedVec;

fn bench(mut f: impl FnMut(), reps: usize) -> f64 {
    let mut ts = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        f();
        ts.push(t.elapsed().as_secs_f64());
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ts[ts.len() / 2]
}

fn main() {
    println!(
        "{:<10}{:>11}{:>10}{:>12}{:>13}",
        "规模", "时间", "GFLOP/s", "建模访存MB", "访存GB/s"
    );
    for &s in &[128usize, 256, 384, 512, 768, 1024] {
        let (m, k, n) = (s, s, s);
        let mut rng = 0x1234_5678u64;
        let mut next = move || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let mut a = AlignedVec::<f32>::fill_with(m * k, |_| next());
        let b = AlignedVec::<f32>::fill_with(k * n, |_| next());
        let mut c = AlignedVec::<f32>::new(m * n);

        let reps = if s >= 768 { 3 } else { 5 };
        let t = bench(
            || {
                lasx_rs::lasx_matmul(
                    m as i32,
                    k as i32,
                    n as i32,
                    a.as_ptr(),
                    b.as_ptr(),
                    c.as_mut_ptr(),
                );
            },
            reps,
        );
        let flop = 2.0 * m as f64 * k as f64 * n as f64;
        let b_bytes = (m.div_ceil(4) * k * n * 4) as f64;
        let a_bytes = (n.div_ceil(32) * m * k * 4) as f64;
        let c_bytes = (m * n * 4) as f64;
        let traffic = a_bytes + b_bytes + c_bytes;
        println!(
            "{s:>4}³{:>6}{:>11}{:>10.1}{:>12.1}{:>13.1}",
            "",
            format!("{:.1} ms", t * 1e3),
            flop / t / 1e9,
            traffic / 1e6,
            traffic / t / 1e9
        );
        let _ = black_box(&mut a);
    }
}
