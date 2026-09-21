//! 专测「小 m、大 n」形状：`C[100×19147] = A[100×448] · B[448×19147]`。
//!
//! 运行：`cargo run --release --example matmul_wide`
//!
//! 这个形状的性质与方阵完全不同：
//! - `A` 只有 179 KB，能整块待在 L2 里；
//! - `B` 是 34.3 MB，比 L3（32 MB）还大，**流一遍**的代价就是它的下限；
//! - 算术强度 ≈ 41 FLOP/字节，远高于本机平衡点（93 GFLOP/s ÷ 25 GB/s ≈ 3.7），
//!   所以只要 B 只读一遍，它就是**算力受限**，不是带宽受限。
//!
//! 原来的循环是"每个 4 行块把 B 扫一遍"：m=100 ⇒ B 被读 25 遍（858 MB），而且每次只读
//! 128 B 就跳 `n×4` = 76 KB，实测有效带宽只有 ~5 GB/s。**改成"列块在外 + 打包 B 面板"**
//! （Goto/BLIS 的 packing，见 perf-report §19）之后 B 只流一遍且顺序读，
//! 单线程 165.9 → ~40 ms。
//!
//! 多核**按列切**也试过（每线程只读自己那段 B）：成对实测只有 0.88–1.20×（8 线程反而
//! 更慢），不值得为它加一条裸指针路径，故未落地——见 `docs/perf-report.md` §18。

use std::hint::black_box;
use std::time::Instant;

use lasx_rs::aligned::AlignedVec;
use lasx_rs::pool::WorkerPool;

const M: usize = 100;
const K: usize = 448;
const N: usize = 19147;

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
    let flop = 2.0 * M as f64 * K as f64 * N as f64;
    let mut rng = 0x1234_5678u64;
    let mut next = move || {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let mut a = AlignedVec::<f32>::fill_with(M * K, |_| next());
    let b = AlignedVec::<f32>::fill_with(K * N, |_| next());
    let mut c = AlignedVec::<f32>::new(M * N);

    let bytes = (M * K + K * N + M * N) as f64 * 4.0;
    println!("形状 {M}×{K} × {K}×{N}");
    println!(
        "  A {:.0} KB（L2 可驻留）  B {:.1} MB（> L3 32 MB）  C {:.1} MB",
        (M * K * 4) as f64 / 1024.0,
        (K * N * 4) as f64 / 1e6,
        (M * N * 4) as f64 / 1e6
    );
    println!(
        "  工作 {flop:.3} GFLOP，理论最少搬运 {:.1} MB，算术强度 {:.1} FLOP/字节",
        bytes / 1e6,
        flop / bytes
    );
    println!();

    let t1 = bench(
        || {
            lasx_rs::lasx_matmul(
                M as i32,
                K as i32,
                N as i32,
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
            );
        },
        5,
    );
    println!(
        "单线程（打包 B 面板）: {:.1} ms  → {:.1} GFLOP/s",
        t1 * 1e3,
        flop / t1 / 1e9
    );

    println!();
    println!("| 线程 | 多核（按行切，每线程跑同一个内核） | 加速比 |");
    println!("|---|---|---|");
    for &th in &[4usize, 8, 12, 16] {
        let mut pool = WorkerPool::new(th);
        let t = bench(
            || {
                lasx_rs::parallel::matmul_f32(
                    &mut pool,
                    M,
                    K,
                    N,
                    a.as_mut_slice(),
                    b.as_slice(),
                    c.as_mut_slice(),
                );
            },
            5,
        );
        println!("| {th} | {:.1} ms | {:.2}× |", t * 1e3, t1 / t);
    }
    let _ = black_box(&mut a);
}
