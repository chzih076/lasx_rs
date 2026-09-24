//! 多核矩阵乘示例：`C[m×n] = A[m×k] · B[k×n]`，按行块铺到常驻线程池。
//!
//! 运行：`cargo run --release --example matmul_pooled`
//!
//! 这个例子同时回答两个问题：
//!
//! 1. **怎么用**：`lasx_rs::parallel::matmul_f32` 一个调用即可（它内部是池的按行块
//!    切分接口 —— `A` 每行 `k` 个、`C` 每行 `n` 个，切在同一批行上）；
//! 2. **能拿到多少**：扫线程数看加速比曲线，并逐位对照单线程结果。
//!
//! 结论（本机 12 物理核 /24 逻辑核，后台负载 load ≈ 5.7）：
//!
//! - **12 线程 ≈ 5.4×，16 线程 ≈ 8.5×，24 线程退化到 ≈ 3.8×**（12/16 两行在安静时段
//!   稳到 ±1%；偶发抢占会打乱任何一行，故示例取 9 次的中位数）；
//! - 瓶颈不是派活，而是 **B 的流量**：`lasx_matmul` 一次算 4 行 × 32 列，每个 4 行块
//!   都要重读整个 B，于是 B 流量 = `(m/4) × k×n × 4 字节`（256³ = 16 MB/次调用），
//!   多核一起跑就把共享缓存的带宽压满了。吃满带宽后吞吐只跟**同时活跃的块数**成正比；
//! - 所以 12 线程那行反而慢：块大小按行粒度 4 取整后只有 11 块在跑（10×24 + 16 行），
//!   而 16 线程恰好是 16 块 × 16 行——既没有退化的尾块，也没有空转的 worker。
//!   这也解释了为什么块大小要是行精度 4 的倍数（`parallel::matmul_f32` 已内置）。

use std::time::Instant;

use lasx_rs::aligned::AlignedVec;
use lasx_rs::pool::WorkerPool;

/// 取 `reps` 次里的中位数（机器有后台负载，单次不可信）。
fn bench(mut f: impl FnMut(), reps: usize) -> f64 {
    let mut ts: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        f();
        ts.push(t.elapsed().as_secs_f64());
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ts[ts.len() / 2]
}

fn main() {
    let (m, k, n) = (256usize, 256usize, 256usize);
    let flop = 2.0 * m as f64 * k as f64 * n as f64;

    let mut rng = 0x1234_5678u64;
    let mut next = move || {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let mut a = AlignedVec::<f32>::fill_with(m * k, |_| next());
    let b = AlignedVec::<f32>::fill_with(k * n, |_| next());
    let mut want = AlignedVec::<f32>::new(m * n);
    lasx_rs::lasx_matmul(
        m as i32,
        k as i32,
        n as i32,
        a.as_ptr(),
        b.as_ptr(),
        want.as_mut_ptr(),
    );

    let t1 = bench(
        || {
            lasx_rs::lasx_matmul(
                m as i32,
                k as i32,
                n as i32,
                a.as_ptr(),
                b.as_ptr(),
                want.as_mut_ptr(),
            );
        },
        9,
    );
    println!(
        "{m}×{k}×{n} 单线程：{:.1} µs（{:.1} GFLOP/s）",
        t1 * 1e6,
        flop / t1 / 1e9
    );
    println!();
    println!("| 线程数 | 时间 | 加速比 | GFLOP/s |");
    println!("|---|---|---|---|");
    println!(
        "| 1 | {:.1} µs | 1.00× | {:.1} |",
        t1 * 1e6,
        flop / t1 / 1e9
    );

    for &threads in &[2usize, 4, 8, 12, 16, 24] {
        let pool = WorkerPool::new(threads);
        let mut got = AlignedVec::<f32>::new(m * n);
        let t = bench(
            || {
                lasx_rs::parallel::matmul_f32(
                    &pool,
                    m,
                    k,
                    n,
                    a.as_mut_slice(),
                    b.as_slice(),
                    got.as_mut_slice(),
                )
                .unwrap();
            },
            9,
        );
        assert_eq!(
            want.as_slice(),
            got.as_slice(),
            "{threads} 线程结果与单线程不一致"
        );
        println!(
            "| {threads} | {:.1} µs | **{:.2}×** | {:.1} |",
            t * 1e6,
            t1 / t,
            flop / t / 1e9
        );
    }
    println!();
    println!("所有线程数的结果都与单线程逐位一致（切行不改变任何输出元素的计算过程）。");
    println!("提示：本机后台负载 load ≈ 5.7，绝对值会飘（尤其线程数多时），");
    println!("      安静时段 12 线程 ≈ 5.4×、16 线程 ≈ 8.5×、24 线程 ≈ 3.8× 是可重复的。");
}
