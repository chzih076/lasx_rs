//! f16 GEMV 的单线程 / 多线程 A/B：按参数只跑一条路径并打印中位数，用于**进程级交替**
//! （同进程交替会被彼此的缓存足迹与原语差异干扰；跨进程交替只受机器负载漂移影响，跑多轮取中位）。
//!
//! `cargo run --release --example gemv_pool_ab -- <m> <k> <serial|pool> [线程数]`
//!
//! 例（3 轮交替，`docs/dev.md` §6.4 的口径）：
//!
//! ```text
//! for i in 1 2 3; do
//!   cargo run --release --example gemv_pool_ab -- 4096 4096 serial
//!   cargo run --release --example gemv_pool_ab -- 4096 4096 pool 12
//! done
//! ```
//!
//! 口径与 CLI 的 `timeit` 一致：样品内重复到 ~25 ms，5 个样品取中位；
//! 吞吐按**权重字节数**算（`m·k·2` + 输入 + 输出），与 CLI 的 `f16` 套件同一算式。

use std::hint::black_box;
use std::time::Instant;

use lasx_rs::aligned::AlignedVec;

fn main() {
    let arg: Vec<String> = std::env::args().collect();
    let m: usize = arg[1].parse().unwrap();
    let k: usize = arg[2].parse().unwrap();
    let path = arg.get(3).cloned().unwrap_or_else(|| "serial".into());
    let threads: usize = arg.get(4).map(|s| s.parse().unwrap()).unwrap_or(12);

    // f16 权重：0x3c00 = 1.0，逐元素 +1/64（**只有位型影响速度**，值本身无关）
    let a: Vec<u16> = (0..m * k).map(|i| 0x3c00 + (i as u16 % 17)).collect();
    let x = AlignedVec::<f32>::fill_with(k, |i| (i as f32).mul_add(0.01, 0.5));
    let mut y = AlignedVec::<f32>::new(m);

    // 池只建一次（与真实调用一致）
    let pool = lasx_rs::pool::WorkerPool::new(threads);

    let mut run = || match path.as_str() {
        "serial" => {
            lasx_rs::lasx_gemv_f16(a.as_ptr(), x.as_ptr(), y.as_mut_ptr(), m as i32, k as i32);
            let _ = black_box(y[0]);
        }
        "pool" => {
            lasx_rs::parallel::gemv_f16(&pool, m, k, &a, x.as_slice(), y.as_mut_slice()).unwrap();
            let _ = black_box(y[0]);
        }
        other => panic!("未知路径 {other}（用 serial|pool）"),
    };

    run();
    let est = {
        let t = Instant::now();
        run();
        t.elapsed().as_secs_f64().max(1e-9)
    };
    let reps = ((0.025 / est) as usize).clamp(1, 100_000);
    let mut ts = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..reps {
            run();
        }
        ts.push(t.elapsed().as_secs_f64() / reps as f64);
    }
    ts.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let t = ts[ts.len() / 2];
    let bytes = (m * k * 2 + k * 4 + m * 4) as f64;
    println!(
        "{:.3} ms  {:.2} GB/s  (m={m} k={k} {path} threads={threads} reps={reps})",
        t * 1e3,
        bytes / t / 1e9
    );
}
