//! **阈值校准 + 1t/auto 对比**（§7.28 实测入口；**分进程**跑：`OnceLock` 全局池每进程只能一种配置）。
//!
//! 用法：
//! ```text
//! LRS_THREADS=1   cargo run --release --example matmul_auto_threshold -- 256 320 384 512
//! LRS_THREADS=auto cargo run --release --example matmul_auto_threshold -- 256 320 384 512
//! ```
//! `LRS_THREADS=1` ⇒ `pool::init_global(1)`（逃生门）；`auto` ⇒ 不调用（`WorkerPool::auto()` = 物理核）。
//!
//! 输出：每档 `ms 中位数 + GFLOPS`（同进程内多轮取中位；**跨配置必须分进程**，不可同进程交替）。

use std::time::Instant;

fn main() {
    let mode = std::env::var("LRS_THREADS").unwrap_or_else(|_| "auto".into());
    if mode == "1" {
        let ok = lasx_rs::pool::init_global(1);
        println!("init_global(1) => {ok}");
    } else {
        println!("threads = auto（未调用 init_global）");
    }
    let sizes: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|s| s.parse().ok())
        .collect();
    for size in sizes {
        let (m, k, n) = (size, size, size);
        let a: Vec<f32> = (0..m * k).map(|i| (i % 97) as f32 * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 89) as f32 * 0.01).collect();
        let mut out = vec![0.0f32; m * n];
        // 预热一次（避免首调用建池/分配影响）
        lasx_rs::api::matmul_into(m, k, n, &a, &b, &mut out).expect("matmul_into");
        let mut times = Vec::new();
        for _ in 0..5 {
            let t0 = Instant::now();
            lasx_rs::api::matmul_into(m, k, n, &a, &b, &mut out).expect("matmul_into");
            times.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        times.remove(0); // 丢首轮
        times.sort_by(f64::total_cmp);
        let med = times[times.len() / 2];
        let gflops = 2.0 * (m as f64) * (k as f64) * (n as f64) / (med / 1e3) / 1e9;
        let work = (m as u128) * (k as u128) * (n as u128);
        println!("size={size} work={work} med={med:.3} ms gflops={gflops:.2} (mode={mode})");
    }
    println!(
        "loadavg = {}",
        std::fs::read_to_string("/proc/loadavg")
            .unwrap_or_default()
            .trim()
    );
}
