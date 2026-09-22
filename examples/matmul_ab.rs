//! 单形状 A/B：按参数跑一条路径，打印中位数。用于**进程级交替**测量
//! （同进程交替会被彼此的缓存足迹干扰，跨进程交替只受机器负载漂移影响，取多轮即可）。
//!
//! `cargo run --release --example matmul_ab -- <m> <k> <n> <stream|packed|cols|packed64|stream64|pool> [线程数]`
//!
//! `pool` 走 `parallel::matmul_f32`（常驻池按行块切分），用于核对多核口径——
//! 例如 `matmul_ab 1024 1024 1024 pool 24`。

use std::time::Instant;

use lasx_rs::aligned::AlignedVec;

fn main() {
    let arg: Vec<String> = std::env::args().collect();
    let m: usize = arg[1].parse().unwrap();
    let k: usize = arg[2].parse().unwrap();
    let n: usize = arg[3].parse().unwrap();
    let path = arg.get(4).cloned().unwrap_or_else(|| "stream".into());

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
    let flop = 2.0 * m as f64 * k as f64 * n as f64;
    // f64 侧（`packed64` / `stream64`）另开一份缓冲
    let a64 = AlignedVec::<f64>::fill_with(m * k, |_| next() as f64);
    let b64 = AlignedVec::<f64>::fill_with(k * n, |_| next() as f64);
    let mut c64 = AlignedVec::<f64>::new(m * n);

    // 池只建一次（与真实调用一致），供 `pool` 路径复用
    let threads: usize = arg.get(5).map(|s| s.parse().unwrap()).unwrap_or(12);
    // `pool` 路径的调度策略（第 6 个参数）：auto/chunk/rowblock/blocked/dynamic
    let strategy = arg.get(6).cloned().unwrap_or_else(|| "auto".into());
    let pick = || match strategy.as_str() {
        "auto" => lasx_rs::pool::pick_rows(m, threads, 4),
        "chunk" => lasx_rs::pool::Pick::Chunk,
        "rowblock" => lasx_rs::pool::Pick::RowBlock,
        "blocked" => lasx_rs::pool::Pick::Blocked {
            block_rows: (m.div_ceil(threads) / 4).max(4),
        },
        "dynamic" => lasx_rs::pool::Pick::Dynamic {
            block_rows: (m.div_ceil(threads) / 4).max(4),
        },
        other => panic!("未知策略 {other}"),
    };
    let mut pool = lasx_rs::pool::WorkerPool::new(threads);

    let mut run = || match path.as_str() {
        "pool" => {
            lasx_rs::parallel::matmul_f32_with_pick(&mut pool, m, k, n, &mut a, &b, &mut c, pick())
        }
        "packed" => lasx_rs::ops_bench_packed(m, k, n, &a, &b, &mut c),
        "stream" => lasx_rs::lasx_matmul(
            m as i32,
            k as i32,
            n as i32,
            a.as_ptr(),
            b.as_ptr(),
            c.as_mut_ptr(),
        ),
        "cols" => lasx_rs::ops_bench_cols_f32(m, k, n, &a, &b, &mut c),
        "packed64" => lasx_rs::ops_bench_packed_f64(m, k, n, &a64, &b64, &mut c64),
        "stream64" => lasx_rs::ops_bench_stream_f64(m, k, n, &a64, &b64, &mut c64),
        other => panic!("未知路径 {other}"),
    };
    run();
    let mut ts = Vec::new();
    for _ in 0..7 {
        let t = Instant::now();
        run();
        ts.push(t.elapsed().as_secs_f64());
    }
    ts.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let t = ts[ts.len() / 2];
    println!("{:.2} {:.1}", t * 1e3, flop / t / 1e9);
}
