//! **P0-3：行归约/归一化自动并行的 A/B**（单线程池 vs 自动池，**同进程**对照）。
//!
//! 为什么能同进程：自动分派包装在 `src/ops/*` 里，但**切分层** `parallel::*` 接收显式
//! `&WorkerPool` ⇒ 这里用 `WorkerPool::new(1)` 与 `WorkerPool::auto()` 两个池直接对照，
//! 不受 `pool::global()` 单例限制（GEMM 阈值实验当初只能分进程，就是因为用全局池）。
//!
//! 口径：同进程交替、6 对、丢首对、取中位；先等价类（两个池的结果必须 `to_bits` 相等 —— 切分
//! 不改变任何元素的累加次序）；打印 `threads`/`ms`/`1t/auto`/`loadavg`。判据：**≥10% 才采纳**
//! （`1t/auto ≥ 1.10` 记为可采纳；并发收益若 <10% 就如实记 BelowThreshold）。
//!
//! 用法：`cargo run --release --features dsl-backend --example row_par_ab`

use std::hint::black_box;
use std::time::{Duration, Instant};

use lasx_rs::parallel;
use lasx_rs::pool::WorkerPool;

const ROUNDS: usize = 6;

fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "<unreadable>".to_string())
}
fn timeit(f: &mut impl FnMut()) -> Duration {
    let t = Instant::now();
    f();
    t.elapsed()
}
fn median(mut v: Vec<Duration>) -> Duration {
    v.sort_unstable();
    v[v.len() / 2]
}
fn ab(mut a: impl FnMut(), mut b: impl FnMut()) -> (Duration, Duration) {
    let (mut ta, mut tb) = (Vec::new(), Vec::new());
    for round in 0..ROUNDS {
        if round % 2 == 0 {
            ta.push(timeit(&mut a));
            tb.push(timeit(&mut b));
        } else {
            tb.push(timeit(&mut b));
            ta.push(timeit(&mut a));
        }
    }
    (median(ta[1..].to_vec()), median(tb[1..].to_vec()))
}
fn sample(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 37 % 101) as f32) * 0.1 - 6.07)
        .collect()
}
fn bitwise(a: &[f32], b: &[f32]) -> bool {
    a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}
fn row(name: &str, elems: usize, a: Duration, b: Duration) {
    let (ams, bms) = (a.as_secs_f64() * 1e3, b.as_secs_f64() * 1e3);
    let ratio = ams / bms;
    println!(
        "| {name} | {elems} | {ams:.3} | {bms:.3} | **{ratio:.3}×** | {} |",
        if ratio >= 1.10 {
            "采纳候选"
        } else {
            "BelowThreshold"
        }
    );
}

fn main() {
    let p1 = WorkerPool::new(1);
    let pn = WorkerPool::auto();
    println!("P0-3 行归约/归一化并行 A/B（1t vs auto；{ROUNDS} 对丢首对取中位）");
    println!("threads: 1t={} auto={}", p1.threads(), pn.threads());
    println!("loadavg(开始) = {}", loadavg());
    println!("| 算子 | elems | 1t ms | auto ms | 1t/auto | 判定 |");
    println!("|---|---|---|---|---|---|");

    for (rows, cols) in [
        (16usize, 512usize), // 8K：小规模门槛下界探针
        (64, 512),           // 32K
        (128, 128),          // 16K
        (1024, 768),
        (4096, 1024),
        (8192, 2048),  // 16.7M：列归约阈值探针
        (16384, 2048), // 33.5M
        (16384, 4096),
    ] {
        let n = rows * cols;
        let x = sample(n);
        let w = sample(cols.max(1));
        let b = sample(cols.max(1));
        let scale = 1.0f32;
        let eps = 1e-5f32;
        let (mut o1, mut on) = (vec![0.0f32; n], vec![0.0f32; n]);
        let (mut c1, mut cn) = (vec![0.0f32; cols], vec![0.0f32; cols]);
        let (mut q1, mut qn) = (vec![0.0f32; cols], vec![0.0f32; cols]);

        // 列和
        parallel::reduce_sum_cols(&p1, &x, rows, cols, &mut c1);
        parallel::reduce_sum_cols(&pn, &x, rows, cols, &mut cn);
        let same = bitwise(&c1, &cn);
        println!(
            "  [reduce_sum_cols {rows}×{cols}] 等价类：**{}**",
            if same { "逐位" } else { "不等（BUG）" }
        );
        assert!(same, "列和并行不逐位");
        let (t1, tn) = ab(
            || {
                parallel::reduce_sum_cols(&p1, &x, rows, cols, &mut c1);
                black_box(&c1);
            },
            || {
                parallel::reduce_sum_cols(&pn, &x, rows, cols, &mut cn);
                black_box(&cn);
            },
        );
        row("reduce_sum_cols", n, t1, tn);

        // 带权列和
        parallel::reduce_weighted_sum_cols(&p1, &x, &w[..rows.min(cols)], rows, cols, &mut q1);
        parallel::reduce_weighted_sum_cols(&pn, &x, &w[..rows.min(cols)], rows, cols, &mut qn);
        assert!(bitwise(&q1, &qn), "带权列和并行不逐位");
        let (t1, tn) = ab(
            || {
                parallel::reduce_weighted_sum_cols(
                    &p1,
                    &x,
                    &w[..rows.min(cols)],
                    rows,
                    cols,
                    &mut q1,
                );
                black_box(&q1);
            },
            || {
                parallel::reduce_weighted_sum_cols(
                    &pn,
                    &x,
                    &w[..rows.min(cols)],
                    rows,
                    cols,
                    &mut qn,
                );
                black_box(&qn);
            },
        );
        row("reduce_weighted_sum_cols", n, t1, tn);

        // 行 softmax
        parallel::softmax_rows(&p1, &x, &[], scale, rows, cols, &mut o1);
        parallel::softmax_rows(&pn, &x, &[], scale, rows, cols, &mut on);
        assert!(bitwise(&o1, &on), "softmax 并行不逐位");
        let (t1, tn) = ab(
            || {
                parallel::softmax_rows(&p1, &x, &[], scale, rows, cols, &mut o1);
                black_box(&o1);
            },
            || {
                parallel::softmax_rows(&pn, &x, &[], scale, rows, cols, &mut on);
                black_box(&on);
            },
        );
        row("softmax_rows", n, t1, tn);

        // 行 RMSNorm
        parallel::rms_norm(&p1, &x, &w, eps, rows, cols, &mut o1);
        parallel::rms_norm(&pn, &x, &w, eps, rows, cols, &mut on);
        assert!(bitwise(&o1, &on), "rms 并行不逐位");
        let (t1, tn) = ab(
            || {
                parallel::rms_norm(&p1, &x, &w, eps, rows, cols, &mut o1);
                black_box(&o1);
            },
            || {
                parallel::rms_norm(&pn, &x, &w, eps, rows, cols, &mut on);
                black_box(&on);
            },
        );
        row("rms_norm", n, t1, tn);

        // 行 LayerNorm
        parallel::layer_norm(&p1, &x, &w, &b, eps, rows, cols, &mut o1);
        parallel::layer_norm(&pn, &x, &w, &b, eps, rows, cols, &mut on);
        assert!(bitwise(&o1, &on), "layer 并行不逐位");
        let (t1, tn) = ab(
            || {
                parallel::layer_norm(&p1, &x, &w, &b, eps, rows, cols, &mut o1);
                black_box(&o1);
            },
            || {
                parallel::layer_norm(&pn, &x, &w, &b, eps, rows, cols, &mut on);
                black_box(&on);
            },
        );
        row("layer_norm", n, t1, tn);
    }

    println!("loadavg(结束) = {}", loadavg());
    println!(
        "判据：1t/auto ≥1.10 才采纳（≥10%）；<1.10 如实记 BelowThreshold，并保留 1t 回退阈值。"
    );
}
