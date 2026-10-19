//! **B1/task-106（M5）A/B：`axpy_inplace_f32` override vs portable 默认**。
//!
//! **线程控制（Lead 口径）**：融合/组合混跑时分不清"SIMD 收益"与"多线程收益"。
//! 本示例**只测纯逐元素原语**（无 auto-parallel），但仍按同一口径在
//! `pool::init_global(1)` 与 `init_global(12)` 下各跑一组，并打印 threads / loadavg / grade。
//!
//! 用法：`cargo run --release --example prims_ab --features dsl-backend -- 1`（或 `12`）。
//! 输出：同进程交替 5 对丢首对中位（portable/LASX 比值；>1 表示 LASX 更快）。

use std::hint::black_box;
use std::time::{Duration, Instant};

use lasx_rs::dsl_backend::LasxBackend;
use simd_dsl::{Backend, ScalarBackend};

const ROUNDS: usize = 5;

fn timeit(f: &mut impl FnMut()) -> Duration {
    let t = Instant::now();
    f();
    t.elapsed()
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort_unstable();
    v[v.len() / 2]
}

fn ab(mut a: impl FnMut(), mut b: impl FnMut()) -> (f64, f64) {
    let (mut av, mut bv) = (Vec::new(), Vec::new());
    for r in 0..ROUNDS {
        let (ta, tb) = if r % 2 == 0 {
            (timeit(&mut a), timeit(&mut b))
        } else {
            let tb = timeit(&mut b);
            let ta = timeit(&mut a);
            (ta, tb)
        };
        av.push(ta);
        bv.push(tb);
    }
    av.remove(0);
    bv.remove(0);
    (
        median(av).as_secs_f64() * 1e3,
        median(bv).as_secs_f64() * 1e3,
    )
}

fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "<unreadable>".into())
}

fn sample(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed) % 101) as f32) * 0.05 - 2.5)
        .collect()
}

fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn main() {
    let threads: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let installed = lasx_rs::pool::init_global(threads);
    let be = LasxBackend;
    println!("# M5 组合原语 A/B：axpy_inplace_f32 override vs portable 默认");
    println!(
        "- threads = {}（init_global({threads}) = {installed}）；loadavg = `{}`（同进程交替 {ROUNDS} 对丢首对中位）",
        lasx_rs::pool::global().threads(),
        loadavg()
    );
    println!("| 用例 | A=Portable ms | B=LASX ms | A/B | 等价类 |");
    println!("|---|---|---|---|---|");
    for n in [1usize << 20, 1usize << 23] {
        let x = sample(n, 3);
        let base = sample(n, 11);
        let alpha = 0.375f32;
        let mut y_l = base.clone();
        let mut y_p = base.clone();
        be.axpy_inplace_f32(&mut y_l, &x, alpha).unwrap();
        ScalarBackend.axpy_inplace_f32(&mut y_p, &x, alpha).unwrap();
        let eq = if bits_eq(&y_l, &y_p) {
            "**逐位**"
        } else {
            "**不逐位！**"
        };
        let (a, b) = ab(
            || {
                ScalarBackend.axpy_inplace_f32(&mut y_p, &x, alpha).unwrap();
                black_box(y_p[0]);
            },
            || {
                be.axpy_inplace_f32(&mut y_l, &x, alpha).unwrap();
                black_box(y_l[0]);
            },
        );
        println!(
            "| axpy_inplace n={n} | {a:.3} | {b:.3} | {:.3}× | {eq} |",
            a / b
        );
    }
    println!("loadavg（结束）= `{}`", loadavg());
}
