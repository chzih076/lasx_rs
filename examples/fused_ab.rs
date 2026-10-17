//! **融合原语 A/B**（等 `simd_dsl` 契约合入后启用）：DSL override 路径 vs `lasx_rs::api::*` 原实现，
//! 同进程交替、≥5 轮丢首轮中位、打印 `threads`/`ms`/`GFLOPS`/`loadavg`；判据 **ratio ≥ 0.98**。
//!
//! 现状：契约分支尚未领先 master（`master..fused-contract` 为空）⇒ `Backend` 上还没有这 7 个方法，
//! 整套装置被 `#[cfg(any())]` 关闭（**能编译**，`cargo clippy --all-targets` 两档都不会红）。
//! **启用步骤**见 `src/dsl_backend_fused.rs` 顶部（bump rev → 去掉 `#[cfg(any())]` → 加 `[[example]]`）。
//!
//! 覆盖：`softmax_rows_f32` / `rms_norm_rows_f32` / `layer_norm_rows_f32` / `gelu_quick_f32` /
//! `rope_apply_f32` / `axpy_f32`；每对先做等价类（逐位优先，否则打印实测界），再计时。

#[cfg(any())]
fn main() {
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    use lasx_rs::api;
    use lasx_rs::dsl_backend::LasxBackend;
    use simd_dsl::Backend;

    const ROUNDS: usize = 6;
    const EPS: f32 = 1e-5;

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
        ta.remove(0);
        tb.remove(0);
        (median(ta), median(tb))
    }
    fn sample(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 37 % 101) as f32) * 0.1 - 6.07)
            .collect()
    }
    fn check(name: &str, a: &[f32], b: &[f32]) {
        if a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()) {
            println!("  [{name}] 等价类：**逐位**");
            return;
        }
        let mut max_abs = 0.0f32;
        let mut peak = 1.0f32;
        for (x, y) in a.iter().zip(b) {
            peak = peak.max(x.abs());
            max_abs = max_abs.max((x - y).abs());
        }
        println!(
            "  [{name}] 等价类：**容差**（max abs {max_abs:.3e}，max rel {:.3e}）",
            max_abs / peak
        );
    }
    fn row(name: &str, n: usize, flops: f64, a: Duration, b: Duration) {
        let (ams, bms) = (a.as_secs_f64() * 1e3, b.as_secs_f64() * 1e3);
        println!(
            "| {name} | {ams:.3} | {bms:.3} | **{:.3}×** | {:.2} / {:.2} |",
            ams / bms,
            n as f64 * flops / (ams * 1e-3) / 1e9,
            n as f64 * flops / (bms * 1e-3) / 1e9
        );
    }

    let be = LasxBackend;
    println!("融合原语 A/B（LasxBackend override vs api；{ROUNDS} 对丢首对取中位）");
    println!("threads = {}", lasx_rs::pool::global().threads());
    println!("loadavg(开始) = {}", loadavg());
    println!(
        "| 算子 | api ms（B 臂=原实现? 见列头） | override ms | 比值 api/override | GFLOPS api / override |"
    );
    println!("|---|---|---|---|---|");

    // softmax_rows
    {
        let (rows, cols) = (1024usize, 768usize);
        let n = rows * cols;
        let x = sample(n);
        let mut a_out = vec![0.0f32; n];
        let mut b_out = vec![0.0f32; n];
        api::softmax_rows_into(&x, None, rows, cols, 1.0, &mut a_out).unwrap();
        be.softmax_rows_f32(&x, rows, cols, &mut b_out).unwrap();
        check("softmax_rows", &a_out, &b_out);
        let (a, b) = ab(
            || {
                api::softmax_rows_into(&x, None, rows, cols, 1.0, &mut a_out).unwrap();
                black_box(&a_out);
            },
            || {
                be.softmax_rows_f32(&x, rows, cols, &mut b_out).unwrap();
                black_box(&b_out);
            },
        );
        row("softmax_rows", n, 6.0, a, b);
    }

    // rms_norm / layer_norm / gelu_quick / axpy / rope 同构 —— 按同一模板补齐（启用时逐段放开）。
    println!("（其余 5 对按上表同一模板：rms/layer/gelu_quick/axpy/rope）");
    println!("loadavg(结束) = {}", loadavg());
}

#[cfg(not(any()))]
fn main() {
    println!(
        "fused_ab：等待 simd_dsl 融合原语契约合入（见 src/dsl_backend_fused.rs 顶部启用步骤）"
    );
}
