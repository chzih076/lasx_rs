//! **批 2 组合算子 A/B**（阶段 3 批 2 收尾；`plan_lasx_to_ops.md` §4）：
//! `ops_dsl` 的**纯宏组合版**（经 [`LasxBackend`]）vs `lasx_rs::api::*` 的**融合核**。
//!
//! 口径（与 `migrate_ab.rs` 同一套）：**同进程交替**、`ROUNDS` 对、**丢首对**、取**中位**；
//! 打印 `loadavg`、`ms`、`GFLOPS`/`GB/s`；每对**先做等价类检查**（`to_bits` 逐位优先，不逐位则打印
//! `max abs/max rel`）。判据：`ratio = api_ms / ops_ms`，**≥0.98 算"不输"**；<0.98 如实记。
//!
//! 用法：`cargo run --release --features dsl-backend --example composed_ab`
//!
use std::hint::black_box;
use std::time::{Duration, Instant};

use lasx_rs::api;
use lasx_rs::dsl_backend::LasxBackend;

const ROUNDS: usize = 6; // 丢首对 ⇒ 5 对取中位
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

/// 同进程交替 A/B：偶数对 A→B、奇数对 B→A；**丢首对**；返回（A 中位、B 中位）。
fn ab(mut a: impl FnMut(), mut b: impl FnMut()) -> (Duration, Duration) {
    let (mut ta_all, mut tb_all) = (Vec::new(), Vec::new());
    for round in 0..ROUNDS {
        if round % 2 == 0 {
            ta_all.push(timeit(&mut a));
            tb_all.push(timeit(&mut b));
        } else {
            tb_all.push(timeit(&mut b));
            ta_all.push(timeit(&mut a));
        }
    }
    ta_all.remove(0);
    tb_all.remove(0);
    (median(ta_all), median(tb_all))
}

fn sample(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 37 % 101) as f32) * 0.1 - 6.07)
        .collect()
}

/// 等价类检查：优先逐位；不逐位则打印实测 `max abs` / `max rel`（相对峰值）。
fn check_class(name: &str, a: &[f32], b: &[f32]) -> bool {
    if a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()) {
        println!("  [{name}] 等价类：**逐位**");
        return true;
    }
    let mut max_abs = 0.0f32;
    let mut peak = 1.0f32;
    for (x, y) in a.iter().zip(b) {
        if x.abs() > peak {
            peak = x.abs();
        }
        let d = (x - y).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    println!(
        "  [{name}] 等价类：**容差**（max abs = {max_abs:.3e}，max rel = {:.3e}）",
        max_abs / peak
    );
    false
}

fn row(name: &str, n: usize, flops_per_elem: f64, a: Duration, b: Duration, exact: bool) {
    let (ams, bms) = (a.as_secs_f64() * 1e3, b.as_secs_f64() * 1e3);
    let ratio = ams / bms; // api / ops：>1 ⇒ 组合版更快
    let gflops = |ms: f64| n as f64 * flops_per_elem / (ms * 1e-3) / 1e9;
    println!(
        "| {name} | {ams:.3} | {bms:.3} | **{ratio:.3}×** | {} | {:.2} / {:.2} |",
        if exact { "逐位" } else { "容差" },
        gflops(ams),
        gflops(bms)
    );
}

fn main() {
    let backend = LasxBackend;
    println!("批 2 组合算子 A/B（LasxBackend；{ROUNDS} 对、丢首对、取中位）");
    println!("threads = {}", lasx_rs::pool::global().threads());
    println!("loadavg(开始) = {}", loadavg());
    println!("| 算子 | api ms | ops 组合 ms | 比值 api/ops | 等价类 | GFLOPS api / ops |");
    println!("|---|---|---|---|---|---|");

    // ---- softmax_rows：rows×cols ----
    {
        let (rows, cols) = (1024usize, 768usize);
        let n = rows * cols;
        let x = sample(n);
        let (mut stat, mut shifted, mut ex) =
            (vec![0.0f32; rows], vec![0.0f32; n], vec![0.0f32; n]);
        let mut out = vec![0.0f32; n];
        let api_out = api::softmax_rows(&x, None, rows, cols, 1.0).unwrap();
        ops_dsl::norm::softmax_rows_composed(
            &backend,
            &x,
            rows,
            cols,
            &mut stat,
            &mut shifted,
            &mut ex,
            &mut out,
        )
        .unwrap();
        let exact = check_class("softmax_rows", &api_out, &out);
        let (a, b) = ab(
            || {
                black_box(api::softmax_rows(&x, None, rows, cols, 1.0).unwrap());
            },
            || {
                ops_dsl::norm::softmax_rows_composed(
                    &backend,
                    &x,
                    rows,
                    cols,
                    &mut stat,
                    &mut shifted,
                    &mut ex,
                    &mut out,
                )
                .unwrap();
                black_box(&out);
            },
        );
        row("softmax_rows", n, 6.0, a, b, exact);
    }

    // ---- rms_norm ----
    {
        let (rows, cols) = (1024usize, 768usize);
        let n = rows * cols;
        let x = sample(n);
        let (mut sq, mut stat, mut inv_std) =
            (vec![0.0f32; n], vec![0.0f32; rows], vec![0.0f32; rows]);
        let mut out = vec![0.0f32; n];
        let api_out = api::rms_norm(&x, None, rows, cols, EPS).unwrap();
        ops_dsl::norm::rms_norm_composed(
            &backend,
            &x,
            rows,
            cols,
            EPS,
            &mut sq,
            &mut stat,
            &mut inv_std,
            &mut out,
        )
        .unwrap();
        let exact = check_class("rms_norm", &api_out, &out);
        let (a, b) = ab(
            || {
                black_box(api::rms_norm(&x, None, rows, cols, EPS).unwrap());
            },
            || {
                ops_dsl::norm::rms_norm_composed(
                    &backend,
                    &x,
                    rows,
                    cols,
                    EPS,
                    &mut sq,
                    &mut stat,
                    &mut inv_std,
                    &mut out,
                )
                .unwrap();
                black_box(&out);
            },
        );
        row("rms_norm", n, 4.0, a, b, exact);
    }

    // ---- layer_norm ----
    {
        let (rows, cols) = (1024usize, 768usize);
        let n = rows * cols;
        let x = sample(n);
        let (mut d, mut stat, mut inv_std) =
            (vec![0.0f32; n], vec![0.0f32; rows], vec![0.0f32; rows]);
        let mut out = vec![0.0f32; n];
        let api_out = api::layer_norm(&x, None, None, rows, cols, EPS).unwrap();
        ops_dsl::norm::layer_norm_composed(
            &backend,
            &x,
            rows,
            cols,
            EPS,
            &mut d,
            &mut stat,
            &mut inv_std,
            &mut out,
        )
        .unwrap();
        let exact = check_class("layer_norm", &api_out, &out);
        let (a, b) = ab(
            || {
                black_box(api::layer_norm(&x, None, None, rows, cols, EPS).unwrap());
            },
            || {
                ops_dsl::norm::layer_norm_composed(
                    &backend,
                    &x,
                    rows,
                    cols,
                    EPS,
                    &mut d,
                    &mut stat,
                    &mut inv_std,
                    &mut out,
                )
                .unwrap();
                black_box(&out);
            },
        );
        row("layer_norm", n, 6.0, a, b, exact);
    }

    // ---- gelu_quick（逐元素 8M）----
    {
        let n = 8 * 1024 * 1024;
        let x = sample(n);
        let (mut t1, mut t2, mut t3) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
        let mut out = vec![0.0f32; n];
        let api_out = api::gelu_quick(&x);
        ops_dsl::activations::gelu_quick(&backend, &x, &mut t1, &mut t2, &mut t3, &mut out)
            .unwrap();
        let exact = check_class("gelu_quick", &api_out, &out);
        let (a, b) = ab(
            || {
                black_box(api::gelu_quick(&x));
            },
            || {
                ops_dsl::activations::gelu_quick(&backend, &x, &mut t1, &mut t2, &mut t3, &mut out)
                    .unwrap();
                black_box(&out);
            },
        );
        row("gelu_quick", n, 9.0, a, b, exact);
    }

    // ---- axpy（原地）----
    {
        let n = 8 * 1024 * 1024;
        let x = sample(n);
        let y0 = sample(n + 5)[5..].to_vec();
        let (mut y_api, mut y_ops) = (y0.clone(), y0.clone());
        let (mut tmp, mut acc) = (vec![0.0f32; n], vec![0.0f32; n]);
        api::axpy(1.25, &x, &mut y_api).unwrap();
        ops_dsl::reduce::axpy_f32(&backend, 1.25, &x, &mut y_ops, &mut tmp, &mut acc).unwrap();
        let exact = check_class("axpy", &y_api, &y_ops);
        let (a, b) = ab(
            || {
                api::axpy(1.25, &x, &mut y_api).unwrap();
                black_box(&y_api);
            },
            || {
                ops_dsl::reduce::axpy_f32(&backend, 1.25, &x, &mut y_ops, &mut tmp, &mut acc)
                    .unwrap();
                black_box(&y_ops);
            },
        );
        row("axpy", n, 2.0, a, b, exact);
    }

    // ---- amax ----
    {
        let n = 8 * 1024 * 1024;
        let x = sample(n);
        let (mut neg, mut abs) = (vec![0.0f32; n], vec![0.0f32; n]);
        let api_v = api::amax(&x).unwrap();
        let ops_v = ops_dsl::reduce::amax_f32(&backend, &x, &mut neg, &mut abs);
        let class = if api_v.to_bits() == ops_v.to_bits() {
            "逐位"
        } else {
            "容差"
        };
        println!("  [amax] 等价类：**{class}**（api = {api_v}, ops = {ops_v}）");
        let (a, b) = ab(
            || {
                black_box(api::amax(&x).unwrap());
            },
            || {
                black_box(ops_dsl::reduce::amax_f32(&backend, &x, &mut neg, &mut abs));
            },
        );
        row("amax", n, 1.0, a, b, class == "逐位");
    }

    // ---- rope（NeoX = half-split）----
    {
        let (rows, cols, n_dims) = (1024usize, 768usize, 64usize);
        let n = rows * cols;
        let d = n_dims / 2;
        let x = sample(n);
        let cos: Vec<f32> = (0..rows * d).map(|i| (0.1 * i as f32).cos()).collect();
        let sin: Vec<f32> = (0..rows * d).map(|i| (0.1 * i as f32).sin()).collect();
        let mut api_out = vec![0.0f32; n];
        api::rope_into(
            &x,
            &cos,
            &sin,
            rows,
            cols,
            n_dims,
            api::RopeMode::NeoX,
            &mut api_out,
        )
        .unwrap();
        let (mut t1, mut t2, mut t3, mut t4) = (
            vec![0.0f32; rows * d],
            vec![0.0f32; rows * d],
            vec![0.0f32; rows * d],
            vec![0.0f32; rows * d],
        );
        let mut ops_out = vec![0.0f32; n];
        ops_dsl::rope::rope_apply_half_f32(
            &backend,
            &x,
            rows,
            cols,
            n_dims,
            &cos,
            &sin,
            &mut t1,
            &mut t2,
            &mut t3,
            &mut t4,
            &mut ops_out,
        )
        .unwrap();
        let exact = check_class("rope", &api_out, &ops_out);
        let (a, b) = ab(
            || {
                api::rope_into(
                    &x,
                    &cos,
                    &sin,
                    rows,
                    cols,
                    n_dims,
                    api::RopeMode::NeoX,
                    &mut api_out,
                )
                .unwrap();
                black_box(&api_out);
            },
            || {
                ops_dsl::rope::rope_apply_half_f32(
                    &backend,
                    &x,
                    rows,
                    cols,
                    n_dims,
                    &cos,
                    &sin,
                    &mut t1,
                    &mut t2,
                    &mut t3,
                    &mut t4,
                    &mut ops_out,
                )
                .unwrap();
                black_box(&ops_out);
            },
        );
        row("rope (NeoX)", n, 8.0, a, b, exact);
    }

    println!("loadavg(结束) = {}", loadavg());
    println!(
        "判据：比值 = api_ms / ops_ms；**≥0.98 算不输**，<0.98 如实记 BelowThreshold + 融合原语提案。"
    );
}
