//! **阶段 4 最终 A/B**（`task-44`）：`*_auto` 路径（LASX `supports(Fused*)` 为真 ⇒ 1 趟融合入口）
//! vs `lasx_rs::api::*`；外加批1 的 matmul/matmul_f64 复核行。
//!
//! 口径：同进程交替、6 对、**丢首对**、取中位；打印 `threads`/`ms`/`GFLOPS`/`loadavg`；
//! 每对先等价类（逐位优先，否则实测 `max abs/max rel`）。判据 `ratio = api_ms / dsl_ms ≥ 0.98`。
//!
//! 组2（批2 组合版 vs api）与组3（批3 `BelowThreshold`）的数字**已在**
//! `examples/composed_ab.rs`（lasx master 已合并）与 `lasx_rs/docs/dev.md §7.30/§7.31` 记录，
//! 本装置不重复造第二份数字（数字家唯一）；master 总表由 Lead 汇总两处。
//!
//! 用法：`cargo run --release --features dsl-backend --example auto_ab`

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
    (median(ta[1..].to_vec()), median(tb[1..].to_vec()))
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
        "| {name} | {n} | {ams:.3} | {bms:.3} | **{:.3}×** | {:.2} / {:.2} |",
        ams / bms,
        n as f64 * flops / (ams * 1e-3) / 1e9,
        n as f64 * flops / (bms * 1e-3) / 1e9
    );
}

fn main() {
    let be = LasxBackend;
    println!(
        "阶段4 最终 A/B（api 直调 vs ops_dsl::*_auto 经 LasxBackend；{ROUNDS} 对丢首对取中位）"
    );
    println!("threads = {}", lasx_rs::pool::global().threads());
    println!(
        "cap: FusedNorms={} FusedElementwise={}",
        be.supports(simd_dsl::Capability::FusedNorms),
        be.supports(simd_dsl::Capability::FusedElementwise)
    );
    println!("loadavg(开始) = {}", loadavg());
    println!("| 算子 | n | api ms | auto ms | 比值 api/auto | GFLOPS api / auto |");
    println!("|---|---|---|---|---|---|");

    // ---- softmax_rows_auto（形状 1024×768；scratch 与组合版同）----
    {
        let (rows, cols) = (1024usize, 768usize);
        let n = rows * cols;
        let x = sample(n);
        let (mut stat, mut shifted, mut ex) =
            (vec![0.0f32; rows], vec![0.0f32; n], vec![0.0f32; n]);
        let (mut ao, mut bo) = (vec![0.0f32; n], vec![0.0f32; n]);
        api::softmax_rows_into(&x, None, rows, cols, 1.0, &mut ao).unwrap();
        ops_dsl::norm::softmax_rows_auto(
            &be,
            &x,
            rows,
            cols,
            &mut stat,
            &mut shifted,
            &mut ex,
            &mut bo,
        )
        .unwrap();
        check("softmax_rows_auto", &ao, &bo);
        let (a, b) = ab(
            || {
                api::softmax_rows_into(&x, None, rows, cols, 1.0, &mut ao).unwrap();
                black_box(&ao);
            },
            || {
                ops_dsl::norm::softmax_rows_auto(
                    &be,
                    &x,
                    rows,
                    cols,
                    &mut stat,
                    &mut shifted,
                    &mut ex,
                    &mut bo,
                )
                .unwrap();
                black_box(&bo);
            },
        );
        row("softmax_rows_auto", n, 6.0, a, b);
    }
    // ---- rms_norm_auto ----
    {
        let (rows, cols) = (1024usize, 768usize);
        let n = rows * cols;
        let x = sample(n);
        let (mut sq, mut stat, mut inv_std) =
            (vec![0.0f32; n], vec![0.0f32; rows], vec![0.0f32; rows]);
        let (mut ao, mut bo) = (vec![0.0f32; n], vec![0.0f32; n]);
        api::rms_norm_into(&x, None, rows, cols, EPS, &mut ao).unwrap();
        ops_dsl::norm::rms_norm_auto(
            &be,
            &x,
            rows,
            cols,
            EPS,
            &mut sq,
            &mut stat,
            &mut inv_std,
            &mut bo,
        )
        .unwrap();
        check("rms_norm_auto", &ao, &bo);
        let (a, b) = ab(
            || {
                api::rms_norm_into(&x, None, rows, cols, EPS, &mut ao).unwrap();
                black_box(&ao);
            },
            || {
                ops_dsl::norm::rms_norm_auto(
                    &be,
                    &x,
                    rows,
                    cols,
                    EPS,
                    &mut sq,
                    &mut stat,
                    &mut inv_std,
                    &mut bo,
                )
                .unwrap();
                black_box(&bo);
            },
        );
        row("rms_norm_auto", n, 4.0, a, b);
    }
    // ---- layer_norm_auto ----
    {
        let (rows, cols) = (1024usize, 768usize);
        let n = rows * cols;
        let x = sample(n);
        let (mut d, mut stat, mut inv_std) =
            (vec![0.0f32; n], vec![0.0f32; rows], vec![0.0f32; rows]);
        let (mut ao, mut bo) = (vec![0.0f32; n], vec![0.0f32; n]);
        api::layer_norm_into(&x, None, None, rows, cols, EPS, &mut ao).unwrap();
        ops_dsl::norm::layer_norm_auto(
            &be,
            &x,
            rows,
            cols,
            EPS,
            &mut d,
            &mut stat,
            &mut inv_std,
            &mut bo,
        )
        .unwrap();
        check("layer_norm_auto", &ao, &bo);
        let (a, b) = ab(
            || {
                api::layer_norm_into(&x, None, None, rows, cols, EPS, &mut ao).unwrap();
                black_box(&ao);
            },
            || {
                ops_dsl::norm::layer_norm_auto(
                    &be,
                    &x,
                    rows,
                    cols,
                    EPS,
                    &mut d,
                    &mut stat,
                    &mut inv_std,
                    &mut bo,
                )
                .unwrap();
                black_box(&bo);
            },
        );
        row("layer_norm_auto", n, 6.0, a, b);
    }
    // ---- gelu_quick_auto（8M）----
    {
        let n = 8 * 1024 * 1024;
        let x = sample(n);
        let (mut t1, mut t2, mut t3) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
        let (mut ao, mut bo) = (vec![0.0f32; n], vec![0.0f32; n]);
        api::gelu_quick_into(&x, &mut ao).unwrap();
        ops_dsl::activations::gelu_quick_auto(&be, &x, &mut t1, &mut t2, &mut t3, &mut bo).unwrap();
        check("gelu_quick_auto", &ao, &bo);
        let (a, b) = ab(
            || {
                api::gelu_quick_into(&x, &mut ao).unwrap();
                black_box(&ao);
            },
            || {
                ops_dsl::activations::gelu_quick_auto(&be, &x, &mut t1, &mut t2, &mut t3, &mut bo)
                    .unwrap();
                black_box(&bo);
            },
        );
        row("gelu_quick_auto", n, 9.0, a, b);
    }
    // ---- 批1 复核：matmul（三个尺寸）+ matmul_f64 ----
    {
        for (m, k, nn) in [
            (1024usize, 1024usize, 1024usize),
            (512, 512, 512),
            (192, 768, 3072),
        ] {
            let (a_v, b_v) = (sample(m * k), sample(k * nn));
            let (mut ao, mut bo) = (vec![0.0f32; m * nn], vec![0.0f32; m * nn]);
            api::matmul_into(m, k, nn, &a_v, &b_v, &mut ao).unwrap();
            ops_dsl::matmul::matmul_f32(&be, m, k, nn, &a_v, &b_v, &mut bo).unwrap();
            check("matmul", &ao, &bo);
            let (a, b) = ab(
                || {
                    api::matmul_into(m, k, nn, &a_v, &b_v, &mut ao).unwrap();
                    black_box(&ao);
                },
                || {
                    ops_dsl::matmul::matmul_f32(&be, m, k, nn, &a_v, &b_v, &mut bo).unwrap();
                    black_box(&bo);
                },
            );
            row("matmul_f32", m * k * nn, 2.0, a, b);
        }
        let (m, k, nn) = (512usize, 512usize, 512usize);
        let (a_v, b_v) = (
            sample(m * k)
                .iter()
                .map(|&v| v as f64)
                .collect::<Vec<f64>>(),
            sample(k * nn)
                .iter()
                .map(|&v| v as f64)
                .collect::<Vec<f64>>(),
        );
        let (mut ao, mut bo) = (vec![0.0f64; m * nn], vec![0.0f64; m * nn]);
        api::matmul_f64_into(m, k, nn, &a_v, &b_v, &mut ao).unwrap();
        ops_dsl::matmul::matmul_f64(&be, m, k, nn, &a_v, &b_v, &mut bo).unwrap();
        let same = ao.iter().zip(&bo).all(|(x, y)| x.to_bits() == y.to_bits());
        println!(
            "  [matmul_f64] 等价类：**{}**",
            if same { "逐位" } else { "容差" }
        );
        let (a, b) = ab(
            || {
                api::matmul_f64_into(m, k, nn, &a_v, &b_v, &mut ao).unwrap();
                black_box(&ao);
            },
            || {
                ops_dsl::matmul::matmul_f64(&be, m, k, nn, &a_v, &b_v, &mut bo).unwrap();
                black_box(&bo);
            },
        );
        row("matmul_f64", m * k * nn, 2.0, a, b);
    }

    println!("loadavg(结束) = {}", loadavg());
    println!(
        "判据：比值 api/dsl ≥0.98 不输；组2 组合版列与组3 BelowThreshold 数字见 composed_ab / dev.md §7.30–§7.31。"
    );
}
