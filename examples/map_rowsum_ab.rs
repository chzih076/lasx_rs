//! **P0-2：单趟 map+行和（`map_rows_sum_f32`）override vs portable 的 A/B**。
//!
//! A 臂 = `simd_dsl::kernels::scalar::map_rows_sum_f32`（portable 默认实现，两输出一次循环）；
//! B 臂 = [`LasxBackend::map_rows_sum_f32`]（LASX 单趟向量化：`exp_vec` + per-lane 累加）。
//!
//! 口径：同进程交替、6 对、**丢首对**、取中位；打印 `ms`/`ratio`/`loadavg`；先等价类：
//! * **映射值**：两个 exp 实现不同（LASX `exp_vec` vs libm）⇒ 期望**容差**，打印实测 rel 界；
//! * **行和**：per-lane 累加 vs 单累加器 ⇒ **容差**，打印实测 rel 界。
//!
//! 判据：`ratio = portable_ms / lasx_ms ≥ 1.10`（≥10%）才保留 override；否则回退并如实记。
//!
//! 用法：`cargo run --release --features dsl-backend --example map_rowsum_ab`

use std::hint::black_box;
use std::time::{Duration, Instant};

use lasx_rs::dsl_backend::LasxBackend;
use simd_dsl::kernels::scalar;
use simd_dsl::{Backend, ReduceMapOp};

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
/// 相对峰值误差（两输出都按同一口径）。
fn rel(a: &[f32], b: &[f32]) -> f64 {
    let mut max_abs = 0.0f64;
    let mut peak = 1.0f64;
    for (x, y) in a.iter().zip(b) {
        peak = peak.max(x.abs() as f64);
        max_abs = max_abs.max((x - y).abs() as f64);
    }
    max_abs / peak
}
fn row(name: &str, rows: usize, cols: usize, a: Duration, b: Duration) {
    let (ams, bms) = (a.as_secs_f64() * 1e3, b.as_secs_f64() * 1e3);
    let ratio = ams / bms;
    println!(
        "| {name} | {rows}×{cols} | {ams:.3} | {bms:.3} | **{ratio:.3}×** | {} |",
        if ratio >= 1.10 {
            "采纳"
        } else {
            "BelowThreshold"
        }
    );
}

fn main() {
    let be = LasxBackend;
    println!("P0-2 单趟 map+行和 A/B（portable vs LASX override；{ROUNDS} 对丢首对取中位）");
    println!("threads = {}", lasx_rs::pool::global().threads());
    println!("loadavg(开始) = {}", loadavg());
    println!("| 算子 | 形状 | portable ms | lasx ms | portable/lasx | 判定 |");
    println!("|---|---|---|---|---|---|");

    for (rows, cols) in [(64usize, 64usize), (1024, 1024), (2048, 4096)] {
        let n = rows * cols;
        let x = sample(n);
        // 行 max 作 shift（`ExpShiftedRow` 的典型输入）
        let shift: Vec<f32> = (0..rows)
            .map(|r| {
                x[r * cols..(r + 1) * cols]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .collect();

        for (label, map) in [
            ("ExpShiftedRow", ReduceMapOp::ExpShiftedRow),
            ("Exp", ReduceMapOp::Exp),
        ] {
            let sh: &[f32] = if matches!(map, ReduceMapOp::Exp) {
                &[]
            } else {
                &shift
            };
            let (mut ma, mut sa, mut mb, mut sb) = (
                vec![0.0f32; n],
                vec![0.0f32; rows],
                vec![0.0f32; n],
                vec![0.0f32; rows],
            );
            scalar::map_rows_sum_f32(&x, rows, cols, sh, map, &mut ma, &mut sa).unwrap();
            be.map_rows_sum_f32(&x, rows, cols, sh, map, &mut mb, &mut sb)
                .unwrap();
            let mapped_rel = rel(&ma, &mb);
            let sum_rel = rel(&sa, &sb);
            let bits_same = ma.iter().zip(&mb).all(|(a, b)| a.to_bits() == b.to_bits());
            println!(
                "  [{label} {rows}×{cols}] 等价类：映射值 **{}**（rel {mapped_rel:.3e}），行和容差（rel {sum_rel:.3e}）",
                if bits_same { "逐位" } else { "容差" }
            );
            let (ta, tb) = ab(
                || {
                    scalar::map_rows_sum_f32(&x, rows, cols, sh, map, &mut ma, &mut sa).unwrap();
                    black_box(&ma);
                },
                || {
                    be.map_rows_sum_f32(&x, rows, cols, sh, map, &mut mb, &mut sb)
                        .unwrap();
                    black_box(&mb);
                },
            );
            row(label, rows, cols, ta, tb);
        }
    }

    println!("loadavg(结束) = {}", loadavg());
    println!(
        "判据：portable/lasx ≥1.10 才保留 override（≥10%）；否则回退 portable 并如实记 BelowThreshold。"
    );
}
