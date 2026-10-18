//! **融合原语 A/B**（阶段 4）：DSL override 路径（`Backend::*_f32`）vs `lasx_rs::api::*_into` 直调。
//!
//! **定位**：override 是 `supports(FusedNorms|FusedElementwise)` 为真时的**加速选项**；
//! 源实现是 `simd_dsl` 的 portable 组合版（`ops_dsl` 侧，另由 `examples/composed_ab.rs` 对照）。
//!
//! 口径：同进程交替、6 对、**丢首对**、取中位；打印 `threads`/`ms`/`GFLOPS`/`loadavg`；
//! 先等价类（逐位优先，否则实测 `max abs/max rel`）。判据 `ratio = api_ms / dsl_ms ≥ 0.98`。
//!
//! 用法：`cargo run --release --features dsl-backend --example fused_ab`

use std::hint::black_box;
use std::time::{Duration, Instant};

use lasx_rs::api;
use lasx_rs::dsl_backend::LasxBackend;
use simd_dsl::{Backend, BackendError, FusedIo, FusedPlan};

const ROUNDS: usize = 6;
const EPS: f32 = 1e-5;

// **P1 批1（task-66）起**：`rms_norm_rows_f32`/`layer_norm_rows_f32`/`gelu_quick_f32`/
// `rope_apply_f32` 四个**具名方法已从契约删除** ⇒ 本示例改用 Plan 入口（同一 LASX 内核，
// 数值逐位不变）。保留 `check()` 逐位对照，确保持久等价。
fn dsl_rms(
    be: &LasxBackend,
    x: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> Result<(), BackendError> {
    be.exec_fused_plan(
        FusedPlan::RmsNormRows {
            rows,
            cols,
            eps: EPS,
        },
        FusedIo {
            x,
            y: &mut [],
            out,
            cos: &[],
            sin: &[],
        },
    )
}
fn dsl_layer(
    be: &LasxBackend,
    x: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> Result<(), BackendError> {
    be.exec_fused_plan(
        FusedPlan::LayerNormRows {
            rows,
            cols,
            eps: EPS,
        },
        FusedIo {
            x,
            y: &mut [],
            out,
            cos: &[],
            sin: &[],
        },
    )
}
fn dsl_gelu(be: &LasxBackend, x: &[f32], out: &mut [f32]) -> Result<(), BackendError> {
    be.exec_fused_plan(
        FusedPlan::GeluQuick,
        FusedIo {
            x,
            y: &mut [],
            out,
            cos: &[],
            sin: &[],
        },
    )
}
#[allow(clippy::too_many_arguments)]
fn dsl_rope(
    be: &LasxBackend,
    x: &[f32],
    rows: usize,
    cols: usize,
    n_dims: usize,
    cos: &[f32],
    sin: &[f32],
    out: &mut [f32],
) -> Result<(), BackendError> {
    be.exec_fused_plan(
        FusedPlan::RopeApply { rows, cols, n_dims },
        FusedIo {
            x,
            y: &mut [],
            out,
            cos,
            sin,
        },
    )
}

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
        "| {name} | {ams:.3} | {bms:.3} | **{:.3}×** | {:.2} / {:.2} |",
        ams / bms,
        n as f64 * flops / (ams * 1e-3) / 1e9,
        n as f64 * flops / (bms * 1e-3) / 1e9
    );
}

fn main() {
    let be = LasxBackend;
    println!("融合原语 A/B（api 直调 vs Backend override；{ROUNDS} 对丢首对取中位）");
    println!("threads = {}", lasx_rs::pool::global().threads());
    println!(
        "cap: FusedNorms={} FusedElementwise={}",
        be.supports(simd_dsl::Capability::FusedNorms),
        be.supports(simd_dsl::Capability::FusedElementwise)
    );
    println!("loadavg(开始) = {}", loadavg());
    println!("| 算子 | api ms | dsl ms | 比值 api/dsl | GFLOPS api / dsl |");
    println!("|---|---|---|---|---|");

    // softmax_rows
    {
        let (rows, cols) = (1024usize, 768usize);
        let n = rows * cols;
        let x = sample(n);
        let (mut ao, mut bo) = (vec![0.0f32; n], vec![0.0f32; n]);
        api::softmax_rows_into(&x, None, rows, cols, 1.0, &mut ao).unwrap();
        be.softmax_rows_f32(&x, rows, cols, &mut bo).unwrap();
        check("softmax_rows", &ao, &bo);
        let (a, b) = ab(
            || {
                api::softmax_rows_into(&x, None, rows, cols, 1.0, &mut ao).unwrap();
                black_box(&ao);
            },
            || {
                be.softmax_rows_f32(&x, rows, cols, &mut bo).unwrap();
                black_box(&bo);
            },
        );
        row("softmax_rows", n, 6.0, a, b);
    }
    // rms_norm
    {
        let (rows, cols) = (1024usize, 768usize);
        let n = rows * cols;
        let x = sample(n);
        let (mut ao, mut bo) = (vec![0.0f32; n], vec![0.0f32; n]);
        api::rms_norm_into(&x, None, rows, cols, EPS, &mut ao).unwrap();
        dsl_rms(&be, &x, rows, cols, &mut bo).unwrap();
        check("rms_norm", &ao, &bo);
        let (a, b) = ab(
            || {
                api::rms_norm_into(&x, None, rows, cols, EPS, &mut ao).unwrap();
                black_box(&ao);
            },
            || {
                dsl_rms(&be, &x, rows, cols, &mut bo).unwrap();
                black_box(&bo);
            },
        );
        row("rms_norm", n, 4.0, a, b);
    }
    // layer_norm
    {
        let (rows, cols) = (1024usize, 768usize);
        let n = rows * cols;
        let x = sample(n);
        let (mut ao, mut bo) = (vec![0.0f32; n], vec![0.0f32; n]);
        api::layer_norm_into(&x, None, None, rows, cols, EPS, &mut ao).unwrap();
        dsl_layer(&be, &x, rows, cols, &mut bo).unwrap();
        check("layer_norm", &ao, &bo);
        let (a, b) = ab(
            || {
                api::layer_norm_into(&x, None, None, rows, cols, EPS, &mut ao).unwrap();
                black_box(&ao);
            },
            || {
                dsl_layer(&be, &x, rows, cols, &mut bo).unwrap();
                black_box(&bo);
            },
        );
        row("layer_norm", n, 6.0, a, b);
    }
    // gelu_quick
    {
        let n = 8 * 1024 * 1024;
        let x = sample(n);
        let (mut ao, mut bo) = (vec![0.0f32; n], vec![0.0f32; n]);
        api::gelu_quick_into(&x, &mut ao).unwrap();
        dsl_gelu(&be, &x, &mut bo).unwrap();
        check("gelu_quick", &ao, &bo);
        let (a, b) = ab(
            || {
                api::gelu_quick_into(&x, &mut ao).unwrap();
                black_box(&ao);
            },
            || {
                dsl_gelu(&be, &x, &mut bo).unwrap();
                black_box(&bo);
            },
        );
        row("gelu_quick", n, 9.0, a, b);
    }
    // axpy
    {
        let n = 8 * 1024 * 1024;
        let x = sample(n);
        let y0 = sample(n + 5)[5..].to_vec();
        let (mut ya, mut yb) = (y0.clone(), y0);
        api::axpy(1.25, &x, &mut ya).unwrap();
        be.axpy_f32(&mut yb, &x, 1.25).unwrap();
        check("axpy", &ya, &yb);
        let (a, b) = ab(
            || {
                api::axpy(1.25, &x, &mut ya).unwrap();
                black_box(&ya);
            },
            || {
                be.axpy_f32(&mut yb, &x, 1.25).unwrap();
                black_box(&yb);
            },
        );
        row("axpy", n, 2.0, a, b);
    }
    // rope（NeoX = half-split）
    {
        let (rows, cols, n_dims) = (1024usize, 768usize, 64usize);
        let n = rows * cols;
        let d = n_dims / 2;
        let x = sample(n);
        let cos: Vec<f32> = (0..rows * d).map(|i| (0.1 * i as f32).cos()).collect();
        let sin: Vec<f32> = (0..rows * d).map(|i| (0.1 * i as f32).sin()).collect();
        let (mut ao, mut bo) = (vec![0.0f32; n], vec![0.0f32; n]);
        api::rope_into(
            &x,
            &cos,
            &sin,
            rows,
            cols,
            n_dims,
            api::RopeMode::NeoX,
            &mut ao,
        )
        .unwrap();
        dsl_rope(&be, &x, rows, cols, n_dims, &cos, &sin, &mut bo).unwrap();
        check("rope", &ao, &bo);
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
                    &mut ao,
                )
                .unwrap();
                black_box(&ao);
            },
            || {
                dsl_rope(&be, &x, rows, cols, n_dims, &cos, &sin, &mut bo).unwrap();
                black_box(&bo);
            },
        );
        row("rope", n, 8.0, a, b);
    }
    println!("loadavg(结束) = {}", loadavg());
    println!(
        "判据：比值 api/dsl ≥0.98 算不输（override 是薄转发，预期 ≈1.00）；组合版对照见 composed_ab。"
    );
}
