//! **P1 批1 后端 A/B（task-66）**：`FusedPlan` 路径 vs 直接 `api::*`（6 个 kind）。
//!
//! 臂：
//! * **A = Plan 路**：`LasxBackend::exec_fused_plan(FusedPlan::…)`（本仓 override → LASX 融合核）；
//! * **B = 对比臂**：能直调的 `api::*`（`axpy`/`softmax_rows_into`/`rms_norm_into`/`layer_norm_into`/
//!   `gelu_quick_into`/`rope_into(RopeMode::NeoX)`）；`Axpy`/`SoftmaxRows` 另有具名方法
//!   （`axpy_f32`/`softmax_rows_f32`，薄委托同内核）一并对照；
//! * **C = portable 计划解释**（`ScalarBackend::exec_fused_plan`）：只做数值参照（慢，不进性能比）。
//!
//! 判据：**A ≡ B 逐位**（同一内核）；A vs C = 既有容差类；性能 **B/A ≥0.98**（Plan 路径不掉性能）。
//!
//! 用法（`--release`）：
//! ```text
//! cargo run --release --features dsl-backend --example plan_ab
//! ```

use std::hint::black_box;
use std::time::{Duration, Instant};

use lasx_rs::dsl_backend::LasxBackend;
use simd_dsl::{Backend, FusedIo, FusedPlan, PatternKind, ScalarBackend};

const ROUNDS: usize = 6;

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
        .unwrap_or_else(|_| "<unreadable>".to_string())
}
fn sample(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed) % 101) as f32) * 0.05 - 2.5)
        .collect()
}
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}
fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}
fn row(name: &str, a: f64, b: f64, eq: &str, dport: f32) {
    println!(
        "| {name} | {a:.3} | {b:.3} | {:.3}× | {eq}（vs portable 计划 max|d|={dport:.2e}） |",
        b / a
    );
}

/// 空 `FusedIo`（只有 RopeApply 用 `cos`/`sin`）。
fn io<'a>(x: &'a [f32], y: &'a mut [f32], out: &'a mut [f32]) -> FusedIo<'a> {
    FusedIo {
        x,
        y,
        out,
        cos: &[],
        sin: &[],
    }
}

fn main() {
    let be = LasxBackend;
    println!(
        "# P1 批1 后端 A/B：`exec_fused_plan`（6 kind）vs 直接 `api::*`（后端 = LasxBackend）"
    );
    println!(
        "- threads = {}；loadavg = `{}`（同进程交替 {ROUNDS} 对丢首对取中位）",
        lasx_rs::pool::global().threads(),
        loadavg()
    );
    let kinds = [
        PatternKind::Axpy,
        PatternKind::SoftmaxRows,
        PatternKind::RmsNormRows,
        PatternKind::LayerNormRows,
        PatternKind::GeluQuick,
        PatternKind::RopeApply,
    ];
    let flags = kinds
        .iter()
        .map(|k| format!("{k:?}={}", be.supports_plan(*k)))
        .collect::<Vec<_>>()
        .join(" ");
    println!("- `supports_plan`（LASX）：{flags}；portable 全 false（如实）");
    println!("| 用例 | A=Plan ms | B=对比臂 ms | B/A | 等价类 |");
    println!("|---|---|---|---|---|");

    // ---- Axpy ----
    {
        let n = 1usize << 20;
        let x = sample(n, 3);
        let base = sample(n, 11);
        let alpha = 0.375f32;
        let mut ya = base.clone();
        let mut yb = base.clone();
        let mut yn = base.clone();
        let mut yd = base.clone();
        be.exec_fused_plan(FusedPlan::Axpy { alpha }, io(&x, &mut ya, &mut []))
            .unwrap();
        lasx_rs::api::axpy(alpha, &x, &mut yb).unwrap();
        be.axpy_f32(&mut yn, &x, alpha).unwrap();
        ScalarBackend
            .exec_fused_plan(FusedPlan::Axpy { alpha }, io(&x, &mut yd, &mut []))
            .unwrap();
        let eq = if bits_eq(&ya, &yb) && bits_eq(&ya, &yn) {
            "A≡api≡命名方法 **逐位**"
        } else {
            "**不逐位！**"
        };
        // **在计时前**取数值差：Axpy 是**就地累加**，计时循环会反复把 ya/yb 再累加，事后比会产生假差。
        let dport = max_abs_diff(&ya, &yd);
        let (a, b) = ab(
            || {
                be.exec_fused_plan(FusedPlan::Axpy { alpha }, io(&x, &mut ya, &mut []))
                    .unwrap();
                black_box(ya[0]);
            },
            || {
                lasx_rs::api::axpy(alpha, &x, &mut yb).unwrap();
                black_box(yb[0]);
            },
        );
        row("Axpy（1M）", a, b, eq, dport);
    }

    // ---- 行 pattern：Softmax / RmsNorm / LayerNorm ----
    let (rows, cols) = (512usize, 2048usize);
    let n = rows * cols;
    let xs = sample(n, 5);
    let eps = 1e-5f32;
    {
        let plan = FusedPlan::SoftmaxRows { rows, cols };
        let mut oa = vec![0.0f32; n];
        let mut ob = vec![0.0f32; n];
        let mut od = vec![0.0f32; n];
        be.exec_fused_plan(plan, io(&xs, &mut [], &mut oa)).unwrap();
        lasx_rs::api::softmax_rows_into(&xs, None, rows, cols, 1.0, &mut ob).unwrap();
        ScalarBackend
            .exec_fused_plan(plan, io(&xs, &mut [], &mut od))
            .unwrap();
        let eq = if bits_eq(&oa, &ob) {
            "A≡api **逐位**"
        } else {
            "**不逐位！**"
        };
        let (a, b) = ab(
            || {
                be.exec_fused_plan(plan, io(&xs, &mut [], &mut oa)).unwrap();
                black_box(oa[0]);
            },
            || {
                lasx_rs::api::softmax_rows_into(&xs, None, rows, cols, 1.0, &mut ob).unwrap();
                black_box(ob[0]);
            },
        );
        row("SoftmaxRows（512×2048）", a, b, eq, max_abs_diff(&oa, &od));
    }
    for (name, plan) in [
        ("RmsNormRows", FusedPlan::RmsNormRows { rows, cols, eps }),
        (
            "LayerNormRows",
            FusedPlan::LayerNormRows { rows, cols, eps },
        ),
    ] {
        let mut oa = vec![0.0f32; n];
        let mut ob = vec![0.0f32; n];
        let mut od = vec![0.0f32; n];
        be.exec_fused_plan(plan, io(&xs, &mut [], &mut oa)).unwrap();
        match plan {
            FusedPlan::RmsNormRows { .. } => {
                lasx_rs::api::rms_norm_into(&xs, None, rows, cols, eps, &mut ob).unwrap()
            }
            _ => lasx_rs::api::layer_norm_into(&xs, None, None, rows, cols, eps, &mut ob).unwrap(),
        }
        ScalarBackend
            .exec_fused_plan(plan, io(&xs, &mut [], &mut od))
            .unwrap();
        let eq = if bits_eq(&oa, &ob) {
            "A≡api **逐位**"
        } else {
            "**不逐位！**"
        };
        let (a, b) = ab(
            || {
                be.exec_fused_plan(plan, io(&xs, &mut [], &mut oa)).unwrap();
                black_box(oa[0]);
            },
            || match plan {
                FusedPlan::RmsNormRows { .. } => {
                    lasx_rs::api::rms_norm_into(&xs, None, rows, cols, eps, &mut ob).unwrap()
                }
                _ => lasx_rs::api::layer_norm_into(&xs, None, None, rows, cols, eps, &mut ob)
                    .unwrap(),
            },
        );
        row(name, a, b, eq, max_abs_diff(&oa, &od));
    }

    // ---- GeluQuick ----
    {
        let xg = sample(1usize << 20, 7);
        let m = xg.len();
        let mut oa = vec![0.0f32; m];
        let mut ob = vec![0.0f32; m];
        let mut od = vec![0.0f32; m];
        be.exec_fused_plan(FusedPlan::GeluQuick, io(&xg, &mut [], &mut oa))
            .unwrap();
        lasx_rs::api::gelu_quick_into(&xg, &mut ob).unwrap();
        ScalarBackend
            .exec_fused_plan(FusedPlan::GeluQuick, io(&xg, &mut [], &mut od))
            .unwrap();
        let eq = if bits_eq(&oa, &ob) {
            "A≡api **逐位**"
        } else {
            "**不逐位！**"
        };
        let (a, b) = ab(
            || {
                be.exec_fused_plan(FusedPlan::GeluQuick, io(&xg, &mut [], &mut oa))
                    .unwrap();
                black_box(oa[0]);
            },
            || {
                lasx_rs::api::gelu_quick_into(&xg, &mut ob).unwrap();
                black_box(ob[0]);
            },
        );
        row("GeluQuick（1M）", a, b, eq, max_abs_diff(&oa, &od));
    }

    // ---- RopeApply ----
    {
        let (rr, cc, nd) = (1024usize, 128usize, 64usize);
        let total = rr * cc;
        let half = nd / 2;
        let cos: Vec<f32> = (0..rr * half)
            .map(|i| ((i * 7 % 41) as f32 * 0.03).cos())
            .collect();
        let sin: Vec<f32> = (0..rr * half)
            .map(|i| ((i * 7 % 41) as f32 * 0.03).sin())
            .collect();
        let xr = sample(total, 9);
        let plan = FusedPlan::RopeApply {
            rows: rr,
            cols: cc,
            n_dims: nd,
        };
        let mut oa = vec![0.0f32; total];
        let mut ob = vec![0.0f32; total];
        let mut od = vec![0.0f32; total];
        be.exec_fused_plan(
            plan,
            FusedIo {
                x: &xr,
                y: &mut [],
                out: &mut oa,
                cos: &cos,
                sin: &sin,
            },
        )
        .unwrap();
        lasx_rs::api::rope_into(
            &xr,
            &cos,
            &sin,
            rr,
            cc,
            nd,
            lasx_rs::api::RopeMode::NeoX,
            &mut ob,
        )
        .unwrap();
        ScalarBackend
            .exec_fused_plan(
                plan,
                FusedIo {
                    x: &xr,
                    y: &mut [],
                    out: &mut od,
                    cos: &cos,
                    sin: &sin,
                },
            )
            .unwrap();
        let eq = if bits_eq(&oa, &ob) {
            "A≡api **逐位**"
        } else {
            "**不逐位！**"
        };
        let (a, b) = ab(
            || {
                be.exec_fused_plan(
                    plan,
                    FusedIo {
                        x: &xr,
                        y: &mut [],
                        out: &mut oa,
                        cos: &cos,
                        sin: &sin,
                    },
                )
                .unwrap();
                black_box(oa[0]);
            },
            || {
                lasx_rs::api::rope_into(
                    &xr,
                    &cos,
                    &sin,
                    rr,
                    cc,
                    nd,
                    lasx_rs::api::RopeMode::NeoX,
                    &mut ob,
                )
                .unwrap();
                black_box(ob[0]);
            },
        );
        row(
            "RopeApply（1024×128, n_dims=64）",
            a,
            b,
            eq,
            max_abs_diff(&oa, &od),
        );
    }

    println!("\nloadavg（结束）= `{}`", loadavg());
    println!(
        "\n判定：**B/A ≥0.98 视为 Plan 路径不掉性能**（<0.98 如实记：薄转发噪声 vs 内核差异）。\
         Plan = `LasxBackend::exec_fused_plan`（override → LASX 融合核）；B = 直接 `api::*`（同一内核）。\
         `Axpy`/`SoftmaxRows` 的具名方法（`axpy_f32`/`softmax_rows_f32`）仍在 trait 里（薄委托）；\
         其余 4 个具名方法已按批1 删除（只剩 Plan 入口）。"
    );
}
