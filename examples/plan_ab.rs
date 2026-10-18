//! **P3 Plan 试点后端 A/B（task-63）**：`FusedPlan` 路径 vs 命名方法 vs 直接 `api::*`。
//!
//! 臂：
//! * **A = Plan 路**：`LasxBackend::exec_fused_plan(FusedPlan::…)`（本仓 override → LASX 融合核）；
//! * **B = 命名方法**：`axpy_f32` / `softmax_rows_f32`（本仓 override，薄委托同一内核）；
//! * **C = 直接 api**：`api::axpy` / `api::softmax_rows_into`（不经 DSL 层）；
//! * **D = portable 计划解释**：`ScalarBackend::exec_fused_plan`（数值参照，慢，不参与性能比）。
//!
//! 判据：**A ≡ B ≡ C 逐位**（axpy 与 softmax 都是同一内核）；A vs D 容差（axpy 逐位/softmax ≤1e-5 实测界）。
//! 性能：同进程交替 6 对丢首对取中位；**目标 ≥0.98**（Plan 路径不能因抽象掉性能）；
//! 若 <0.98 如实记并说明原因（薄转发噪声 vs 内核差异）。
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
/// 同进程交替（A/B），丢首对取中位；返回 (A ms, B ms)。
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

fn main() {
    let be = LasxBackend;
    println!("# P3 Plan 后端 A/B：`exec_fused_plan` vs 命名方法 vs 直接 api（后端 = LasxBackend）");
    println!(
        "- threads = {}；loadavg = `{}`（同进程交替 {ROUNDS} 对丢首对取中位）",
        lasx_rs::pool::global().threads(),
        loadavg()
    );
    println!(
        "- `supports_plan`：LASX Axpy={} SoftmaxRows={}；portable Axpy={} SoftmaxRows={}（如实）",
        be.supports_plan(PatternKind::Axpy),
        be.supports_plan(PatternKind::SoftmaxRows),
        ScalarBackend.supports_plan(PatternKind::Axpy),
        ScalarBackend.supports_plan(PatternKind::SoftmaxRows)
    );
    println!("| 用例 | A=Plan ms | B=命名 ms | C=api ms | B/A | C/A | 等价类 |");
    println!("|---|---|---|---|---|---|---|");

    // ---------------- Axpy：n = 1M ----------------
    {
        let n = 1usize << 20;
        let x = sample(n, 3);
        let base = sample(n, 11);
        let alpha = 0.375f32;
        let mut ya = base.clone();
        let mut yb = base.clone();
        let mut yc = base.clone();
        let mut yd = base.clone();
        be.exec_fused_plan(
            FusedPlan::Axpy { alpha },
            FusedIo {
                x: &x,
                y: &mut ya,
                out: &mut [],
            },
        )
        .unwrap();
        be.axpy_f32(&mut yb, &x, alpha).unwrap();
        lasx_rs::api::axpy(alpha, &x, &mut yc).unwrap();
        ScalarBackend
            .exec_fused_plan(
                FusedPlan::Axpy { alpha },
                FusedIo {
                    x: &x,
                    y: &mut yd,
                    out: &mut [],
                },
            )
            .unwrap();
        let eq = if bits_eq(&ya, &yb) && bits_eq(&ya, &yc) {
            "A≡B≡C **逐位**"
        } else {
            "**不逐位！**"
        };
        let d = max_abs_diff(&ya, &yd);
        let (a, b) = ab(
            || {
                be.exec_fused_plan(
                    FusedPlan::Axpy { alpha },
                    FusedIo {
                        x: &x,
                        y: &mut ya,
                        out: &mut [],
                    },
                )
                .unwrap();
                black_box(ya[0]);
            },
            || {
                be.axpy_f32(&mut yb, &x, alpha).unwrap();
                black_box(yb[0]);
            },
        );
        let (a2, c) = ab(
            || {
                be.exec_fused_plan(
                    FusedPlan::Axpy { alpha },
                    FusedIo {
                        x: &x,
                        y: &mut ya,
                        out: &mut [],
                    },
                )
                .unwrap();
                black_box(ya[0]);
            },
            || {
                lasx_rs::api::axpy(alpha, &x, &mut yc).unwrap();
                black_box(yc[0]);
            },
        );
        println!(
            "| Axpy（1M） | {a:.3} | {b:.3} | {c:.3} | {:.3}× | {:.3}× | {eq}（vs portable 计划 max|d|={d:.2e}） |",
            b / a,
            c / a2
        );
    }

    // ---------------- SoftmaxRows：512 × 2048 ----------------
    {
        let (rows, cols) = (512usize, 2048usize);
        let n = rows * cols;
        let x = sample(n, 5);
        let plan = FusedPlan::SoftmaxRows { rows, cols };
        let mut outa = vec![0.0f32; n];
        let mut outb = vec![0.0f32; n];
        let mut outc = vec![0.0f32; n];
        let mut outd = vec![0.0f32; n];
        be.exec_fused_plan(
            plan,
            FusedIo {
                x: &x,
                y: &mut [],
                out: &mut outa,
            },
        )
        .unwrap();
        be.softmax_rows_f32(&x, rows, cols, &mut outb).unwrap();
        lasx_rs::api::softmax_rows_into(&x, None, rows, cols, 1.0, &mut outc).unwrap();
        ScalarBackend
            .exec_fused_plan(
                plan,
                FusedIo {
                    x: &x,
                    y: &mut [],
                    out: &mut outd,
                },
            )
            .unwrap();
        let eq = if bits_eq(&outa, &outb) && bits_eq(&outa, &outc) {
            "A≡B≡C **逐位**"
        } else {
            "**不逐位！**"
        };
        let d = max_abs_diff(&outa, &outd);
        let (a, b) = ab(
            || {
                be.exec_fused_plan(
                    plan,
                    FusedIo {
                        x: &x,
                        y: &mut [],
                        out: &mut outa,
                    },
                )
                .unwrap();
                black_box(outa[0]);
            },
            || {
                be.softmax_rows_f32(&x, rows, cols, &mut outb).unwrap();
                black_box(outb[0]);
            },
        );
        let (a2, c) = ab(
            || {
                be.exec_fused_plan(
                    plan,
                    FusedIo {
                        x: &x,
                        y: &mut [],
                        out: &mut outa,
                    },
                )
                .unwrap();
                black_box(outa[0]);
            },
            || {
                lasx_rs::api::softmax_rows_into(&x, None, rows, cols, 1.0, &mut outc).unwrap();
                black_box(outc[0]);
            },
        );
        println!(
            "| SoftmaxRows（512×2048） | {a:.3} | {b:.3} | {c:.3} | {:.3}× | {:.3}× | {eq}（vs portable 计划 max|d|={d:.2e}） |",
            b / a,
            c / a2
        );
    }

    println!("\nloadavg（结束）= `{}`", loadavg());
    println!(
        "\n判定：B/A、C/A **≥0.98 视为 Plan 路径不掉性能**（<0.98 如实记：薄转发噪声 vs 内核差异）。\
         Plan 路 = `LasxBackend::exec_fused_plan`（override → `api::axpy`/`api::softmax_rows_into`）；\
         命名方法 = `axpy_f32`/`softmax_rows_f32`（同一内核）；api = 直调。"
    );
}
