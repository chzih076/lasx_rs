//! **B1/B1b（M5）组合原语 A/B：LASX override vs portable 默认**。
//!
//! 两节：
//! 1. `axpy_inplace_f32`（B1）：LASX `api::axpy` vs portable 默认（**逐位**）；
//! 2. **组合路径改前/改后（B1b）**：`ops_dsl::{rms_norm, layer_norm, softmax_rows}` 在
//!    `LASX_NO_PLAN=1`（同一 Lasx 后端走**生产组合回退**）下：
//!    A = `ScalarBackend`（**改前**的等价物：旧 Lasx 在这两个原语上就是 portable 标量默认），
//!    B = `LasxBackend` + `LASX_NO_PLAN=1`（**改后**，命中 B1b 的 `map_unary_f32`/
//!    `reduce_rows_map_f32` override），
//!    F = `LasxBackend` 默认（融合路径，仅作参考）。
//!
//! **口径**：同进程轮转（丢首对、中位），逐轮 loadavg + A/B/C 分级；
//! **打印 threads**（`init_global(1|12)`）；**没收益如实写**。
//! 用法：`cargo run --release --example prims_ab --features dsl-backend -- 1`（或 `12`）。
//! 绝对 ms 只在同一次运行内可比（跨窗口/跨线程配置不可横比）。
//!
//! **已知偏差（如实）**：A 臂用 `ScalarBackend` 近似"改前"，而旧 Lasx 回退的 softmax 里
//! 有 1 个调用（`reduce_sum_f32`）是 LASX 原生核 ⇒ A 臂比旧值略慢（只会低估改后收益方向）。

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

fn loadavg() -> (f64, f64) {
    let s = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    let mut it = s.split_whitespace();
    (
        it.next().and_then(|x| x.parse().ok()).unwrap_or(99.0),
        it.next().and_then(|x| x.parse().ok()).unwrap_or(99.0),
    )
}

fn grade(l1: f64, l5: f64) -> char {
    if l1 <= 3.0 && l5 <= 3.0 {
        'A'
    } else if l1 <= 4.5 {
        'B'
    } else {
        'C'
    }
}

fn loadavg_tag() -> String {
    let (l1, l5) = loadavg();
    format!("{l1:.2}/{l5:.2} grade={}", grade(l1, l5))
}

/// 设置 `LASX_NO_PLAN`（组合路径）。`std::env::set_var` 在 edition 2024 起是 `unsafe`。
///
/// # Safety（调用方约定）
/// 本探针里**只有主线程**会调 `no_plan_hook()`（`Backend` 方法均由测量循环在本线程调用）；
/// 池线程只做数值内核、不读环境变量。计时窗口内不会有其它线程 `getenv`，因此这里的
/// `set_var/remove_var` 不构成数据竞争。（这是基准探针的约定，不是库内可复制的模式。）
fn set_no_plan() {
    // SAFETY: 见函数文档"调用方约定"——仅主线程读该变量、池线程不读。
    unsafe { std::env::set_var("LASX_NO_PLAN", "1") }
}

/// 清除 `LASX_NO_PLAN`（回到融合路径）。
fn clear_no_plan() {
    // SAFETY: 同 [`set_no_plan`]。
    unsafe { std::env::remove_var("LASX_NO_PLAN") }
}

fn sample(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed) % 101) as f32) * 0.05 - 2.5)
        .collect()
}

fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// 两臂交替（A/B），返回 (A 中位 ms, B 中位 ms)。
fn ab(mut a: impl FnMut(), mut b: impl FnMut()) -> (f64, f64) {
    let (mut av, mut bv) = (Vec::new(), Vec::new());
    for r in 0..=ROUNDS {
        let (ta, tb) = if r % 2 == 0 {
            (timeit(&mut a), timeit(&mut b))
        } else {
            let tb = timeit(&mut b);
            let ta = timeit(&mut a);
            (ta, tb)
        };
        if r > 0 {
            av.push(ta);
            bv.push(tb);
        }
    }
    (
        median(av).as_secs_f64() * 1e3,
        median(bv).as_secs_f64() * 1e3,
    )
}

/// 三臂轮转（A/B/F），返回三臂中位 ms。
fn ab3(mut a: impl FnMut(), mut b: impl FnMut(), mut f: impl FnMut()) -> (f64, f64, f64) {
    let orders: [(u8, u8, u8); 5] = [(0, 1, 2), (2, 1, 0), (1, 2, 0), (0, 2, 1), (2, 0, 1)];
    let (mut av, mut bv, mut fv) = (Vec::new(), Vec::new(), Vec::new());
    for (i, ord) in orders.iter().enumerate() {
        let (mut ta, mut tb, mut tf) = (Duration::ZERO, Duration::ZERO, Duration::ZERO);
        for pick in [ord.0, ord.1, ord.2] {
            match pick {
                0 => ta = timeit(&mut a),
                1 => tb = timeit(&mut b),
                _ => tf = timeit(&mut f),
            }
        }
        if i > 0 {
            av.push(ta);
            bv.push(tb);
            fv.push(tf);
        }
    }
    (
        median(av).as_secs_f64() * 1e3,
        median(bv).as_secs_f64() * 1e3,
        median(fv).as_secs_f64() * 1e3,
    )
}

fn main() {
    let threads: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let installed = lasx_rs::pool::init_global(threads);
    let be = LasxBackend;
    println!("# M5 组合原语 A/B（B1: axpy + B1b: map_unary/reduce_rows_map）");
    println!(
        "- threads = {}（init_global({threads}) = {installed}）；loadavg = `{}`；轮次 {ROUNDS} 对丢首对中位",
        lasx_rs::pool::global().threads(),
        loadavg_tag()
    );

    // ---------------- 1) axpy_inplace_f32（B1：逐位，核效率对照） ----------------
    println!("\n## 1) axpy_inplace_f32：A=Portable 默认 vs B=LASX override");
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
        println!("   —— loadavg `{}`", loadavg_tag());
    }

    // ---------------- 2) 组合路径改前/改后（B1b：ops_dsl 生产组合回退） ----------------
    println!(
        "\n## 2) 组合路径（LASX_NO_PLAN=1）：A=改前(ScalarBackend) / B=改后(LASX prims) / F=融合参考"
    );
    println!(
        "| 用例 | A=改前 ms | B=改后 ms | F=融合 ms | A/B（>1 ⇒ 改后更快） | B/F | 等价类/说明 |"
    );
    println!("|---|---|---|---|---|---|---|");
    for &(rows, cols) in &[(1024usize, 1024usize), (4096usize, 128usize)] {
        let n = rows * cols;
        let x = sample(n, 5);
        let xp: Vec<f32> = x.iter().map(|v| v.abs() + 0.05).collect();

        // rms_norm：组合回退 = fuse!(sum_sq, sqrt_mean_plus_eps, div) ⇒ 命中两个新 override。
        {
            let eps = 1e-6f32;
            let mut sq_a = vec![0.0f32; n];
            let mut sq_b = vec![0.0f32; n];
            let mut sq_f = vec![0.0f32; n];
            let mut st_a = vec![0.0f32; rows];
            let mut st_b = vec![0.0f32; rows];
            let mut st_f = vec![0.0f32; rows];
            let mut inv_a = vec![0.0f32; rows];
            let mut inv_b = vec![0.0f32; rows];
            let mut inv_f = vec![0.0f32; rows];
            let mut oa = vec![0.0f32; n];
            let mut ob = vec![0.0f32; n];
            let mut of = vec![0.0f32; n];
            ops_dsl::norm::rms_norm(
                &ScalarBackend,
                &xp,
                rows,
                cols,
                eps,
                &mut sq_a,
                &mut st_a,
                &mut inv_a,
                &mut oa,
            )
            .unwrap();
            set_no_plan();
            ops_dsl::norm::rms_norm(
                &be, &xp, rows, cols, eps, &mut sq_b, &mut st_b, &mut inv_b, &mut ob,
            )
            .unwrap();
            clear_no_plan();
            ops_dsl::norm::rms_norm(
                &be, &xp, rows, cols, eps, &mut sq_f, &mut st_f, &mut inv_f, &mut of,
            )
            .unwrap();
            let eq = format!("A≡B max_abs={:.2e}", max_abs(&oa, &ob));
            let (a, b, f) = ab3(
                || {
                    ops_dsl::norm::rms_norm(
                        &ScalarBackend,
                        &xp,
                        rows,
                        cols,
                        eps,
                        &mut sq_a,
                        &mut st_a,
                        &mut inv_a,
                        &mut oa,
                    )
                    .unwrap();
                },
                || {
                    set_no_plan();
                    ops_dsl::norm::rms_norm(
                        &be, &xp, rows, cols, eps, &mut sq_b, &mut st_b, &mut inv_b, &mut ob,
                    )
                    .unwrap();
                    clear_no_plan();
                },
                || {
                    clear_no_plan();
                    ops_dsl::norm::rms_norm(
                        &be, &xp, rows, cols, eps, &mut sq_f, &mut st_f, &mut inv_f, &mut of,
                    )
                    .unwrap();
                },
            );
            println!(
                "| rms_norm {rows}×{cols}（组合） | {a:.3} | {b:.3} | {f:.3} | {:.3}× | {:.3}× | {eq} |",
                a / b,
                b / f
            );
            println!("   —— loadavg `{}`", loadavg_tag());
        }

        // layer_norm：组合回退 = 2×fuse! ⇒ 命中两个新 override。
        {
            let eps = 1e-6f32;
            let mut d_a = vec![0.0f32; n];
            let mut d_b = vec![0.0f32; n];
            let mut d_f = vec![0.0f32; n];
            let mut st_a = vec![0.0f32; rows];
            let mut st_b = vec![0.0f32; rows];
            let mut st_f = vec![0.0f32; rows];
            let mut inv_a = vec![0.0f32; rows];
            let mut inv_b = vec![0.0f32; rows];
            let mut inv_f = vec![0.0f32; rows];
            let mut oa = vec![0.0f32; n];
            let mut ob = vec![0.0f32; n];
            let mut of = vec![0.0f32; n];
            ops_dsl::norm::layer_norm(
                &ScalarBackend,
                &x,
                rows,
                cols,
                eps,
                &mut d_a,
                &mut st_a,
                &mut inv_a,
                &mut oa,
            )
            .unwrap();
            set_no_plan();
            ops_dsl::norm::layer_norm(
                &be, &x, rows, cols, eps, &mut d_b, &mut st_b, &mut inv_b, &mut ob,
            )
            .unwrap();
            clear_no_plan();
            ops_dsl::norm::layer_norm(
                &be, &x, rows, cols, eps, &mut d_f, &mut st_f, &mut inv_f, &mut of,
            )
            .unwrap();
            let eq = format!("A≡B max_abs={:.2e}", max_abs(&oa, &ob));
            let (a, b, f) = ab3(
                || {
                    ops_dsl::norm::layer_norm(
                        &ScalarBackend,
                        &x,
                        rows,
                        cols,
                        eps,
                        &mut d_a,
                        &mut st_a,
                        &mut inv_a,
                        &mut oa,
                    )
                    .unwrap();
                },
                || {
                    set_no_plan();
                    ops_dsl::norm::layer_norm(
                        &be, &x, rows, cols, eps, &mut d_b, &mut st_b, &mut inv_b, &mut ob,
                    )
                    .unwrap();
                    clear_no_plan();
                },
                || {
                    clear_no_plan();
                    ops_dsl::norm::layer_norm(
                        &be, &x, rows, cols, eps, &mut d_f, &mut st_f, &mut inv_f, &mut of,
                    )
                    .unwrap();
                },
            );
            println!(
                "| layer_norm {rows}×{cols}（组合） | {a:.3} | {b:.3} | {f:.3} | {:.3}× | {:.3}× | {eq} |",
                a / b,
                b / f
            );
            println!("   —— loadavg `{}`", loadavg_tag());
        }

        // softmax_rows：组合回退 = rewrite_softmax_rows（逐行 5 个后端调用；Exp 命中新 override）。
        {
            let mut st_a = vec![0.0f32; rows];
            let mut st_b = vec![0.0f32; rows];
            let mut st_f = vec![0.0f32; rows];
            let mut sh_a = vec![0.0f32; n];
            let mut sh_b = vec![0.0f32; n];
            let mut sh_f = vec![0.0f32; n];
            let mut ex_a = vec![0.0f32; n];
            let mut ex_b = vec![0.0f32; n];
            let mut ex_f = vec![0.0f32; n];
            let mut oa = vec![0.0f32; n];
            let mut ob = vec![0.0f32; n];
            let mut of = vec![0.0f32; n];
            ops_dsl::norm::softmax_rows(
                &ScalarBackend,
                &x,
                rows,
                cols,
                &mut st_a,
                &mut sh_a,
                &mut ex_a,
                &mut oa,
            )
            .unwrap();
            set_no_plan();
            ops_dsl::norm::softmax_rows(
                &be, &x, rows, cols, &mut st_b, &mut sh_b, &mut ex_b, &mut ob,
            )
            .unwrap();
            clear_no_plan();
            ops_dsl::norm::softmax_rows(
                &be, &x, rows, cols, &mut st_f, &mut sh_f, &mut ex_f, &mut of,
            )
            .unwrap();
            let eq = format!("A≡B max_abs={:.2e}", max_abs(&oa, &ob));
            let (a, b, f) = ab3(
                || {
                    ops_dsl::norm::softmax_rows(
                        &ScalarBackend,
                        &x,
                        rows,
                        cols,
                        &mut st_a,
                        &mut sh_a,
                        &mut ex_a,
                        &mut oa,
                    )
                    .unwrap();
                },
                || {
                    set_no_plan();
                    ops_dsl::norm::softmax_rows(
                        &be, &x, rows, cols, &mut st_b, &mut sh_b, &mut ex_b, &mut ob,
                    )
                    .unwrap();
                    clear_no_plan();
                },
                || {
                    clear_no_plan();
                    ops_dsl::norm::softmax_rows(
                        &be, &x, rows, cols, &mut st_f, &mut sh_f, &mut ex_f, &mut of,
                    )
                    .unwrap();
                },
            );
            println!(
                "| softmax_rows {rows}×{cols}（组合） | {a:.3} | {b:.3} | {f:.3} | {:.3}× | {:.3}× | {eq} |",
                a / b,
                b / f
            );
            println!("   —— loadavg `{}`", loadavg_tag());
        }
    }

    println!("\nloadavg（结束）= `{}`", loadavg_tag());
}
