//! **R6-B-3：三档后端横向对比**——`LasxBackend`（手写 SIMD / LASX）vs
//! `simd_dsl::IdiomaticBackend`（可移植优化档）vs `simd_dsl::ScalarBackend`（**下限臂**）。
//!
//! 口径严格照 `simd_dsl/docs/bench.md` §5：**同进程交替**、**所有臂写同一块目的缓冲**、
//! **≥5 对且丢弃首对**、报**中位 + 极差 + `iters` + `ms_max` + `loadavg`**；
//! **正确性在计时之外断言**（先跑一遍、比完再计时）；**每行标等价类**（逐位 / 容差 + 实测界 + 参照物）。
//! **CI 不对耗时设阈值**，本 example 只在 `--features dsl-backend` 下构建。
//!
//! 运行：
//! ```text
//! # 大跑
//! cargo run --release --features dsl-backend --example lasx_vs_portable
//! # 秒级冒烟（轮数不满足 §5，输出里显式标注）
//! cargo run --release --features dsl-backend --example lasx_vs_portable -- --smoke
//! ```
//!
//! **负结果照写**：小形状上"手写 SIMD"可能被**调用/打包开销**吃掉甚至更慢；本节不做结论性断言，
//! 结论与归因见 `docs/dev.md` 的新小节（三档表 + 归因 + 负结果 + 复跑命令）。

use std::cell::RefCell;
use std::time::Instant;

use lasx_rs::dsl_backend::LasxBackend;
use simd_dsl::{
    Backend, BinaryOp, IdiomaticBackend, RowEpilogue, RowFuse, RowReduce, ScalarBackend,
};

#[derive(Clone, Copy)]
struct Timing {
    median_ms: f64,
    range_ms: f64,
    max_ms: f64,
    iters: u32,
}

fn sample<F: FnMut()>(f: &mut F, iters: u32) -> f64 {
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    t0.elapsed().as_secs_f64() * 1e3 / f64::from(iters)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("无 NaN"));
    let mid = v.len() / 2;
    if v.len().is_multiple_of(2) {
        (v[mid - 1] + v[mid]) / 2.0
    } else {
        v[mid]
    }
}

fn timing(v: &mut [f64], iters: u32) -> Timing {
    let max_ms = v.iter().copied().fold(f64::MIN, f64::max);
    let min_ms = v.iter().copied().fold(f64::MAX, f64::min);
    Timing {
        median_ms: median(v),
        range_ms: max_ms - min_ms,
        max_ms,
        iters,
    }
}

fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .map(|s| s.split_whitespace().take(3).collect::<Vec<_>>().join(" "))
        .unwrap_or_else(|_| "n/a".into())
}

/// 掩码相对误差（`simd_dsl/docs/bench.md` 的既有口径）。
fn masked_rel(got: &[f32], want: &[f32]) -> f32 {
    let peak = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0);
    let floor = 1.0e-3 * peak;
    got.iter()
        .zip(want)
        .filter(|(_, w)| w.abs() > floor)
        .fold(0.0f32, |m, (g, w)| m.max((g - w).abs() / w.abs()))
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// 一行等价类（逐位优先；否则容差 + **实测界** + **参照物**）。
fn class_line(work: &str, arm: &str, got: &[f32], want: &[f32], reference: &str) -> String {
    if bits_equal(got, want) {
        format!("{work:<24} {arm:<22} **逐位**（参照物：{reference}）")
    } else {
        format!(
            "{work:<24} {arm:<22} 容差（参照物：{reference}，**实测界 masked_rel ≤ {:.3e}**）",
            masked_rel(got, want)
        )
    }
}

type Arm<'a> = (&'a str, &'a str, Box<dyn FnMut() + 'a>);

/// 计时：同进程**交替**、每轮每臂各一次、**丢首轮**；`A` = 第一臂（手写 SIMD）作基准。
fn run_workload(
    work: &str,
    shape: &str,
    arms: &mut [Arm<'_>],
    pairs: u32,
    iters: u32,
    smoke: bool,
) {
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    for i in 0..pairs {
        for (k, (_, _, f)) in arms.iter_mut().enumerate() {
            samples[k].push(sample(f, iters));
        }
        if i == 0 {
            for v in samples.iter_mut() {
                v.clear();
            }
        }
    }
    println!(
        "\n-- {work}（{shape}；pairs {pairs}{}，iters {iters}）--",
        if smoke {
            "，**smoke：轮数不满足 §5**"
        } else {
            ""
        }
    );
    let base = timing(&mut samples[0].clone(), iters).median_ms;
    for (k, (name, note, _)) in arms.iter().enumerate() {
        let t = timing(&mut samples[k].clone(), iters);
        println!(
            "  {name:<26} 中位 {:>10.4} ms（极差 {:>9.4} / ms_max {:>10.4}）| 相对手写 {:.3}× | iters {} | {note}",
            t.median_ms,
            t.range_ms,
            t.max_ms,
            base / t.median_ms,
            t.iters
        );
    }
}

// ============================================================ 臂（同一 DSL 调用，只换后端）

fn arm_dot<B: Backend + ?Sized>(be: &B, x: &[f32], y: &[f32]) -> f32 {
    be.dot_f32(x, y).unwrap()
}

fn arm_matmul<B: Backend + ?Sized>(
    be: &B,
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
) {
    be.matmul_f32(m, k, n, a, b, out).unwrap();
}

fn arm_rows<B: Backend + ?Sized>(be: &B, x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    be.reduce_sum_rows_f32(x, rows, cols, out).unwrap();
}

/// ④ 窄三元组 `rms_norm`：`SumSq → sqrt(mean+eps) → div`（**一趟**；`docs/fusion.md` 的形态③）。
fn arm_rms<B: Backend + ?Sized>(
    be: &B,
    x: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
    out: &mut [f32],
) {
    be.reduce_rows_map_f32(
        x,
        rows,
        cols,
        out,
        RowFuse {
            reduce: RowReduce::SumSq,
            epilogue: RowEpilogue::SqrtMeanPlusEps,
            map: BinaryOp::Div,
            eps,
        },
    )
    .unwrap();
}

fn vec_of(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed) % 101) as f32) * 0.05 - 2.5)
        .collect()
}

fn main() {
    let smoke = std::env::args().any(|a| a == "--smoke");
    let (pairs, iters) = if smoke { (2u32, 1u32) } else { (5u32, 4u32) };
    let lasx = LasxBackend;
    let idio = IdiomaticBackend;
    let scalar = ScalarBackend;
    let eps = 1.0e-5f32;

    println!(
        "== R6-B-3：LasxBackend（手写 SIMD）vs IdiomaticBackend（可移植优化）vs ScalarBackend（下限）=="
    );
    println!(
        "口径：simd_dsl/docs/bench.md §5（同进程交替、同一块目的缓冲、丢首对、中位 + 极差 + ms_max）"
    );
    println!("loadavg = {}", loadavg());
    println!(
        "模式：{}；pairs {pairs}，iters {iters}{}",
        if smoke { "--smoke" } else { "大跑" },
        if smoke {
            "（**冒烟轮数不满足 §5，只验证能跑**）"
        } else {
            ""
        }
    );

    // ---------------- ① 点积 ----------------
    let dots: Vec<usize> = if smoke { vec![256] } else { vec![1024, 65536] };
    println!("\n== 等价类（**计时之外**；参照物 = ScalarBackend）==");
    for kk in dots {
        let x = vec_of(kk, 1);
        let y = vec_of(kk, 7);
        let r_scalar = arm_dot(&scalar, &x, &y);
        let r_lasx = arm_dot(&lasx, &x, &y);
        let r_idio = arm_dot(&idio, &x, &y);
        println!(
            "{}",
            class_line(
                &format!("① dot K={kk}"),
                "Lasx",
                &[r_lasx],
                &[r_scalar],
                "Scalar"
            )
        );
        println!(
            "{}",
            class_line(
                &format!("① dot K={kk}"),
                "Idiomatic",
                &[r_idio],
                &[r_scalar],
                "Scalar"
            )
        );
        if !smoke {
            let shared = RefCell::new(0.0f32);
            let mut arms: Vec<Arm<'_>> = vec![
                (
                    "LasxBackend（手写 SIMD）",
                    "LASX 内核",
                    Box::new(|| {
                        let mut s = shared.borrow_mut();
                        *s = arm_dot(&lasx, &x, &y);
                    }),
                ),
                (
                    "IdiomaticBackend",
                    "可移植优化档（`std::simd` 风格分块）",
                    Box::new(|| {
                        let mut s = shared.borrow_mut();
                        *s = arm_dot(&idio, &x, &y);
                    }),
                ),
                (
                    "ScalarBackend",
                    "**下限臂**（顺序单累加器）",
                    Box::new(|| {
                        let mut s = shared.borrow_mut();
                        *s = arm_dot(&scalar, &x, &y);
                    }),
                ),
            ];
            run_workload("① 点积", &format!("K={kk}"), &mut arms, pairs, iters, smoke);
        }
    }

    // ---------------- ② matmul ----------------
    let mats: Vec<(usize, usize, usize)> = if smoke {
        vec![(16, 16, 16)]
    } else {
        vec![(64, 64, 64), (192, 768, 3072)]
    };
    for (m, kk, n) in mats {
        let a = vec_of(m * kk, 3);
        let b = vec_of(kk * n, 5);
        let mut w_scalar = vec![0.0f32; m * n];
        arm_matmul(&scalar, m, kk, n, &a, &b, &mut w_scalar);
        let mut w_lasx = vec![0.0f32; m * n];
        arm_matmul(&lasx, m, kk, n, &a, &b, &mut w_lasx);
        let mut w_idio = vec![0.0f32; m * n];
        arm_matmul(&idio, m, kk, n, &a, &b, &mut w_idio);
        let label = format!("② matmul {m}×{kk}×{n}");
        println!(
            "{}",
            class_line(&label, "Lasx", &w_lasx, &w_scalar, "Scalar")
        );
        println!(
            "{}",
            class_line(&label, "Idiomatic", &w_idio, &w_scalar, "Scalar")
        );

        // 大形状允许减轮（如实报 iters/pairs）
        let (p2, i2) = if smoke {
            (2, 1)
        } else if m * kk * n > 100_000_000 {
            (3u32, 1u32)
        } else {
            (pairs, iters)
        };
        let shared = RefCell::new(vec![0.0f32; m * n]);
        let mut arms: Vec<Arm<'_>> = vec![
            (
                "LasxBackend（手写 SIMD）",
                "LASX 内核（含内部打包）",
                Box::new(|| {
                    let mut o = shared.borrow_mut();
                    arm_matmul(&lasx, m, kk, n, &a, &b, &mut o);
                }),
            ),
            (
                "IdiomaticBackend",
                "可移植优化档",
                Box::new(|| {
                    let mut o = shared.borrow_mut();
                    arm_matmul(&idio, m, kk, n, &a, &b, &mut o);
                }),
            ),
            (
                "ScalarBackend",
                "**下限臂**",
                Box::new(|| {
                    let mut o = shared.borrow_mut();
                    arm_matmul(&scalar, m, kk, n, &a, &b, &mut o);
                }),
            ),
        ];
        run_workload(
            "② matmul",
            &format!("{m}×{kk}×{n}（{} MFLOP·2）", m * kk * n / 1_000_000),
            &mut arms,
            p2,
            i2,
            smoke,
        );
    }

    // ---------------- ③ 轴归约（rows 方向） + ④ 窄三元组 rms_norm ----------------
    let shapes: Vec<(usize, usize)> = if smoke {
        vec![(32, 64)]
    } else {
        vec![(128, 1024), (4096, 256)]
    };
    for (rows, cols) in shapes {
        let x = vec_of(rows * cols, 11);
        let mut w_scalar = vec![0.0f32; rows];
        arm_rows(&scalar, &x, rows, cols, &mut w_scalar);
        let mut w_lasx = vec![0.0f32; rows];
        arm_rows(&lasx, &x, rows, cols, &mut w_lasx);
        let mut w_idio = vec![0.0f32; rows];
        arm_rows(&idio, &x, rows, cols, &mut w_idio);
        let label = format!("③ reduce_sum_rows {rows}×{cols}");
        println!(
            "{}",
            class_line(&label, "Lasx", &w_lasx, &w_scalar, "Scalar")
        );
        println!(
            "{}",
            class_line(&label, "Idiomatic", &w_idio, &w_scalar, "Scalar")
        );

        let shared = RefCell::new(vec![0.0f32; rows]);
        let mut arms: Vec<Arm<'_>> = vec![
            (
                "LasxBackend（手写 SIMD）",
                "轴归约（rows）",
                Box::new(|| {
                    let mut o = shared.borrow_mut();
                    arm_rows(&lasx, &x, rows, cols, &mut o);
                }),
            ),
            (
                "IdiomaticBackend",
                "可移植优化档",
                Box::new(|| {
                    let mut o = shared.borrow_mut();
                    arm_rows(&idio, &x, rows, cols, &mut o);
                }),
            ),
            (
                "ScalarBackend",
                "**下限臂**",
                Box::new(|| {
                    let mut o = shared.borrow_mut();
                    arm_rows(&scalar, &x, rows, cols, &mut o);
                }),
            ),
        ];
        run_workload(
            "③ 轴归约 rows",
            &format!("{rows}×{cols}"),
            &mut arms,
            pairs,
            iters,
            smoke,
        );

        // ④ rms_norm（窄三元组，一趟）
        let mut w_scalar = vec![0.0f32; rows * cols];
        arm_rms(&scalar, &x, rows, cols, eps, &mut w_scalar);
        let mut w_lasx = vec![0.0f32; rows * cols];
        arm_rms(&lasx, &x, rows, cols, eps, &mut w_lasx);
        let mut w_idio = vec![0.0f32; rows * cols];
        arm_rms(&idio, &x, rows, cols, eps, &mut w_idio);
        let label = format!("④ rms_norm {rows}×{cols}");
        println!(
            "{}",
            class_line(&label, "Lasx", &w_lasx, &w_scalar, "Scalar")
        );
        println!(
            "{}",
            class_line(&label, "Idiomatic", &w_idio, &w_scalar, "Scalar")
        );

        let shared = RefCell::new(vec![0.0f32; rows * cols]);
        let mut arms: Vec<Arm<'_>> = vec![
            (
                "LasxBackend（手写 SIMD）",
                "`reduce_rows_map_f32`（SumSq→sqrt(mean+eps)→div）",
                Box::new(|| {
                    let mut o = shared.borrow_mut();
                    arm_rms(&lasx, &x, rows, cols, eps, &mut o);
                }),
            ),
            (
                "IdiomaticBackend",
                "同一计划、可移植档",
                Box::new(|| {
                    let mut o = shared.borrow_mut();
                    arm_rms(&idio, &x, rows, cols, eps, &mut o);
                }),
            ),
            (
                "ScalarBackend",
                "**下限臂**（顺序行统计量）",
                Box::new(|| {
                    let mut o = shared.borrow_mut();
                    arm_rms(&scalar, &x, rows, cols, eps, &mut o);
                }),
            ),
        ];
        run_workload(
            "④ 窄三元组 rms_norm",
            &format!("{rows}×{cols}"),
            &mut arms,
            pairs,
            iters,
            smoke,
        );
    }

    println!(
        "\n== 装置只出数，不做结论性断言；三档归因与**负结果**（手写 SIMD 在哪些形状不快）见 `docs/dev.md` 的新小节。=="
    );
}
