//! **迁移 A/B 统一装置（阶段3 批1 / IS-9，`plan_lasx_to_ops.md` §4）**：
//! 旧 `lasx_rs::api::*` vs 新 `ops_dsl::*`（经 [`LasxBackend`]）**同进程交替**、≥5 轮丢首轮、取中位。
//!
//! # 每行强制信息（缺一条，"平局"就不可解释）
//! * `threads`：`lasx_rs::pool::global().threads()`；
//! * `work`：matmul 为 `m·k·n`，并标注是否越过 `PAR_MIN_WORK = 32M`（§7.28）——越过时**内核自己**
//!   走全局池，两臂的线程数可能不同；
//! * `loadavg`：`/proc/loadavg` 三值（跑前/跑后各打一次）。
//!
//! # 判据
//! `ratio = api_ms / ops_ms`：**≥ 0.98 通过**（不慢于 2%，在 5 轮中位的噪声带内）；
//! `< 0.98` ⇒ 优化或如实记 `BelowThreshold` + 归因。**本装置只报数，不做 CI 阈值**。
//!
//! # 等价类（先逐位，再谈性能）
//! 每一对先 `to_bits()` 比较；不逐位 ⇒ 立刻算 `max abs` / `max rel`（相对峰值）并打印，
//! 供 `docs/dev.md` 的等价类登记（**禁止未测就标容差**）。
//!
//! # 本批覆盖 vs 待补（**如实**）
//! 当前 `simd_dsl` pin（`4e97bdb`）的 `ops_dsl` 只有 `matmul_f32`/`matmul_const`/`dot_f32`/
//! `sum_f32`/`weighted_sum_f32` ⇒ 本装置先覆盖这 4 类"迁移 A/B"，外加 **IS-1/IS-2 的接线前后对照**
//! （A 臂 = 接线前的标量默认实现，B 臂 = [`LasxBackend`] 的 override）。
//! 批1 的其余新算子（batch/NT 包装、轴归约、索引/跨步）在 `simd_dsl` rev bump 后按同一模板加行。
//!
//! 用法（`--release` 才有效；`CARGO` 别并发跑别的编译任务，会污染计时）：
//! ```text
//! cargo run --release --features dsl-backend --example migrate_ab
//! ```

use std::hint::black_box;
use std::time::{Duration, Instant};

use lasx_rs::dsl_backend::LasxBackend;
use simd_dsl::kernels::scalar;
use simd_dsl::{Backend, ScalarKind};

/// 与 `ops::matmul` 的并行阈值同值（**只用于打印"是否越过"**，不参与计算）。
const PAR_MIN_WORK: u128 = 32 * 1024 * 1024;
/// ≥5 对；**丢首对**后剩 4 对取中位。
const ROUNDS: usize = 6;

fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "<unreadable>".to_string())
}

/// 单次计时的最小单元。
fn timeit(f: &mut impl FnMut()) -> Duration {
    let t = Instant::now();
    f();
    t.elapsed()
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort_unstable();
    v[v.len() / 2]
}

/// **同进程交替 A/B**：偶数对 A→B、奇数对 B→A；**丢首对**；返回中位。
fn ab(mut a: impl FnMut(), mut b: impl FnMut()) -> (Duration, Duration) {
    let mut a_ms = Vec::with_capacity(ROUNDS);
    let mut b_ms = Vec::with_capacity(ROUNDS);
    for round in 0..ROUNDS {
        let (ta, tb) = if round % 2 == 0 {
            let ta = timeit(&mut a);
            let tb = timeit(&mut b);
            (ta, tb)
        } else {
            let tb = timeit(&mut b);
            let ta = timeit(&mut a);
            (ta, tb)
        };
        a_ms.push(ta);
        b_ms.push(tb);
    }
    a_ms.remove(0);
    b_ms.remove(0);
    (median(a_ms), median(b_ms))
}

/// 逐位/容差判定（f32）；返回 `(逐位, max_abs, max_rel)`。
fn equiv_f32(got: &[f32], want: &[f32]) -> (bool, f64, f64) {
    assert_eq!(got.len(), want.len(), "A/B 输出长度必须一致");
    let same = got
        .iter()
        .zip(want)
        .all(|(g, w)| g.to_bits() == w.to_bits());
    let peak = want
        .iter()
        .fold(0.0f64, |p, v| p.max(v.abs() as f64))
        .max(1e-30);
    let abs = got
        .iter()
        .zip(want)
        .fold(0.0f64, |w, (g, x)| w.max((*g as f64 - *x as f64).abs()));
    (same, abs, abs / peak)
}

/// 逐位/容差判定（f64）。
fn equiv_f64(got: &[f64], want: &[f64]) -> (bool, f64, f64) {
    assert_eq!(got.len(), want.len(), "A/B 输出长度必须一致");
    let same = got
        .iter()
        .zip(want)
        .all(|(g, w)| g.to_bits() == w.to_bits());
    let peak = want.iter().fold(0.0f64, |p, v| p.max(v.abs())).max(1e-30);
    let abs = got
        .iter()
        .zip(want)
        .fold(0.0f64, |w, (g, x)| w.max((*g - *x).abs()));
    (same, abs, abs / peak)
}

fn class_f32(eq: (bool, f64, f64)) -> String {
    let (same, abs, rel) = eq;
    if same {
        "**逐位**".to_string()
    } else {
        format!("容差（abs≤{abs:.2e} / rel≤{rel:.2e}）")
    }
}
fn class_f64(eq: (bool, f64, f64)) -> String {
    let (same, abs, rel) = eq;
    if same {
        "**逐位**".to_string()
    } else {
        format!("容差（abs≤{abs:.2e} / rel≤{rel:.2e}）")
    }
}

/// 一行结果（`work = None` 表示非 matmul，不适用并行阈值）。
struct Row {
    name: &'static str,
    threads: usize,
    work: Option<u128>,
    a_ms: f64,
    b_ms: f64,
    class: String,
    note: &'static str,
}

fn gflops(work: u128, ms: f64) -> f64 {
    // matmul: 2·m·k·n / t
    2.0 * work as f64 / (ms / 1000.0) / 1e9
}

fn print_rows(rows: &[Row], title: &str) {
    println!("\n### {title}\n");
    println!(
        "| 用例 | threads | work | A(api/旧) ms | B(ops/新) ms | ratio A/B | GFLOPS(B) | 等价类 | 备注 |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    for r in rows {
        let ratio = r.a_ms / r.b_ms;
        let work = match r.work {
            Some(w) => format!(
                "{:.2e}{}",
                w as f64,
                if w >= PAR_MIN_WORK {
                    "（越阈值⇒并行）"
                } else {
                    ""
                }
            ),
            None => "—".to_string(),
        };
        let gf = r
            .work
            .map(|w| format!("{:.1}", gflops(w, r.b_ms)))
            .unwrap_or_else(|| "—".to_string());
        println!(
            "| {} | {} | {} | {:.4} | {:.4} | {:.3} | {} | {} | {} |",
            r.name, r.threads, work, r.a_ms, r.b_ms, ratio, gf, r.class, r.note
        );
    }
}

fn matmul_case(name: &'static str, m: usize, k: usize, n: usize) -> Row {
    let (be, threads) = (LasxBackend, lasx_rs::pool::global().threads());
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i * 37 % 101) as f32) * 0.01 - 0.5)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i * 53 % 97) as f32) * 0.01 + 0.25)
        .collect();
    let mut out = vec![0.0f32; m * n];

    // 等价类：先各跑一次，`to_bits()` 比较
    let api = lasx_rs::api::matmul(m, k, n, &a, &b).expect("api::matmul");
    ops_dsl::matmul::matmul_f32(&be, m, k, n, &a, &b, &mut out).expect("ops matmul_f32");
    let class = class_f32(equiv_f32(&out, api.as_slice()));

    let (da, db) = ab(
        || {
            let v = lasx_rs::api::matmul(m, k, n, &a, &b).expect("api::matmul");
            black_box(v.as_slice()[0]);
        },
        || {
            ops_dsl::matmul::matmul_f32(&be, m, k, n, &a, &b, &mut out).expect("ops matmul_f32");
            black_box(out[0]);
        },
    );
    Row {
        name,
        threads,
        work: Some(m as u128 * k as u128 * n as u128),
        a_ms: da.as_secs_f64() * 1e3,
        b_ms: db.as_secs_f64() * 1e3,
        class,
        note: "A 每轮分配输出；B 复用缓冲",
    }
}

fn matmul_const_case(name: &'static str, m: usize, k: usize, n: usize) -> Row {
    // const 形状入口要求 const 泛型 ⇒ 固定一组形状（64³），与运行期入口对照。
    let _ = (m, k, n);
    let (be, threads) = (LasxBackend, lasx_rs::pool::global().threads());
    const M: usize = 64;
    const K: usize = 64;
    const N: usize = 64;
    let a: Vec<f32> = (0..M * K)
        .map(|i| ((i * 37 % 101) as f32) * 0.01 - 0.5)
        .collect();
    let b: Vec<f32> = (0..K * N)
        .map(|i| ((i * 53 % 97) as f32) * 0.01 + 0.25)
        .collect();
    let mut out = vec![0.0f32; M * N];

    let api = lasx_rs::api::matmul(M, K, N, &a, &b).expect("api::matmul");
    ops_dsl::matmul::matmul_const::<M, K, N, _>(&be, &a, &b, &mut out).expect("ops matmul_const");
    let class = class_f32(equiv_f32(&out, api.as_slice()));
    let (da, db) = ab(
        || {
            let v = lasx_rs::api::matmul(M, K, N, &a, &b).expect("api::matmul");
            black_box(v.as_slice()[0]);
        },
        || {
            ops_dsl::matmul::matmul_const::<M, K, N, _>(&be, &a, &b, &mut out)
                .expect("ops matmul_const");
            black_box(out[0]);
        },
    );
    Row {
        name,
        threads,
        work: Some(M as u128 * K as u128 * N as u128),
        a_ms: da.as_secs_f64() * 1e3,
        b_ms: db.as_secs_f64() * 1e3,
        class,
        note: "const 形状：A 运行期入口 / B 编译期入口",
    }
}

fn dot_case(name: &'static str, n: usize) -> Row {
    let (be, threads) = (LasxBackend, lasx_rs::pool::global().threads());
    let x: Vec<f32> = (0..n)
        .map(|i| ((i * 11 % 37) as f32) * 0.01 - 0.2)
        .collect();
    let y: Vec<f32> = (0..n).map(|i| ((i * 7 % 53) as f32) * 0.01 + 0.1).collect();

    let a = lasx_rs::api::dot(&x, &y).expect("api::dot");
    let b = ops_dsl::reduce::dot_f32(&be, &x, &y);
    let class = if a.to_bits() == b.to_bits() {
        "**逐位**".to_string()
    } else {
        format!(
            "容差（abs={:.2e} / rel={:.2e}）",
            (a - b).abs(),
            (a - b).abs() as f64 / a.abs().max(1e-30) as f64
        )
    };
    let (da, db) = ab(
        || {
            black_box(lasx_rs::api::dot(&x, &y).expect("api::dot"));
        },
        || {
            black_box(ops_dsl::reduce::dot_f32(&be, &x, &y));
        },
    );
    Row {
        name,
        threads,
        work: None,
        a_ms: da.as_secs_f64() * 1e3,
        b_ms: db.as_secs_f64() * 1e3,
        class,
        note: "整片归约",
    }
}

fn sum_case(name: &'static str, n: usize) -> Row {
    let (be, threads) = (LasxBackend, lasx_rs::pool::global().threads());
    let x: Vec<f32> = (0..n)
        .map(|i| ((i * 11 % 37) as f32) * 0.01 - 0.2)
        .collect();

    let a = lasx_rs::api::sum(&x);
    let b = ops_dsl::reduce::sum_f32(&be, &x);
    // 归约是**容差类**（参照 = 多累加器）；这里如实给界。
    let class = if a.to_bits() == b.to_bits() {
        "**逐位**".to_string()
    } else {
        format!(
            "容差（abs={:.2e} / rel={:.2e}）",
            (a - b).abs(),
            (a - b).abs() as f64 / a.abs().max(1e-30) as f64
        )
    };
    let (da, db) = ab(
        || {
            black_box(lasx_rs::api::sum(&x));
        },
        || {
            black_box(ops_dsl::reduce::sum_f32(&be, &x));
        },
    );
    Row {
        name,
        threads,
        work: None,
        a_ms: da.as_secs_f64() * 1e3,
        b_ms: db.as_secs_f64() * 1e3,
        class,
        note: "整片归约",
    }
}

/// **IS-2 接线前后**：A = 接线前的 `simd_dsl` 标量默认，B = [`LasxBackend`] 的 LASX override。
fn matmul_f64_case(name: &'static str, m: usize, k: usize, n: usize) -> Row {
    let (be, threads) = (LasxBackend, lasx_rs::pool::global().threads());
    let a: Vec<f64> = (0..m * k)
        .map(|i| ((i * 37 % 101) as f64) * 0.01 - 0.5)
        .collect();
    let b: Vec<f64> = (0..k * n)
        .map(|i| ((i * 53 % 97) as f64) * 0.01 + 0.25)
        .collect();
    let mut out = vec![0.0f64; m * n];

    // 等价类：对"接线前默认"是容差类（实测 ≤ 3.4e-16）；对 api::matmul_f64 是逐位，另由单测钉住。
    let mut sc = vec![0.0f64; m * n];
    scalar::matmul_f64(m, k, n, &a, &b, &mut sc).expect("scalar");
    be.matmul_f64(m, k, n, &a, &b, &mut out).expect("backend");
    let class = class_f64(equiv_f64(&out, &sc));
    let (da, db) = ab(
        || {
            scalar::matmul_f64(m, k, n, &a, &b, &mut sc).expect("scalar");
            black_box(sc[0]);
        },
        || {
            be.matmul_f64(m, k, n, &a, &b, &mut out).expect("backend");
            black_box(out[0]);
        },
    );
    Row {
        name,
        threads,
        work: Some(m as u128 * k as u128 * n as u128),
        a_ms: da.as_secs_f64() * 1e3,
        b_ms: db.as_secs_f64() * 1e3,
        class,
        note: "A=接线前标量默认；B=LASX override（IS-2）",
    }
}

fn dot_f64_case(name: &'static str, n: usize) -> Row {
    let (be, threads) = (LasxBackend, lasx_rs::pool::global().threads());
    let x: Vec<f64> = (0..n)
        .map(|i| ((i * 11 % 37) as f64) * 0.01 - 0.2)
        .collect();
    let y: Vec<f64> = (0..n).map(|i| ((i * 7 % 53) as f64) * 0.01 + 0.1).collect();
    let a = scalar::dot_f64(&x, &y).expect("scalar");
    let b = be.dot_f64(&x, &y).expect("backend");
    let class = class_f64(equiv_f64(&[b], &[a]));
    let (da, db) = ab(
        || {
            black_box(scalar::dot_f64(&x, &y).expect("scalar"));
        },
        || {
            black_box(be.dot_f64(&x, &y).expect("backend"));
        },
    );
    Row {
        name,
        threads,
        work: None,
        a_ms: da.as_secs_f64() * 1e3,
        b_ms: db.as_secs_f64() * 1e3,
        class,
        note: "A=接线前标量默认；B=LASX override（IS-2）",
    }
}

/// **IS-1 接线前后**：A = 接线前 `simd_dsl` 标量默认（`matmul_bt_f32`），B = LASX NT-GEMM override。
/// `note` 里另给"物化转置 + `api::matmul`"的 **替代算法**耗时作对照（不是同一等价类）。
fn matmul_bt_case(name: &'static str, m: usize, k: usize, n: usize) -> Row {
    let (be, threads) = (LasxBackend, lasx_rs::pool::global().threads());
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i * 37 % 101) as f32) * 0.01 - 0.5)
        .collect();
    // NT：B 是 n×k（按 j 行主序、k 连续）
    let bt: Vec<f32> = (0..n * k)
        .map(|i| ((i * 53 % 97) as f32) * 0.01 + 0.25)
        .collect();
    let mut out = vec![0.0f32; m * n];

    // 等价类：对"接线前标量默认"⇒ 逐位（同序）；对物化转置+api::matmul ⇒ 容差（印在 note）
    let mut sc = vec![0.0f32; m * n];
    scalar::matmul_bt_f32(m, k, n, &a, &bt, &mut sc).expect("scalar");
    be.matmul_bt_f32(m, k, n, &a, &bt, &mut out)
        .expect("backend");
    let class = class_f32(equiv_f32(&out, &sc));
    // 替代算法（物化转置 + api::matmul）的耗时：**仅供决策参考**。
    let mut bmat = vec![0.0f32; k * n];
    for (j, row) in bmat.chunks_exact_mut(n).enumerate() {
        for (t, v) in row.iter_mut().enumerate() {
            *v = bt[j * k + t];
        }
    }
    let mut alt_out = vec![0.0f32; m * n];
    let t_alt = {
        let t = Instant::now();
        lasx_rs::api::matmul_into(m, k, n, &a, &bmat, &mut alt_out).expect("api::matmul_into");
        black_box(alt_out[0]);
        t.elapsed()
    };
    let (da, db) = ab(
        || {
            scalar::matmul_bt_f32(m, k, n, &a, &bt, &mut sc).expect("scalar");
            black_box(sc[0]);
        },
        || {
            be.matmul_bt_f32(m, k, n, &a, &bt, &mut out)
                .expect("backend");
            black_box(out[0]);
        },
    );
    // `note` 要 `&'static str` ⇒ 替代算法的数进 stderr，表里给静态说明。
    eprintln!(
        "[参考] {} 物化转置+api::matmul（另一等价类）= {:.4} ms（比值 B/替代 = {:.2}）",
        name,
        t_alt.as_secs_f64() * 1e3,
        db.as_secs_f64() / t_alt.as_secs_f64()
    );
    Row {
        name,
        threads,
        work: Some(m as u128 * k as u128 * n as u128),
        a_ms: da.as_secs_f64() * 1e3,
        b_ms: db.as_secs_f64() * 1e3,
        class,
        note: "A=接线前标量默认；B=LASX NT 内核（IS-1）；替代算法见 stderr",
    }
}

fn main() {
    let be = LasxBackend;
    println!("# 迁移 A/B（阶段3 批1）——同进程交替、{ROUNDS} 对丢首对、取中位");
    println!();
    println!(
        "- 后端 `{:?}`（`name()` = `{}`）",
        ScalarKind::F32,
        be.name()
    );
    println!(
        "- `pool::global().threads()` = {}",
        lasx_rs::pool::global().threads()
    );
    println!("- `PAR_MIN_WORK` = {PAR_MIN_WORK}（work 越过它时 GEMM 内核自走全局池）");
    println!("- loadavg（开始）= `{}`", loadavg());
    println!(
        "- 等价类声明：`{:?}`（本装置先逐位、再谈性能）",
        be.precision_claim(simd_dsl::EnginePath::Default, ScalarKind::F32)
            .map(|c| c.scope)
            .unwrap_or("<none>")
    );

    let mut migrate = vec![
        matmul_case("matmul 64³", 64, 64, 64),
        matmul_case("matmul 256³", 256, 256, 256),
        matmul_case("matmul 512³", 512, 512, 512),
        matmul_case("matmul 1024×768×1024", 1024, 768, 1024),
        matmul_case("matmul 2048³", 2048, 2048, 2048),
        matmul_const_case("matmul_const 64³", 64, 64, 64),
        dot_case("dot 1M", 1 << 20),
        dot_case("dot 16M", 1 << 24),
        sum_case("sum 1M", 1 << 20),
        sum_case("sum 16M", 1 << 24),
    ];
    migrate.sort_by(|x, y| x.a_ms.total_cmp(&y.a_ms));
    print_rows(
        &migrate,
        "① 迁移 A/B：旧 `api::*` vs 新 `ops_dsl::*`（判据 ratio ≥ 0.98）",
    );

    println!("\n--- loadavg（迁移段后）= `{}`", loadavg());

    let wiring = vec![
        matmul_f64_case("matmul_f64 256³（IS-2 接线前后）", 256, 256, 256),
        matmul_f64_case("matmul_f64 512³（IS-2 接线前后）", 512, 512, 512),
        dot_f64_case("dot_f64 1M（IS-2 接线前后）", 1 << 20),
        matmul_bt_case("matmul_bt 256×256×256（IS-1 接线前后）", 256, 256, 256),
        matmul_bt_case("matmul_bt 512×512×512（IS-1 接线前后）", 512, 512, 512),
    ];
    print_rows(
        &wiring,
        "② 接线前后：A = 接线前标量默认，B = LasxBackend override",
    );

    println!("\n--- loadavg（接线段后）= `{}`", loadavg());
    println!(
        "\n注：① 的 ratio < 0.98 的用例需优化或如实记 `BelowThreshold`（归因：趟数/分配/缓存）；\
         \n② 是**接线收益**（越大越好），不是迁移门槛；`matmul_bt` 的替代算法（物化转置+api::matmul）\
         是另一等价类，见 stderr 的参考行。"
    );
}
