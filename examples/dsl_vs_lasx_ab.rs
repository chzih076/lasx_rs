//! **性能比赛·测试台第一批**：DSL 算子（`ops_dsl::*` + `LasxBackend`）vs LASX 库算子（`lasx_rs::api::*`）。
//!
//! # 用法（**每个进程只跑一个 threads 值**：`SIMD_DSL_ENGINE_THREADS` 引擎只读一次）
//!
//! ```text
//! cargo run --release --features dsl-backend --example dsl_vs_lasx_ab -- \
//!     --op=matmul --arm=lasx --threads=1 --iters=5
//! cargo run --release --features dsl-backend --example dsl_vs_lasx_ab -- \
//!     --op=matmul --arm=dsl  --threads=1 --iters=5
//! cargo run … --example dsl_vs_lasx_ab -- --list          # 支持/缺口的 op 清单
//! ```
//!
//! 输出两行：人类可读行 + 机器行
//! `RESULT op=<..> arm=<..> threads=<..> work=<..> eq=bitwise|tol:<界> median_ns=.. min_ns=.. max_ns=.. iters=..`。
//! 配套 `scripts/dsl_vs_lasx_ab.sh` 做**进程级交替** A/B（每轮 A、B 各起一个进程，≥5 轮丢首轮），
//! 汇总成 markdown 表。
//!
//! # 公平性（两臂对等）
//!
//! * **计时区外**构造输入、做一次两臂等价性校验（A 臂输出 vs B 臂输出；逐位优先，否则打印实测 rel 界）；
//! * 两臂都用**预分配缓冲**（`_into`/显式 out 参数），计时区内**无输入生成、无校验、无额外分配**；
//! * 线程数同一：`pool::init_global(N)`（LASX 内核的池版分派）+ DSL 臂另设
//!   `SIMD_DSL_ENGINE_THREADS=N`（引擎组合路径的线程数）；**不做任何 prepare/打包**（两臂都不做）；
//! * `cmatmul` 两臂都走 **Direct 路线**（4 条实矩阵乘）：A = `lasx_rs::cmatmul::cmatmul_f32_into`
//!   （其默认是 `Route::Direct`）；B = `ops_dsl::complex::cmatmul_f32`（引擎 4 matmul + 2 逐元素）。
//!
//! # 覆盖与缺口（如实）
//!
//! * **第一批实现**：`matmul`（1024³ NN）、`dot`（8M）、`sum`（16M）、`softmax_rows`、`rms_norm`、
//!   `layer_norm`（512×2048）、`gelu_quick`（4M）、`axpy`（4M）、`absmax_rows`（512×8192）、
//!   `cmatmul`（512³ 复数）。
//! * **缺口（LASX 无公开 `api::*` 入口 ⇒ 本轮不测，不拿内部函数冒充）**：
//!   `sum_cols`、`sum_rows`、`weighted_sum_cols`（只有 `parallel::reduce_*` 池版内部函数，不是库算子面）。

use std::hint::black_box;
use std::time::Instant;

use lasx_rs::aligned::AlignedVec;
use lasx_rs::dsl_backend::LasxBackend;

/// 支持的算子（第一批）。
const OPS: &[&str] = &[
    "matmul",
    "dot",
    "sum",
    "softmax_rows",
    "rms_norm",
    "layer_norm",
    "gelu_quick",
    "axpy",
    "absmax_rows",
    "cmatmul",
];

/// **缺口算子**：LASX 侧没有公开 `api::*` 入口（只有内部/池版函数）⇒ 不做 A/B。
const GAP_OPS: &[&str] = &["sum_cols", "sum_rows", "weighted_sum_cols"];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    Lasx,
    Dsl,
}

impl Arm {
    fn as_str(self) -> &'static str {
        match self {
            Arm::Lasx => "lasx",
            Arm::Dsl => "dsl",
        }
    }
}

struct Args {
    op: String,
    arm: Arm,
    threads: usize,
    /// DSL 臂的引擎线程数（默认 = `threads`）：用于诊断"引擎行切块 × 后端池"双重并行。
    engine_threads: Option<usize>,
    iters: usize,
}

fn usage(msg: &str) -> ! {
    eprintln!(
        "{msg}\n\n用法：--op=<name> --arm=lasx|dsl --threads=N [--engine-threads=M] --iters=N [--list]"
    );
    std::process::exit(2)
}

fn parse_args() -> Args {
    let (mut op, mut arm, mut threads, mut iters) = (String::new(), Arm::Lasx, 1usize, 5usize);
    let mut engine_threads: Option<usize> = None;
    for a in std::env::args().skip(1) {
        if a == "--list" {
            for o in OPS {
                println!("op={o} status=supported");
            }
            for o in GAP_OPS {
                println!("op={o} status=gap:lasx 无公开 api 入口");
            }
            std::process::exit(0);
        } else if let Some(v) = a.strip_prefix("--op=") {
            op = v.to_string();
        } else if let Some(v) = a.strip_prefix("--arm=") {
            arm = match v {
                "lasx" => Arm::Lasx,
                "dsl" => Arm::Dsl,
                other => usage(&format!("未知 --arm={other}")),
            };
        } else if let Some(v) = a.strip_prefix("--threads=") {
            threads = v.parse().unwrap_or_else(|_| usage("--threads 需为整数"));
        } else if let Some(v) = a.strip_prefix("--iters=") {
            iters = v.parse().unwrap_or_else(|_| usage("--iters 需为整数"));
        } else if let Some(v) = a.strip_prefix("--engine-threads=") {
            engine_threads = Some(
                v.parse()
                    .unwrap_or_else(|_| usage("--engine-threads 需为整数")),
            );
        } else {
            usage(&format!("未知参数 {a}"));
        }
    }
    if op.is_empty() {
        usage("缺少 --op=<name>（--list 看清单）");
    }
    if GAP_OPS.contains(&op.as_str()) {
        eprintln!(
            "op={op} status=gap: LASX 没有公开 api 入口（只有 `parallel::reduce_*` 池版内部函数）⇒ 不做 A/B"
        );
        std::process::exit(3);
    }
    if !OPS.contains(&op.as_str()) {
        usage(&format!("未知 --op={op}（--list 看清单）"));
    }
    if iters == 0 {
        usage("--iters 必须 ≥1");
    }
    Args {
        op,
        arm,
        threads,
        engine_threads,
        iters,
    }
}

/// 确定性输入（LCG；值域约 `[-1, 1)`，用于避免全零/溢出等退化）。
fn genv(n: usize, seed: u64) -> AlignedVec<f32> {
    let mut v = AlignedVec::<f32>::new(n);
    let mut s = seed | 1;
    for x in v.as_mut_slice() {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // 取高位 → [0,1) → [-1,1)
        *x = (((s >> 40) as f32) / ((1u64 << 24) as f32)) * 2.0 - 1.0;
    }
    v
}

struct Stats {
    median_ns: u128,
    min_ns: u128,
    max_ns: u128,
}

fn time_iters(mut f: impl FnMut(), iters: usize) -> Stats {
    let mut v = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        f();
        v.push(t.elapsed().as_nanos());
    }
    v.sort_unstable();
    Stats {
        median_ns: v[v.len() / 2],
        min_ns: v[0],
        max_ns: v[v.len() - 1],
    }
}

/// 选中的臂计时（两臂的闭包都传进来，只执行被选中的那个）。
fn time_selected<FA: FnMut(), FB: FnMut()>(
    arm: Arm,
    iters: usize,
    mut fa: FA,
    mut fb: FB,
) -> Stats {
    time_iters(
        || match arm {
            Arm::Lasx => fa(),
            Arm::Dsl => fb(),
        },
        iters,
    )
}

/// `(bitwise?, max_rel)`：逐位优先，否则给相对峰值的实测界。
fn eq_slices(a: &[f32], b: &[f32]) -> (bool, f64) {
    assert_eq!(a.len(), b.len(), "两臂输出长度不一致");
    if a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()) {
        return (true, 0.0);
    }
    let peak = a
        .iter()
        .fold(0.0f64, |m, v| m.max((*v).abs() as f64))
        .max(1e-30);
    let rel = a
        .iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((*x as f64 - *y as f64).abs()))
        / peak;
    (false, rel)
}

fn eq_scalar(a: f32, b: f32) -> (bool, f64) {
    if a.to_bits() == b.to_bits() {
        return (true, 0.0);
    }
    let peak = (a.abs() as f64).max(1e-30);
    (false, ((a as f64) - (b as f64)).abs() / peak)
}

fn eq_str(bitwise: bool, rel: f64) -> String {
    if bitwise {
        "bitwise".to_string()
    } else {
        format!("tol:{rel:.3e}")
    }
}

#[allow(clippy::too_many_arguments)]
fn print_result(
    op: &str,
    arm: Arm,
    threads: usize,
    work: &str,
    eq: &str,
    st: &Stats,
    iters: usize,
) {
    let arm_s = arm.as_str();
    println!(
        "  {op:<12} {arm_s:<4} t={threads:<2} {work:<20} median={:.3} ms  range=[{:.3},{:.3}] ms  eq={eq}",
        st.median_ns as f64 / 1e6,
        st.min_ns as f64 / 1e6,
        st.max_ns as f64 / 1e6,
    );
    println!(
        "RESULT op={op} arm={arm_s} threads={threads} work={work} eq={eq} median_ns={} min_ns={} max_ns={} iters={iters}",
        st.median_ns, st.min_ns, st.max_ns,
    );
}

fn main() {
    let args = parse_args();
    let threads = args.threads;
    let engine_threads = args.engine_threads.unwrap_or(threads);
    let iters = args.iters;
    // 诊断用：同时打印 pool 线程与引擎线程（RESULT 行格式不变，脚本仍按原格式解析）。
    println!("META threads={threads} engine_threads={engine_threads}");

    // **引擎线程数（DSL 臂）**：`SIMD_DSL_ENGINE_THREADS` 只读一次 ⇒ 必须在**任何引擎调用之前**设置；
    // 本进程只跑一个 threads 值（脚本每轮/每臂各起一个进程）。
    if args.arm == Arm::Dsl {
        // SAFETY: 进程刚启动、尚未调用任何 simd_dsl 引擎入口（`OnceLock` 未初始化），也无线程并发读该环境变量。
        unsafe {
            std::env::set_var("SIMD_DSL_ENGINE_THREADS", engine_threads.to_string());
        }
    }
    lasx_rs::pool::init_global(threads);

    let be = LasxBackend;
    let loadavg = std::fs::read_to_string("/proc/loadavg")
        .unwrap_or_default()
        .trim()
        .to_string();
    println!(
        "# dsl_vs_lasx_ab op={} arm={} threads_requested={} pool_threads={} loadavg=`{loadavg}`",
        args.op,
        args.arm.as_str(),
        threads,
        lasx_rs::pool::global().threads(),
    );

    match args.op.as_str() {
        "matmul" => bench_matmul(&be, args.arm, threads, iters),
        "dot" => bench_dot(&be, args.arm, threads, iters),
        "sum" => bench_sum(&be, args.arm, threads, iters),
        "softmax_rows" => bench_softmax(&be, args.arm, threads, iters),
        "rms_norm" => bench_rms(&be, args.arm, threads, iters),
        "layer_norm" => bench_layer(&be, args.arm, threads, iters),
        "gelu_quick" => bench_gelu(&be, args.arm, threads, iters),
        "axpy" => bench_axpy(&be, args.arm, threads, iters),
        "absmax_rows" => bench_absmax(&be, args.arm, threads, iters),
        "cmatmul" => bench_cmatmul(&be, args.arm, threads, iters),
        other => unreachable!("已由 parse_args 校验：{other}"),
    }
}

// ============================== 各算子 ==============================

fn bench_matmul(be: &LasxBackend, arm: Arm, threads: usize, iters: usize) {
    let (m, k, n) = (1024usize, 1024usize, 1024usize);
    let a = genv(m * k, 11);
    let b = genv(k * n, 12);
    let (a, b) = (a.as_slice(), b.as_slice());
    let mut out_a = AlignedVec::<f32>::new(m * n);
    let mut out_b = AlignedVec::<f32>::new(m * n);
    lasx_rs::api::matmul_into(m, k, n, a, b, out_a.as_mut_slice()).expect("lasx matmul");
    ops_dsl::matmul::matmul_f32(be, m, k, n, a, b, out_b.as_mut_slice()).expect("dsl matmul");
    let (bw, rel) = eq_slices(out_a.as_slice(), out_b.as_slice());
    let eq = eq_str(bw, rel);
    // 计时：两臂都写**同一块**自己的输出缓冲（不重复分配）
    let st = time_selected(
        arm,
        iters,
        || {
            lasx_rs::api::matmul_into(m, k, n, a, b, out_a.as_mut_slice()).expect("lasx");
            black_box(out_a.as_slice()[0]);
        },
        || {
            ops_dsl::matmul::matmul_f32(be, m, k, n, a, b, out_b.as_mut_slice()).expect("dsl");
            black_box(out_b.as_slice()[0]);
        },
    );
    print_result(
        "matmul",
        arm,
        threads,
        &format!("{m}x{k}x{n}"),
        &eq,
        &st,
        iters,
    );
}

fn bench_dot(be: &LasxBackend, arm: Arm, threads: usize, iters: usize) {
    let n = 8_388_608usize;
    let x = genv(n, 21);
    let y = genv(n, 22);
    let (x, y) = (x.as_slice(), y.as_slice());
    let a = lasx_rs::api::dot(x, y).expect("lasx dot");
    let b = ops_dsl::reduce::dot_f32(be, x, y);
    let (bw, rel) = eq_scalar(a, b);
    let eq = eq_str(bw, rel);
    let st = time_selected(
        arm,
        iters,
        || {
            black_box(lasx_rs::api::dot(x, y).expect("lasx"));
        },
        || {
            black_box(ops_dsl::reduce::dot_f32(be, x, y));
        },
    );
    print_result("dot", arm, threads, &format!("n={n}"), &eq, &st, iters);
}

fn bench_sum(be: &LasxBackend, arm: Arm, threads: usize, iters: usize) {
    let n = 16_777_216usize;
    let x = genv(n, 31);
    let x = x.as_slice();
    let a = lasx_rs::api::sum(x);
    let b = ops_dsl::reduce::sum_f32(be, x);
    let (bw, rel) = eq_scalar(a, b);
    let eq = eq_str(bw, rel);
    let st = time_selected(
        arm,
        iters,
        || {
            black_box(lasx_rs::api::sum(x));
        },
        || {
            black_box(ops_dsl::reduce::sum_f32(be, x));
        },
    );
    print_result("sum", arm, threads, &format!("n={n}"), &eq, &st, iters);
}

/// 行 softmax（`rows × cols`）；两臂同输入、同输出缓冲预分配。
fn bench_softmax(be: &LasxBackend, arm: Arm, threads: usize, iters: usize) {
    let (rows, cols) = (512usize, 2048usize);
    let n = rows * cols;
    let x = genv(n, 41);
    let x = x.as_slice();
    let mut out_a = AlignedVec::<f32>::new(n);
    let mut out_b = AlignedVec::<f32>::new(n);
    let (mut stat, mut shifted, mut ex) = (
        AlignedVec::<f32>::new(rows),
        AlignedVec::<f32>::new(n),
        AlignedVec::<f32>::new(n),
    );
    lasx_rs::api::softmax_rows_into(x, None, rows, cols, 1.0, out_a.as_mut_slice())
        .expect("lasx softmax");
    ops_dsl::norm::softmax_rows(
        be,
        x,
        rows,
        cols,
        stat.as_mut_slice(),
        shifted.as_mut_slice(),
        ex.as_mut_slice(),
        out_b.as_mut_slice(),
    )
    .expect("dsl softmax");
    let (bw, rel) = eq_slices(out_a.as_slice(), out_b.as_slice());
    let eq = eq_str(bw, rel);
    let st = time_selected(
        arm,
        iters,
        || {
            lasx_rs::api::softmax_rows_into(x, None, rows, cols, 1.0, out_a.as_mut_slice())
                .expect("lasx");
            black_box(out_a.as_slice()[0]);
        },
        || {
            ops_dsl::norm::softmax_rows(
                be,
                x,
                rows,
                cols,
                stat.as_mut_slice(),
                shifted.as_mut_slice(),
                ex.as_mut_slice(),
                out_b.as_mut_slice(),
            )
            .expect("dsl");
            black_box(out_b.as_slice()[0]);
        },
    );
    print_result(
        "softmax_rows",
        arm,
        threads,
        &format!("{rows}x{cols}"),
        &eq,
        &st,
        iters,
    );
}

fn bench_rms(be: &LasxBackend, arm: Arm, threads: usize, iters: usize) {
    let (rows, cols) = (512usize, 2048usize);
    let eps = 1e-5f32;
    let n = rows * cols;
    let x = genv(n, 51);
    let x = x.as_slice();
    let mut out_a = AlignedVec::<f32>::new(n);
    let mut out_b = AlignedVec::<f32>::new(n);
    let (mut sq, mut stat, mut inv) = (
        AlignedVec::<f32>::new(n),
        AlignedVec::<f32>::new(rows),
        AlignedVec::<f32>::new(rows),
    );
    lasx_rs::api::rms_norm_into(x, None, rows, cols, eps, out_a.as_mut_slice()).expect("lasx rms");
    ops_dsl::norm::rms_norm(
        be,
        x,
        rows,
        cols,
        eps,
        sq.as_mut_slice(),
        stat.as_mut_slice(),
        inv.as_mut_slice(),
        out_b.as_mut_slice(),
    )
    .expect("dsl rms");
    let (bw, rel) = eq_slices(out_a.as_slice(), out_b.as_slice());
    let eq = eq_str(bw, rel);
    let st = time_selected(
        arm,
        iters,
        || {
            lasx_rs::api::rms_norm_into(x, None, rows, cols, eps, out_a.as_mut_slice())
                .expect("lasx");
            black_box(out_a.as_slice()[0]);
        },
        || {
            ops_dsl::norm::rms_norm(
                be,
                x,
                rows,
                cols,
                eps,
                sq.as_mut_slice(),
                stat.as_mut_slice(),
                inv.as_mut_slice(),
                out_b.as_mut_slice(),
            )
            .expect("dsl");
            black_box(out_b.as_slice()[0]);
        },
    );
    print_result(
        "rms_norm",
        arm,
        threads,
        &format!("{rows}x{cols}"),
        &eq,
        &st,
        iters,
    );
}

fn bench_layer(be: &LasxBackend, arm: Arm, threads: usize, iters: usize) {
    let (rows, cols) = (512usize, 2048usize);
    let eps = 1e-5f32;
    let n = rows * cols;
    let x = genv(n, 61);
    let x = x.as_slice();
    let mut out_a = AlignedVec::<f32>::new(n);
    let mut out_b = AlignedVec::<f32>::new(n);
    let (mut d, mut stat, mut inv) = (
        AlignedVec::<f32>::new(n),
        AlignedVec::<f32>::new(rows),
        AlignedVec::<f32>::new(rows),
    );
    lasx_rs::api::layer_norm_into(x, None, None, rows, cols, eps, out_a.as_mut_slice())
        .expect("lasx layer");
    ops_dsl::norm::layer_norm(
        be,
        x,
        rows,
        cols,
        eps,
        d.as_mut_slice(),
        stat.as_mut_slice(),
        inv.as_mut_slice(),
        out_b.as_mut_slice(),
    )
    .expect("dsl layer");
    let (bw, rel) = eq_slices(out_a.as_slice(), out_b.as_slice());
    let eq = eq_str(bw, rel);
    let st = time_selected(
        arm,
        iters,
        || {
            lasx_rs::api::layer_norm_into(x, None, None, rows, cols, eps, out_a.as_mut_slice())
                .expect("lasx");
            black_box(out_a.as_slice()[0]);
        },
        || {
            ops_dsl::norm::layer_norm(
                be,
                x,
                rows,
                cols,
                eps,
                d.as_mut_slice(),
                stat.as_mut_slice(),
                inv.as_mut_slice(),
                out_b.as_mut_slice(),
            )
            .expect("dsl");
            black_box(out_b.as_slice()[0]);
        },
    );
    print_result(
        "layer_norm",
        arm,
        threads,
        &format!("{rows}x{cols}"),
        &eq,
        &st,
        iters,
    );
}

fn bench_gelu(be: &LasxBackend, arm: Arm, threads: usize, iters: usize) {
    let n = 4_194_304usize;
    let x = genv(n, 71);
    let x = x.as_slice();
    let mut out_a = AlignedVec::<f32>::new(n);
    let mut out_b = AlignedVec::<f32>::new(n);
    let (mut t1, mut t2, mut t3) = (
        AlignedVec::<f32>::new(n),
        AlignedVec::<f32>::new(n),
        AlignedVec::<f32>::new(n),
    );
    lasx_rs::api::gelu_quick_into(x, out_a.as_mut_slice()).expect("lasx gelu");
    ops_dsl::activations::gelu_quick(
        be,
        x,
        t1.as_mut_slice(),
        t2.as_mut_slice(),
        t3.as_mut_slice(),
        out_b.as_mut_slice(),
    )
    .expect("dsl gelu");
    let (bw, rel) = eq_slices(out_a.as_slice(), out_b.as_slice());
    let eq = eq_str(bw, rel);
    let st = time_selected(
        arm,
        iters,
        || {
            lasx_rs::api::gelu_quick_into(x, out_a.as_mut_slice()).expect("lasx");
            black_box(out_a.as_slice()[0]);
        },
        || {
            ops_dsl::activations::gelu_quick(
                be,
                x,
                t1.as_mut_slice(),
                t2.as_mut_slice(),
                t3.as_mut_slice(),
                out_b.as_mut_slice(),
            )
            .expect("dsl");
            black_box(out_b.as_slice()[0]);
        },
    );
    print_result(
        "gelu_quick",
        arm,
        threads,
        &format!("n={n}"),
        &eq,
        &st,
        iters,
    );
}

/// `y := alpha·x + y`（**就地**）：两臂各用自己的 `y` 缓冲（同初值），等价性用计时区外的一次调用校验。
fn bench_axpy(be: &LasxBackend, arm: Arm, threads: usize, iters: usize) {
    let n = 4_194_304usize;
    let alpha = 0.375f32;
    let x = genv(n, 81);
    let x = x.as_slice();
    let mut y_a = genv(n, 82);
    let mut y_b = genv(n, 82);
    let (mut tmp, mut acc) = (AlignedVec::<f32>::new(n), AlignedVec::<f32>::new(n));
    // 校验：用**沙盒副本**各跑一次（不污染计时用的 y）
    let mut ca = AlignedVec::<f32>::new(n);
    let mut cb = AlignedVec::<f32>::new(n);
    ca.as_mut_slice().copy_from_slice(y_a.as_slice());
    cb.as_mut_slice().copy_from_slice(y_b.as_slice());
    lasx_rs::api::axpy(alpha, x, ca.as_mut_slice()).expect("lasx axpy");
    ops_dsl::reduce::axpy_f32(
        be,
        alpha,
        x,
        cb.as_mut_slice(),
        tmp.as_mut_slice(),
        acc.as_mut_slice(),
    )
    .expect("dsl axpy");
    let (bw, rel) = eq_slices(ca.as_slice(), cb.as_slice());
    let eq = eq_str(bw, rel);
    let st = time_selected(
        arm,
        iters,
        || {
            lasx_rs::api::axpy(alpha, x, y_a.as_mut_slice()).expect("lasx");
            black_box(y_a.as_slice()[0]);
        },
        || {
            ops_dsl::reduce::axpy_f32(
                be,
                alpha,
                x,
                y_b.as_mut_slice(),
                tmp.as_mut_slice(),
                acc.as_mut_slice(),
            )
            .expect("dsl");
            black_box(y_b.as_slice()[0]);
        },
    );
    print_result("axpy", arm, threads, &format!("n={n}"), &eq, &st, iters);
}

fn bench_absmax(be: &LasxBackend, arm: Arm, threads: usize, iters: usize) {
    let (rows, cols) = (512usize, 8192usize);
    let n = rows * cols;
    let x = genv(n, 91);
    let x = x.as_slice();
    let mut out_a = AlignedVec::<f32>::new(rows);
    let mut out_b = AlignedVec::<f32>::new(rows);
    let (mut neg, mut abs) = (AlignedVec::<f32>::new(n), AlignedVec::<f32>::new(n));
    lasx_rs::api::absmax_rows_into(x, rows, cols, out_a.as_mut_slice()).expect("lasx absmax");
    ops_dsl::quant::absmax_rows_f32(
        be,
        x,
        rows,
        cols,
        neg.as_mut_slice(),
        abs.as_mut_slice(),
        out_b.as_mut_slice(),
    )
    .expect("dsl absmax");
    let (bw, rel) = eq_slices(out_a.as_slice(), out_b.as_slice());
    let eq = eq_str(bw, rel);
    let st = time_selected(
        arm,
        iters,
        || {
            lasx_rs::api::absmax_rows_into(x, rows, cols, out_a.as_mut_slice()).expect("lasx");
            black_box(out_a.as_slice()[0]);
        },
        || {
            ops_dsl::quant::absmax_rows_f32(
                be,
                x,
                rows,
                cols,
                neg.as_mut_slice(),
                abs.as_mut_slice(),
                out_b.as_mut_slice(),
            )
            .expect("dsl");
            black_box(out_b.as_slice()[0]);
        },
    );
    print_result(
        "absmax_rows",
        arm,
        threads,
        &format!("{rows}x{cols}"),
        &eq,
        &st,
        iters,
    );
}

/// 复数矩阵乘（分平面）：两臂都走 **Direct 路线**（4 条实矩阵乘）。
fn bench_cmatmul(be: &LasxBackend, arm: Arm, threads: usize, iters: usize) {
    let (m, k, n) = (512usize, 512usize, 512usize);
    let (ar, ai) = (genv(m * k, 101), genv(m * k, 102));
    let (br, bi) = (genv(k * n, 103), genv(k * n, 104));
    let mut cr_a = AlignedVec::<f32>::new(m * n);
    let mut ci_a = AlignedVec::<f32>::new(m * n);
    let mut cr_b = AlignedVec::<f32>::new(m * n);
    let mut ci_b = AlignedVec::<f32>::new(m * n);
    let (mut t1, mut t2) = (AlignedVec::<f32>::new(m * n), AlignedVec::<f32>::new(m * n));
    lasx_rs::cmatmul::cmatmul_f32_into(
        m,
        k,
        n,
        ar.as_slice(),
        ai.as_slice(),
        br.as_slice(),
        bi.as_slice(),
        cr_a.as_mut_slice(),
        ci_a.as_mut_slice(),
    );
    ops_dsl::complex::cmatmul_f32(
        be,
        m,
        k,
        n,
        ar.as_slice(),
        ai.as_slice(),
        br.as_slice(),
        bi.as_slice(),
        cr_b.as_mut_slice(),
        ci_b.as_mut_slice(),
        t1.as_mut_slice(),
        t2.as_mut_slice(),
    )
    .expect("dsl cmatmul");
    let (bw_r, rel_r) = eq_slices(cr_a.as_slice(), cr_b.as_slice());
    let (bw_i, rel_i) = eq_slices(ci_a.as_slice(), ci_b.as_slice());
    let eq = eq_str(bw_r && bw_i, rel_r.max(rel_i));
    let st = time_selected(
        arm,
        iters,
        || {
            lasx_rs::cmatmul::cmatmul_f32_into(
                m,
                k,
                n,
                ar.as_slice(),
                ai.as_slice(),
                br.as_slice(),
                bi.as_slice(),
                cr_a.as_mut_slice(),
                ci_a.as_mut_slice(),
            );
            black_box(cr_a.as_slice()[0]);
        },
        || {
            ops_dsl::complex::cmatmul_f32(
                be,
                m,
                k,
                n,
                ar.as_slice(),
                ai.as_slice(),
                br.as_slice(),
                bi.as_slice(),
                cr_b.as_mut_slice(),
                ci_b.as_mut_slice(),
                t1.as_mut_slice(),
                t2.as_mut_slice(),
            )
            .expect("dsl");
            black_box(cr_b.as_slice()[0]);
        },
    );
    print_result(
        "cmatmul",
        arm,
        threads,
        &format!("{m}x{k}x{n} 复数(Direct)"),
        &eq,
        &st,
        iters,
    );
}
