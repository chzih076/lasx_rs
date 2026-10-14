//! 计划复用（[`lasx_rs::plan::MatmulPlan`]）：`B` 固定时"打包一次"到底省多少。
//!
//! 三列对照，逐列剥掉一层开销（都算同一次 `A·B`）：
//!
//! | 列 | 输出缓冲 | `B` 打包 |
//! |---|---|---|
//! | ① `api::matmul` | 每次新分配 + 清零 | 每次重打包 |
//! | ② `lasx_matmul`（C 复用） | 复用调用方的 | 每次重打包 |
//! | ③ `MatmulPlan::run_into` | 复用调用方的 | **只打包一次** |
//!
//! 所以"③ / ②"就是**纯粹**的打包复用收益，不受输出分配干扰。
//!
//! 最后一段 `dsl_plan_reuse` 把同一件事搬到**公式 DSL** 上：`matmul!(y = x * w[K,N])` 生成的
//! `x.mul_into(&w, &mut y)` 内部每次 `prepare()`；新形态 `matmul!(y = x * p)`（右操作数不带
//! 方括号 = `Prepared`）生成 `p.apply_into(&x, &mut y)`。两臂进程级交替、丢弃首轮、取中位数。

use crate::data::{AlignedBuf, Lcg};
use crate::timing::{fmt_t, median, timeit};
use lasx_rs::api;
use lasx_rs::lasx_matmul;
use lasx_rs::lasx_matmul_f64;
use lasx_rs::plan::MatmulPlan;
use std::hint::black_box;

pub fn run() {
    println!("### 计划复用：`B` 固定，反复算 `A·B`（单线程，`MatmulPlan`）");
    println!();
    println!(
        "| 形状 | ① `api::matmul` | ② `lasx_matmul`（C 复用） | ③ `run_into`（C 复用+免打包） | ③/② | 建计划（一次性） | 计划占用 |"
    );
    println!("|---|---|---|---|---|---|---|");
    for &(m, k, n) in &[
        (64usize, 64usize, 64usize),
        (256, 256, 256),
        (512, 512, 512),
        (1000, 64, 4096),
        // 极宽 n + 小 k：`matmul_f32` 会选"列块"路径，而计划固定走打包路径。
        (100, 64, 19147),
    ] {
        row_f32(m, k, n);
    }
    for &(m, k, n) in &[
        (128usize, 128usize, 128usize),
        (256, 256, 256),
        (512, 512, 512),
    ] {
        row_f64(m, k, n);
    }
    println!();
    println!("③/② 是「只把 `B` 的打包省掉」的收益：`k·n` 相对 `m·k·n` 越大越明显。");
    println!("小形状上计划固定走打包路径，而 `lasx_matmul` 会按形状走流式路径，两者可能持平。");
    println!();
    shape_layer();
    dsl_plan_reuse();
}

/// DSL 侧的同一件事：`matmul!` 默认**每次调用都重新打包**右操作数。
///
/// 根因是 `Mat::mul_into` 的实现 `rhs.prepare().apply_into(self, out)`：`matmul!(y[M,N] =
/// x[M,K] * w[K,N])` 生成 `x.mul_into(&w, &mut y)`，所以层循环里复用同一权重会重复打包。
/// 新形态让右操作数**不写方括号**（`Prepared` 自带形状）：`matmul!(y[M,N] = x[M,K] * p)`
/// 生成 `p.apply_into(&x, &mut y)`，打包只在循环外的 `w.prepare()` 发生一次。
///
/// 口径（`docs/dev.md` §6.4）：两臂**进程级交替**、≥3 轮、**丢弃首轮**、取中位数；每轮内部是
/// [`timeit`]（自适应重复 + 5 取样中位数）。输出第一行是 `/proc/loadavg`——load ≤3 才算正式
/// 数，否则只能当趋势。计时**不含计划构建**（`p` 在计时区之外建好）。
fn dsl_plan_reuse() {
    use lasx_rs::matmul;
    use lasx_rs::shape::{Mat, MatBuf};

    const M: usize = 256;
    const K: usize = 256;
    const N: usize = 256;
    const ROUNDS: usize = 4; // 丢弃首轮后仍有 3 轮

    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    println!(
        "\n## DSL 计划复用：每次重新打包 vs 复用 plan（{M}×{K}×{N}，loadavg = {}）",
        load.trim()
    );
    println!();
    println!(
        "| 轮次 | ① `matmul!(y = x * w[K,N])`（每次 pack） | ② `matmul!(y = x * p)`（plan 复用） | ②/① |"
    );
    println!("|---|---|---|---|");

    let mut rng = Lcg::new(0x5eed_1234);
    let weights = AlignedBuf::fill_with(K * N, |_| rng.f32());
    let batch = AlignedBuf::fill_with(M * K, |_| rng.f32());
    let w = Mat::<f32, { K }, { N }>::new(weights.as_slice()).unwrap();
    let x = Mat::<f32, { M }, { K }>::new(batch.as_slice()).unwrap();
    let mut y_pack = MatBuf::<f32, { M }, { N }>::new();
    let mut y_plan = MatBuf::<f32, { M }, { N }>::new();

    // 计划在**计时区之外**建一次：这就是"循环外准备"的形态本身。
    let p = w.prepare();

    let mut pack_rounds = Vec::with_capacity(ROUNDS - 1);
    let mut plan_rounds = Vec::with_capacity(ROUNDS - 1);

    {
        let mut arm_pack = || {
            timeit(|| {
                matmul!(y_pack[M, N] = x[M, K] * w[K, N]); // → x.mul_into(&w, &mut y_pack)
                let _ = black_box(y_pack.as_slice()[0]);
            })
        };
        let mut arm_plan = || {
            timeit(|| {
                matmul!(y_plan[M, N] = x[M, K] * p); // → p.apply_into(&x, &mut y_plan)
                let _ = black_box(y_plan.as_slice()[0]);
            })
        };
        for round in 0..ROUNDS {
            // 交替顺序（奇偶轮互换），抵消"第二个被测者占便宜/吃亏"的顺序效应
            let (t_pack, t_plan) = if round % 2 == 0 {
                let a = arm_pack();
                let b = arm_plan();
                (a, b)
            } else {
                let b = arm_plan();
                let a = arm_pack();
                (a, b)
            };
            println!(
                "| r{} | {} | {} | {:.2}× |",
                round + 1,
                fmt_t(t_pack),
                fmt_t(t_plan),
                t_pack.as_secs_f64() / t_plan.as_secs_f64()
            );
            if round > 0 {
                pack_rounds.push(t_pack);
                plan_rounds.push(t_plan);
            }
        }
    }

    let t_pack = median(pack_rounds);
    let t_plan = median(plan_rounds);
    let speedup = t_pack.as_secs_f64() / t_plan.as_secs_f64();
    println!(
        "| **中位数（丢弃首轮，{} 轮）** | **{}** | **{}** | **{:.2}×（省 {:.0}%）** |",
        ROUNDS - 1,
        fmt_t(t_pack),
        fmt_t(t_plan),
        speedup,
        100.0 * (1.0 - 1.0 / speedup)
    );

    // 快但不能错：两臂互相比、再与 `api::matmul` 逐位比（`to_bits()`，不是近似）
    let want = api::matmul(M, K, N, batch.as_slice(), weights.as_slice()).unwrap();
    let want_bits: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
    let arms = [
        ("① 每次 pack", y_pack.as_slice()),
        ("② plan 复用", y_plan.as_slice()),
    ];
    let mut arm_bits: Vec<Vec<u32>> = Vec::with_capacity(arms.len());
    for (label, got) in arms {
        let bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
        assert!(
            bits == want_bits,
            "{M}×{K}×{N}: {label} 与 api::matmul 不逐位一致"
        );
        arm_bits.push(bits);
    }
    assert!(
        arm_bits[0] == arm_bits[1],
        "{M}×{K}×{N}: 两臂之间不逐位一致"
    );
    println!();
    println!("两臂与 `api::matmul` 逐位一致（`to_bits()` 全量比对）——② 只是少了重复打包，");
    println!("数值路径完全相同，没有近似的余地。load > 3 时上表只能当趋势。");
}

/// `shape` 层自己的验收：**同一个 `Prepared` 类型、只换策略参数**（`Auto` vs `Single`）。
///
/// 它测的是"策略层有没有正确接线"（`Auto` 有没有真把活铺到多核、块切得对不对），
/// 与上面那张表的"shape 层有没有正确复用底层"是两件事，两个都要。
///
/// 形状是 const 泛型，没法在运行期循环——所以每个形状一行显式 case（这本来就是这层
/// 的用法：形状写死在类型里）。输入走 DYN 入口吃运行期行数，权重侧仍是全静态。
fn shape_layer() {
    use lasx_rs::shape::{Mat, MatBufDyn, MatDyn, Single};

    println!("### `shape` 层验收：`Prepared<Auto>` vs `Prepared<Single>`（同一份打包权重）");
    println!();
    println!("| 形状 | `Single` | `Auto` | 加速 | Auto 线程数 |");
    println!("|---|---|---|---|---|");

    macro_rules! case {
        ($m:expr, $k:expr, $n:expr) => {{
            let (m, k, n) = ($m, $k, $n);
            let mut rng = Lcg::new((m * 31 + k * 7 + n) as u64);
            let weights = AlignedBuf::fill_with(k * n, |_| rng.f32());
            let batch = AlignedBuf::fill_with(m * k, |_| rng.f32());
            let w = Mat::<f32, $k, $n>::new(weights.as_slice()).unwrap();
            let single = w.prepare_with::<Single>();
            let auto = w.prepare();

            let x = MatDyn::<f32, $k>::new(batch.as_slice()).unwrap();
            let mut out = MatBufDyn::<f32, $n>::with_rows(m);
            single.apply_dyn_into(&x, &mut out);
            let mut out_auto = MatBufDyn::<f32, $n>::with_rows(m);
            auto.apply_dyn_into(&x, &mut out_auto);
            assert_eq!(
                out.as_slice()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                out_auto
                    .as_slice()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "{}×{}×{}: Auto 与 Single 不逐位一致",
                m,
                k,
                n
            );

            let t_single = timeit(|| {
                single.apply_dyn_into(&x, &mut out);
                let _ = black_box(out.as_slice()[0]);
            });
            let t_auto = timeit(|| {
                auto.apply_dyn_into(&x, &mut out_auto);
                let _ = black_box(out_auto.as_slice()[0]);
            });
            println!(
                "| f32 {m}×{k}×{n} | {}（{}） | {}（{}） | {:.2}× | {} |",
                fmt_t(t_single),
                gf(flops(m, k, n), t_single),
                fmt_t(t_auto),
                gf(flops(m, k, n), t_auto),
                t_single.as_secs_f64() / t_auto.as_secs_f64(),
                auto.threads::<{ $m }>(),
            );
        }};
    }
    // 小形状：`Auto` 会判成"不派活"，所以这里量的其实是**判据本身的开销**
    // （`Auto::threads` 每次都问一遍机器并行度——那是系统调用）
    case!(16, 64, 64);
    case!(64, 256, 256);
    case!(256, 256, 256);
    case!(512, 512, 512);
    case!(1024, 1024, 1024);
    println!();
    println!("（`Auto` 判据见 `shape::auto_threads`：每线程 ≥4 行 **且** 工作量 ≥4 M 乘加；");
    println!("`Single` 恒 1、完全不碰池。两者逐位一致是本函数开头的断言。）");
}

fn flops(m: usize, k: usize, n: usize) -> f64 {
    2.0 * (m * k * n) as f64
}

/// 把耗时换算成 GFLOP/s。
fn gf(work: f64, d: std::time::Duration) -> String {
    format!("{:.1} GF/s", work / d.as_secs_f64() / 1e9)
}

fn row_f32(m: usize, k: usize, n: usize) {
    let mut rng = Lcg::new((m * 31 + k * 7 + n) as u64);
    let a = AlignedBuf::fill_with(m * k, |_| rng.f32());
    let b = AlignedBuf::fill_with(k * n, |_| rng.f32());
    let mut c = AlignedBuf::new(m * n);
    let work = flops(m, k, n);

    let build = timeit(|| {
        let p = MatmulPlan::from_row_major(&b, k, n).unwrap();
        let _ = black_box(&p);
    });
    let plan = MatmulPlan::from_row_major(&b, k, n).unwrap();
    let api_t = timeit(|| {
        let out = api::matmul(m, k, n, &a, &b).unwrap();
        let _ = black_box(out.as_slice()[0]);
    });
    let ffi_t = timeit(|| {
        lasx_matmul(
            m as i32,
            k as i32,
            n as i32,
            a.as_ptr(),
            b.as_ptr(),
            c.as_mut_ptr(),
        );
        let _ = black_box(c[0]);
    });
    let plan_t = timeit(|| {
        plan.run_into(&a, &mut c).unwrap();
        let _ = black_box(c[0]);
    });
    // 快但不能错：与 api::matmul 逐位比一遍
    let want = api::matmul(m, k, n, &a, &b).unwrap();
    assert!(
        want.iter()
            .zip(c.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits()),
        "{m}×{k}×{n}: 计划与 api::matmul 不逐位一致"
    );

    report("f32", m, k, n, work, api_t, ffi_t, plan_t, build, &plan);
}

fn row_f64(m: usize, k: usize, n: usize) {
    let mut rng = Lcg::new((m * 17 + k * 5 + n) as u64);
    let a = AlignedBuf::fill_with(m * k, |_| rng.f64());
    let b = AlignedBuf::fill_with(k * n, |_| rng.f64());
    let mut c = AlignedBuf::new(m * n);
    let work = flops(m, k, n);

    let build = timeit(|| {
        let p = MatmulPlan::from_row_major(&b, k, n).unwrap();
        let _ = black_box(&p);
    });
    let plan = MatmulPlan::from_row_major(&b, k, n).unwrap();
    let api_t = timeit(|| {
        let out = api::matmul_f64(m, k, n, &a, &b).unwrap();
        let _ = black_box(out.as_slice()[0]);
    });
    let ffi_t = timeit(|| {
        lasx_matmul_f64(
            m as i32,
            k as i32,
            n as i32,
            a.as_ptr(),
            b.as_ptr(),
            c.as_mut_ptr(),
        );
        let _ = black_box(c[0]);
    });
    let plan_t = timeit(|| {
        plan.run_into(&a, &mut c).unwrap();
        let _ = black_box(c[0]);
    });
    let want = api::matmul_f64(m, k, n, &a, &b).unwrap();
    assert!(
        want.iter()
            .zip(c.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits()),
        "{m}×{k}×{n}: 计划与 api::matmul_f64 不逐位一致"
    );

    report("f64", m, k, n, work, api_t, ffi_t, plan_t, build, &plan);
}

#[allow(clippy::too_many_arguments)]
fn report<T: lasx_rs::plan::PackedKernel>(
    ty: &str,
    m: usize,
    k: usize,
    n: usize,
    work: f64,
    api_t: std::time::Duration,
    ffi_t: std::time::Duration,
    plan_t: std::time::Duration,
    build: std::time::Duration,
    plan: &MatmulPlan<T>,
) {
    println!(
        "| {ty} {m}×{k}×{n} | {}（{}） | {}（{}） | {}（{}） | {:.2}× | {} | {} |",
        fmt_t(api_t),
        gf(work, api_t),
        fmt_t(ffi_t),
        gf(work, ffi_t),
        fmt_t(plan_t),
        gf(work, plan_t),
        ffi_t.as_secs_f64() / plan_t.as_secs_f64(),
        fmt_t(build),
        fmt_bytes(plan.packed_bytes()),
    );
}

fn fmt_bytes(b: usize) -> String {
    if b >= 1024 * 1024 {
        format!("{:.1} MiB", b as f64 / (1024.0 * 1024.0))
    } else if b >= 1024 {
        format!("{:.0} KiB", b as f64 / 1024.0)
    } else {
        format!("{b} B")
    }
}
