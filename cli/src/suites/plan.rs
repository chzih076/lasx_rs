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

use crate::data::{AlignedBuf, Lcg};
use crate::timing::{fmt_t, timeit};
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
        // 极宽 n + 小 k：`matmul_f32` 会选"列块"路径，而计划固定走打包路径（见 dev.md §18）
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
