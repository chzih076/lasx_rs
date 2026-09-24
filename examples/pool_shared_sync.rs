//! `&self` 的同步开销与池的起效档位：给上层 `Auto` 判据提供输入。
//!
//! 三个问题：
//! 1. `&self` 之后每次派活必然多一层同步（`OnceLock::get` + 原子，最坏还要一把互斥）。
//!    它在 GEMM 的调用频率下可见吗？
//! 2. 可见性的边界在哪——**更小的 DYN 场景**（`M=16, K=N=64`，单次约 2 µs）同步占比多少？
//! 3. 池从哪一档开始值？判据是"看行数"还是"看总工作量"？
//!
//! `cargo run --release --example pool_shared_sync`

use lasx_rs::parallel;
use lasx_rs::plan::MatmulPlan;
use lasx_rs::pool::WorkerPool;
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// 自适应重复 + 5 取样取中位数（与 `lasx_bench` 同口径，避免单次取样骗人）。
fn timeit<F: FnMut()>(mut f: F) -> Duration {
    f();
    f();
    let t = Instant::now();
    f();
    let single = t.elapsed().max(Duration::from_nanos(1));
    let reps =
        (Duration::from_millis(20).as_nanos() / single.as_nanos()).clamp(1, 50_000_000) as u32;
    let mut v = Vec::with_capacity(5);
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        v.push(t.elapsed() / reps);
    }
    v.sort_unstable();
    v[2]
}

fn ns(d: Duration) -> f64 {
    d.as_secs_f64() * 1e9
}

/* ---------- 候选同步原语 ---------- */

static ONCE: OnceLock<usize> = OnceLock::new();
static COUNTER: AtomicUsize = AtomicUsize::new(0);
static LOCK: Mutex<()> = Mutex::new(());

/// 最坏情况：一次派活要付的全部同步（全局池取一次 + 一个序号原子 + 一把互斥）。
#[inline]
fn sync_worst_case() -> usize {
    let id = *ONCE.get_or_init(|| 1);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let _guard = LOCK.lock().unwrap();
    id + seq
}

/// 乐观情况：只用原子，不加锁。
#[inline]
fn sync_atomic_only() -> usize {
    let id = *ONCE.get_or_init(|| 1);
    id + COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// 造一对 `(a, b)` 输入。
fn inputs(m: usize, k: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    (
        (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect(),
        (0..k * n).map(|i| ((i % 13) as f32 - 6.0) * 0.25).collect(),
    )
}

/// 形状清单：小 DYN 场景、以及"行很多但每行很薄"（用来分离两个判据）。
const SHAPES: &[(usize, usize, usize)] = &[
    (16, 64, 64),   // 单次约 2 µs：同步开销开始可见的那一档
    (64, 64, 64),   // 约 34 µs 的四分之一
    (256, 64, 64),  // 行不少、每行薄
    (1024, 64, 64), // 行很多、每行薄
    (1024, 8, 8),   // 行极多、工作量极小：判据"看行数"还是"看工作量"的分水岭
    (16, 256, 256), // 工作量与 64³ 同量级，但行数少
    (32, 256, 256), // 已知盈亏平衡点
    (64, 256, 256),
    (256, 256, 256),
    (1024, 256, 256),
];

fn main() {
    println!("### `&self` 的同步单价、可见性边界，以及池的起效档位");
    println!();

    /* ============ 1. 同步原语单价 ============ */
    let t_once = timeit(|| {
        black_box(ONCE.get_or_init(|| 1));
    });
    let t_atomic = timeit(|| {
        black_box(COUNTER.fetch_add(1, Ordering::Relaxed));
    });
    let t_lock = timeit(|| {
        let g = LOCK.lock().unwrap();
        drop(g);
    });
    let t_atomic_only = timeit(|| {
        black_box(sync_atomic_only());
    });
    let t_worst = timeit(|| {
        black_box(sync_worst_case());
    });
    // `Auto` 策略每次调用都会问一遍机器并行度——这是系统调用，必须量它的单价
    let t_avail = timeit(|| {
        black_box(
            std::thread::available_parallelism()
                .map(|v| v.get())
                .unwrap_or(1),
        );
    });

    println!("| 同步动作 | 每次耗时 |");
    println!("|---|---|");
    println!(
        "| `OnceLock::get_or_init`（已初始化） | {:.1} ns |",
        ns(t_once)
    );
    println!(
        "| `AtomicUsize::fetch_add(Relaxed)` | {:.1} ns |",
        ns(t_atomic)
    );
    println!("| `Mutex::lock/unlock`（无竞争） | {:.1} ns |", ns(t_lock));
    println!(
        "| `thread::available_parallelism()`（**系统调用**） | {:.1} ns |",
        ns(t_avail)
    );
    println!(
        "| 合计：原子版（OnceLock+原子） | {:.1} ns |",
        ns(t_atomic_only)
    );
    println!("| 合计：最坏版（再加一把互斥） | {:.1} ns |", ns(t_worst));
    println!();

    /* ============ 2. 端到端：加在 DYN 热路径上 ============ */
    let pool_threads = 12;
    println!("「+同步」= 每次调用前先付一遍最坏版同步；单线程热路径是 `MatmulPlan::run_into`。");
    println!();
    println!("| M×K×N | 干净 | +同步（原子版） | +同步（最坏版） | 最坏版占比 |");
    println!("|---|---|---|---|---|");

    for &(m, k, n) in SHAPES {
        let (a, b) = inputs(m, k, n);
        let mut c = vec![0f32; m * n];
        let plan = MatmulPlan::from_row_major(&b, k, n).unwrap();

        let clean = timeit(|| {
            plan.run_into(&a, &mut c).unwrap();
            black_box(c[0]);
        });
        let with_atomic = timeit(|| {
            black_box(sync_atomic_only());
            plan.run_into(&a, &mut c).unwrap();
            black_box(c[0]);
        });
        let with_worst = timeit(|| {
            black_box(sync_worst_case());
            plan.run_into(&a, &mut c).unwrap();
            black_box(c[0]);
        });
        println!(
            "| {m}×{k}×{n} | {:.2} µs | {:.2} µs | {:.2} µs | {:.2}% |",
            ns(clean) / 1e3,
            ns(with_atomic) / 1e3,
            ns(with_worst) / 1e3,
            (ns(with_worst) - ns(clean)) / ns(with_worst) * 100.0
        );
    }
    println!();

    /* ============ 3. 池从哪一档开始值 ============ */
    println!(
        "同一形状：单线程（`MatmulPlan`）vs `{pool_threads}` 线程池（`parallel::matmul_f32`）。"
    );
    println!("两列结果逐位比对，确保比的是同一份工作；工作量 = `M×K×N`。");
    println!();
    println!("| M×K×N | 工作量 | 单线程 | {pool_threads} 线程 | 池/单线程 |");
    println!("|---|---|---|---|---|");

    let pool = WorkerPool::new(pool_threads);
    for &(m, k, n) in SHAPES {
        let (a, b) = inputs(m, k, n);
        let mut a_par = a.clone();
        let mut c = vec![0f32; m * n];
        let mut c_par = vec![0f32; m * n];
        let plan = MatmulPlan::from_row_major(&b, k, n).unwrap();

        let serial = timeit(|| {
            plan.run_into(&a, &mut c).unwrap();
            black_box(c[0]);
        });
        let par = timeit(|| {
            parallel::matmul_f32(&pool, m, k, n, &mut a_par, &b, &mut c_par).unwrap();
            black_box(c_par[0]);
        });
        assert_eq!(
            c.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            c_par.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "M×K×N={m}×{k}×{n}：池化与单线程不逐位一致"
        );
        println!(
            "| {m}×{k}×{n} | {:>8} kFLOP | {:.2} µs | {:.2} µs | {:.2}× |",
            m * k * n / 1000,
            ns(serial) / 1e3,
            ns(par) / 1e3,
            ns(par) / ns(serial)
        );
    }
    println!();
    println!("（倍数 < 1 表示池更快；≈1 或 > 1 表示这一档并行不划算。）");
}
