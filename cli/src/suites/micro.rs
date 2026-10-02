//! 微基准：纯寄存器 FMA 吞吐与多线程扩展性。

use crate::data::{states, velocities, Soa6, J2, MU, RE};
use crate::timing::{fmt_t, timeit};
use lasx_rs::lasx_force_lsx_thread;
use lasx_rs::pool::WorkerPool;
use lasx_rs::*;
use std::arch::loongarch64::*;
use std::hint::black_box;
use std::time::Duration;

/// 一个 SOA 样本块：位置与速度各三分量。
type Soa6Chunk<'a> = (
    &'a mut [f64],
    &'a mut [f64],
    &'a mut [f64],
    &'a mut [f64],
    &'a mut [f64],
    &'a mut [f64],
);

/// FMA 吞吐微基准的迭代次数。
const FMA_ITERS: usize = 20_000_000;

/// 把 n 个样本按线程数切成对齐到 4 样本（32 字节）的块，**每次调用都新建线程**推进一步。
pub fn rk4_parallel_scope(buf: &mut Soa6, threads: usize) {
    let n = buf.len();
    let chunk = n.div_ceil(threads).next_multiple_of(4).max(4);

    let rxv: Vec<&mut [f64]> = buf.rx.chunks_mut(chunk).collect();
    let ryv: Vec<&mut [f64]> = buf.ry.chunks_mut(chunk).collect();
    let rzv: Vec<&mut [f64]> = buf.rz.chunks_mut(chunk).collect();
    let vxv: Vec<&mut [f64]> = buf.vx.chunks_mut(chunk).collect();
    let vyv: Vec<&mut [f64]> = buf.vy.chunks_mut(chunk).collect();
    let vzv: Vec<&mut [f64]> = buf.vz.chunks_mut(chunk).collect();

    let parts: Vec<Soa6Chunk<'_>> = rxv
        .into_iter()
        .zip(ryv)
        .zip(rzv)
        .zip(vxv)
        .zip(vyv)
        .zip(vzv)
        .map(|(((((rx, ry), rz), vx), vy), vz)| (rx, ry, rz, vx, vy, vz))
        .collect();

    std::thread::scope(|s| {
        for (rx, ry, rz, vx, vy, vz) in parts {
            s.spawn(move || {
                let m = rx.len() as i32;
                lasx_force_lsx_thread(false);
                lasx_rk4_j2_step_batch(
                    rx.as_mut_ptr(),
                    ry.as_mut_ptr(),
                    rz.as_mut_ptr(),
                    vx.as_mut_ptr(),
                    vy.as_mut_ptr(),
                    vz.as_mut_ptr(),
                    MU,
                    J2,
                    RE,
                    10.0,
                    m,
                );
            });
        }
    });
}

pub fn thread_scaling() {
    let n = 1 << 18;
    let (rx, ry, rz) = states(n);
    let (vx, vy, vz) = velocities(n);
    let base = Soa6::new(rx, ry, rz, vx, vy, vz);

    let t1 = {
        let mut b = base.clone();
        timeit(|| {
            b.step(false);
            let _ = black_box(b.rx[0]);
        })
    };

    println!();
    println!("## 多线程扩展性：`lasx_rk4_j2_step_batch`（n = 2^18 = {n}，原地单步）");
    println!();
    println!("两种派活方式对比：**每次调用新建线程**（`std::thread::scope`）与");
    println!("**库内常驻池**（`lasx_rs::pool::WorkerPool`：原子代次派活，worker 自适应自旋/休眠，无系统调用）。");
    println!();
    println!("| 线程数 | 每次新建线程 | 库内常驻池 | 池相对 1 线程 | 池效率 |");
    println!("|---|---|---|---|---|");
    println!("| 1 | — | {} | 1.00× | 100% |", fmt_t(t1));

    // 池里有裸指针切分，先对一次结果：池推进一步应与单线程逐位一致
    {
        let mut seq = base.clone();
        let mut par = base.clone();
        seq.step(false);
        par.step_pooled(&WorkerPool::new(8));
        for k in 0..n {
            assert_eq!(
                seq.rx[k].to_bits(),
                par.rx[k].to_bits(),
                "常驻池结果与单线程不一致 @ {k}"
            );
            assert_eq!(seq.vz[k].to_bits(), par.vz[k].to_bits(), "vz 不一致 @ {k}");
        }
    }

    for &th in &[2usize, 4, 6, 8, 12, 24] {
        let mut b = base.clone();
        let d_scope = timeit(|| {
            rk4_parallel_scope(&mut b, th);
            let _ = black_box(b.rx[0]);
        });
        // 池要建一次、复用多次，才能把"建池成本"摊掉（这才是常驻池的用法）
        let pool = WorkerPool::new(th);
        let mut b2 = base.clone();
        let d_pool = timeit(|| {
            b2.step_pooled(&pool);
            let _ = black_box(b2.rx[0]);
        });
        let sp = t1.as_secs_f64() / d_pool.as_secs_f64();
        println!(
            "| {th} | {} | {} | {sp:.2}× | {:.0}% |",
            fmt_t(d_scope),
            fmt_t(d_pool),
            100.0 * sp / th as f64
        );
    }

    let ov = spawn_overhead(24);
    println!();
    println!(
        "> 参考：24 次线程创建/回收的固定开销 ≈ {}（只影响「每次新建线程」一列）",
        fmt_t(ov)
    );
}

/// f32 GEMM 的多线程扩展（`parallel::matmul_f32`，常驻池）。
///
/// 动机见 `docs/dev.md` §21.8：批量形态下 **f32 `lasx_matmul` 比 int8 快 2.5×**，
/// 所以"端到端能不能进分档目标"要按 **f32 + 多核**来算——这张表就是那个乘数。
/// 形状取 §21.2 消费者的三层形状（token = 192）。线程表**不绑核**（§7 开头口径）。
pub fn matmul_thread_scaling() {
    let tokens = 192usize;
    println!();
    println!("## 多线程扩展性：`parallel::matmul_f32`（token = {tokens}，f32 GEMM）");
    println!();
    println!("| 形状 (m×k×n) | 1 线程 | 2 | 4 | 8 | 12 | 24 | 12 线程扩展 | 12 线程效率 |");
    println!("|---|---|---|---|---|---|---|---|---|");
    for &(k, n) in &[(768usize, 768usize), (768, 3072), (3072, 768)] {
        let mut rng = crate::data::Lcg::new((k * 11 + n) as u64);
        let mut a: Vec<f32> = (0..tokens * k).map(|_| rng.f32() * 2.0 - 1.0).collect();
        let b: Vec<f32> = (0..k * n).map(|_| rng.f32() * 2.0 - 1.0).collect();
        let mut c = vec![0f32; tokens * n];
        let t1 = timeit(|| {
            lasx_matmul(
                tokens as i32,
                k as i32,
                n as i32,
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
            );
            let _ = black_box(c[0]);
        });
        let mut cells = Vec::new();
        let mut t12 = Duration::ZERO;
        for &th in &[2usize, 4, 8, 12, 24] {
            let pool = WorkerPool::new(th);
            let d = timeit(|| {
                parallel::matmul_f32(&pool, tokens, k, n, &mut a, &b, &mut c).unwrap();
                let _ = black_box(c[0]);
            });
            if th == 12 {
                t12 = d;
            }
            cells.push(fmt_t(d));
        }
        let sp = t1.as_secs_f64() / t12.as_secs_f64();
        println!(
            "| {tokens}×{k}×{n} | {} | {} | {sp:.2}× | {:.0}% |",
            fmt_t(t1),
            cells.join(" | "),
            100.0 * sp / 12.0
        );
    }
}

#[inline(never)]
pub fn fma_peak_lasx() -> f64 {
    unsafe {
        let a: m256 = std::mem::transmute(lasx_xvreplgr2vr_w(0x3f80_0000));
        let b: m256 = std::mem::transmute(lasx_xvreplgr2vr_w(0x3f00_0000));
        // **16 条独立累加链**：8 条链时量到的是 FMA 延迟上限（≈2.66 条/周期），
        // 16 条才够盖住延迟、量到流水线吞吐（≈3.96 条/周期）。见 docs/dev.md §6.2。
        let (mut v0, mut v1, mut v2, mut v3) = (a, a, a, a);
        let (mut v4, mut v5, mut v6, mut v7) = (a, a, a, a);
        let (mut v8, mut v9, mut v10, mut v11) = (a, a, a, a);
        let (mut v12, mut v13, mut v14, mut v15) = (a, a, a, a);
        for _ in 0..FMA_ITERS {
            v0 = lasx_xvfmadd_s(a, b, v0);
            v1 = lasx_xvfmadd_s(a, b, v1);
            v2 = lasx_xvfmadd_s(a, b, v2);
            v3 = lasx_xvfmadd_s(a, b, v3);
            v4 = lasx_xvfmadd_s(a, b, v4);
            v5 = lasx_xvfmadd_s(a, b, v5);
            v6 = lasx_xvfmadd_s(a, b, v6);
            v7 = lasx_xvfmadd_s(a, b, v7);
            v8 = lasx_xvfmadd_s(a, b, v8);
            v9 = lasx_xvfmadd_s(a, b, v9);
            v10 = lasx_xvfmadd_s(a, b, v10);
            v11 = lasx_xvfmadd_s(a, b, v11);
            v12 = lasx_xvfmadd_s(a, b, v12);
            v13 = lasx_xvfmadd_s(a, b, v13);
            v14 = lasx_xvfmadd_s(a, b, v14);
            v15 = lasx_xvfmadd_s(a, b, v15);
        }
        let s = lasx_xvfadd_s(
            lasx_xvfadd_s(lasx_xvfadd_s(v0, v1), lasx_xvfadd_s(v2, v3)),
            lasx_xvfadd_s(
                lasx_xvfadd_s(lasx_xvfadd_s(v4, v5), lasx_xvfadd_s(v6, v7)),
                lasx_xvfadd_s(
                    lasx_xvfadd_s(lasx_xvfadd_s(v8, v9), lasx_xvfadd_s(v10, v11)),
                    lasx_xvfadd_s(lasx_xvfadd_s(v12, v13), lasx_xvfadd_s(v14, v15)),
                ),
            ),
        );
        let mut tmp = [0f32; 8];
        lasx_xvst(
            std::mem::transmute::<m256, m256i>(s),
            tmp.as_mut_ptr() as *mut i8,
            0,
        );
        black_box(tmp.iter().sum::<f32>()) as f64
    }
}

#[inline(never)]
pub fn fma_peak_lsx() -> f64 {
    unsafe {
        let a: m128 = std::mem::transmute(lsx_vreplgr2vr_w(0x3f80_0000));
        let b: m128 = std::mem::transmute(lsx_vreplgr2vr_w(0x3f00_0000));
        let (mut v0, mut v1, mut v2, mut v3) = (a, a, a, a);
        let (mut v4, mut v5, mut v6, mut v7) = (a, a, a, a);
        let (mut v8, mut v9, mut v10, mut v11) = (a, a, a, a);
        let (mut v12, mut v13, mut v14, mut v15) = (a, a, a, a);
        for _ in 0..FMA_ITERS {
            v0 = lsx_vfmadd_s(a, b, v0);
            v1 = lsx_vfmadd_s(a, b, v1);
            v2 = lsx_vfmadd_s(a, b, v2);
            v3 = lsx_vfmadd_s(a, b, v3);
            v4 = lsx_vfmadd_s(a, b, v4);
            v5 = lsx_vfmadd_s(a, b, v5);
            v6 = lsx_vfmadd_s(a, b, v6);
            v7 = lsx_vfmadd_s(a, b, v7);
            v8 = lsx_vfmadd_s(a, b, v8);
            v9 = lsx_vfmadd_s(a, b, v9);
            v10 = lsx_vfmadd_s(a, b, v10);
            v11 = lsx_vfmadd_s(a, b, v11);
            v12 = lsx_vfmadd_s(a, b, v12);
            v13 = lsx_vfmadd_s(a, b, v13);
            v14 = lsx_vfmadd_s(a, b, v14);
            v15 = lsx_vfmadd_s(a, b, v15);
        }
        let s = lsx_vfadd_s(
            lsx_vfadd_s(lsx_vfadd_s(v0, v1), lsx_vfadd_s(v2, v3)),
            lsx_vfadd_s(
                lsx_vfadd_s(lsx_vfadd_s(v4, v5), lsx_vfadd_s(v6, v7)),
                lsx_vfadd_s(
                    lsx_vfadd_s(lsx_vfadd_s(v8, v9), lsx_vfadd_s(v10, v11)),
                    lsx_vfadd_s(lsx_vfadd_s(v12, v13), lsx_vfadd_s(v14, v15)),
                ),
            ),
        );
        let mut tmp = [0f32; 4];
        lsx_vst(
            std::mem::transmute::<m128, m128i>(s),
            tmp.as_mut_ptr() as *mut i8,
            0,
        );
        black_box(tmp.iter().sum::<f32>()) as f64
    }
}

pub fn fma_peak() {
    // FLOP = 累加器数 × 通道数 × 迭代数 × 2（乘 + 加）
    let flops_lasx = (16 * 8 * FMA_ITERS * 2) as f64;
    let flops_lsx = (16 * 4 * FMA_ITERS * 2) as f64;

    let t_lasx = timeit(|| {
        let _ = black_box(fma_peak_lasx());
    });
    let t_lsx = timeit(|| {
        let _ = black_box(fma_peak_lsx());
    });

    let g_lasx = flops_lasx / t_lasx.as_secs_f64() / 1e9;
    let g_lsx = flops_lsx / t_lsx.as_secs_f64() / 1e9;
    println!();
    println!("## 纯寄存器 FMA 吞吐（无内存访问，**16 条独立累加链**）");
    println!();
    println!("| 路径 | 宽度 | FLOP/s | 相对峰值 |");
    println!("|---|---|---|---|");
    println!("| LASX | 256 位（8×f32） | {g_lasx:.2} GFLOP/s | 1.00× |");
    println!(
        "| LSX | 128 位（4×f32） | {g_lsx:.2} GFLOP/s | {:.2}× |",
        g_lasx / g_lsx
    );
    println!();
    println!(
        "> LASX/LSX 吞吐比 = **{:.2}×**（若 LA664 的 256 位 FMA 为满宽实现应接近 2.00×）",
        g_lasx / g_lsx
    );
    println!(">");
    println!(
        "> 口径说明：链数必须 ≥ 16 才量得到**吞吐**上限；8 条链量到的是 FMA 延迟上限\n\
         > （延迟 ≈3 周期）。这里不写死任何读数——想复核就把 `CHAINS` 改小再跑一次。"
    );
}

/// 线程创建/回收的固定开销（用于校正 24 线程扩展比）
pub fn spawn_overhead(threads: usize) -> Duration {
    timeit(|| {
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    let _ = black_box(1u32);
                });
            }
        });
    })
}

/// 分派开销：每次内核调用都要过的"全局状态"（能力探测缓存 + 线程级强制降级标志）。
///
/// 这两项在热路径上每次调用都要过。当前实现是"写一次的 relaxed 原子读"，相对 `OnceLock`
/// 或 TLS 的读取路径，它把每次调用的成本压到对小规模内核（十几 ns）也可忽略的水平。
///
/// lasx_rs 是**无状态**计算库，热路径上唯一的共享状态就是这两个；这里单独量它们，
/// 用来判断值不值得换成自旋/无锁读。
pub fn dispatch_overhead() {
    use lasx_rs::arch::{hardware, SimdPath};

    let n = 4_000_000usize;
    let per = |f: &mut dyn FnMut()| {
        let t = std::time::Instant::now();
        for _ in 0..n {
            f(); // 被测值由各闭包内部 black_box，这里只需保证循环不被消除
        }
        t.elapsed().as_nanos() as f64 / n as f64
    };

    let empty = per(&mut || {
        black_box(0u8);
    });
    let detect = per(&mut || {
        black_box(SimdPath::detect());
    });
    let hw = per(&mut || {
        black_box(hardware());
    });

    println!();
    println!("## 分派开销（每次内核调用都要过的全局状态）");
    println!();
    println!("| 项 | ns/次 | 相对空循环 |");
    println!("|---|---|---|");
    println!("| 空循环基线 | {empty:.2} | 1.00× |");
    println!(
        "| `SimdPath::detect()`（原子读 + 线程级 TLS） | {detect:.2} | {:.1}× |",
        detect / empty
    );
    println!(
        "| `hardware()`（仅原子读） | {hw:.2} | {:.1}× |",
        hw / empty
    );
    println!();
    println!(
        "> 这三次取值的量级就是**每次内核调用都要付的固定开销**：规模越小，它占的比例越高。\n\
         > 具体到 `lasx_dot` 的绝对值看 `dot` 套件（本页不再写死数字——写死过一次，很快就过期了）。"
    );
}

/// 池**在核数附近**的表现：`docs/dev.md` §7.9 要求的那一步测量。
///
/// 为什么要单独一条：§7.9 那张 f32 GEMM 表在 `load ≈1.6–1.9` 的窗口里出现过"8/12 线程比
/// 4 线程慢"，首轮复测却是单调的。要把它判成"池的缺陷"还是"邻居负载造成的超订"，
/// 必须**记录负载**并做对照。所以本函数的输出**第一行就是 `/proc/loadavg`**，
/// 线程数扫得比 §7.9 那张表更密（1,2,3,4,6,8,10,12,16,24）——判据是：
/// **在"物理核数 − 邻居数"以内是否单调**。当前 12 物理核、load ≈2 ⇒ 10 是分界。
///
/// 用法：高负载窗口与低负载窗口各跑 ≥3 轮，两份输出对照。
pub fn pool_scaling() {
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    let cores = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1);
    println!(
        "\n## 池在核数附近的表现（逻辑核 {cores}，loadavg = {}）",
        load.trim()
    );
    println!("\n| 形状 (m×k×n) | 线程 | 时间 | 扩展 | 效率 |");
    println!("|---|---|---|---|---|");
    let tokens = 192usize;
    for &(k, n) in &[(768usize, 3072usize), (3072, 768)] {
        let mut rng = crate::data::Lcg::new((k * 17 + n) as u64);
        let a: Vec<f32> = (0..tokens * k).map(|_| rng.f32() * 2.0 - 1.0).collect();
        let b: Vec<f32> = (0..k * n).map(|_| rng.f32() * 2.0 - 1.0).collect();
        let mut c = vec![0f32; tokens * n];
        let mut base = Duration::ZERO;
        for &th in &[1usize, 2, 3, 4, 6, 8, 10, 12, 16, 24] {
            let pool = WorkerPool::new(th);
            let mut a = a.clone();
            let d = timeit(|| {
                parallel::matmul_f32(&pool, tokens, k, n, &mut a, &b, &mut c).unwrap();
                let _ = black_box(c[0]);
            });
            if th == 1 {
                base = d;
            }
            let sp = base.as_secs_f64() / d.as_secs_f64();
            println!(
                "| {tokens}×{k}×{n} | {th} | {} | {sp:.2}× | {:.0}% |",
                fmt_t(d),
                100.0 * sp / th as f64
            );
        }
    }
}

///
/// `parallel::gemv_i8_k`（**k 方向切分**，为 `m = 1` 的单 token 档位而做）的线程扩展。
///
/// **判别实验**：8 线程处那次系统性下降，是"缓存/L3 争用"还是"调度/块粒度"？
/// 两个只读对照（见 `docs/dev.md` §7.9）：① 换工作集（9.4 MB → 512 KB）；② 固定 8 线程换 `Pick`。
pub fn pool_attribution() {
    use lasx_rs::pool::Pick;
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    println!("\n## 8 线程下降的判别实验（loadavg = {}）", load.trim());
    let tokens = 192usize;

    println!("\n### ① 换工作集（默认 pick，线程 4/6/8/12）");
    println!("\n| 形状 | 权重(B) | 4 | 6 | 8 | 12 |");
    println!("|---|---|---|---|---|---|");
    for &(k, n) in &[(256usize, 512usize), (768, 3072)] {
        let mut rng = crate::data::Lcg::new((k * 23 + n) as u64);
        let a0: Vec<f32> = (0..tokens * k).map(|_| rng.f32() * 2.0 - 1.0).collect();
        let b: Vec<f32> = (0..k * n).map(|_| rng.f32() * 2.0 - 1.0).collect();
        let mut c = vec![0f32; tokens * n];
        let mut cells = Vec::new();
        for &th in &[4usize, 6, 8, 12] {
            let pool = WorkerPool::new(th);
            let mut a = a0.clone();
            let d = timeit(|| {
                parallel::matmul_f32(&pool, tokens, k, n, &mut a, &b, &mut c).unwrap();
                let _ = black_box(c[0]);
            });
            cells.push(fmt_t(d));
        }
        println!(
            "| {tokens}×{k}×{n} | {} KiB | {} |",
            (k * n * 4) / 1024,
            cells.join(" | ")
        );
    }

    println!("\n### ② 换调度策略（固定 8 线程，`192×768×3072`）");
    println!("\n| 策略 | 时间 | 相对默认 |");
    println!("|---|---|---|");
    let (k, n) = (768usize, 3072usize);
    let mut rng = crate::data::Lcg::new(0xbeef);
    let a0: Vec<f32> = (0..tokens * k).map(|_| rng.f32() * 2.0 - 1.0).collect();
    let b: Vec<f32> = (0..k * n).map(|_| rng.f32() * 2.0 - 1.0).collect();
    let mut c = vec![0f32; tokens * n];
    let pool = WorkerPool::new(8);
    let mut base = Duration::ZERO;
    for (name, pick) in [
        ("默认 pick_rows", None),
        ("RowBlock（每线程一块）", Some(Pick::RowBlock)),
        ("Blocked{4}", Some(Pick::Blocked { block_rows: 4 })),
        ("Dynamic{4}", Some(Pick::Dynamic { block_rows: 4 })),
    ] {
        let mut a = a0.clone();
        let d = timeit(|| {
            match pick {
                None => parallel::matmul_f32(&pool, tokens, k, n, &mut a, &b, &mut c),
                Some(p) => {
                    parallel::matmul_f32_with_pick(&pool, tokens, k, n, &mut a, &b, &mut c, p)
                }
            }
            .unwrap();
            let _ = black_box(c[0]);
        });
        if base == Duration::ZERO {
            base = d;
        }
        println!(
            "| {name} | {} | {:.2}× |",
            fmt_t(d),
            base.as_secs_f64() / d.as_secs_f64()
        );
    }
}

/// 动机：int8 唯一占优的档位是"单 token、权重流 DRAM"（`docs/ops.md` §5.15），那一档 `m = 1`
/// ⇒ 按行切分没有并行度；本函数量的是"按 `k` 切块"能拿回多少。
/// 三个 `k` 分别代表：模型单层（3072，L2 内）、1 MiB（L2 边缘）、16 MiB（DRAM 流）。
/// 线程表**不绑核**（§7 开头口径）。
pub fn gemv_i8_k_thread_scaling() {
    println!();
    println!("## 多线程扩展性：`parallel::gemv_i8_k`（`m = 1`，k 方向切块，int8 GEMV）");
    println!();
    println!("| k（权重字节） | 1 线程 | 2 | 4 | 8 | 12 | 24 | 12 线程扩展 |");
    println!("|---|---|---|---|---|---|---|---|");
    for &k in &[3072usize, 1 << 20, 1 << 24] {
        let mut rng = crate::data::Lcg::new(k as u64);
        let w: Vec<i8> = (0..k).map(|_| rng.i8()).collect();
        let x: Vec<i8> = (0..k).map(|_| rng.i8()).collect();
        let sw = [0.001f32];
        let mut y = [0f32; 1];
        // 池建在计时闭包**外面**：建池本身几十 µs，第一版把 `WorkerPool::new(1)` 写在闭包里，
        // 于是"1 线程"那列混进了建池开销，跑出 200× 的假加速（探针要先证明自己在测什么）
        let pool1 = WorkerPool::new(1);
        let t1 = timeit(|| {
            parallel::gemv_i8_k(&pool1, 1, k, &w, &sw, &x, 0.002, &mut y).unwrap();
            let _ = black_box(y[0]);
        });
        let mut cells = Vec::new();
        let mut t12 = Duration::ZERO;
        for &th in &[2usize, 4, 8, 12, 24] {
            let pool = WorkerPool::new(th);
            let d = timeit(|| {
                parallel::gemv_i8_k(&pool, 1, k, &w, &sw, &x, 0.002, &mut y).unwrap();
                let _ = black_box(y[0]);
            });
            if th == 12 {
                t12 = d;
            }
            cells.push(fmt_t(d));
        }
        println!(
            "| {k}（{} KiB） | {} | {} | {:.2}× |",
            k / 1024,
            fmt_t(t1),
            cells.join(" | "),
            t1.as_secs_f64() / t12.as_secs_f64()
        );
    }
}
