//! 微基准：纯寄存器 FMA 吞吐与多线程扩展性。

use crate::data::{J2, MU, RE, Soa6, states, velocities};
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
    println!(
        "**库内常驻池**（`lasx_rs::pool::WorkerPool`：原子代次派活，worker 自适应自旋/休眠，无系统调用）。"
    );
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
    use lasx_rs::arch::{SimdPath, hardware};

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

    // 分配的固定开销：`AlignedVec::new(n)` = 分配 + 清零（`vec![T::default(); n + 对齐余量]`）。
    // 这就是"每次调用分配输出"的 api 在**调用方改用 `_into` 后能省掉的那笔**
    // （对照数据与筛选结论见 `docs/dev.md` §7.9 的 `_into` 一节）。
    println!("\n### 输出缓冲的分配成本（`AlignedVec::new`：分配 + 清零）");
    println!("\n| 元素数（`f32`） | 字节 | 时间 | 等效带宽 |");
    println!("|---|---|---|---|");
    for &n in &[147_456usize, 589_824, 1_048_576, 4_194_304, 8_388_608] {
        let d = timeit(|| {
            let v = lasx_rs::aligned::AlignedVec::<f32>::new(n);
            let _ = black_box(v[0]);
        });
        let bytes = (n * 4) as f64;
        println!(
            "| {n} | {:.1} MiB | {} | {:.1} GB/s |",
            bytes / (1024.0 * 1024.0),
            crate::timing::fmt_t(d),
            bytes / d.as_secs_f64() / 1e9
        );
    }

    // `_into` 候选的 A/B：**api（每次分配输出）vs 同内核的 FFI（调用方给缓冲）**。
    // 两者是同一个内核，差的就是"分配 + 清零 + 释放" ⇒ 这条 A/B 直接给出分配占比，
    // 不需要先写 `_into`（`docs/dev.md` §7.9 ③ 筛选口径：先测后铺）。
    println!("\n### `_into` 候选：api（分配） vs FFI（预分配）");
    println!("\n| 算子 | 规模 | api（分配） | FFI（预分配） | 分配占比 |");
    println!("|---|---|---|---|---|");
    let act_n = 192 * 3072;
    let xa = vec![0.5f32; act_n];
    let mut ya = vec![0f32; act_n];
    let rows: [(&str, usize); 3] = [("`silu`", 0), ("`gelu_quick`", 1), ("`gelu_erf`", 2)];
    for (name, which) in rows {
        let a = timeit(|| {
            let v = match which {
                0 => lasx_rs::api::silu(&xa),
                1 => lasx_rs::api::gelu_quick(&xa),
                _ => lasx_rs::api::gelu_erf(&xa),
            };
            let _ = black_box(v[0]);
        });
        let f = timeit(|| {
            match which {
                0 => lasx_silu(xa.as_ptr(), ya.as_mut_ptr(), act_n as i32),
                1 => lasx_gelu_quick(xa.as_ptr(), ya.as_mut_ptr(), act_n as i32),
                _ => lasx_gelu_erf(xa.as_ptr(), ya.as_mut_ptr(), act_n as i32),
            }
            let _ = black_box(ya[0]);
        });
        println!(
            "| {name} | {act_n} | {} | {} | **{:.0}%** |",
            crate::timing::fmt_t(a),
            crate::timing::fmt_t(f),
            100.0 * a.saturating_sub(f).as_secs_f64() / a.as_secs_f64()
        );
    }
    for &(m, k, n) in &[(192usize, 768usize, 3072usize), (192, 768, 768)] {
        let am = vec![0.01f32; m * k];
        let bm = vec![0.01f32; k * n];
        let mut cm = vec![0f32; m * n];
        let a = timeit(|| {
            let v = lasx_rs::api::matmul(m, k, n, &am, &bm).unwrap();
            let _ = black_box(v[0]);
        });
        let f = timeit(|| {
            lasx_matmul(
                m as i32,
                k as i32,
                n as i32,
                am.as_ptr(),
                bm.as_ptr(),
                cm.as_mut_ptr(),
            );
            let _ = black_box(cm[0]);
        });
        println!(
            "| `matmul` | {m}×{k}×{n} | {} | {} | **{:.0}%** |",
            crate::timing::fmt_t(a),
            crate::timing::fmt_t(f),
            100.0 * a.saturating_sub(f).as_secs_f64() / a.as_secs_f64()
        );
    }
    // `rope`：输出 = rows×cols（与激活同量级）；`gemv_f16`：输出只有 m 个 f32。
    {
        let (rows, cols, n_dims) = (192usize, 768usize, 128usize);
        let mut rng = crate::data::Lcg::new(0x1234_5678_u64);
        let xr: Vec<f32> = (0..rows * cols).map(|_| rng.f32() * 2.0 - 1.0).collect();
        let half = rows * (n_dims / 2);
        let cos: Vec<f32> = (0..half).map(|_| rng.f32() * 2.0 - 1.0).collect();
        let sin: Vec<f32> = (0..half).map(|_| rng.f32() * 2.0 - 1.0).collect();
        let mut out = vec![0f32; rows * cols];
        let mode = lasx_rs::api::RopeMode::NeoX as i32;
        let a = timeit(|| {
            let v = lasx_rs::api::rope(
                &xr,
                &cos,
                &sin,
                rows,
                cols,
                n_dims,
                lasx_rs::api::RopeMode::NeoX,
            )
            .unwrap();
            let _ = black_box(v[0]);
        });
        let f = timeit(|| {
            lasx_rope(
                xr.as_ptr(),
                cos.as_ptr(),
                sin.as_ptr(),
                out.as_mut_ptr(),
                rows as i32,
                cols as i32,
                n_dims as i32,
                mode,
            );
            let _ = black_box(out[0]);
        });
        println!(
            "| `rope` | {rows}×{cols}（n_dims={n_dims}） | {} | {} | **{:.0}%** |",
            crate::timing::fmt_t(a),
            crate::timing::fmt_t(f),
            100.0 * a.saturating_sub(f).as_secs_f64() / a.as_secs_f64()
        );
        // `rope_at`（**每次调用重算表**：2 次分配 + 2 次逐元素拷贝）vs 预建表 + `rope`（1 次分配）。
        // 这不是"分配 vs 预分配"，而是**用法级**对照：表该不该由调用方 hoist。
        let positions: Vec<f32> = (0..rows).map(|i| i as f32 * 0.5).collect();
        let (ctab, stab) = lasx_rs::api::rope_tables(&positions, n_dims, 10_000.0).unwrap();
        let mode2 = lasx_rs::api::RopeMode::NeoX;
        let at_call = timeit(|| {
            let v = lasx_rs::api::rope_at(&xr, &positions, rows, cols, n_dims, 10_000.0, mode2)
                .unwrap();
            let _ = black_box(v[0]);
        });
        let hoisted = timeit(|| {
            let v = lasx_rs::api::rope(&xr, &ctab, &stab, rows, cols, n_dims, mode2).unwrap();
            let _ = black_box(v[0]);
        });
        println!(
            "| `rope_at` vs 预建表 `rope` | {rows}×{cols} | 每次重算 {} | 预建表 {} | **{:.0}%** |",
            crate::timing::fmt_t(at_call),
            crate::timing::fmt_t(hoisted),
            100.0 * at_call.saturating_sub(hoisted).as_secs_f64() / at_call.as_secs_f64()
        );
        // 并行建表值不值得：先量**线程启动开销的下界**（12 个空线程 spawn+join）。`rope_tables`
        // 的活儿只有 ~232 µs，若启动开销与它同量级，并行就是白搭（每元素独立 ⇒ 位精确与线程数
        // 无关，所以"安不安全"不是问题，**"划不划算"才是**）。
        let spawn = timeit(|| {
            std::thread::scope(|s| {
                let hs: Vec<_> = (0..12).map(|_| s.spawn(|| {})).collect();
                for h in hs {
                    h.join().unwrap();
                }
            });
        });
        let serial_tables = timeit(|| {
            let t = lasx_rs::api::rope_tables(&positions, n_dims, 10_000.0).unwrap();
            let _ = black_box(t.0[0]);
        });
        println!(
            "| 串行 `rope_tables`（{rows} 位置）vs 12 线程空 spawn+join | — | 串行 {} | 启动下界 {} | — |",
            crate::timing::fmt_t(serial_tables),
            crate::timing::fmt_t(spawn)
        );
        // **池的派发开销**（常驻线程；上一轮量的 `std::thread` 不是这个）。空闭包 + 每线程一个块。
        {
            let pool = lasx_rs::pool::WorkerPool::auto();
            let nt = pool.threads();
            // **真正的派发开销 = (24 线程跑大块) − (1 线程跑同量大块)/24 的理想值**。上一版用
            // 小数据（24×64 B）测出 16 ns——那不是派发成本：**数据太小时池不派发、直接内联执行**
            // （这本身是个好设计，但用"小数据"量不到派发）。这里用 24 MB 让 24 个线程都真动起来。
            let mut big = vec![0u8; 24 * 1_000_000];
            let p1 = lasx_rs::pool::WorkerPool::new(1);
            let t1 = timeit(|| {
                p1.for_each_chunk_mut(&mut big, |c| {
                    for b in c.iter_mut() {
                        *b = 3;
                    }
                });
                let _ = black_box(big[0]);
            });
            let t24 = timeit(|| {
                pool.for_each_chunk_mut(&mut big, |c| {
                    for b in c.iter_mut() {
                        *b = 3;
                    }
                });
                let _ = black_box(big[0]);
            });
            println!(
                "| 池：24 MB 写（{nt} 线程） | 1 线程 {} | {nt} 线程 {} | 理想 1/{nt} = {}；差含带宽饱和 |",
                crate::timing::fmt_t(t1),
                crate::timing::fmt_t(t24),
                crate::timing::fmt_t(t1 / nt as u32)
            );
            // 串行建表的**规模扫描**：池能不能回本取决于"工作 / 派发"之比。
            for &np in &[192usize, 4096, 16384] {
                let ps: Vec<f32> = (0..np).map(|i| i as f32 * 0.5).collect();
                let t = timeit(|| {
                    let v = lasx_rs::api::rope_tables(&ps, n_dims, 10_000.0).unwrap();
                    let _ = black_box(v.0[0]);
                });
                println!(
                    "| 串行 `rope_tables` | {np} 位置 × {n_dims} | — | {} | — |",
                    crate::timing::fmt_t(t)
                );
            }
        }
        // **DSL vs 直接调用**：两臂都写**预分配**缓冲（`MatBuf` / `lasx_matmul` 的 `c` 指针），
        // 唯一差别是"宏生成的代码 vs 手写直接调用"。**不能用 `api::matmul`**——它分配，
        // 那测出来的是"分配开销"而不是 DSL 开销（本库 matmul 没有 `_into` 变体）。
        // **对照臂改用 `parallel::matmul_f32`（池化），而不是 `lasx_matmul`（串行 FFI）**：
        // 只有这样才能回答"DSL 会不会漏掉并行"（256³ 串行 ~533 µs、12 线程应 ~100 µs 量级）。
        // const 形状取 256³：`MatBuf` 是 **inline 数组**，256×256×4 = 256 KB 栈上安全；
        // 再大（如 192×3072 = 2.36 MB）有栈溢出风险，得换 `MatDyn`（但那就不再是纯 const 形状）。
        {
            use lasx_rs::matmul;
            use lasx_rs::shape::{Mat, MatBuf};
            const M: usize = 256;
            const K: usize = 256;
            const N: usize = 256;
            let wv: Vec<f32> = (0..K * N).map(|i| (i % 13) as f32).collect();
            let xv2: Vec<f32> = (0..M * K).map(|i| (i % 7) as f32).collect();
            let w = Mat::<f32, { K }, { N }>::new(&wv).unwrap();
            let x = Mat::<f32, { M }, { K }>::new(&xv2).unwrap();
            let mut yb = MatBuf::<f32, { M }, { N }>::new();
            let mut yc = vec![0f32; M * N];
            let pool12 = WorkerPool::new(12);
            let pool24 = WorkerPool::new(24);
            let poolg = lasx_rs::pool::global();
            let mut xm = xv2.clone();
            let dsl = timeit(|| {
                matmul!(yb[M, N] = x[M, K] * w[K, N]);
                let _ = black_box(yb.as_slice()[0]);
            });
            let direct = timeit(|| {
                parallel::matmul_f32(&pool12, M, K, N, &mut xm, &wv, &mut yc).unwrap();
                let _ = black_box(yc[0]);
            });
            // **池线程数三臂对照**：全局池（本机 24 线程，DSL 走的就是它）vs 显式 12 vs 显式 24。
            let t24 = timeit(|| {
                parallel::matmul_f32(&pool24, M, K, N, &mut xm, &wv, &mut yc).unwrap();
                let _ = black_box(yc[0]);
            });
            let tg = timeit(|| {
                parallel::matmul_f32(poolg, M, K, N, &mut xm, &wv, &mut yc).unwrap();
                let _ = black_box(yc[0]);
            });
            println!(
                "| 池线程数 @ {M}³ | 12 线程 {} | 24 线程 {} | **全局（{} 线程）{}** | — |",
                crate::timing::fmt_t(direct),
                crate::timing::fmt_t(t24),
                poolg.threads(),
                crate::timing::fmt_t(tg)
            );
            // **第三臂：`prepare()` 一次 + `apply_into` 复用**——库文档在 `apply_dyn_into` 上写着
            // "内部仍会打包一次；**反复用请走 `Mat::prepare`**"。`matmul!` 生成的 `mul_into` 每次
            // 都 `rhs.prepare().apply_into(...)`（`src/shape.rs:488`）⇒ 每次调用打包一次 B。
            // 这一臂量"把准备阶段 hoist 出循环"能省多少（与 RoPE 的 hoist 表同型）。
            let plan = w.prepare();
            let reused = timeit(|| {
                plan.apply_into(&x, &mut yb);
                let _ = black_box(yb.as_slice()[0]);
            });
            println!(
                "| **准备阶段 hoist**（{M}³） | `matmul!` 每次 {} | `prepare()` 复用 {} | 省 **{:.0}%** | 池化 {} |",
                crate::timing::fmt_t(dsl),
                crate::timing::fmt_t(reused),
                100.0 * (dsl.as_secs_f64() - reused.as_secs_f64()).max(0.0) / dsl.as_secs_f64(),
                crate::timing::fmt_t(direct)
            );
            // **派发/屏障开销的判别实验**（§7.11 ⑩ 的调度假设）：**同一份活儿**（192 行 × 3072 列）
            // 切成 **N 次** `for_each_row_block_mut` 派发 vs **1 次**。`matmul_f32_picked` 是
            // **逐面板派发**（每面板一次整队 + 屏障），所以"每次派发的固定成本"直接乘上面板数。
            // 闭包用 `black_box` 保证真跑（别再用"空闭包"——那会被优化掉，我早前踩过）。
            {
                let (nr, stride) = (192usize, 3072usize);
                let mut buf = vec![0f32; nr * stride];
                let pool12 = WorkerPool::new(12);
                for &nseg in &[1usize, 4, 8, 16, 24] {
                    let rows_per = nr / nseg;
                    let t = timeit(|| {
                        for s in 0..nseg {
                            let off = s * rows_per * stride;
                            let seg = &mut buf[off..off + rows_per * stride];
                            pool12.for_each_row_block_mut(
                                rows_per,
                                4,
                                [(seg, stride)],
                                |_start, _rows, [d]| {
                                    let _ = black_box(d[0]);
                                },
                            );
                        }
                    });
                    println!(
                        "| 派发切分 | **{nseg} 次派发**（每次 {rows_per} 行，12 线程） | {} | — |",
                        crate::timing::fmt_t(t)
                    );
                }
            }
            // **打包成本单独量**（公开 API）：m 取 4 ⇒ 计算只有 4×768×3072（可忽略），而 B 仍是
            // 768×3072 = **9.44 MB** ⇒ `matmul_f32_packed` 的时间 ≈ **打包 B 一遍的成本**。
            // 这是判断"主线程打包 vs worker 计算的重叠（双缓冲）够不够"的关键输入：
            // 若打包 ≥ 单面板计算，worker 在面板边界必然空转（调度问题的一种）。
            {
                let (mp, kp, np) = (4usize, 768usize, 3072usize);
                let ap: Vec<f32> = (0..mp * kp).map(|i| (i % 17) as f32 * 0.0625).collect();
                let bp: Vec<f32> = (0..kp * np).map(|i| (i % 13) as f32 * 0.0625).collect();
                let stream = timeit(|| {
                    let v = lasx_rs::api::matmul(mp, kp, np, &ap, &bp).unwrap();
                    let _ = black_box(v[0]);
                });
                let packed = timeit(|| {
                    let v = lasx_rs::api::matmul_f32_packed(mp, kp, np, &ap, &bp).unwrap();
                    let _ = black_box(v[0]);
                });
                println!(
                    "| **打包 B 的成本**（m=4、B=9.44 MB） | 自动路径 {} | `f32_packed` {} | 差 **{:.0}%** | — |",
                    crate::timing::fmt_t(stream),
                    crate::timing::fmt_t(packed),
                    100.0 * (packed.as_secs_f64() - stream.as_secs_f64()).abs()
                        / stream.as_secs_f64()
                );
            }
            // 位精确对拍：两条路必须逐位一致（否则它们不是同一个内核）
            let mut bits_ok = true;
            for (i, v) in yc.iter().enumerate() {
                if yb.as_slice()[i].to_bits() != v.to_bits() {
                    bits_ok = false;
                    break;
                }
            }
            println!(
                "| DSL `matmul!` vs **池化** `parallel::matmul_f32` | {M}×{K}×{N} | DSL {} | 池化 {} | 差 **{:.1}%**（逐位一致：{bits_ok}） |",
                crate::timing::fmt_t(dsl),
                crate::timing::fmt_t(direct),
                100.0 * (dsl.as_secs_f64() - direct.as_secs_f64()).abs() / dsl.as_secs_f64()
            );
        }
        // **"真标量"基线**：`scalar_ref.rs` 那一列在 `+lasx` 下会被 LLVM **自动向量化**，
        // 所以它回答的是"手写 intrinsic vs 编译器自动向量化"。这里用逐元素 `black_box`
        // 掐断向量化（编译器屏障，运行期是 no-op）⇒ 这才是"vs 朴素循环"的分母。
        {
            /// 朴素三重循环 GEMM（阻止自动向量化；也不允许 FMA 收缩）。
            #[inline(never)]
            fn naive_gemm(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
                for i in 0..m {
                    for j in 0..n {
                        let mut s = 0f32;
                        for p in 0..k {
                            s = black_box(s) + black_box(a[i * k + p]) * black_box(b[p * n + j]);
                        }
                        c[i * n + j] = s;
                    }
                }
            }
            let n = 256usize;
            let av: Vec<f32> = (0..n * n).map(|i| (i % 17) as f32 * 0.0625).collect();
            let bv: Vec<f32> = (0..n * n).map(|i| (i % 13) as f32 * 0.0625).collect();
            let mut cv = vec![0f32; n * n];
            let mut cv2 = vec![0f32; n * n];
            let naive = timeit(|| {
                naive_gemm(n, n, n, &av, &bv, &mut cv);
                let _ = black_box(cv[0]);
            });
            let simd = timeit(|| {
                lasx_matmul(
                    n as i32,
                    n as i32,
                    n as i32,
                    av.as_ptr(),
                    bv.as_ptr(),
                    cv2.as_mut_ptr(),
                );
                let _ = black_box(cv2[0]);
            });
            println!(
                "| 真标量 vs LASX（GEMM {n}³） | — | 朴素三重循环 {} | LASX {} | **{:.1}×** |",
                crate::timing::fmt_t(naive),
                crate::timing::fmt_t(simd),
                naive.as_secs_f64() / simd.as_secs_f64()
            );
        }
        // **窄列分解本身是否低效？**（串行、只用公开 API）：1 次宽 `n` vs 32 次窄 `n/32` 的**同量工作**。
        // 这是把"窄列"从"并发"里单独拎出来的判别实验（§7.11 ⑥ 的 12× 归因需要它）。
        // 注意口径：窄臂把 B 的列段**在计时外**拷成连续缓冲（否则拷 B 的成本会混进来）；两臂都含分配。
        {
            let (m, k, n, seg) = (192usize, 768usize, 3072usize, 96usize);
            let av: Vec<f32> = (0..m * k).map(|i| (i % 17) as f32 * 0.0625).collect();
            let bv: Vec<f32> = (0..k * n).map(|i| (i % 13) as f32 * 0.0625).collect();
            let bcols: Vec<Vec<f32>> = (0..n / seg)
                .map(|s| {
                    let j0 = s * seg;
                    let mut chunk = Vec::with_capacity(k * seg);
                    for p in 0..k {
                        chunk.extend_from_slice(&bv[p * n + j0..p * n + j0 + seg]);
                    }
                    chunk
                })
                .collect();
            let wide = timeit(|| {
                let v = lasx_rs::api::matmul(m, k, n, &av, &bv).unwrap();
                let _ = black_box(v[0]);
            });
            let narrow = timeit(|| {
                for bc in &bcols {
                    let v = lasx_rs::api::matmul(m, k, seg, &av, bc).unwrap();
                    let _ = black_box(v[0]);
                }
            });
            println!(
                "| 宽 1 次 vs **窄 {} 次**（串行、同量工作） | {m}×{k}×{n} | 宽 {} | 窄 {} | **{:.1}×** |",
                n / seg,
                crate::timing::fmt_t(wide),
                crate::timing::fmt_t(narrow),
                narrow.as_secs_f64() / wide.as_secs_f64()
            );
        }
        // **池在"多而小"工作项上的行为**（§7.11 ⑥ 的 12× 归因落到这里）：同一份活儿分别切 32 项/8 项，
        // 在 1 线程与 8 线程下计时。`black_box(item)` 保证闭包不可被优化掉（否则空活儿会被消除，
        // 我之前就踩过"空闭包 2 ns"这个假数）。
        {
            let mut items32: Vec<(usize, usize)> = (0..32).map(|s| (s * 96, s * 96 + 96)).collect();
            let mut items8: Vec<(usize, usize)> =
                (0..8).map(|s| (s * 384, s * 384 + 384)).collect();
            for &nt in &[1usize, 8] {
                let pool = WorkerPool::new(nt);
                let t32 = timeit(|| {
                    pool.for_each_chunk_mut(&mut items32, |my| {
                        for &(j0, j1) in my.iter() {
                            let _ = black_box(j1 - j0);
                        }
                    });
                });
                let t8 = timeit(|| {
                    pool.for_each_chunk_mut(&mut items8, |my| {
                        for &(j0, j1) in my.iter() {
                            let _ = black_box(j1 - j0);
                        }
                    });
                });
                println!(
                    "| 池：空活儿切 **32 项 vs 8 项** | {nt} 线程 | 32 项 {} | 8 项 {} | — |",
                    crate::timing::fmt_t(t32),
                    crate::timing::fmt_t(t8)
                );
            }
        }
        let (m, k) = (4096usize, 4096usize);
        let aw: Vec<u16> = (0..m * k).map(|i| (i % 1024) as u16).collect();
        let xv: Vec<f32> = vec![0.25; k];
        let mut yv = vec![0f32; m];
        let a = timeit(|| {
            let v = lasx_rs::api::gemv_f16(m, k, &aw, &xv).unwrap();
            let _ = black_box(v[0]);
        });
        let f = timeit(|| {
            lasx_gemv_f16(
                aw.as_ptr(),
                xv.as_ptr(),
                yv.as_mut_ptr(),
                m as i32,
                k as i32,
            );
            let _ = black_box(yv[0]);
        });
        println!(
            "| `gemv_f16` | m={m} k={k} | {} | {} | **{:.1}%** |",
            crate::timing::fmt_t(a),
            crate::timing::fmt_t(f),
            100.0 * a.saturating_sub(f).as_secs_f64() / a.as_secs_f64()
        );
    }
    // 物理/几何批量族（f64）：api 分配版 vs FFI 预分配版。**注意它们一次返回多个缓冲**
    // （`norm3_batch` 1 个、`quat_to_dcm_batch` **9 个**）⇒ 差额里含"每个缓冲一次分配"。
    {
        let n = 65536usize;
        let a1: Vec<f64> = (0..n).map(|i| i as f64 * 1e-3 + 1.0).collect();
        let a2: Vec<f64> = (0..n).map(|i| i as f64 * 2e-3).collect();
        let a3: Vec<f64> = (0..n).map(|i| i as f64 * 3e-3).collect();
        let mut o = vec![0f64; n];
        // 对称设计（同 `quat_to_dcm_batch` 那条）：两臂都走同一个 FFI 内核，只差"预分配 vs 每次分配"。
        let y = timeit(|| {
            lasx_norm3_batch(
                a1.as_ptr(),
                a2.as_ptr(),
                a3.as_ptr(),
                o.as_mut_ptr(),
                n as i32,
            );
            let _ = black_box(o[0]);
        });
        let x = timeit(|| {
            let mut o2 = vec![0f64; n];
            lasx_norm3_batch(
                a1.as_ptr(),
                a2.as_ptr(),
                a3.as_ptr(),
                o2.as_mut_ptr(),
                n as i32,
            );
            let _ = black_box(o2[0]);
        });
        println!(
            "| `norm3_batch`（1 缓冲，对称 A/B） | {n} | 每次分配 {} | 预分配 {} | **{:.1}%** |",
            crate::timing::fmt_t(x),
            crate::timing::fmt_t(y),
            100.0 * x.saturating_sub(y).as_secs_f64() / x.as_secs_f64()
        );
        let q: Vec<f64> = (0..n).map(|i| 0.5 + 1e-6 * i as f64).collect();
        let mut outs: Vec<Vec<f64>> = (0..9).map(|_| vec![0f64; n]).collect();
        // **对称设计**（上一版"api vs FFI"给出反直觉的 2.3×，已查明两者调的是**同一个 op**）：
        // 两个闭包都走**同一个 FFI 内核**，唯一差别是"9 个输出缓冲预分配 vs 每次分配"
        // ⇒ 差额才是**纯粹的分配成本**。
        let y = timeit(|| {
            lasx_quat_to_dcm_batch(
                q.as_ptr(),
                q.as_ptr(),
                q.as_ptr(),
                q.as_ptr(),
                outs[0].as_mut_ptr(),
                outs[1].as_mut_ptr(),
                outs[2].as_mut_ptr(),
                outs[3].as_mut_ptr(),
                outs[4].as_mut_ptr(),
                outs[5].as_mut_ptr(),
                outs[6].as_mut_ptr(),
                outs[7].as_mut_ptr(),
                outs[8].as_mut_ptr(),
                n as i32,
            );
            let _ = black_box(outs[0][0]);
        });
        let x = timeit(|| {
            let mut o: Vec<Vec<f64>> = (0..9).map(|_| vec![0f64; n]).collect();
            lasx_quat_to_dcm_batch(
                q.as_ptr(),
                q.as_ptr(),
                q.as_ptr(),
                q.as_ptr(),
                o[0].as_mut_ptr(),
                o[1].as_mut_ptr(),
                o[2].as_mut_ptr(),
                o[3].as_mut_ptr(),
                o[4].as_mut_ptr(),
                o[5].as_mut_ptr(),
                o[6].as_mut_ptr(),
                o[7].as_mut_ptr(),
                o[8].as_mut_ptr(),
                n as i32,
            );
            let _ = black_box(o[0][0]);
        });
        println!(
            "| `quat_to_dcm_batch`（9 缓冲，对称 A/B） | {n} | 每次分配 {} | 预分配 {} | **{:.1}%** |",
            crate::timing::fmt_t(x),
            crate::timing::fmt_t(y),
            100.0 * x.saturating_sub(y).as_secs_f64() / x.as_secs_f64()
        );
    }
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
    // 线程数可用 `LASX_BENCH_THREADS=8`（或 `6,8`）覆盖：perf profile 要把样本集中在
    // **一个**线程数上，扫 1..24 会把符号占比稀释掉（`docs/dev.md` §7.9 的归因步骤）。
    let threads: Vec<usize> = match std::env::var("LASX_BENCH_THREADS") {
        Ok(s) => s
            .split(',')
            .filter_map(|x| x.trim().parse::<usize>().ok())
            .collect(),
        Err(_) => vec![1, 2, 3, 4, 6, 8, 10, 12, 16, 24],
    };
    for &(k, n) in &[(768usize, 3072usize), (3072, 768)] {
        let mut rng = crate::data::Lcg::new((k * 17 + n) as u64);
        let a: Vec<f32> = (0..tokens * k).map(|_| rng.f32() * 2.0 - 1.0).collect();
        let b: Vec<f32> = (0..k * n).map(|_| rng.f32() * 2.0 - 1.0).collect();
        let mut c = vec![0f32; tokens * n];
        let mut base = Duration::ZERO;
        for &th in &threads {
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
    // 立方形状：文档里"回退赢/持平/共享赢"的**原始数据点**就在这些形状上（`docs/dev.md` §14.3
    // 与 §7.9 的"待改"注记）⇒ 改 `need_shared_pack` 前必须在这里扫过，否则就是"只按一个形状改"。
    for &s in &[256usize, 512, 1024] {
        let (m, k, n) = (s, s, s);
        let mut rng = crate::data::Lcg::new((s * 31) as u64);
        let a: Vec<f32> = (0..m * k).map(|_| rng.f32() * 2.0 - 1.0).collect();
        let b: Vec<f32> = (0..k * n).map(|_| rng.f32() * 2.0 - 1.0).collect();
        let mut c = vec![0f32; m * n];
        let mut base = Duration::ZERO;
        for &th in &threads {
            let pool = WorkerPool::new(th);
            let mut a = a.clone();
            let d = timeit(|| {
                parallel::matmul_f32(&pool, m, k, n, &mut a, &b, &mut c).unwrap();
                let _ = black_box(c[0]);
            });
            if th == 1 {
                base = d;
            }
            let sp = base.as_secs_f64() / d.as_secs_f64();
            println!(
                "| {m}×{k}×{n} | {th} | {} | {sp:.2}× | {:.0}% |",
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
    println!(
        "\n> ⚠️ **本表的\"相对默认\"列不要引用**（2026-10-04 判定，`docs/dev.md` §7.11 ⑤）：\
         \n> \"默认 `pick_rows`\"与显式 `RowBlock` 本是**同一段代码、同一策略值**（`pick_rows` 在 \
         threads<20 时返回的就是 `RowBlock`）；把两臂**对调顺序**后差距消失（2.71 vs 2.54 ms），\
         且两次跑**都是第一臂更慢** ⇒ 那是**顺序效应**，不是策略差异。\
         \n> 另外：同代码臂的摆动可达 **65%**（2.47 vs 4.10 ms）⇒ 要重测请按 §7.11 ⑤ 三条：\
         **≥5 轮取中位数 + 报离散度、只在 load ≤3 取数、逐臂交错**。"
    );
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
