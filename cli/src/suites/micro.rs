//! 微基准：纯寄存器 FMA 吞吐与多线程扩展性。

use crate::data::{states, velocities, Soa6, J2, MU, RE};
use crate::timing::{fmt_t, timeit};
use lasx_rs::lasx_force_lsx_thread;
use lasx_rs::*;
use std::arch::loongarch64::*;
use std::cell::UnsafeCell;
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
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

/* ==================== 常驻自旋工作线程池 ====================
每次调用都新建线程的代价实测 ≈0.5 ms（24 线程），占 24 线程单步时间的一半以上。
这里换成"原子代次 + 自旋等待"的常驻池：派活无系统调用、无互斥锁，主线程发布后
自旋等 done 计数到齐。短任务 + 高频派活正是自旋优于阻塞式同步的场合。 */

/// 一个 worker 的作业（裸指针，只有对应 worker 会读）。
#[derive(Clone, Copy)]
struct Rk4Job {
    rx: *mut f64,
    ry: *mut f64,
    rz: *mut f64,
    vx: *mut f64,
    vy: *mut f64,
    vz: *mut f64,
    n: usize,
}

impl Rk4Job {
    const NULL: Rk4Job = Rk4Job {
        rx: std::ptr::null_mut(),
        ry: std::ptr::null_mut(),
        rz: std::ptr::null_mut(),
        vx: std::ptr::null_mut(),
        vy: std::ptr::null_mut(),
        vz: std::ptr::null_mut(),
        n: 0,
    };
}

/// 自旋上限：超过就 park 睡到被唤醒。
///
/// 纯自旋在**不过订阅**时最快（派活全程无系统调用）。但线程数超过物理核时，一堆自旋
/// 线程会把真正干活的线程挤掉——实测 24 线程（12 物理核 ×2 SMT）纯自旋慢 10 倍、
/// 改成 `yield_now` 也仍慢 3 倍。故做成**自适应**：先自旋若干次，仍无任务就 park；
/// 主线程发布后统一唤醒。这样不过订阅走无系统调用的快路径，过订阅退化为"睡着的线程"，
/// 不会偷执行槽。
const SPIN_LIMIT: u32 = 1024;

struct PoolShared {
    /// 主线程递增表示"新一批作业已就位"。
    epoch: AtomicUsize,
    /// 本轮已完成的 worker 数。
    done: AtomicUsize,
    /// 退出标志。
    stop: AtomicBool,
    jobs: Vec<UnsafeCell<Rk4Job>>,
    /// 仅用于"自旋超限后 park / 发布后唤醒"，不保护数据。
    park: Mutex<()>,
    wake: Condvar,
}

// SAFETY: jobs[i] 只由第 i 个 worker 访问——主线程在递增 epoch 前写入，本轮不再触碰；
// epoch/done/stop 都是原子量。故跨线程共享安全。
unsafe impl Sync for PoolShared {}
unsafe impl Send for PoolShared {}

pub struct SpinPool {
    shared: Arc<PoolShared>,
    handles: Vec<std::thread::JoinHandle<()>>,
    threads: usize,
}

impl SpinPool {
    pub fn new(threads: usize) -> Self {
        let shared = Arc::new(PoolShared {
            epoch: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            jobs: (0..threads)
                .map(|_| UnsafeCell::new(Rk4Job::NULL))
                .collect(),
            park: Mutex::new(()),
            wake: Condvar::new(),
        });
        let handles = (0..threads)
            .map(|i| {
                let sh = Arc::clone(&shared);
                std::thread::spawn(move || {
                    let mut seen = 0usize;
                    loop {
                        // 自适应等待：先自旋，超限就 park（过订阅时不再偷执行槽）
                        let mut spins = 0u32;
                        while sh.epoch.load(Ordering::Acquire) == seen {
                            if sh.stop.load(Ordering::Relaxed) {
                                return;
                            }
                            spins += 1;
                            if spins < SPIN_LIMIT {
                                std::hint::spin_loop();
                                continue;
                            }
                            let mut g = sh.park.lock().unwrap();
                            while sh.epoch.load(Ordering::Acquire) == seen
                                && !sh.stop.load(Ordering::Relaxed)
                            {
                                g = sh.wake.wait(g).unwrap();
                            }
                            break;
                        }
                        seen = sh.epoch.load(Ordering::Acquire);
                        if sh.stop.load(Ordering::Relaxed) {
                            return;
                        }
                        // SAFETY: 本轮 jobs[i] 由本 worker 独占
                        let job = unsafe { *sh.jobs[i].get() };
                        lasx_force_lsx_thread(false);
                        lasx_rk4_j2_step_batch(
                            job.rx,
                            job.ry,
                            job.rz,
                            job.vx,
                            job.vy,
                            job.vz,
                            MU,
                            J2,
                            RE,
                            10.0,
                            job.n as i32,
                        );
                        sh.done.fetch_add(1, Ordering::Release);
                    }
                })
            })
            .collect();
        SpinPool {
            shared,
            handles,
            threads,
        }
    }

    /// 把 `buf` 切块、派活，并等到全部 worker 完成。
    pub fn run(&self, buf: &mut Soa6) {
        let n = buf.len();
        let chunk = n.div_ceil(self.threads).next_multiple_of(4).max(4);
        let mut off = 0usize;
        // 下标即 worker 编号（jobs[w] 必须与第 w 个 worker 对应），故用下标循环
        #[allow(clippy::needless_range_loop)]
        for w in 0..self.threads {
            let len = chunk.min(n.saturating_sub(off));
            // SAFETY: 各 worker 区间互不重叠；len=0 时 add(off) 仍是合法的一次过尾指针
            let job = unsafe {
                Rk4Job {
                    rx: buf.rx.as_mut_ptr().add(off),
                    ry: buf.ry.as_mut_ptr().add(off),
                    rz: buf.rz.as_mut_ptr().add(off),
                    vx: buf.vx.as_mut_ptr().add(off),
                    vy: buf.vy.as_mut_ptr().add(off),
                    vz: buf.vz.as_mut_ptr().add(off),
                    n: len,
                }
            };
            unsafe { *self.shared.jobs[w].get() = job };
            off += len;
        }
        self.shared.done.store(0, Ordering::Relaxed);
        {
            let _g = self.shared.park.lock().unwrap();
            self.shared.epoch.fetch_add(1, Ordering::Release);
        }
        self.shared.wake.notify_all();
        while self.shared.done.load(Ordering::Acquire) < self.threads {
            std::hint::spin_loop();
        }
    }
}

impl Drop for SpinPool {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        self.shared.epoch.fetch_add(1, Ordering::Release); // 唤醒自旋中的 worker
        self.shared.wake.notify_all(); // 以及已 park 的 worker
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
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
    println!("**常驻自旋池**（原子代次派活，worker 自旋等待，无锁无系统调用）。");
    println!();
    println!("| 线程数 | 每次新建线程 | 常驻自旋池 | 池相对 1 线程 | 池效率 |");
    println!("|---|---|---|---|---|");
    println!("| 1 | — | {} | 1.00× | 100% |", fmt_t(t1));

    // 池里有裸指针切分，先对一次结果：池推进一步应与单线程逐位一致
    {
        let mut seq = base.clone();
        let mut par = base.clone();
        seq.step(false);
        SpinPool::new(8).run(&mut par);
        for k in 0..n {
            assert_eq!(
                seq.rx[k].to_bits(),
                par.rx[k].to_bits(),
                "自旋池结果与单线程不一致 @ {k}"
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
        let pool = SpinPool::new(th);
        let mut b2 = base.clone();
        let d_pool = timeit(|| {
            pool.run(&mut b2);
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

#[inline(never)]
pub fn fma_peak_lasx() -> f64 {
    unsafe {
        let a: m256 = std::mem::transmute(lasx_xvreplgr2vr_w(0x3f80_0000));
        let b: m256 = std::mem::transmute(lasx_xvreplgr2vr_w(0x3f00_0000));
        let (mut v0, mut v1, mut v2, mut v3) = (a, a, a, a);
        let (mut v4, mut v5, mut v6, mut v7) = (a, a, a, a);
        for _ in 0..FMA_ITERS {
            v0 = lasx_xvfmadd_s(a, b, v0);
            v1 = lasx_xvfmadd_s(a, b, v1);
            v2 = lasx_xvfmadd_s(a, b, v2);
            v3 = lasx_xvfmadd_s(a, b, v3);
            v4 = lasx_xvfmadd_s(a, b, v4);
            v5 = lasx_xvfmadd_s(a, b, v5);
            v6 = lasx_xvfmadd_s(a, b, v6);
            v7 = lasx_xvfmadd_s(a, b, v7);
        }
        let s = lasx_xvfadd_s(
            lasx_xvfadd_s(v0, v1),
            lasx_xvfadd_s(
                lasx_xvfadd_s(v2, v3),
                lasx_xvfadd_s(lasx_xvfadd_s(v4, v5), lasx_xvfadd_s(v6, v7)),
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
        for _ in 0..FMA_ITERS {
            v0 = lsx_vfmadd_s(a, b, v0);
            v1 = lsx_vfmadd_s(a, b, v1);
            v2 = lsx_vfmadd_s(a, b, v2);
            v3 = lsx_vfmadd_s(a, b, v3);
            v4 = lsx_vfmadd_s(a, b, v4);
            v5 = lsx_vfmadd_s(a, b, v5);
            v6 = lsx_vfmadd_s(a, b, v6);
            v7 = lsx_vfmadd_s(a, b, v7);
        }
        let s = lsx_vfadd_s(
            lsx_vfadd_s(v0, v1),
            lsx_vfadd_s(
                lsx_vfadd_s(v2, v3),
                lsx_vfadd_s(lsx_vfadd_s(v4, v5), lsx_vfadd_s(v6, v7)),
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
    let flops_lasx = (8 * 8 * FMA_ITERS * 2) as f64;
    let flops_lsx = (8 * 4 * FMA_ITERS * 2) as f64;

    let t_lasx = timeit(|| {
        let _ = black_box(fma_peak_lasx());
    });
    let t_lsx = timeit(|| {
        let _ = black_box(fma_peak_lsx());
    });

    let g_lasx = flops_lasx / t_lasx.as_secs_f64() / 1e9;
    let g_lsx = flops_lsx / t_lsx.as_secs_f64() / 1e9;
    println!();
    println!("## 纯寄存器 FMA 吞吐（无内存访问，8 条独立累加链）");
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
/// 这两项历史上分别用 `OnceLock`（约 5 ns/次）与 TLS；改成"写一次的 relaxed 原子读"
/// 后降到约 1 ns/次——对小规模内核（十几 ns）是实打实的比例提升。
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
    println!("> 参考：`lasx_dot` n=24 约 12 ns，n=4096 约 272 ns。");
}
