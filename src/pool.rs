//! 可选的常驻工作线程池：把批量内核铺到多核，并**跨调用复用线程**。
//!
//! # 为什么放进库里
//!
//! 多步传播这类真实负载里，"每次调用都新建线程"的代价摊不掉：实测每步约 0.5 ms
//! 的创建/回收开销，而多星多步传播端到端因此慢 2.1–2.3×（见 `docs/perf-report.md`
//! §13.3）。常驻池把它降到接近零，并让**建池只发生一次**。
//!
//! # 为什么是 Rust API 而不是 C ABI
//!
//! 池需要调用方的数据切片与一个闭包，走 `extern "C"` 就得引入不透明句柄、C 函数指针
//! 与手工生命周期管理。本库同时提供 `rlib`，Rust 调用方（本仓库的基准、YouLiLong
//! 原生扩展、loong-sci 等）直接 `use` 即可，**不新增任何导出符号**。
//!
//! ```
//! use lasx_rs::aligned::AlignedVec;
//! use lasx_rs::pool::WorkerPool;
//!
//! let pool = WorkerPool::new(4);
//! let mut data = AlignedVec::<f64>::fill_with(10_000, |i| i as f64);
//! pool.for_each_chunk_mut(data.as_mut_slice(), |chunk| {
//!     for v in chunk.iter_mut() {
//!         *v += 1.0;
//!     }
//! });
//! assert_eq!(data[0], 1.0);
//! ```
//!
//! # 等待策略（实测过的边界）
//!
//! 先自旋 `SPIN_LIMIT` 次，仍无任务就 park（`Condvar`）。**纯自旋在线程数超过物理核时
//! 会灾难性地互相抢执行槽**——24 逻辑核 /12 物理核上实测慢 10 倍（perf-report §12.2）。
//!
//! # 什么时候不要用
//!
//! 数据量小于 [`MIN_PARALLEL_LEN`] 时自动原地串行执行：派活/唤醒的开销不值得。

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// 自旋上限：超过就 park。
const SPIN_LIMIT: u32 = 1024;

/// 低于该长度不做并行（直接原地串行），避免派活开销超过收益。
pub const MIN_PARALLEL_LEN: usize = 4096;

/// 单个 worker 本轮的工作槽。
struct Slot {
    /// 每个数组在本 worker 上的起始地址（SOA 多数组时长度 > 1）。
    ptrs: Vec<*mut u8>,
    /// 本 worker 的样本数（各数组相同）；0 表示本轮无活。
    len: usize,
}

/// 类型擦除的入口：把裸指针还原成 `[&mut [T]; N]` 并调用闭包。
///
/// `ptrs` 是各数组的起始地址，`len` 是样本数，`func` 指向调用方栈上的闭包。
type Thunk = unsafe fn(ptrs: &[*mut u8], len: usize, func: *const ());

struct Shared {
    /// 递增表示"新一批作业已就位"。
    epoch: AtomicUsize,
    /// 本轮已完成的 worker 数。
    done: AtomicUsize,
    /// 退出标志。
    stop: AtomicBool,
    slots: Vec<UnsafeCell<Slot>>,
    /// 本轮的类型擦除入口与闭包指针。
    call: UnsafeCell<(Thunk, *const ())>,
    /// 仅用于"自旋超限后 park / 发布后唤醒"，不保护数据。
    park: Mutex<()>,
    wake: Condvar,
}

// SAFETY: 数据竞争由"代次发布 + done 计数"排除——
// - `slots[i]` 只由第 i 个 worker 访问；
// - `call` 与所有 `slots` 都在 `epoch` 递增（Release）**之前**由调用方写入，
//   而调用方在 `done == threads`（Acquire）之后才返回，因此 worker 读取期间
//   调用方不会改动它们；
// - epoch/done/stop 都是原子量。
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

/// 常驻工作线程池。
///
/// 线程在 [`WorkerPool::new`] 时创建一次，之后每次并行调用只做一次原子发布 + 自旋等待。
pub struct WorkerPool {
    shared: Arc<Shared>,
    handles: Vec<std::thread::JoinHandle<()>>,
    threads: usize,
}

impl WorkerPool {
    /// 建池，`threads` 至少为 1。
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let shared = Arc::new(Shared {
            epoch: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            slots: (0..threads)
                .map(|_| {
                    UnsafeCell::new(Slot {
                        ptrs: Vec::new(),
                        len: 0,
                    })
                })
                .collect(),
            call: UnsafeCell::new((noop_thunk::<u8, 1>, std::ptr::null())),
            park: Mutex::new(()),
            wake: Condvar::new(),
        });
        let handles = (0..threads)
            .map(|i| {
                let sh = Arc::clone(&shared);
                std::thread::spawn(move || worker_loop(sh, i))
            })
            .collect();
        WorkerPool {
            shared,
            handles,
            threads,
        }
    }

    /// 按 `available_parallelism` 建池。
    pub fn auto() -> Self {
        Self::new(
            std::thread::available_parallelism()
                .map(|v| v.get())
                .unwrap_or(1),
        )
    }

    /// 池内线程数。
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// 把 `data` 切成 `threads` 段并行处理，返回前保证全部完成。
    pub fn for_each_chunk_mut<T: Send, F: Fn(&mut [T]) + Sync>(&self, data: &mut [T], f: F) {
        self.for_each_chunks_mut([data], |[c]| f(c));
    }

    /// 把 `N` 个**等长**数组同步切段并行处理（SOA 批量内核的常见形状），
    /// 每个 worker 拿到的是同一下标区间上、各数组互不重叠的可变切片。
    ///
    /// # Panics
    /// 各数组长度不一致时 panic。
    pub fn for_each_chunks_mut<T: Send, const N: usize, F>(&self, arrays: [&mut [T]; N], f: F)
    where
        F: Fn([&mut [T]; N]) + Sync,
    {
        let len = arrays[0].len();
        for a in &arrays[1..] {
            assert_eq!(a.len(), len, "并行处理要求各数组等长");
        }
        // 单线程或数据太小：原地串行，省掉派活开销
        if self.threads == 1 || len < MIN_PARALLEL_LEN {
            f(arrays);
            return;
        }

        let chunk = len.div_ceil(self.threads).max(1);
        // 每个数组各自切块；`chunks[k][w]` 是第 k 个数组的第 w 块
        let mut chunks: [Vec<&mut [T]>; N] = arrays.map(|a| a.chunks_mut(chunk).collect());
        let nchunks = chunks[0].len();
        debug_assert!(nchunks <= self.threads);

        for w in 0..self.threads {
            // SAFETY: slots[w] 只由第 w 个 worker 访问，且它在 done 计数之前不会推进到下一轮
            let slot = unsafe { &mut *self.shared.slots[w].get() };
            if w < nchunks {
                slot.ptrs.clear();
                slot.ptrs
                    .extend((0..N).map(|k| chunks[k][w].as_mut_ptr() as *mut u8));
                slot.len = chunks[0][w].len();
            } else {
                slot.ptrs.clear();
                slot.len = 0; // 无活：thunk 见 len==0 会直接返回，不会解引用空 ptrs
            }
        }

        // SAFETY: `chunks` 里的可变借用覆盖本次调用的全部数据，且在本函数返回前一直存活；
        // 闭包 `f` 同样在栈上存活到返回。publish 返回时所有 worker 已执行完毕。
        unsafe { self.publish(thunk::<T, N, F>, &f as *const F as *const ()) };
    }

    /// 发布一轮并等到全部 worker 完成。
    ///
    /// # Safety
    /// 调用方必须保证：所有 `slots` 已填好、`func` 与槽位里的指针在本次调用期间有效，
    /// 且 `func` 指向的闭包类型与 `t` 期望的一致。
    unsafe fn publish(&self, t: Thunk, func: *const ()) {
        *self.shared.call.get() = (t, func);
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

impl Drop for WorkerPool {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        self.shared.epoch.fetch_add(1, Ordering::Release);
        self.shared.wake.notify_all();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

impl std::fmt::Debug for WorkerPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerPool")
            .field("threads", &self.threads)
            .finish_non_exhaustive()
    }
}

/// worker 主循环：自适应等待新代次 → 取出本轮入口 → 执行 → 计数。
fn worker_loop(sh: Arc<Shared>, i: usize) {
    let mut seen = 0usize;
    loop {
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
            // 自旋够久：park 睡到被唤醒（过订阅时不再偷执行槽）
            let mut g = sh.park.lock().unwrap();
            while sh.epoch.load(Ordering::Acquire) == seen && !sh.stop.load(Ordering::Relaxed) {
                g = sh.wake.wait(g).unwrap();
            }
            break;
        }
        seen = sh.epoch.load(Ordering::Acquire);
        if sh.stop.load(Ordering::Relaxed) {
            return;
        }
        // SAFETY: 见 Shared 的 Sync 说明——本 worker 只读 slots[i]，调用方在
        // done 计数到齐前不会改它们。
        let (thunk, func) = unsafe { *sh.call.get() };
        let slot = unsafe { &*sh.slots[i].get() };
        unsafe { thunk(&slot.ptrs, slot.len, func) };
        sh.done.fetch_add(1, Ordering::Release);
    }
}

/// 类型擦除入口的具体化。
///
/// # Safety
/// `ptrs` 必须是 `N` 个有效地址（`len > 0` 时），`func` 必须指向一个
/// `F: Fn([&mut [T]; N])`，且两者在调用期间有效、各块互不重叠。
unsafe fn thunk<T, const N: usize, F>(ptrs: &[*mut u8], len: usize, func: *const ())
where
    F: Fn([&mut [T]; N]),
{
    if len == 0 {
        return; // 无活的 worker：ptrs 可能为空，不能解引用
    }
    let arrays: [&mut [T]; N] =
        std::array::from_fn(|k| std::slice::from_raw_parts_mut(ptrs[k] as *mut T, len));
    let f = &*(func as *const F);
    f(arrays);
}

/// 初始占位入口（`call` 需要初值；任何真实调用都会覆盖它）。
unsafe fn noop_thunk<T, const N: usize>(_p: &[*mut u8], _len: usize, _f: *const ()) {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize as AtomicCount;

    /// 每个元素必须被**恰好处理一次**（用原子计数与逐元素自增双重验证）。
    #[test]
    fn test_every_element_processed_exactly_once() {
        for &threads in &[1usize, 2, 3, 8] {
            for &len in &[
                0usize,
                1,
                7,
                MIN_PARALLEL_LEN - 1,
                MIN_PARALLEL_LEN,
                10_000,
                65_537,
            ] {
                let pool = WorkerPool::new(threads);
                let mut data: Vec<u64> = vec![0; len];
                let calls = AtomicCount::new(0);
                pool.for_each_chunk_mut(data.as_mut_slice(), |chunk| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    for v in chunk.iter_mut() {
                        *v += 1;
                    }
                });
                assert!(
                    data.iter().all(|&v| v == 1),
                    "threads={threads} len={len}: 有人被处理了 0 次或多次"
                );
                // 长度够时应当真的切成多块（并行生效的证据）
                if len >= MIN_PARALLEL_LEN && threads > 1 {
                    assert!(calls.load(Ordering::Relaxed) > 1, "len={len} 未被切块");
                }
            }
        }
    }

    /// 多数组版本：各数组的同一区间必须对齐、互不重叠。
    #[test]
    fn test_multi_array_chunks_are_aligned_and_disjoint() {
        let pool = WorkerPool::new(4);
        let n = 50_000usize;
        let (mut a, mut b, mut c) = (vec![0u32; n], vec![0u32; n], vec![0u32; n]);
        pool.for_each_chunks_mut(
            [a.as_mut_slice(), b.as_mut_slice(), c.as_mut_slice()],
            |[x, y, z]| {
                assert_eq!(x.len(), y.len());
                assert_eq!(y.len(), z.len());
                for i in 0..x.len() {
                    x[i] += 1;
                    y[i] += 1;
                    z[i] += 1;
                }
            },
        );
        for i in 0..n {
            assert_eq!((a[i], b[i], c[i]), (1, 1, 1), "下标 {i} 未被处理或重复处理");
        }
    }

    /// 求和结果必须与串行一致（验证数据确实被完整覆盖）。
    #[test]
    fn test_sum_matches_serial() {
        let pool = WorkerPool::new(6);
        let n = 100_003usize;
        let mut data: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
        let want: f64 = data.iter().sum();
        let total = Mutex::new(0f64);
        pool.for_each_chunk_mut(data.as_mut_slice(), |chunk| {
            let s: f64 = chunk.iter().sum();
            *total.lock().unwrap() += s;
        });
        let got = *total.lock().unwrap();
        assert!((got - want).abs() < 1e-6, "并行求和 {got} != 串行 {want}");
    }

    #[test]
    fn test_unequal_lengths_panic() {
        let pool = WorkerPool::new(2);
        let mut a = vec![0u8; 10];
        let mut b = vec![0u8; 11];
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.for_each_chunks_mut([a.as_mut_slice(), b.as_mut_slice()], |_| {});
        }));
        assert!(r.is_err(), "长度不一致应当 panic");
    }

    #[test]
    fn test_pool_is_reusable_across_many_calls() {
        let pool = WorkerPool::new(4);
        let mut data = vec![0u64; 20_000];
        for round in 0..50 {
            pool.for_each_chunk_mut(data.as_mut_slice(), |chunk| {
                for v in chunk.iter_mut() {
                    *v += 1;
                }
            });
            assert!(
                data.iter().all(|&v| v == round + 1),
                "第 {round} 轮结果不对——池跨调用复用有问题"
            );
        }
    }
}
