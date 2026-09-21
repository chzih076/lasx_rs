//! 可选的常驻工作线程池：把批量内核铺到多核，并**跨调用复用线程**。
//!
//! # 为什么放进库里
//!
//! 多步传播这类真实负载里，"每次调用都新建线程"的代价摊不掉：实测每步约 0.5 ms
//! 的创建/回收开销，而多星多步传播端到端因此慢 2.1–2.3×（见 `docs/perf-report.md`
//! §13.3）。常驻池把它降到接近零，并让**建池只发生一次**。
//!
//! # 三种切分形状
//!
//! | 接口 | 形状 | 典型用途 |
//! |---|---|---|
//! | [`WorkerPool::for_each_chunk_mut`] | 一个数组，按元素等分 | `sum`/`dot`/`axpy` |
//! | [`WorkerPool::for_each_chunks_mut`] | N 个**等长**数组，同步等分 | SOA 批量物理内核（6 个分量数组） |
//! | [`WorkerPool::for_each_row_block_mut`] | N 个数组，**行宽可不同**，按同一行区间切块 | 矩阵乘（A 每行 k 个、C 每行 n 个） |
//!
//! 行块接口要求调用方声明**行粒度**（内核一次处理几行）：块大小会向上取整到该粒度的
//! 倍数。这不是可有可无的参数——`lasx_matmul` 按 4 行分块、尾块退化成单行（B 复用从
//! 4 次掉到 1 次），256 行 12 线程默认切出 22 行/块 = 5×4+2 行，实测**慢一倍**。
//!
//! ```
//! use lasx_rs::aligned::AlignedVec;
//! use lasx_rs::pool::WorkerPool;
//!
//! let mut pool = WorkerPool::new(4);
//! let mut data = AlignedVec::<f64>::fill_with(10_000, |i| i as f64);
//! pool.for_each_chunk_mut(data.as_mut_slice(), |chunk| {
//!     for v in chunk.iter_mut() {
//!         *v += 1.0;
//!     }
//! });
//! assert_eq!(data[0], 1.0);
//! ```
//!
//! 矩阵乘这种"只读共享 B、按行切 A 与 C"的形状：
//!
//! ```no_run
//! use lasx_rs::pool::WorkerPool;
//! # fn main() {
//! # let (m, k, n) = (256usize, 256usize, 256usize);
//! # let (mut a, b, mut c) = (vec![0f32; m * k], vec![0f32; k * n], vec![0f32; m * n]);
//! let mut pool = WorkerPool::new(12);
//! let (bs, cs) = (b.as_slice(), c.as_mut_slice()); // 只读的 B 用 & 捕获即可
//! pool.for_each_row_block_mut(m, 4, [(&mut a, k), (cs, n)], |rows, [ab, cb]| {
//!     lasx_rs::lasx_matmul(
//!         rows as i32, k as i32, n as i32,
//!         ab.as_ptr(), bs.as_ptr(), cb.as_mut_ptr(),
//!     );
//! });
//! # }
//! ```
//!
//! # 为什么是 Rust API 而不是 C ABI
//!
//! 池需要调用方的数据切片与一个闭包，走 `extern "C"` 就得引入不透明句柄、C 函数指针
//! 与手工生命周期管理。本库同时提供 `rlib`，Rust 调用方（本仓库的基准、YouLiLong
//! 原生扩展、loong-sci 等）直接 `use` 即可，**不新增任何导出符号**。
//!
//! # 派活是独占的（`&mut self`）
//!
//! 池只有一套作业槽与代次计数器，**同时只允许一个调用方派活**——所以三个 `for_each_*`
//! 都取 `&mut self`：并发调用在编译期就被拒绝，而不是变成静默的数据竞争。要在多个
//! 线程间共享一个池，自己套 `Mutex`（YouLiLong 扩展就是这么做的）。
//!
//! # panic 不会把池挂死
//!
//! worker 里的闭包 panic 会被 `catch_unwind` 拦下并计数，主线程等到全部 worker 收工后
//! 把 panic **原样续抛**给调用方（`resume_unwind`）。不这么做的话，`done` 永远到不齐，
//! 主线程会永久自旋——那是比崩溃更糟的失败形态。
//!
//! # 每次调用零分配
//!
//! 作业槽是定长数组（[`MAX_ARRAYS`] 个指针 + 长度），派活只做指针算术与几次写入，
//! 不分配、不加锁、不产生系统调用。
//!
//! # 等待策略（实测过的边界）
//!
//! 先自旋 `SPIN_LIMIT` 次，仍无任务就 park（`Condvar`）。**纯自旋在线程数超过物理核时
//! 会灾难性地互相抢执行槽**——24 逻辑核 /12 物理核上实测慢 10 倍（perf-report §12.2）。
//!
//! # 什么时候不要用
//!
//! 一次调用涉及的元素总数小于 [`MIN_PARALLEL_LEN`] 时自动原地串行执行：派活/唤醒的
//! 开销不值得。

use std::any::Any;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// 自旋上限：超过就 park。
const SPIN_LIMIT: u32 = 1024;

/// 低于该长度不做并行（直接原地串行），避免派活开销超过收益。
///
/// 口径是**一次调用涉及的元素总数**：单数组接口即数组长度；SOA 接口是各数组长度之和
/// （6 个分量数组时相当于长度之和）。
pub const MIN_PARALLEL_LEN: usize = 4096;

/// 单次派活最多支持的数组个数（SOA 内核一般 2–9 个）。
pub const MAX_ARRAYS: usize = 16;

/// 单个 worker 本轮的工作槽。
///
/// 定长数组而非 `Vec`：派活路径上零分配。
struct Slot {
    /// 每个数组在本 worker 上的起始地址。
    ptrs: [*mut u8; MAX_ARRAYS],
    /// 每个数组在本 worker 上的元素个数（行块切分时各数组可以不同）。
    lens: [usize; MAX_ARRAYS],
    /// 数组个数；0 表示本轮无活（worker 不调用入口）。
    n: usize,
    /// 本块的行数（按元素切分时即元素个数）——闭包拿它省得自己反推。
    rows: usize,
}

/// 类型擦除的入口：把裸指针还原成 `[&mut [T]; N]` 并调用闭包。
///
/// `ptrs[k]`/`lens[k]` 是第 k 个数组的起始地址与元素个数，`rows` 是本块行数
/// （按元素切分时就是元素个数），`func` 指向调用方栈上的闭包。
type Thunk = unsafe fn(rows: usize, ptrs: &[*mut u8], lens: &[usize], func: *const ());

struct Shared {
    /// 递增表示"新一批作业已就位"。
    epoch: AtomicUsize,
    /// 本轮已完成的 worker 数。
    done: AtomicUsize,
    /// 退出标志。
    stop: AtomicBool,
    /// 是否有 worker 在闭包里 panic 了。
    panicked: AtomicBool,
    /// 第一个 panic 的载荷（只在 `panicked` 为真时上锁读取）。
    payload: Mutex<Option<Box<dyn Any + Send>>>,
    slots: Vec<UnsafeCell<Slot>>,
    /// 本轮的类型擦除入口与闭包指针。
    call: UnsafeCell<(Thunk, *const ())>,
    /// 仅用于"自旋超限后 park / 发布后唤醒"，不保护数据。
    park: Mutex<()>,
    wake: Condvar,
}

// SAFETY: 数据竞争由"独占派活（`&mut self`）+ 代次发布 + done 计数"排除——
// - 三个 `for_each_*` 都取 `&mut self`，故同一时刻只有一个调用方在写 `slots`/`call`；
// - `slots[i]` 只由第 i 个 worker 访问；
// - `call` 与所有 `slots` 都在 `epoch` 递增（Release）**之前**由调用方写入，
//   而调用方在 `done == threads`（Acquire）之后才返回，因此 worker 读取期间
//   调用方不会改动它们；
// - epoch/done/stop/panicked 都是原子量，payload 由 `Mutex` 保护。
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
            panicked: AtomicBool::new(false),
            payload: Mutex::new(None),
            slots: (0..threads)
                .map(|_| {
                    UnsafeCell::new(Slot {
                        ptrs: [std::ptr::null_mut(); MAX_ARRAYS],
                        lens: [0; MAX_ARRAYS],
                        n: 0,
                        rows: 0,
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
    ///
    /// 注意它给的是**逻辑**核数（本机 24 = 12 物理核 ×2 SMT）；超线程对这类带宽受限的
    /// 批量内核没有额外收益，也不会更差（perf-report §12.2）。
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
    pub fn for_each_chunk_mut<T: Send, F: Fn(&mut [T]) + Sync>(&mut self, data: &mut [T], f: F) {
        let len = data.len();
        if self.threads == 1 || len < MIN_PARALLEL_LEN {
            f(data);
            return;
        }
        let base = data.as_mut_ptr();
        let rows_per = len.div_ceil(self.threads).max(1);
        self.dispatch_rows::<T, 1, _>([base], [1], len, rows_per, |_rows, [c]| f(c));
    }

    /// 把 `N` 个**等长**数组同步切段并行处理（SOA 批量内核的常见形状），
    /// 每个 worker 拿到的是同一下标区间上、各数组互不重叠的可变切片。
    ///
    /// # Panics
    /// 各数组长度不一致时 panic。
    pub fn for_each_chunks_mut<T: Send, const N: usize, F>(&mut self, arrays: [&mut [T]; N], f: F)
    where
        F: Fn([&mut [T]; N]) + Sync,
    {
        let len = arrays[0].len();
        for a in &arrays[1..] {
            assert_eq!(a.len(), len, "并行处理要求各数组等长");
        }
        if self.threads == 1 || len.saturating_mul(N) < MIN_PARALLEL_LEN {
            f(arrays);
            return;
        }
        let bases = arrays.map(|a| a.as_mut_ptr());
        let rows_per = len.div_ceil(self.threads).max(1);
        self.dispatch_rows::<T, N, _>(bases, [1; N], len, rows_per, |_rows, a| f(a));
    }

    /// 按**行块**切分：`arrays[k]` 是 `(行主序数组, 每行元素数)`，
    /// 所有数组按**同一段行区间**切块——数组间的行宽可以不同。
    ///
    /// 这是矩阵乘的形状：`A` 每行 `k` 个元素、`C` 每行 `n` 个元素，但两者必须同步
    /// 切在同一批行上；`B` 只读共享，由闭包以 `&` 捕获即可。
    ///
    /// 闭包第一个参数是**本块的行数**（`rows × 行宽` 反推会在行宽为 0 时除零）。
    ///
    /// `row_gran` 是内核一次处理的行数（如 `lasx_matmul` 为 4；没有行结构就传 1）：
    /// 块大小会**向上**取整到它的倍数，代价是块数可能少于线程数，换来的是块内没有
    /// 退化的尾块。别省这个参数——实测矩阵乘 256 行 12 线程，`row_gran=1` 切出
    /// 22 行/块（5×4+2，两个单行尾块）比 `row_gran=4` 切出 24 行/块**慢一倍**
    /// （见 `docs/perf-report.md` §14.6）。
    ///
    /// # Panics
    /// `row_gran == 0`，或第 k 个数组的长度不等于 `rows × 行宽` 时 panic。
    pub fn for_each_row_block_mut<T: Send, const N: usize, F>(
        &mut self,
        rows: usize,
        row_gran: usize,
        arrays: [(&mut [T], usize); N],
        f: F,
    ) where
        F: Fn(usize, [&mut [T]; N]) + Sync,
    {
        assert!(row_gran > 0, "行粒度至少为 1");
        let mut total = 0usize;
        for (k, (s, w)) in arrays.iter().enumerate() {
            let want = rows.checked_mul(*w).expect("rows × 行宽 溢出");
            assert_eq!(
                s.len(),
                want,
                "第 {k} 个数组长度 {} 与 rows × 行宽 = {want} 不符",
                s.len()
            );
            total = total.saturating_add(want);
        }
        if self.threads == 1 || rows < 2 || total < MIN_PARALLEL_LEN {
            f(rows, arrays.map(|(s, _)| s));
            return;
        }
        let mut bases = [std::ptr::null_mut(); N];
        let mut widths = [0usize; N];
        for (k, (s, w)) in arrays.into_iter().enumerate() {
            bases[k] = s.as_mut_ptr();
            widths[k] = w;
        }
        // 块大小向上取整到行粒度的倍数：块数只会变少（仍 <= threads），但不会出现
        // "n 行 + 一个退化的尾块"这种把整个块拖慢的组合。
        let rows_per = rows
            .div_ceil(self.threads)
            .max(1)
            .next_multiple_of(row_gran);
        self.dispatch_rows::<T, N, _>(bases, widths, rows, rows_per, f);
    }

    /// 派活内核：第 k 个数组按行宽 `widths[k]` 解释，共 `rows` 行；每个 worker 取一段连续行。
    ///
    /// 各数组长度必须等于 `rows × widths[k]`（由上面的公开接口校验），
    /// 否则下面的指针算术就越界了。`rows_per` 是本轮的块大小（≥ 1）。
    fn dispatch_rows<T: Send, const N: usize, F>(
        &mut self,
        bases: [*mut T; N],
        widths: [usize; N],
        rows: usize,
        rows_per: usize,
        f: F,
    ) where
        F: Fn(usize, [&mut [T]; N]) + Sync,
    {
        const { assert!(N <= MAX_ARRAYS, "一次派活的数组个数超过 MAX_ARRAYS") };
        // 下标即 worker 编号（slots[w] 必须与第 w 个 worker 对应），故用下标循环
        #[allow(clippy::needless_range_loop)]
        for w in 0..self.threads {
            // 第 w 段：行区间 [start, start + r)
            let start = w * rows_per;
            let r = rows_per.min(rows.saturating_sub(start));
            // SAFETY: slots[w] 只由第 w 个 worker 访问，且它在 done 计数之前不会推进到下一轮
            let slot = unsafe { &mut *self.shared.slots[w].get() };
            if r == 0 {
                slot.n = 0; // 无活：worker 见到 n==0 就不调用入口
                continue;
            }
            slot.n = N;
            slot.rows = r;
            for k in 0..N {
                let off = start * widths[k];
                // SAFETY: 数组长度为 rows*widths[k]（调用方已校验），
                // 而 off <= rows*widths[k]，故地址在界内（至多尾后一格）。
                slot.ptrs[k] = unsafe { bases[k].add(off) } as *mut u8;
                slot.lens[k] = r * widths[k];
            }
        }

        // SAFETY: 上面的槽位覆盖本次调用的全部数据，且各数组长度由公开接口校验为
        // `rows × widths[k]`、行区间互不重叠；`f` 在栈上存活到本函数返回。
        // publish 返回时所有 worker 已执行完毕（或已被拦下的 panic 已取回）。
        unsafe { self.publish(thunk::<T, N, F>, &f as *const F as *const ()) };
    }

    /// 发布一轮并等到全部 worker 完成（有 panic 则续抛）。
    ///
    /// # Safety
    /// 调用方必须保证：所有 `slots` 已填好、`func` 与槽位里的指针在本次调用期间有效，
    /// 且 `func` 指向的闭包类型与 `t` 期望的一致。
    unsafe fn publish(&mut self, t: Thunk, func: *const ()) {
        *self.shared.call.get() = (t, func);
        self.shared.done.store(0, Ordering::Relaxed);
        self.shared.panicked.store(false, Ordering::Relaxed);
        {
            let _g = self.shared.park.lock().unwrap();
            self.shared.epoch.fetch_add(1, Ordering::Release);
        }
        self.shared.wake.notify_all();
        while self.shared.done.load(Ordering::Acquire) < self.threads {
            std::hint::spin_loop();
        }
        if self.shared.panicked.load(Ordering::Acquire) {
            // 池已收工，状态是干净的：把 panic 交给调用方，池本身可继续复用。
            self.shared.panicked.store(false, Ordering::Relaxed);
            let payload = self.shared.payload.lock().unwrap().take();
            if let Some(p) = payload {
                std::panic::resume_unwind(p);
            }
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
        if slot.n > 0 {
            let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                thunk(slot.rows, &slot.ptrs[..slot.n], &slot.lens[..slot.n], func);
            }))
            .err();
            if let Some(p) = payload {
                // 只留第一个载荷；无论怎样都要计数，否则主线程永远等不到 done
                let mut guard = sh.payload.lock().unwrap();
                if guard.is_none() {
                    *guard = Some(p);
                }
                drop(guard);
                sh.panicked.store(true, Ordering::Release);
            }
        }
        sh.done.fetch_add(1, Ordering::Release);
    }
}

/// 类型擦除入口的具体化。
///
/// # Safety
/// `ptrs`/`lens` 必须等长且为 `N` 个有效地址与元素个数（`N` 个数组互不重叠），
/// `func` 必须指向一个 `F: Fn([&mut [T]; N])`，且两者在调用期间有效。
unsafe fn thunk<T, const N: usize, F>(
    rows: usize,
    ptrs: &[*mut u8],
    lens: &[usize],
    func: *const (),
) where
    F: Fn(usize, [&mut [T]; N]),
{
    let arrays: [&mut [T]; N] =
        std::array::from_fn(|k| std::slice::from_raw_parts_mut(ptrs[k] as *mut T, lens[k]));
    let f = &*(func as *const F);
    f(rows, arrays);
}

/// 初始占位入口（`call` 需要初值；任何真实调用都会覆盖它）。
unsafe fn noop_thunk<T, const N: usize>(_rows: usize, _p: &[*mut u8], _l: &[usize], _f: *const ()) {
}

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
                let mut pool = WorkerPool::new(threads);
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
        let mut pool = WorkerPool::new(4);
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

    /// 行块版本：行宽不同的两个数组必须切在**同一批行**上，且恰好覆盖每一行。
    #[test]
    fn test_row_blocks_pair_arrays_by_row_range() {
        let mut pool = WorkerPool::new(5);
        let (m, k, n) = (1000usize, 7usize, 11usize);
        // 两个数组都用"行号"预填：行宽不同（k vs n），能对齐才说明切的是同一批行
        let mut a: Vec<u64> = (0..m * k).map(|i| (i / k) as u64).collect();
        let mut c: Vec<u64> = (0..m * n).map(|i| (i / n) as u64).collect();
        let seen = Mutex::new(Vec::new());

        pool.for_each_row_block_mut(m, 4, [(&mut a, k), (&mut c, n)], |r, [ab, cb]| {
            assert_eq!(ab.len(), r * k, "A 块长度与块内行数不符");
            assert_eq!(cb.len(), r * n, "C 块长度与块内行数不符");
            // 预填的值就是行号本身，所以块首元素即本块起始行
            let row0 = ab[0] as usize;
            assert_eq!(cb[0], row0 as u64, "两个数组的行区间没对齐");
            for i in 0..r {
                assert_eq!(ab[i * k], (row0 + i) as u64, "A 块内行号不连续");
                assert_eq!(cb[i * n], (row0 + i) as u64, "C 块内行号不连续");
                for j in 0..n {
                    cb[i * n + j] += 1; // 每行元素恰好 +1 一次 ⇔ 覆盖恰好一次
                }
            }
            seen.lock().unwrap().push((row0, r));
        });

        let mut seen = seen.into_inner().unwrap();
        seen.sort_unstable();
        let mut next = 0usize;
        for (row0, r) in seen {
            assert_eq!(row0, next, "行区间有重叠或空洞");
            next = row0 + r;
        }
        assert_eq!(next, m, "没有覆盖全部行");
        let want: Vec<u64> = (0..m * n).map(|i| (i / n) as u64 + 1).collect();
        assert_eq!(c, want, "行块写回的位置不对");
    }

    /// 求和结果必须与串行一致（验证数据确实被完整覆盖）。
    #[test]
    fn test_sum_matches_serial() {
        let mut pool = WorkerPool::new(6);
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
        let mut pool = WorkerPool::new(2);
        let mut a = vec![0u8; 10];
        let mut b = vec![0u8; 11];
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.for_each_chunks_mut([a.as_mut_slice(), b.as_mut_slice()], |_| {});
        }));
        assert!(r.is_err(), "长度不一致应当 panic");
    }

    /// 行块接口也要拒绝"长度与 rows×行宽 不符"，不做静默截断。
    #[test]
    fn test_row_block_shape_mismatch_panics() {
        let mut pool = WorkerPool::new(4);
        let mut a = vec![0f32; 100];
        let mut c = vec![0f32; 99];
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.for_each_row_block_mut(10, 4, [(&mut a, 10), (&mut c, 10)], |_, _| {});
        }));
        assert!(r.is_err(), "形状不符应当 panic");
    }

    /// worker 里 panic 必须上抛给调用方，**不能让主线程永久自旋**。
    #[test]
    fn test_panic_in_closure_propagates_and_pool_survives() {
        let mut pool = WorkerPool::new(4);
        let mut data = vec![0u32; 100_000];
        // 只让**一个** chunk panic：其余 worker 仍要正常收工，done 才能凑齐
        let calls = AtomicCount::new(0);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.for_each_chunk_mut(data.as_mut_slice(), |chunk| {
                if calls.fetch_add(1, Ordering::Relaxed) == 0 {
                    panic!("闭包里的故障");
                }
                let _ = chunk;
            });
        }));
        let msg = r.expect_err("worker 的 panic 必须上抛");
        let text = msg
            .downcast_ref::<&str>()
            .map(|s| (*s).to_string())
            .or_else(|| msg.downcast_ref::<String>().cloned())
            .unwrap_or_default();
        assert!(text.contains("闭包里的故障"), "panic 载荷丢失：{text:?}");

        // panic 之后池还能正常干活（状态干净、没有 worker 卡在旧代次）
        let mut data = vec![0u32; 100_000];
        pool.for_each_chunk_mut(data.as_mut_slice(), |chunk| {
            for v in chunk.iter_mut() {
                *v += 1;
            }
        });
        assert!(data.iter().all(|&v| v == 1), "panic 之后池不可复用");
    }

    #[test]
    fn test_pool_is_reusable_across_many_calls() {
        let mut pool = WorkerPool::new(4);
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
