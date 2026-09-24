//! 可选的常驻工作线程池：把批量内核铺到多核，并**跨调用复用线程**。
//!
//! # 为什么放进库里
//!
//! 多步传播这类真实负载里，"每次调用都新建线程"的代价摊不掉：实测每步约 0.5 ms
//! 的创建/回收开销，而多星多步传播端到端因此慢 2.1–2.3×（见 `docs/dev.md`
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
//! 矩阵乘这种"只读共享 B、按行切 A 与 C"的形状：
//!
//! ```no_run
//! use lasx_rs::pool::WorkerPool;
//! # fn main() {
//! # let (m, k, n) = (256usize, 256usize, 256usize);
//! # let (mut a, b, mut c) = (vec![0f32; m * k], vec![0f32; k * n], vec![0f32; m * n]);
//! let pool = WorkerPool::new(12);
//! let (bs, cs) = (b.as_slice(), c.as_mut_slice()); // 只读的 B 用 & 捕获即可
//! pool.for_each_row_block_mut(m, 4, [(&mut a, k), (cs, n)], |_start, rows, [ab, cb]| {
//!     lasx_rs::lasx_matmul(
//!         rows as i32, k as i32, n as i32,
//!         ab.as_ptr(), bs.as_ptr(), cb.as_mut_ptr(),
//!     );
//! });
//! # }
//! ```
//!
//! # 行块闭包的契约（`start` 是干什么用的）
//!
//! 行块接口的闭包签名是 `f(start: usize, rows: usize, arrays)`：`start` 是**本块起始行**。
//! 池只把"本块"的切片交给闭包，所以：
//!
//! | 数组怎么进来的 | 闭包能做什么 | 越界会怎样 |
//! |---|---|---|
//! | 作为 `arrays` 里的 `&mut [T]` | 读写**本块**（切片长度就是 `rows × 行宽`） | 下标越界 → `panic`（安全，不是 UB） |
//! | 闭包**捕获**的 `&[T]`（如只读的 `A`） | **只读 `[start, start + rows)` 这一段** | 读错别的块 → **静默错误结果**（类型系统查不了） |
//!
//! 第二条是这套接口唯一"靠约定而不是靠编译器"的地方，所以 `start` 必须传出来：只读数组
//! 可以整个捕获进闭包，**按 `start` 切片**即可，不需要 `unsafe`、也不需要"把只读入参
//! 假装成 `&mut`"。上层（`shape::Prepared`）生成这个闭包时必须用 `start` 切片。
//!
//! 第三条相关约定（"同一个数组既作只读捕获又作 `&mut` 输出"，即 `y = a·x + y` 里的 `y`）：
//! 在安全 Rust 里**已经被借用检查器拒绝**——不可能同时持有同一缓冲的 `&` 与 `&mut`。
//! 真要走这条（比如原地累加）必须自己用 `unsafe`，那时契约是"该数组只能读写本块"。
//!
//! # 为什么是 Rust API 而不是 C ABI
//!
//! 池需要调用方的数据切片与一个闭包，走 `extern "C"` 就得引入不透明句柄、C 函数指针
//! 与手工生命周期管理。本库同时提供 `rlib`，Rust 调用方（本仓库的基准、YouLiLong
//! 原生扩展、loong-sci 等）直接 `use` 即可，**不新增任何导出符号**。
//!
//! # 派活接口取 `&self`，内部把派活者串行化
//!
//! 池只有一套作业槽与代次计数器，**同时只允许一个调用方派活**。这个约束现在由池内部
//! 一把 `Mutex` 在运行期保证（`WorkerPool::busy`）：多个线程可以同时持有 `&WorkerPool`
//! 并发调用派活接口，它们会**排队**而不是变成静默的数据竞争；worker 的读写仍由
//! "代次发布（Release）+ `done` 计数（Acquire）" 这对握手保护。
//!
//! 为什么要 `&self` 而不是 `&mut self`：要有一个**进程级共享池**（[`global`]），
//! 让 `Auto` 策略不必每次派活都从调用点接一个 `&mut WorkerPool`。代价是重入派活
//! （例如在 [`WorkerPool::for_each_row_block_mut_picked_deferred`] 的 `during` 回调里
//! 再调派活接口）不再被借用检查器挡住 —— 池用一个**线程局部标志**把这种情况变成一条
//! 明确的 panic，而不是在 `Mutex` 上死等。
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
//! 会灾难性地互相抢执行槽**——24 逻辑核 /12 物理核上实测慢 10 倍（dev.md §3.1）。
//!
//! # 什么时候不要用
//!
//! 一次调用涉及的元素总数小于 [`MIN_PARALLEL_LEN`] 时自动原地串行执行：派活/唤醒的
//! 开销不值得。

pub mod sched;

pub use sched::{pick_rows, Job, Pick, Strategy};

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
    /// **多块模式**：各数组的基址与**每行字节数**（行主序），worker 逐块自行派生指针。
    /// 存字节数是因为 worker 循环对元素类型 `T` 不透明（它只有擦除后的 `*mut u8`）。
    /// 单块模式下不用（`ptrs`/`lens` 已经算好）。
    bases: [*mut u8; MAX_ARRAYS],
    row_bytes: [usize; MAX_ARRAYS],
    /// 每行**元素个数**（切片长度用它，指针偏移用上面的 `row_bytes`）。
    widths: [usize; MAX_ARRAYS],
    /// 本轮作业表长度；`0` 表示"单块模式"（用 `ptrs`/`lens`/`rows`）。
    n_jobs: usize,
    /// 本块起始行（多块模式下每个作业各自的起始行）——闭包拿它去索引自己捕获的只读数组。
    start: usize,
}

/// 类型擦除的入口：把裸指针还原成 `[&mut [T]; N]` 并调用闭包。
///
/// `ptrs[k]`/`lens[k]` 是第 k 个数组的起始地址与元素个数，`rows` 是本块行数
/// （按元素切分时就是元素个数），`func` 指向调用方栈上的闭包。
type Thunk =
    unsafe fn(start: usize, rows: usize, ptrs: &[*mut u8], lens: &[usize], func: *const ());

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
    /// 本轮作业表（多块模式）。调用方在 epoch 递增前写入，worker 在轮内只读。
    jobs: UnsafeCell<sched::Jobs>,
    /// 动态领取的下一个作业号（仅 `dynamic` 为真时使用）。
    next_job: AtomicUsize,
    /// 本轮是否动态领取。
    dynamic: AtomicBool,
    slots: Vec<UnsafeCell<Slot>>,
    /// 本轮的类型擦除入口与闭包指针。
    call: UnsafeCell<(Thunk, *const ())>,
    /// 仅用于"自旋超限后 park / 发布后唤醒"，不保护数据。
    park: Mutex<()>,
    wake: Condvar,
}

// SAFETY: 数据竞争由"派活互斥 + 代次发布 + done 计数"排除——
// - `WorkerPool::busy` 这把 `Mutex` 把**派活者**串行化：同一时刻只有一个调用方在写
//   `slots`/`call`/`jobs`（它覆盖了原来 `&mut self` 提供的那个保证）；
// - `slots[i]` 只由第 i 个 worker 访问；
// - `call` 与所有 `slots` 都在 `epoch` 递增（Release）**之前**由调用方写入，
//   而调用方在 `done == threads`（Acquire）之后才返回，因此 worker 读取期间
//   调用方不会改动它们；
// - 串行快路径（不派活）只读不可变字段，与派活者并发是安全的；
// - epoch/done/stop/panicked 都是原子量，payload 由 `Mutex` 保护。
// SAFETY: 见上面那条不变量清单（派活互斥 + 代次发布 + 原子计数）。
unsafe impl Sync for Shared {}
// SAFETY: 同上——`Shared` 只被 `Arc` 在线程间传递，内部可变状态由上面的规则保护。
unsafe impl Send for Shared {}

thread_local! {
    /// 本线程是否**正在**某个池里派活。
    ///
    /// 取 `&self` 之后，借用检查器不再能阻止重入派活（例如在 `during` 回调里再调
    /// 派活接口）。那种情况会在 `busy` 上死等，比崩溃更难查——所以用这个标志把它
    /// 变成一条明确的 panic。
    static DISPATCHING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// 派活令牌：先置线程局部标志（重入即 panic），再拿池的 `busy` 锁（跨线程排队）。
///
/// 字段顺序即释放顺序的反序：先建的先析构，所以 `busy` 的 `MutexGuard` 比标志后释放，
/// 标志清零时锁已经确定会随后释放——两者都在本函数的调用栈上，不存在中间态。
struct DispatchToken<'a> {
    _busy: std::sync::MutexGuard<'a, ()>,
    _reentry: ReentryCleared,
}

/// 析构时清掉线程局部标志。
struct ReentryCleared;

impl Drop for ReentryCleared {
    fn drop(&mut self) {
        DISPATCHING.with(|d| d.set(false));
    }
}

impl<'a> DispatchToken<'a> {
    /// 取令牌。**重入**（本线程已在派活）直接 panic；跨线程则排队等锁。
    ///
    /// `busy` 这把锁保护的是"同一时刻只有一个派活者"这条不变量，锁里的载荷是 `()`；
    /// 因此中毒（上一次派活 panic 过）不影响正确性——panic 会被原样续抛给调用方，池
    /// 状态在此之前已经收干净——所以这里忽略中毒，取回内部值即可。
    fn acquire(pool: &'a WorkerPool) -> Self {
        if DISPATCHING.with(|d| d.replace(true)) {
            panic!("WorkerPool 不支持重入派活：在派活期间（含 during 回调）再次调用派活接口会死锁");
        }
        let busy = pool
            .busy
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        DispatchToken {
            _busy: busy,
            _reentry: ReentryCleared,
        }
    }
}

/// 常驻工作线程池。
///
/// 线程在 [`WorkerPool::new`] 时创建一次，之后每次并行调用只做一次原子发布 + 自旋等待。
pub struct WorkerPool {
    shared: Arc<Shared>,
    handles: Vec<std::thread::JoinHandle<()>>,
    threads: usize,
    /// 是否有已发布但未 `wait()` 的作业（`publish_no_wait`/`wait` 配对用）。
    in_flight: AtomicBool,
    /// 派活互斥：同一时刻只允许一个派活者写作业槽（见 [`DispatchToken`]）。
    busy: Mutex<()>,
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
            jobs: UnsafeCell::new(sched::Jobs::new()),
            next_job: AtomicUsize::new(0),
            dynamic: AtomicBool::new(false),
            slots: (0..threads)
                .map(|_| {
                    UnsafeCell::new(Slot {
                        ptrs: [std::ptr::null_mut(); MAX_ARRAYS],
                        lens: [0; MAX_ARRAYS],
                        n: 0,
                        rows: 0,
                        bases: [std::ptr::null_mut(); MAX_ARRAYS],
                        row_bytes: [0; MAX_ARRAYS],
                        widths: [0; MAX_ARRAYS],
                        n_jobs: 0,
                        start: 0,
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
            in_flight: AtomicBool::new(false),
            busy: Mutex::new(()),
        }
    }

    /// 按 `available_parallelism` 建池。
    ///
    /// 注意它给的是**逻辑**核数（本机 24 = 12 物理核 ×2 SMT）；超线程对这类带宽受限的
    /// 批量内核没有额外收益，也不会更差（dev.md §3.1）。
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
        let len = data.len();
        if self.threads == 1 || len < MIN_PARALLEL_LEN {
            f(data);
            return;
        }
        let base = data.as_mut_ptr();
        let rows_per = len.div_ceil(self.threads).max(1);
        self.dispatch_rows::<T, 1, _, _>([base], [1], len, rows_per, |_s, _rows, [c]| f(c), || {});
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
        if self.threads == 1 || len.saturating_mul(N) < MIN_PARALLEL_LEN {
            f(arrays);
            return;
        }
        let bases = arrays.map(|a| a.as_mut_ptr());
        let rows_per = len.div_ceil(self.threads).max(1);
        self.dispatch_rows::<T, N, _, _>(bases, [1; N], len, rows_per, |_s, _rows, a| f(a), || {});
    }

    /// 按**行块**切分：`arrays[k]` 是 `(行主序数组, 每行元素数)`，
    /// 所有数组按**同一段行区间**切块——数组间的行宽可以不同。
    ///
    /// 这是矩阵乘的形状：`A` 每行 `k` 个元素、`C` 每行 `n` 个元素，但两者必须同步
    /// 切在同一批行上；`B` 只读共享，由闭包以 `&` 捕获即可。
    ///
    /// 闭包参数是 **`(start, rows, arrays)`**：`start` 是本块起始行、`rows` 是本块行数
    /// （`rows × 行宽` 反推会在行宽为 0 时除零，所以直接给）。只读的输入数组（如 `A`）
    /// 由闭包**按 `start` 切片**使用——这是本接口唯一靠约定的地方，见模块头
    /// "行块闭包的契约"。
    ///
    /// `row_gran` 是内核一次处理的行数（如 `lasx_matmul` 为 4；没有行结构就传 1）：
    /// 块大小会**向上**取整到它的倍数，代价是块数可能少于线程数，换来的是块内没有
    /// 退化的尾块。别省这个参数——实测矩阵乘 256 行 12 线程，`row_gran=1` 切出
    /// 22 行/块（5×4+2，两个单行尾块）比 `row_gran=4` 切出 24 行/块**慢一倍**
    /// （见 `docs/dev.md` §14.6）。
    ///
    /// # Panics
    /// `row_gran == 0`，或第 k 个数组的长度不等于 `rows × 行宽` 时 panic。
    pub fn for_each_row_block_mut<T: Send, const N: usize, F>(
        &self,
        rows: usize,
        row_gran: usize,
        arrays: [(&mut [T], usize); N],
        f: F,
    ) where
        F: Fn(usize, usize, [&mut [T]; N]) + Sync,
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
            // 原地串行：整段就是"一块"，起始行恒为 0
            f(0, rows, arrays.map(|(s, _)| s));
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
        self.dispatch_rows::<T, N, _, _>(bases, widths, rows, rows_per, f, || {});
    }

    /// 按**编译期策略** `S` 切分派活（行块形状）。语义与
    /// [`Self::for_each_row_block_mut`] 完全一致，只是"怎么切"由策略决定：
    /// 策略是零尺寸类型 ⇒ 每处调用各自单态化，派活路径上**没有运行时分支**。
    ///
    /// ```ignore
    /// pool.for_each_row_block_mut_with::<sched::RowBlock, _, 2, _>(
    ///     m, 4, [(&mut a, k), (c, n)], |rows, [ab, cb]| { /* … */ });
    /// ```
    pub fn for_each_row_block_mut_with<S: Strategy, T: Send, const N: usize, F>(
        &self,
        rows: usize,
        row_gran: usize,
        arrays: [(&mut [T], usize); N],
        f: F,
    ) where
        F: Fn(usize, usize, [&mut [T]; N]) + Sync,
    {
        self.row_block_static::<S, T, N, F, fn()>(rows, row_gran, arrays, f, || {});
    }

    /// 与上面同，但**动态领取**：块长 `block_rows`，worker 用原子计数器抢块。
    ///
    /// 适合负载不均或机器有后台抢占的场景（块足够碎时把线程间差异压到最小）。
    pub fn for_each_row_block_mut_dynamic<T: Send, const N: usize, F>(
        &self,
        rows: usize,
        row_gran: usize,
        block_rows: usize,
        arrays: [(&mut [T], usize); N],
        f: F,
    ) where
        F: Fn(usize, usize, [&mut [T]; N]) + Sync,
    {
        self.row_block_run(
            rows,
            row_gran,
            arrays,
            Pick::Dynamic { block_rows },
            f,
            || {},
        );
    }

    pub fn for_each_row_block_mut_picked<T: Send, const N: usize, F>(
        &self,
        rows: usize,
        row_gran: usize,
        arrays: [(&mut [T], usize); N],
        pick: Pick,
        f: F,
    ) where
        F: Fn(usize, usize, [&mut [T]; N]) + Sync,
    {
        self.row_block_run(rows, row_gran, arrays, pick, f, || {});
    }

    /// 与上面同，但额外接受一个 **`during` 闭包**：它在本轮**已发布、尚未等待**时
    /// 由主线程执行——也就是"趁 worker 干活，主线程做点别的事"（矩阵乘的双缓冲打包）。
    ///
    /// 为什么用 scoped 回调而不是 `submit()`/`wait()` 两个方法：闭包 `f` 活在本函数的
    /// 栈帧上，一旦 `submit()` 提前返回、worker 还在调它，就是悬垂引用。把"等待期间做的事"
    /// 变成传进来的 `during`，`f` 与数据在整个过程内都活着 ⇒ **不需要 unsafe**；
    /// 而且 `during` 无法捕获 `arrays`（已被本调用移走）或 `&mut pool`（已借出），
    /// 借用检查器替我们挡掉了数据竞争。
    pub fn for_each_row_block_mut_picked_deferred<T: Send, const N: usize, F, D>(
        &self,
        rows: usize,
        row_gran: usize,
        arrays: [(&mut [T], usize); N],
        pick: Pick,
        f: F,
        during: D,
    ) where
        F: Fn(usize, usize, [&mut [T]; N]) + Sync,
        D: FnOnce(),
    {
        self.row_block_run(rows, row_gran, arrays, pick, f, during);
    }

    /// [`Self::for_each_row_block_mut_picked`] 与 [`Self::submit_row_block_mut_picked`]
    /// 的公共实现；`wait_now = false` 表示只发布不等（见 `publish_no_wait` 的 Safety）。
    #[allow(clippy::too_many_arguments)]
    fn row_block_run<T: Send, const N: usize, F, D>(
        &self,
        rows: usize,
        row_gran: usize,
        mut arrays: [(&mut [T], usize); N],
        pick: Pick,
        f: F,
        during: D,
    ) where
        F: Fn(usize, usize, [&mut [T]; N]) + Sync,
        D: FnOnce(),
    {
        match pick {
            Pick::Chunk => {
                self.row_block_static::<sched::Chunk, T, N, F, D>(rows, row_gran, arrays, f, during)
            }
            Pick::RowBlock => self
                .row_block_static::<sched::RowBlock, T, N, F, D>(rows, row_gran, arrays, f, during),
            Pick::Blocked { block_rows } => {
                if self.row_blocks_serial(rows, row_gran, &arrays) {
                    // 原地串行：整段一块，起始行 0
                    f(0, rows, arrays.map(|(s, _)| s));
                    during();
                    return;
                }
                let (bases, widths) = row_block_ptrs(&mut arrays);
                let jobs = sched::plan_blocked(rows, block_rows, row_gran);
                self.dispatch_jobs::<T, N, _, _>(bases, widths, jobs, false, f, during);
            }
            Pick::Dynamic { block_rows } => {
                if self.row_blocks_serial(rows, row_gran, &arrays) {
                    // 原地串行：整段一块，起始行 0
                    f(0, rows, arrays.map(|(s, _)| s));
                    during();
                    return;
                }
                let (bases, widths) = row_block_ptrs(&mut arrays);
                let jobs = sched::plan_blocked(rows, block_rows, row_gran);
                self.dispatch_jobs::<T, N, _, _>(bases, widths, jobs, true, f, during);
            }
        }
    }

    /// 静态策略（`Chunk`/`RowBlock`）的公共实现。
    fn row_block_static<S: Strategy, T: Send, const N: usize, F, D>(
        &self,
        rows: usize,
        row_gran: usize,
        mut arrays: [(&mut [T], usize); N],
        f: F,
        during: D,
    ) where
        F: Fn(usize, usize, [&mut [T]; N]) + Sync,
        D: FnOnce(),
    {
        if self.row_blocks_serial(rows, row_gran, &arrays) {
            // 原地串行：整段一块，起始行 0
            f(0, rows, arrays.map(|(s, _)| s));
            during();
            return;
        }
        let (bases, widths) = row_block_ptrs(&mut arrays);
        let jobs = S::plan(rows, self.threads, row_gran);
        if !S::MULTI && !jobs.is_empty() {
            self.dispatch_rows(bases, widths, rows, jobs[0].rows, f, during);
            return;
        }
        self.dispatch_jobs::<T, N, _, _>(bases, widths, jobs, false, f, during);
    }

    /// 行块接口的公共前置：校验各数组长度，并判断本次是否**原地串行**（不派活）。
    ///
    /// 只借用切片（不取指针），这样串行分支可以直接把原切片交给闭包，
    /// 不需要从裸指针重建、也就没有生命周期把戏。
    fn row_blocks_serial<T, const N: usize>(
        &self,
        rows: usize,
        row_gran: usize,
        arrays: &[(&mut [T], usize); N],
    ) -> bool {
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
        self.threads == 1 || rows < 2 || total < MIN_PARALLEL_LEN
    }

    /// 派活内核：第 k 个数组按行宽 `widths[k]` 解释，共 `rows` 行；每个 worker 取一段连续行。
    ///
    /// 各数组长度必须等于 `rows × widths[k]`（由上面的公开接口校验），
    /// 否则下面的指针算术就越界了。`rows_per` 是本轮的块大小（≥ 1）。
    fn dispatch_rows<T: Send, const N: usize, F, D>(
        &self,
        bases: [*mut T; N],
        widths: [usize; N],
        rows: usize,
        rows_per: usize,
        f: F,
        during: D,
    ) where
        F: Fn(usize, usize, [&mut [T]; N]) + Sync,
        D: FnOnce(),
    {
        const { assert!(N <= MAX_ARRAYS, "一次派活的数组个数超过 MAX_ARRAYS") };
        // 派活令牌：串行化所有派活者（跨线程排队），并把重入派活变成明确 panic。
        // 从填槽一直持到 `wait()` 返回——这正是 worker 读写这些槽的整个窗口。
        let _token = DispatchToken::acquire(self);
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
            slot.start = start;
            slot.n_jobs = 0; // 单块模式
            for k in 0..N {
                let off = start * widths[k];
                // SAFETY: 数组长度为 rows*widths[k]（调用方已校验），
                // 而 off <= rows*widths[k]，故地址在界内（至多尾后一格）。
                slot.ptrs[k] = unsafe { bases[k].add(off) } as *mut u8;
                slot.lens[k] = r * widths[k];
            }
        }

        // 发布 → **主线程做 `during`（worker 正在算）** → 等待。
        // SAFETY: 槽位覆盖本次调用的全部数据、各数组长度由公开接口校验为 `rows × widths[k]`、
        // 行区间互不重叠；`f` 与 `during` 都在本函数栈帧上活到 `wait()` 之后。
        // SAFETY: 槽位已填好、长度由公开接口校验；`f` 与 `during` 都在本栈帧上活到 `wait()` 之后。
        unsafe { self.publish_no_wait(thunk::<T, N, F>, &f as *const F as *const ()) };
        // 发布之后立刻上守卫：`during` 是用户代码，panic 展开也必须先收工（否则 worker 在
        // 调一个已经析构的栈上闭包）。正常路径下面 disarm，交给 `wait()` 收尾并续抛 panic。
        let mut guard = WaitGuard::new(self);
        during();
        guard.disarm();
        self.wait();
    }

    /// 按**作业表**派活：支持"每 worker 多块"（静态轮转）与"动态领取"两种模式。
    ///
    /// 与 [`Self::dispatch_rows`] 的区别只有分配方式：那个是"每 worker 一段连续行"，
    /// 这里是"一张 `(起始行, 行数)` 作业表 + 领取规则"。规则由 [`Pick`] 穷尽决定：
    ///
    /// - `Pick::Blocked`：静态轮转——worker `w` 领 `jobs[w], jobs[w+T], …`（`T` = 线程数）。
    ///   块数多于线程数时能吸收轻度负载不均，且没有原子竞争。
    /// - `Pick::Dynamic`：动态领取——所有 worker 用 `next_job` 的 `fetch_add` 抢块。
    ///   块足够碎时把线程间的不均压到最小（机器有后台抢占时尤其有用），代价是每块一次原子操作。
    ///
    /// # Panics
    /// 各数组长度必须等于 `rows × 行宽`；`jobs` 必须恰好覆盖 `[0, rows)`。
    fn dispatch_jobs<T: Send, const N: usize, F, D>(
        &self,
        bases: [*mut T; N],
        widths: [usize; N],
        jobs: sched::Jobs,
        dynamic: bool,
        f: F,
        during: D,
    ) where
        F: Fn(usize, usize, [&mut [T]; N]) + Sync,
        D: FnOnce(),
    {
        const { assert!(N <= MAX_ARRAYS, "一次派活的数组个数超过 MAX_ARRAYS") };
        let threads = self.threads;
        // 派活令牌：从填槽一直持到 `wait()` 返回，替代原来 `&mut self` 提供的独占保证
        let _token = DispatchToken::acquire(self);
        // SAFETY: 令牌保证同一时刻只有一个派活者在写 `jobs`，且写入发生在 epoch 递增之前。
        unsafe { *self.shared.jobs.get() = jobs };
        self.shared.dynamic.store(dynamic, Ordering::Relaxed);
        self.shared.next_job.store(0, Ordering::Relaxed);
        // SAFETY: `jobs` 刚由本函数写入（令牌在手，且在发布之前），本轮内无别名写入。
        let n_jobs = unsafe { (*self.shared.jobs.get()).len() };
        // 下标即 worker 编号（slots[w] 必须与第 w 个 worker 对应），故用下标循环
        #[allow(clippy::needless_range_loop)]
        for w in 0..threads {
            // SAFETY: slots[w] 只由第 w 个 worker 访问
            let slot = unsafe { &mut *self.shared.slots[w].get() };
            // 静态轮转下第 w 个 worker 至少有一个作业才要干活（动态模式则人人可能要抢）
            let has_work = dynamic || w < n_jobs;
            if !has_work {
                slot.n = 0;
                continue;
            }
            slot.n = N;
            slot.rows = 0;
            slot.n_jobs = n_jobs;
            slot.start = 0; // 多块模式：起始行由每个作业带（worker 侧传）
            for k in 0..N {
                slot.bases[k] = bases[k] as *mut u8;
                // 行宽同时存"元素个数"（切片长度）与"字节数"（指针偏移）：
                // worker 侧对 `T` 不透明，指针只能按字节算。
                slot.widths[k] = widths[k];
                slot.row_bytes[k] = widths[k] * std::mem::size_of::<T>();
            }
        }
        // 发布 → **主线程做 `during`（worker 正在算）** → 等待
        // SAFETY: 槽位已填好、长度由公开接口校验；`f` 与 `during` 都在本栈帧上活到 `wait()` 之后。
        unsafe { self.publish_no_wait(thunk::<T, N, F>, &f as *const F as *const ()) };
        // 同上：`during` panic 时必须先收工，避免栈上闭包被 worker 继续调用
        let mut guard = WaitGuard::new(self);
        during();
        guard.disarm();
        self.wait();
    }

    /// 发布一轮并等到全部 worker 完成（有 panic 则续抛）。
    ///
    /// # Safety
    /// 调用方必须保证：所有 `slots` 已填好、`func` 与槽位里的指针在本次调用期间有效，
    /// 且 `func` 指向的闭包类型与 `t` 期望的一致。
    /// 只发布、不等待（**unsafe**）：调用方**必须**随后调用 [`Self::wait`]。
    ///
    /// 存在的意义是让调用方在 worker 干活时做"只有主线程能做的事"——典型用途是矩阵乘
    /// 的**双缓冲打包**：worker 算当前面板时，主线程打包下一个面板（`docs/dev.md` §16）。
    /// 对外它只经 [`WorkerPool::for_each_row_block_mut_picked_deferred`] 暴露：那个门面把
    /// "等待期间做的事"做成 `during` 回调，闭包与数据都活在同一个栈帧里，因而不需要 unsafe。
    ///
    /// # Safety
    /// 在 `wait()` 返回之前，调用方不得访问本次派活的数组或闭包捕获的数据（worker 正在
    /// 读写它们），也不得再次调用本池的派活接口（池只有一套作业槽与代次）。
    unsafe fn publish_no_wait(&self, t: Thunk, func: *const ()) {
        *self.shared.call.get() = (t, func);
        self.shared.done.store(0, Ordering::Relaxed);
        self.shared.panicked.store(false, Ordering::Relaxed);
        {
            let _g = self.shared.park.lock().unwrap();
            self.shared.epoch.fetch_add(1, Ordering::Release);
        }
        self.shared.wake.notify_all();
        self.in_flight.store(true, Ordering::Relaxed);
    }

    /// 等到本轮全部 worker 完成（有 panic 则续抛）。没有在飞作业时直接返回。
    ///
    /// 取 `&self` 之后，这里只做"自己刚发布的那一轮"的收尾：调用方必须已经持有派活令牌
    /// （[`DispatchToken`]），所以 `in_flight` 不会被另一个派活者中途改掉。
    ///
    /// **不对外**：它必须与私有的 [`Self::publish_no_wait`] 成对使用，单独调用没有意义
    /// （还可能替别人收尾）。对外只需要三个 `for_each_*` 门面。
    fn wait(&self) {
        if !self.in_flight.load(Ordering::Relaxed) {
            return;
        }
        let payload = self.finish_round();
        if let Some(p) = payload {
            std::panic::resume_unwind(p);
        }
    }

    /// 等本轮收工并取走 worker 的 panic 载荷（**不续抛**）。没有在飞作业时返回 `None`。
    ///
    /// 与 [`Self::wait`] 的区别只有一个：不 panic。用于**已经在展开**的路径
    /// （[`WaitGuard`] 的析构）——那里再 `resume_unwind` 会变成双重 panic 而 abort。
    fn wait_quiet(&self) -> Option<Box<dyn Any + Send>> {
        if !self.in_flight.load(Ordering::Relaxed) {
            return None;
        }
        self.finish_round()
    }

    /// 收尾内核：清 `in_flight`、等 `done` 到齐、取走（并清掉）panic 载荷。
    fn finish_round(&self) -> Option<Box<dyn Any + Send>> {
        self.in_flight.store(false, Ordering::Relaxed);
        while self.shared.done.load(Ordering::Acquire) < self.threads {
            std::hint::spin_loop();
        }
        if self.shared.panicked.load(Ordering::Acquire) {
            // 池已收工，状态是干净的：把 panic 交给调用方，池本身可继续复用。
            self.shared.panicked.store(false, Ordering::Relaxed);
            return self.shared.payload.lock().unwrap().take();
        }
        None
    }
}

/// "发布之后一定要收工"的析构守卫。
///
/// 为什么需要它：`during` 是**用户代码**，它 panic 时展开会跳过 `wait()`。而 worker 此刻
/// 正在调用 `f`——`f` 活在本函数的栈帧上，栈一展开就是**悬垂闭包**，那是 UB，
/// 比"池挂死"严重得多。所以发布之后立刻挂上这个守卫，任何路径（含 panic 展开）都会先
/// 收工再走。
///
/// 正常路径由调用方 [`WaitGuard::disarm`] 交还给 `wait()`——那一步会**续抛** worker 的
/// panic；守卫自己只在展开路径上兜底，且只收工、不续抛（展开中再 panic 会 abort）。
struct WaitGuard<'a> {
    pool: &'a WorkerPool,
    armed: bool,
}

impl<'a> WaitGuard<'a> {
    fn new(pool: &'a WorkerPool) -> Self {
        WaitGuard { pool, armed: true }
    }

    /// 正常路径：把收尾交还给 `wait()`。
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for WaitGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            // 可能已经在展开中：只收工，绝不续抛 worker 的 panic。
            drop(self.pool.wait_quiet());
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

/// 进程级共享池的存储；[`global`] 与 [`init_global`] 共用同一个。
static GLOBAL: std::sync::OnceLock<WorkerPool> = std::sync::OnceLock::new();

/// 进程级共享池：首次使用时就地建一个按 `available_parallelism` 配线程数的池。
///
/// 这是"一个进程一个池"的默认入口，[`parallel`](crate::parallel) 与上层的 `Auto` 策略
/// 都走它。多个线程可以同时拿 `&'static WorkerPool` 调用派活接口——池内部会把派活者
/// 串行化（见模块头"派活接口取 `&self`"），所以同一个池可以被整进程共用，不会像
/// "每个线程各建一个池"那样把线程数乘起来。
///
/// 想固定线程数就在**首次使用前**调 [`init_global`]。
pub fn global() -> &'static WorkerPool {
    GLOBAL.get_or_init(WorkerPool::auto)
}

/// 指定全局池的线程数；返回 `false` 表示全局池**已经**建好，本次调用无效。
///
/// 典型用法是在程序启动时按机器/配置定死：
///
/// ```
/// // 已经建过就返回 false（测试进程里别的用例可能先用过），两种都正确
/// let _ = lasx_rs::pool::init_global(4);
/// assert!(lasx_rs::pool::global().threads() >= 1);
/// ```
pub fn init_global(threads: usize) -> bool {
    GLOBAL.set(WorkerPool::new(threads)).is_ok()
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
        // SAFETY: 调用方在 `done` 计数到齐前不会改写 `call`/`slots`（见 Shared 的不变量清单）。
        let (thunk, func) = unsafe { *sh.call.get() };
        // SAFETY: 同上，且 `slots[i]` 只由第 i 个 worker 访问。
        let slot = unsafe { &*sh.slots[i].get() };
        if slot.n > 0 {
            // 跑一块：多块模式下由 `bases`/`widths`/作业行区间现场派生指针；
            // 单块模式下 `ptrs`/`lens` 已经在派活时算好（热路径不变）。
            let run_block = |start: usize, rows: usize, ptrs: &[*mut u8], lens: &[usize]| {
                // SAFETY: 入口与闭包指针由派活方在发布前写入；`ptrs`/`lens` 是本轮该 worker
                // 自己那段（互不重叠），由派活方按 `rows × widths` 校验过。
                let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                    thunk(start, rows, ptrs, lens, func);
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
                    return false;
                }
                true
            };
            if slot.n_jobs == 0 {
                run_block(
                    slot.start,
                    slot.rows,
                    &slot.ptrs[..slot.n],
                    &slot.lens[..slot.n],
                );
            } else {
                // SAFETY: 作业表在本轮内只读（调用方在 done 计数到齐前不会改它）
                let jobs = unsafe { &*sh.jobs.get() };
                let mut ptrs = [std::ptr::null_mut(); MAX_ARRAYS];
                let mut lens = [0usize; MAX_ARRAYS];
                let mut one = |idx: usize| -> bool {
                    let job = jobs[idx];
                    for k in 0..slot.n {
                        // SAFETY: 数组长度为 rows×widths[k]，job 覆盖在 [0, rows) 内
                        // 字节偏移：bases[k] + 起始行 × 每行字节数
                        ptrs[k] = unsafe { slot.bases[k].add(job.start * slot.row_bytes[k]) };
                        lens[k] = job.rows * slot.widths[k];
                    }
                    run_block(job.start, job.rows, &ptrs[..slot.n], &lens[..slot.n])
                };
                if sh.dynamic.load(Ordering::Relaxed) {
                    loop {
                        let idx = sh.next_job.fetch_add(1, Ordering::Relaxed);
                        if idx >= jobs.len() || !one(idx) {
                            break;
                        }
                    }
                } else {
                    let mut idx = i;
                    while idx < jobs.len() {
                        if !one(idx) {
                            break;
                        }
                        idx += sh.slots.len();
                    }
                }
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
    start: usize,
    rows: usize,
    ptrs: &[*mut u8],
    lens: &[usize],
    func: *const (),
) where
    F: Fn(usize, usize, [&mut [T]; N]),
{
    let arrays: [&mut [T]; N] =
        std::array::from_fn(|k| std::slice::from_raw_parts_mut(ptrs[k] as *mut T, lens[k]));
    let f = &*(func as *const F);
    f(start, rows, arrays);
}

/// 从"行主序切片 + 行宽"取出基址与行宽（派活用）。
fn row_block_ptrs<T, const N: usize>(
    arrays: &mut [(&mut [T], usize); N],
) -> ([*mut T; N], [usize; N]) {
    let mut bases = [std::ptr::null_mut(); N];
    let mut widths = [0usize; N];
    for (k, (s, w)) in arrays.iter_mut().enumerate() {
        bases[k] = s.as_mut_ptr();
        widths[k] = *w;
    }
    (bases, widths)
}

/// 初始占位入口（`call` 需要初值；任何真实调用都会覆盖它）。
unsafe fn noop_thunk<T, const N: usize>(
    _start: usize,
    _rows: usize,
    _p: &[*mut u8],
    _l: &[usize],
    _f: *const (),
) {
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

    /// 行块版本：行宽不同的两个数组必须切在**同一批行**上，且恰好覆盖每一行。
    #[test]
    fn test_row_blocks_pair_arrays_by_row_range() {
        let pool = WorkerPool::new(5);
        let (m, k, n) = (1000usize, 7usize, 11usize);
        // 两个数组都用"行号"预填：行宽不同（k vs n），能对齐才说明切的是同一批行
        let mut a: Vec<u64> = (0..m * k).map(|i| (i / k) as u64).collect();
        let mut c: Vec<u64> = (0..m * n).map(|i| (i / n) as u64).collect();
        let seen = Mutex::new(Vec::new());

        pool.for_each_row_block_mut(m, 4, [(&mut a, k), (&mut c, n)], |_s, r, [ab, cb]| {
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

    /// 行块接口也要拒绝"长度与 rows×行宽 不符"，不做静默截断。
    #[test]
    fn test_row_block_shape_mismatch_panics() {
        let pool = WorkerPool::new(4);
        let mut a = vec![0f32; 100];
        let mut c = vec![0f32; 99];
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.for_each_row_block_mut(10, 4, [(&mut a, 10), (&mut c, 10)], |_, _, _| {});
        }));
        assert!(r.is_err(), "形状不符应当 panic");
    }

    /// worker 里 panic 必须上抛给调用方，**不能让主线程永久自旋**。
    #[test]
    fn test_panic_in_closure_propagates_and_pool_survives() {
        let pool = WorkerPool::new(4);
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

    /// 每种调度策略都必须**恰好访问每一行一次**（不多、不少、不重）。
    /// 这是策略层唯一真正的正确性不变式——`sched` 里的纯函数测试管切分，
    /// 这里管"派活 + 领取规则"真的按切分执行。
    #[test]
    fn row_block_strategies_visit_every_row_once() {
        let pool = WorkerPool::new(5);
        let (rows, width) = (2000usize, 3usize);
        let picks = [
            Pick::Chunk,
            Pick::RowBlock,
            Pick::Blocked { block_rows: 7 },
            Pick::Dynamic { block_rows: 7 },
        ];
        for pick in picks {
            let mut data = vec![0usize; rows * width];
            pool.for_each_row_block_mut_picked(
                rows,
                1,
                [(&mut data, width)],
                pick,
                |_s, n, [d]| {
                    for i in 0..n {
                        for k in 0..width {
                            d[i * width + k] += 1;
                        }
                    }
                },
            );
            assert!(
                data.iter().all(|&v| v == 1),
                "{}: 有行没跑到或跑了多次",
                pick.name()
            );
        }
    }

    /// `&self` 的核心承诺：多个线程可以**同时**拿同一个池派活——内部排队，结果各自正确。
    ///
    /// 这条测试覆盖的是"把 `&mut self` 换成 `&self` + 内部互斥"这件事本身：如果互斥漏了，
    /// 两个派活者会同时写作业槽；如果排队写错了（例如 `in_flight` 被另一个线程清掉），
    /// 就会出现少算/多算或提前返回。
    #[test]
    fn test_shared_pool_concurrent_dispatch() {
        let pool = WorkerPool::new(4);
        let mut bufs: Vec<Vec<u64>> = (0..4).map(|_| (0..20_000).collect()).collect();
        std::thread::scope(|s| {
            for buf in bufs.iter_mut() {
                let pool = &pool;
                s.spawn(move || {
                    for _ in 0..20 {
                        pool.for_each_chunk_mut(buf.as_mut_slice(), |chunk| {
                            for v in chunk.iter_mut() {
                                *v += 1;
                            }
                        });
                    }
                });
            }
        });
        for (n, buf) in bufs.iter().enumerate() {
            assert!(
                buf.iter().enumerate().all(|(i, &v)| v == i as u64 + 20),
                "第 {n} 个缓冲区结果不对——并发派活没有正确排队"
            );
        }
    }

    /// 重入派活必须在**当下** panic，而不是在 `busy` 上死等。
    #[test]
    fn test_reentrant_dispatch_panics_instead_of_deadlock() {
        let pool = WorkerPool::new(2);
        let mut data = vec![0u64; 64 * 128];
        let mut other = vec![0u64; 20_000];
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.for_each_row_block_mut_picked_deferred(
                64,
                1,
                [(&mut data, 128)],
                Pick::RowBlock,
                |_s, _rows, [d]| d[0] += 1,
                || {
                    // 派活期间（`during` 回调）再派活：过去是借用检查器拦，现在必须运行期拦
                    pool.for_each_chunk_mut(other.as_mut_slice(), |c| c[0] += 1);
                },
            );
        }));
        let payload = r.unwrap_err();
        let msg = payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .expect("panic 载荷应当是字符串");
        assert!(msg.contains("重入"), "panic 消息应说明是重入派活：{msg}");

        // 池必须已经收工且可继续用（panic 展开路径由 WaitGuard 兜住，闭包不会悬垂）
        let mut after = vec![0u64; 20_000];
        pool.for_each_chunk_mut(after.as_mut_slice(), |chunk| {
            for v in chunk.iter_mut() {
                *v += 1;
            }
        });
        assert!(
            after.iter().all(|&v| v == 1),
            "重入 panic 之后池应当照常可用"
        );
    }

    /// `during` 在展开路径上必须先把 worker 收干净：否则 worker 会继续调用一个已经
    /// 随着栈展开而析构的闭包（**悬垂闭包 → UB**）。这里验证"展开之后池还能照常派活"，
    /// 池干净只是附带结果；真正守的是"不 UB、不挂死"。
    #[test]
    fn test_during_panic_no_dangling_closure() {
        let pool = WorkerPool::new(4);
        let mut data = vec![0u64; 64 * 128];
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.for_each_row_block_mut_picked_deferred(
                64,
                1,
                [(&mut data, 128)],
                Pick::RowBlock,
                |_s, _rows, [d]| d[0] += 1,
                || panic!("during 故意 panic"),
            );
        }));
        assert!(r.is_err());

        let mut after = vec![0u64; 20_000];
        pool.for_each_chunk_mut(after.as_mut_slice(), |chunk| {
            for v in chunk.iter_mut() {
                *v += 1;
            }
        });
        assert!(
            after.iter().all(|&v| v == 1),
            "during panic 之后池应当照常可用"
        );
    }

    /// `start` 必须与池交给闭包的那一段切片**一一对应**——上层（`shape::Prepared`）正是
    /// 靠它去索引自己捕获的只读数组，错位就是静默的错误结果。这里让每个元素的初值等于
    /// 它所在的行号，闭包就能自查"我拿到的块确实从 `start` 行开始"，并检查各块首尾相接、
    /// 恰好覆盖 `[0, rows)`。
    #[test]
    fn test_row_block_start_matches_slice() {
        use std::sync::atomic::AtomicUsize;
        let pool = WorkerPool::new(5);
        let (rows, width) = (2048usize, 3usize);
        let mut data: Vec<usize> = (0..rows * width).collect();
        for pick in [
            Pick::Chunk,
            Pick::RowBlock,
            Pick::Blocked { block_rows: 7 },
            Pick::Dynamic { block_rows: 7 },
        ] {
            for (i, v) in data.iter_mut().enumerate() {
                *v = i / width; // 每个元素记住自己的行号
            }
            let covered = AtomicUsize::new(0);
            pool.for_each_row_block_mut_picked(
                rows,
                1,
                [(&mut data, width)],
                pick,
                |start, n, [d]| {
                    assert_eq!(
                        d[0],
                        start,
                        "{}: 块的第一个元素不属于 start 行",
                        pick.name()
                    );
                    assert_eq!(
                        d[(n - 1) * width],
                        start + n - 1,
                        "{}: 块的最后一个元素不属于 start+n-1 行",
                        pick.name()
                    );
                    covered.fetch_add(n, Ordering::Relaxed);
                },
            );
            assert_eq!(
                covered.load(Ordering::Relaxed),
                rows,
                "{}: 块没有恰好覆盖全部行",
                pick.name()
            );
        }
    }

    /// 全局池：同一个进程里拿到的必须是**同一个**池（不是每线程/每次各建一个）。
    #[test]
    fn test_global_pool_is_a_singleton() {
        let a = global() as *const WorkerPool;
        let b = global() as *const WorkerPool;
        assert_eq!(a, b);
        assert!(global().threads() >= 1);
        std::thread::scope(|s| {
            // 传 `usize` 而不是裸指针：裸指针不是 `Send`
            let h = s.spawn(|| global() as *const WorkerPool as usize);
            assert_eq!(
                h.join().unwrap(),
                a as usize,
                "另一个线程必须拿到同一个全局池"
            );
        });
        // 已经建过之后 `init_global` 必须无效（返回 false），线程数不变
        let threads = global().threads();
        assert!(!init_global(threads + 1));
        assert_eq!(global().threads(), threads);
    }
}
