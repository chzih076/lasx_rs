//! 矩阵乘**计划句柄**的 C ABI 导出：**打包一次 `B`，之后按块/按线程反复用**。
//!
//! 这一层只做"裸指针 ⇄ Rust 计划"的转换与校验，**不含计算逻辑**；语义权威在
//! `docs/c-abi.md` 的"计划与句柄"一节（C 读者）与 [`crate::plan`]（Rust 读者）。
//!
//! # 线程模型（**不要死板的线程池**）
//!
//! - `lasx_matmul_run` 是**纯调用方驱动**：一次只算一个行块，**线程怎么开、块怎么切由调用方定**
//!   （OpenMP / `std::thread` / 协程 / 单线程串行都行），库不夺控制权。同一个计划可以被多个
//!   线程**并发** `run` 不同行块——计划在 `lasx_matmul_prepare` 之后是**只读**的。
//! - `lasx_matmul_run_threaded` 是**便利层，不是唯一路径**：库内派发，`threads` 可调
//!   （`<=0` = 自动取物理核、`1` = 串行、`N` = N 个 worker），分块用池的**动态取块**（不是固定等分）。
//!   **不用它就完全零进程级状态**：没有全局池、没有环境变量、没有 `atexit`——池是每次调用
//!   现建现拆的（热循环里想省这笔开销就用 `run` + 调用方自己的线程/池）。
//!
//! # 位精确
//!
//! 两条入口都走 [`crate::plan::MatmulPlan`] 的按块内核，与单线程 `lasx_matmul` **逐位一致**
//! （每个输出元素仍沿 `k` 单累加器升序）；`scripts/bench_openblas.c` 的 `--mode=verify`
//! 把这件事当作计时前的断言。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（分层策略：内核模块在函数级统一写 SAFETY，不逐块重复）。
#![allow(clippy::undocumented_unsafe_blocks)]

use super::status::{
    LasxStatus, checked_len, checked_mul, checked_slice, checked_slice_mut, or_fail,
};
use crate::api::Error;
use crate::plan::MatmulPlan;

/// 不透明计划句柄：`lasx_matmul_prepare` 造、`lasx_matmul_plan_destroy` 释放。
///
/// C 侧**只能当 token 传**：不要解引用、不要 `sizeof`、不要复制它的内容。它是
/// "打包好的 B + 几何"，在 `prepare` 之后**只读**，因此可以跨线程共享（各线程写自己那块 `c`）。
pub struct LasxMatmulPlan {
    plan: MatmulPlan<f32>,
}

/// `api::Error` → `LasxStatus`（本模块只会遇到这两种；`checked.rs` 的 `or_fail!` 同口径）。
fn status_of(e: &Error) -> LasxStatus {
    match e {
        Error::Overflow { .. } => LasxStatus::SizeOverflow,
        _ => LasxStatus::BadShape,
    }
}

/// 写一个可选出参（`NULL` = 不关心）。
fn write_i32(out: *mut i32, v: i32) {
    if !out.is_null() {
        // SAFETY: FFI 约定——非空时调用方保证 out 可写一个 i32。
        unsafe { out.write(v) };
    }
}

/// 写一个可选出参（`NULL` = 不关心）。
fn write_f64(out: *mut f64, v: f64) {
    if !out.is_null() {
        // SAFETY: FFI 约定——非空时调用方保证 out 可写一个 f64。
        unsafe { out.write(v) };
    }
}

/// 把 `threads` 解析成实际 worker 数：`<= 0` = 自动（物理核）、`N > 物理核` 夹到物理核、`1` = 串行。
///
/// 与 `crate::pool::WorkerPool::auto()` 同一判据（`parallelism().physical`），不读环境变量。
fn resolve_threads(threads: i32) -> usize {
    let physical = crate::arch::parallelism().physical.max(1);
    if threads <= 0 {
        physical
    } else {
        (threads as usize).min(physical)
    }
}

/// **打包 `B` 一次**，返回计划句柄（裸入口，不校验：误用即 UB）。
///
/// C 签名：`lasx_matmul_plan *lasx_matmul_prepare(int k, int n, const float *b)`
///
/// - `b` 是行主序 `k×n`；**`prepare` 会复制并打包它**，返回后调用方可以释放 `b`；
/// - 失败（尺寸溢出、`from_row_major` 拒绝）返回 `NULL`；要错误通道用
///   [`lasx_matmul_prepare_checked`]；
/// - 内存 ≈ `k·n·4` 字节（打包面板；`n` 不是 32 的倍数时另加一小段列尾副本）。
#[unsafe(no_mangle)]
pub extern "C" fn lasx_matmul_prepare(k: i32, n: i32, b: *const f32) -> *mut LasxMatmulPlan {
    let (k, n) = (k as usize, n as usize);
    // SAFETY: FFI 约定——`b` 为 k×n 可读。
    let b = unsafe { std::slice::from_raw_parts(b, k * n) };
    match MatmulPlan::from_row_major(b, k, n) {
        Ok(plan) => Box::into_raw(Box::new(LasxMatmulPlan { plan })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// 带错误通道的 `lasx_matmul_prepare`。
///
/// C 签名：`lasx_matmul_plan *lasx_matmul_prepare_checked(int k, int n, const float *b, int *status)`
///
/// 校验顺序：`k`/`n` 非负（否则 `NegativeLength`）→ `k×n` 不溢出（否则 `SizeOverflow`）→
/// `b` 非空（长度非 0 时，否则 `NullPointer`）→ 计划构造（否则 `BadShape`）。
/// 失败时返回 `NULL` **且只写 `status`**。`status` 可以传 `NULL`（表示不关心原因，但校验照做）。
#[unsafe(no_mangle)]
pub extern "C" fn lasx_matmul_prepare_checked(
    k: i32,
    n: i32,
    b: *const f32,
    status: *mut i32,
) -> *mut LasxMatmulPlan {
    let k = or_fail!(checked_len(k), status, std::ptr::null_mut());
    let n = or_fail!(checked_len(n), status, std::ptr::null_mut());
    let kn = or_fail!(checked_mul(k, n), status, std::ptr::null_mut());
    // SAFETY: `b` 需可读 k×n（`checked_slice` 校验非空与长度约定）。
    let b = unsafe { or_fail!(checked_slice(b, kn), status, std::ptr::null_mut()) };
    match MatmulPlan::from_row_major(b, k, n) {
        Ok(plan) => {
            LasxStatus::Ok.write(status);
            Box::into_raw(Box::new(LasxMatmulPlan { plan }))
        }
        Err(e) => {
            status_of(&e).write(status);
            std::ptr::null_mut()
        }
    }
}

/// 用计划算**一个行块** `C[rows×n] = A[rows×k]·B`（裸入口，不校验：误用即 UB）。
///
/// C 签名：`void lasx_matmul_run(const lasx_matmul_plan *plan, int rows, const float *a, float *c)`
///
/// - `a` 指向**该块自己**的第一行（`rows×k`），`c` 同理（`rows×n`）——所以调用方可以用任意
///   方式切块（`a + r0*k`、`c + r0*n`）而**不需要**传偏移或块描述；
/// - `rows` 与 4 无关、不整除也支持；`rows == 0` 是合法的空操作；
/// - **同一个 `plan` 可以被多个线程并发调用**（只读共享），各线程写自己的 `c`；
/// - 与整块一次算（`lasx_matmul`）**逐位一致**。
#[unsafe(no_mangle)]
pub extern "C" fn lasx_matmul_run(
    plan: *const LasxMatmulPlan,
    rows: i32,
    a: *const f32,
    c: *mut f32,
) {
    // SAFETY: FFI 约定——`plan` 来自 `lasx_matmul_prepare` 且未 destroy。
    let plan = unsafe { &*plan };
    let rows = rows as usize;
    let (k, n) = (plan.plan.k(), plan.plan.n());
    // SAFETY: FFI 约定——`a` 为 rows×k 可读、`c` 为 rows×n 可写。
    let (a, c) = unsafe {
        (
            std::slice::from_raw_parts(a, rows * k),
            std::slice::from_raw_parts_mut(c, rows * n),
        )
    };
    // 裸入口不校验长度；`run_block_into` 长度不符时返回 `Err` 而不写 `c`（不会 UB）。
    let _ = plan.plan.run_block_into(a, c);
}

/// 带错误通道的 `lasx_matmul_run`。
///
/// C 签名：`void lasx_matmul_run_checked(const lasx_matmul_plan *plan, int rows,
/// const float *a, float *c, int *status)`
///
/// 校验顺序：`plan` 非空（否则 `NullPointer`）→ `rows` 非负（否则 `NegativeLength`）→
/// `rows×k`/`rows×n` 不溢出（否则 `SizeOverflow`）→ `a`/`c` 非空且长度约定（否则 `NullPointer`）。
/// **失败时不触碰 `c`**（可以在原地重试）。`status` 可传 `NULL`。
///
/// 注意：`plan` 里没有 `rows`，所以对**块**的校验只能看"块自己"的 `rows×k` / `rows×n`——
/// "这个块是否落在调用方真正的缓冲里"只有调用方判得了（C 侧拿不到缓冲区长度）。
#[unsafe(no_mangle)]
pub extern "C" fn lasx_matmul_run_checked(
    plan: *const LasxMatmulPlan,
    rows: i32,
    a: *const f32,
    c: *mut f32,
    status: *mut i32,
) {
    // SAFETY: 只读借用；调用方保证句柄有效（destroy 之后再用是 UB）。
    let plan = or_fail!(
        unsafe { plan.as_ref() }.ok_or(LasxStatus::NullPointer),
        status,
        ()
    );
    let rows = or_fail!(checked_len(rows), status, ());
    let (k, n) = (plan.plan.k(), plan.plan.n());
    let (la, lc) = (
        or_fail!(checked_mul(rows, k), status, ()),
        or_fail!(checked_mul(rows, n), status, ()),
    );
    // SAFETY: `checked_slice*` 校验非空与长度约定。
    let (a, c) = unsafe {
        (
            or_fail!(checked_slice(a, la), status, ()),
            or_fail!(checked_slice_mut(c, lc), status, ()),
        )
    };
    match plan.plan.run_block_into(a, c) {
        Ok(()) => LasxStatus::Ok.write(status),
        Err(e) => status_of(&e).write(status),
    }
}

/// 释放计划句柄（**重复释放 / 释放后再用 = UB**；`NULL` 是合法的空操作）。
///
/// C 签名：`void lasx_matmul_plan_destroy(lasx_matmul_plan *plan)`
///
/// 名字用 `_destroy` 而不是 `_free`：`free` 是 `docs/c-abi.md` §9.1 保留给**库级**符号的词根
/// （白名单里已有 `lasx_free`），算子级名字用它会被 `scripts/namecheck.sh` 拦下。
#[unsafe(no_mangle)]
pub extern "C" fn lasx_matmul_plan_destroy(plan: *mut LasxMatmulPlan) {
    if plan.is_null() {
        return;
    }
    // SAFETY: `plan` 来自 `Box::into_raw(lasx_matmul_prepare)`，且按契约只释放一次。
    drop(unsafe { Box::from_raw(plan) });
}

/// **便利层**：库内派发，用计划算整块 `C[m×n] = A[m×k]·B`（裸入口，不校验）。
///
/// C 签名：`void lasx_matmul_run_threaded(const lasx_matmul_plan *plan, int m,
/// const float *a, float *c, int threads)`
///
/// `threads`：`<= 0` = 自动（物理核）、`1` = 串行（**不建线程**）、`N` = N 个 worker
/// （超过物理核夹到物理核）。分块用池的**动态取块**（大 `m` 时不是固定等分，避免长尾）。
///
/// 池是**每次调用现建现拆**的：热循环里请改用 `lasx_matmul_run` + 调用方自己的线程/池
/// （见 `docs/c-abi.md` 的线程模型）；想确认这次有没有真的并行起来，用
/// [`lasx_matmul_run_threaded_checked`] 的 `threads_out`/`chunks_out`/`ms_max_out`/`ms_min_out`。
#[unsafe(no_mangle)]
pub extern "C" fn lasx_matmul_run_threaded(
    plan: *const LasxMatmulPlan,
    m: i32,
    a: *const f32,
    c: *mut f32,
    threads: i32,
) {
    let _ = run_threaded(plan, m, a, c, threads, std::ptr::null_mut(), None);
}

/// 带错误通道 + **执行自检**出参的 `lasx_matmul_run_threaded`。
///
/// C 签名：`void lasx_matmul_run_threaded_checked(const lasx_matmul_plan *plan, int m,
/// const float *a, float *c, int threads, int *threads_out, int *chunks_out,
/// double *ms_max_out, double *ms_min_out, int *status)`
///
/// 四个出参**都可以传 `NULL`**（不关心就不写）：
/// - `threads_out`：这次实际用的 worker 数（`1` = 串行）——**用来确认没有静默退化成串行**；
/// - `chunks_out`：实际执行了几块（动态取块的结果）；
/// - `ms_max_out` / `ms_min_out`：最慢/最快一块的耗时（毫秒）——`ms_max/ms_min` 接近 1
///   且 `threads_out > 1` 时，说明块负载均匀；差距大说明长尾（或机器被抢）。
///
/// 校验顺序同 [`lasx_matmul_run_checked`]，另加 `threads` 无要求（负值 = 自动）。
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn lasx_matmul_run_threaded_checked(
    plan: *const LasxMatmulPlan,
    m: i32,
    a: *const f32,
    c: *mut f32,
    threads: i32,
    threads_out: *mut i32,
    chunks_out: *mut i32,
    ms_max_out: *mut f64,
    ms_min_out: *mut f64,
    status: *mut i32,
) {
    let stats = or_fail!(
        run_threaded(plan, m, a, c, threads, status, Some(threads_out)),
        status,
        ()
    );
    write_i32(chunks_out, stats.chunks as i32);
    write_f64(ms_max_out, stats.ns_max as f64 / 1e6);
    write_f64(ms_min_out, stats.ns_min as f64 / 1e6);
}

/// `lasx_matmul_run_threaded` / `_checked` 的公共实现。
///
/// 校验判据与 `_checked` 版完全一致；裸入口只是**不写状态**（`status == NULL`）。
/// `threads_out != NULL` 时写实际 worker 数。
#[allow(clippy::too_many_arguments)]
fn run_threaded(
    plan: *const LasxMatmulPlan,
    m: i32,
    a: *const f32,
    c: *mut f32,
    threads: i32,
    status: *mut i32,
    threads_out: Option<*mut i32>,
) -> Result<crate::parallel::RunStats, LasxStatus> {
    /// 校验失败：写状态（`status` 可为 `NULL`，由 `LasxStatus::write` 处理）并提前返回。
    macro_rules! chk {
        ($expr:expr) => {
            match $expr {
                Ok(v) => v,
                Err(e) => {
                    e.write(status);
                    return Err(e);
                }
            }
        };
    }
    // SAFETY: 只读借用；句柄有效性由调用方保证（destroy 之后再用是 UB）。
    let plan_ref = chk!(unsafe { plan.as_ref() }.ok_or(LasxStatus::NullPointer));
    let m = chk!(checked_len(m));
    let (k, n) = (plan_ref.plan.k(), plan_ref.plan.n());
    let (la, lc) = (chk!(checked_mul(m, k)), chk!(checked_mul(m, n)));
    // SAFETY: `checked_slice*` 校验非空与长度约定。
    let (a, c) = unsafe { (chk!(checked_slice(a, la)), chk!(checked_slice_mut(c, lc))) };

    let nt = resolve_threads(threads);
    if let Some(out) = threads_out {
        write_i32(out, nt as i32);
    }
    if nt == 1 {
        // 串行档：**不建池、不建线程**（`threads=1` = 纯串行，与 `run` 等价 + 统计量）
        let t0 = std::time::Instant::now();
        if let Err(e) = plan_ref.plan.run_block_into(a, c) {
            let st = status_of(&e);
            st.write(status);
            return Err(st);
        }
        LasxStatus::Ok.write(status);
        let dt = u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX);
        return Ok(crate::parallel::RunStats {
            chunks: if m == 0 || n == 0 { 0 } else { 1 },
            ns_max: dt,
            ns_min: dt,
        });
    }
    // 便利层：池**现建现拆**（不用它就零进程级状态）；分块交给池的动态取块。
    let pool = crate::pool::WorkerPool::new(nt);
    match crate::parallel::matmul_f32_plan_with_stats(&pool, &plan_ref.plan, m, a, c) {
        Ok(stats) => {
            LasxStatus::Ok.write(status);
            Ok(stats)
        }
        Err(e) => {
            let st = status_of(&e);
            st.write(status);
            Err(st)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::testutil::Lcg;

    /// 造一份确定性数据（与别的 FFI 测试同一套 LCG 风格）。
    fn data(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = Lcg(seed);
        (0..n).map(|_| rng.f64() as f32).collect()
    }

    /// 参考实现：单线程整块 `lasx_matmul`，返回逐位可比的 u32 位模式。
    fn ref_bits(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<u32> {
        let mut c = vec![0f32; m * n];
        crate::ops::matmul::matmul_f32(m, k, n, a, b, &mut c);
        c.iter().map(|v| v.to_bits()).collect()
    }

    fn bits(c: &[f32]) -> Vec<u32> {
        c.iter().map(|v| v.to_bits()).collect()
    }

    /// 形状覆盖：k=384/385（打包路径 k 分块边界两侧）、`m` 不整除块大小、
    /// `rows` 不是 4 的倍数、`n` 不是 32 的倍数（列尾副本那条路）。
    const SHAPES: &[(usize, usize, usize, usize)] = &[
        (40, 384, 257, 7),
        (40, 385, 257, 7),
        (100, 384, 96, 13),
        (13, 769, 384, 5),
        (64, 64, 64, 64),
    ];

    #[test]
    fn run_and_threaded_are_bit_exact_with_matmul() {
        for &(m, k, n, block) in SHAPES {
            let a = data(m * k, 0x9e37_79b9);
            let b = data(k * n, 0x85eb_ca6b);
            let want = ref_bits(m, k, n, &a, &b);

            let mut st = -1;
            let plan = lasx_matmul_prepare_checked(k as i32, n as i32, b.as_ptr(), &mut st);
            assert!(
                !plan.is_null() && st == LasxStatus::Ok as i32,
                "prepare({m},{k},{n})"
            );

            // ① 调用方按行切块（块大小故意不整除）
            let mut c = vec![0f32; m * n];
            let mut r0 = 0;
            while r0 < m {
                let rows = block.min(m - r0);
                lasx_matmul_run_checked(
                    plan,
                    rows as i32,
                    a[r0 * k..].as_ptr(),
                    c[r0 * n..].as_mut_ptr(),
                    &mut st,
                );
                assert_eq!(st, LasxStatus::Ok as i32);
                r0 += rows;
            }
            assert_eq!(
                bits(&c),
                want,
                "按行切块不一致（m={m} k={k} n={n} block={block}）"
            );

            // ② 便利层 threads=1（串行、不建线程）
            let mut c1 = vec![0f32; m * n];
            let (mut used, mut chunks) = (-1i32, -1i32);
            let (mut mx, mut mn) = (-1.0f64, -1.0f64);
            lasx_matmul_run_threaded_checked(
                plan,
                m as i32,
                a.as_ptr(),
                c1.as_mut_ptr(),
                1,
                &mut used,
                &mut chunks,
                &mut mx,
                &mut mn,
                &mut st,
            );
            assert_eq!(st, LasxStatus::Ok as i32);
            assert_eq!(used, 1, "threads=1 应报告 1 个 worker");
            assert_eq!(bits(&c1), want, "threads=1 不一致");

            // ③ 便利层 threads=0（自动 = 物理核）：结果仍须逐位一致，且如实报告 worker 数
            let mut c2 = vec![0f32; m * n];
            lasx_matmul_run_threaded_checked(
                plan,
                m as i32,
                a.as_ptr(),
                c2.as_mut_ptr(),
                0,
                &mut used,
                &mut chunks,
                &mut mx,
                &mut mn,
                &mut st,
            );
            assert_eq!(st, LasxStatus::Ok as i32);
            assert!(used >= 1 && chunks >= 1, "used={used} chunks={chunks}");
            assert!(mx >= mn, "ms_max 应 >= ms_min");
            assert_eq!(bits(&c2), want, "threads=0 不一致");

            // ④ 句柄生命周期：NULL 空操作 + 正常释放
            lasx_matmul_plan_destroy(std::ptr::null_mut());
            lasx_matmul_plan_destroy(plan);
        }
    }

    /// 校验路径：状态码与"失败不触碰输出"。
    #[test]
    fn checked_paths_report_statuses() {
        let (k, n) = (64usize, 96usize);
        let b = data(k * n, 7);
        let a = data(4 * k, 8);
        let mut c = vec![1.5f32; 4 * n];
        let mut st = -999;

        // prepare_checked：k 为负、b 为空指针
        assert!(lasx_matmul_prepare_checked(-1, n as i32, b.as_ptr(), &mut st).is_null());
        assert_eq!(st, LasxStatus::NegativeLength as i32);
        assert!(
            lasx_matmul_prepare_checked(k as i32, n as i32, std::ptr::null(), &mut st).is_null()
        );
        assert_eq!(st, LasxStatus::NullPointer as i32);
        // status 传 NULL 也不崩（裸入口语义）
        assert!(
            lasx_matmul_prepare_checked(-1, n as i32, b.as_ptr(), std::ptr::null_mut()).is_null()
        );

        let plan = lasx_matmul_prepare_checked(k as i32, n as i32, b.as_ptr(), &mut st);
        assert!(!plan.is_null());

        // run_checked：plan 空 / rows 负 / a 空；失败时 c 原样
        lasx_matmul_run_checked(std::ptr::null(), 4, a.as_ptr(), c.as_mut_ptr(), &mut st);
        assert_eq!(st, LasxStatus::NullPointer as i32);
        lasx_matmul_run_checked(plan, -1, a.as_ptr(), c.as_mut_ptr(), &mut st);
        assert_eq!(st, LasxStatus::NegativeLength as i32);
        lasx_matmul_run_checked(plan, 4, std::ptr::null(), c.as_mut_ptr(), &mut st);
        assert_eq!(st, LasxStatus::NullPointer as i32);
        lasx_matmul_run_checked(plan, 4, a.as_ptr(), std::ptr::null_mut(), &mut st);
        assert_eq!(st, LasxStatus::NullPointer as i32);
        assert!(c.iter().all(|&v| v == 1.5), "校验失败时不得触碰输出");
        lasx_matmul_plan_destroy(plan);
    }

    /// **不用便利层时零进程级状态**：只走 `prepare` + `run` 不得**建池**。
    ///
    /// 这是"不引入全局池"的可执行证据（`docs/c-abi.md` 第 11.4 节）。
    ///
    /// 证据口径：**本线程的建池次数**（线程本地计数器），不是进程级线程数——`cargo test`
    /// 默认并行，别的用例（`api::matmul` 会初始化进程级全局池）会把 `/proc/self/task`
    /// 抬高 ⇒ 那种写法在并行框架下**必然 flaky**（CI run 1792063289 就是这么红的）。
    /// 线程本地的"建池次数"恰好就是契约要证明的事：`prepare`/`run` 的代码路径里
    /// 一次都不该碰 `WorkerPool::new`。
    #[test]
    fn plan_path_does_not_construct_a_pool() {
        let (m, k, n) = (64usize, 128usize, 96usize);
        let a = data(m * k, 11);
        let b = data(k * n, 12);
        let before = crate::pool::pool_ctors_this_thread();

        let plan = lasx_matmul_prepare(k as i32, n as i32, b.as_ptr());
        assert!(!plan.is_null());
        let mut c = vec![0f32; m * n];
        // 按行切块多调几次（推荐用法）：每次都不该建池
        for r0 in (0..m).step_by(7) {
            let rows = 7.min(m - r0);
            lasx_matmul_run(
                plan,
                rows as i32,
                a[r0 * k..].as_ptr(),
                c[r0 * n..].as_mut_ptr(),
            );
        }
        lasx_matmul_plan_destroy(plan);
        assert_eq!(
            crate::pool::pool_ctors_this_thread(),
            before,
            "prepare/run 不得走建池路径（契约：不用便利层 = 零进程级状态）"
        );

        // **正对照**：便利层 `threads=2` 必须建池。没有这一步，上面那条断言可能因为
        //"计数器根本没被触发"而**假绿**——这是结构性断言最容易犯的错。
        let plan = lasx_matmul_prepare(k as i32, n as i32, b.as_ptr());
        assert!(!plan.is_null());
        let mut st = -1;
        lasx_matmul_run_threaded_checked(
            plan,
            m as i32,
            a.as_ptr(),
            c.as_mut_ptr(),
            2,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut st,
        );
        assert_eq!(st, LasxStatus::Ok as i32);
        assert!(
            crate::pool::pool_ctors_this_thread() > before,
            "正对照失败：便利层 threads=2 应当建池（计数器没动 ⇒ 上面那条断言是假绿）"
        );
        lasx_matmul_plan_destroy(plan);
    }
}
