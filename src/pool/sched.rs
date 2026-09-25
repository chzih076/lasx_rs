//! 池的**调度策略**：把"n 行怎么分给 T 个 worker"从派活内核里拆出来。
//!
//! # 为什么要有这一层
//!
//! 原来 `dispatch_rows` 把"第 w 个 worker 拿 `[w·rows_per, …)`"写死了。但"怎么切"其实是
//! 与负载相关的决策：均匀内核要**连续等分**（每线程一份，缓存友好），矩阵乘要**对齐行粒度**
//! （与 4 行微块的边界对齐），而负载不均或机器有后台抢占时要**多块 + 动态领取**。
//! 把策略独立出来的收益有三条：
//!
//! 1. **编译期确认路径**：策略是零尺寸类型（[`Chunk`]、[`RowBlock`]、[`Blocked`]），
//!    池的入口对 `S: Strategy` 泛型化 ⇒ 每种策略各自单态化，派活路径上没有运行时分支。
//! 2. **新增策略时编译器会指着所有分派点**：[`Pick`] 是穷尽枚举，`match` 少一个分支就
//!    编译不过；不会出现"新策略悄悄走了旧路径"这种静默回退。
//! 3. **纯函数、可单测**：[`Strategy::plan`] 只做算术，不碰线程。测试断言"每个作业
//!    恰好被覆盖一次、块不重叠、块长满足粒度"，不必起池。
//!
//! # 编码一种新策略
//!
//! ```ignore
//! pub struct MyStrategy;
//! impl Strategy for MyStrategy {
//!     const NAME: &'static str = "my";
//!     fn plan(rows: usize, threads: usize, gran: usize) -> Jobs {
//!         // 只允许读这三个参数；返回的作业表必须恰好覆盖 [0, rows)
//!         todo!()
//!     }
//! }
//! // 然后在 Pick 里加一个变体 —— 编译器会指着所有需要补分支的地方。
//! ```

/// 一个作业块：从 `start` 行开始、`rows` 行。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Job {
    /// 起始行。
    pub start: usize,
    /// 行数（≥ 1，除非总行数为 0）。
    pub rows: usize,
}

/// 一次派活的作业表（行块按执行顺序排列）。
pub type Jobs = Vec<Job>;

/// 静态调度策略：一次算出整张作业表，worker `w` 领第 `w` 块（块数 ≤ 线程数）。
///
/// 之所以叫"静态"：作业表在派活前就定好，worker 之间没有竞争、没有原子操作。
pub trait Strategy {
    /// 诊断/文档用的名字。
    const NAME: &'static str;

    /// 是否允许"块数 > 线程数"（块按 `i % threads` 轮转领取）。
    ///
    /// `false` 的策略是"每 worker 恰好一块"的静态等分；`true` 的是多块策略
    /// （[`Blocked`]），块数由分块粒度决定。派活内核按这个常量决定是否需要轮转。
    const MULTI: bool = false;

    /// 把 `rows` 行按 `threads` 个 worker 切块，块长对齐到 `gran` 的倍数。
    ///
    /// 约定（[`crate::pool`] 的派活内核依赖它）：
    /// - 返回的块**按顺序恰好覆盖** `[0, rows)`，不重叠、不留空；
    /// - 块数 ≤ `threads`（`MULTI = false` 时；`MULTI = true` 的策略可以更多，
    ///   由派活内核按 `i % threads` 轮转分配）；
    /// - 每块行数是 `gran` 的倍数（最后一块可以是 `rows` 的余数），除非 `gran == 0/1`。
    fn plan(rows: usize, threads: usize, gran: usize) -> Jobs;
}

/// 等分（现状）：每线程一段连续行，块长 `ceil(rows/threads)`，最后一块接住余数。
///
/// 适合**每行代价相同**的内核（`sum`/`dot`/`axpy`、SOA 批量物理内核）：连续区间对缓存最友好。
pub struct Chunk;

impl Strategy for Chunk {
    const NAME: &'static str = "chunk";
    fn plan(rows: usize, threads: usize, gran: usize) -> Jobs {
        let per = rows.div_ceil(threads).max(1);
        // 对齐到 gran 的倍数：块数只会变少，且块边界落在微块边界上（不留半块）
        let per = if gran > 1 {
            per.next_multiple_of(gran)
        } else {
            per
        };
        let mut jobs = Jobs::new();
        let mut start = 0;
        while start < rows {
            let r = per.min(rows - start);
            jobs.push(Job { start, rows: r });
            start += r;
        }
        jobs
    }
}

/// 行块（矩阵乘的现状）：与 [`Chunk`] 同形，但**必须**对齐行粒度。
///
/// 单独列一个类型而不是给 `Chunk` 传参，是为了让调用点一眼看出"这里依赖行粒度"：
/// 矩阵乘按 4 行微块复用 B，块边界对齐到 4 行更整齐。**注意"`row_gran=1` 慢一倍"那条旧理由
/// 已被 2026-09-25 的复测推翻**（gran=1 从没更慢，256³ 上还快 8%；见 `docs/dev.md` §14）。
pub struct RowBlock;

impl Strategy for RowBlock {
    const NAME: &'static str = "row-block";
    fn plan(rows: usize, threads: usize, gran: usize) -> Jobs {
        // gran 至少 1；RowBlock 的语义就是"一定要对齐"，所以这里不接受 gran = 0
        Chunk::plan(rows, threads, gran.max(1))
    }
}

/// 固定块长：每块 `BLOCK` 行（末块接余数），按顺序分给各 worker 轮转。
///
/// 块数可以**多于**线程数（`Blocked<8>` 在 1024 行 × 24 线程下是 128 块），于是：
/// - 每线程领到的块数不完全相等 ⇒ 对"每块代价略有差异"的负载更稳；
/// - 仍然是静态分配，没有原子竞争。
///
/// `BLOCK` 是 const 泛型 ⇒ 块长参与单态化，循环边界是编译期常量。
pub struct Blocked<const BLOCK: usize>;

impl<const BLOCK: usize> Strategy for Blocked<BLOCK> {
    const NAME: &'static str = "blocked";
    const MULTI: bool = true;
    fn plan(rows: usize, _threads: usize, gran: usize) -> Jobs {
        // 块长对齐到粒度（矩阵乘的行粒度是 4），再按固定块长切
        plan_blocked(rows, BLOCK.max(gran.max(1)), gran)
    }
}

/// 按**运行时**块长切固定块（[`Blocked`] 的非常量版本）。
///
/// 块长对齐到 `gran` 的倍数：`gran > 1` 的内核（矩阵乘按 4 行复用 B）不能让某块
/// 只剩 1–3 行，否则那块会退化。最后一块允许不足块长（它接住余数）。
pub fn plan_blocked(rows: usize, block_rows: usize, gran: usize) -> Jobs {
    let gran = gran.max(1);
    let block = block_rows.max(1).next_multiple_of(gran).max(gran);
    let mut jobs = Jobs::new();
    let mut start = 0;
    while start < rows {
        let r = block.min(rows - start);
        jobs.push(Job { start, rows: r });
        start += r;
    }
    jobs
}

/// 运行期的调度决策：调用方按形状**穷尽匹配**一次，之后走各自的单态化实现。
///
/// 这就是"编译期确认路径"的落点：策略实现是泛型的（单态化、无分支），而"选哪个"是
/// 一个显式枚举 + 穷尽 `match`。加新策略时，所有 `match Pick` 的地方都会编译失败，
/// 逼着作者去每一处想清楚该不该用它 —— 而不是悄悄走默认分支。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    /// 连续等分（均匀内核）。
    Chunk,
    /// 对齐行粒度的连续等分（矩阵乘）。
    RowBlock,
    /// 固定块长的多块静态分配（块数 > 线程数，抗轻度负载不均）。
    Blocked {
        /// 每块行数。
        block_rows: usize,
    },
    /// 动态领取：块表固定，worker 用原子计数器抢块（抗重度负载不均/后台抢占）。
    Dynamic {
        /// 每块行数。
        block_rows: usize,
    },
}

impl Pick {
    /// 名字（诊断与基准输出用）。
    pub fn name(self) -> &'static str {
        match self {
            Pick::Chunk => "chunk",
            Pick::RowBlock => "row-block",
            Pick::Blocked { .. } => "blocked",
            Pick::Dynamic { .. } => "dynamic",
        }
    }
}

/// 按"行数 / 线程数 / 行粒度"挑一个策略。
///
/// 判据（都在本机实测过，见 `docs/dev.md` §14）：
///
/// - **块数太少就换多块静态**：静态等分时块数 = 线程数，一旦某块遇到 SMT 争抢或后台抢占，
///   整轮就等它。块数 ≥ 2×线程数（[`Blocked`]）能吸收这种抖动，代价只是每线程多几次
///   循环边界判断。
/// - **行粒度 ≥ 4 的内核（矩阵乘）**：块必须对齐粒度，否则最后一个退化块把整块拖慢。
/// - **块数 ≥ 8×线程数**：块足够碎，[`Pick::Dynamic`] 的原子领取（每次一块一次
///   `fetch_add`）才摊得薄，同时把负载不均压到最小。
/// - **`threads == 1` 或行数很少**：不切。
pub fn pick_rows(rows: usize, threads: usize, gran: usize) -> Pick {
    if threads <= 1 || rows <= 1 {
        return Pick::RowBlock;
    }
    let per_thread = rows.div_ceil(threads).max(1);
    // 动态/多块策略的块长：取"每线程工作量的 1/4" ⇒ 块数 ≈ 4×线程数。
    // 块太粗（= per_thread）等于静态等分，抢不到均衡；太细则每块一次原子操作不划算。
    let quarter = |per: usize, g: usize| (per / 4).max(g).next_multiple_of(g.max(1));
    if gran >= 4 {
        // 行结构内核（矩阵乘）：块需对齐行粒度，让块边界落在微块边界上。
        //
        // 静态 vs 动态是**实测**定的（本机 12 物理核 / 24 逻辑核，`docs/dev.md` §14）：
        //   1024³ 24 线程：静态 345 → 动态 426 GFLOP/s（+23%）
        //   1024³ 16 线程：静态 418 → 动态 375（−11%，块切碎后每块的面板复用变差）
        //   512³  24 线程：静态 275 → 动态 255（−8%）
        // 即：**线程数超过物理核（SMT 兄弟互相抢 L1/L2）且每线程仍有 ≥32 行**时，
        // 块间方差才大到值得用原子领取；否则一个大块留在缓存里更好。
        if threads >= 20 && per_thread >= 32 {
            return Pick::Dynamic {
                block_rows: quarter(per_thread, gran),
            };
        }
        return Pick::RowBlock;
    }
    if per_thread <= 2 {
        return Pick::Chunk;
    }
    if per_thread >= 64 {
        return Pick::Dynamic {
            block_rows: quarter(per_thread, 4),
        };
    }
    Pick::Blocked {
        block_rows: quarter(per_thread, 4),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 每个策略都必须**恰好覆盖** `[0, rows)`：不重叠、不留空、块长满足粒度。
    fn check<S: Strategy>(rows: usize, threads: usize, gran: usize) {
        let jobs = S::plan(rows, threads, gran);
        let mut covered = vec![0usize; rows];
        for j in &jobs {
            assert!(j.rows > 0, "{}: 出现空块 {j:?}", S::NAME);
            assert!(j.start + j.rows <= rows, "{}: 越界 {j:?}", S::NAME);
            if gran > 1 && j.start + j.rows < rows {
                assert_eq!(j.rows % gran, 0, "{}: 非末尾块未对齐粒度 {j:?}", S::NAME);
            }
            for slot in covered[j.start..j.start + j.rows].iter_mut() {
                *slot += 1;
            }
        }
        assert!(
            covered.iter().all(|&c| c == 1),
            "{}: 覆盖不恰好一次",
            S::NAME
        );
        if rows > 0 && !S::MULTI {
            assert!(jobs.len() <= threads.max(1), "{}: 块数超过线程数", S::NAME);
        }
        // （多块策略在小 `rows` 下也可能只切出一块 —— 那是块长决定的，不是不变式）
    }

    #[test]
    fn all_strategies_cover_exactly_once() {
        for rows in [
            0usize, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 255, 256, 1000, 1024, 4097,
        ] {
            for threads in [1usize, 2, 3, 7, 12, 24] {
                for gran in [1usize, 2, 4, 8] {
                    check::<Chunk>(rows, threads, gran);
                    check::<RowBlock>(rows, threads, gran);
                    check::<Blocked<8>>(rows, threads, gran);
                    check::<Blocked<64>>(rows, threads, gran);
                }
            }
        }
    }

    #[test]
    fn pick_is_sane() {
        assert_eq!(pick_rows(1024, 1, 4), Pick::RowBlock);
        // 实测收敛出的两条判据（见函数注释）
        assert_eq!(pick_rows(1024, 24, 4), Pick::Dynamic { block_rows: 12 });
        assert_eq!(pick_rows(1024, 16, 4), Pick::RowBlock); // 16 线程静态更好
        assert_eq!(pick_rows(512, 24, 4), Pick::RowBlock); // 每线程 22 行，太碎
        assert_eq!(pick_rows(48, 16, 4), Pick::RowBlock);
        assert_eq!(pick_rows(1 << 20, 12, 1).name(), "dynamic");
    }
}
