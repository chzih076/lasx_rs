//! `lasx_matmul` —— f32 矩阵乘 `C[m×n] = A[m×k]·B[k×n]`（行主序，列方向向量化）。
//!
//! 微内核是 i-k-j 形式：A 的行元素广播、B 沿列方向连续流式读取，**不做 B 转置、
//! 不做跨 lane 水平归约**（这两点是 v1 慢于朴素三重循环的原因）。
//!
//! v3 叠加**行分块**：一次算 4 行 × 32 列（16 个 8 通道累加器），同一段 B 被 4
//! 行复用，B 的重复读取量降到 1/4。
//!
//! v4 试过两级 L2 分块与 6 行微块，**都被实测否掉了**：4×32 已经是这台机器上
//! 16 个累加器能做到的最优形状。
//!
//! v6 = **k 分块**：打包条带按 k 主序存放，所以一段 k 就是一段连续内存。把 k 切成
//! `K_CHUNK` 步的块、**k 块在外层、行块在内层**，就能让每个 k 块的 B（32 KB）被所有
//! 行块反复命中，并与 A 的 4 行、C 的 4×32 一起待在 L1（64 KB）里。实测收益：
//! 512³ +13~26%、1024³ +7~26%、2048³ +91%、256³ 持平
//! （k=256 的条带本来就装得进 L1）。
//! 代价是 C 每块读写一次（k=1024 时多 3 次），相对算力可忽略。
//!
//! 数值行为不变：每个输出元素仍是**沿 k 的单个 f32 累加器**，累加次序与 v2 完全一致
//! （k 分块只是把同一个 f32 部分和分段落到 C 再读回，落盘/读回是精确的）。
//!
//! **LASX-only**：没有降级分支，无 LASX 的 CPU 上会执行 LASX 指令（见手册 Caveats）。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（分层策略：内核模块在函数级统一写 SAFETY，不逐块重复）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX/LSX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]

use crate::arch::lasx;

/// 列块宽度：B 的面板 `k × COL_BLOCK` 要能留在 L2（3 MiB/核）里被整轮行扫描复用。
/// 取值来自 `COL_BLOCK` 的扫描：128 最优，32/64 接近，256 起明显变慢。
pub(crate) const COL_BLOCK: usize = 128;

/// 走"列块在外"次序的 `m` 上限。`m/4` 是流式次序下 B 被读的遍数：m=128 时 32 遍，
/// 已经把 B 的读量放大 32 倍；再大一档（方阵）实测两种次序持平，故保守取 128。
const COL_ORDER_MAX_M: usize = 128;

/// 走"列块在外"次序的 `n` 下限：n 太小时 A 的重读代价相对更高，不值得切换。
const COL_ORDER_MIN_N: usize = 1024;

/// 打包路径的 `m` 下限：打包本身要走两遍 B（读+写），只有当行块足够多
/// （流式次序会因此把 B 读很多遍）时才摊得掉。按行切并行的**每线程**只有 m/线程数 行，
/// 所以这个下限同时保护了并行路径——否则每个线程都会把整块 B 打包一遍。
pub(crate) const PACK_MIN_M: usize = 32;

/// 打包路径的 `k` 下限。重标定过：配合 k 分块，k=64 起打包
/// 就大胜"列块/流式"（`64×96×8192` 21 → 45 GFLOP/s、`256×64×4096` 23 → 58），
/// 所以从 192 降到 64。
pub(crate) const PACK_MIN_K: usize = 64;

/// 极宽 `n` 的例外：当 `k` 很小而 `n` 极大时，列块路径只复制一遍 B 面板、且面板只有
/// `k×128×4`，比打包（要整体复制一遍 B = `k×n×4` 字节）更划算。实测 `100×64×19147`
/// 列块 39.7 vs 打包 33.5 GFLOP/s；而 `n ≤ 16384` 时打包仍然领先（`64×96×8192`、
/// `128×128×2048` 分别是 +80%、+71%）。所以只在这个交叉区间保留列块路径。
const WIDE_N_FOR_COLS: usize = 16384;

/// `k` 大于等于这个值时，打包**无条件**胜出（与 `n` 多大无关）：这是"打包 `k` 下限"
/// 与"列块 `n` 下限"两条判据的交叉点。
const K_ALWAYS_PACK: usize = 192;

/// 打包路径的工作量下限（`m×k×n`，即乘加次数）：128³（4.2 MFLOP）打包 0.85×，
/// 192³（14 MFLOP）起转为正收益。
pub(crate) const PACK_MIN_WORK: u128 = 8_000_000;

/// k 方向分块的"L1 预算"与块长。打包条带按 k 主序存放，所以一个 k 块就是一段连续内存。
///
/// 动机（条带扫描，`512×k×512`）：条带 24–48 KB 时 59–61 GFLOP/s，正好等于 L1 的
/// 64 KB 时掉到 54.7，96–128 KB 掉到 47–49。k=1024 不分块时
/// 条带就是 128 KB。所以条带本身超过 `L1_BUDGET` 才分块，块长取 `K_CHUNK`（32 KB）；
/// k ≤ 384 时条带 ≤ 48 KB，直接一整块跑，避免白白多读写 C。
const L1_BUDGET: usize = 48 * 1024;
const K_CHUNK: usize = 256;

/// **路线 B 实验旋钮**（`task-73`）：把上面两个常量 + 三个次序/写策略变体参数化，
/// 生产路径默认用 `KernelCfg::default()`（= 与本轮改动前**逐条指令相同**），
/// 仅在 `#[cfg(test)]` 的 perf 探针里用别的组合做 A/B；采纳时只改 `default()` 的字段值。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct KernelCfg {
    /// k 块长（条带 `k_chunk×32×4` 字节）。
    pub k_chunk: usize,
    /// 条带 ≤ 该字节数时不分块。
    pub l1_budget: usize,
    /// 循环次序：`false` = `k块 → strip → 行块`（现状）；`true` = `strip → k块 → 行块`。
    pub strip_outer: bool,
    /// C 写策略：`true` = 每个 strip 先在**独立缓冲**里累完全部 k 块，最后写 C 一次
    /// （消除每 k 块一次 C 读+写；缓冲逐位精确，累加次序不变）。
    pub c_strip_buf: bool,
    /// 对下一个 strip 的 B 做 `preld` 预取（只在标量循环边界处发 2 条预取，不改变数值）。
    pub prefetch: bool,
    /// 打包面板的 L2 预算（字节）：`nc_strips = panel_bytes / (k×32×4)`。
    pub panel_bytes: usize,
}

impl Default for KernelCfg {
    /// **生产默认**：`strip_outer = true`（条带主序）——`task-73` 实测 1t 内核态
    /// **1024³ +5.1%、1536³ +11.3%**（逐位不变；`k块→strip` 旧次序是现默认的对照）。
    ///
    /// 采纳依据：① 亲和性修复（`pool::auto` 尊重可用 CPU 数）后 1t C ABI 不再被 12 线程
    /// 派发污染，实测 1t 对内核对齐到 **≈0.98–0.99×** OB 纯内核；② 叠加本项后 1536³ 有望
    /// 越过 0.95× 目标；③ `legacy()`（并行路径）保持旧次序不变 ⇒ 多线程路径逐位/行为不变。
    fn default() -> Self {
        KernelCfg {
            k_chunk: K_CHUNK,
            l1_budget: L1_BUDGET,
            strip_outer: true,
            c_strip_buf: false,
            prefetch: false,
            panel_bytes: 2 * 1024 * 1024,
        }
    }
}

impl KernelCfg {
    /// **改动前行为**（并行路径固定用它，保证多线程路径的循环次序与本轮改动前完全一致）。
    pub(crate) fn legacy() -> Self {
        KernelCfg {
            k_chunk: K_CHUNK,
            l1_budget: L1_BUDGET,
            strip_outer: false,
            c_strip_buf: false,
            prefetch: false,
            panel_bytes: 2 * 1024 * 1024,
        }
    }
}
use std::arch::loongarch64::*;

/// 输出那一遍的缩放口径：`C = alpha·acc + beta·C_old`（`acc` 是沿 k 从 0 起的完整累加）。
///
/// 这是"融合"的全部内容——把缩放**融进最终的 store**，不再单独跑一遍 elementwise。
/// 三种模式的差别只在收尾，k 循环与既有内核**逐条指令相同**：
///
/// | 模式 | 触发 | 收尾 | 读 `C_old` |
/// |---|---|---|---|
/// | `Plain` | `alpha == 1 && beta == 0` | `store(acc)` | 否（与既有内核逐字节相同） |
/// | `Scale` | `beta == 0` | `store(alpha·acc)` | **否**（CBLAS 语义：`beta = 0` 时 `C` 无需初始化） |
/// | `Fma` | `beta != 0` | `store(fma(beta, C_old, alpha·acc))` | 是 |
///
/// `Fma` 与参考实现 `beta.mul_add(C_old, alpha * acc)` **逐位一致**（两次舍入：`alpha·acc`
/// 一次、FMA 一次）；`Scale` 与 `alpha * acc` 逐位一致（一次舍入）。列尾的标量段不是 FMA，
/// 所以口径按**表达式**定义（参考实现同样写成 `beta.mul_add(old, alpha * acc)`），不按
/// "数学值"定义；NaN 的 payload 不在逐位承诺内。
#[derive(Clone, Copy, PartialEq)]
pub enum Epi {
    /// `C = acc`：不读 `C`，store 与既有内核逐字节相同。
    Plain,
    /// `C = alpha·acc`：不读 `C`。
    Scale(f32),
    /// `C = fma(beta, C_old, alpha·acc)`：读 `C_old`。
    Fma(f32, f32),
}

impl Epi {
    /// 按 `(alpha, beta)` 选口径；`beta == 0` 一律不读 `C`。
    pub(crate) fn new(alpha: f32, beta: f32) -> Epi {
        if beta == 0.0 {
            if alpha == 1.0 {
                Epi::Plain
            } else {
                Epi::Scale(alpha)
            }
        } else {
            Epi::Fma(alpha, beta)
        }
    }
    /// 写回连续的 8 个 `f32`（`p` 指向该段起点）。
    ///
    /// # Safety
    /// `Epi::Fma` 要求 `p` 可读写 8 个 `f32`；另两种模式只要求可写 8 个 `f32`。
    #[inline]
    unsafe fn store8(self, p: *mut f32, acc: lasx::F32x8) {
        match self {
            // SAFETY: 调用方保证 `p` 可写 8 个 `f32`（见内核函数的 SAFETY 段）。
            Epi::Plain => unsafe { lasx::store_f32x8(p, acc) },
            // SAFETY: 同上，只写不读。
            Epi::Scale(a) => unsafe {
                lasx::store_f32x8(p, lasx_xvfmul_s(acc, lasx::splat_f32(a)))
            },
            // SAFETY: 调用方保证 `p` 可读写 8 个 `f32`。
            Epi::Fma(a, b) => unsafe {
                let old = lasx::load_f32x8(p);
                let out = lasx_xvfmadd_s(
                    old,
                    lasx::splat_f32(b),
                    lasx_xvfmul_s(acc, lasx::splat_f32(a)),
                );
                lasx::store_f32x8(p, out);
            },
        }
    }
    /// 标量列尾的同一口径。`old` 惰性取值：`Plain`/`Scale` 下**不读** `C`。
    #[inline]
    fn scalar<F: FnOnce() -> f32>(self, old: F, acc: f32) -> f32 {
        match self {
            Epi::Plain => acc,
            Epi::Scale(a) => a * acc,
            Epi::Fma(a, b) => b.mul_add(old(), a * acc),
        }
    }
}

/// f32 矩阵乘 `C[m×n] = A[m×k]·B[k×n]`（行主序）。
///
/// 循环顺序是"每个 4 行块把 B 扫一遍"——**这个顺序是实测选出来的**：
/// 把它改成两级 L2 分块（列块在外、行块在内）在多核下慢 8–12%，单线程也无收益，
/// 因为 4×32 微块本来就让 B 被复用 4 次，分块只是把流量从 L3 挪到 L2、体量不变。
///
/// **这条生产路径的判据与流式循环保持内联**（不抽成 `pick_f32_path` + `matmul_f32_stream`）：
/// 抽出去以后最小形状会慢——64³ 实测 7.6 → 7.9 µs（+3.9%），是"多一层调用 + 微内核拿不到
/// 同样的内联"造成的（基线 A/B 测到，改回内联后回到 7.6）。融合入口自己复制那十几行判据。
#[inline]
pub(crate) fn matmul_f32(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    // **自动并行默认开（§7.28；用户裁决 Q1=默认开，数值逐位不变）**。
    //
    // 判据：`work = m·k·n` 低于阈值 ⇒ 走原单线程主体（小形状并行只会更慢）；
    // 否则走**进程级全局池** `pool::global()`（`OnceLock`，默认 `WorkerPool::auto()` = 物理核）。
    // 线程数由 `pool::init_global(n)` 或环境决定；**`init_global(1)` 是逃生门**（强制单线程）。
    //
    // **逐位**：池版只按输出行切、**绝不 split-K** ⇒ 每个输出元素的累加顺序与单线程相同。
    // **scratch 说明（如实）**：池版入口要求 `a: &mut [f32]`（行内核就地整理），本轮用**内部
    // scratch** 消化这次 `&mut` ⇒ 调用方（含 DSL override）签名仍是 `&[f32]`、**不需要自己复制**。
    // 这次复制发生在后端内核内部，**不算 DSL 抽象税**（但真实成本存在，见 §7.28 的实测）。
    const PAR_MIN_WORK: u128 = 32 * 1024 * 1024; // 阈值：work ≥ 32M ≈ 320³（实测来源 `docs/dev.md` §7.28.2：320³ 中性、56.6M 起 4.11× ⇒ 32M 是安全下界）
    let work = (m as u128) * (k as u128) * (n as u128);
    let pool = crate::pool::global();
    if work >= PAR_MIN_WORK && pool.threads() > 1 {
        let mut a_scratch = a.to_vec();
        if crate::parallel::matmul_f32(pool, m, k, n, &mut a_scratch, b, c).is_ok() {
            return;
        }
    }
    matmul_f32_single(m, k, n, a, b, c)
}

/// **单线程主体**（原 `matmul_f32` 的实现，逐行未改）：按形状选 打包/列块/流式 三条路径。
#[inline]
fn matmul_f32_single(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    if m == 0 || n == 0 {
        return;
    }
    // 路径选择（三条都**逐位一致**，只是访存次序不同）：
    //   1) 打包 B 面板：让微内核的 k 循环顺序读。交叉点实测 work ≈ 8 MFLOP；k 下限
    //      在 k 分块落地后从 192 降到 64（见 `PACK_MIN_K`），
    //      只有"小 k + 极宽 n"例外仍走列块（见 `WIDE_N_FOR_COLS`）。
    //   2) 列块在外：k 小、m 小、n 大——打包摊不掉，但仍要让 B 只流一遍。
    //   3) 流式：其余（工作集本来就装得下缓存）。
    // 极宽 n + 小 k 时列块路径更划算（只复制一遍 B 面板），见 `WIDE_N_FOR_COLS`
    let cols_regime = m <= COL_ORDER_MAX_M && n >= COL_ORDER_MIN_N;
    let wide_n_small_k = cols_regime && k < K_ALWAYS_PACK && n > WIDE_N_FOR_COLS;
    if m >= PACK_MIN_M
        && k >= PACK_MIN_K
        && !wide_n_small_k
        && (m as u128) * (k as u128) * (n as u128) >= PACK_MIN_WORK
    {
        matmul_f32_packed(m, k, n, a, b, c, Epi::Plain);
        return;
    }
    if m <= COL_ORDER_MAX_M && n >= COL_ORDER_MIN_N {
        matmul_f32_cols_block(m, k, n, 0, n, a, b, c, COL_BLOCK, Epi::Plain);
        return;
    }

    let m4 = m / 4 * 4;
    let n32 = n / 32 * 32;

    let mut i = 0;
    while i < m4 {
        let mut j = 0;
        while j < n32 {
            tile4x32_f32(i, j, k, n, a, b, c);
            j += 32;
        }
        i += 4;
    }
    // 主体行的列尾 [n32, n)
    let mut i = 0;
    while i < m4 {
        for r in 0..4 {
            let a_row = &a[(i + r) * k..(i + r + 1) * k];
            let c_row = &mut c[(i + r) * n..(i + r + 1) * n];
            row_tail_f32(a_row, c_row, b, k, n, n32);
        }
        i += 4;
    }
    // 行尾：不足 4 行
    while i < m {
        let a_row = &a[i * k..(i + 1) * k];
        let c_row = &mut c[i * n..(i + 1) * n];
        // 第 4 个参数是 B 的**行跨距**（= 真实 n），不是列数上界
        cols32_f32(a_row, c_row, b, n);
        row_tail_f32(a_row, c_row, b, k, n, n32);
        i += 1;
    }
}

/// `C = alpha·(A·B) + beta·C`（原地）——把缩放**融进输出那一遍**。
///
/// 与 [`matmul_f32`] 的关系：`alpha = 1, beta = 0` 时逐位等于 [`matmul_f32`]（同一条内核、
/// 收尾 `Plain`，连一次乘法都不多）；其余情况下每个输出元素的收尾是
/// `C = fma(beta, C_old, alpha·acc)`，而 `acc` 与 [`matmul_f32`] 算出的值**逐位相同**
/// （k 循环一条指令都没动）。
///
/// **`beta == 0` 不读 `C`**（CBLAS 语义），此时 `c` 可以是未初始化内容的缓冲；`beta != 0`
/// 时读 `c` 的旧值。
///
/// **哪些形状真的融合**：融合的硬前提是"收尾时 `C_old` 还没被部分和覆盖"。打包路径在
/// `k > 384`（条带超过 `L1_BUDGET`）时会把部分和落进 `C` 再读回，此时 `beta != 0` 无法
/// 融合，退回二遍法（[`matmul_f32_scaled_two_pass`]，与朴素实现等价、不会更快）；`beta == 0`
/// 没有这个约束（根本不读 `C_old`），任何 `k` 都能融合。
///
/// # Panics
/// 不 panic：形状由调用方保证（`api` 层先做长度与溢出检查）。
#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_f32_scaled(
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    beta: f32,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    if m == 0 || n == 0 {
        return;
    }
    let epi = Epi::new(alpha, beta);
    if epi == Epi::Plain {
        matmul_f32(m, k, n, a, b, c);
        return;
    }
    // k 不分块时部分和不落 C，收尾读 C_old 才安全；`beta == 0` 时压根不读。
    let single_chunk = k * 32 * 4 <= L1_BUDGET;
    match pick_f32_path(m, k, n) {
        Path::Packed if single_chunk || beta == 0.0 => matmul_f32_packed(m, k, n, a, b, c, epi),
        Path::Packed => matmul_f32_scaled_two_pass(m, k, n, alpha, beta, a, b, c),
        Path::Cols => matmul_f32_cols_block(m, k, n, 0, n, a, b, c, COL_BLOCK, epi),
        // 流式档（三条判据都不满足的小形状）：融合收益在噪声里，直接走二遍法，
        // 让那条路径保持与改动前逐字节相同。见 `matmul_f32_stream` 的注释。
        Path::Stream => matmul_f32_scaled_two_pass(m, k, n, alpha, beta, a, b, c),
    }
}

/// `beta != 0` 且打包路径要 k 分块时的兜底：`acc = A·B` 算进临时缓冲，再
/// `C = fma(beta, C_old, alpha·acc)`。
///
/// 这就是"朴素二遍法"本身——融合在这里**不成立**：k 分块会把部分和落进输出缓冲再读回，
/// 若输出就是 `C`，`C_old` 会被部分和覆盖；而给累加器播种 `beta·C_old` 会改结合次序、
/// 破坏与参考实现逐位一致的口径。所以本函数**不假装更快**：它逐位等于参考实现，
/// 访存量也与调用方手写的两遍法相同（省下的是"融合"本身，这里没有）。
///
/// 临时缓冲走线程局部复用：每次调用新分配一个 `m×n` 的 `Vec` 会带来一次 `malloc` +
/// 首次触页（512³ 实测 +1.6%，正好是"退化成两遍法"之外多出来的那点），而这本该是零成本。
fn matmul_f32_scaled_two_pass(
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    beta: f32,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    SCALED_ACC.with(|buf| {
        let mut acc = buf.borrow_mut();
        acc.resize(m * n, 0.0);
        let acc = &mut acc[..m * n];
        matmul_f32(m, k, n, a, b, acc);
        // 索引走切片、不用 `AlignedVec` 的 `Deref` 索引，也不用 `zip`：这一遍是要与
        // 调用方手写的 `for i in 0..m*n { … }` 比时间的，写法差一点就是测量误差。
        let (c, acc) = (&mut c[..m * n], &acc[..m * n]);
        if beta == 0.0 {
            // `beta == 0` **不读 C**（CBLAS 语义）在所有路径上都要成立——写成
            // `0.0.mul_add(C_old[i], ·)` 会读 C，NaN 会从"不该读的旧值"传播进输出
            // （测试用 NaN 探针钉住这条）。
            for i in 0..m * n {
                c[i] = alpha * acc[i];
            }
        } else {
            for i in 0..m * n {
                c[i] = beta.mul_add(c[i], alpha * acc[i]);
            }
        }
    });
}

/// f32 的三条访存次序（都**逐位一致**，只是循环嵌套不同）。
#[derive(Clone, Copy, PartialEq, Eq)]
enum Path {
    Packed,
    Cols,
    Stream,
}

/// 路径选择（只看形状）：
///   1) 打包 B 面板：让微内核的 k 循环顺序读。交叉点实测 work ≈ 8 MFLOP；k 下限在 k 分块
///      落地后从 192 降到 64（见 `PACK_MIN_K`），只有"小 k + 极宽 n"例外仍走列块
///      （见 `WIDE_N_FOR_COLS`）。
///   2) 列块在外：k 小、m 小、n 大——打包摊不掉，但仍要让 B 只流一遍。
///   3) 流式：其余（工作集本来就装得下缓存）。
fn pick_f32_path(m: usize, k: usize, n: usize) -> Path {
    let cols_regime = m <= COL_ORDER_MAX_M && n >= COL_ORDER_MIN_N;
    let wide_n_small_k = cols_regime && k < K_ALWAYS_PACK && n > WIDE_N_FOR_COLS;
    if m >= PACK_MIN_M
        && k >= PACK_MIN_K
        && !wide_n_small_k
        && (m as u128) * (k as u128) * (n as u128) >= PACK_MIN_WORK
    {
        Path::Packed
    } else if cols_regime {
        Path::Cols
    } else {
        Path::Stream
    }
}

/// 列区间入口：只算 `C[:, j0..j1)`（`b`/`c` 的行跨距仍是完整的 `n`），列块宽度可调。
///
/// 与 [`matmul_f32`] **逐位一致**（每个输出元素的累加次序相同），区别只在循环嵌套：
/// 这里**列块在外、行扫在内**，于是 B 的一段列面板（`k × col_block`）在整轮行扫描期间
/// 留在 L1/L2 里被复用，**B 只从内存流一遍**。
///
/// 动机：`m` 小、`n` 大时（100×448 × 448×19147），[`matmul_f32`] 每个 4 行块都要把
/// B 扫一遍 ⇒ 100 行读 25 遍 = 858 MB，且每次只读 128 B 就跳 `n×4` 字节，实测有效带宽
/// 只有 ~5 GB/s。改成列块在外后同一形状 **165.9 ms → 42.7 ms（3.9×）**。
///
/// # 为什么要 32 对齐
/// 主内核把 `[0, n/32*32)` 交给 4×32 微内核、其余交给 8 宽 + 标量的尾路径；尾路径里的
/// 标量段用的是**普通乘加**（不是 FMA），所以只有 32 对齐的切分才能保证每一列都落在与
/// 主内核相同的路径上、结果逐位相同。
///
/// # Panics
/// `debug` 构建下要求 `j0 % 32 == 0` 且 `j1 <= n`。
///
/// `epi` 是输出那一遍的缩放口径（`Epi::Plain` 时与不带缩放逐位相同）；列块次序**不分块
/// k**，所以累加器一直在寄存器里，`Epi::Fma` 读 `C_old` 是安全的。
#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_f32_cols_block(
    m: usize,
    k: usize,
    n: usize,
    j0: usize,
    j1: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    col_block: usize,
    epi: Epi,
) {
    debug_assert_eq!(j0 % 32, 0, "列区间起点需 32 对齐（尾路径的标量段不是 FMA）");
    debug_assert!(j1 <= n && j1 >= j0);
    if m == 0 || j1 <= j0 {
        return;
    }
    let m4 = m / 4 * 4;
    let jb = j1 / 32 * 32; // 完整 32 列的上界；j1 不是 32 倍数时留出列尾

    let mut jc = j0;
    while jc < jb {
        let jce = (jc + col_block).min(jb);
        // 主体行：4 行 × 32 列微内核
        let mut i = 0;
        while i < m4 {
            let mut j = jc;
            while j < jce {
                // `Epi::Plain` 走**与改动前逐字节相同**的微内核（它是 64³ 那类小形状的热点，
                // 给它加运行时口径参数实测 +3.9%）；只有要缩放时才进 scaled 微内核。
                if let Epi::Plain = epi {
                    tile4x32_f32(i, j, k, n, a, b, c);
                } else {
                    tile4x32_f32_scaled(i, j, k, n, a, b, c, epi);
                }
                j += 32;
            }
            i += 4;
        }
        // 不足 4 行的尾部行
        while i < m {
            let a_row = &a[i * k..(i + 1) * k];
            let c_row = &mut c[i * n..(i + 1) * n];
            row_tail_range_f32(a_row, c_row, b, k, n, jc, jce, epi);
            i += 1;
        }
        jc = jce;
    }
    // 列尾 [max(jb, j0), j1)：与主内核同一条尾路径
    let from = jb.max(j0);
    if j1 > from {
        for i in 0..m {
            let a_row = &a[i * k..(i + 1) * k];
            let c_row = &mut c[i * n..(i + 1) * n];
            row_tail_range_f32(a_row, c_row, b, k, n, from, j1, epi);
        }
    }
}

// B 面板打包用的线程局部缓冲（跨调用复用，避免每次 malloc 1.8 MB）。
thread_local! {
    static PACK_BUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

// 融合退化成二遍法时用的 `A·B` 临时缓冲（`beta != 0` + k 分块；见
// `matmul_f32_scaled_two_pass`）。同样跨调用复用，否则每次调用都要付一次 malloc + 触页。
thread_local! {
    static SCALED_ACC: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// "打包 B + 宏内核"路径：小 m、大 n 时的主力。
///
/// Goto/BLIS 的做法是**打包**（packing）：把 B 的一段列面板复制成"k 方向连续"的布局，
/// 于是微内核的 k 循环是顺序读，而不是 [`matmul_f32`] 里那种"读 128 B 跳 `n×4`"。
/// 打包本身要走一遍 B（顺序读、128 B 连续写），但这**一遍**换来后面所有行扫描的
/// 顺序访问——B 被读 `m/4` 遍，省下的是 `m/4 - 1` 遍的跨距访问。
///
/// 布局：`packed[(s*k + p)*32 + r] = B[p][jb + s*32 + r]`（s 是 32 列条带号）。
/// 数值与 [`matmul_f32`] **逐位一致**（每个输出元素的 p 升序累加不变）。
///
/// 参考水位：同形状 OpenBLAS 0.3.34（LASX，la464）单线程 32.6 ms / 52.7 GFLOP/s。
///
/// `epi` 是输出那一遍的缩放口径（`Epi::Plain` 时与不带缩放逐位相同）。**只有最后一个 k 块**
/// 才应用 `epi`：中间的 k 块落盘的是部分和，必须原样写回，否则读回来继续累加就错了。
/// 于是 `beta != 0` 的原地融合只在 `k` 不分块时成立（`k` 分块时部分和会覆盖 `C_old`）。
#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_f32_packed(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    epi: Epi,
) {
    matmul_f32_packed_at(m, k, n, a, b, c, epi, KernelCfg::default());
}

/// 路线 B 的可配置整条乘法（生产包装 [`matmul_f32_packed`] 用默认配置）。
pub(crate) fn matmul_f32_packed_at(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    epi: Epi,
    cfg: KernelCfg,
) {
    if m == 0 || n == 0 {
        return;
    }
    let m4 = m / 4 * 4;
    let n32 = n / 32 * 32;
    let strips_total = n32 / 32;
    // 一次打包的列数：面板 `k×nc×4` 要留在 L2（3 MiB/核）里，预算默认取 2 MiB
    let nc_strips = (cfg.panel_bytes / (k * 32 * 4).max(1)).clamp(1, 256);

    PACK_BUF.with(|buf| {
        let mut packed = buf.borrow_mut();
        packed.resize(k * nc_strips * 32, 0.0);
        let mut s0 = 0;
        while s0 < strips_total {
            let s1 = (s0 + nc_strips).min(strips_total);
            let jb = s0 * 32;
            let nc = (s1 - s0) * 32;
            pack_b(k, n, jb, nc, b, &mut packed);
            // 面板打包好后，交给"用已打包面板算一个行块"的入口（并行层复用同一个入口，
            // 于是打包只做一次、面板跨线程共享）。
            matmul_f32_packed_rows_at(k, n, jb, nc, &packed, a, c, m4, epi, cfg);
            s0 = s1;
        }
    });
    // 列尾 [n32, n)：走原来的跨距尾路径（列数 < 32，代价可忽略）。这条尾路径按整体 k
    // 累加（不分块），所以收尾可以直接应用 `epi`。
    if n > n32 {
        for i in 0..m {
            let a_row = &a[i * k..(i + 1) * k];
            let c_row = &mut c[i * n..(i + 1) * n];
            row_tail_range_f32(a_row, c_row, b, k, n, n32, n, epi);
        }
    }
}

/// 用**已打包**的列面板计算 `a_rows`/`c_rows` 对应的那些行。
///
/// `packed` 是 [`pack_b`] 的产物（`k × nc`，32 列条带主序）；`a_rows.len()/k` 是本块行数。
/// 并行层（[`crate::parallel::matmul_f32`]）用它复用**同一份**打包面板：打包只做一遍，
/// 各线程只读共享面板。每行的列尾 `[n32, n)` 由 [`row_tail_range_f32`] 单独处理。
///
/// 顺序版的 [`matmul_f32_packed`] 走的是同一个函数（`m4` 之后的行尾由调用方补），
/// 所以并行与单线程**逐位一致**（每个输出元素沿 k 的升序累加不变）。
#[allow(clippy::too_many_arguments)]
pub fn matmul_f32_packed_rows(
    k: usize,
    n: usize,
    jb: usize,
    nc: usize,
    packed: &[f32],
    a_rows: &[f32],
    c_rows: &mut [f32],
    m4: usize,
    epi: Epi,
) {
    matmul_f32_packed_rows_at(
        k,
        n,
        jb,
        nc,
        packed,
        a_rows,
        c_rows,
        m4,
        epi,
        // **并行路径固定用改动前次序**（任务红线："多线程路径不变"）。
        KernelCfg::legacy(),
    );
}

/// 单 strip、单 k 块的行块主体（路线 B 的公共内核）：`use_buf` 为真时写到 `cbuf`（stride 32），
/// 否则直接写 `c_rows`（stride `n`、列偏移 `j`）。两者累加次序相同 ⇒ 逐位一致。
#[inline]
#[allow(clippy::too_many_arguments)]
fn run_strip_chunk(
    m4: usize,
    m: usize,
    n: usize,
    k: usize,
    j: usize,
    kb0: usize,
    kb: usize,
    strip: &[f32],
    a_rows: &[f32],
    first: bool,
    epi: Epi,
    c_rows: &mut [f32],
    cbuf: &mut [f32],
    use_buf: bool,
) {
    let mut i = 0;
    if use_buf {
        while i < m4 {
            let ct = &mut cbuf[i * 32..(i + 4) * 32];
            tile4x32_chunk(
                kb,
                first,
                strip,
                &a_rows[i * k + kb0..i * k + kb0 + kb],
                &a_rows[(i + 1) * k + kb0..(i + 1) * k + kb0 + kb],
                &a_rows[(i + 2) * k + kb0..(i + 2) * k + kb0 + kb],
                &a_rows[(i + 3) * k + kb0..(i + 3) * k + kb0 + kb],
                ct,
                32,
                0,
                epi,
            );
            i += 4;
        }
        while i < m {
            row_tail_packed(
                &a_rows[i * k + kb0..i * k + kb0 + kb],
                strip,
                &mut cbuf[i * 32..(i + 1) * 32],
                first,
                0,
                epi,
            );
            i += 1;
        }
    } else {
        while i < m4 {
            let ct = &mut c_rows[i * n..(i + 4) * n];
            tile4x32_chunk(
                kb,
                first,
                strip,
                &a_rows[i * k + kb0..i * k + kb0 + kb],
                &a_rows[(i + 1) * k + kb0..(i + 1) * k + kb0 + kb],
                &a_rows[(i + 2) * k + kb0..(i + 2) * k + kb0 + kb],
                &a_rows[(i + 3) * k + kb0..(i + 3) * k + kb0 + kb],
                ct,
                n,
                j,
                epi,
            );
            i += 4;
        }
        while i < m {
            row_tail_packed(
                &a_rows[i * k + kb0..i * k + kb0 + kb],
                strip,
                &mut c_rows[i * n..(i + 1) * n],
                first,
                j,
                epi,
            );
            i += 1;
        }
    }
}

/// 把每 strip 的 C 缓冲（`m×32`，已含完整 k 累加）按 `epi` 写到 C 的真实位置。
fn finish_cbuf(cbuf: &[f32], c_rows: &mut [f32], m: usize, n: usize, j: usize, epi: Epi) {
    for i in 0..m {
        let src = &cbuf[i * 32..i * 32 + 32];
        let dst = &mut c_rows[i * n + j..i * n + j + 32];
        match epi {
            Epi::Plain => dst.copy_from_slice(src),
            Epi::Scale(alpha) => {
                for (d, v) in dst.iter_mut().zip(src) {
                    *d = alpha * *v;
                }
            }
            Epi::Fma(alpha, beta) => {
                for (d, v) in dst.iter_mut().zip(src) {
                    *d = beta.mul_add(*d, alpha * *v);
                }
            }
        }
    }
}

/// `preld` 预取一条缓存行（LASX 上就是普通标量预取；不改变任何数值语义）。
#[inline]
unsafe fn prefetch_line(p: *const f32) {
    // SAFETY: 调用方传的是仍存活的 `packed` 切片内地址；`preld` 对非法地址也只是 no-op，
    // 不影响正确性（这里额外保证地址有效）。
    unsafe {
        core::arch::asm!("preld 0, {}, 0", in(reg) p, options(nostack, preserves_flags));
    }
}

/// 预取下一个 strip 的 B 面板（均匀抽样：每 8 行预取 1 条）。
fn prefetch_strip(packed: &[f32], s: usize, k: usize) {
    let base = s * k * 32;
    let mut q = 0;
    while q < k {
        // SAFETY: `base + q*32 < packed.len()`（调用方保证 s < strips）。
        unsafe { prefetch_line(packed.as_ptr().add(base + q * 32)) };
        q += 8;
    }
}

/// 路线 B 的可配置入口：生产包装 [`matmul_f32_packed_rows`] 用默认配置；
/// 只有 perf 探针/实验会传别的 [`KernelCfg`]。
#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_f32_packed_rows_at(
    k: usize,
    n: usize,
    jb: usize,
    nc: usize,
    packed: &[f32],
    a_rows: &[f32],
    c_rows: &mut [f32],
    m4: usize,
    epi: Epi,
    cfg: KernelCfg,
) {
    // k == 0 时不会走进这个入口（调用方已提前返回），用 checked_div 让 clippy 与读者都放心
    let m = a_rows.len().checked_div(k).unwrap_or(0);
    let strips = nc / 32;
    // 宏内核：**k 分块在外、行块在内**。这样每个 k 块打包后的 B
    //（K_CHUNK×32×4 = 32 KB）会被所有行块反复命中，并与 A 的 4 行、C 的 4×32
    // 一起待在 L1 里。代价是 C 每块读写一次（k=1024 时多 3 次），相对算力可忽略。
    let kb_size = if k * 32 * 4 <= cfg.l1_budget {
        k
    } else {
        cfg.k_chunk.max(1)
    };
    let use_buf = cfg.c_strip_buf && kb_size < k;
    let mut cbuf_storage: Vec<f32> = Vec::new();
    if use_buf {
        cbuf_storage.resize(m * 32, 0.0);
    }
    // C 独立缓冲**只对单个 strip 有意义**：它的生命周期是"一个 strip 的全部 k 块"，
    // 所以必须配 strip 主序（`strip → k块 → 行块`）；chunk 主序下缓冲会被下一个 strip 覆盖。
    let strip_major = cfg.strip_outer || use_buf;
    if strip_major {
        for s in 0..strips {
            let j = jb + s * 32;
            if cfg.prefetch && s + 1 < strips {
                prefetch_strip(packed, s + 1, k);
            }
            let mut kb0 = 0;
            while kb0 < k {
                let kb = kb_size.min(k - kb0);
                let first = kb0 == 0;
                let base = s * k * 32 + kb0 * 32;
                let strip = &packed[base..base + kb * 32];
                if use_buf {
                    run_strip_chunk(
                        m4,
                        m,
                        n,
                        k,
                        j,
                        kb0,
                        kb,
                        strip,
                        a_rows,
                        first,
                        Epi::Plain,
                        c_rows,
                        &mut cbuf_storage,
                        true,
                    );
                } else {
                    let epi = if kb0 + kb == k { epi } else { Epi::Plain };
                    run_strip_chunk(
                        m4,
                        m,
                        n,
                        k,
                        j,
                        kb0,
                        kb,
                        strip,
                        a_rows,
                        first,
                        epi,
                        c_rows,
                        &mut [],
                        false,
                    );
                }
                kb0 += kb;
            }
            if use_buf {
                finish_cbuf(&cbuf_storage, c_rows, m, n, j, epi);
            }
        }
    } else {
        let mut kb0 = 0;
        while kb0 < k {
            let kb = kb_size.min(k - kb0);
            let first = kb0 == 0;
            // 只有最后一个 k 块才应用输出口径；中间的块写的是**部分和**，必须原样落盘。
            let epi_chunk = if kb0 + kb == k { epi } else { Epi::Plain };
            for s in 0..strips {
                let base = s * k * 32 + kb0 * 32;
                let strip = &packed[base..base + kb * 32];
                let j = jb + s * 32;
                if cfg.prefetch && s + 1 < strips {
                    prefetch_strip(packed, s + 1, k);
                }
                if use_buf {
                    run_strip_chunk(
                        m4,
                        m,
                        n,
                        k,
                        j,
                        kb0,
                        kb,
                        strip,
                        a_rows,
                        first,
                        Epi::Plain,
                        c_rows,
                        &mut cbuf_storage,
                        true,
                    );
                } else {
                    run_strip_chunk(
                        m4,
                        m,
                        n,
                        k,
                        j,
                        kb0,
                        kb,
                        strip,
                        a_rows,
                        first,
                        epi_chunk,
                        c_rows,
                        &mut [],
                        false,
                    );
                }
            }
            kb0 += kb;
        }
        if use_buf {
            for s in 0..strips {
                let j = jb + s * 32;
                finish_cbuf(&cbuf_storage, c_rows, m, n, j, epi);
            }
        }
    }
}

/// **融合入口（核不落地）的整条乘法**：与 [`matmul_f32_packed`] 同构，但 **A 的行由回调按需生成**，
/// 不落地 `m×k` 的 A。
///
/// - **主 strips** 走 [`matmul_f32_packed_rows_gen`]（每 k 块只生成 `4×k` 面板）；
/// - **列尾 `[n32, n)`** 复用**现有** `row_tail_range_f32`（它本来就吃"整行 A"）⇒ **不新增标量路径**。
///
/// ⇒ **strip 布局、k 分块、每个输出元素的累加次序与物化版完全一致**：只要 `gen_row` 写出的值与
/// 物化版逐字相同，结果与 [`matmul_f32_packed`] **逐位一致**。
pub(crate) fn matmul_f32_packed_gen<G: FnMut(usize, &mut [f32])>(
    m: usize,
    k: usize,
    n: usize,
    b: &[f32],
    mut gen_row: G,
    c: &mut [f32],
    epi: Epi,
) {
    if m == 0 || n == 0 {
        return;
    }
    let m4 = m / 4 * 4;
    let n32 = n / 32 * 32;
    let strips_total = n32 / 32;
    let nc_strips = pack_strips(k);
    PACK_BUF.with(|buf| {
        let mut packed = buf.borrow_mut();
        packed.resize(k * nc_strips * 32, 0.0);
        let mut s0 = 0;
        while s0 < strips_total {
            let s1 = (s0 + nc_strips).min(strips_total);
            let jb = s0 * 32;
            let nc = (s1 - s0) * 32;
            pack_b(k, n, jb, nc, b, &mut packed);
            // 注意：`gen_row` 是 `FnMut`，这里需要可变借用 ⇒ 用 `&mut gen_row` 转发。
            matmul_f32_packed_rows_gen(m, k, n, jb, nc, &packed, &mut gen_row, c, m4, epi);
            s0 = s1;
        }
    });
    // 列尾：复用现有尾路径（吃整行 A）——行由回调生成到暂存
    if n > n32 {
        let mut row = vec![0.0f32; k];
        for i in 0..m {
            gen_row(i, &mut row);
            let c_row = &mut c[i * n..(i + 1) * n];
            row_tail_range_f32(&row, c_row, b, k, n, n32, n, epi);
        }
    }
}

/// **融合入口：核不落地**——把 `a_rows`（`m×k` 的整块 A）换成**按需生成 A 行**的回调。
///
/// `gen_row(row, out)` 必须把 A 的第 `row` 行 `[0..k]` 写进 `out`。
/// 除"A 从哪来"之外与 [`matmul_f32_packed_rows`] 同构：同一 strip 打包布局、同一
/// `tile4x32_chunk` / `row_tail_packed` 调用、**每个输出元素的 k 累加次序不变**
/// ⇒ 若 `gen_row` 写出的值与物化版**相同**，结果**逐位一致**。
///
/// 内存上只保留 `4×k` 的暂存面板（`k=1024` 时 16 KB），**不落地 `m×k` 的核**。
pub(crate) fn matmul_f32_packed_rows_gen<G: FnMut(usize, &mut [f32])>(
    m: usize,
    k: usize,
    n: usize,
    jb: usize,
    nc: usize,
    packed: &[f32],
    mut gen_row: G,
    c_rows: &mut [f32],
    m4: usize,
    epi: Epi,
) {
    let strips = nc / 32;
    let kb_size = if k * 32 * 4 <= L1_BUDGET { k } else { K_CHUNK };
    let mut scratch = vec![0.0f32; 4 * k];
    let mut kb0 = 0;
    while kb0 < k {
        let kb = kb_size.min(k - kb0);
        let first = kb0 == 0;
        let epi = if kb0 + kb == k { epi } else { Epi::Plain };
        let mut i = 0;
        while i < m4 {
            for r in 0..4 {
                gen_row(i + r, &mut scratch[r * k..(r + 1) * k]);
            }
            for s in 0..strips {
                let base = s * k * 32 + kb0 * 32;
                let strip = &packed[base..base + kb * 32];
                let j = jb + s * 32;
                let ct = &mut c_rows[i * n..(i + 4) * n];
                tile4x32_chunk(
                    kb,
                    first,
                    strip,
                    &scratch[kb0..kb0 + kb],
                    &scratch[k + kb0..k + kb0 + kb],
                    &scratch[2 * k + kb0..2 * k + kb0 + kb],
                    &scratch[3 * k + kb0..3 * k + kb0 + kb],
                    ct,
                    n,
                    j,
                    epi,
                );
            }
            i += 4;
        }
        while i < m {
            gen_row(i, &mut scratch[..k]);
            for s in 0..strips {
                let base = s * k * 32 + kb0 * 32;
                let strip = &packed[base..base + kb * 32];
                let j = jb + s * 32;
                let c_row = &mut c_rows[i * n..(i + 1) * n];
                row_tail_packed(&scratch[kb0..kb0 + kb], strip, c_row, first, j, epi);
            }
            i += 1;
        }
        kb0 += kb;
    }
}

/// 把 `B[p][jb..jb+nc]` 打成 `packed[(s*k + p)*32 + r]`。
///
/// 布局决定了**只能让一侧顺序**：读是"每行 32 个 `f32`（128 B）、跨行跳 `n×4`"，
/// 写是"每条带 `k×32` 个 `f32`（`k×128` B）连续"。这里两侧都按 4 条带一组走：
/// 读侧每行取 512 B 连续（4 条带在同一行里是挨着的），写侧每条带仍连续。
///
/// **必须用向量拷贝**：`copy_from_slice` 会落进通用 memcpy——先按地址做对齐剥离与重叠检查、
/// 再按块起分支——实测打包吞吐只有 10–18 GB/s；显式按布局发 4 条 256 位载入/存储后是
/// 18–46 GB/s（探针与端到端都测过）。循环顺序本身不是瓶颈（两种都实测过）。
/// 一次处理 4 条带（读侧每行 512 B 连续）是端到端选的：宽 B 形状比单条带快 2%，其余打平。
#[inline]
pub fn pack_b(k: usize, n: usize, jb: usize, nc: usize, b: &[f32], packed: &mut [f32]) {
    let strips = nc / 32;
    let mut s0 = 0;
    while s0 < strips {
        let g = (strips - s0).min(4);
        for p in 0..k {
            for t in 0..g {
                let src = p * n + jb + (s0 + t) * 32;
                let dst = (s0 + t) * k * 32 + p * 32;
                // SAFETY: `strips = nc/32` 且 `jb + nc <= n`（三个调用点都按"完全覆盖的列面板"
                // 调用，`nc` 是 32 的倍数），于是 `src + 31 <= (k-1)*n + jb + nc - 1 <= k*n - 1`；
                // `packed` 的长度是 `k*nc`，`dst + 31 <= k*32*strips - 1 <= k*nc - 1`。
                // 两块不重叠：`packed` 是本模块自己的缓冲，`b` 是调用方的只读矩阵，
                // 由类型系统保证（安全代码里 `&[f32]` 与 `&mut [f32]` 不可能别名）。
                unsafe {
                    copy_block32(b.as_ptr().add(src), packed.as_mut_ptr().add(dst));
                }
            }
        }
        s0 += g;
    }
}

/// 拷贝 32 个连续 `f32`（4 条 256 位向量）。
///
/// # Safety
/// `src` 至少可读 32 个 `f32`，`dst` 至少可写 32 个 `f32`，且两者不重叠。
#[inline]
unsafe fn copy_block32(src: *const f32, dst: *mut f32) {
    // SAFETY: 调用方保证 `src` 可读、`dst` 可写各 32 个 `f32` 且不重叠（见上面的 `# Safety`）；
    // 8 次 `add` 的偏移都落在 0..32 内。
    unsafe {
        let v0 = lasx::load_f32x8(src);
        let v1 = lasx::load_f32x8(src.add(8));
        let v2 = lasx::load_f32x8(src.add(16));
        let v3 = lasx::load_f32x8(src.add(24));
        lasx::store_f32x8(dst, v0);
        lasx::store_f32x8(dst.add(8), v1);
        lasx::store_f32x8(dst.add(16), v2);
        lasx::store_f32x8(dst.add(24), v3);
    }
}

/// 微内核：4 行 × 32 列 × **kb 个 k 步**，B 从打包缓冲连续读。
///
/// `first` 为真表示这是该 (行块, 列条带) 的第一个 k 块，累加器从 0 起；否则先把 C 里
/// 已有的部分和读进累加器再继续加——**结合次序与不分块时完全一致**（都是按 p 递增累加
/// f32 部分和），所以结果逐位不变。
///
/// `epi` 只该在**最后一个 k 块**传非 `Plain`（见 [`matmul_f32_packed_rows`]）：中间块落盘的
/// 是部分和，必须原样写。
#[inline]
#[allow(clippy::too_many_arguments)]
fn tile4x32_chunk(
    kb: usize,
    first: bool,
    strip: &[f32],
    a0: &[f32],
    a1: &[f32],
    a2: &[f32],
    a3: &[f32],
    ct: &mut [f32],
    n: usize,
    j: usize,
    epi: Epi,
) {
    let mut rows = ct.chunks_mut(n);
    let c0 = rows.next().expect("4 行");
    let c1 = rows.next().expect("4 行");
    let c2 = rows.next().expect("4 行");
    let c3 = rows.next().expect("4 行");

    let (mut r0a, mut r0b, mut r0c, mut r0d) = z4();
    let (mut r1a, mut r1b, mut r1c, mut r1d) = z4();
    let (mut r2a, mut r2b, mut r2c, mut r2d) = z4();
    let (mut r3a, mut r3b, mut r3c, mut r3d) = z4();
    if !first {
        // SAFETY: j..j+32 在行内（调用方保证 32 列条带完整）。
        unsafe {
            r0a = lasx::load_f32x8(c0.as_ptr().add(j));
            r0b = lasx::load_f32x8(c0.as_ptr().add(j + 8));
            r0c = lasx::load_f32x8(c0.as_ptr().add(j + 16));
            r0d = lasx::load_f32x8(c0.as_ptr().add(j + 24));
            r1a = lasx::load_f32x8(c1.as_ptr().add(j));
            r1b = lasx::load_f32x8(c1.as_ptr().add(j + 8));
            r1c = lasx::load_f32x8(c1.as_ptr().add(j + 16));
            r1d = lasx::load_f32x8(c1.as_ptr().add(j + 24));
            r2a = lasx::load_f32x8(c2.as_ptr().add(j));
            r2b = lasx::load_f32x8(c2.as_ptr().add(j + 8));
            r2c = lasx::load_f32x8(c2.as_ptr().add(j + 16));
            r2d = lasx::load_f32x8(c2.as_ptr().add(j + 24));
            r3a = lasx::load_f32x8(c3.as_ptr().add(j));
            r3b = lasx::load_f32x8(c3.as_ptr().add(j + 8));
            r3c = lasx::load_f32x8(c3.as_ptr().add(j + 16));
            r3d = lasx::load_f32x8(c3.as_ptr().add(j + 24));
        }
    }
    for p in 0..kb {
        let base = p * 32;
        // 顺序读：p 前进 32 个 f32
        let vb0 = unsafe { lasx::load_f32x8(strip.as_ptr().add(base)) };
        let vb1 = unsafe { lasx::load_f32x8(strip.as_ptr().add(base + 8)) };
        let vb2 = unsafe { lasx::load_f32x8(strip.as_ptr().add(base + 16)) };
        let vb3 = unsafe { lasx::load_f32x8(strip.as_ptr().add(base + 24)) };
        let s0 = lasx::splat_f32(a0[p]);
        let s1 = lasx::splat_f32(a1[p]);
        let s2 = lasx::splat_f32(a2[p]);
        let s3 = lasx::splat_f32(a3[p]);
        unsafe {
            r0a = lasx_xvfmadd_s(vb0, s0, r0a);
            r0b = lasx_xvfmadd_s(vb1, s0, r0b);
            r0c = lasx_xvfmadd_s(vb2, s0, r0c);
            r0d = lasx_xvfmadd_s(vb3, s0, r0d);
            r1a = lasx_xvfmadd_s(vb0, s1, r1a);
            r1b = lasx_xvfmadd_s(vb1, s1, r1b);
            r1c = lasx_xvfmadd_s(vb2, s1, r1c);
            r1d = lasx_xvfmadd_s(vb3, s1, r1d);
            r2a = lasx_xvfmadd_s(vb0, s2, r2a);
            r2b = lasx_xvfmadd_s(vb1, s2, r2b);
            r2c = lasx_xvfmadd_s(vb2, s2, r2c);
            r2d = lasx_xvfmadd_s(vb3, s2, r2d);
            r3a = lasx_xvfmadd_s(vb0, s3, r3a);
            r3b = lasx_xvfmadd_s(vb1, s3, r3b);
            r3c = lasx_xvfmadd_s(vb2, s3, r3c);
            r3d = lasx_xvfmadd_s(vb3, s3, r3d);
        }
    }
    // SAFETY: j..j+32 在行内（调用方保证 32 列条带完整）；`Epi::Fma` 会读同一段 C。
    unsafe {
        epi.store8(c0.as_mut_ptr().add(j), r0a);
        epi.store8(c0.as_mut_ptr().add(j + 8), r0b);
        epi.store8(c0.as_mut_ptr().add(j + 16), r0c);
        epi.store8(c0.as_mut_ptr().add(j + 24), r0d);
        epi.store8(c1.as_mut_ptr().add(j), r1a);
        epi.store8(c1.as_mut_ptr().add(j + 8), r1b);
        epi.store8(c1.as_mut_ptr().add(j + 16), r1c);
        epi.store8(c1.as_mut_ptr().add(j + 24), r1d);
        epi.store8(c2.as_mut_ptr().add(j), r2a);
        epi.store8(c2.as_mut_ptr().add(j + 8), r2b);
        epi.store8(c2.as_mut_ptr().add(j + 16), r2c);
        epi.store8(c2.as_mut_ptr().add(j + 24), r2d);
        epi.store8(c3.as_mut_ptr().add(j), r3a);
        epi.store8(c3.as_mut_ptr().add(j + 8), r3b);
        epi.store8(c3.as_mut_ptr().add(j + 16), r3c);
        epi.store8(c3.as_mut_ptr().add(j + 24), r3d);
    }
}

/// 单行 × 32 列，B 从打包缓冲读（不足 4 行的尾部行）。`epi` 同 [`tile4x32_chunk`]：
/// 只该在最后一个 k 块传非 `Plain`。
#[inline]
fn row_tail_packed(
    a_row: &[f32],
    strip: &[f32],
    c_row: &mut [f32],
    first: bool,
    j: usize,
    epi: Epi,
) {
    let mut acc0 = lasx::zero_f32x8();
    let mut acc1 = lasx::zero_f32x8();
    let mut acc2 = lasx::zero_f32x8();
    let mut acc3 = lasx::zero_f32x8();
    if !first {
        // SAFETY: j..j+32 在行内（调用方保证 32 列条带完整）。
        unsafe {
            acc0 = lasx::load_f32x8(c_row.as_ptr().add(j));
            acc1 = lasx::load_f32x8(c_row.as_ptr().add(j + 8));
            acc2 = lasx::load_f32x8(c_row.as_ptr().add(j + 16));
            acc3 = lasx::load_f32x8(c_row.as_ptr().add(j + 24));
        }
    }
    for (p, &a_p) in a_row.iter().enumerate() {
        let va = lasx::splat_f32(a_p);
        let base = p * 32;
        unsafe {
            let b0 = lasx::load_f32x8(strip.as_ptr().add(base));
            let b1 = lasx::load_f32x8(strip.as_ptr().add(base + 8));
            let b2 = lasx::load_f32x8(strip.as_ptr().add(base + 16));
            let b3 = lasx::load_f32x8(strip.as_ptr().add(base + 24));
            acc0 = lasx_xvfmadd_s(b0, va, acc0);
            acc1 = lasx_xvfmadd_s(b1, va, acc1);
            acc2 = lasx_xvfmadd_s(b2, va, acc2);
            acc3 = lasx_xvfmadd_s(b3, va, acc3);
        }
    }
    // SAFETY: j..j+32 在行内（调用方保证 32 列条带完整）；`Epi::Fma` 会读同一段 C。
    unsafe {
        epi.store8(c_row.as_mut_ptr().add(j), acc0);
        epi.store8(c_row.as_mut_ptr().add(j + 8), acc1);
        epi.store8(c_row.as_mut_ptr().add(j + 16), acc2);
        epi.store8(c_row.as_mut_ptr().add(j + 24), acc3);
    }
}

/// 一个打包面板放几条 32 列条带：面板要能留在 L2（3 MiB/核）里被整轮行扫描复用，
/// 预算取 2 MiB，所以 `nc_strips = 2 MiB / (k×32×4)`，上限 256 条（f32 = 8192 列）。
///
/// 注意条带宽度是 **32 列**（f64 侧是 16 列，见 [`crate::ops::matmul_f64::pack_strips`]）：
/// 曾经考虑过 8 行 × 16 列的微内核（L2 复用遍数减半、条带 28 KB），但 4×32 的
/// "16 累加器 + 4 载入 + 4 广播" 已经是寄存器可容纳的最优 ops/FMA 组合——
/// `cRxC` 扫描里 `c8x16` 实测同样是 2.00 FMA/周期；改 f64 内核形状的同类尝试也实测否掉了。
#[inline]
pub fn pack_strips(k: usize) -> usize {
    const L2_BUDGET: usize = 2 * 1024 * 1024;
    let per_strip = (k * 32 * 4).max(1);
    (L2_BUDGET / per_strip).clamp(1, 256)
}

/// 强制走"流式"次序（**测试对照用**；生产路径由 [`matmul_f32`] 按形状分派）。
#[cfg(test)]
pub(crate) fn matmul_f32_stream_ref(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    let m4 = m / 4 * 4;
    let n32 = n / 32 * 32;
    let mut i = 0;
    while i < m4 {
        let mut j = 0;
        while j < n32 {
            tile4x32_f32(i, j, k, n, a, b, c);
            j += 32;
        }
        i += 4;
    }
    let mut i = 0;
    while i < m4 {
        for r in 0..4 {
            let a_row = &a[(i + r) * k..(i + r + 1) * k];
            let c_row = &mut c[(i + r) * n..(i + r + 1) * n];
            row_tail_range_f32(a_row, c_row, b, k, n, n32, n, Epi::Plain);
        }
        i += 4;
    }
    while i < m {
        let a_row = &a[i * k..(i + 1) * k];
        let c_row = &mut c[i * n..(i + 1) * n];
        cols32_f32(a_row, c_row, b, n);
        row_tail_range_f32(a_row, c_row, b, k, n, n32, n, Epi::Plain);
        i += 1;
    }
}

/// 微内核：一次算 4 行 × 32 列（16 个累加器），B 的 4 个向量被 4 行共享。
///
/// `j + 32 <= n` 由调用方保证（只走 `[0, n/32*32)`）；列尾另由 [`row_tail_f32`] 处理。
/// `p` 既用于索引 4 行 A 又用于计算 B 的地址，故保留下标循环。
///
/// **不带缩放**（流式档专用）：这条内核是小形状的热点，给它加运行时口径参数实测
/// 64³ 7.6 → 7.9 µs（+3.9%）；所以融合走下面的 [`tile4x32_f32_scaled`]，本函数与改动前
/// 逐字节相同。
#[inline]
#[allow(clippy::needless_range_loop)]
fn tile4x32_f32(i0: usize, j: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let a0 = &a[i0 * k..(i0 + 1) * k];
    let a1 = &a[(i0 + 1) * k..(i0 + 2) * k];
    let a2 = &a[(i0 + 2) * k..(i0 + 3) * k];
    let a3 = &a[(i0 + 3) * k..(i0 + 4) * k];

    // 4 行 C 互不重叠，用 chunks_mut 安全地同时取出
    let mut rows = c[i0 * n..(i0 + 4) * n].chunks_mut(n);
    let c0 = rows.next().expect("4 行");
    let c1 = rows.next().expect("4 行");
    let c2 = rows.next().expect("4 行");
    let c3 = rows.next().expect("4 行");

    let (mut r0a, mut r0b, mut r0c, mut r0d) = z4();
    let (mut r1a, mut r1b, mut r1c, mut r1d) = z4();
    let (mut r2a, mut r2b, mut r2c, mut r2d) = z4();
    let (mut r3a, mut r3b, mut r3c, mut r3d) = z4();
    for p in 0..k {
        let base = p * n + j;
        let vb0 = unsafe { lasx::load_f32x8(b.as_ptr().add(base)) };
        let vb1 = unsafe { lasx::load_f32x8(b.as_ptr().add(base + 8)) };
        let vb2 = unsafe { lasx::load_f32x8(b.as_ptr().add(base + 16)) };
        let vb3 = unsafe { lasx::load_f32x8(b.as_ptr().add(base + 24)) };
        let s0 = lasx::splat_f32(a0[p]);
        let s1 = lasx::splat_f32(a1[p]);
        let s2 = lasx::splat_f32(a2[p]);
        let s3 = lasx::splat_f32(a3[p]);
        unsafe {
            r0a = lasx_xvfmadd_s(vb0, s0, r0a);
            r0b = lasx_xvfmadd_s(vb1, s0, r0b);
            r0c = lasx_xvfmadd_s(vb2, s0, r0c);
            r0d = lasx_xvfmadd_s(vb3, s0, r0d);
            r1a = lasx_xvfmadd_s(vb0, s1, r1a);
            r1b = lasx_xvfmadd_s(vb1, s1, r1b);
            r1c = lasx_xvfmadd_s(vb2, s1, r1c);
            r1d = lasx_xvfmadd_s(vb3, s1, r1d);
            r2a = lasx_xvfmadd_s(vb0, s2, r2a);
            r2b = lasx_xvfmadd_s(vb1, s2, r2b);
            r2c = lasx_xvfmadd_s(vb2, s2, r2c);
            r2d = lasx_xvfmadd_s(vb3, s2, r2d);
            r3a = lasx_xvfmadd_s(vb0, s3, r3a);
            r3b = lasx_xvfmadd_s(vb1, s3, r3b);
            r3c = lasx_xvfmadd_s(vb2, s3, r3c);
            r3d = lasx_xvfmadd_s(vb3, s3, r3d);
        }
    }
    unsafe {
        lasx::store_f32x8(c0.as_mut_ptr().add(j), r0a);
        lasx::store_f32x8(c0.as_mut_ptr().add(j + 8), r0b);
        lasx::store_f32x8(c0.as_mut_ptr().add(j + 16), r0c);
        lasx::store_f32x8(c0.as_mut_ptr().add(j + 24), r0d);
        lasx::store_f32x8(c1.as_mut_ptr().add(j), r1a);
        lasx::store_f32x8(c1.as_mut_ptr().add(j + 8), r1b);
        lasx::store_f32x8(c1.as_mut_ptr().add(j + 16), r1c);
        lasx::store_f32x8(c1.as_mut_ptr().add(j + 24), r1d);
        lasx::store_f32x8(c2.as_mut_ptr().add(j), r2a);
        lasx::store_f32x8(c2.as_mut_ptr().add(j + 8), r2b);
        lasx::store_f32x8(c2.as_mut_ptr().add(j + 16), r2c);
        lasx::store_f32x8(c2.as_mut_ptr().add(j + 24), r2d);
        lasx::store_f32x8(c3.as_mut_ptr().add(j), r3a);
        lasx::store_f32x8(c3.as_mut_ptr().add(j + 8), r3b);
        lasx::store_f32x8(c3.as_mut_ptr().add(j + 16), r3c);
        lasx::store_f32x8(c3.as_mut_ptr().add(j + 24), r3d);
    }
}

/// [`tile4x32_f32`] 的**融合收尾版**（列块路径用）：k 循环逐条指令相同，只是收尾走
/// [`Epi::store8`]（`alpha·acc` 或 `fma(beta, C_old, alpha·acc)`）。累加器一直在寄存器里
/// （列块次序不分块 k），所以 `Epi::Fma` 读 `C_old` 时 C 还没被写过。
#[inline]
#[allow(clippy::needless_range_loop)]
fn tile4x32_f32_scaled(
    i0: usize,
    j: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    epi: Epi,
) {
    let a0 = &a[i0 * k..(i0 + 1) * k];
    let a1 = &a[(i0 + 1) * k..(i0 + 2) * k];
    let a2 = &a[(i0 + 2) * k..(i0 + 3) * k];
    let a3 = &a[(i0 + 3) * k..(i0 + 4) * k];

    // 4 行 C 互不重叠，用 chunks_mut 安全地同时取出
    let mut rows = c[i0 * n..(i0 + 4) * n].chunks_mut(n);
    let c0 = rows.next().expect("4 行");
    let c1 = rows.next().expect("4 行");
    let c2 = rows.next().expect("4 行");
    let c3 = rows.next().expect("4 行");

    let (mut r0a, mut r0b, mut r0c, mut r0d) = z4();
    let (mut r1a, mut r1b, mut r1c, mut r1d) = z4();
    let (mut r2a, mut r2b, mut r2c, mut r2d) = z4();
    let (mut r3a, mut r3b, mut r3c, mut r3d) = z4();
    for p in 0..k {
        let base = p * n + j;
        let vb0 = unsafe { lasx::load_f32x8(b.as_ptr().add(base)) };
        let vb1 = unsafe { lasx::load_f32x8(b.as_ptr().add(base + 8)) };
        let vb2 = unsafe { lasx::load_f32x8(b.as_ptr().add(base + 16)) };
        let vb3 = unsafe { lasx::load_f32x8(b.as_ptr().add(base + 24)) };
        let s0 = lasx::splat_f32(a0[p]);
        let s1 = lasx::splat_f32(a1[p]);
        let s2 = lasx::splat_f32(a2[p]);
        let s3 = lasx::splat_f32(a3[p]);
        unsafe {
            r0a = lasx_xvfmadd_s(vb0, s0, r0a);
            r0b = lasx_xvfmadd_s(vb1, s0, r0b);
            r0c = lasx_xvfmadd_s(vb2, s0, r0c);
            r0d = lasx_xvfmadd_s(vb3, s0, r0d);
            r1a = lasx_xvfmadd_s(vb0, s1, r1a);
            r1b = lasx_xvfmadd_s(vb1, s1, r1b);
            r1c = lasx_xvfmadd_s(vb2, s1, r1c);
            r1d = lasx_xvfmadd_s(vb3, s1, r1d);
            r2a = lasx_xvfmadd_s(vb0, s2, r2a);
            r2b = lasx_xvfmadd_s(vb1, s2, r2b);
            r2c = lasx_xvfmadd_s(vb2, s2, r2c);
            r2d = lasx_xvfmadd_s(vb3, s2, r2d);
            r3a = lasx_xvfmadd_s(vb0, s3, r3a);
            r3b = lasx_xvfmadd_s(vb1, s3, r3b);
            r3c = lasx_xvfmadd_s(vb2, s3, r3c);
            r3d = lasx_xvfmadd_s(vb3, s3, r3d);
        }
    }
    // SAFETY: j..j+32 在行内（调用方保证 32 列条带完整）；`Epi::Fma` 会读同一段 C。
    unsafe {
        epi.store8(c0.as_mut_ptr().add(j), r0a);
        epi.store8(c0.as_mut_ptr().add(j + 8), r0b);
        epi.store8(c0.as_mut_ptr().add(j + 16), r0c);
        epi.store8(c0.as_mut_ptr().add(j + 24), r0d);
        epi.store8(c1.as_mut_ptr().add(j), r1a);
        epi.store8(c1.as_mut_ptr().add(j + 8), r1b);
        epi.store8(c1.as_mut_ptr().add(j + 16), r1c);
        epi.store8(c1.as_mut_ptr().add(j + 24), r1d);
        epi.store8(c2.as_mut_ptr().add(j), r2a);
        epi.store8(c2.as_mut_ptr().add(j + 8), r2b);
        epi.store8(c2.as_mut_ptr().add(j + 16), r2c);
        epi.store8(c2.as_mut_ptr().add(j + 24), r2d);
        epi.store8(c3.as_mut_ptr().add(j), r3a);
        epi.store8(c3.as_mut_ptr().add(j + 8), r3b);
        epi.store8(c3.as_mut_ptr().add(j + 16), r3c);
        epi.store8(c3.as_mut_ptr().add(j + 24), r3d);
    }
}

/// 单行主循环：32 列一块（4 个累加器）。**不带缩放**（流式档专用，理由同
/// [`tile4x32_f32`]：小形状的热点，加口径参数实测 +3.9%）。
#[inline]
fn cols32_f32(a_row: &[f32], c_row: &mut [f32], b: &[f32], n: usize) {
    let mut j = 0;
    while j + 32 <= n {
        let (mut acc0, mut acc1, mut acc2, mut acc3) = z4();
        for (p, &a_p) in a_row.iter().enumerate() {
            let va = lasx::splat_f32(a_p);
            let base = p * n + j;
            let b0 = unsafe { lasx::load_f32x8(b.as_ptr().add(base)) };
            let b1 = unsafe { lasx::load_f32x8(b.as_ptr().add(base + 8)) };
            let b2 = unsafe { lasx::load_f32x8(b.as_ptr().add(base + 16)) };
            let b3 = unsafe { lasx::load_f32x8(b.as_ptr().add(base + 24)) };
            unsafe {
                acc0 = lasx_xvfmadd_s(b0, va, acc0);
                acc1 = lasx_xvfmadd_s(b1, va, acc1);
                acc2 = lasx_xvfmadd_s(b2, va, acc2);
                acc3 = lasx_xvfmadd_s(b3, va, acc3);
            }
        }
        unsafe {
            lasx::store_f32x8(c_row.as_mut_ptr().add(j), acc0);
            lasx::store_f32x8(c_row.as_mut_ptr().add(j + 8), acc1);
            lasx::store_f32x8(c_row.as_mut_ptr().add(j + 16), acc2);
            lasx::store_f32x8(c_row.as_mut_ptr().add(j + 24), acc3);
        }
        j += 32;
    }
}

/// 单行的列尾：`[j0, n)`，先 8 列块再标量。**不带缩放**（流式档专用）。
#[inline]
fn row_tail_f32(a_row: &[f32], c_row: &mut [f32], b: &[f32], k: usize, n: usize, j0: usize) {
    row_tail_range_f32(a_row, c_row, b, k, n, j0, n, Epi::Plain);
}

/// 单行的列尾 `[j0, j1)`：先 8 列块再标量。`n` 始终是 B/C 的**行跨距**。
///
/// `epi` 的标量段走 [`Epi::scalar`]（与参考实现同一个表达式）；`beta == 0` 时**不读**
/// `c_row`。
#[inline]
pub fn row_tail_range_f32(
    a_row: &[f32],
    c_row: &mut [f32],
    b: &[f32],
    k: usize,
    n: usize,
    j0: usize,
    j1: usize,
    epi: Epi,
) {
    let mut j = j0;
    while j + 8 <= j1 {
        let mut acc = lasx::zero_f32x8();
        for (p, &a_p) in a_row.iter().enumerate() {
            let va = lasx::splat_f32(a_p);
            let vb = unsafe { lasx::load_f32x8(b.as_ptr().add(p * n + j)) };
            acc = unsafe { lasx_xvfmadd_s(vb, va, acc) };
        }
        // SAFETY: j..j+8 在行内；`Epi::Fma` 会读同一段 C。
        unsafe { epi.store8(c_row.as_mut_ptr().add(j), acc) };
        j += 8;
    }
    for jj in j..j1 {
        let mut s = 0f32;
        for p in 0..k {
            s += a_row[p] * b[p * n + jj];
        }
        c_row[jj] = epi.scalar(|| c_row[jj], s);
    }
}

/// 4 个零累加器。
#[inline]
fn z4() -> (lasx::F32x8, lasx::F32x8, lasx::F32x8, lasx::F32x8) {
    (
        lasx::zero_f32x8(),
        lasx::zero_f32x8(),
        lasx::zero_f32x8(),
        lasx::zero_f32x8(),
    )
}

/// 数值回归测试：对照独立参考实现。
#[cfg(test)]
mod tests {
    use crate::ffi::matmul::lasx_matmul;
    use crate::ops::testutil::{Lcg, reference, rel};

    #[test]
    fn test_matmul_f32_matches_reference() {
        let shapes = [
            (0usize, 4usize, 3usize),
            (1, 1, 1),
            (1, 0, 5),
            (1, 8, 1),
            (3, 5, 7),
            (4, 4, 32),
            (4, 4, 33),
            (5, 7, 40),
            (8, 8, 8),
            (16, 33, 17),
            (32, 32, 32),
            (33, 65, 31),
            (64, 64, 64),
            (128, 128, 128),
        ];
        for &(m, k, n) in &shapes {
            let mut rng = Lcg(0x5eed_1234 ^ ((m * 31 + k) * 17 + n) as u64);
            let a: Vec<f32> = (0..m * k).map(|_| rng.f64() as f32).collect();
            let b: Vec<f32> = (0..k * n).map(|_| rng.f64() as f32).collect();
            let mut c = vec![f32::NAN; m * n];
            lasx_matmul(
                m as i32,
                k as i32,
                n as i32,
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
            );
            let want = reference(m, k, n, &a, &b);
            for idx in 0..m * n {
                // f32 沿 k 累加，k≤128 时相对误差应在 1e-4 内
                assert!(
                    rel(c[idx] as f64, want[idx]) < 1e-4,
                    "{m}×{k}×{n} idx={idx}: got {} want {}",
                    c[idx],
                    want[idx]
                );
            }
        }
    }
}

#[cfg(test)]
mod cols_exp {
    use super::*;
    use crate::ops::testutil::Lcg;

    /// 32 对齐的列区间入口必须与全量入口**逐位一致**，且区间外一个字节都不写。
    #[test]
    fn cols_bit_exact() {
        let mut rng = Lcg(0xc015);
        for &(m, k, n) in &[
            (1usize, 1usize, 32usize),
            (3, 5, 64),
            (5, 7, 96),
            (8, 8, 64),
            (13, 17, 128),
            (100, 448, 512),
        ] {
            let a: Vec<f32> = (0..m * k).map(|_| rng.f64() as f32).collect();
            let b: Vec<f32> = (0..k * n).map(|_| rng.f64() as f32).collect();
            let mut want = vec![0f32; m * n];
            matmul_f32(m, k, n, &a, &b, &mut want);
            // 32 对齐区间：整段、每 32 列、以及末尾不足 32 的列尾
            let mut ranges: Vec<(usize, usize)> = vec![(0, n), (0, 0)];
            let mut j = 0;
            while j + 32 <= n {
                ranges.push((j, j + 32));
                j += 32;
            }
            if n % 32 != 0 {
                ranges.push((n / 32 * 32, n));
            }
            for &(j0, j1) in &ranges {
                let mut got = vec![f32::NAN; m * n];
                matmul_f32_cols_block(m, k, n, j0, j1, &a, &b, &mut got, COL_BLOCK, Epi::Plain);
                for i in 0..m {
                    for jj in 0..n {
                        if jj >= j0 && jj < j1 {
                            assert_eq!(
                                got[i * n + jj].to_bits(),
                                want[i * n + jj].to_bits(),
                                "m={m} k={k} n={n} 区间[{j0},{j1}) i={i} j={jj}"
                            );
                        } else {
                            assert!(got[i * n + jj].is_nan(), "区间外不该被写 @ i={i} j={jj}");
                        }
                    }
                }
            }
        }
    }
    /// 三条路径（打包 / 列块 / 流式）必须**逐位一致**。
    #[test]
    fn packed_paths_bit_exact() {
        let mut rng = Lcg(0x9e37);
        for &(m, k, n) in &[
            (1usize, 1usize, 32usize),
            (3, 5, 64),
            (5, 7, 96),
            (8, 8, 64),
            (13, 17, 128),
            (100, 448, 1024),
            (100, 448, 19147),
        ] {
            let a: Vec<f32> = (0..m * k).map(|_| rng.f64() as f32).collect();
            let b: Vec<f32> = (0..k * n).map(|_| rng.f64() as f32).collect();
            let mut want = vec![f32::NAN; m * n];
            matmul_f32_stream_ref(m, k, n, &a, &b, &mut want);
            let mut cols = vec![f32::NAN; m * n];
            matmul_f32_cols_block(m, k, n, 0, n, &a, &b, &mut cols, COL_BLOCK, Epi::Plain);
            let mut pack = vec![f32::NAN; m * n];
            matmul_f32_packed(m, k, n, &a, &b, &mut pack, Epi::Plain);
            for i in 0..m * n {
                assert_eq!(
                    cols[i].to_bits(),
                    want[i].to_bits(),
                    "列块 {m}×{k}×{n} @ {i}"
                );
                assert_eq!(
                    pack[i].to_bits(),
                    want[i].to_bits(),
                    "打包 {m}×{k}×{n} @ {i}"
                );
            }
        }
    }

    /// 融合路径（`api` 层 `matmul_scaled_into` 的内核）必须与"未融合参考"**逐位一致**。
    ///
    /// 参考实现（口径的定义）：先把 `acc = A·B` 算进独立缓冲（走同一条 [`matmul_f32`]），
    /// 再 `C = beta.mul_add(C_old, alpha * acc)`——两次舍入（`alpha·acc` 一次、FMA 一次）。
    /// 融合路径只是把这一步搬进内核收尾，表达式**一模一样**，所以是逐位而不是容差。
    ///
    /// 覆盖三种收尾（`Plain` / `Scale` / `Fma`）× 三条路径（打包 / 列块 / 流式）×
    /// 有列尾与无列尾的形状。`k > 384` 时打包路径会分块 k：`beta == 0` 仍融合（末块才缩放），
    /// `beta != 0` 退回二遍法——两种都必须在表里逐位对上。
    #[test]
    fn scaled_bit_exact_against_two_pass_reference() {
        let mut rng = Lcg(0x5ca1_ed01);
        let cases: &[(usize, usize, usize)] = &[
            (1, 1, 1),
            (3, 5, 33),
            (4, 4, 32),
            (5, 7, 40),
            (16, 33, 17),
            (33, 65, 31),
            (32, 64, 96),     // 打包（小 k）
            (100, 64, 4096),  // 打包：宽薄
            (100, 64, 19147), // 列块：极宽 n + 小 k
            (256, 512, 512),  // 打包 + k 分块（k > 384）
            (100, 448, 1024), // 打包 + k 分块
        ];
        for &(m, k, n) in cases {
            let a: Vec<f32> = (0..m * k).map(|_| rng.f64() as f32).collect();
            let b: Vec<f32> = (0..k * n).map(|_| rng.f64() as f32).collect();
            let c0: Vec<f32> = (0..m * n).map(|_| rng.f64() as f32).collect();
            for &(alpha, beta) in &[
                (1.0f32, 0.0f32),
                (1.5, 0.0),
                (-0.25, 0.0),
                (1.0, 1.0),
                (1.5, 0.5),
                (0.75, -1.25),
            ] {
                // 参考：acc 走同一条 matmul，再按契约表达式收尾。
                let mut acc = vec![0f32; m * n];
                matmul_f32(m, k, n, &a, &b, &mut acc);
                let want: Vec<f32> = acc
                    .iter()
                    .zip(c0.iter())
                    .map(|(&x, &old)| beta.mul_add(old, alpha * x))
                    .collect();
                let mut got = c0.clone();
                matmul_f32_scaled(m, k, n, alpha, beta, &a, &b, &mut got);
                for i in 0..m * n {
                    assert_eq!(
                        got[i].to_bits(),
                        want[i].to_bits(),
                        "{m}×{k}×{n} alpha={alpha} beta={beta} @ {i}: got {} want {}",
                        got[i],
                        want[i]
                    );
                }
            }
            // `beta == 0` 不读 C：把 C 填成 NaN，结果里不许出现 NaN。
            let mut got = vec![f32::NAN; m * n];
            matmul_f32_scaled(m, k, n, 1.5, 0.0, &a, &b, &mut got);
            assert!(
                got.iter().all(|v| v.is_finite()),
                "{m}×{k}×{n}: beta == 0 时读了 C（NaN 传播）"
            );
        }
    }

    /// **路线 B 候选组合逐位等价**（`task-73`）：`KernelCfg` 的 strip 主序 / C 独立缓冲 /
    /// 预取 / 面板预算变体，对 ragged 形状与列尾都要与改动前配置**逐位一致**。
    ///
    /// 生产默认 = `legacy`（候选不采纳），但候选留在探针里供 Lead 叠加；这条测试保证
    /// "一旦有人把默认切到候选，数值不会变"。
    #[test]
    fn gemm_kernel_cfg_variants_stay_bitwise() {
        let cfgs = [
            (
                "strip_outer",
                KernelCfg {
                    strip_outer: true,
                    ..Default::default()
                },
            ),
            (
                "strip_outer+l1=64K",
                KernelCfg {
                    strip_outer: true,
                    l1_budget: 64 * 1024,
                    ..Default::default()
                },
            ),
            (
                "strip_outer+cbuf",
                KernelCfg {
                    strip_outer: true,
                    c_strip_buf: true,
                    ..Default::default()
                },
            ),
            (
                "strip_outer+prefetch",
                KernelCfg {
                    strip_outer: true,
                    prefetch: true,
                    ..Default::default()
                },
            ),
            (
                "strip_outer+panel3M",
                KernelCfg {
                    strip_outer: true,
                    panel_bytes: 3 * 1024 * 1024,
                    ..Default::default()
                },
            ),
        ];
        for (m, k, n) in [(37usize, 64usize, 45usize), (128, 256, 96), (513, 448, 130)] {
            let a: Vec<f32> = (0..m * k)
                .map(|i| (((i * 37 + 11) % 101) as f32) * 0.01 - 0.5)
                .collect();
            let b: Vec<f32> = (0..k * n)
                .map(|i| (((i * 53 + 7) % 97) as f32) * 0.01 - 0.5)
                .collect();
            let mut c_legacy = vec![0f32; m * n];
            matmul_f32_packed_at(
                m,
                k,
                n,
                &a,
                &b,
                &mut c_legacy,
                Epi::Plain,
                KernelCfg::legacy(),
            );
            for (name, cfg) in cfgs {
                let mut c = vec![0f32; m * n];
                matmul_f32_packed_at(m, k, n, &a, &b, &mut c, Epi::Plain, cfg);
                for (i, (x, y)) in c.iter().zip(&c_legacy).enumerate() {
                    assert_eq!(
                        x.to_bits(),
                        y.to_bits(),
                        "{m}×{k}×{n} cfg={name} @ {i}: {} vs {}",
                        x,
                        y
                    );
                }
            }
        }
    }
}

/// **路线 B perf 探针（`task-73`）**：`cargo test --release --lib gemm_l1_bench -- --ignored --nocapture`。
///
/// 对 1024³/1536³ 用多个 [`KernelCfg`] 组合跑 1t 打包内核：先与默认配置**逐位**对比，
/// 再进程内交替 5 轮（丢首轮、取中位）计时。生产路径不受影响（默认配置 = 改动前行为）。
#[cfg(test)]
mod gemm_l1_bench {
    use super::*;
    use std::time::{Duration, Instant};

    fn timeit(f: &mut impl FnMut()) -> Duration {
        let t = Instant::now();
        f();
        t.elapsed()
    }
    fn median(mut v: Vec<Duration>) -> Duration {
        v.sort_unstable();
        v[v.len() / 2]
    }

    #[test]
    #[ignore = "perf probe: cargo test --release --lib gemm_l1_bench -- --ignored --nocapture"]
    fn bench_gemm_l1_variants() {
        let cfgs: Vec<(&str, KernelCfg)> = vec![
            // **改动前行为**（chunk 主序）——真正的 baseline；`KernelCfg::default()` 是候选采纳配置。
            ("legacy(chunk_outer)", KernelCfg::legacy()),
            ("adopted(strip_outer)", KernelCfg::default()),
            (
                "strip_outer",
                KernelCfg {
                    strip_outer: true,
                    ..Default::default()
                },
            ),
            (
                "strip_outer+prefetch",
                KernelCfg {
                    strip_outer: true,
                    prefetch: true,
                    ..Default::default()
                },
            ),
            (
                "strip_outer+l1=64K",
                KernelCfg {
                    strip_outer: true,
                    l1_budget: 64 * 1024,
                    ..Default::default()
                },
            ),
            (
                "strip_outer+l1=64K+prefetch",
                KernelCfg {
                    strip_outer: true,
                    l1_budget: 64 * 1024,
                    prefetch: true,
                    ..Default::default()
                },
            ),
            (
                "strip_outer+panel3M",
                KernelCfg {
                    strip_outer: true,
                    panel_bytes: 3 * 1024 * 1024,
                    ..Default::default()
                },
            ),
            (
                "strip_outer+l1=64K+panel3M",
                KernelCfg {
                    strip_outer: true,
                    l1_budget: 64 * 1024,
                    panel_bytes: 3 * 1024 * 1024,
                    ..Default::default()
                },
            ),
            (
                "strip_outer+cbuf+prefetch",
                KernelCfg {
                    strip_outer: true,
                    c_strip_buf: true,
                    prefetch: true,
                    ..Default::default()
                },
            ),
        ];
        // `GEMM_CFG=<cfg 名>`：只跑该组合 + 1536³（供 `perf stat` 前后对照用；默认跑全部组合）。
        let only = std::env::var("GEMM_CFG").ok();
        let cfgs: Vec<(&str, KernelCfg)> = match &only {
            Some(sel) => cfgs.into_iter().filter(|(n, _)| n == sel).collect(),
            None => cfgs,
        };
        let shapes: Vec<(usize, usize, usize)> = if only.is_some() {
            vec![(1536, 1536, 1536)]
        } else {
            vec![(1024, 1024, 1024), (1536, 1536, 1536)]
        };
        for (m, k, n) in shapes {
            let a: Vec<f32> = (0..m * k)
                .map(|i| (((i * 37 + 11) % 101) as f32) * 0.01 - 0.5)
                .collect();
            let b: Vec<f32> = (0..k * n)
                .map(|i| (((i * 53 + 7) % 97) as f32) * 0.01 - 0.5)
                .collect();
            let gflops = |ms: f64| 2.0 * (m * k * n) as f64 / (ms * 1e-3) * 1e-9;
            // `perf stat` 专用：只跑选中的那个组合（8 轮丢首轮），不混入 base 臂/逐位参照。
            if let Some(sel) = &only {
                let (name, cfg) = cfgs[0];
                assert_eq!(name, sel, "GEMM_CFG 名字不匹配");
                let mut c = vec![0f32; m * n];
                let mut tv = Vec::new();
                for _ in 0..8 {
                    let t = timeit(&mut || {
                        matmul_f32_packed_at(m, k, n, &a, &b, &mut c, Epi::Plain, cfg)
                    });
                    tv.push(t);
                }
                std::hint::black_box(c[0]);
                tv.remove(0);
                let mv = median(tv).as_secs_f64() * 1e3;
                println!("| {name} | {mv:.2} | {:.2} | (perf 模式) | — |", gflops(mv));
                continue;
            }
            let mut c_ref = vec![0f32; m * n];
            matmul_f32_packed_at(
                m,
                k,
                n,
                &a,
                &b,
                &mut c_ref,
                Epi::Plain,
                KernelCfg::default(),
            );
            let gflops = |ms: f64| 2.0 * (m * k * n) as f64 / (ms * 1e-3) * 1e-9;
            println!("\n== {m}×{k}×{n}（1t 打包内核，base 为逐位参照）==");
            println!("| cfg | ms(med) | GFLOP/s | vs base | bitwise |");
            println!("|---|---|---|---|---|");
            for (name, cfg) in &cfgs {
                let mut c_var = vec![0f32; m * n];
                matmul_f32_packed_at(m, k, n, &a, &b, &mut c_var, Epi::Plain, *cfg);
                let bitwise = c_var
                    .iter()
                    .zip(&c_ref)
                    .all(|(x, y)| x.to_bits() == y.to_bits());
                let mut c_base = vec![0f32; m * n];
                let (mut tv, mut tb) = (Vec::new(), Vec::new());
                for r in 0..5 {
                    let (t1, t2) = if r % 2 == 0 {
                        let t1 = timeit(&mut || {
                            matmul_f32_packed_at(m, k, n, &a, &b, &mut c_var, Epi::Plain, *cfg)
                        });
                        let t2 = timeit(&mut || {
                            matmul_f32_packed_at(
                                m,
                                k,
                                n,
                                &a,
                                &b,
                                &mut c_base,
                                Epi::Plain,
                                KernelCfg::default(),
                            )
                        });
                        (t1, t2)
                    } else {
                        let t2 = timeit(&mut || {
                            matmul_f32_packed_at(
                                m,
                                k,
                                n,
                                &a,
                                &b,
                                &mut c_base,
                                Epi::Plain,
                                KernelCfg::default(),
                            )
                        });
                        let t1 = timeit(&mut || {
                            matmul_f32_packed_at(m, k, n, &a, &b, &mut c_var, Epi::Plain, *cfg)
                        });
                        (t1, t2)
                    };
                    tv.push(t1);
                    tb.push(t2);
                }
                std::hint::black_box(c_var[0]);
                std::hint::black_box(c_base[0]);
                tv.remove(0);
                tb.remove(0);
                let mv = median(tv).as_secs_f64() * 1e3;
                let mb = median(tb).as_secs_f64() * 1e3;
                println!(
                    "| {name} | {mv:.2} | {:.2} | {:+.1}% | {bitwise} |",
                    gflops(mv),
                    (mb / mv - 1.0) * 100.0
                );
            }
            // 诊断：整条 `matmul_f32`（C ABI 走的同一条）vs 直接打包内核。
            println!(
                "- 诊断：`pool::threads()` = {}（>1 且 work≥32M 时 `matmul_f32` 会走并行）",
                crate::pool::global().threads()
            );
            {
                let mut cf = vec![0f32; m * n];
                let mut tv = Vec::new();
                let mut tb = Vec::new();
                for r in 0..5 {
                    let (t1, t2) = if r % 2 == 0 {
                        let t1 = timeit(&mut || matmul_f32(m, k, n, &a, &b, &mut cf));
                        let t2 = timeit(&mut || {
                            matmul_f32_packed_at(
                                m,
                                k,
                                n,
                                &a,
                                &b,
                                &mut cf,
                                Epi::Plain,
                                KernelCfg::default(),
                            )
                        });
                        (t1, t2)
                    } else {
                        let t2 = timeit(&mut || {
                            matmul_f32_packed_at(
                                m,
                                k,
                                n,
                                &a,
                                &b,
                                &mut cf,
                                Epi::Plain,
                                KernelCfg::default(),
                            )
                        });
                        let t1 = timeit(&mut || matmul_f32(m, k, n, &a, &b, &mut cf));
                        (t1, t2)
                    };
                    tv.push(t1);
                    tb.push(t2);
                }
                tv.remove(0);
                tb.remove(0);
                let mf = median(tv).as_secs_f64() * 1e3;
                let mk = median(tb).as_secs_f64() * 1e3;
                println!(
                    "- 全派发 {mf:.2} ms（{:.2} GFLOP/s） / 内核 {mk:.2} ms（{:.2} GFLOP/s） ⇒ 派发+并行层 {:+.1}%",
                    gflops(mf),
                    gflops(mk),
                    (mf / mk - 1.0) * 100.0
                );
            }
        }
    }
}
