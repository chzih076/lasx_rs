//! `dsl_map` —— 给 `simd_dsl::Backend` 的**逐元素与轴归约族**提供 LASX 内核（R19-E）。
//!
//! **等价类（逐条；R19-G 起有两类）**：
//! * **逐位**（与 `simd_dsl::kernels::scalar` 参照完全相同的结果）：
//!   `map_binary_*`（逐元素、无归约）、`reduce_sum_cols`（每列一条 lane、按行序累加 ⇒ 次序同参照）；
//! * **容差类**（R19-G，用户 2026-10-17 批准"允许 FMA"）：`map_affine`、`reduce_weighted_sum_cols`
//!   —— 用 **FMA 单指令**（`a·x + b` / `v·w + acc`）替代"先乘后加"两步，**指令数减半**；
//!   代价是舍入与标量参照不同 ⇒ **不再逐位**，实测界由 `dsl_backend.rs` 的测试钉住并打印。
//!   （对照：`reduce_sum_f32` 的参照本身是多累加器 ⇒ 那条也是容差类，见 `dsl_backend` 的说明。）
//!
//! **本文件不依赖 `simd_dsl`**（本仓内核保持独立）⇒ 用一个本地 `BinOp` 枚举做 op 分派。
//! **注意**：`Max`/`Min` 的 NaN 语义（Rust `f32::max`：另一侧是数就返回它）与 LASX `xvfmax` 的
//! IEEE 语义**不同** ⇒ 这两条**不在本文件实现**，由 `LasxBackend` 委托标量参照（见 `dsl_backend.rs`）。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（分层策略：内核模块在函数级统一写 SAFETY，不逐块重复）。
#![allow(clippy::undocumented_unsafe_blocks)]
use crate::arch::lasx;
use std::arch::loongarch64::*;

/// 本地产物：逐元素二元运算种类（与 `simd_dsl::BinaryOp` 的 `Add/Sub/Mul/Div` 一一对应）。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum BinOp {
    /// `x + y`
    Add,
    /// `x − y`
    Sub,
    /// `x · y`
    Mul,
    /// `x / y`
    Div,
}

#[inline]
fn bin_scalar(op: BinOp, a: f32, b: f32) -> f32 {
    match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => a / b,
    }
}

/// `outᵢ = xᵢ op yᵢ`（三切片等长；调用方已校验）。
///
/// # Safety
/// 调用方保证 `x.len() == y.len() == out.len()` 且 `out` 不与 `x`/`y` 重叠。
#[inline]
pub(crate) fn map_binary(x: &[f32], y: &[f32], out: &mut [f32], op: BinOp) {
    let n = x.len();
    // **4 路展开（32 元素/迭代）**：逐元素运算没有归约 ⇒ 展开**不改变任何数值**（按构造逐位安全），
    // 只为摊薄循环开销、给访存更多并行度。动机是实测：单路版本在 1 MiB 档比 LLVM 自动向量化的
    // 可移植档慢约 10%（见 `ops/axpy.rs` 的同款教训）；本文件 W5/W6 的 §5 复测确认了这一点。
    // 展开成宏而不是 `unsafe fn`：edition 2024 下 `unsafe fn` 体内不再隐式 unsafe（会触发
    // `unsafe_op_in_unsafe_fn` 告警），而宏展开进已有的 `unsafe { }` 块里就够了。
    macro_rules! vec_op {
        ($a:expr, $b:expr, $op:expr) => {
            match $op {
                BinOp::Add => lasx_xvfadd_s($a, $b),
                BinOp::Sub => lasx_xvfsub_s($a, $b),
                BinOp::Mul => lasx_xvfmul_s($a, $b),
                BinOp::Div => lasx_xvfdiv_s($a, $b),
            }
        };
    }
    let mut i = 0;
    while i + 32 <= n {
        // SAFETY: `i + 32 <= n` ⇒ 四组 8 元素访存分别落在 [i,i+8)/[i+8,i+16)/[i+16,i+24)/[i+24,i+32)，
        // 全在切片内；非别名由调用方保证。
        unsafe {
            let (pa, pb, po) = (
                x.as_ptr().add(i),
                y.as_ptr().add(i),
                out.as_mut_ptr().add(i),
            );
            for u in [0usize, 8, 16, 24] {
                let a = lasx::load_f32x8(pa.add(u));
                let b = lasx::load_f32x8(pb.add(u));
                lasx::store_f32x8(po.add(u), vec_op!(a, b, op));
            }
        }
        i += 32;
    }
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n` ⇒ 三次 8 元素访存都在切片内。
        unsafe {
            let a = lasx::load_f32x8(x.as_ptr().add(i));
            let b = lasx::load_f32x8(y.as_ptr().add(i));
            lasx::store_f32x8(out.as_mut_ptr().add(i), vec_op!(a, b, op));
        }
        i += 8;
    }
    for j in i..n {
        out[j] = bin_scalar(op, x[j], y[j]);
    }
}

/// `outᵢ = a·xᵢ + b`（**两步运算**，不融 FMA ⇒ 与标量参照逐位）。
///
/// # Safety
/// 调用方保证 `x.len() == out.len()` 且 `out` 不与 `x` 重叠。
#[inline]
pub(crate) fn map_affine(a: f32, b: f32, x: &[f32], out: &mut [f32]) {
    let n = x.len();
    let va = lasx::splat_f32(a);
    let vb = lasx::splat_f32(b);
    // 4 路展开（32 元素/迭代）：逐元素、无归约 ⇒ 展开不改数值；动机同 `map_binary`（访存摊薄）。
    let mut i = 0;
    while i + 32 <= n {
        // SAFETY: `i + 32 <= n` ⇒ 四组访存全在切片内；非别名由调用方保证。
        unsafe {
            let px = x.as_ptr().add(i);
            let po = out.as_mut_ptr().add(i);
            for u in [0usize, 8, 16, 24] {
                let v = lasx::load_f32x8(px.add(u));
                // **R19-G：FMA 单指令**（`a·x + b`）——比"先乘后加"少一条指令；
                // 代价：舍入与标量参照不同 ⇒ 本函数是**容差类**（界见 `dsl_backend.rs` 测试）。
                lasx::store_f32x8(po.add(u), lasx_xvfmadd_s(va, v, vb));
            }
        }
        i += 32;
    }
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n` ⇒ 访存落在切片内；非别名由调用方保证。
        unsafe {
            let v = lasx::load_f32x8(x.as_ptr().add(i));
            lasx::store_f32x8(out.as_mut_ptr().add(i), lasx_xvfmadd_s(va, v, vb));
        }
        i += 8;
    }
    for j in i..n {
        out[j] = a.mul_add(x[j], b);
    }
}

/// **列和**：`out[c] = Σ_r x[r·cols + c]`（**每列一条 lane、按行序累加** ⇒ 与标量参照逐位）。
///
/// P0-3 起含**自动并行分派**：`work = rows·cols ≥ PAR_MIN_REDUCE_ELEMS` 且池线程 > 1 时按
/// **列块**切（每列独立、行序不变 ⇒ **逐位**，见 [`crate::parallel::reduce_sum_cols`]）；
/// 小规模回退单线程（阈值以下**不许变慢**）。
///
/// # Safety
/// 调用方保证 `x.len() == rows·cols`、`out.len() == cols`、`out` 不与 `x` 重叠。
#[inline]
pub(crate) fn reduce_sum_cols(x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    let work = (rows as u128) * (cols as u128);
    let pool = crate::pool::global();
    if work >= PAR_MIN_REDUCE_ELEMS && pool.threads() > 1 && cols >= 8 {
        crate::parallel::reduce_sum_cols(pool, x, rows, cols, out);
        return;
    }
    reduce_sum_cols_range(x, rows, cols, 0, cols, out);
}

/// 自动并行阈值（元素数；与 GEMM 的 `PAR_MIN_WORK` 同思路，按 `examples/row_par_ab.rs` 校准）。
///
/// 实测（12 线程、loadavg≈1.3）：33.5M 元素 **1.007×（中性、无退化）**，67M **4.04×/5.17×**；
/// 4.2M 及以下 0.98–1.04×（无效）⇒ 阈值取 33.5M：**33.5M 以下回退 1t（不许变慢）**，
/// 大矩阵（列足够多）才并行。
pub(crate) const PAR_MIN_REDUCE_ELEMS: u128 = 1 << 25; // 33.5M 元素

/// 列和的**列区间**版本：只算列 `[c0, c1)`，写入 `out_chunk`（长度 `c1-c0`）。
///
/// 每列仍按行序累加 ⇒ 与整段调用**逐位一致**（并行切分的基础）。
pub(crate) fn reduce_sum_cols_range(
    x: &[f32],
    rows: usize,
    cols: usize,
    c0: usize,
    c1: usize,
    out_chunk: &mut [f32],
) {
    let mut c = c0;
    while c + 8 <= c1 {
        let mut acc = lasx::zero_f32x8();
        for r in 0..rows {
            let off = r * cols + c;
            // SAFETY: `c + 8 <= c1 <= cols` 且 `r < rows` ⇒ `off + 8 <= rows*cols`，落在 `x` 内。
            unsafe {
                acc = lasx_xvfadd_s(acc, lasx::load_f32x8(x.as_ptr().add(off)));
            }
        }
        // SAFETY: `c + 8 <= c1`，`out_chunk` 长度 `c1-c0` ⇒ 落在其中。
        unsafe { lasx::store_f32x8(out_chunk.as_mut_ptr().add(c - c0), acc) };
        c += 8;
    }
    // 尾部列：与参照同形（f32 单累加器、按行序）
    for cc in c..c1 {
        let mut acc = 0.0f32;
        for r in 0..rows {
            acc += x[r * cols + cc];
        }
        out_chunk[cc - c0] = acc;
    }
}

/// `out[c] = Σ_r x[r·cols + c]·w[r]`（`out + (x·w)`：**先乘后加**，与标量参照逐位）。
///
/// P0-3 起含**自动并行分派**（同 [`reduce_sum_cols`]：按列块切，每列行序不变 ⇒ **逐位**）。
///
/// # Safety
/// 调用方保证 `x.len() == rows·cols`、`w.len() >= rows`、`out.len() == cols`、`out` 不与 `x` 重叠。
#[inline]
pub(crate) fn reduce_weighted_sum_cols(
    x: &[f32],
    w: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    let work = (rows as u128) * (cols as u128);
    let pool = crate::pool::global();
    if work >= PAR_MIN_REDUCE_ELEMS && pool.threads() > 1 && cols >= 8 {
        crate::parallel::reduce_weighted_sum_cols(pool, x, w, rows, cols, out);
        return;
    }
    reduce_weighted_sum_cols_range(x, w, rows, cols, 0, cols, out);
}

/// 带权列和的**列区间**版本（逐位切分基础）。
pub(crate) fn reduce_weighted_sum_cols_range(
    x: &[f32],
    w: &[f32],
    rows: usize,
    cols: usize,
    c0: usize,
    c1: usize,
    out_chunk: &mut [f32],
) {
    let mut c = c0;
    while c + 8 <= c1 {
        let mut acc = lasx::zero_f32x8();
        for (r, wv) in w.iter().enumerate().take(rows) {
            let off = r * cols + c;
            // SAFETY: 同 `reduce_sum_cols_range`；`*wv` 是标量广播。
            unsafe {
                let v = lasx::load_f32x8(x.as_ptr().add(off));
                // **R19-G：FMA**（`v·w + acc`）——省一条指令；舍入与"先乘后加"不同 ⇒ **容差类**。
                acc = lasx_xvfmadd_s(v, lasx::splat_f32(*wv), acc);
            }
        }
        // SAFETY: `c + 8 <= c1`，`out_chunk` 长度 `c1-c0`。
        unsafe { lasx::store_f32x8(out_chunk.as_mut_ptr().add(c - c0), acc) };
        c += 8;
    }
    for cc in c..c1 {
        let mut acc = 0.0f32;
        for (r, wv) in w.iter().enumerate().take(rows) {
            acc = x[r * cols + cc].mul_add(*wv, acc);
        }
        out_chunk[cc - c0] = acc;
    }
}

/// `Σ x[i]`：直接复用本仓既有的 LASX 归约（`ops::sum`）。
#[inline]
pub(crate) fn reduce_sum(x: &[f32]) -> f32 {
    crate::ops::sum::sum(x)
}

// ================= P4.2：行和（`reduce_sum_rows_f32`）的 LASX override + 自动并行 =================
//
// **等价类 = 逐位**（与 `simd_dsl::kernels::scalar::reduce_sum_rows_f32` 的默认实现逐位相等）：
// 默认实现是"8 个 per-lane 累加器（lane `l` 加下标 `c ≡ l (mod 8)` 的元素）→ **顺序 fold** 8 lane
// → 顺序补尾"。这里用 LASX 的 8-lane 向量加法做**同一件事**：向量加法 = 8 个独立的 per-lane 加法
// （次序相同），水平归约**不**用两两树，而是把 8 lane 落回标量后**顺序 fold**（与参照同序），
// 尾部再按 `c` 递增顺序补加 ⇒ 每条输出的运算次序完全一致 ⇒ `to_bits()` 相等。
//
// **并行**：行与行之间独立（每行的归约次序不变）⇒ 行块切分仍是**逐位类**；只改"哪一行先算"。

/// 行和的**逐行 kernel**：`out[r] = Σ_c x[r·cols + c]`（**逐位**，见本段文档）。
pub(crate) fn reduce_sum_rows_rows(x: &[f32], cols: usize, out: &mut [f32], rows_local: usize) {
    for r in 0..rows_local {
        out[r] = row_sum_bitwise(&x[r * cols..(r + 1) * cols]);
    }
}

/// 单行和：8-lane 向量累加 + **顺序 fold** + 顺序补尾（与标量参照逐位）。
#[inline]
fn row_sum_bitwise(row: &[f32]) -> f32 {
    let (chunks, tail) = row.as_chunks::<8>();
    let mut acc = lasx::zero_f32x8();
    for c in chunks {
        // SAFETY: 每个 chunk 恰有 8 个 f32。
        unsafe {
            acc = lasx_xvfadd_s(acc, lasx::load_f32x8(c.as_ptr()));
        }
    }
    let mut buf = [0.0f32; 8];
    // SAFETY: `buf` 有 8 个元素，store 8 lane 合法。
    unsafe { lasx::store_f32x8(buf.as_mut_ptr(), acc) };
    // **顺序 fold**（不要两两树）：与标量参照 `acc.iter().copied().fold(0.0, |s, v| s + v)` 同序。
    let mut s = 0.0f32;
    for a in buf {
        s += a;
    }
    for v in tail {
        s += v;
    }
    s
}

/// 行和的自动并行阈值（元素数；由 `examples/row_par_ab.rs` 实测定）。
///
/// **实测**（12 线程、loadavg≈2.7–2.9、同进程 1t vs auto、6 对丢首对取中位；判据 1t/auto ≥1.10）：
/// 8K–3.1M **0.99–1.01×（中性，不进并行是对的）**；**4.2M 8.55×**、16.7M 2.65×、33.5M 2.30×、
/// 67M 2.17×。拐点在"数据超出 L3（≈16 MB）"：3.1M 元素 = 12.6 MB 仍在 L3，串行已打满带宽；
/// 4.2M = 16.8 MB 起并行才有效 ⇒ 阈值取 **4M 元素**（`1<<22`）。阈值以下原地串行
/// （**不许变慢**：8K–3.1M 实测 0.99–1.01× 是与 1t 同一路径/或并行无收益）。
pub(crate) const PAR_MIN_SUM_ROWS_ELEMS: u128 = 1 << 22; // 4M 元素

/// 行和入口（`LasxBackend::reduce_sum_rows_f32` 的落点；**自动并行**）。
///
/// `work = rows·cols ≥ PAR_MIN_SUM_ROWS_ELEMS` 且池线程 > 1 时按**行块**并行
/// （`parallel::reduce_sum_rows`）；否则串行。线程数由 `pool::global()` 决定
/// （`init_global(n)` 可控；默认 `WorkerPool::auto()`）。
pub(crate) fn reduce_sum_rows(x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    let work = (rows as u128) * (cols as u128);
    let pool = crate::pool::global();
    if work >= PAR_MIN_SUM_ROWS_ELEMS && pool.threads() > 1 && rows >= 2 {
        crate::parallel::reduce_sum_rows(pool, x, rows, cols, out);
        return;
    }
    reduce_sum_rows_rows(x, cols, out, rows);
}
