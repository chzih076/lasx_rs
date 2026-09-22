//! `lasx_matmul` —— f32 矩阵乘 `C[m×n] = A[m×k]·B[k×n]`（行主序，列方向向量化）。
//!
//! 微内核是 i-k-j 形式：A 的行元素广播、B 沿列方向连续流式读取，**不做 B 转置、
//! 不做跨 lane 水平归约**（这两点是 v1 慢于朴素三重循环的原因）。
//!
//! v3 叠加**行分块**：一次算 4 行 × 32 列（16 个 8 通道累加器），同一段 B 被 4
//! 行复用，B 的重复读取量降到 1/4。
//!
//! v4 试过两级 L2 分块与 6 行微块，**都被实测否掉了**：4×32 已经是这台机器上
//! 16 个累加器能做到的最优形状，详见 `docs/perf-report.md` §16。
//!
//! v6 = **k 分块**：打包条带按 k 主序存放，所以一段 k 就是一段连续内存。把 k 切成
//! `K_CHUNK` 步的块、**k 块在外层、行块在内层**，就能让每个 k 块的 B（32 KB）被所有
//! 行块反复命中，并与 A 的 4 行、C 的 4×32 一起待在 L1（64 KB）里。实测见 §21：
//! 512³ +13~26%、1024³ +7~26%、2048³ +91%，256³ 持平（k=256 的条带本来就装得进 L1）。
//! 代价是 C 每块读写一次（k=1024 时多 3 次），相对算力可忽略。
//!
//! 数值行为不变：每个输出元素仍是**沿 k 的单个 f32 累加器**，累加次序与 v2 完全一致
//! （k 分块只是把同一个 f32 部分和分段落到 C 再读回，落盘/读回是精确的）。
//!
//! **LASX-only**：没有降级分支，无 LASX 的 CPU 上会执行 LASX 指令（见手册 Caveats）。

use crate::arch::lasx;

/// 列块宽度：B 的面板 `k × COL_BLOCK` 要能留在 L2（3 MiB/核）里被整轮行扫描复用。
/// 实测 128 最优（32/64 接近，256 起明显变慢，见 perf-report §18）。
pub(crate) const COL_BLOCK: usize = 128;

/// 走"列块在外"次序的 `m` 上限。`m/4` 是流式次序下 B 被读的遍数：m=128 时 32 遍，
/// 已经把 B 的读量放大 32 倍；再大一档（方阵）实测两种次序持平，故保守取 128。
const COL_ORDER_MAX_M: usize = 128;

/// 走"列块在外"次序的 `n` 下限：n 太小时 A 的重读代价相对更高，不值得切换。
const COL_ORDER_MIN_N: usize = 1024;

/// 打包路径的 `m` 下限：打包本身要走两遍 B（读+写），只有当行块足够多
/// （流式次序会因此把 B 读很多遍）时才摊得掉。按行切并行的**每线程**只有 m/线程数 行，
/// 所以这个下限同时保护了并行路径——否则每个线程都会把整块 B 打包一遍。
const PACK_MIN_M: usize = 32;

/// 打包路径的 `k` 下限。§23 重新测过：配合 k 分块，k=64 起打包就大胜"列块/流式"
/// （`64×96×8192` 21 → 45 GFLOP/s、`256×64×4096` 23 → 58），所以从 192 降到 64。
const PACK_MIN_K: usize = 64;

/// 极宽 `n` 的例外：当 `k` 很小而 `n` 极大时，列块路径只复制一遍 B 面板、且面板只有
/// `k×128×4`，比打包（要整体复制一遍 B = `k×n×4` 字节）更划算。实测 `100×64×19147`
/// 列块 39.7 vs 打包 33.5 GFLOP/s；而 `n ≤ 16384` 时打包仍然领先（`64×96×8192`、
/// `128×128×2048` 分别是 +80%、+71%）。所以只在这个交叉区间保留列块路径。
const WIDE_N_FOR_COLS: usize = 16384;

/// `k` 大于等于这个值时，打包**无条件**胜出（与 `n` 多大无关）：§19/§23 的交叉点。
const K_ALWAYS_PACK: usize = 192;

/// 打包路径的工作量下限（`m×k×n`，即乘加次数）：128³（4.2 MFLOP）打包 0.85×，
/// 192³（14 MFLOP）起转为正收益。
const PACK_MIN_WORK: u128 = 8_000_000;

/// k 方向分块的"L1 预算"与块长。打包条带按 k 主序存放，所以一个 k 块就是一段连续内存。
///
/// 动机（实测，`512×k×512`）：条带 24–48 KB 时 61 GF/s，条带 64 KB（正好等于 L1 的
/// 64 KB）掉到 54.7，96–128 KB 掉到 47–49。k=1024 不分块时条带就是 128 KB。
/// 所以条带本身超过 `L1_BUDGET` 才分块，块长取 `K_CHUNK`（32 KB）；
/// k ≤ 384 时条带 ≤ 48 KB，直接一整块跑，避免白白多读写 C。
const L1_BUDGET: usize = 48 * 1024;
const K_CHUNK: usize = 256;
use std::arch::loongarch64::*;

/// f32 矩阵乘 `C[m×n] = A[m×k]·B[k×n]`（行主序）。
///
/// 循环顺序是"每个 4 行块把 B 扫一遍"——**这个顺序是实测选出来的**：
/// 把它改成两级 L2 分块（列块在外、行块在内）在多核下慢 8–12%，单线程也无收益，
/// 因为 4×32 微块本来就让 B 被复用 4 次，分块只是把流量从 L3 挪到 L2、体量不变。
/// 详见 `docs/perf-report.md` §16。
#[inline]
pub(crate) fn matmul_f32(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    if m == 0 || n == 0 {
        return;
    }
    // 路径选择（三条都**逐位一致**，只是访存次序不同；数字见 perf-report §18/§19）：
    //   1) 打包 B 面板：让微内核的 k 循环顺序读。交叉点实测 work ≈ 8 MFLOP；k 下限
    //      在 §21 的 k 分块落地后从 192 降到 64（见 `PACK_MIN_K`），只有"小 k + 极宽 n"
    //      例外仍走列块（见 `WIDE_N_FOR_COLS`）。
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
        matmul_f32_packed(m, k, n, a, b, c);
        return;
    }
    if m <= COL_ORDER_MAX_M && n >= COL_ORDER_MIN_N {
        matmul_f32_cols_block(m, k, n, 0, n, a, b, c, COL_BLOCK);
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

/// 列区间入口：只算 `C[:, j0..j1)`（`b`/`c` 的行跨距仍是完整的 `n`），列块宽度可调。
///
/// 与 [`matmul_f32`] **逐位一致**（每个输出元素的累加次序相同），区别只在循环嵌套：
/// 这里**列块在外、行扫在内**，于是 B 的一段列面板（`k × col_block`）在整轮行扫描期间
/// 留在 L1/L2 里被复用，**B 只从内存流一遍**。
///
/// 动机：`m` 小、`n` 大时（100×448 × 448×19147），[`matmul_f32`] 每个 4 行块都要把
/// B 扫一遍 ⇒ 100 行读 25 遍 = 858 MB，且每次只读 128 B 就跳 `n×4` 字节，实测有效带宽
/// 只有 ~5 GB/s。改成列块在外后同一形状 **6.6 → 38.6 GFLOP/s**（§18）。
///
/// # 为什么要 32 对齐
/// 主内核把 `[0, n/32*32)` 交给 4×32 微内核、其余交给 8 宽 + 标量的尾路径；尾路径里的
/// 标量段用的是**普通乘加**（不是 FMA），所以只有 32 对齐的切分才能保证每一列都落在与
/// 主内核相同的路径上、结果逐位相同。
///
/// # Panics
/// `debug` 构建下要求 `j0 % 32 == 0` 且 `j1 <= n`。
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
                tile4x32_f32(i, j, k, n, a, b, c);
                j += 32;
            }
            i += 4;
        }
        // 不足 4 行的尾部行
        while i < m {
            let a_row = &a[i * k..(i + 1) * k];
            let c_row = &mut c[i * n..(i + 1) * n];
            row_tail_range_f32(a_row, c_row, b, k, n, jc, jce);
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
            row_tail_range_f32(a_row, c_row, b, k, n, from, j1);
        }
    }
}

// B 面板打包用的线程局部缓冲（跨调用复用，避免每次 malloc 1.8 MB）。
thread_local! {
    static PACK_BUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_f32_packed(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    if m == 0 || n == 0 {
        return;
    }
    let m4 = m / 4 * 4;
    let n32 = n / 32 * 32;
    let strips_total = n32 / 32;
    // 一次打包的列数：面板 `k×nc×4` 要留在 L2（3 MiB/核）里，预算取 2 MiB
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
            // 宏内核：**k 分块在外、行块在内**。这样每个 k 块打包后的 B
            //（K_CHUNK×32×4 = 24 KB）会被所有行块反复命中，并与 A 的 4 行、C 的 4×32
            // 一起待在 L1 里。代价是 C 每块读写一次（k=1024 时多 5 次），相对算力可忽略。
            let kb_size = if k * 32 * 4 <= L1_BUDGET { k } else { K_CHUNK };
            let mut kb0 = 0;
            while kb0 < k {
                let kb = kb_size.min(k - kb0);
                let first = kb0 == 0;
                for s in s0..s1 {
                    let base = (s - s0) * k * 32 + kb0 * 32;
                    let strip = &packed[base..base + kb * 32];
                    let j = s * 32;
                    let mut i = 0;
                    while i < m4 {
                        let ct = &mut c[i * n..(i + 4) * n];
                        tile4x32_chunk(
                            kb,
                            first,
                            strip,
                            &a[i * k + kb0..i * k + kb0 + kb],
                            &a[(i + 1) * k + kb0..(i + 1) * k + kb0 + kb],
                            &a[(i + 2) * k + kb0..(i + 2) * k + kb0 + kb],
                            &a[(i + 3) * k + kb0..(i + 3) * k + kb0 + kb],
                            ct,
                            n,
                            j,
                        );
                        i += 4;
                    }
                    // 不足 4 行的尾部行
                    while i < m {
                        let c_row = &mut c[i * n..(i + 1) * n];
                        row_tail_packed(&a[i * k + kb0..i * k + kb0 + kb], strip, c_row, first, j);
                        i += 1;
                    }
                }
                kb0 += kb;
            }
            s0 = s1;
        }
    });
    // 列尾 [n32, n)：走原来的跨距尾路径（列数 < 32，代价可忽略）
    if n > n32 {
        for i in 0..m {
            let a_row = &a[i * k..(i + 1) * k];
            let c_row = &mut c[i * n..(i + 1) * n];
            row_tail_range_f32(a_row, c_row, b, k, n, n32, n);
        }
    }
}

/// 把 `B[p][jb..jb+nc]` 打成 `packed[(s*k + p)*32 + r]`。
///
/// 外层走 `p`：这样读 B 是**顺序**的（每行 `nc×4` 字节连续），写是 128 B 连续小块。
#[inline]
fn pack_b(k: usize, n: usize, jb: usize, nc: usize, b: &[f32], packed: &mut [f32]) {
    let strips = nc / 32;
    for p in 0..k {
        let row = &b[p * n + jb..p * n + jb + nc];
        for s in 0..strips {
            let dst = (s * k + p) * 32;
            packed[dst..dst + 32].copy_from_slice(&row[s * 32..s * 32 + 32]);
        }
    }
}

/// 微内核：4 行 × 32 列 × **kb 个 k 步**，B 从打包缓冲连续读。
///
/// `first` 为真表示这是该 (行块, 列条带) 的第一个 k 块，累加器从 0 起；否则先把 C 里
/// 已有的部分和读进累加器再继续加——**结合次序与不分块时完全一致**（都是按 p 递增累加
/// f32 部分和），所以结果逐位不变。
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

/// 单行 × 32 列，B 从打包缓冲读（不足 4 行的尾部行）。
#[inline]
fn row_tail_packed(a_row: &[f32], strip: &[f32], c_row: &mut [f32], first: bool, j: usize) {
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
    unsafe {
        lasx::store_f32x8(c_row.as_mut_ptr().add(j), acc0);
        lasx::store_f32x8(c_row.as_mut_ptr().add(j + 8), acc1);
        lasx::store_f32x8(c_row.as_mut_ptr().add(j + 16), acc2);
        lasx::store_f32x8(c_row.as_mut_ptr().add(j + 24), acc3);
    }
}

/// 手写 8 行 × 16 列打包微内核（16 个累加器，2 个 B 向量 + 8 个 A 广播/轮）。
///
/// 相比 4×32：B 的 L2 复用遍数从 25 降到 12.5（`m/MR`），且打包条带只有 `k×16×4`
/// = 28 KB，能连带 A 行一起待在 L1（4×32 的条带是 57 KB，正好卡在 L1 边缘）。
#[inline]
#[allow(clippy::too_many_arguments)]
fn pack_strips(k: usize) -> usize {
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
            row_tail_range_f32(a_row, c_row, b, k, n, n32, n);
        }
        i += 4;
    }
    while i < m {
        let a_row = &a[i * k..(i + 1) * k];
        let c_row = &mut c[i * n..(i + 1) * n];
        cols32_f32(a_row, c_row, b, n);
        row_tail_range_f32(a_row, c_row, b, k, n, n32, n);
        i += 1;
    }
}

/// 微内核：一次算 4 行 × 32 列（16 个累加器），B 的 4 个向量被 4 行共享。
///
/// `j + 32 <= n` 由调用方保证（只走 `[0, n/32*32)`）；列尾另由 [`row_tail_f32`] 处理。
/// `p` 既用于索引 4 行 A 又用于计算 B 的地址，故保留下标循环。
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

/// 单行主循环：32 列一块（4 个累加器）。
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

/// 单行的列尾：`[j0, n)`，先 8 列块再标量。
#[inline]
fn row_tail_f32(a_row: &[f32], c_row: &mut [f32], b: &[f32], k: usize, n: usize, j0: usize) {
    row_tail_range_f32(a_row, c_row, b, k, n, j0, n);
}

/// 单行的列尾 `[j0, j1)`：先 8 列块再标量。`n` 始终是 B/C 的**行跨距**。
#[inline]
fn row_tail_range_f32(
    a_row: &[f32],
    c_row: &mut [f32],
    b: &[f32],
    k: usize,
    n: usize,
    j0: usize,
    j1: usize,
) {
    let mut j = j0;
    while j + 8 <= j1 {
        let mut acc = lasx::zero_f32x8();
        for (p, &a_p) in a_row.iter().enumerate() {
            let va = lasx::splat_f32(a_p);
            let vb = unsafe { lasx::load_f32x8(b.as_ptr().add(p * n + j)) };
            acc = unsafe { lasx_xvfmadd_s(vb, va, acc) };
        }
        unsafe { lasx::store_f32x8(c_row.as_mut_ptr().add(j), acc) };
        j += 8;
    }
    for jj in j..j1 {
        let mut s = 0f32;
        for p in 0..k {
            s += a_row[p] * b[p * n + jj];
        }
        c_row[jj] = s;
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
    use crate::ops::testutil::{reference, rel, Lcg};

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
                matmul_f32_cols_block(m, k, n, j0, j1, &a, &b, &mut got, COL_BLOCK);
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
            matmul_f32_cols_block(m, k, n, 0, n, &a, &b, &mut cols, COL_BLOCK);
            let mut pack = vec![f32::NAN; m * n];
            matmul_f32_packed(m, k, n, &a, &b, &mut pack);
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
}
