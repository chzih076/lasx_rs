//! `lasx_matmul_f64` —— f64 矩阵乘 `C[m×n] = A[m×k]·B[k×n]`（行主序，列方向向量化）。
//!
//! 与 [`crate::ops::matmul`] 同构，只是向量宽度为 4×f64、列块取 16：
//! 一次算 4 行 × 16 列（16 个 4 通道累加器），同一段 B 被 4 行复用。
//! 不做 B 转置、不做跨 lane 水平归约。
//!
//! v2：把 f32 侧验证过的**打包 + k 分块**照搬过来（见 `docs/dev.md` §8.2）。
//! 原路径（`rows4_f64`）每次 k 步都要按 `n` 跨距读 B，B 被 `m/4` 遍重复读；
//! 打包后 B 变成 16 列一条带的连续内存，再把 k 切成 `K_CHUNK` 步的块、**k 块在外、
//! 行块在内**，让每个 k 块的条带与 A 的 4 行、C 的 4×16 一起待在 L1 里。
//! 小形状仍走原来的流式路径（见 `PACK_MIN_*`）。
//!
//! 数值行为不变：每个输出元素仍是沿 k 的单个 f64 累加器，累加次序与 v2 一致
//! （k 分块只是把同一个 f64 部分和分段落到 C 再读回，落盘/读回是精确的）。
//!
//! **LASX-only**：没有降级分支，无 LASX 的 CPU 上会执行 LASX 指令（见手册 Caveats）。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（策略见 `docs/dev.md` §17）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX/LSX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]

use crate::arch::lasx;
use std::arch::loongarch64::*;

/// 打包路径的 `m` 下限：打包要走两遍 B（读+写），只有行块足够多时才摊得掉。
pub(crate) const PACK_MIN_M: usize = 32;
/// 打包路径的 `k` 下限。f64 侧**没有**列块路径可退，所以配合 k 分块后下限可以比 f32
/// 更激进地取 64（重标定数据见 `docs/dev.md` §8.4：`256×64×4096` 11.2 → 28.6–30.2、
/// `128×128×2048` 11.1 → 27.7）；小的方阵由 `PACK_MIN_WORK` 挡住
/// （128³ 只有 2.1 M < 8 M，仍走流式）。
pub(crate) const PACK_MIN_K: usize = 64;
/// 打包路径的工作量下限（`m×k×n` 乘加次数）。
pub(crate) const PACK_MIN_WORK: u128 = 8_000_000;

/// k 分块的 L1 预算与块长：条带是 `k×16×8` 字节，超过 `L1_BUDGET` 就分块。
/// 数值来自 f32 侧的条带扫描（24–48 KB 最优、正好等于 L1 的 64 KB 时开始掉，
/// 见 `docs/dev.md` §8.3），f64 条带同口径。
pub(crate) const L1_BUDGET: usize = 48 * 1024;
pub(crate) const K_CHUNK: usize = 256;

/// 一次打包的条带数：面板 `k×nc×8` 要留在 L2（3 MiB/核）里，预算取 2 MiB。
pub(crate) fn pack_strips(k: usize) -> usize {
    let per_strip = (k * 16 * 8).max(1);
    (2 * 1024 * 1024 / per_strip).clamp(1, 256)
}

thread_local! {
    /// 打包缓冲：按 `(条带, k)` 主序存放，条带内是 16 列一段。
    static PACK_BUF: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// f64 矩阵乘 `C[m×n] = A[m×k]·B[k×n]`（行主序）。
#[inline]
pub(crate) fn matmul_f64(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    if m == 0 || n == 0 {
        return;
    }
    if m >= PACK_MIN_M && k >= PACK_MIN_K && n >= 16 && (m * k * n) as u128 >= PACK_MIN_WORK {
        matmul_f64_packed(m, k, n, a, b, c);
        return;
    }
    matmul_f64_stream(m, k, n, a, b, c);
}

/// 原始的流式次序（`docs/dev.md` §8.5 的 A/B 基准，也是小形状的生产路径）。
pub(crate) fn matmul_f64_stream(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    let mut i = 0;
    while i + 4 <= m {
        rows4_f64(i, k, n, a, b, c);
        i += 4;
    }
    while i < m {
        let a_row = &a[i * k..(i + 1) * k];
        let c_row = &mut c[i * n..(i + 1) * n];
        cols16_f64(a_row, c_row, b, n);
        row_tail_f64(a_row, c_row, b, k, n, n / 16 * 16);
        i += 1;
    }
}

/// 一次算 4 行 × 16 列：16 个累加器，B 的 4 个向量被 4 行共享。
///
/// `p` 既用于索引 4 行 A 又用于计算 B 的地址，故保留下标循环。
#[inline]
#[allow(clippy::needless_range_loop)]
fn rows4_f64(i0: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    let a0 = &a[i0 * k..(i0 + 1) * k];
    let a1 = &a[(i0 + 1) * k..(i0 + 2) * k];
    let a2 = &a[(i0 + 2) * k..(i0 + 3) * k];
    let a3 = &a[(i0 + 3) * k..(i0 + 4) * k];

    let mut rows = c[i0 * n..(i0 + 4) * n].chunks_mut(n);
    let c0 = rows.next().expect("4 行");
    let c1 = rows.next().expect("4 行");
    let c2 = rows.next().expect("4 行");
    let c3 = rows.next().expect("4 行");

    let nb = n / 16 * 16;
    let mut j = 0;
    while j + 16 <= n {
        let (mut r0a, mut r0b, mut r0c, mut r0d) = z4();
        let (mut r1a, mut r1b, mut r1c, mut r1d) = z4();
        let (mut r2a, mut r2b, mut r2c, mut r2d) = z4();
        let (mut r3a, mut r3b, mut r3c, mut r3d) = z4();
        for p in 0..k {
            let base = p * n + j;
            let vb0 = unsafe { lasx::load_f64x4(b.as_ptr().add(base)) };
            let vb1 = unsafe { lasx::load_f64x4(b.as_ptr().add(base + 4)) };
            let vb2 = unsafe { lasx::load_f64x4(b.as_ptr().add(base + 8)) };
            let vb3 = unsafe { lasx::load_f64x4(b.as_ptr().add(base + 12)) };
            let s0 = lasx::splat_f64(a0[p]);
            let s1 = lasx::splat_f64(a1[p]);
            let s2 = lasx::splat_f64(a2[p]);
            let s3 = lasx::splat_f64(a3[p]);
            unsafe {
                r0a = lasx_xvfmadd_d(vb0, s0, r0a);
                r0b = lasx_xvfmadd_d(vb1, s0, r0b);
                r0c = lasx_xvfmadd_d(vb2, s0, r0c);
                r0d = lasx_xvfmadd_d(vb3, s0, r0d);
                r1a = lasx_xvfmadd_d(vb0, s1, r1a);
                r1b = lasx_xvfmadd_d(vb1, s1, r1b);
                r1c = lasx_xvfmadd_d(vb2, s1, r1c);
                r1d = lasx_xvfmadd_d(vb3, s1, r1d);
                r2a = lasx_xvfmadd_d(vb0, s2, r2a);
                r2b = lasx_xvfmadd_d(vb1, s2, r2b);
                r2c = lasx_xvfmadd_d(vb2, s2, r2c);
                r2d = lasx_xvfmadd_d(vb3, s2, r2d);
                r3a = lasx_xvfmadd_d(vb0, s3, r3a);
                r3b = lasx_xvfmadd_d(vb1, s3, r3b);
                r3c = lasx_xvfmadd_d(vb2, s3, r3c);
                r3d = lasx_xvfmadd_d(vb3, s3, r3d);
            }
        }
        unsafe {
            lasx::store_f64x4(c0.as_mut_ptr().add(j), r0a);
            lasx::store_f64x4(c0.as_mut_ptr().add(j + 4), r0b);
            lasx::store_f64x4(c0.as_mut_ptr().add(j + 8), r0c);
            lasx::store_f64x4(c0.as_mut_ptr().add(j + 12), r0d);
            lasx::store_f64x4(c1.as_mut_ptr().add(j), r1a);
            lasx::store_f64x4(c1.as_mut_ptr().add(j + 4), r1b);
            lasx::store_f64x4(c1.as_mut_ptr().add(j + 8), r1c);
            lasx::store_f64x4(c1.as_mut_ptr().add(j + 12), r1d);
            lasx::store_f64x4(c2.as_mut_ptr().add(j), r2a);
            lasx::store_f64x4(c2.as_mut_ptr().add(j + 4), r2b);
            lasx::store_f64x4(c2.as_mut_ptr().add(j + 8), r2c);
            lasx::store_f64x4(c2.as_mut_ptr().add(j + 12), r2d);
            lasx::store_f64x4(c3.as_mut_ptr().add(j), r3a);
            lasx::store_f64x4(c3.as_mut_ptr().add(j + 4), r3b);
            lasx::store_f64x4(c3.as_mut_ptr().add(j + 8), r3c);
            lasx::store_f64x4(c3.as_mut_ptr().add(j + 12), r3d);
        }
        j += 16;
    }
    row_tail_f64(a0, c0, b, k, n, nb);
    row_tail_f64(a1, c1, b, k, n, nb);
    row_tail_f64(a2, c2, b, k, n, nb);
    row_tail_f64(a3, c3, b, k, n, nb);
}

/// 单行主循环：16 列一块（4 个累加器）。
#[inline]
fn cols16_f64(a_row: &[f64], c_row: &mut [f64], b: &[f64], n: usize) {
    let mut j = 0;
    while j + 16 <= n {
        let (mut acc0, mut acc1, mut acc2, mut acc3) = z4();
        for (p, &a_p) in a_row.iter().enumerate() {
            let va = lasx::splat_f64(a_p);
            let base = p * n + j;
            let b0 = unsafe { lasx::load_f64x4(b.as_ptr().add(base)) };
            let b1 = unsafe { lasx::load_f64x4(b.as_ptr().add(base + 4)) };
            let b2 = unsafe { lasx::load_f64x4(b.as_ptr().add(base + 8)) };
            let b3 = unsafe { lasx::load_f64x4(b.as_ptr().add(base + 12)) };
            unsafe {
                acc0 = lasx_xvfmadd_d(b0, va, acc0);
                acc1 = lasx_xvfmadd_d(b1, va, acc1);
                acc2 = lasx_xvfmadd_d(b2, va, acc2);
                acc3 = lasx_xvfmadd_d(b3, va, acc3);
            }
        }
        unsafe {
            lasx::store_f64x4(c_row.as_mut_ptr().add(j), acc0);
            lasx::store_f64x4(c_row.as_mut_ptr().add(j + 4), acc1);
            lasx::store_f64x4(c_row.as_mut_ptr().add(j + 8), acc2);
            lasx::store_f64x4(c_row.as_mut_ptr().add(j + 12), acc3);
        }
        j += 16;
    }
}

/// 单行的列尾：`[j0, n)`，先 4 列块再标量。
#[inline]
pub(crate) fn row_tail_f64(
    a_row: &[f64],
    c_row: &mut [f64],
    b: &[f64],
    k: usize,
    n: usize,
    j0: usize,
) {
    let mut j = j0;
    while j + 4 <= n {
        let mut acc = lasx::zero_f64x4();
        for (p, &a_p) in a_row.iter().enumerate() {
            let va = lasx::splat_f64(a_p);
            let vb = unsafe { lasx::load_f64x4(b.as_ptr().add(p * n + j)) };
            acc = unsafe { lasx_xvfmadd_d(vb, va, acc) };
        }
        unsafe { lasx::store_f64x4(c_row.as_mut_ptr().add(j), acc) };
        j += 4;
    }
    for jj in j..n {
        let mut s = 0f64;
        for p in 0..k {
            s += a_row[p] * b[p * n + jj];
        }
        c_row[jj] = s;
    }
}

/// 打包 + k 分块的 f64 路径（与 f32 侧 [`crate::ops::matmul`] 的 v6 同构）。
pub(crate) fn matmul_f64_packed(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    let m4 = m / 4 * 4;
    let n16 = n / 16 * 16;
    let strips_total = n16 / 16;
    let nc_strips = pack_strips(k);

    PACK_BUF.with(|buf| {
        let mut packed = buf.borrow_mut();
        packed.resize(k * nc_strips * 16, 0.0);
        let mut s0 = 0;
        while s0 < strips_total {
            let s1 = (s0 + nc_strips).min(strips_total);
            let jb = s0 * 16;
            let nc = (s1 - s0) * 16;
            pack_b_f64(k, n, jb, nc, b, &mut packed);
            // 面板打包好后交给公共入口（并行层复用同一个入口 ⇒ 打包只做一次、跨线程共享）
            matmul_f64_packed_rows(k, n, jb, nc, &packed, a, c, m4);
            s0 = s1;
        }
    });
    // 列尾 [n16, n)：走原来的跨距尾路径（列数 < 16，代价可忽略）
    if n > n16 {
        for i in 0..m {
            let a_row = &a[i * k..(i + 1) * k];
            let c_row = &mut c[i * n..(i + 1) * n];
            row_tail_f64(a_row, c_row, b, k, n, n16);
        }
    }
}

/// 用**已打包**的列面板计算 `a_rows`/`c_rows` 对应的那些行（并行层复用的入口）。
///
/// `packed` 是 [`pack_b_f64`] 的产物（`k × nc`，16 列条带主序）；`a_rows.len()/k` 是本块行数。
/// 每行的列尾 `[n16, n)` 由调用方用 [`row_tail_f64`] 补。顺序版的 [`matmul_f64_packed`]
/// 走的是同一个函数，所以并行与单线程**逐位一致**。
#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_f64_packed_rows(
    k: usize,
    n: usize,
    jb: usize,
    nc: usize,
    packed: &[f64],
    a_rows: &[f64],
    c_rows: &mut [f64],
    m4: usize,
) {
    let m = a_rows.len().checked_div(k).unwrap_or(0);
    let strips = nc / 16;
    // k 块在外、行块在内：每个 k 块的条带被所有行块反复命中，留在 L1
    let kb_size = if k * 16 * 8 <= L1_BUDGET { k } else { K_CHUNK };
    let mut kb0 = 0;
    while kb0 < k {
        let kb = kb_size.min(k - kb0);
        let first = kb0 == 0;
        for s in 0..strips {
            let base = s * k * 16 + kb0 * 16;
            let strip = &packed[base..base + kb * 16];
            let j = jb + s * 16;
            let mut i = 0;
            while i < m4 {
                let ct = &mut c_rows[i * n..(i + 4) * n];
                tile4x16_chunk(
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
                );
                i += 4;
            }
            while i < m {
                let c_row = &mut c_rows[i * n..(i + 1) * n];
                row_tail_packed_f64(
                    &a_rows[i * k + kb0..i * k + kb0 + kb],
                    strip,
                    c_row,
                    first,
                    j,
                );
                i += 1;
            }
        }
        kb0 += kb;
    }
}

/// 把 `B[p][jb..jb+nc]` 打成 `packed[(s*k + p)*16 + r]`（16 列一条带）。
#[inline]
pub(crate) fn pack_b_f64(k: usize, n: usize, jb: usize, nc: usize, b: &[f64], packed: &mut [f64]) {
    let strips = nc / 16;
    for p in 0..k {
        let row = &b[p * n + jb..p * n + jb + nc];
        for s in 0..strips {
            let dst = (s * k + p) * 16;
            packed[dst..dst + 16].copy_from_slice(&row[s * 16..s * 16 + 16]);
        }
    }
}

/// 微内核：4 行 × 16 列 × **kb 个 k 步**，B 从打包缓冲连续读。
///
/// `first` 为真表示第一个 k 块（累加器从 0 起），否则先把 C 里已有的部分和读回来，
/// 保证结合次序与不分块时完全一致 —— 结果逐位不变。
#[inline]
#[allow(clippy::too_many_arguments)]
fn tile4x16_chunk(
    kb: usize,
    first: bool,
    strip: &[f64],
    a0: &[f64],
    a1: &[f64],
    a2: &[f64],
    a3: &[f64],
    ct: &mut [f64],
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
        // SAFETY: j..j+16 在行内（调用方保证 16 列条带完整）。
        unsafe {
            r0a = lasx::load_f64x4(c0.as_ptr().add(j));
            r0b = lasx::load_f64x4(c0.as_ptr().add(j + 4));
            r0c = lasx::load_f64x4(c0.as_ptr().add(j + 8));
            r0d = lasx::load_f64x4(c0.as_ptr().add(j + 12));
            r1a = lasx::load_f64x4(c1.as_ptr().add(j));
            r1b = lasx::load_f64x4(c1.as_ptr().add(j + 4));
            r1c = lasx::load_f64x4(c1.as_ptr().add(j + 8));
            r1d = lasx::load_f64x4(c1.as_ptr().add(j + 12));
            r2a = lasx::load_f64x4(c2.as_ptr().add(j));
            r2b = lasx::load_f64x4(c2.as_ptr().add(j + 4));
            r2c = lasx::load_f64x4(c2.as_ptr().add(j + 8));
            r2d = lasx::load_f64x4(c2.as_ptr().add(j + 12));
            r3a = lasx::load_f64x4(c3.as_ptr().add(j));
            r3b = lasx::load_f64x4(c3.as_ptr().add(j + 4));
            r3c = lasx::load_f64x4(c3.as_ptr().add(j + 8));
            r3d = lasx::load_f64x4(c3.as_ptr().add(j + 12));
        }
    }
    for p in 0..kb {
        let base = p * 16;
        let (vb0, vb1, vb2, vb3) = unsafe {
            (
                lasx::load_f64x4(strip.as_ptr().add(base)),
                lasx::load_f64x4(strip.as_ptr().add(base + 4)),
                lasx::load_f64x4(strip.as_ptr().add(base + 8)),
                lasx::load_f64x4(strip.as_ptr().add(base + 12)),
            )
        };
        let s0 = lasx::splat_f64(a0[p]);
        let s1 = lasx::splat_f64(a1[p]);
        let s2 = lasx::splat_f64(a2[p]);
        let s3 = lasx::splat_f64(a3[p]);
        unsafe {
            r0a = lasx_xvfmadd_d(vb0, s0, r0a);
            r0b = lasx_xvfmadd_d(vb1, s0, r0b);
            r0c = lasx_xvfmadd_d(vb2, s0, r0c);
            r0d = lasx_xvfmadd_d(vb3, s0, r0d);
            r1a = lasx_xvfmadd_d(vb0, s1, r1a);
            r1b = lasx_xvfmadd_d(vb1, s1, r1b);
            r1c = lasx_xvfmadd_d(vb2, s1, r1c);
            r1d = lasx_xvfmadd_d(vb3, s1, r1d);
            r2a = lasx_xvfmadd_d(vb0, s2, r2a);
            r2b = lasx_xvfmadd_d(vb1, s2, r2b);
            r2c = lasx_xvfmadd_d(vb2, s2, r2c);
            r2d = lasx_xvfmadd_d(vb3, s2, r2d);
            r3a = lasx_xvfmadd_d(vb0, s3, r3a);
            r3b = lasx_xvfmadd_d(vb1, s3, r3b);
            r3c = lasx_xvfmadd_d(vb2, s3, r3c);
            r3d = lasx_xvfmadd_d(vb3, s3, r3d);
        }
    }
    unsafe {
        lasx::store_f64x4(c0.as_mut_ptr().add(j), r0a);
        lasx::store_f64x4(c0.as_mut_ptr().add(j + 4), r0b);
        lasx::store_f64x4(c0.as_mut_ptr().add(j + 8), r0c);
        lasx::store_f64x4(c0.as_mut_ptr().add(j + 12), r0d);
        lasx::store_f64x4(c1.as_mut_ptr().add(j), r1a);
        lasx::store_f64x4(c1.as_mut_ptr().add(j + 4), r1b);
        lasx::store_f64x4(c1.as_mut_ptr().add(j + 8), r1c);
        lasx::store_f64x4(c1.as_mut_ptr().add(j + 12), r1d);
        lasx::store_f64x4(c2.as_mut_ptr().add(j), r2a);
        lasx::store_f64x4(c2.as_mut_ptr().add(j + 4), r2b);
        lasx::store_f64x4(c2.as_mut_ptr().add(j + 8), r2c);
        lasx::store_f64x4(c2.as_mut_ptr().add(j + 12), r2d);
        lasx::store_f64x4(c3.as_mut_ptr().add(j), r3a);
        lasx::store_f64x4(c3.as_mut_ptr().add(j + 4), r3b);
        lasx::store_f64x4(c3.as_mut_ptr().add(j + 8), r3c);
        lasx::store_f64x4(c3.as_mut_ptr().add(j + 12), r3d);
    }
}

/// 单行 × 16 列，B 从打包缓冲读（不足 4 行的尾部行）。
#[inline]
fn row_tail_packed_f64(a_row: &[f64], strip: &[f64], c_row: &mut [f64], first: bool, j: usize) {
    let mut acc0 = lasx::zero_f64x4();
    let mut acc1 = lasx::zero_f64x4();
    let mut acc2 = lasx::zero_f64x4();
    let mut acc3 = lasx::zero_f64x4();
    if !first {
        // SAFETY: j..j+16 在行内。
        unsafe {
            acc0 = lasx::load_f64x4(c_row.as_ptr().add(j));
            acc1 = lasx::load_f64x4(c_row.as_ptr().add(j + 4));
            acc2 = lasx::load_f64x4(c_row.as_ptr().add(j + 8));
            acc3 = lasx::load_f64x4(c_row.as_ptr().add(j + 12));
        }
    }
    for (p, &a_p) in a_row.iter().enumerate() {
        let va = lasx::splat_f64(a_p);
        let base = p * 16;
        unsafe {
            let b0 = lasx::load_f64x4(strip.as_ptr().add(base));
            let b1 = lasx::load_f64x4(strip.as_ptr().add(base + 4));
            let b2 = lasx::load_f64x4(strip.as_ptr().add(base + 8));
            let b3 = lasx::load_f64x4(strip.as_ptr().add(base + 12));
            acc0 = lasx_xvfmadd_d(b0, va, acc0);
            acc1 = lasx_xvfmadd_d(b1, va, acc1);
            acc2 = lasx_xvfmadd_d(b2, va, acc2);
            acc3 = lasx_xvfmadd_d(b3, va, acc3);
        }
    }
    unsafe {
        lasx::store_f64x4(c_row.as_mut_ptr().add(j), acc0);
        lasx::store_f64x4(c_row.as_mut_ptr().add(j + 4), acc1);
        lasx::store_f64x4(c_row.as_mut_ptr().add(j + 8), acc2);
        lasx::store_f64x4(c_row.as_mut_ptr().add(j + 12), acc3);
    }
}

/// 4 个零累加器。
#[inline]
fn z4() -> (lasx::F64x4, lasx::F64x4, lasx::F64x4, lasx::F64x4) {
    (
        lasx::zero_f64x4(),
        lasx::zero_f64x4(),
        lasx::zero_f64x4(),
        lasx::zero_f64x4(),
    )
}

/// 数值回归测试：对照独立参考实现。
#[cfg(test)]
mod tests {
    use super::{matmul_f64_packed, matmul_f64_stream};
    use crate::ffi::matmul::lasx_matmul_f64;
    use crate::ops::testutil::{reference, rel, Lcg};

    #[test]
    fn test_matmul_f64_matches_reference() {
        let shapes = [
            (0usize, 4usize, 3usize),
            (1, 1, 1),
            (1, 0, 5),
            (1, 4, 1),
            (3, 5, 7),
            (8, 8, 8),
            (16, 17, 19),
            (32, 32, 32),
            (33, 65, 31),
            (64, 64, 64),
        ];
        for &(m, k, n) in &shapes {
            let mut rng = Lcg(0xbeef_5678 ^ ((m * 31 + k) * 17 + n) as u64);
            let a: Vec<f64> = (0..m * k).map(|_| rng.f64()).collect();
            let b: Vec<f64> = (0..k * n).map(|_| rng.f64()).collect();
            let mut c = vec![f64::NAN; m * n];
            lasx_matmul_f64(
                m as i32,
                k as i32,
                n as i32,
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
            );
            let want = reference(m, k, n, &a, &b);
            for idx in 0..m * n {
                assert!(
                    rel(c[idx], want[idx]) < 1e-12,
                    "{m}×{k}×{n} idx={idx}: got {} want {}",
                    c[idx],
                    want[idx]
                );
            }
        }
    }

    /// 打包 + k 分块路径必须与流式路径**逐位一致**（k 方向累加次序不变）。
    #[test]
    fn packed_f64_bit_exact() {
        let mut rng = Lcg(0x5eed_0064);
        for &(m, k, n) in &[
            (1usize, 1usize, 16usize),
            (3, 5, 32),
            (5, 7, 48),
            (8, 8, 32),
            (13, 17, 64),
            (32, 192, 16),
            (33, 200, 33),
            (64, 512, 64),
            (100, 448, 512),
        ] {
            let a: Vec<f64> = (0..m * k).map(|_| rng.f64()).collect();
            let b: Vec<f64> = (0..k * n).map(|_| rng.f64()).collect();
            let mut want = vec![f64::NAN; m * n];
            matmul_f64_stream(m, k, n, &a, &b, &mut want);
            let mut got = vec![f64::NAN; m * n];
            matmul_f64_packed(m, k, n, &a, &b, &mut got);
            for idx in 0..m * n {
                assert_eq!(
                    got[idx].to_bits(),
                    want[idx].to_bits(),
                    "打包 {m}×{k}×{n} @ {idx}"
                );
            }
        }
    }
}
