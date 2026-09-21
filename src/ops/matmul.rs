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
//! 数值行为不变：每个输出元素仍是**沿 k 的单个 f32 累加器**，累加次序与 v2 完全一致。
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
    // 小 m、大 n：B 的复用度只有 m/4，流式次序会把 B 读 m/4 遍。改成"列块在外"，
    // 让 B 只流一遍（实测 100×448×19147 快 5.8×，方阵无回归，见 perf-report §18）。
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
    use std::time::Instant;

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

    fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }

    /// 列块宽度定标（宽形状），以及与"流式"次序的成对比较（含方阵回归）。
    #[test]
    fn paired_wide_vs_stream() {
        // 1) 宽形状：列块宽度扫描
        let (m, k, n) = (100usize, 448usize, 19147usize);
        let mut rng = Lcg(0x77aa);
        let a: Vec<f32> = (0..m * k).map(|_| rng.f64() as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|_| rng.f64() as f32).collect();
        let flop = 2.0 * m as f64 * k as f64 * n as f64;
        let mut c = vec![0f32; m * n];
        let t_stream = median(
            (0..3)
                .map(|_| {
                    let t = Instant::now();
                    matmul_f32(m, k, n, &a, &b, &mut c);
                    t.elapsed().as_secs_f64()
                })
                .collect(),
        );
        println!(
            "宽形状 {m}×{k}×{n}: 流式 {:.1} ms / {:.1} GF/s",
            t_stream * 1e3,
            flop / t_stream / 1e9
        );
        for &cb in &[32usize, 64, 128, 256, 512, 1024, 4096] {
            let t = median(
                (0..3)
                    .map(|_| {
                        let t = Instant::now();
                        matmul_f32_cols_block(m, k, n, 0, n, &a, &b, &mut c, cb);
                        t.elapsed().as_secs_f64()
                    })
                    .collect(),
            );
            println!(
                "  列块 {cb:>5}: {:.1} ms / {:.1} GF/s  （{:.2}×）",
                t * 1e3,
                flop / t / 1e9,
                t_stream / t
            );
        }

        // 2) 方阵回归：列块序不应拖慢
        for &s0 in &[256usize, 512] {
            let mut rng = Lcg(0x33cc);
            let a: Vec<f32> = (0..s0 * s0).map(|_| rng.f64() as f32).collect();
            let b: Vec<f32> = (0..s0 * s0).map(|_| rng.f64() as f32).collect();
            let mut c1 = vec![0f32; s0 * s0];
            let mut c2 = vec![0f32; s0 * s0];
            matmul_f32(s0, s0, s0, &a, &b, &mut c1);
            matmul_f32_cols_block(s0, s0, s0, 0, s0, &a, &b, &mut c2, COL_BLOCK);
            assert_eq!(c1, c2, "方阵 {s0} 两种次序不一致");
            let f = 2.0 * (s0 * s0 * s0) as f64;
            let ts = median(
                (0..3)
                    .map(|_| {
                        let t = Instant::now();
                        matmul_f32(s0, s0, s0, &a, &b, &mut c1);
                        t.elapsed().as_secs_f64()
                    })
                    .collect(),
            );
            let tc = median(
                (0..3)
                    .map(|_| {
                        let t = Instant::now();
                        matmul_f32_cols_block(s0, s0, s0, 0, s0, &a, &b, &mut c2, COL_BLOCK);
                        t.elapsed().as_secs_f64()
                    })
                    .collect(),
            );
            println!(
                "方阵 {s0}³: 流式 {:.1} GF/s  列块 {:.1} GF/s （{:.2}×）",
                f / ts / 1e9,
                f / tc / 1e9,
                ts / tc
            );
        }
    }
}
