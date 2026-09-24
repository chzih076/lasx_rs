//! 把库内核铺到多核的**调用策略层**。
//!
//! 这里每个函数都只是 [`WorkerPool`] 上面几行调用样板。它们存在的意义不是"多一层封装"，
//! 而是**把正确的切分方式固定下来**——让每个调用方自己推导"按行切"该怎么切，很容易
//! 切错：矩阵乘的 `A` 每行 `k` 个元素、`C` 每行 `n` 个元素，只有切在**同一批行**上
//! 结果才对；SOA 物理内核的 6 个数组必须切在同一个下标区间。
//!
//! 收益（2026-09-25 复测，**不绑核**；完整表在 `docs/dev.md` §7.7）：
//!
//! | 负载 | 单线程 | 12 线程 | 倍数 |
//! |---|---|---|---|
//! | RK4 J2 多星传播（每步，12 线程常驻池） | 1.51 ms | 187 µs | **8.1×** |
//! | RK4 J2 单步（n = 2^18，常驻池） | 6.07 ms | 745 µs | **8.15×** |
//!
//! 前提是**池要复用**：把 `WorkerPool::new` 放进热路径等于退化成"每次新建线程"，
//! 那就是 477 µs/步 vs 187 µs/步（**2.55×**）的差距。
//!
//! 数值一致性：这些函数只把**行/下标区间**分给不同线程，每个输出元素的计算过程与
//! 单线程完全相同，因此结果与单线程**逐位一致**（有测试逐位对照）。

use crate::api::{checked_mul, expect_len, Error};
use crate::ops::matmul;
use crate::ops::matmul_f64 as matmul64;
use crate::pool::sched::Pick;

thread_local! {
    /// 双缓冲打包面板，**跨调用复用**。
    ///
    /// 为什么必须复用：`vec![0.0; cap]` 每次调用都会分配并清零（1024×1024×4096 时
    /// 两块共 8 MB、256³ 时 4 MB）。对本就几十微秒的小形状，这笔开销比内核还贵——
    /// 实测 256³/12 线程：每次分配 87 GF/s，复用后 310 GF/s（3.5×，见 dev.md §16.5）。
    static PANEL_BUFS: std::cell::RefCell<[Vec<f32>; 2]> =
        const { std::cell::RefCell::new([Vec::new(), Vec::new()]) };
}

/// 走"共享打包面板"分支的 `m` 下限。
///
/// 面板打包在并行层是**串行**做的（各线程只读共享面板），所以它带来一个 Amdahl 项：
/// 打包只与 `k·n` 成正比，而计算与 `m·k·n` 成正比 ⇒ 打包占比 ≈ 常数/`m`。
/// 实测（16 线程、`examples/matmul_ab ... pool 16`）：`m = 48/100/300` 时共享打包
/// **反而慢 40–75%**（那时旧版按线程各拿 m/线程数 行，落到流式路径、没有打包开销），
/// 而 `m ≥ 512` 时共享打包赢（512³ +19%、1024³ +76%…+240%）。交叉点取 512。
///
/// 注：更彻底的做法是**把面板打包也并行化**（池里按块长切分、各线程打自己那些条带），
/// 那样这个阈值就不需要了 —— 记为下一步（`docs/dev.md` §12）。
/// 判据（**替代**原先的固定阈值 `SHARED_PACK_MIN_M = 512`）：回退路径每线程拿 `m/线程数` 行——
/// - `per_thread >= PACK_MIN_M` 时，回退路径**会各自打包一整份 B**（病态，见 `docs/dev.md` §13.7）
///   ⇒ 必须走共享打包；
/// - 否则回退路径落在**列块路径**（不打包、B 只从 DRAM 读一遍）⇒ 共享打包要额外读写
///   一遍 B，通常更慢（实测 256³/12 线程：回退 327 vs 共享 223 GF/s）。
///
/// 实测交叉点与这条判据一致（12 线程）：384³（per_thread = 32）共享赢 287 vs 216；
/// 256³（21）回退赢 327 vs 223；宽形状 m=300/100（25/8）回退赢；512³ 及以上两者持平偏共享。
fn need_shared_pack(m: usize, threads: usize, pack_min_m: usize) -> bool {
    threads > 1 && m / threads >= pack_min_m
}

use crate::pool::WorkerPool;

/// 多核 `C[m×n] = A[m×k] · B[k×n]`（行主序，`B` 只读共享）。
///
/// `B` 是只读的，闭包以 `&` 捕获即可——池的按行块接口正是为这种形状准备的。
///
/// # Errors
/// 数组长度与 `m/k/n` 不符（[`Error::Shape`]），或 `m×k`/`k×n`/`m×n` 溢出
/// `usize`（[`Error::Overflow`]）——与 [`crate::api::matmul`] 同一套错误与消息口径。
pub fn matmul_f32(
    pool: &WorkerPool,
    m: usize,
    k: usize,
    n: usize,
    a: &mut [f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<(), Error> {
    // 调度策略按形状选一次（穷尽枚举，见 `pool::sched`）：矩阵乘的行内核按 4 行复用 B，
    // 所以 `gran = 4`；块数/动态与否由 `pick_rows` 的工作量判据决定。
    let pick = crate::pool::pick_rows(m, pool.threads(), 4);
    matmul_f32_picked(pool, m, k, n, a, b, c, pick, "matmul_f32")
}

/// 与 [`matmul_f32`] 同，但**显式指定调度策略**（A/B 与调优用）。
///
/// # Errors
/// 同 [`matmul_f32`]。
#[allow(clippy::too_many_arguments)]
pub fn matmul_f32_with_pick(
    pool: &WorkerPool,
    m: usize,
    k: usize,
    n: usize,
    a: &mut [f32],
    b: &[f32],
    c: &mut [f32],
    pick: Pick,
) -> Result<(), Error> {
    matmul_f32_picked(pool, m, k, n, a, b, c, pick, "matmul_f32_with_pick")
}

/// 两个 f32 入口的公共实现；`op` 只用于错误信息里的算子名。
#[allow(clippy::too_many_arguments)]
fn matmul_f32_picked(
    pool: &WorkerPool,
    m: usize,
    k: usize,
    n: usize,
    a: &mut [f32],
    b: &[f32],
    c: &mut [f32],
    pick: Pick,
    op: &'static str,
) -> Result<(), Error> {
    expect_len(op, "a", a.len(), checked_mul(op, "m×k", m, k)?)?;
    expect_len(op, "b", b.len(), checked_mul(op, "k×n", k, n)?)?;
    expect_len(op, "c", c.len(), checked_mul(op, "m×n", m, n)?)?;
    if m == 0 || n == 0 {
        return Ok(());
    }
    if k == 0 {
        c.fill(0.0); // 空内积：结果是全零矩阵（内核的 k 循环走 0 次，不会写 C）
        return Ok(());
    }
    // ---- 打包分支：**打包一次、跨线程共享** ----
    //
    // 不这样做的话，每个线程各自走一遍 `lasx_matmul`，而它内部会**各自打包一整份 B**
    // （1024³ 每份 4 MB）：24 线程 = 96 MB，远超 L3（32 MB），于是线程越多越慢
    // （实测 6 线程 274 GF/s → 12 线程 175 → 24 线程 93）。这里改成：并行层按面板
    // 打包一次，再把行块分给各线程，线程只读共享面板（面板本身留在共享 L3 里）。
    let work = (m as u128) * (k as u128) * (n as u128);
    if need_shared_pack(m, pool.threads(), matmul::PACK_MIN_M)
        && k >= matmul::PACK_MIN_K
        && n >= 32
        && work >= matmul::PACK_MIN_WORK
    {
        let n32 = n / 32 * 32;
        let strips_total = n32 / 32;
        let nc_strips = matmul::pack_strips(k);
        // 面板清单（每个面板 = nc_strips 个 32 列条带；最后一个可能更短）
        let panels: Vec<(usize, usize)> = {
            let mut v = Vec::new();
            let mut s0 = 0;
            while s0 < strips_total {
                let s1 = (s0 + nc_strips).min(strips_total);
                v.push((s0 * 32, (s1 - s0) * 32));
                s0 = s1;
            }
            v
        };
        // **双缓冲打包流水**：worker 算第 i 个面板时，主线程（否则只是空转自旋）打包第 i+1 个。
        // 打包是 B 的一遍读+写，原来串行做，是 Amdahl 项。实测（多面板形状，见 dev.md §16）：
        // 1024×1024×4096 +24%、512×512×8192 +31%；单面板形状没有可重叠的部分，与原来一致。
        let cap = k * nc_strips * 32;
        PANEL_BUFS.with(|cell| {
            let mut bufs = cell.borrow_mut();
            for b in bufs.iter_mut() {
                if b.len() < cap {
                    b.resize(cap, 0.0); // 只在增长时清零；之后跨调用复用
                }
            }
            let (buf_a, buf_b) = {
                let (first, rest) = bufs.split_at_mut(1);
                (&mut first[0], &mut rest[0])
            };
            matmul::pack_b(k, n, panels[0].0, panels[0].1, b, buf_a);
            for (i, &(jb, nc)) in panels.iter().enumerate() {
                let next = panels.get(i + 1).copied();
                if i % 2 == 0 {
                    pool.for_each_row_block_mut_picked_deferred(
                        m,
                        4,
                        [(a, k), (c, n)],
                        pick,
                        |_start, rows, [ab, cb]| {
                            let m4 = rows / 4 * 4;
                            matmul::matmul_f32_packed_rows(
                                k,
                                n,
                                jb,
                                nc,
                                &buf_a[..k * nc],
                                ab,
                                cb,
                                m4,
                            );
                        },
                        || {
                            if let Some((njb, nnc)) = next {
                                matmul::pack_b(k, n, njb, nnc, b, &mut buf_b[..k * nnc]);
                            }
                        },
                    );
                } else {
                    pool.for_each_row_block_mut_picked_deferred(
                        m,
                        4,
                        [(a, k), (c, n)],
                        pick,
                        |_start, rows, [ab, cb]| {
                            let m4 = rows / 4 * 4;
                            matmul::matmul_f32_packed_rows(
                                k,
                                n,
                                jb,
                                nc,
                                &buf_b[..k * nc],
                                ab,
                                cb,
                                m4,
                            );
                        },
                        || {
                            if let Some((njb, nnc)) = next {
                                matmul::pack_b(k, n, njb, nnc, b, &mut buf_a[..k * nnc]);
                            }
                        },
                    );
                }
            }
        });
        // 列尾 [n32, n)：每个线程处理自己那些行（与顺序路径同一函数 ⇒ 逐位一致）
        if n > n32 {
            pool.for_each_row_block_mut_picked(
                m,
                4,
                [(a, k), (c, n)],
                pick,
                |_start, rows, [ab, cb]| {
                    for i in 0..rows {
                        let a_row = &ab[i * k..(i + 1) * k];
                        let c_row = &mut cb[i * n..(i + 1) * n];
                        matmul::row_tail_range_f32(a_row, c_row, b, k, n, n32, n);
                    }
                },
            );
        }
        return Ok(());
    }

    let bs = b; // `&[T]` 是 Copy，闭包按值捕获这个引用即可
                // 行粒度 4：`lasx_matmul` 按 4 行分块（块内 B 复用 4 次），尾块只有 1 行
    pool.for_each_row_block_mut(m, 4, [(a, k), (c, n)], |_start, rows, [ab, cb]| {
        crate::lasx_matmul(
            rows as i32,
            k as i32,
            n as i32,
            ab.as_ptr(),
            bs.as_ptr(),
            cb.as_mut_ptr(),
        );
    });
    Ok(())
}

/// 多核 f64 版 [`matmul_f32`]。
///
/// # Panics
/// 数组长度与 `m/k/n` 不符时 panic。
pub fn matmul_f64(
    pool: &WorkerPool,
    m: usize,
    k: usize,
    n: usize,
    a: &mut [f64],
    b: &[f64],
    c: &mut [f64],
) -> Result<(), Error> {
    expect_len(
        "matmul_f64",
        "a",
        a.len(),
        checked_mul("matmul_f64", "m×k", m, k)?,
    )?;
    expect_len(
        "matmul_f64",
        "b",
        b.len(),
        checked_mul("matmul_f64", "k×n", k, n)?,
    )?;
    expect_len(
        "matmul_f64",
        "c",
        c.len(),
        checked_mul("matmul_f64", "m×n", m, n)?,
    )?;
    if m == 0 || n == 0 {
        return Ok(());
    }
    let pick = crate::pool::pick_rows(m, pool.threads(), 4);
    if k == 0 {
        c.fill(0.0);
        return Ok(());
    }
    // ---- 打包分支：与 f32 侧同源（打包一次、跨线程共享 + 双缓冲流水）----
    //
    // 不做这一步的话，每个线程各自走 `lasx_matmul_f64`，而它会各自打包一整份 B
    // （f64 的 B 字节数是 f32 的两倍：1024³ 每份 8 MB），24 线程 = 192 MB ≫ L3（32 MB），
    // 实测线程越多越慢：6 线程 110 GF/s → 12 线程 77 → 16 线程 62 → 24 线程 46。
    let work = (m as u128) * (k as u128) * (n as u128);
    if need_shared_pack(m, pool.threads(), matmul64::PACK_MIN_M)
        && k >= matmul64::PACK_MIN_K
        && n >= 16
        && work >= matmul64::PACK_MIN_WORK
    {
        let n16 = n / 16 * 16;
        let strips_total = n16 / 16;
        let nc_strips = matmul64::pack_strips(k);
        let panels: Vec<(usize, usize)> = {
            let mut v = Vec::new();
            let mut s0 = 0;
            while s0 < strips_total {
                let s1 = (s0 + nc_strips).min(strips_total);
                v.push((s0 * 16, (s1 - s0) * 16));
                s0 = s1;
            }
            v
        };
        let cap = k * nc_strips * 16;
        let mut buf_a = vec![0f64; cap];
        let mut buf_b = vec![0f64; cap];
        matmul64::pack_b_f64(k, n, panels[0].0, panels[0].1, b, &mut buf_a);
        for (i, &(jb, nc)) in panels.iter().enumerate() {
            let next = panels.get(i + 1).copied();
            if i % 2 == 0 {
                pool.for_each_row_block_mut_picked_deferred(
                    m,
                    4,
                    [(a, k), (c, n)],
                    pick,
                    |_start, rows, [ab, cb]| {
                        let m4 = rows / 4 * 4;
                        matmul64::matmul_f64_packed_rows(
                            k,
                            n,
                            jb,
                            nc,
                            &buf_a[..k * nc],
                            ab,
                            cb,
                            m4,
                        );
                    },
                    || {
                        if let Some((njb, nnc)) = next {
                            matmul64::pack_b_f64(k, n, njb, nnc, b, &mut buf_b[..k * nnc]);
                        }
                    },
                );
            } else {
                pool.for_each_row_block_mut_picked_deferred(
                    m,
                    4,
                    [(a, k), (c, n)],
                    pick,
                    |_start, rows, [ab, cb]| {
                        let m4 = rows / 4 * 4;
                        matmul64::matmul_f64_packed_rows(
                            k,
                            n,
                            jb,
                            nc,
                            &buf_b[..k * nc],
                            ab,
                            cb,
                            m4,
                        );
                    },
                    || {
                        if let Some((njb, nnc)) = next {
                            matmul64::pack_b_f64(k, n, njb, nnc, b, &mut buf_a[..k * nnc]);
                        }
                    },
                );
            }
        }
        // 列尾 [n16, n)：与顺序路径同一函数 ⇒ 逐位一致
        if n > n16 {
            pool.for_each_row_block_mut_picked(
                m,
                4,
                [(a, k), (c, n)],
                pick,
                |_start, rows, [ab, cb]| {
                    for i in 0..rows {
                        let a_row = &ab[i * k..(i + 1) * k];
                        let c_row = &mut cb[i * n..(i + 1) * n];
                        matmul64::row_tail_f64(a_row, c_row, b, k, n, n16);
                    }
                },
            );
        }
        return Ok(());
    }

    let bs = b; // `&[T]` 是 Copy，闭包按值捕获这个引用即可
                // 行粒度 4：`lasx_matmul` 按 4 行分块（块内 B 复用 4 次），尾块只有 1 行
    pool.for_each_row_block_mut(m, 4, [(a, k), (c, n)], |_start, rows, [ab, cb]| {
        crate::lasx_matmul_f64(
            rows as i32,
            k as i32,
            n as i32,
            ab.as_ptr(),
            bs.as_ptr(),
            cb.as_mut_ptr(),
        );
    });
    Ok(())
}

/// 多核批量 RK4 J2 单步（6 个 SOA 数组一次派活）。
///
/// 对应单线程的 `lasx_rk4_j2_step_batch`，数值**逐位一致**；多步传播时把池建在循环外面，
/// 每步调用本函数即可（这正是 `docs/dev.md` §7.7「多星多步轨道传播」里 8.1× 的用法）。
///
/// # 线程级降级钩子对它无效
/// 池的 worker 是复用的，其线程本地状态与调用线程无关，故本函数在每块开头显式
/// `lasx_force_lsx_thread(false)`——池化路径固定走自动分派（LASX）。
/// 要逐位比对 LSX 路径，请用单线程的 `lasx_rk4_j2_step_batch`。
///
/// # Panics
/// 6 个数组长度不一致时 panic。
#[allow(clippy::too_many_arguments)]
pub fn rk4_j2_step_batch(
    pool: &WorkerPool,
    mu: f64,
    j2: f64,
    re: f64,
    dt: f64,
    rx: &mut [f64],
    ry: &mut [f64],
    rz: &mut [f64],
    vx: &mut [f64],
    vy: &mut [f64],
    vz: &mut [f64],
) -> Result<(), Error> {
    let n = rx.len();
    expect_len("rk4_j2_step_batch", "ry", ry.len(), n)?;
    expect_len("rk4_j2_step_batch", "rz", rz.len(), n)?;
    expect_len("rk4_j2_step_batch", "vx", vx.len(), n)?;
    expect_len("rk4_j2_step_batch", "vy", vy.len(), n)?;
    expect_len("rk4_j2_step_batch", "vz", vz.len(), n)?;
    if n == 0 {
        return Ok(());
    }
    pool.for_each_chunks_mut([rx, ry, rz, vx, vy, vz], |[rx, ry, rz, vx, vy, vz]| {
        let m = rx.len() as i32;
        crate::lasx_force_lsx_thread(false);
        crate::lasx_rk4_j2_step_batch(
            rx.as_mut_ptr(),
            ry.as_mut_ptr(),
            rz.as_mut_ptr(),
            vx.as_mut_ptr(),
            vy.as_mut_ptr(),
            vz.as_mut_ptr(),
            mu,
            j2,
            re,
            dt,
            m,
        );
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aligned::AlignedVec;

    /// 池化矩阵乘必须与单线程**逐位一致**。形状要同时覆盖三条分支：
    /// 小形状（走 `lasx_matmul` 的流式/列块路径）、`m < SHARED_PACK_MIN_M` 的回退、
    /// 以及 `m ≥ 512` 的**共享打包面板**分支（含行尾 513 = 128·4+1 与列尾 130 = 4·32+2）。
    #[test]
    fn test_matmul_f32_matches_serial_bit_for_bit() {
        let pool = WorkerPool::new(7);
        for &(m, k, n) in &[
            (1usize, 1usize, 1usize),
            (13, 5, 3),
            (64, 64, 64),
            (129, 33, 17),
            (256, 96, 40),
            (128, 256, 96),  // 回退分支（m < 512）
            (512, 512, 64),  // 共享打包分支
            (513, 448, 130), // 共享打包 + 行尾 + 列尾
        ] {
            let (mut a, b) = (
                AlignedVec::<f32>::fill_with(m * k, |i| ((i % 23) as f32 - 11.0) * 0.25),
                AlignedVec::<f32>::fill_with(k * n, |i| ((i % 19) as f32 - 9.0) * 0.5),
            );
            let mut want = AlignedVec::<f32>::new(m * n);
            let mut got = AlignedVec::<f32>::new(m * n);
            crate::lasx_matmul(
                m as i32,
                k as i32,
                n as i32,
                a.as_ptr(),
                b.as_ptr(),
                want.as_mut_ptr(),
            );
            matmul_f32(
                &pool,
                m,
                k,
                n,
                a.as_mut_slice(),
                b.as_slice(),
                got.as_mut_slice(),
            )
            .unwrap();
            assert_eq!(
                want.as_slice(),
                got.as_slice(),
                "{m}×{k}×{n} 与单线程不一致"
            );
        }
    }

    /// f64 版同上，形状覆盖三条分支：小形状（流式）、共享打包 + 单面板、
    /// 以及**共享打包 + 多面板 + 双缓冲流水**（`(512, 512, 2048)`：16 列一条带 ⇒ 128 条带
    /// ⇒ 8 个面板）与行尾/列尾。
    #[test]
    fn test_matmul_f64_matches_serial_bit_for_bit() {
        let pool = WorkerPool::new(5);
        for &(m, k, n) in &[
            (100usize, 48usize, 37usize),
            (128, 256, 96),
            (512, 512, 64),   // 共享打包、单面板
            (512, 512, 2048), // 共享打包、多面板（流水）
            (513, 448, 130),  // 行尾 + 列尾
        ] {
            let (mut a, b) = (
                AlignedVec::<f64>::fill_with(m * k, |i| ((i % 29) as f64 - 14.0) * 0.125),
                AlignedVec::<f64>::fill_with(k * n, |i| ((i % 31) as f64 - 15.0) * 0.0625),
            );
            let mut want = AlignedVec::<f64>::new(m * n);
            let mut got = AlignedVec::<f64>::new(m * n);
            crate::lasx_matmul_f64(
                m as i32,
                k as i32,
                n as i32,
                a.as_ptr(),
                b.as_ptr(),
                want.as_mut_ptr(),
            );
            matmul_f64(
                &pool,
                m,
                k,
                n,
                a.as_mut_slice(),
                b.as_slice(),
                got.as_mut_slice(),
            )
            .unwrap();
            for i in 0..m * n {
                // 逐位比较：f64 的 == 会把 -0.0/0.0 视为相等，也会漏掉 NaN 的位型差异
                assert_eq!(
                    got[i].to_bits(),
                    want[i].to_bits(),
                    "f64 池化 {m}×{k}×{n} @ {i}"
                );
            }
        }
    }

    /// 池化 RK4 步必须与单线程逐位一致（切块边界不得影响结果）。
    #[test]
    fn test_rk4_pooled_matches_serial_bit_for_bit() {
        let pool = WorkerPool::new(6);
        let n = 20_000usize;
        let make = || {
            (
                AlignedVec::<f64>::fill_with(n, |i| 7.0e6 + (i % 97) as f64 * 1.0e3),
                AlignedVec::<f64>::fill_with(n, |i| (i % 89) as f64 * 1.0e3 - 4.0e4),
                AlignedVec::<f64>::fill_with(n, |i| (i % 71) as f64 * 1.0e3 - 3.0e4),
                AlignedVec::<f64>::fill_with(n, |i| (i % 53) as f64 * 2.0 - 50.0),
                AlignedVec::<f64>::fill_with(n, |i| (i % 43) as f64 * 1.5 - 30.0),
                AlignedVec::<f64>::fill_with(n, |i| (i % 37) as f64 * 1.25 - 20.0),
            )
        };
        let (mu, j2, re, dt) = (3.986_004_418e14, 1.082_626_68e-3, 6.378_137e6, 10.0);

        let (mut rx, mut ry, mut rz, mut vx, mut vy, mut vz) = make();
        for _ in 0..3 {
            crate::lasx_rk4_j2_step_batch(
                rx.as_mut_ptr(),
                ry.as_mut_ptr(),
                rz.as_mut_ptr(),
                vx.as_mut_ptr(),
                vy.as_mut_ptr(),
                vz.as_mut_ptr(),
                mu,
                j2,
                re,
                dt,
                n as i32,
            );
        }

        let (mut px, mut py, mut pz, mut qx, mut qy, mut qz) = make();
        for _ in 0..3 {
            rk4_j2_step_batch(
                &pool,
                mu,
                j2,
                re,
                dt,
                px.as_mut_slice(),
                py.as_mut_slice(),
                pz.as_mut_slice(),
                qx.as_mut_slice(),
                qy.as_mut_slice(),
                qz.as_mut_slice(),
            )
            .unwrap();
        }

        for i in 0..n {
            assert_eq!(rx[i].to_bits(), px[i].to_bits(), "rx 不一致 @ {i}");
            assert_eq!(vz[i].to_bits(), qz[i].to_bits(), "vz 不一致 @ {i}");
        }
    }

    /// 空内积（k = 0）与零维：返回 `Ok`，且 `k=0` 时结果是全零。
    #[test]
    fn test_degenerate_shapes() {
        let pool = WorkerPool::new(3);
        let mut a = vec![1.0f32; 0];
        let b = vec![1.0f32; 0];
        let mut c = vec![1.0f32; 4];
        matmul_f32(&pool, 2, 0, 2, &mut a, &b, &mut c).unwrap();
        assert_eq!(c, vec![0.0f32; 4], "k=0 时结果应为全零");
        matmul_f32(&pool, 0, 0, 0, &mut [], &[], &mut []).unwrap();
    }

    /// **形状不符返回 `Err` 而不是 panic**（这是本层接口债清掉的那一条）：
    /// 错误口径与 `api::matmul` 一致（同样是 `Error::Shape`，带算子名/参数名/期望/实际）。
    #[test]
    fn test_shape_mismatch_returns_err_not_panic() {
        let pool = WorkerPool::new(4);
        let (mut a, b) = (vec![0f32; 6], vec![0f32; 6]);
        let mut c = vec![0f32; 6];

        // A 长度不对
        match matmul_f32(&pool, 2, 3, 2, &mut a[..5], &b, &mut c) {
            Err(crate::api::Error::Shape {
                op,
                what,
                expected,
                got,
            }) => {
                assert_eq!((op, what, expected, got), ("matmul_f32", "a", 6, 5));
            }
            other => panic!("期望 Shape 错误，得到 {other:?}"),
        }
        // B 长度不对
        assert!(matmul_f32(&pool, 2, 3, 2, &mut a, &b[..5], &mut c).is_err());
        // C 长度不对
        assert!(matmul_f32(&pool, 2, 3, 2, &mut a, &b, &mut c[..5]).is_err());
        // 溢出：`m×k` 超过 usize
        assert!(matches!(
            matmul_f32(&pool, usize::MAX, 2, 2, &mut a, &b, &mut c),
            Err(crate::api::Error::Overflow { .. })
        ));
        // f64 与 rk4 同样返回 Err
        let (mut a64, b64) = (vec![0f64; 6], vec![0f64; 6]);
        let mut c64 = [0f64; 6];
        assert!(matmul_f64(&pool, 2, 3, 2, &mut a64, &b64, &mut c64[..5]).is_err());

        let mut r0 = vec![0f64; 8];
        let mut r1 = vec![0f64; 7]; // ← 故意不等长
        let mut r2 = vec![0f64; 8];
        let mut v0 = vec![0f64; 8];
        let mut v1 = vec![0f64; 8];
        let mut v2 = vec![0f64; 8];
        match rk4_j2_step_batch(
            &pool, 1.0, 1e-3, 1.0, 1.0, &mut r0, &mut r1, &mut r2, &mut v0, &mut v1, &mut v2,
        ) {
            Err(crate::api::Error::Shape { op, what, .. }) => {
                assert_eq!((op, what), ("rk4_j2_step_batch", "ry"));
            }
            other => panic!("期望 Shape 错误，得到 {other:?}"),
        }
    }
}
