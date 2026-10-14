//! 融合对照：`C = alpha·(A·B) + beta·C`（原地）vs"先算完再整遍读写 `C`"。
//!
//! **结论（随形状与 `beta` 变，不是"一律更快"）**：
//!
//! - `beta == 0`（CBLAS 的 `C = alpha·A·B`）：融合**稳定更快**——省掉的正是"对 `C` 的一整遍
//!   读写"。宽薄形状（小 `k`、大 `m×n`）收益最大，方阵 ≈0（上限本来就只有 0.8%）；
//! - `beta != 0` 且 `k ≤ 384`：融合仍小胜（收尾读 `C_old` 代替一遍三流 elementwise）；
//! - `beta != 0` 且 `k > 384`：打包路径会把部分和落进 `C`，`C_old` 会被覆盖 ⇒ **融合不成立**，
//!   `api::matmul_scaled_into` 退回**与朴素二遍法同构**的实现（下表这一行会显示 ≈1.00×）。
//!
//! 这个示例是**同进程**的快速视图，数字只当趋势。正式验收走 `lasx_bench matmul a|b`
//! 的**进程级交替**（`docs/dev.md` §6.4：≥3 轮、丢弃首轮、中位数 + 极差、每轮记 loadavg；
//! `load ≤ 3` 才算正式数）。
//!
//! 融合的正式实现在 `src/ops/matmul.rs`：`Epi::{Plain, Scale, Fma}` 把缩放融进**既有内核的
//! 收尾**（打包 / 列块 / 流式三条路径共用同一套 k 循环），没有另起一条简化 tile 路径——
//! 上一轮（`docs/dev.md` §19.11）否掉的是"另起路径 + 收尾无条件读写 C"那个形态。
//!
//! `cargo run --release --example fusion_kernel`

use lasx_rs::aligned::AlignedVec;
use lasx_rs::api;
use lasx_rs::lasx_matmul;
use std::hint::black_box;
use std::time::{Duration, Instant};

fn timeit<F: FnMut()>(mut f: F) -> Duration {
    f();
    f();
    let t = Instant::now();
    f();
    let single = t.elapsed().max(Duration::from_nanos(1));
    let reps =
        (Duration::from_millis(25).as_nanos() / single.as_nanos()).clamp(1, 10_000_000) as u32;
    let mut v = Vec::with_capacity(5);
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        v.push(t.elapsed() / reps);
    }
    v.sort_unstable();
    v[2]
}

/// 两遍法的第二遍（口径的定义）：`C = fma(beta, C_old, alpha·acc)`，两次舍入。
///
/// 注意用**切片索引**而不是 `AlignedVec` 的 `Deref` 索引：后者每个元素多一层调用，
/// 实测能把这一遍拖慢 10× 以上（3.75 ms vs 0.21 ms），那会让"融合"白得一个假胜。
fn scale_add(alpha: f32, acc: &[f32], beta: f32, c: &mut [f32]) {
    for i in 0..c.len() {
        c[i] = beta.mul_add(c[i], alpha * acc[i]);
    }
}

fn main() {
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    println!("### 融合 vs 两遍法：`C = alpha·(A·B) + beta·C`（同进程快速视图）");
    println!();
    println!("loadavg = {}（load > 3 只能当趋势）", load.trim());
    println!();
    println!("| 形状 | α / β | 两遍法 | 融合 | 融合/两遍 | 逐位一致 |");
    println!("|---|---|---|---|---|---|");

    // 形状覆盖：宽薄（打包 + 小 k）、极宽 n（列块路径）、方阵（k ≤ 384）、
    // 方阵 + k 分块（k > 384，beta≠0 时融合不成立）。n = 19147 不是 32 的倍数，
    // 顺带把列尾路径也走一遍（旧实验内核要求 n%32==0，这版没有这个限制）。
    for &(m, k, n, alpha, beta) in &[
        (256usize, 64usize, 4096usize, 1.5f32, 0.0f32),
        (100, 64, 19147, 1.5, 0.0),
        (256, 64, 4096, 1.5, 1.0),
        (512, 512, 512, 1.5, 0.0),
        (512, 512, 512, 1.5, 1.0),
    ] {
        let a = AlignedVec::<f32>::fill_with(m * k, |i| (i % 23) as f32 * 0.25 - 2.0);
        let b = AlignedVec::<f32>::fill_with(k * n, |i| (i % 19) as f32 * 0.5 - 4.0);

        // 参考（口径的定义）：acc = A·B，再 C = fma(beta, C_old, alpha·acc)
        let mut acc = AlignedVec::<f32>::new(m * n);
        lasx_matmul(
            m as i32,
            k as i32,
            n as i32,
            a.as_ptr(),
            b.as_ptr(),
            acc.as_mut_ptr(),
        );
        let mut c_two = AlignedVec::<f32>::fill_with(m * n, |i| (i % 7) as f32 + 1.0);
        scale_add(alpha, acc.as_slice(), beta, c_two.as_mut_slice());

        // 融合（原地）：公开入口
        let mut c_fused = AlignedVec::<f32>::fill_with(m * n, |i| (i % 7) as f32 + 1.0);
        api::matmul_scaled_into(
            m,
            k,
            n,
            alpha,
            beta,
            a.as_slice(),
            b.as_slice(),
            c_fused.as_mut_slice(),
        )
        .unwrap();
        let same = c_two
            .iter()
            .zip(c_fused.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits());

        let t_two = timeit(|| {
            lasx_matmul(
                m as i32,
                k as i32,
                n as i32,
                a.as_ptr(),
                b.as_ptr(),
                acc.as_mut_ptr(),
            );
            scale_add(alpha, acc.as_slice(), beta, c_two.as_mut_slice());
            black_box(c_two[0]);
        });
        let t_fused = timeit(|| {
            api::matmul_scaled_into(
                m,
                k,
                n,
                alpha,
                beta,
                a.as_slice(),
                b.as_slice(),
                c_fused.as_mut_slice(),
            )
            .unwrap();
            black_box(c_fused[0]);
        });
        println!(
            "| f32 {m}×{k}×{n} | {alpha} / {beta} | {} | {} | {:.3}× | {} |",
            fmt_us(t_two),
            fmt_us(t_fused),
            t_fused.as_secs_f64() / t_two.as_secs_f64(),
            if same { "是" } else { "**否**" },
        );
        assert!(same, "{m}×{k}×{n} α={alpha} β={beta}：融合与参考不逐位一致");
    }
    println!();
    println!(
        "参考用 `beta.mul_add(C_old, alpha * acc)`（两次舍入）；融合收尾是 `fma(beta, C_old, alpha·acc)`"
    );
    println!("——同一件事，所以上表「逐位一致」列全是「是」。`beta = 0` 时融合**不读** `C_old`。");
}

fn fmt_us(d: Duration) -> String {
    let v = d.as_secs_f64() * 1e6;
    if v < 1000.0 {
        format!("{v:.1} µs")
    } else {
        format!("{:.2} ms", v / 1000.0)
    }
}
