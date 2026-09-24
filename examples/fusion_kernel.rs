//! 融合内核实验：`C = alpha·(A·B) + beta·C`（原地）到底能不能比二遍法快。
//!
//! 上一轮（`examples/fusion_ceiling.rs`）量到的是**上限**：宽薄形状 7.5–11.1%，方阵 0.4–0.9%。
//! 但那个上限假设"融合内核能保持非融合内核的速度"——**没有内核就没法验**。这里补上内核：
//! `ops_bench_scaled`（实验性，只覆盖 k 不分块、`m%4==0`、`n%32==0` 的形状）。
//!
//! 对比两件事：
//! 1. **速度**：融合 vs 二遍法（`lasx_matmul` 写 C + elementwise `C = alpha·C + beta·C_old`）；
//! 2. **逐位一致**：融合结果必须等于契约的参考实现
//!    `beta.mul_add(C_old, alpha * acc)`（两次舍入；普通 `alpha*acc + beta*C_old` 是三次
//!    舍入，会差位）。
//!
//! `cargo run --release --example fusion_kernel`

use lasx_rs::aligned::AlignedVec;
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

/// 二遍法的第二遍（契约的两次舍入：`alpha·acc` 一次、`fma(beta, old, ·)` 一次）。
fn scale_add(alpha: f32, acc: &[f32], beta: f32, c: &mut [f32]) {
    for i in 0..c.len() {
        c[i] = beta.mul_add(c[i], alpha * acc[i]);
    }
}

fn main() {
    println!("### 融合内核实验：`C = alpha·(A·B) + beta·C`（实验性，k 不分块）");
    println!();
    println!("| 形状 | 二遍法(融合) | 融合内核 | 融合/二遍 | 省下 | 逐位一致 |");
    println!("|---|---|---|---|---|---|");

    // 只挑实验内核覆盖的形状：k ≤ 384、m%4==0、n%32==0。
    // 两个宽薄形状正是上一轮上限最高的那一档（7.5%/11.1%），一个方阵作对照（上限 0.8%）。
    for &(m, k, n) in &[
        (256usize, 64usize, 4096usize),
        (100, 64, 19072),
        (512, 256, 512),
    ] {
        let a = AlignedVec::<f32>::fill_with(m * k, |i| (i % 23) as f32 * 0.25 - 2.0);
        let b = AlignedVec::<f32>::fill_with(k * n, |i| (i % 19) as f32 * 0.5 - 4.0);
        let (alpha, beta) = (1.5f32, 0.5f32);

        // 参考：先把 C_old 抄下来，用 acc = A·B 做契约形式的 elementwise 收尾
        let mut c_fused = AlignedVec::<f32>::fill_with(m * n, |i| (i % 7) as f32 + 1.0);
        let mut c_two = c_fused.clone();
        let mut acc = AlignedVec::<f32>::new(m * n);
        lasx_matmul(
            m as i32,
            k as i32,
            n as i32,
            a.as_ptr(),
            b.as_ptr(),
            acc.as_mut_ptr(),
        );
        scale_add(alpha, acc.as_slice(), beta, c_two.as_mut_slice());

        // 融合内核（原地）
        lasx_rs::ops_bench_scaled(
            m,
            k,
            n,
            alpha,
            beta,
            a.as_slice(),
            b.as_slice(),
            c_fused.as_mut_slice(),
        );
        let same = c_two
            .iter()
            .zip(c_fused.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits());

        // 速度：二遍法 = matmul 写 acc + elementwise 收尾；融合 = 一次调用
        let mut scratch = AlignedVec::<f32>::new(m * n);
        // 拆开量：① 生产打包内核（不融合）② 我这份内核传 alpha=1/beta=0（收尾照样读写 C）
        let t_packed = timeit(|| {
            lasx_rs::ops_bench_packed(m, k, n, a.as_slice(), b.as_slice(), scratch.as_mut_slice());
            black_box(scratch[0]);
        });
        let t_scaled_10 = timeit(|| {
            lasx_rs::ops_bench_scaled(
                m,
                k,
                n,
                1.0,
                0.0,
                a.as_slice(),
                b.as_slice(),
                scratch.as_mut_slice(),
            );
            black_box(scratch[0]);
        });
        let t_two = timeit(|| {
            lasx_matmul(
                m as i32,
                k as i32,
                n as i32,
                a.as_ptr(),
                b.as_ptr(),
                scratch.as_mut_ptr(),
            );
            scale_add(alpha, scratch.as_slice(), beta, c_two.as_mut_slice());
            black_box(c_two[0]);
        });
        let t_fused = timeit(|| {
            lasx_rs::ops_bench_scaled(
                m,
                k,
                n,
                alpha,
                beta,
                a.as_slice(),
                b.as_slice(),
                c_fused.as_mut_slice(),
            );
            black_box(c_fused[0]);
        });
        println!(
            "| f32 {m}×{k}×{n} | {} | {} | {:.3}× | {:.1}% | {} |",
            fmt_us(t_two),
            fmt_us(t_fused),
            t_fused.as_secs_f64() / t_two.as_secs_f64(),
            (1.0 - t_fused.as_secs_f64() / t_two.as_secs_f64()) * 100.0,
            if same { "是" } else { "**否**" },
        );
        println!(
            "|   ↳ 拆解：生产打包内核 {}、本内核(α=1,β=0，收尾仍读写 C) {} → 收尾本身 {} |",
            fmt_us(t_packed),
            fmt_us(t_scaled_10),
            fmt_us(t_scaled_10.saturating_sub(t_packed)),
        );
        assert!(same, "{m}×{k}×{n}：融合内核与契约参考实现不逐位一致");
    }
    println!();
    println!("参考实现用 `beta.mul_add(old, alpha * acc)`（契约的两次舍入）；");
    println!("融合内核收尾是 `fma(beta, C_old, alpha·acc)`——同一件事。");
}

fn fmt_us(d: Duration) -> String {
    let v = d.as_secs_f64() * 1e6;
    if v < 1000.0 {
        format!("{v:.1} µs")
    } else {
        format!("{:.2} ms", v / 1000.0)
    }
}
