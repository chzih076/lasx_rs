//! **P4-e 前置测量**：融合原语 vs 现有"两步/多步"路径的**同进程 A/B**（§5 口径）。
//!
//! 目的：在把融合原语接进 `engine`/帧路径（P4-c）之前，先回答**值不值得接**。
//! 注意：本轮两个 LASX 覆盖是**标量**实现（FMA 版待封装名核对），所以这里的对比是
//! "**标量融合核** vs **现有向量化两步路径**"——若融合核更慢，如实记录，不做接线。
//!
//! 口径（抗噪）：同进程、交替 A/B、**8 对丢首对（7 对）**，同时打印**中位**与 **min** 比值
//! （负载只增不减 ⇒ min 更接近无噪速度；本测试**没有时间阈值断言**，只打印证据，不会因负载抖动）。
#![cfg(feature = "dsl-backend")]

use std::time::Instant;

use lasx_rs::dsl_backend::LasxBackend;
use simd_dsl::{Backend, BinaryOp};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// 单臂计时：`iters` 次调用的总毫秒（调用方取中位数）。
fn time<F: FnMut()>(mut f: F, iters: usize) -> f64 {
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed().as_secs_f64() * 1e3
}

#[test]
fn fused_primitives_ab_same_process() {
    let b = LasxBackend;
    let rows = 4096usize; // n
    let cols = 256usize; // b
    let mut g = 0x1234_5678_u64;
    let mut rnd = move || {
        g = g
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((g >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
    };
    let x: Vec<f32> = (0..rows * cols).map(|_| rnd()).collect();
    let mean: Vec<f32> = (0..cols).map(|_| rnd()).collect();
    let mean_rep: Vec<f32> = (0..rows * cols).map(|i| mean[i % cols]).collect();

    // ---- 臂 A：现有向量化多步路径（Sub → Mul → 列和，三个全尺寸缓冲）----
    let mut tmp = vec![0.0f32; rows * cols];
    let mut tmp2 = vec![0.0f32; rows * cols];
    let mut out_a = vec![0.0f32; cols];
    let arm_a = |b: &LasxBackend, tmp: &mut [f32], tmp2: &mut [f32], out: &mut [f32]| {
        b.map_binary_f32(&x, &mean_rep, tmp, BinaryOp::Sub).unwrap();
        b.map_binary_f32(tmp, tmp, tmp2, BinaryOp::Mul).unwrap();
        b.reduce_sum_cols_f32(tmp2, rows, cols, out).unwrap();
    };

    // ---- 臂 B：融合原语（单遍）----
    let mut out_b = vec![0.0f32; cols];
    let arm_b = |b: &LasxBackend, out: &mut [f32]| {
        b.reduce_centered_sq_sum_cols_f32(&x, rows, cols, &mean, out)
            .unwrap();
    };

    // 预热 + 交替测量。
    // **抗噪口径（本测试无时间阈值断言，只影响打印证据质量）**：同进程交替 **8 对、丢首对（7 对）**；
    // 同时给**中位比值**与 **min 比值**——共享机器负载只增不减，min 更接近"无噪速度"；
    // 正式验收（若将来加门限）仍用进程级交替 + 中位，**min 只用于趋势判断**。
    arm_b(&b, &mut out_b);
    let iters = 3;
    let mut ta = Vec::new();
    let mut tb = Vec::new();
    for _ in 0..8 {
        ta.push(time(|| arm_a(&b, &mut tmp, &mut tmp2, &mut out_a), iters));
        tb.push(time(|| arm_b(&b, &mut out_b), iters));
    }
    ta.remove(0); // 丢首对（预热）
    tb.remove(0);
    let (ma, mb) = (median(ta.clone()), median(tb.clone()));
    let (mina, minb) = (
        ta.iter().copied().fold(f64::INFINITY, f64::min),
        tb.iter().copied().fold(f64::INFINITY, f64::min),
    );
    eprintln!(
        "energy: 多步 {ma:.4} ms（min {mina:.4}）/ 融合 {mb:.4} ms（min {minb:.4}）\
         ⇒ 融合/多步 = {:.3}×（中位）/ {:.3}×（min；<1 才是更快）",
        mb / ma,
        minb / mina
    );
    // 数值对齐（仅确认两臂算的是同一个东西；容差类给实测界）
    let mut worst = 0.0f32;
    for c in 0..cols {
        let d = (out_a[c] - out_b[c]).abs() / out_a[c].abs().max(1e-30);
        worst = worst.max(d);
    }
    eprintln!("energy: 两臂 max rel = {worst:e}");
    assert!(worst < 1e-5, "两臂应算同一量：max rel = {worst:e}");
}
