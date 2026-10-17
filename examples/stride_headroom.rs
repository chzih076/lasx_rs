//! **P5-b 负载调研探针**：`Q·Kᵀ` 形态里"**物化转置**"的真实成本 vs `matmul` 成本。
//!
//! 为什么测这个：P5-b 要解冻"跨步读写"，前提是**真实负载 + 可测收益**。本仓**没有 2D stencil 算子**，
//! 最接近的"真实跨步形态"是 attention 的 `scores = Q[S,D] · Kᵀ[D,S]`：K 在内存里是 `[S,D]` 行主序，
//! 要么**先物化 Kᵀ**（一趟 `S·D` 拷贝）再调 `matmul`，要么让 matmul 直接跨步读 K。
//!
//! 本探针只量**现状工作法的成本**（物化转置），据此算"跨步能省多少"的**上限**：
//! `headroom ≈ 转置时间 / (转置时间 + matmul 时间)`。跑 5 轮丢首轮取中位数（§5 口径精神）。
//!
//! 结论会写进 `simd_dsl/docs/optimizations.md §8`（P5-b 负载调研），数字家仍是 `lasx_rs/docs/dev.md`。
#![allow(clippy::needless_range_loop)] // 探针里按下标写转置，保持"用户会怎么写"的形态

use std::time::Instant;

use lasx_rs::api;

/// 朴素物化转置：`dst[c*S + r] = src[r*D + c]`（这就是没有 transpose 原语时用户会写的代码）。
fn transpose_copy(src: &[f32], s: usize, d: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; s * d];
    for r in 0..s {
        for c in 0..d {
            dst[c * s + r] = src[r * d + c];
        }
    }
    dst
}

/// 分块转置（32×32 tile，内层按行整块拷贝）——"有经验的人会写的版本"，用它当**公正基线**。
fn transpose_blocked(src: &[f32], s: usize, d: usize) -> Vec<f32> {
    const T: usize = 32;
    let mut dst = vec![0.0f32; s * d];
    let mut r0 = 0;
    while r0 < s {
        let mut c0 = 0;
        while c0 < d {
            let r1 = (r0 + T).min(s);
            let c1 = (c0 + T).min(d);
            for r in r0..r1 {
                for c in c0..c1 {
                    dst[c * s + r] = src[r * d + c];
                }
            }
            c0 += T;
        }
        r0 += T;
    }
    dst
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    // attention 常见形状：S=序列长、D=head dim
    for &(s, d) in &[(1024usize, 64usize), (2048, 128)] {
        let q = vec![0.25f32; s * d];
        let k = vec![0.5f32; s * d];
        let mut t_tr = Vec::new();
        let mut t_bl = Vec::new();
        let mut t_mm = Vec::new();
        for round in 0..5 {
            let t0 = Instant::now();
            let kt = transpose_copy(&k, s, d);
            let tr = t0.elapsed().as_secs_f64() * 1e3;
            let t0b = Instant::now();
            let ktb = transpose_blocked(&k, s, d);
            let bl = t0b.elapsed().as_secs_f64() * 1e3;
            // scores = Q·Kᵀ  ⇒  [S,D]·[D,S] = [S,S]
            let t1 = Instant::now();
            let _scores = api::matmul(s, d, s, &q, &kt).expect("matmul");
            let mm = t1.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(&ktb);
            if round > 0 {
                t_tr.push(tr);
                t_bl.push(bl);
                t_mm.push(mm);
            }
        }
        let (tr, bl, mm) = (median(t_tr), median(t_bl), median(t_mm));
        let hr_naive = tr / (tr + mm) * 100.0;
        let hr_blocked = bl / (bl + mm) * 100.0;
        eprintln!(
            "S={s:>4} D={d:>3}: 朴素转置 {tr:.3} ms | **分块转置 {bl:.3} ms** | matmul {mm:.3} ms ⇒ 跨步上限 朴素 {hr_naive:.1}% / **分块 {hr_blocked:.1}%**"
        );
    }
}
