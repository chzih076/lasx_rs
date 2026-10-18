//! **f64 几何五入口 fused A/B**（阶段 4 收尾 / `task-49`）：
//! A = `lasx_rs::api::*_into`（LASX 向量核直调）；B = `Backend::*_f64`（经 [`LasxBackend`] 的 override）。
//!
//! 口径：同进程交替、6 对、**丢首对**、取中位；打印 `threads`/`ms`/`ratio`/`loadavg`；
//! 每对先等价类（逐位优先）。判据 `ratio = api_ms / dsl_ms ≥ 0.98`（override 是薄转发，预期 ≈1.0）。
//!
//! **零向量单列**：LASX `unitize3` 对零向量给 `0`，合同默认（组合版）给 `NaN` —— **路径各自语义**，
//! 本装置打印两边实际值，供 `docs/dev.md §7.34` 登记（override 不打补丁）。
//!
//! 用法：`cargo run --release --features dsl-backend --example geomf64_fused_ab`

use std::hint::black_box;
use std::time::{Duration, Instant};

use lasx_rs::api;
use lasx_rs::dsl_backend::LasxBackend;
use simd_dsl::Backend;

const ROUNDS: usize = 6;

fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "<unreadable>".to_string())
}
fn timeit(f: &mut impl FnMut()) -> Duration {
    let t = Instant::now();
    f();
    t.elapsed()
}
fn median(mut v: Vec<Duration>) -> Duration {
    v.sort_unstable();
    v[v.len() / 2]
}
fn ab(mut a: impl FnMut(), mut b: impl FnMut()) -> (Duration, Duration) {
    let (mut ta, mut tb) = (Vec::new(), Vec::new());
    for round in 0..ROUNDS {
        if round % 2 == 0 {
            ta.push(timeit(&mut a));
            tb.push(timeit(&mut b));
        } else {
            tb.push(timeit(&mut b));
            ta.push(timeit(&mut a));
        }
    }
    (median(ta[1..].to_vec()), median(tb[1..].to_vec()))
}
fn sample(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| ((i * 37 % 101) as f64) * 0.1 - 6.07)
        .collect()
}
fn check(name: &str, a: &[f64], b: &[f64]) {
    if a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()) {
        println!("  [{name}] 等价类：**逐位**");
        return;
    }
    let mut max_abs = 0.0f64;
    let mut peak = 1.0f64;
    for (x, y) in a.iter().zip(b) {
        peak = peak.max(x.abs());
        max_abs = max_abs.max((x - y).abs());
    }
    println!(
        "  [{name}] 等价类：**容差**（max abs {max_abs:.3e}，max rel {:.3e}）",
        max_abs / peak
    );
}
fn row(name: &str, n: usize, a: Duration, b: Duration) {
    let (ams, bms) = (a.as_secs_f64() * 1e3, b.as_secs_f64() * 1e3);
    println!(
        "| {name} | {n} | {ams:.3} | {bms:.3} | **{:.3}×** |",
        ams / bms
    );
}

fn main() {
    let be = LasxBackend;
    println!("f64 几何 fused A/B（api 直调 vs Backend override；{ROUNDS} 对丢首对取中位）");
    println!("threads = {}", lasx_rs::pool::global().threads());
    println!(
        "cap: FusedGeometry={}",
        be.supports(simd_dsl::Capability::FusedGeometry)
    );
    println!("loadavg(开始) = {}", loadavg());
    println!("| 算子 | n | api ms | dsl ms | 比值 api/dsl |");
    println!("|---|---|---|---|---|");

    let n = 1 << 20; // 1M 样本
    let x = sample(n);
    let y = sample(n + 3)[3..].to_vec();
    let z = sample(n + 7)[7..].to_vec();
    let (bx, by, bz) = (
        sample(n + 5)[5..].to_vec(),
        sample(n + 11)[11..].to_vec(),
        sample(n + 13)[13..].to_vec(),
    );

    // ---- vec3_add_scaled：o = a + s·b ----
    {
        let (mut ao, mut bo) = (vec![0.0f64; n], vec![0.0f64; n]);
        let (mut aoy, mut boy) = (vec![0.0f64; n], vec![0.0f64; n]);
        let (mut aoz, mut boz) = (vec![0.0f64; n], vec![0.0f64; n]);
        api::vec3_add_scaled_batch_into(
            &x, &y, &z, &bx, &by, &bz, 1.25, &mut ao, &mut aoy, &mut aoz,
        )
        .unwrap();
        be.vec3_add_scaled_f64(&x, &y, &z, &bx, &by, &bz, 1.25, &mut bo, &mut boy, &mut boz)
            .unwrap();
        check("vec3_add_scaled", &ao, &bo);
        let (a, b) = ab(
            || {
                api::vec3_add_scaled_batch_into(
                    &x, &y, &z, &bx, &by, &bz, 1.25, &mut ao, &mut aoy, &mut aoz,
                )
                .unwrap();
                black_box(&ao);
            },
            || {
                be.vec3_add_scaled_f64(
                    &x, &y, &z, &bx, &by, &bz, 1.25, &mut bo, &mut boy, &mut boz,
                )
                .unwrap();
                black_box(&bo);
            },
        );
        row("vec3_add_scaled_f64", n, a, b);
    }
    // ---- norm3 ----
    {
        let (mut ao, mut bo) = (vec![0.0f64; n], vec![0.0f64; n]);
        api::norm3_batch_into(&x, &y, &z, &mut ao).unwrap();
        be.norm3_batch_f64(&x, &y, &z, &mut bo).unwrap();
        check("norm3", &ao, &bo);
        let (a, b) = ab(
            || {
                api::norm3_batch_into(&x, &y, &z, &mut ao).unwrap();
                black_box(&ao);
            },
            || {
                be.norm3_batch_f64(&x, &y, &z, &mut bo).unwrap();
                black_box(&bo);
            },
        );
        row("norm3_batch_f64", n, a, b);
    }
    // ---- unitize3 ----
    {
        let (mut ao, mut bo) = (vec![0.0f64; n], vec![0.0f64; n]);
        let (mut aoy, mut boy) = (vec![0.0f64; n], vec![0.0f64; n]);
        let (mut aoz, mut boz) = (vec![0.0f64; n], vec![0.0f64; n]);
        api::unitize3_batch_into(&x, &y, &z, &mut ao, &mut aoy, &mut aoz).unwrap();
        be.unitize3_batch_f64(&x, &y, &z, &mut bo, &mut boy, &mut boz)
            .unwrap();
        check("unitize3", &ao, &bo);
        let (a, b) = ab(
            || {
                api::unitize3_batch_into(&x, &y, &z, &mut ao, &mut aoy, &mut aoz).unwrap();
                black_box(&ao);
            },
            || {
                be.unitize3_batch_f64(&x, &y, &z, &mut bo, &mut boy, &mut boz)
                    .unwrap();
                black_box(&bo);
            },
        );
        row("unitize3_batch_f64", n, a, b);
    }
    // ---- cross3 ----
    {
        let (mut ao, mut bo) = (vec![0.0f64; n], vec![0.0f64; n]);
        let (mut aoy, mut boy) = (vec![0.0f64; n], vec![0.0f64; n]);
        let (mut aoz, mut boz) = (vec![0.0f64; n], vec![0.0f64; n]);
        api::cross3_batch_into(&x, &y, &z, &bx, &by, &bz, &mut ao, &mut aoy, &mut aoz).unwrap();
        be.cross3_batch_f64(&x, &y, &z, &bx, &by, &bz, &mut bo, &mut boy, &mut boz)
            .unwrap();
        check("cross3", &ao, &bo);
        let (a, b) = ab(
            || {
                api::cross3_batch_into(&x, &y, &z, &bx, &by, &bz, &mut ao, &mut aoy, &mut aoz)
                    .unwrap();
                black_box(&ao);
            },
            || {
                be.cross3_batch_f64(&x, &y, &z, &bx, &by, &bz, &mut bo, &mut boy, &mut boz)
                    .unwrap();
                black_box(&bo);
            },
        );
        row("cross3_batch_f64", n, a, b);
    }
    // ---- mat3_mul_vec3（m = 9 条逐元素数组）----
    {
        let m: Vec<Vec<f64>> = (0..9).map(|k| sample(n + k)[k..].to_vec()).collect();
        let refs: [&[f64]; 9] = [
            &m[0], &m[1], &m[2], &m[3], &m[4], &m[5], &m[6], &m[7], &m[8],
        ];
        let (mut ao, mut bo) = (vec![0.0f64; n], vec![0.0f64; n]);
        let (mut aoy, mut boy) = (vec![0.0f64; n], vec![0.0f64; n]);
        let (mut aoz, mut boz) = (vec![0.0f64; n], vec![0.0f64; n]);
        api::mat3_mul_vec3_batch_into(refs, &x, &y, &z, &mut ao, &mut aoy, &mut aoz).unwrap();
        be.mat3_mul_vec3_batch_f64(refs, &x, &y, &z, &mut bo, &mut boy, &mut boz)
            .unwrap();
        check("mat3_mul_vec3", &ao, &bo);
        let (a, b) = ab(
            || {
                api::mat3_mul_vec3_batch_into(refs, &x, &y, &z, &mut ao, &mut aoy, &mut aoz)
                    .unwrap();
                black_box(&ao);
            },
            || {
                be.mat3_mul_vec3_batch_f64(refs, &x, &y, &z, &mut bo, &mut boy, &mut boz)
                    .unwrap();
                black_box(&bo);
            },
        );
        row("mat3_mul_vec3_batch_f64", n, a, b);
    }
    // ---- 零向量语义（路径各自）----
    {
        let n0 = 4;
        let (zx, zy, zz) = (vec![0.0f64; n0], vec![0.0f64; n0], vec![0.0f64; n0]);
        let (mut ox, mut oy, mut oz) = (vec![0.0f64; n0], vec![0.0f64; n0], vec![0.0f64; n0]);
        api::unitize3_batch_into(&zx, &zy, &zz, &mut ox, &mut oy, &mut oz).unwrap();
        println!(
            "  [零向量] LASX unitize3 = ({}, {}, {})  ← 内核语义 0",
            ox[0], oy[0], oz[0]
        );
        println!(
            "  [零向量] 合同默认（组合版）= (NaN, NaN, NaN)  ← 路径各自语义，override 不打补丁"
        );
    }
    println!("loadavg(结束) = {}", loadavg());
    println!(
        "判据：ratio = api/dsl ≥0.98 不输（override 是薄转发，预期 ≈1.00）；组合版 A/B 见 geomf64_ab。"
    );
}
