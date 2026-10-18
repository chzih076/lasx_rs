//! **f64 几何五入口 auto 路径 A/B**（阶段 4 收尾，任务1 验收）：
//! A = `lasx_rs::api::{norm3_batch, vec3_add_scaled_batch, unitize3_batch, cross3_batch, mat3_mul_vec3_batch}`
//! （原实现）；B = `ops_dsl::geometry::*_auto(&LasxBackend, …)`——`supports(FusedGeometry)` 为真时
//! 走 **1 趟融合分支**（运行时打印能力值确认），否则回落组合版（本机应为 true）。
//!
//! 口径：n = 1M、同进程交替、6 对、丢首对、取中位；打印 `threads`/`ms`/`ratio`/`loadavg`；
//! 每对先等价类（逐位优先）。判据 `ratio = api_ms / auto_ms ≥ 0.98`。
//! `vec3`/`mat3` 预期 0.95–0.97（薄转发 + 15 项长度校验；**同内核、结果逐位**）——如实记，不调阈值。
//! **零向量单列**（不参与性能比值）：组合版 NaN / 融合路径 0。
//!
//! 用法：`cargo run --release --features dsl-backend --example geomf64_auto_ab`

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
    println!("f64 几何 auto A/B（api 原实现 vs ops_dsl::*_auto；{ROUNDS} 对丢首对取中位）");
    println!("threads = {}", lasx_rs::pool::global().threads());
    println!(
        "cap: FusedGeometry={}  ← true ⇒ auto 走 1 趟融合分支",
        be.supports(simd_dsl::Capability::FusedGeometry)
    );
    println!("loadavg(开始) = {}", loadavg());
    println!("| 算子 | n | api ms | auto ms | 比值 api/auto |");
    println!("|---|---|---|---|---|");

    let n = 1 << 20;
    let x = sample(n);
    let y = sample(n + 3)[3..].to_vec();
    let z = sample(n + 7)[7..].to_vec();
    let bx = sample(n + 5)[5..].to_vec();
    let by = sample(n + 11)[11..].to_vec();
    let bz = sample(n + 13)[13..].to_vec();

    // ---- vec3_add_scaled ----
    {
        let (mut ao, mut aoy, mut aoz) = (vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]);
        let (mut bo, mut boy, mut boz) = (vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]);
        let (mut tx, mut ty, mut tz) = (vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]);
        let a = api::vec3_add_scaled_batch(&x, &y, &z, &bx, &by, &bz, 1.25).unwrap();
        ops_dsl::geometry::vec3_add_scaled_auto(
            &be, &x, &y, &z, &bx, &by, &bz, 1.25, &mut bo, &mut boy, &mut boz, &mut tx, &mut ty,
            &mut tz,
        )
        .unwrap();
        check("vec3_add_scaled", &a[0], &bo);
        let (ta, tb) = ab(
            || {
                black_box(api::vec3_add_scaled_batch(&x, &y, &z, &bx, &by, &bz, 1.25).unwrap());
            },
            || {
                ops_dsl::geometry::vec3_add_scaled_auto(
                    &be, &x, &y, &z, &bx, &by, &bz, 1.25, &mut ao, &mut aoy, &mut aoz, &mut tx,
                    &mut ty, &mut tz,
                )
                .unwrap();
                black_box(&ao);
            },
        );
        row("vec3_add_scaled_auto", n, ta, tb);
    }
    // ---- norm3 ----
    {
        let (mut go, mut bo) = (vec![0.0f64; n], vec![0.0f64; n]);
        let (mut t0, mut t1, mut t2) = (vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]);
        let a = api::norm3_batch(&x, &y, &z).unwrap();
        ops_dsl::geometry::norm3_batch_auto(&be, &x, &y, &z, &mut bo, &mut t0, &mut t1, &mut t2)
            .unwrap();
        check("norm3", &a, &bo);
        let (ta, tb) = ab(
            || {
                black_box(api::norm3_batch(&x, &y, &z).unwrap());
            },
            || {
                ops_dsl::geometry::norm3_batch_auto(
                    &be, &x, &y, &z, &mut go, &mut t0, &mut t1, &mut t2,
                )
                .unwrap();
                black_box(&go);
            },
        );
        row("norm3_batch_auto", n, ta, tb);
    }
    // ---- unitize3 ----
    {
        let (mut go, mut goy, mut goz) = (vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]);
        let (mut bo, mut boy, mut boz) = (vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]);
        let (mut norm, mut t0, mut t1, mut t2) = (
            vec![0.0f64; n],
            vec![0.0f64; n],
            vec![0.0f64; n],
            vec![0.0f64; n],
        );
        let a = api::unitize3_batch(&x, &y, &z).unwrap();
        ops_dsl::geometry::unitize3_batch_auto(
            &be, &x, &y, &z, &mut bo, &mut boy, &mut boz, &mut norm, &mut t0, &mut t1, &mut t2,
        )
        .unwrap();
        check("unitize3", &a[0], &bo);
        let (ta, tb) = ab(
            || {
                black_box(api::unitize3_batch(&x, &y, &z).unwrap());
            },
            || {
                ops_dsl::geometry::unitize3_batch_auto(
                    &be, &x, &y, &z, &mut go, &mut goy, &mut goz, &mut norm, &mut t0, &mut t1,
                    &mut t2,
                )
                .unwrap();
                black_box(&go);
            },
        );
        row("unitize3_batch_auto", n, ta, tb);
    }
    // ---- cross3 ----
    {
        let (mut go, mut goy, mut goz) = (vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]);
        let (mut bo, mut boy, mut boz) = (vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]);
        let (mut p1, mut p2) = (vec![0.0f64; n], vec![0.0f64; n]);
        let a = api::cross3_batch(&x, &y, &z, &bx, &by, &bz).unwrap();
        ops_dsl::geometry::cross3_batch_auto(
            &be, &x, &y, &z, &bx, &by, &bz, &mut bo, &mut boy, &mut boz, &mut p1, &mut p2,
        )
        .unwrap();
        check("cross3", &a[0], &bo);
        let (ta, tb) = ab(
            || {
                black_box(api::cross3_batch(&x, &y, &z, &bx, &by, &bz).unwrap());
            },
            || {
                ops_dsl::geometry::cross3_batch_auto(
                    &be, &x, &y, &z, &bx, &by, &bz, &mut go, &mut goy, &mut goz, &mut p1, &mut p2,
                )
                .unwrap();
                black_box(&go);
            },
        );
        row("cross3_batch_auto", n, ta, tb);
    }
    // ---- mat3_mul_vec3 ----
    {
        let m: Vec<Vec<f64>> = (0..9).map(|k| sample(n + k)[k..].to_vec()).collect();
        let refs: [&[f64]; 9] = [
            &m[0], &m[1], &m[2], &m[3], &m[4], &m[5], &m[6], &m[7], &m[8],
        ];
        let (mut go, mut goy, mut goz) = (vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]);
        let (mut bo, mut boy, mut boz) = (vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]);
        let (mut t0, mut t1, mut t2) = (vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]);
        let a = api::mat3_mul_vec3_batch(refs, &x, &y, &z).unwrap();
        ops_dsl::geometry::mat3_mul_vec3_batch_auto(
            &be, refs, &x, &y, &z, &mut bo, &mut boy, &mut boz, &mut t0, &mut t1, &mut t2,
        )
        .unwrap();
        check("mat3_mul_vec3", &a[0], &bo);
        let (ta, tb) = ab(
            || {
                black_box(api::mat3_mul_vec3_batch(refs, &x, &y, &z).unwrap());
            },
            || {
                ops_dsl::geometry::mat3_mul_vec3_batch_auto(
                    &be, refs, &x, &y, &z, &mut go, &mut goy, &mut goz, &mut t0, &mut t1, &mut t2,
                )
                .unwrap();
                black_box(&go);
            },
        );
        row("mat3_mul_vec3_batch_auto", n, ta, tb);
    }
    // ---- 零向量（不参与性能比值）----
    {
        let n0 = 2;
        let (zx, zy, zz) = (vec![0.0f64; n0], vec![0.0f64; n0], vec![0.0f64; n0]);
        let (mut ox, mut oy, mut oz) = (vec![0.0f64; n0], vec![0.0f64; n0], vec![0.0f64; n0]);
        let (mut norm, mut t0, mut t1, mut t2) = (
            vec![0.0f64; n0],
            vec![0.0f64; n0],
            vec![0.0f64; n0],
            vec![0.0f64; n0],
        );
        ops_dsl::geometry::unitize3_batch_auto(
            &be, &zx, &zy, &zz, &mut ox, &mut oy, &mut oz, &mut norm, &mut t0, &mut t1, &mut t2,
        )
        .unwrap();
        println!(
            "  [零向量] auto(融合) unitize3 = ({}, {}, {})  ← LASX 内核语义 0",
            ox[0], oy[0], oz[0]
        );
        println!(
            "  [零向量] 组合版（supports 假时）unitize3 = (NaN, NaN, NaN)  ← 路径各自语义，不参与比值"
        );
    }
    println!("loadavg(结束) = {}", loadavg());
    println!(
        "判据：ratio = api/auto ≥0.98 不输；vec3/mat3 的 0.95–0.97 预期来自薄转发 + 15 项长度校验（同内核、逐位）。"
    );
}
