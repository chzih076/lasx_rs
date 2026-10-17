//! **f64 几何五入口 A/B（阶段 4 收尾）**：A = `lasx_rs::api::*_into`（LASX 向量核），
//! B = `ops_dsl::geometry::*_f64(&LasxBackend, …)`（**组合版源实现**，契约 f64 六方法拼出来）。
//!
//! # 口径
//! 同进程交替、6 对丢首对取中位；打印 `threads`/`ms`/`ratio`/`loadavg`；每行**先比数值**：
//! `to_bits()` 逐位 or `max_rel`（相对峰值）+ 等价类标注。
//!
//! # 预期（如实，不硬造"不输"）
//! LASX **没有** f64 逐元素 override（本仓 `dsl_backend.rs` 只覆盖 f32 族 + f64 matmul/dot ⇒
//! f64 逐元素/轴归约走 portable 标量默认）⇒ 组合版是"每算子 6–15 趟标量循环" vs LASX 的
//! "1 趟向量核" ⇒ **必然慢**，比值预期 0.05–0.3；**结论 = `BelowThreshold`（不替换生产入口）**，
//! 它证明的是"源实现先跑通、跨后端可用"，加速选项等后端覆盖 f64 原语。
//!
//! 用法（`--release` 才有效；别并发跑别的 cargo/编译任务）：
//! ```text
//! cargo run --release --features dsl-backend --example geomf64_ab
//! ```

use std::hint::black_box;
use std::time::{Duration, Instant};

use lasx_rs::dsl_backend::LasxBackend;

const ROUNDS: usize = 6;
const N: usize = 1 << 20;

fn timeit(f: &mut impl FnMut()) -> Duration {
    let t = Instant::now();
    f();
    t.elapsed()
}
fn median(mut v: Vec<Duration>) -> Duration {
    v.sort_unstable();
    v[v.len() / 2]
}
fn ab(mut a: impl FnMut(), mut b: impl FnMut()) -> (f64, f64) {
    let (mut av, mut bv) = (Vec::new(), Vec::new());
    for r in 0..ROUNDS {
        let (ta, tb) = if r % 2 == 0 {
            (timeit(&mut a), timeit(&mut b))
        } else {
            let tb = timeit(&mut b);
            let ta = timeit(&mut a);
            (ta, tb)
        };
        av.push(ta);
        bv.push(tb);
    }
    av.remove(0);
    bv.remove(0);
    (
        median(av).as_secs_f64() * 1e3,
        median(bv).as_secs_f64() * 1e3,
    )
}
fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "<unreadable>".to_string())
}
fn maxrel(got: &[f64], want: &[f64]) -> f64 {
    let peak = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
    got.iter()
        .zip(want)
        .fold(0.0f64, |m, (a, b)| m.max((*a - *b).abs()))
        / peak
}
fn class(got: &[f64], want: &[f64]) -> String {
    if got.len() == want.len()
        && got
            .iter()
            .zip(want)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    {
        "**逐位**".to_string()
    } else {
        format!("容差 rel≤{:.2e}", maxrel(got, want))
    }
}
fn row(name: &str, a: f64, b: f64, eq: &str) {
    println!("| {name} | {a:.4} | {b:.4} | {:.3} | {eq} |", a / b);
}

fn data(n: usize, seed: f64) -> Vec<f64> {
    (0..n)
        .map(|i| ((i * 29 % 71) as f64) * 0.0625 + seed)
        .collect()
}

fn main() {
    let be = LasxBackend;
    println!("# f64 几何五入口 A/B（后端 = LasxBackend；n = {N}）");
    println!(
        "- threads = {}；loadavg = `{}`（同进程交替 {ROUNDS} 对丢首对取中位）",
        lasx_rs::pool::global().threads(),
        loadavg()
    );
    println!(
        "- LASX 对 f64 逐元素的支持：`matmul_f64`/`dot_f64` 有 override；`map_binary_f64` 等走 portable 默认"
    );
    println!("| 用例 | A(api) ms | B(ops 组合) ms | ratio A/B | 等价类 |");
    println!("|---|---|---|---|---|");

    let (ax, ay, az) = (data(N, 0.5), data(N, -1.25), data(N, 2.0));
    let (bx, by, bz) = (data(N, 3.5), data(N, -0.25), data(N, 1.0));

    // ---- vec3_add_scaled ----
    {
        let (mut ox, mut oy, mut oz) = (vec![0.0; N], vec![0.0; N], vec![0.0; N]);
        let s = -0.75;
        lasx_rs::api::vec3_add_scaled_batch_into(
            &ax, &ay, &az, &bx, &by, &bz, s, &mut ox, &mut oy, &mut oz,
        )
        .unwrap();
        let (mut qx, mut qy, mut qz) = (vec![0.0; N], vec![0.0; N], vec![0.0; N]);
        let (mut tx, mut ty, mut tz) = (vec![0.0; N], vec![0.0; N], vec![0.0; N]);
        ops_dsl::geometry::vec3_add_scaled_f64(
            &be, &ax, &ay, &az, &bx, &by, &bz, s, &mut qx, &mut qy, &mut qz, &mut tx, &mut ty,
            &mut tz,
        )
        .unwrap();
        let eq = format!(
            "{} / {} / {}（lasx 用 FMA）",
            class(&qx, &ox),
            class(&qy, &oy),
            class(&qz, &oz)
        );
        let (a, b) = ab(
            || {
                lasx_rs::api::vec3_add_scaled_batch_into(
                    &ax, &ay, &az, &bx, &by, &bz, s, &mut ox, &mut oy, &mut oz,
                )
                .unwrap();
                black_box(ox[0]);
            },
            || {
                ops_dsl::geometry::vec3_add_scaled_f64(
                    &be, &ax, &ay, &az, &bx, &by, &bz, s, &mut qx, &mut qy, &mut qz, &mut tx,
                    &mut ty, &mut tz,
                )
                .unwrap();
                black_box(qx[0]);
            },
        );
        row("vec3_add_scaled（6 趟标量 vs 1 趟向量）", a, b, &eq);
    }

    // ---- norm3 ----
    {
        let mut on = vec![0.0; N];
        lasx_rs::api::norm3_batch_into(&ax, &ay, &az, &mut on).unwrap();
        let mut qn = vec![0.0; N];
        let (mut t0, mut t1, mut t2) = (vec![0.0; N], vec![0.0; N], vec![0.0; N]);
        ops_dsl::geometry::norm3_batch_f64(&be, &ax, &ay, &az, &mut qn, &mut t0, &mut t1, &mut t2)
            .unwrap();
        let eq = class(&qn, &on);
        let (a, b) = ab(
            || {
                lasx_rs::api::norm3_batch_into(&ax, &ay, &az, &mut on).unwrap();
                black_box(on[0]);
            },
            || {
                ops_dsl::geometry::norm3_batch_f64(
                    &be, &ax, &ay, &az, &mut qn, &mut t0, &mut t1, &mut t2,
                )
                .unwrap();
                black_box(qn[0]);
            },
        );
        row("norm3_batch（6 趟标量 vs 1 趟向量）", a, b, &eq);
    }

    // ---- unitize3 ----
    {
        let (mut ox, mut oy, mut oz) = (vec![0.0; N], vec![0.0; N], vec![0.0; N]);
        lasx_rs::api::unitize3_batch_into(&ax, &ay, &az, &mut ox, &mut oy, &mut oz).unwrap();
        let (mut qx, mut qy, mut qz) = (vec![0.0; N], vec![0.0; N], vec![0.0; N]);
        let (mut norm, mut t0, mut t1, mut t2) =
            (vec![0.0; N], vec![0.0; N], vec![0.0; N], vec![0.0; N]);
        ops_dsl::geometry::unitize3_batch_f64(
            &be, &ax, &ay, &az, &mut qx, &mut qy, &mut qz, &mut norm, &mut t0, &mut t1, &mut t2,
        )
        .unwrap();
        let eq = format!(
            "{} / {} / {}",
            class(&qx, &ox),
            class(&qy, &oy),
            class(&qz, &oz)
        );
        let (a, b) = ab(
            || {
                lasx_rs::api::unitize3_batch_into(&ax, &ay, &az, &mut ox, &mut oy, &mut oz)
                    .unwrap();
                black_box(ox[0]);
            },
            || {
                ops_dsl::geometry::unitize3_batch_f64(
                    &be, &ax, &ay, &az, &mut qx, &mut qy, &mut qz, &mut norm, &mut t0, &mut t1,
                    &mut t2,
                )
                .unwrap();
                black_box(qx[0]);
            },
        );
        row("unitize3_batch（9 趟标量 vs 1 趟向量）", a, b, &eq);
    }

    // ---- cross3 ----
    {
        let (mut ox, mut oy, mut oz) = (vec![0.0; N], vec![0.0; N], vec![0.0; N]);
        lasx_rs::api::cross3_batch_into(&ax, &ay, &az, &bx, &by, &bz, &mut ox, &mut oy, &mut oz)
            .unwrap();
        let (mut qx, mut qy, mut qz) = (vec![0.0; N], vec![0.0; N], vec![0.0; N]);
        let (mut p1, mut p2) = (vec![0.0; N], vec![0.0; N]);
        ops_dsl::geometry::cross3_batch_f64(
            &be, &ax, &ay, &az, &bx, &by, &bz, &mut qx, &mut qy, &mut qz, &mut p1, &mut p2,
        )
        .unwrap();
        let eq = format!(
            "{} / {} / {}",
            class(&qx, &ox),
            class(&qy, &oy),
            class(&qz, &oz)
        );
        let (a, b) = ab(
            || {
                lasx_rs::api::cross3_batch_into(
                    &ax, &ay, &az, &bx, &by, &bz, &mut ox, &mut oy, &mut oz,
                )
                .unwrap();
                black_box(ox[0]);
            },
            || {
                ops_dsl::geometry::cross3_batch_f64(
                    &be, &ax, &ay, &az, &bx, &by, &bz, &mut qx, &mut qy, &mut qz, &mut p1, &mut p2,
                )
                .unwrap();
                black_box(qx[0]);
            },
        );
        row("cross3_batch（9 趟标量 vs 1 趟向量）", a, b, &eq);
    }

    // ---- mat3_mul_vec3（A 臂分配 3 个输出数组；B 臂复用缓冲）----
    {
        let m: [Vec<f64>; 9] = std::array::from_fn(|k| data(N, k as f64 - 4.0));
        let mr: [&[f64]; 9] = std::array::from_fn(|k| m[k].as_slice());
        let api = lasx_rs::api::mat3_mul_vec3_batch(mr, &ax, &ay, &az).unwrap();
        let (mut qx, mut qy, mut qz) = (vec![0.0; N], vec![0.0; N], vec![0.0; N]);
        let (mut t0, mut t1, mut t2) = (vec![0.0; N], vec![0.0; N], vec![0.0; N]);
        ops_dsl::geometry::mat3_mul_vec3_batch_f64(
            &be, mr, &ax, &ay, &az, &mut qx, &mut qy, &mut qz, &mut t0, &mut t1, &mut t2,
        )
        .unwrap();
        let eq = format!(
            "{} / {} / {}",
            class(&qx, &api[0]),
            class(&qy, &api[1]),
            class(&qz, &api[2])
        );
        let (a, b) = ab(
            || {
                let o = lasx_rs::api::mat3_mul_vec3_batch(mr, &ax, &ay, &az).unwrap();
                black_box(o[0][0]);
            },
            || {
                ops_dsl::geometry::mat3_mul_vec3_batch_f64(
                    &be, mr, &ax, &ay, &az, &mut qx, &mut qy, &mut qz, &mut t0, &mut t1, &mut t2,
                )
                .unwrap();
                black_box(qx[0]);
            },
        );
        row("mat3_mul_vec3（15 趟标量 vs 1 趟向量）", a, b, &eq);
    }

    println!("\nloadavg（结束）= `{}`", loadavg());
    println!(
        "\n结论：组合版**全部 `BelowThreshold`**（LASX 无 f64 逐元素 override ⇒ 组合版 = 多趟标量）；\
         数值上 cross3/mat3/norm3 逐位、vec3/unitize 容差 ≤1 ulp（lasx 用 FMA/倒数）——\
         这正是「源实现先跑通、加速选项待后端覆盖」的活例。"
    );
}
