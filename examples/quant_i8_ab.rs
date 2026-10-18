//! **P1-5 A/B：i8 量化族 override（LASX 内核）vs portable 默认**。
//!
//! A 臂 = `simd_dsl::kernels::scalar::*`（契约默认实现）；B 臂 = `LasxBackend` 的 override
//! （`quantize_i8_rows_f32` / `dot_i8` / `gemv_i8`；`dequantize_i8_rows_f32` 本批**未接**，见提交说明）。
//!
//! 口径：同进程交替、6 对丢首对取中位；每项先比**逐位/整数相等**再谈倍率；打印
//! `threads`/`ms`/ratio/`loadavg`。**≥1.10× 才保留该 override**（不够就回退并如实记）。
//!
//! 用法（`--release` 才有效；别并发跑别的 cargo/编译任务）：
//! ```text
//! cargo run --release --features dsl-backend --example quant_i8_ab
//! ```

use std::hint::black_box;
use std::time::{Duration, Instant};

use lasx_rs::dsl_backend::LasxBackend;
use simd_dsl::Backend;
use simd_dsl::kernels::scalar;

const ROUNDS: usize = 6;

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
fn row(name: &str, a: f64, b: f64, eq: &str) {
    println!("| {name} | {a:.3} | {b:.3} | {:.2}× | {eq} |", a / b);
}

fn main() {
    let be = LasxBackend;
    println!("# P1-5 A/B：i8 量化族 override vs portable 默认（后端 = LasxBackend）");
    println!(
        "- threads = {}；loadavg = `{}`（同进程交替 {ROUNDS} 对丢首对取中位）",
        lasx_rs::pool::global().threads(),
        loadavg()
    );
    println!("| 用例 | A(portable) ms | B(LASX) ms | 加速比 A/B | 等价 |");
    println!("|---|---|---|---|---|");

    // ---- quantize_i8_rows_f32：512 行 × 2048 列 = 1M 元素 ----
    {
        let (rows, cols) = (512usize, 2048usize);
        let total = rows * cols;
        let x: Vec<f32> = (0..total)
            .map(|i| ((i * 37 % 101) as f32) * 0.03125 - 1.5)
            .collect();
        let scale: Vec<f32> = (0..rows).map(|r| 0.02 + (r % 7) as f32 * 0.01).collect();
        let mut qa = vec![0i8; total];
        let mut qb = vec![0i8; total];
        scalar::quantize_i8_rows_f32(&x, rows, cols, &scale, &mut qa).unwrap();
        be.quantize_i8_rows_f32(&x, rows, cols, &scale, &mut qb)
            .unwrap();
        let eq = if qa == qb {
            "**逐位**"
        } else {
            "不逐位！"
        };
        let (a, b) = ab(
            || {
                scalar::quantize_i8_rows_f32(&x, rows, cols, &scale, &mut qa).unwrap();
                black_box(qa[0]);
            },
            || {
                be.quantize_i8_rows_f32(&x, rows, cols, &scale, &mut qb)
                    .unwrap();
                black_box(qb[0]);
            },
        );
        row("quantize_i8_rows_f32（1M）", a, b, eq);
    }

    // ---- dot_i8：n = 1M ----
    {
        let n = 1usize << 20;
        let a: Vec<i8> = (0..n)
            .map(|i| ((i * 53 % 251) as i32 - 125) as i8)
            .collect();
        let b: Vec<i8> = (0..n)
            .map(|i| ((i * 97 % 241) as i32 - 120) as i8)
            .collect();
        let want = scalar::dot_i8(&a, &b).unwrap();
        let got = be.dot_i8(&a, &b).unwrap();
        let eq = if want == got {
            "**逐位**（整数）"
        } else {
            "不相等！"
        };
        let (ta, tb) = ab(
            || {
                black_box(scalar::dot_i8(&a, &b).unwrap());
            },
            || {
                black_box(be.dot_i8(&a, &b).unwrap());
            },
        );
        row("dot_i8（1M）", ta, tb, eq);
    }

    // ---- gemv_i8：j = 1024, k = 1024 ----
    {
        let (j, k) = (1024usize, 1024usize);
        let a: Vec<i8> = (0..j * k)
            .map(|i| ((i * 41 % 251) as i32 - 125) as i8)
            .collect();
        let x: Vec<i8> = (0..k)
            .map(|i| ((i * 59 % 241) as i32 - 120) as i8)
            .collect();
        let mut outa = vec![0i32; j];
        let mut outb = vec![0i32; j];
        scalar::gemv_i8(j, k, &a, &x, &mut outa).unwrap();
        be.gemv_i8(j, k, &a, &x, &mut outb).unwrap();
        let eq = if outa == outb {
            "**逐位**（整数）"
        } else {
            "不相等！"
        };
        let (ta, tb) = ab(
            || {
                scalar::gemv_i8(j, k, &a, &x, &mut outa).unwrap();
                black_box(outa[0]);
            },
            || {
                be.gemv_i8(j, k, &a, &x, &mut outb).unwrap();
                black_box(outb[0]);
            },
        );
        row("gemv_i8（1024×1024）", ta, tb, eq);
    }

    println!("\nloadavg（结束）= `{}`", loadavg());
    println!(
        "\n说明：A = `kernels::scalar`（portable 默认），B = `LasxBackend` override。\
         判据 **≥1.10× 才保留该 override**；`dequantize_i8_rows_f32` 本批未接（LASX 拓宽指令 128-bit \
         lane 语义，连续 i8 块需多次 load+重排，收益不足）⇒ `Capability::Quant` 不置真。"
    );
}
