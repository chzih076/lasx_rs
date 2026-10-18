//! **P1-4 A/B：LASX f64 逐元素 override vs portable 默认**（`map_binary/ternary/affine_f64`）。
//!
//! A 臂 = `simd_dsl::kernels::scalar::*_f64`（契约的 portable 默认实现，即"没有 override 的后端"走的路）；
//! B 臂 = `LasxBackend::*_f64`（本仓 `ops::f64_map` 的 LASX 4×f64 向量内核）。
//!
//! 口径：同进程交替、6 对丢首对取中位；每行先比 `to_bits()`（逐位/容差界）再谈性能；
//! 打印 `threads`/`ms`/ratio/`loadavg`。**目标 ≥2×**（标量→向量）；若某 op 因 f64 指令吞吐差 <1.1×
//! 或更慢，**如实记录**（不硬造"不输"）。
//!
//! 用法（`--release` 才有效；别并发跑别的 cargo/编译任务）：
//! ```text
//! cargo run --release --features dsl-backend --example f64_map_ab
//! ```

use std::hint::black_box;
use std::time::{Duration, Instant};

use lasx_rs::dsl_backend::LasxBackend;
use simd_dsl::kernels::scalar;
use simd_dsl::{Backend, BinaryOp, TernaryOp};

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
/// 逐位/容差判定：返回展示串。
fn class(got: &[f64], want: &[f64]) -> String {
    if got.len() == want.len()
        && got
            .iter()
            .zip(want)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    {
        "**逐位**".to_string()
    } else {
        let peak = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
        let rel = got
            .iter()
            .zip(want)
            .fold(0.0f64, |m, (a, b)| m.max((*a - *b).abs()))
            / peak;
        format!("容差 rel≤{rel:.2e}")
    }
}
fn row(name: &str, n: usize, a: f64, b: f64, eq: &str) {
    // 带宽口径：读 2 块 + 写 1 块 × 8B（近似；ternary 读 3 块，行内说明）
    let gbps = |ms: f64| (n as f64 * 24.0) / (ms / 1000.0) / 1e9;
    println!(
        "| {name} | {n} | {a:.3} | {b:.3} | {:.2}× | {:.1}/{:.1} | {eq} |",
        a / b,
        gbps(a),
        gbps(b)
    );
}

fn main() {
    let be = LasxBackend;
    println!("# P1-4 A/B：LASX f64 逐元素 override vs portable 默认（kernels::scalar）");
    println!(
        "- threads = {}；loadavg = `{}`（同进程交替 {ROUNDS} 对丢首对取中位）",
        lasx_rs::pool::global().threads(),
        loadavg()
    );
    println!("| 用例 | n | A(portable) ms | B(LASX) ms | 加速比 A/B | GB/s A/B | 等价类 |");
    println!("|---|---|---|---|---|---|---|");

    for &n in &[1usize << 20, 8 << 20] {
        let x: Vec<f64> = (0..n)
            .map(|i| ((i * 37 % 101) as f64) * 0.1 - 5.05)
            .collect();
        let y: Vec<f64> = (0..n).map(|i| ((i * 53 % 97) as f64) * 0.1 + 1.3).collect();
        let z: Vec<f64> = (0..n).map(|i| ((i * 11 % 89) as f64) * 0.1 - 0.7).collect();
        let mut out_a = vec![0.0f64; n];
        let mut out_b = vec![0.0f64; n];

        // ---- map_binary_f64：四种 op ----
        for (op, name) in [
            (BinaryOp::Add, "map_binary_f64 Add"),
            (BinaryOp::Sub, "map_binary_f64 Sub"),
            (BinaryOp::Mul, "map_binary_f64 Mul"),
            (BinaryOp::Div, "map_binary_f64 Div"),
        ] {
            scalar::map_binary_f64(&x, &y, &mut out_a, op).expect("scalar");
            let want = out_a.clone();
            be.map_binary_f64(&x, &y, &mut out_b, op).expect("backend");
            let eq = class(&out_b, &want);
            let (a, b) = ab(
                || {
                    scalar::map_binary_f64(&x, &y, &mut out_a, op).expect("scalar");
                    black_box(out_a[0]);
                },
                || {
                    be.map_binary_f64(&x, &y, &mut out_b, op).expect("backend");
                    black_box(out_b[0]);
                },
            );
            row(name, n, a, b, &eq);
        }

        // ---- map_ternary_f64（SubSub；读 3 块）----
        {
            scalar::map_ternary_f64(&x, &y, &z, &mut out_a, TernaryOp::SubSub).expect("scalar");
            let want = out_a.clone();
            be.map_ternary_f64(&x, &y, &z, &mut out_b, TernaryOp::SubSub)
                .expect("backend");
            let eq = class(&out_b, &want);
            let (a, b) = ab(
                || {
                    scalar::map_ternary_f64(&x, &y, &z, &mut out_a, TernaryOp::SubSub)
                        .expect("scalar");
                    black_box(out_a[0]);
                },
                || {
                    be.map_ternary_f64(&x, &y, &z, &mut out_b, TernaryOp::SubSub)
                        .expect("backend");
                    black_box(out_b[0]);
                },
            );
            row("map_ternary_f64 SubSub", n, a, b, &eq);
        }

        // ---- map_affine_f64（非 FMA ⇒ 逐位）----
        {
            scalar::map_affine_f64(&x, 0.3, -1.7, &mut out_a).expect("scalar");
            let want = out_a.clone();
            be.map_affine_f64(&x, 0.3, -1.7, &mut out_b)
                .expect("backend");
            let eq = class(&out_b, &want);
            let (a, b) = ab(
                || {
                    scalar::map_affine_f64(&x, 0.3, -1.7, &mut out_a).expect("scalar");
                    black_box(out_a[0]);
                },
                || {
                    be.map_affine_f64(&x, 0.3, -1.7, &mut out_b)
                        .expect("backend");
                    black_box(out_b[0]);
                },
            );
            row("map_affine_f64 a·x+b", n, a, b, &eq);
        }
    }
    println!("\nloadavg（结束）= `{}`", loadavg());
    println!(
        "\n说明：A 臂 = `kernels::scalar::*_f64`（portable 默认）；B 臂 = `LasxBackend` 的 \
         `map_binary_f64`/`map_ternary_f64`/`map_affine_f64`（LASX 4×f64 向量内核，4 路展开 + 尾部标量）。\
         等价类全部**逐位**（affine 故意不融 FMA）。"
    );
}
