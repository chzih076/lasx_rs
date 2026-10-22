//! **完整性能报表（R19-F）**：同一进程内测三件事——
//! ① **DSL + LASX 后端**的完整性能；② **DSL + 通用可移植后端**（`IdiomaticBackend`）；
//! ③ **中间抽象层损失**（同一后端下 `DSL 宏+引擎` vs `原语直调`）。
//!
//! **口径（`simd_dsl/docs/performance.md` §5）**：同一目的缓冲、同进程交替（6 轮丢第 1 轮）、
//! 每臂每轮 `iters` 次取均值，输出**中位 + 极差 + `iters`**；打印 `loadavg`；**正确性在计时之外**。
//!
//! **三层术语**：
//! * **L0（原语直调）**＝不经 DSL 层：`lasx_rs::api::*`（手写 floor）或 `Backend::方法` 直调；
//! * **DSL**＝`matmul!`/`dot!`/`sum!`/`wsum!`/`affine!`/`combine!`/`kernel!` → 引擎 → 后端；
//! * **抽象层损失** ＝ `DSL 臂 / 同后端 L0 臂`（>1 的部分＝宏解析+引擎编排+分派的代价）。
//!
//! **无 L0 的行**：轴归约/加权归约在本仓没有手写公开入口（LASX 内核只经 `Backend`
//! 暴露）⇒ 只给档位差，**不编造 L0**。整帧谱检测已迁 `simd_dsl` 的
//! `ops_dsl` 示例 `spectral_detect`：
//! `cargo run --release -p ops_dsl --example spectral_detect`（`--mode frame-ab` = 可移植档
//! rebuild vs reuse 跨帧 A/B + 逐位；`--check` = digest/池峰值/入口 stages 自检）。
//!
//! 复现：`cargo run --release --features dsl-backend --example dsl_perf_report`
#![cfg(feature = "dsl-backend")]

use std::cell::RefCell;
use std::time::Instant;

use lasx_rs::dsl_backend::LasxBackend;
use simd_dsl::{
    Backend, BinaryOp, IdiomaticBackend, KernelSpec, affine, combine, dot, kernel, matmul, sum,
    wsum,
};

fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .map(|s| s.split_whitespace().take(3).collect::<Vec<_>>().join(" "))
        .unwrap_or_else(|_| "?".into())
}

fn measure_once<F: FnMut()>(f: &mut F, budget_s: f64) -> (f64, u32) {
    let mut iters = 0u32;
    let mut sum_s = 0.0f64;
    let t0 = Instant::now();
    while t0.elapsed().as_secs_f64() < budget_s {
        let t1 = Instant::now();
        f();
        sum_s += t1.elapsed().as_secs_f64();
        iters += 1;
    }
    let mean = if iters == 0 {
        f64::NAN
    } else {
        sum_s / iters as f64
    };
    (mean * 1e3, iters)
}

fn median_range(v: &[f64]) -> (f64, f64) {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    (s[s.len() / 2], s[s.len() - 1] - s[0])
}

struct Arm<'a> {
    name: &'static str,
    f: Box<dyn FnMut() + 'a>,
    ms: Vec<f64>,
    iters: u32,
}

impl<'a> Arm<'a> {
    fn new(name: &'static str, f: impl FnMut() + 'a) -> Self {
        Self {
            name,
            f: Box::new(f),
            ms: Vec::new(),
            iters: 0,
        }
    }
    fn med(&self) -> f64 {
        median_range(&self.ms).0
    }
}

/// 6 轮轮转（丢第 1 轮）；打印 markdown 表 + 派生比值。
fn run<'a>(tag: &str, title: &str, mut arms: Vec<Arm<'a>>) {
    for _round in 0..6 {
        for arm in arms.iter_mut() {
            let (ms, it) = measure_once(&mut arm.f, 0.12);
            arm.iters = it;
            arm.ms.push(ms);
        }
    }
    for arm in &mut arms {
        arm.ms.remove(0);
    }
    println!("### {tag} {title}");
    println!("| 臂 | 中位 ms | 极差 | iters |");
    println!("|---|---|---|---|");
    for arm in &arms {
        let (med, rng) = median_range(&arm.ms);
        println!("| {} | {med:.4} | {rng:.4} | {} |", arm.name, arm.iters);
    }
    let get = |name: &str| arms.iter().find(|a| a.name == name).map(Arm::med);
    let mut parts = Vec::new();
    if let (Some(l0), Some(d)) = (get("L0"), get("DSL+LASX")) {
        parts.push(format!("层损(LASX)=**{:.3}×**", d / l0));
    }
    if let (Some(l0), Some(d)) = (get("L0_port"), get("DSL+可移植")) {
        parts.push(format!("层损(可移植)=**{:.3}×**", d / l0));
    }
    if let (Some(l), Some(p)) = (get("DSL+LASX"), get("DSL+可移植")) {
        parts.push(format!("可移植/LASX=**{:.3}×**", p / l));
    }
    if !parts.is_empty() {
        println!("⇒ {}", parts.join("；"));
    }
    println!();
}

fn main() {
    println!("# DSL 完整性能报表（R19-F）\n");
    println!("loadavg = {}\n", loadavg());
    let (lasx, port) = (LasxBackend, IdiomaticBackend);

    // ---- W1: 方阵 matmul 512³（计算密集）----
    {
        let (m, k, n) = (512usize, 512usize, 512usize);
        let a: Vec<f32> = (0..m * k).map(|i| (i % 97) as f32 * 0.01).collect();
        let w: Vec<f32> = (0..k * n).map(|i| (i % 89) as f32 * 0.01).collect();
        let o = RefCell::new(vec![0.0f32; m * n]);
        let arms = vec![
            Arm::new("L0", || {
                let mut b = o.borrow_mut();
                lasx_rs::api::matmul_into(m, k, n, &a, &w, b.as_mut_slice()).unwrap()
            }),
            Arm::new("L0_port", || {
                let mut b = o.borrow_mut();
                port.matmul_f32(m, k, n, &a, &w, b.as_mut_slice()).unwrap()
            }),
            Arm::new("DSL+LASX", || {
                let mut b = o.borrow_mut();
                let mut out = b.as_mut_slice();
                matmul!(lasx; out[m, n] = a[m, k] * w[k, n]).unwrap()
            }),
            Arm::new("DSL+可移植", || {
                let mut b = o.borrow_mut();
                let mut out = b.as_mut_slice();
                matmul!(port; out[m, n] = a[m, k] * w[k, n]).unwrap()
            }),
        ];
        run("W1", &format!("matmul {m}³"), arms);
    }

    // ---- W2: 帧形状 matmul 513×256×4096（瘦长，`p≪k`）----
    {
        let (m, k, n) = (513usize, 4096usize, 256usize);
        let a: Vec<f32> = (0..m * k).map(|i| (i % 97) as f32 * 0.01).collect();
        let w: Vec<f32> = (0..k * n).map(|i| (i % 89) as f32 * 0.01).collect();
        let o = RefCell::new(vec![0.0f32; m * n]);
        let arms = vec![
            Arm::new("L0", || {
                let mut b = o.borrow_mut();
                lasx_rs::api::matmul_into(m, k, n, &a, &w, b.as_mut_slice()).unwrap()
            }),
            Arm::new("L0_port", || {
                let mut b = o.borrow_mut();
                port.matmul_f32(m, k, n, &a, &w, b.as_mut_slice()).unwrap()
            }),
            Arm::new("DSL+LASX", || {
                let mut b = o.borrow_mut();
                let mut out = b.as_mut_slice();
                matmul!(lasx; out[m, n] = a[m, k] * w[k, n]).unwrap()
            }),
            Arm::new("DSL+可移植", || {
                let mut b = o.borrow_mut();
                let mut out = b.as_mut_slice();
                matmul!(port; out[m, n] = a[m, k] * w[k, n]).unwrap()
            }),
        ];
        run("W2", &format!("matmul 帧形状 {m}×{n}×{k}"), arms);
    }

    // ---- W3: dot 1M ----
    {
        let k = 1usize << 20;
        let a: Vec<f32> = (0..k).map(|i| (i % 97) as f32 * 0.01).collect();
        let b: Vec<f32> = (0..k).map(|i| (i % 89) as f32 * 0.01).collect();
        let out = RefCell::new(0.0f32);
        let arms = vec![
            Arm::new("L0", || {
                *out.borrow_mut() = lasx_rs::api::dot(&a, &b).unwrap();
            }),
            Arm::new("L0_port", || {
                *out.borrow_mut() = port.dot_f32(&a, &b).unwrap();
            }),
            Arm::new("DSL+LASX", || {
                *out.borrow_mut() = dot!(lasx; a[k] * b[k]).unwrap();
            }),
            Arm::new("DSL+可移植", || {
                *out.borrow_mut() = dot!(port; a[k] * b[k]).unwrap();
            }),
        ];
        run("W3", &format!("dot n={k}"), arms);
    }

    // ---- W4: 归约 Σ 1M ----
    {
        let k = 1usize << 20;
        let x: Vec<f32> = (0..k).map(|i| (i % 97) as f32 * 0.01).collect();
        let out = RefCell::new(0.0f32);
        let arms = vec![
            Arm::new("L0", || {
                *out.borrow_mut() = lasx_rs::api::sum(&x);
            }),
            Arm::new("L0_port", || {
                *out.borrow_mut() = port.reduce_sum_f32(&x).unwrap();
            }),
            Arm::new("DSL+LASX", || {
                *out.borrow_mut() = sum!(lasx; x[k]).unwrap();
            }),
            Arm::new("DSL+可移植", || {
                *out.borrow_mut() = sum!(port; x[k]).unwrap();
            }),
        ];
        run("W4", &format!("reduce_sum n={k}"), arms);
    }

    // ---- W5: 仿射 out = a·x + b（1M）----
    {
        let k = 1usize << 20;
        let x: Vec<f32> = (0..k).map(|i| (i % 97) as f32 * 0.01).collect();
        let o = RefCell::new(vec![0.0f32; k]);
        let arms = vec![
            Arm::new("L0", || {
                let mut b = o.borrow_mut();
                lasx.map_affine_f32(&x, 2.5, 1.0, b.as_mut_slice()).unwrap()
            }),
            Arm::new("L0_port", || {
                let mut b = o.borrow_mut();
                port.map_affine_f32(&x, 2.5, 1.0, b.as_mut_slice()).unwrap()
            }),
            Arm::new("DSL+LASX", || {
                let mut b = o.borrow_mut();
                let mut out = b.as_mut_slice();
                affine!(lasx; out[k] = 2.5 * x[k] + 1.0).unwrap()
            }),
            Arm::new("DSL+可移植", || {
                let mut b = o.borrow_mut();
                let mut out = b.as_mut_slice();
                affine!(port; out[k] = 2.5 * x[k] + 1.0).unwrap()
            }),
        ];
        run("W5", &format!("map_affine n={k}"), arms);
    }

    // ---- W6: 逐元素二元 out = x + y（1M）----
    {
        let k = 1usize << 20;
        let x: Vec<f32> = (0..k).map(|i| (i % 97) as f32 * 0.01).collect();
        let y: Vec<f32> = (0..k).map(|i| (i % 89) as f32 * 0.01).collect();
        let o = RefCell::new(vec![0.0f32; k]);
        let arms = vec![
            Arm::new("L0", || {
                let mut b = o.borrow_mut();
                lasx.map_binary_f32(&x, &y, b.as_mut_slice(), BinaryOp::Add)
                    .unwrap()
            }),
            Arm::new("L0_port", || {
                let mut b = o.borrow_mut();
                port.map_binary_f32(&x, &y, b.as_mut_slice(), BinaryOp::Add)
                    .unwrap()
            }),
            Arm::new("DSL+LASX", || {
                let mut b = o.borrow_mut();
                let mut out = b.as_mut_slice();
                combine!(lasx; out[k] = x[k] + y[k]).unwrap()
            }),
            Arm::new("DSL+可移植", || {
                let mut b = o.borrow_mut();
                let mut out = b.as_mut_slice();
                combine!(port; out[k] = x[k] + y[k]).unwrap()
            }),
        ];
        run("W6", &format!("map_binary add n={k}"), arms);
    }

    // ---- W7: 轴归约（沿第一轴，4096×256）----
    {
        let (rows, cols) = (4096usize, 256usize);
        let (r, c) = (rows, cols); // 宏要求**同名维度变量**在作用域内（`x[r, c]`、`out[c]`）
        let x: Vec<f32> = (0..rows * cols).map(|i| (i % 97) as f32 * 0.01).collect();
        let o = RefCell::new(vec![0.0f32; cols]);
        let arms = vec![
            Arm::new("L0", || {
                let mut b = o.borrow_mut();
                lasx.reduce_sum_cols_f32(&x, rows, cols, b.as_mut_slice())
                    .unwrap()
            }),
            Arm::new("L0_port", || {
                let mut b = o.borrow_mut();
                port.reduce_sum_cols_f32(&x, rows, cols, b.as_mut_slice())
                    .unwrap()
            }),
            Arm::new("DSL+LASX", || {
                let mut b = o.borrow_mut();
                let mut out = b.as_mut_slice();
                sum!(lasx; out[c] = x[r, c]).unwrap()
            }),
            Arm::new("DSL+可移植", || {
                let mut b = o.borrow_mut();
                let mut out = b.as_mut_slice();
                sum!(port; out[c] = x[r, c]).unwrap()
            }),
        ];
        run("W7", &format!("轴归约 cols {rows}×{cols}"), arms);
    }

    // ---- W8: 加权轴归约（513×256）----
    {
        let (rows, cols) = (513usize, 256usize);
        let (r, c) = (rows, cols); // 同上：宏要 `x[r, c]`/`w[r]`/`out[c]` 同名维度变量
        let x: Vec<f32> = (0..rows * cols).map(|i| (i % 97) as f32 * 0.003).collect();
        let w: Vec<f32> = (0..rows).map(|i| 1.0 / (1.0 + i as f32)).collect();
        let o = RefCell::new(vec![0.0f32; cols]);
        let arms = vec![
            Arm::new("L0", || {
                let mut b = o.borrow_mut();
                lasx.reduce_weighted_sum_cols_f32(&x, &w, rows, cols, b.as_mut_slice())
                    .unwrap()
            }),
            Arm::new("L0_port", || {
                let mut b = o.borrow_mut();
                port.reduce_weighted_sum_cols_f32(&x, &w, rows, cols, b.as_mut_slice())
                    .unwrap()
            }),
            Arm::new("DSL+LASX", || {
                let mut b = o.borrow_mut();
                let mut out = b.as_mut_slice();
                wsum!(lasx; out[c] = x[r, c] * w[r]).unwrap()
            }),
            Arm::new("DSL+可移植", || {
                let mut b = o.borrow_mut();
                let mut out = b.as_mut_slice();
                wsum!(port; out[c] = x[r, c] * w[r]).unwrap()
            }),
        ];
        run("W8", &format!("加权轴归约 {rows}×{cols}"), arms);
    }

    // ---- W9: 等距核构造 kernel!（rows=64, n=1024）----
    {
        let (rows, n) = (64usize, 1024usize);
        let dt = 1.0f32 / 4096.0;
        let sigma: Vec<f32> = (0..rows).map(|i| 1.0 + i as f32 * 0.05).collect();
        let omega: Vec<f32> = (0..rows).map(|i| 0.01 * (1 + i % 7) as f32).collect();
        let spec = KernelSpec::new(rows, n, dt, &sigma, &omega, true);
        let kc = RefCell::new(vec![0.0f32; rows * n]);
        let ks = RefCell::new(vec![0.0f32; rows * n]);
        let arms = vec![
            Arm::new("L0", || {
                let mut a = kc.borrow_mut();
                let mut b = ks.borrow_mut();
                lasx.build_kernel_f32(&spec, a.as_mut_slice(), b.as_mut_slice())
                    .unwrap();
            }),
            Arm::new("L0_port", || {
                let mut a = kc.borrow_mut();
                let mut b = ks.borrow_mut();
                port.build_kernel_f32(&spec, a.as_mut_slice(), b.as_mut_slice())
                    .unwrap();
            }),
            Arm::new("DSL+LASX", || {
                let mut a = kc.borrow_mut();
                let mut b = ks.borrow_mut();
                let mut kc = a.as_mut_slice();
                let mut ks = b.as_mut_slice();
                kernel!(lasx; kc, ks, spec).unwrap();
            }),
            Arm::new("DSL+可移植", || {
                let mut a = kc.borrow_mut();
                let mut b = ks.borrow_mut();
                let mut kc = a.as_mut_slice();
                let mut ks = b.as_mut_slice();
                kernel!(port; kc, ks, spec).unwrap();
            }),
        ];
        run("W9", &format!("等距核构造 {rows}×{n}"), arms);
    }

    // ---- W11: 原地按列 FFT 本体（4096×256，帧里 DFT 的核心；R19-H 接 LASX）----
    {
        let (n, b) = (4096usize, 256usize);
        let re0: Vec<f32> = (0..n * b)
            .map(|i| ((i * 37 % 101) as f32) * 0.05 - 2.5)
            .collect();
        let im0: Vec<f32> = (0..n * b)
            .map(|i| ((i * 11 % 71) as f32) * 0.03 - 1.0)
            .collect();
        let (rc, ic) = (RefCell::new(re0), RefCell::new(im0));
        let arms = vec![
            // 可移植参照（`kernels::scalar` 的 f64 内部实现）**直调** ⇒ LASX 臂的"手写照"。
            // 注意：每跑一次输入就被变换一次（数据随时序变化，但耗时与数据无关）；**正确性另在测试里验**。
            Arm::new("L0_port", || {
                let mut r = rc.borrow_mut();
                let mut i = ic.borrow_mut();
                simd_dsl::kernels::scalar::fft_radix2_strided_f32(
                    r.as_mut_slice(),
                    i.as_mut_slice(),
                    n,
                    b,
                )
                .unwrap();
            }),
            Arm::new("DSL+LASX", || {
                let mut r = rc.borrow_mut();
                let mut i = ic.borrow_mut();
                lasx.fft_strided_f32(r.as_mut_slice(), i.as_mut_slice(), n, b)
                    .unwrap();
            }),
            Arm::new("DSL+可移植", || {
                let mut r = rc.borrow_mut();
                let mut i = ic.borrow_mut();
                port.fft_strided_f32(r.as_mut_slice(), i.as_mut_slice(), n, b)
                    .unwrap();
            }),
        ];
        run("W11", &format!("按列 FFT n={n} b={b}"), arms);
    }
}
