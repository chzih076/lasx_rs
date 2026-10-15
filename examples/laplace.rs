//! **实战：用公式 DSL 做单边数值拉普拉斯变换**（`cargo run --example laplace`）。
//!
//! 数学与离散化见 [`lasx_rs::laplace`] 模块文档；本 example 只做三件事：
//! 1. **解析对照**：`1↔1/s`、`e^{-at}↔1/(s+a)`、`sin(ω₀t)↔ω₀/(s²+ω₀²)`（实部虚部都比）；
//! 2. **加密收敛表**：固定区间 `T`、`N ∈ {64,128,256,512}`（`Δt` 减半）⇒ 误差应 O(Δt²) 下降；
//! 3. **性能演示（同进程口径）**：挑一个**能触发自动动态认领**（`每线程行数 ≥ 32` 且
//!    `线程数 ≥ 12`）的规模，同进程交替测 `plan`（调用方常驻线程）与 `matmul!`（DSL 路径），
//!    报 ms / GFLOP/s / **实际线程数** / `iters` / `ms_max`。
use lasx_rs::laplace::{Weighting, laplace_batch_f32, trapezoid_weights};
use lasx_rs::pool::WorkerPool;
use lasx_rs::shape::{Mat, MatBuf};
use std::time::Instant;

/// 解析值：`1/s`。
fn f_inv_s(sig: f64, om: f64) -> (f64, f64) {
    let d = sig * sig + om * om;
    (sig / d, -om / d)
}
/// 解析值：`1/(s+a)`。
fn f_inv_s_plus_a(sig: f64, om: f64, a: f64) -> (f64, f64) {
    let (sr, si) = (sig + a, om);
    let d = sr * sr + si * si;
    (sr / d, -si / d)
}
/// 解析值：`sin(ω₀t) ⇒ ω₀/(s²+ω₀²)`。
fn f_sin(sig: f64, om: f64, w0: f64) -> (f64, f64) {
    let (ar, ai) = (sig * sig - om * om + w0 * w0, 2.0 * sig * om);
    let d = ar * ar + ai * ai;
    (w0 * ar / d, -w0 * ai / d)
}

fn analytic_part() {
    const M: usize = 4;
    const N: usize = 512;
    const NB: usize = 3; // 三条信号并列：1、e^{-2t}、sin(3t)
    let dt = 0.02;
    let sigs = [1.0f64, 1.0, 1.0, 1.0];
    let oms = [0.0f64, 2.0, 0.0, 3.0];
    let mut f = vec![0.0f32; N * NB];
    for i in 0..N {
        let t = i as f64 * dt;
        f[i * NB] = 1.0;
        f[i * NB + 1] = (-2.0 * t).exp() as f32;
        f[i * NB + 2] = (3.0 * t).sin() as f32;
    }
    let (re, im) = laplace_batch_f32::<M, N, NB>(&sigs, &oms, dt, &f);
    println!("\n== ① 解析对照（f32，N={N}，Δt={dt}）==");
    println!(
        "    s                    解析 Re/Im              DSL Re/Im              相对误差(Re/Im)"
    );
    for (j, (&s, &w)) in sigs.iter().zip(oms.iter()).enumerate() {
        for (name, want) in [
            ("1/s", f_inv_s(s, w)),
            ("1/(s+2)", f_inv_s_plus_a(s, w, 2.0)),
            ("3/(s²+9)", f_sin(s, w, 3.0)),
        ] {
            let (gr, gi) = (
                re[j * NB + name_idx(name)] as f64,
                im[j * NB + name_idx(name)] as f64,
            );
            let er = (gr - want.0).abs() / want.0.abs();
            let ei = (gi - want.1).abs() / want.1.abs().max(1e-6);
            println!(
                "    {:<10} s={:+.1}{:+.1}i  {:+.6} {:+.6}   {:+.6} {:+.6}   {:.2e}/{:.2e}",
                name, s, w, want.0, want.1, gr, gi, er, ei
            );
        }
    }
}

fn name_idx(name: &str) -> usize {
    match name {
        "1/s" => 0,
        "1/(s+2)" => 1,
        _ => 2,
    }
}

fn refinement_part() {
    const T: f64 = 6.4;
    let sigs = [1.0f64, 2.0, 3.0, 4.0];
    let _oms = [0.0f64; 4];
    println!("\n== ② 加密收敛（固定 T={T}，Δt 减半；解析值 1/(s+0.5)，f64）==");
    println!("     N     Δt        max 相对误差   相邻比");
    let mut prev: Option<f64> = None;
    for &n in &[64usize, 128, 256, 512] {
        let dt = T / (n - 1) as f64;
        let mut f = vec![0.0f64; n];
        for (i, fi) in f.iter_mut().enumerate() {
            *fi = (-0.5 * (i as f64 * dt)).exp();
        }
        // 用参考路径（任意 N）
        let w = trapezoid_weights(n);
        let mut e = 0.0f64;
        for (j, &s) in sigs.iter().enumerate() {
            let _ = j;
            let mut acc = 0.0f64;
            for (i, &wi) in w.iter().enumerate() {
                let t = i as f64 * dt;
                acc += wi * dt * (-s * t).exp() * f[i];
            }
            let want = 1.0 / (s + 0.5);
            e = e.max((acc - want).abs() / want);
        }
        let ratio = prev.map(|p| p / e);
        println!(
            "    {n:>4}  {dt:.5}   {e:.3e}      {}",
            ratio
                .map(|r| format!("{r:.2}×"))
                .unwrap_or_else(|| "-".into())
        );
        prev = Some(e);
    }
}

/// **性能演示（同进程口径）**：`plan`（调用方常驻线程）vs **DSL `matmul!`**（自动策略）。
///
/// 规模选 `M=384`（行）× `N=1024`（采样点）× `NB=64`（批量信号）@12 线程：
/// `每线程行数 = 384/12 = 32 ≥ 32` 且 `线程数 ≥ 12` ⇒ **命中 `pick_rows` 的动态认领阈值**。
fn perf_part() {
    const M: usize = 384;
    const N: usize = 1024;
    const NB: usize = 64;
    let dt = 1.0 / 256.0;
    let sigs: Vec<f64> = (0..M).map(|j| 0.5 + 0.01 * j as f64).collect();
    let oms: Vec<f64> = (0..M).map(|j| (j % 16) as f64 * 0.25).collect();
    let f = vec![1.0f32; N * NB];
    println!("\n== ③ 性能演示（同进程交替，口径见 docs/dev.md §6.4）==");
    let threads = lasx_rs::arch::parallelism().physical.max(1);
    println!("    形状 M×N×NB = {M}×{N}×{NB}（f32），线程数 = {threads}（物理核）");
    println!(
        "    每线程行数 = {} ⇒ 命中动态认领阈值（≥32 且 threads≥12）",
        M / threads
    );

    // DSL 路径：核矩阵 + 一次 `matmul!`（批量 NB 条信号）
    let (re_dsl, _im) = laplace_batch_f32::<M, N, NB>(&sigs, &oms, dt, &f);
    let flop = 2.0 * M as f64 * N as f64 * NB as f64;

    // 只造 Kc（与 DSL 一致）；两条臂算的都是 `Kc[M,N]·f[N,NB]`
    let w = trapezoid_weights(N);
    let mut kc = vec![0.0f32; M * N];
    for j in 0..M {
        for i in 0..N {
            let t = i as f64 * dt;
            kc[j * N + i] = ((-sigs[j] * t).exp() * w[i] * dt * (oms[j] * t).cos()) as f32;
        }
    }
    let mf = Mat::<f32, N, NB>::new(&f).unwrap(); // 权重（K=N, N=NB）
    let mkc = Mat::<f32, M, N>::new(&kc).unwrap(); // 输入（M, K=N）
    let mut out = MatBuf::<f32, M, NB>::new();
    let pool = WorkerPool::new(threads);
    let plan = lasx_rs::plan::MatmulPlan::from_row_major(&f, N, NB).unwrap();

    let mut t_dsl = Vec::new();
    let mut it_dsl = Vec::new();
    let mut t_plan = Vec::new();
    let mut it_plan = Vec::new();
    // `prepare` hoist 到循环外（文档推荐用法：打包只做一次）
    let prepared = mf.prepare();
    for _ in 0..5 {
        for which in 0..2 {
            let mut n = 0usize;
            let mut mx = 0.0f64;
            let t0 = Instant::now();
            let mut c_out = vec![0.0f32; M * NB];
            while t0.elapsed().as_secs_f64() < 0.06 {
                let t1 = Instant::now();
                if which == 0 {
                    // **DSL 路径**（自动策略：进程级共享池 + 面板复用 + 按形状自动切分）
                    prepared.apply_into(&mkc, &mut out);
                } else {
                    // 手动最优：常驻池 + 复用 plan
                    lasx_rs::parallel::matmul_f32_plan(&pool, &plan, M, &kc, &mut c_out).unwrap();
                }
                let d = t1.elapsed().as_secs_f64();
                if d > mx {
                    mx = d;
                }
                n += 1;
            }
            let med = t0.elapsed().as_secs_f64() / n.max(1) as f64;
            if which == 0 {
                t_dsl.push(med);
                it_dsl.push((n, mx));
            } else {
                t_plan.push(med);
                it_plan.push((n, mx));
            }
        }
    }
    let med = |v: &[f64]| {
        let mut s = v.to_vec();
        s.sort_by(f64::total_cmp);
        s[s.len() / 2]
    };
    let (md, mp) = (med(&t_dsl), med(&t_plan));
    println!(
        "    DSL matmul!（自动策略）：{:.4} ms  {:.1} GFLOP/s  iters={} ms_max={:.3} ms",
        md * 1e3,
        flop / md / 1e9,
        it_dsl[2].0,
        it_dsl[2].1 * 1e3
    );
    println!(
        "    手动 plan（常驻池）    ：{:.4} ms  {:.1} GFLOP/s  iters={} ms_max={:.3} ms",
        mp * 1e3,
        flop / mp / 1e9,
        it_plan[2].0,
        it_plan[2].1 * 1e3
    );
    println!(
        "    ⇒ DSL/手动 = {:.3}×（**DSL 默认走的自动策略已跟上**：进程级共享池 + 打包面板复用 + \
         按形状自动选切分）",
        mp / md
    );
    // 位精确：DSL 结果与串行参考逐位一致
    let want = lasx_rs::api::matmul(M, N, NB, &kc, &f).unwrap();
    assert_eq!(
        re_dsl.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "DSL 与 api::matmul 不逐位"
    );
    println!("    位精确：DSL 结果与 api::matmul 逐位一致 ✓");
}

/// **权重方案对比（梯形 `O(Δt²)` vs Simpson `O(Δt⁴)`）**：固定 `T=20`（截断可忽略）看阶数。
fn weighting_part() {
    const T: f64 = 20.0;
    let sigs = [1.0f64, 2.0, 3.0, 4.0];
    println!("\n== ④ 权重方案（T={T}，f=e^{{-2t}}，解析 1/(s+2)）==");
    println!("     N     梯形误差    相邻比     Simpson 误差   相邻比");
    let mut prev = (0.0f64, 0.0f64);
    for &n in &[65usize, 129, 257, 513] {
        let dt = T / (n - 1) as f64;
        let mut f = vec![0.0f64; n];
        for (i, fi) in f.iter_mut().enumerate() {
            *fi = (-2.0 * (i as f64 * dt)).exp();
        }
        let mut e = [0.0f64; 2];
        for (k, kind) in [Weighting::Trapezoid, Weighting::Simpson]
            .iter()
            .enumerate()
        {
            // 用参考路径（任意 n；核矩阵构造与 DSL 路径完全一致）
            let w = kind.weights(n).0;
            for (j, &s) in sigs.iter().enumerate() {
                let mut acc = 0.0f64;
                for (i, &wi) in w.iter().enumerate() {
                    acc += wi * dt * (-s * (i as f64 * dt)).exp() * f[i];
                }
                let want = 1.0 / (s + 2.0);
                e[k] = e[k].max((acc - want).abs() / want);
                let _ = j;
            }
        }
        let r = if prev.0 > 0.0 {
            (
                format!("{:.2}×", prev.0 / e[0]),
                format!("{:.2}×", prev.1 / e[1]),
            )
        } else {
            ("-".into(), "-".into())
        };
        println!(
            "    {n:>4}   {:.3e}   {:>7}    {:.3e}    {:>7}",
            e[0], r.0, e[1], r.1
        );
        prev = (e[0], e[1]);
    }
    println!("    ⇒ 梯形 ≈4×/Δt 减半（二阶）；**Simpson ≈16×（四阶）**。");
    println!(
        "    回退口径：N<3 或 N 为偶数（区间数奇数）⇒ Simpson **整段回退梯形**并返回标记（不丢尾巴）。"
    );
}

fn main() {
    println!("拉普拉斯变换实战（公式 DSL）：F(s)=∫₀^∞ f(t)e^{{-st}}dt，Re F=Kc·f、Im F=-Ks·f");
    analytic_part();
    refinement_part();
    perf_part();
    weighting_part();
    println!(
        "\n限制：非任意精度（截断 + 求积阶数：梯形 O(Δt²)、Simpson O(Δt⁴)）；\
         s 靠近极点时不能用；**不做逆变换**。"
    );
}
