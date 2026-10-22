//! **复杂算法对照装置（LASX 后端侧，测量用）**：跑 `attention_block` 库里的**同一份 DSL 算法**
//! （512×512 单头自注意力块：5 GEMM + softmax + 残差 + LayerNorm + amax），后端为
//! [`LasxBackend`]，输出机器可读计时行，供 `simd_dsl/scripts/bench_algo_vs_openblas.sh`
//! 做三方对照（可移植 DSL / LASX DSL / OpenBLAS）。
//!
//! **这不是算法示例**：算法与示例在 DSL 仓（`attention_block` crate /
//! `examples/algo_attention.rs`）；本文件是 lasx 仓的**同算法测量入口**，只做后端替换与计时，
//! 保证两侧跑的是**同一份 DSL 代码**（不复制算法）。
//!
//! 用法：
//! ```bash
//! cargo run --release --features dsl-backend --example algo_attention_lasx -- --bench --iters 3
//! cargo build --release --features dsl-backend --example algo_attention_lasx
//! ```

use std::time::Instant;

use attention_block::{DIM, EPS, Inputs, ROWS, digest, loadavg, run};
use lasx_rs::dsl_backend::LasxBackend;

fn bench(be: &LasxBackend, b: &Inputs, iters: usize) -> (f64, f64, Vec<f32>, f32) {
    let mut samples = Vec::with_capacity(iters);
    let mut last = (Vec::new(), 0.0);
    for i in 0..=iters {
        let t = Instant::now();
        last = run(be, b).expect("算法执行失败");
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if i > 0 {
            samples.push(ms);
        }
    }
    samples.sort_by(f64::total_cmp);
    let med = samples[samples.len() / 2];
    let min = samples[0];
    (min, med, last.0, last.1)
}

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut iters = 3usize;
    let mut dump_z: Option<String> = None;
    let mut compare: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--bench" => {}
            "--iters" => {
                iters = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(3);
                i += 1;
            }
            "--dump-z" => {
                dump_z = args.get(i + 1).cloned();
                i += 1;
            }
            "--compare" => {
                compare = args.get(i + 1).cloned();
                i += 1;
            }
            other => {
                eprintln!("未知参数 {other}");
                return std::process::ExitCode::from(2);
            }
        }
        i += 1;
    }
    let be = LasxBackend;
    let b = Inputs::new();
    if let Some(path) = compare {
        let bytes = std::fs::read(&path).expect("读对方 Z 失败");
        let other: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let (z, _amax) = run(&be, &b).expect("算法失败");
        let mut max_abs = 0.0f64;
        let mut max_rel = 0.0f64;
        let mut ref_scale = 1.0f64;
        for &v in &other {
            ref_scale = ref_scale.max(f64::from(v.abs()));
        }
        for (a, o) in z.iter().zip(other.iter()) {
            let d = (f64::from(*a) - f64::from(*o)).abs();
            max_abs = max_abs.max(d);
            max_rel = max_rel.max(d / ref_scale.max(1.0));
        }
        let ok = max_rel <= 1.0e-3;
        println!(
            "CONSISTENCY n={} max_abs={max_abs:.3e} max_rel={max_rel:.3e} ok={}",
            z.len().min(other.len()),
            if ok { "yes" } else { "no" }
        );
        return std::process::ExitCode::from(u8::from(!ok));
    }
    let (min, med, z, amax) = bench(&be, &b, iters);
    if let Some(path) = dump_z {
        let mut bytes = Vec::with_capacity(z.len() * 4);
        for v in &z {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(path, bytes).expect("写 Z 失败");
    }
    let dg = digest(&z, amax);
    println!(
        "ALGO impl=lasx threads={} rows={ROWS} dim={DIM} eps={EPS:e} iters={iters} ms_min={min:.6} ms_med={med:.6} digest=0x{dg:016x} amax={amax:.6e} loadavg={}",
        std::thread::available_parallelism().map_or(1, |n| n.get()),
        loadavg()
    );
    std::process::ExitCode::SUCCESS
}
