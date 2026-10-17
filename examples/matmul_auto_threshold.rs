//! **§7.28 实测入口：matmul 四形状 × {1t, auto} × {DSL, 直调}**。
//!
//! **分进程**（`OnceLock` 全局池每进程只能一种线程配置）：
//! ```text
//! LRS_THREADS=1    cargo run --release --example matmul_auto_threshold
//! LRS_THREADS=auto cargo run --release --example matmul_auto_threshold
//! ```
//! 臂定义：
//! - **直调** = `lasx_rs::api::matmul_into(m,k,n,a,b,out)`（后端公开自动入口）
//! - **DSL** = `simd_dsl::Backend::matmul_f32(&LasxBackend, …)`（DSL 的 Backend 契约入口；
//!   其 override 现在也调 `api::matmul_into` ⇒ 两臂**应 ≈1.00**，这就是"抽象层税"的直接度量）
//!
//! 输出：每形状 `中位 ms + GFLOPS + DSL/直调比`；**跨线程配置必须分进程**，不可同进程交替。

use std::time::Instant;

#[cfg(feature = "dsl-backend")]
use simd_dsl::Backend;

fn main() {
    let mode = std::env::var("LRS_THREADS").unwrap_or_else(|_| "auto".into());
    if mode == "1" {
        println!("init_global(1) => {}", lasx_rs::pool::init_global(1));
    } else {
        println!("threads = auto（未调用 init_global）");
    }
    // (m, k, n)
    let shapes = [
        (512usize, 512usize, 512usize),
        (1024, 1024, 1024),
        (192, 768, 3072),
        (256, 64, 4096),
    ];
    for (m, k, n) in shapes {
        let a: Vec<f32> = (0..m * k).map(|i| (i % 97) as f32 * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 89) as f32 * 0.01).collect();
        let mut out = vec![0.0f32; m * n];
        // 预热（建池/首调用开销不计入）
        lasx_rs::api::matmul_into(m, k, n, &a, &b, &mut out).expect("direct");
        let med = |f: &mut dyn FnMut()| -> f64 {
            let mut v = Vec::new();
            for _ in 0..5 {
                let t0 = Instant::now();
                f();
                v.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            v.remove(0);
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        // 臂① 直调
        let t_direct = med(&mut || {
            lasx_rs::api::matmul_into(m, k, n, &a, &b, &mut out).expect("direct");
        });
        // 臂② DSL（Backend 契约入口）
        #[cfg(feature = "dsl-backend")]
        let t_dsl = {
            let be = lasx_rs::dsl_backend::LasxBackend;
            med(&mut || {
                be.matmul_f32(m, k, n, &a, &b, &mut out).expect("dsl");
            })
        };
        #[cfg(not(feature = "dsl-backend"))]
        let t_dsl = f64::NAN;
        let gflops = 2.0 * (m as f64) * (k as f64) * (n as f64) / (t_direct / 1e3) / 1e9;
        println!(
            "shape={m}x{k}x{n} mode={mode} direct={t_direct:.3}ms dsl={t_dsl:.3}ms \
             dsl/direct={:.4} gflops(direct)={gflops:.2}",
            t_dsl / t_direct
        );
    }
    println!(
        "loadavg = {}",
        std::fs::read_to_string("/proc/loadavg")
            .unwrap_or_default()
            .trim()
    );
}
