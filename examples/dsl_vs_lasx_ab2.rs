//! **性能比赛·测试台第二批**：几何 f64 的 `lasx api` vs `ops_dsl::geometry`（DSL 后端）。
//!
//! 用法（两臂分进程，`scripts/dsl_vs_lasx_ab.sh` 或手动交替）：
//! ```text
//! cargo run --release --features dsl-backend --example dsl_vs_lasx_ab2 -- --op=norm3 --arm=lasx --iters=5
//! cargo run --release --features dsl-backend --example dsl_vs_lasx_ab2 -- --op=norm3 --arm=dsl  --iters=5
//! ```
//! 输出：`RESULT op=.. arm=.. median_ms=.. range_ms=.. bitwise=.. n=.. loadavg=..`
//!
//! # 覆盖与对齐（如实）
//! - **可对齐**：`norm3`（LASX `api::norm3_batch_into` ↔ DSL `ops_dsl::geometry::norm3_batch_f64`）、
//!   `unitize3`（`api::unitize3_batch_into` ↔ `unitize3_batch_f64`）。
//! - **`unitize3` 零向量语义差异**：组合路径 `0/0 = NaN`、融合核（LASX plan 路径）为 0；本台架用
//!   非零输入比性能与等价，零向量语义差异由 DSL 侧测试覆盖（文件头注明，不混入性能结论）。
//! - **未实现/跳过（如实，不用近似臂冒充）**：`vec3_add_scaled`、`cross3`（DSL 侧带额外暂存参数，
//!   本轮未读全签名）；`spectral detect`/`stencil2d`（LASX 无公开对应入口）；几何 f32（LASX 仅有 f64 版本）。
//!
//! # 公平性
//! 计时区外构造输入；两臂输出在计时区外做一次 `to_bits` 全等校验；两臂均为"一进一出/原地写"
//! 的原生入口，无 prepare/打包；`threads` 仅打印（几何 f64 两臂均无按线程切分入口）。

use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Op {
    Norm3,
    Unitize3,
}

fn parse_args() -> (Op, String, usize) {
    let mut op = Op::Norm3;
    let mut arm = "lasx".to_string();
    let mut iters = 5usize;
    for a in std::env::args().skip(1) {
        if let Some(v) = a.strip_prefix("--op=") {
            op = match v {
                "norm3" => Op::Norm3,
                "unitize3" => Op::Unitize3,
                other => panic!(
                    "未知 --op={other}（只支持 norm3/unitize3；vec3_add_scaled/cross3 未实现，见文件头）"
                ),
            };
        } else if let Some(v) = a.strip_prefix("--arm=") {
            arm = v.to_string();
        } else if let Some(v) = a.strip_prefix("--iters=") {
            iters = v.parse().expect("--iters 需为整数");
        } else if let Some(v) = a.strip_prefix("--threads=") {
            let _ = v; // 两臂均无按线程切分入口；仅保持 CLI 兼容
        }
    }
    (op, arm, iters)
}

fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .map(|s| s.split_whitespace().take(3).collect::<Vec<_>>().join(" "))
        .unwrap_or_else(|_| "?".to_string())
}

type Bufs = (
    Vec<f64>, // out / ox
    Vec<f64>, // oy
    Vec<f64>, // oz
    Vec<f64>, // norm
    Vec<f64>, // t0
    Vec<f64>, // t1
    Vec<f64>, // t2
);

fn new_bufs(n: usize) -> Bufs {
    (
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
    )
}

#[cfg(feature = "dsl-backend")]
fn run_dsl(op: Op, x: &[f64], y: &[f64], z: &[f64], b: &mut Bufs) -> Result<(), String> {
    let be = lasx_rs::dsl_backend::LasxBackend;
    match op {
        Op::Norm3 => {
            ops_dsl::geometry::norm3_batch_f64(&be, x, y, z, &mut b.3, &mut b.4, &mut b.5, &mut b.6)
                .map_err(|e| format!("{e:?}"))
        }
        Op::Unitize3 => ops_dsl::geometry::unitize3_batch_f64(
            &be, x, y, z, &mut b.0, &mut b.1, &mut b.2, &mut b.3, &mut b.4, &mut b.5, &mut b.6,
        )
        .map_err(|e| format!("{e:?}")),
    }
}

fn run_lasx(op: Op, x: &[f64], y: &[f64], z: &[f64], b: &mut Bufs) -> Result<(), String> {
    match op {
        Op::Norm3 => {
            lasx_rs::api::norm3_batch_into(x, y, z, &mut b.3).map_err(|e| format!("{e:?}"))
        }
        Op::Unitize3 => lasx_rs::api::unitize3_batch_into(x, y, z, &mut b.0, &mut b.1, &mut b.2)
            .map_err(|e| format!("{e:?}")),
    }
}

fn outputs(op: Op, b: &Bufs) -> Vec<f64> {
    match op {
        Op::Norm3 => b.3.clone(),
        Op::Unitize3 => [&b.0[..], &b.1[..], &b.2[..]].concat(),
    }
}

fn one_arm(op: Op, arm: &str, x: &[f64], y: &[f64], z: &[f64], b: &mut Bufs) -> Result<(), String> {
    match arm {
        "lasx" => run_lasx(op, x, y, z, b),
        #[cfg(feature = "dsl-backend")]
        "dsl" => run_dsl(op, x, y, z, b),
        #[cfg(not(feature = "dsl-backend"))]
        "dsl" => Err("dsl 臂需要 --features dsl-backend".to_string()),
        other => Err(format!("未知 --arm={other}（只支持 lasx|dsl）")),
    }
}

fn main() {
    let (op, arm, iters) = parse_args();
    let n = 1_000_000usize;
    let x: Vec<f64> = (0..n)
        .map(|i| ((i % 97) as f64 - 48.0) * 0.5 + 0.25)
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| ((i % 89) as f64 - 44.0) * 0.5 + 0.5)
        .collect();
    let z: Vec<f64> = (0..n)
        .map(|i| ((i % 83) as f64 - 41.0) * 0.5 + 0.75)
        .collect();

    // 计时区外：参考值（lasx）与两臂等价性校验。
    let mut ref_bufs = new_bufs(n);
    run_lasx(op, &x, &y, &z, &mut ref_bufs).expect("lasx 参考");
    let want = outputs(op, &ref_bufs);
    let mut check = new_bufs(n);
    one_arm(op, &arm, &x, &y, &z, &mut check).expect("arm 预热");
    let got = outputs(op, &check);
    let bitwise = got
        .iter()
        .map(|v| v.to_bits())
        .eq(want.iter().map(|v| v.to_bits()));

    let mut work = new_bufs(n);
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        one_arm(op, &arm, &x, &y, &z, &mut work).expect("arm 计时");
        times.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(f64::total_cmp);
    let median = times[times.len() / 2];
    let range = times[times.len() - 1] - times[0];
    let op_name = match op {
        Op::Norm3 => "norm3",
        Op::Unitize3 => "unitize3",
    };
    println!(
        "RESULT op={op_name} arm={arm} median_ms={median:.4} range_ms={range:.4} bitwise={bitwise} n={n} loadavg={}",
        loadavg()
    );
}
