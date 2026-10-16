//! **后端效率对照（③b-3a）**：`LasxBackend` 内部"递推档 vs 全回退档"，以及
//! `LasxBackend`（手写）vs `simd_dsl::mock::ScalarBackend`（笨可移植）。
//!
//! **口径严格照 `simd_dsl/docs/bench.md` §5**：
//! - **目的缓冲口径 E（强制）**：一条对比的两条臂**写同一块目的缓冲**（本文件里就是同一对 `kc/ks`）；
//! - **同进程交替、≥5 对、丢第一对**；每对内部 `iters` 次取均值，输出**中位 + 极差 + `ms_max` + `iters`**；
//! - 打印 **loadavg**：**≥1.0 时绝对毫秒不可比**，只看同进程比值；
//! - **每行标等价类**（调 `Backend::precision_claim`，未声明就打印"未声明"）；
//! - **正确性在计时之外断言**（全回退档与 `ScalarBackend` 在整数域应 `==`）。
//!
//! 复现：`cargo run --release --features dsl-backend --example dsl_efficiency`
#![cfg(feature = "dsl-backend")]

use std::cell::RefCell;
use std::time::Instant;

use lasx_rs::dsl_backend::LasxBackend;
use simd_dsl::IdiomaticBackend;
use simd_dsl::backend::Backend;
use simd_dsl::equivalence::{EnginePath, ScalarKind};
use simd_dsl::kernel::KernelSpec;
use simd_dsl::mock::ScalarBackend;

const ROWS: usize = 64;
const N: usize = 1024;

fn loadavg() -> f64 {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next().and_then(|v| v.parse().ok()))
        .unwrap_or(f64::NAN)
}

/// 一次测量：同一块目的缓冲、`iters` 次取均值；返回 `(mean_ms, ms_max, iters)`。
fn measure<F: FnMut()>(mut f: F, budget_s: f64) -> (f64, f64, u32) {
    let mut iters = 0u32;
    let mut sum = 0.0f64;
    let mut ms_max = 0.0f64;
    let t0 = Instant::now();
    while t0.elapsed().as_secs_f64() < budget_s {
        let t1 = Instant::now();
        f();
        let d = t1.elapsed().as_secs_f64();
        sum += d;
        if d > ms_max {
            ms_max = d;
        }
        iters += 1;
    }
    let mean = if iters == 0 {
        f64::NAN
    } else {
        sum / iters as f64
    };
    (mean * 1e3, ms_max * 1e3, iters)
}

fn median_range(v: &[f64]) -> (f64, f64) {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    (s[s.len() / 2], s[s.len() - 1] - s[0])
}

/// 等价类标注（未声明 ⇒ 打印"未声明"）。
fn class_of(b: &dyn Backend, path: EnginePath) -> String {
    match b.precision_claim(path, ScalarKind::F32) {
        Some(c) => format!("{c:?}"),
        None => "未声明".to_string(),
    }
}

fn main() {
    let dt = 1.0f32 / 256.0;
    let sigma = vec![0.5f32; ROWS];
    // **全在安全区**（ω·Δt 很小）：方向② 的"递推档"
    let omega_safe: Vec<f32> = (0..ROWS).map(|j| (j % 8) as f32 * 0.003).collect();

    let spec_rec = KernelSpec::new(ROWS, N, dt, &sigma, &omega_safe, true);
    let spec_fb = KernelSpec::new(ROWS, N, dt, &sigma, &omega_safe, false);

    let lasx = LasxBackend;
    let scalar = ScalarBackend;

    // ---- 正确性（**计时之外**）：全回退档 vs ScalarBackend（整数域 ⇒ 应 `==`）----
    let (mut kc, mut ks) = (vec![0.0f32; ROWS * N], vec![0.0f32; ROWS * N]);
    let (mut kc2, mut ks2) = (vec![0.0f32; ROWS * N], vec![0.0f32; ROWS * N]);
    lasx.build_kernel_f32(&spec_fb, &mut kc, &mut ks).unwrap();
    scalar
        .build_kernel_f32(&spec_fb, &mut kc2, &mut ks2)
        .unwrap();
    let same = kc == kc2 && ks == ks2;
    println!(
        "[正确性·计时外] LasxBackend(全回退) == ScalarBackend：{}",
        if same {
            "逐值相等 ✓"
        } else {
            "不相等 ✗"
        }
    );

    println!("\nloadavg = {:.2}", loadavg());
    println!(
        "形状：rows={ROWS}, n={N}, dt={dt}；口径：同一对 kc/ks 缓冲、同进程交替、6 对丢首对、每对内多次取均值\n"
    );

    // ================= 方向②：同一后端内 递推档 vs 全回退档 =================
    let dest = RefCell::new((vec![0.0f32; ROWS * N], vec![0.0f32; ROWS * N]));
    let mut a_ms = Vec::new();
    let mut b_ms = Vec::new();
    let (mut a_max, mut b_max) = (0.0f64, 0.0f64);
    let (mut a_it, mut b_it) = (0u32, 0u32);
    for pair in 0..6 {
        // 交替起跑次序：偶数对先测递推、奇数对先测全回退（"一对" = 两臂各测一次）
        let first_is_rec = pair % 2 == 0;
        let order = if first_is_rec { [0usize, 1] } else { [1, 0] };
        for which in order {
            let mut d = dest.borrow_mut();
            let (kc, ks) = &mut *d;
            let spec = if which == 0 { &spec_rec } else { &spec_fb };
            let (mean, mx, it) = measure(
                || {
                    lasx.build_kernel_f32(spec, kc, ks).unwrap();
                },
                0.05,
            );
            if which == 0 {
                a_ms.push(mean);
                a_max = a_max.max(mx);
                a_it = a_it.max(it);
            } else {
                b_ms.push(mean);
                b_max = b_max.max(mx);
                b_it = b_it.max(it);
            }
        }
    }
    // 丢掉第一对（各 2 个样本）后再取中位/极差
    let (a_med, a_rng) = median_range(&a_ms[2..]);
    let (b_med, b_rng) = median_range(&b_ms[2..]);
    println!("== 方向②：同一 LasxBackend 内，递推档 vs 全回退档 ==");
    println!(
        "  递推档(opt_in=true)  {a_med:.4} ms  极差 {a_rng:.4}  ms_max {a_max:.3}  iters {a_it}  等价类 {}",
        class_of(&lasx, EnginePath::IsometricRecurrence)
    );
    println!(
        "  全回退档(opt_in=false) {b_med:.4} ms  极差 {b_rng:.4}  ms_max {b_max:.3}  iters {b_it}  等价类 {}",
        class_of(&lasx, EnginePath::Default)
    );
    println!(
        "  ⇒ 递推/全回退 = **{:.3}×**（<1 表示递推更快）",
        a_med / b_med
    );

    // ================= 方向①：LasxBackend vs ScalarBackend =================
    let mut l_ms = Vec::new();
    let mut s_ms = Vec::new();
    let (mut l_max, mut s_max) = (0.0f64, 0.0f64);
    let (mut l_it, mut s_it) = (0u32, 0u32);
    for pair in 0..6 {
        for step in 0..2 {
            let use_lasx = (pair + step) % 2 == 0;
            let mut d = dest.borrow_mut();
            let (kc, ks) = &mut *d;
            let (mean, mx, it) = if use_lasx {
                measure(
                    || {
                        lasx.build_kernel_f32(&spec_rec, kc, ks).unwrap();
                    },
                    0.05,
                )
            } else {
                measure(
                    || {
                        scalar.build_kernel_f32(&spec_rec, kc, ks).unwrap();
                    },
                    0.05,
                )
            };
            if use_lasx {
                l_ms.push(mean);
                l_max = l_max.max(mx);
                l_it = l_it.max(it);
            } else {
                s_ms.push(mean);
                s_max = s_max.max(mx);
                s_it = s_it.max(it);
            }
        }
    }
    let (l_med, l_rng) = median_range(&l_ms[2..]);
    let (s_med, s_rng) = median_range(&s_ms[2..]);
    println!(
        "\n== 方向①：LasxBackend（手写）vs ScalarBackend（笨可移植），同一 `build_kernel_f32` 请求 =="
    );
    println!(
        "  LasxBackend    {l_med:.4} ms  极差 {l_rng:.4}  ms_max {l_max:.3}  iters {l_it}  等价类 {}",
        class_of(&lasx, EnginePath::IsometricRecurrence)
    );
    println!(
        "  ScalarBackend  {s_med:.4} ms  极差 {s_rng:.4}  ms_max {s_max:.3}  iters {s_it}  等价类 {}",
        class_of(&scalar, EnginePath::Default)
    );
    println!(
        "  ⇒ Scalar/Lasx = **{:.3}×**（>1 表示手写多买到）",
        s_med / l_med
    );

    // ============ ③b-3b 第三臂：惯用 Rust（自动向量化）—— 三臂同口径 ============
    let idio = IdiomaticBackend;
    let mut i_ms = Vec::new();
    let (mut i_max, mut i_it) = (0.0f64, 0u32);
    let mut l2_ms = Vec::new();
    let (mut l2_max, mut l2_it) = (0.0f64, 0u32);
    for pair in 0..6 {
        let first_is_idio = pair % 2 == 0;
        let order = if first_is_idio { [0usize, 1] } else { [1, 0] };
        for which in order {
            let mut d = dest.borrow_mut();
            let (kc, ks) = &mut *d;
            let (mean, mx, it) = if which == 0 {
                measure(
                    || {
                        idio.build_kernel_f32(&spec_rec, kc, ks).unwrap();
                    },
                    0.05,
                )
            } else {
                measure(
                    || {
                        lasx.build_kernel_f32(&spec_rec, kc, ks).unwrap();
                    },
                    0.05,
                )
            };
            if which == 0 {
                i_ms.push(mean);
                i_max = i_max.max(mx);
                i_it = i_it.max(it);
            } else {
                l2_ms.push(mean);
                l2_max = l2_max.max(mx);
                l2_it = l2_it.max(it);
            }
        }
    }
    let (i_med, i_rng) = median_range(&i_ms[2..]);
    let (l2_med, l2_rng) = median_range(&l2_ms[2..]);
    println!("\n== ③b-3b 三臂（同一 `build_kernel_f32` 请求、同一对 kc/ks 缓冲）==");
    println!(
        "  LasxBackend（手写 LASX 通路的递推核） {l2_med:.4} ms  极差 {l2_rng:.4}  ms_max {l2_max:.3}  iters {l2_it}  等价类 {}",
        class_of(&lasx, EnginePath::IsometricRecurrence)
    );
    println!(
        "  惯用 Rust（`simd_dsl::IdiomaticBackend`，库内导出）  {i_med:.4} ms  极差 {i_rng:.4}  ms_max {i_max:.3}  iters {i_it}  等价类 {}",
        class_of(&idio, EnginePath::IsometricRecurrence)
    );
    println!(
        "  ScalarBackend（笨标量）            {s_med:.4} ms  极差 {s_rng:.4}  ms_max {s_max:.3}  iters {s_it}  等价类 {}",
        class_of(&scalar, EnginePath::Default)
    );
    println!(
        "  ⇒ 惯用/Lasx = **{:.3}×**（>1 ⇒ 手写领先；≈1 ⇒ 持平）｜惯用/Scalar = **{:.3}×**",
        i_med / l2_med,
        i_med / s_med
    );
    println!(
        "\n③b-3c：本档现在用的是**库内导出**的 `simd_dsl::IdiomaticBackend`（`521e837` 起 `pub use idiomatic::*`），\n\
         **本地副本已删除**。⚠️ **流程规则**：`Cargo.toml` 里钉死的 `rev` **必须在引擎仓库移动时显式 bump**\n\
         （`git ls-remote <panel>` 取实值 + `cargo update -p simd_dsl`），否则本仓看到的是旧快照——\n\
         ③b-3b 就因此误判『未导出』。"
    );

    // ======================== ③b-5a′：dot K=4096 三臂（算法由 DSL 宏生成） ========================
    // **宏约束（本轮试出来的三条，也转给 simd_dsl 侧记档）**：
    //  1. `dot!(arm, x[k] * w[k])` 里的 `k` **必须是作用域内的值/const 泛型**（不能只写字面量）；
    //  2. 后端以**第一个 `ident`** 出现 ⇒ 需要作用域里有 `lasx`/`idio`/`scalar` 三个绑定；
    //  3. dot 无输出缓冲 ⇒ §5 的"同缓冲"口径对它**自动成立**（matmul 场景才需要共享 `RefCell`，
    //     且输出必须是 `let mut o = shared.borrow_mut();` 那个 `DerefMut` 绑定）。
    println!("\n============ ③b-5a′：dot K=4096 三臂（宏驱动） ============");
    println!("loadavg = {:.2}", loadavg());
    {
        let k = 4096usize;
        let x: Vec<f32> = (0..k).map(|i| (i % 7) as f32 - 3.0).collect();
        let w: Vec<f32> = (0..k).map(|i| (i % 5) as f32 - 2.0).collect();
        // **计时外**：三臂值与逐位判定（整数域 ⇒ 各种累加次序都可能精确）
        let v = [
            simd_dsl::dot!(lasx, x[k] * w[k]).unwrap(),
            simd_dsl::dot!(idio, x[k] * w[k]).unwrap(),
            simd_dsl::dot!(scalar, x[k] * w[k]).unwrap(),
        ];
        let bitwise = v[0].to_bits() == v[1].to_bits() && v[0].to_bits() == v[2].to_bits();
        println!(
            "  [计时外] dot 三臂值：lasx={} idio={} scalar={} ⇒ {}",
            v[0],
            v[1],
            v[2],
            if bitwise {
                "三臂逐位一致（整数域）⇒ **逐位类**"
            } else {
                "**非逐位** ⇒ 本场景按容差类报告（不假设逐位）"
            }
        );
        let names = ["lasx", "idio", "scalar"];
        let cls = [
            class_of(&lasx, EnginePath::Default),
            class_of(&idio, EnginePath::Default),
            class_of(&scalar, EnginePath::Default),
        ];
        let mut samples: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let (mut mx, mut it) = ([0.0f64; 3], [0u32; 3]);
        for pair in 0..5 {
            // 交替起跑次序（配合"互换"：正序 / 逆序各半）
            let order = if pair % 2 == 0 {
                [0usize, 1, 2]
            } else {
                [2, 1, 0]
            };
            for arm in order {
                let (mean, m2, i2) = measure(
                    || {
                        let got = match arm {
                            0 => simd_dsl::dot!(lasx, x[k] * w[k]).unwrap(),
                            1 => simd_dsl::dot!(idio, x[k] * w[k]).unwrap(),
                            _ => simd_dsl::dot!(scalar, x[k] * w[k]).unwrap(),
                        };
                        std::hint::black_box(got);
                    },
                    0.05,
                );
                samples[arm].push(mean);
                mx[arm] = mx[arm].max(m2);
                it[arm] = it[arm].max(i2);
            }
        }
        println!("  == dot K=4096（5 对丢首对；正/逆序交替）==");
        let mut med = [0.0f64; 3];
        for arm in 0..3 {
            let (md, rg) = median_range(&samples[arm][1..]);
            med[arm] = md;
            println!(
                "  {:6} {:.5} ms  极差 {:.5}  ms_max {:.4}  iters {}  等价类 {}",
                names[arm], md, rg, mx[arm], it[arm], cls[arm]
            );
        }
        println!(
            "  ⇒ lasx/idio = **{:.3}×**｜lasx/scalar = **{:.3}×**｜idio/scalar = **{:.3}×**",
            med[0] / med[1],
            med[0] / med[2],
            med[1] / med[2]
        );
        println!(
            "  注：判据先写死 —— `lasx/idio ≈ 0.99–1.01×` ⇒ 如实写『手写没多买到、自动向量化已吃满』。\n\
           未做：matmul 64³（③b-5a″）、`192×768×3072`（③b-5b）、③b-4（打包接 `Prepared`、`CallOpts`、`Capability`）。"
        );
    }
}
