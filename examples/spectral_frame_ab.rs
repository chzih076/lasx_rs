//! **谱检测整帧在 LASX 档上的 A/B（R19-D）**：`DftMode::Direct`（两次 LASX 瘦长 matmul）
//! vs `DftMode::Auto`（= 可移植**免转置按列 FFT**）。**这是"完整性能"的帧级口径**：
//! 建计划、帧体（预建计划）、一次性 `detect` 三个数都在**同一进程内交替**测。
//!
//! **口径（§5）**：同进程交替、**同一目的缓冲**、5 对丢首对、中位 + 极差、正确性在计时之外。
//! 形状 = 真实工作点 `n=4096, p=513, b=256`；参考档 = `IdiomaticBackend`（可移植）。
//!
//! **如实边界**：LASX 档只覆盖 `matmul_f32`/`dot_f32`/`build_kernel_f32`/`matmul_f32_packed`；
//! 帧里其余步骤（归约/逐元素/超越函数）走 `Backend` 的**可移植默认实现** ⇒ 两臂在这部分开销相同；
//! FFT 臂用的也是**可移植** `fft_strided_f32`（LASX 尚无向量化 FFT）。
use lasx_rs::dsl_backend::{LasxBackend, LasxBackendHybrid};
use ops_dsl::spectral::{
    DftMode, SpectralCfg, build_plan, detect, detect_stages, detect_with_plan,
};
use simd_dsl::IdiomaticBackend;
use std::time::Instant;

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}
fn range(v: &[f64]) -> f64 {
    let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    hi - lo
}
/// §5：交替采样，丢弃首对，返回 (A 中位, A 极差, B 中位, B 极差)。

fn masked_rel(got: &[f32], want: &[f32]) -> f64 {
    let peak = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0) as f64;
    got.iter()
        .zip(want)
        .filter(|(_, w)| (**w).abs() as f64 > 1.0e-3 * peak)
        .fold(0.0f64, |m, (g, w)| {
            m.max((*g - *w).abs() as f64 / w.abs() as f64)
        })
}

// `alternate` 的两个闭包需要同时可变借 `ta`/`tb` 之外的资源 ⇒ 用 macro 化简调用处的借用冲突
macro_rules! ab {
    ($pairs:expr, $a:block, $b:block) => {{
        let (mut ta, mut tb) = (Vec::new(), Vec::new());
        for pair in 0..$pairs {
            let t0 = Instant::now();
            $a
            let da = t0.elapsed().as_secs_f64() * 1e3;
            let t1 = Instant::now();
            $b
            let db = t1.elapsed().as_secs_f64() * 1e3;
            if pair > 0 {
                ta.push(da);
                tb.push(db);
            }
        }
        (median(&mut ta), range(&ta), median(&mut tb), range(&tb))
    }};
}

fn main() {
    let lasx = LasxBackend;
    let hybrid = LasxBackendHybrid; // "接 R19-E 之前"的形态：逐元素/归约走可移植默认
    let port = IdiomaticBackend;
    let (n, p, b) = (4096usize, 513usize, 256usize);
    let base = SpectralCfg {
        n,
        p,
        b,
        dt: 1.0 / n as f32,
        eps: 1.0e-12,
        k_soft: 6.0,
        dft: DftMode::Auto,
    };
    let mut c_dir = base;
    c_dir.dft = DftMode::Direct;
    let mut c_fft = base;
    c_fft.dft = DftMode::Fft;
    assert!(c_fft.uses_fft() && !c_dir.uses_fft() && base.uses_fft());
    println!(
        "loadavg = {}",
        std::fs::read_to_string("/proc/loadavg")
            .unwrap_or_default()
            .trim()
    );

    let x: Vec<f32> = (0..n * b)
        .map(|i| ((i * 37 % 101) as f32) * 0.05 - 2.5)
        .collect();
    let threshold = vec![-40.0f32; b];
    let fw = base.normalized_fw();
    let mut out_a = vec![0.0f32; b];
    let mut out_b = vec![0.0f32; b];

    // ---- 正确性（计时之外）----
    let st_dir = detect_stages(&lasx, &c_dir, &x, &threshold, &fw).unwrap();
    let st_fft = detect_stages(&lasx, &c_fft, &x, &threshold, &fw).unwrap();
    println!(
        "正确性（LASX 档，FFT vs Direct）：out masked_rel = {:.3e} | psd masked_rel = {:.3e}",
        masked_rel(&st_fft.out, &st_dir.out),
        masked_rel(&st_fft.psd, &st_dir.psd)
    );

    // ---- ① 建计划（Direct 建 kc/ks；FFT 不建）----
    let (b_dir, b_dir_r, b_fft, b_fft_r) = ab!(
        6,
        {
            let _ = build_plan(&lasx, &c_dir).unwrap();
        },
        {
            let _ = build_plan(&lasx, &c_fft).unwrap();
        }
    );
    let plan_dir = build_plan(&lasx, &c_dir).unwrap();
    let plan_fft = build_plan(&lasx, &c_fft).unwrap();
    println!(
        "① build_plan（LASX）  Direct {b_dir:9.4} ms（极差 {b_dir_r:7.4}） | FFT {b_fft:9.4} ms（极差 {b_fft_r:7.4}） ⇒ **{:.2}×**",
        b_dir / b_fft
    );

    // ---- ② 帧体（预建计划）----
    let (f_dir, f_dir_r, f_fft, f_fft_r) = ab!(
        6,
        {
            detect_with_plan(&lasx, &plan_dir, &x, &threshold, &fw, &mut out_a).unwrap();
        },
        {
            detect_with_plan(&lasx, &plan_fft, &x, &threshold, &fw, &mut out_b).unwrap();
        }
    );
    println!(
        "② 帧体（LASX，预建） Direct {f_dir:9.4} ms（极差 {f_dir_r:7.4}） | FFT {f_fft:9.4} ms（极差 {f_fft_r:7.4}） ⇒ **{:.3}×**",
        f_dir / f_fft
    );

    // ---- ③ 一次性 detect（含建计划）----
    let (o_dir, o_dir_r, o_fft, o_fft_r) = ab!(
        6,
        {
            detect(&lasx, &c_dir, &x, &threshold, &fw, &mut out_a).unwrap();
        },
        {
            detect(&lasx, &c_fft, &x, &threshold, &fw, &mut out_b).unwrap();
        }
    );
    println!(
        "③ 一次性 detect（LASX） Direct {o_dir:9.4} ms（极差 {o_dir_r:7.4}） | FFT {o_fft:9.4} ms（极差 {o_fft_r:7.4}） ⇒ **{:.3}×**",
        o_dir / o_fft
    );

    // ---- ④ 两档并列（同一 `Auto` 配置，看"完整性能"的档位差）----
    let (p_fft, p_fft_r, l_fft, l_fft_r) = ab!(
        6,
        {
            detect(&port, &base, &x, &threshold, &fw, &mut out_a).unwrap();
        },
        {
            detect(&lasx, &base, &x, &threshold, &fw, &mut out_b).unwrap();
        }
    );
    println!(
        "④ 一次性 detect（Auto=FFT） 可移植 {p_fft:9.4} ms（极差 {p_fft_r:7.4}） | **LASX {l_fft:9.4} ms（极差 {l_fft_r:7.4}）** ⇒ 可移植/LASX = **{:.3}×**",
        p_fft / l_fft
    );
    // 同口径再测 Direct 的两档差（看"手写 matmul 在帧里的占比"）
    let (p_dir, p_dir_r, l_dir, l_dir_r) = ab!(
        6,
        {
            detect(&port, &c_dir, &x, &threshold, &fw, &mut out_a).unwrap();
        },
        {
            detect(&lasx, &c_dir, &x, &threshold, &fw, &mut out_b).unwrap();
        }
    );
    println!(
        "⑤ 一次性 detect（Direct）    可移植 {p_dir:9.4} ms（极差 {p_dir_r:7.4}） | **LASX {l_dir:9.4} ms（极差 {l_dir_r:7.4}）** ⇒ 可移植/LASX = **{:.3}×**",
        p_dir / l_dir
    );

    // ---- ⑥⑦ **同进程 A/B：接族后 vs 接族前**（`LasxBackend` vs `LasxBackendHybrid`）----
    // hybrid = matmul/建核/打包照旧 LASX，但逐元素/归约走**可移植默认** ⇒ "接 R19-E 之前"的形态。
    let (h_frame, h_frame_r, n_frame, n_frame_r) = ab!(
        6,
        {
            detect_with_plan(&hybrid, &plan_fft, &x, &threshold, &fw, &mut out_a).unwrap();
        },
        {
            detect_with_plan(&lasx, &plan_fft, &x, &threshold, &fw, &mut out_b).unwrap();
        }
    );
    println!(
        "⑥ 帧体（FFT，预建）  接族前 {h_frame:9.4} ms（极差 {h_frame_r:7.4}） | **接族后 {n_frame:9.4} ms（极差 {n_frame_r:7.4}）** ⇒ **{:.3}×**",
        h_frame / n_frame
    );
    let (h_one, h_one_r, n_one, n_one_r) = ab!(
        6,
        {
            detect(&hybrid, &c_fft, &x, &threshold, &fw, &mut out_a).unwrap();
        },
        {
            detect(&lasx, &c_fft, &x, &threshold, &fw, &mut out_b).unwrap();
        }
    );
    println!(
        "⑦ 一次性 detect（FFT）接族前 {h_one:9.4} ms（极差 {h_one_r:7.4}） | **接族后 {n_one:9.4} ms（极差 {n_one_r:7.4}）** ⇒ **{:.3}×**",
        h_one / n_one
    );
    // 正确性：接族前后的输出应当**非常接近**（逐元素/归约都声明与标量参照逐位/同类）
    let mut oa = vec![0.0f32; b];
    let mut ob = vec![0.0f32; b];
    detect(&hybrid, &c_fft, &x, &threshold, &fw, &mut oa).unwrap();
    detect(&lasx, &c_fft, &x, &threshold, &fw, &mut ob).unwrap();
    println!(
        "   ⑥⑦ 正确性（接族前 vs 后）：out masked_rel = {:.3e}",
        masked_rel(&ob, &oa)
    );
}
