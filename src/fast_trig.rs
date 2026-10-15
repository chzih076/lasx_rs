//! **超越函数 recon（原型，opt-in，不改默认行为）**：拉普拉斯核构造的三条路线。
//!
//! 现状（基线）走 `libm` 的 `exp/cos/sin`（`laplace::kernels64`）。本模块提供两条**可选**
//! 快路径，用于测量"值不值得扩 DSL 语法"：
//!
//! | 路线 | 做法 | 契约 |
//! |---|---|---|
//! | (a) 基线 | `libm` 逐元素 `exp/cos/sin` | **位精确基准**（现有硬承诺） |
//! | (b) 表 + 多项式 | `const` 编译期表（**零运行时初始化**）+ 2 项 Taylor 修正 | **与 libm 不逐位一致**（例外清单；误差界见下） |
//! | (c) 等距递推 | `t_i = i·Δt` 上 `e^{−σΔt}` 逐次乘、`cos/sin` 旋转递推 | **与 libm 不逐位一致**；**误差随 `N` 累积**（实测界见下） |
//!
//! **共享表**：(b) 的表是 `static`，**每个（类型, 精度）一份**、`const fn` **编译期生成**、
//! **零运行时初始化**、只读共享 ⇒ **多线程只读安全**（无 `static mut`、无 `UnsafeCell`）。
//!
//! **为什么需要显式"范围声明"**：DSL 只有**类型级形状**，**没有取值范围信息**；表驱动路线的
//! 精度/范围取舍（表大小、归约区间）**必须由显式声明驱动**，否则"编译期分析查表精度"是空话
//! —— 见 `docs/dev.md` 的"DSL 能/不能知道什么"小节与 `docs/ops.md` §9.10。

/// 表步长：`[0, π/2]` 上 128 段（`N_TAB = 129` 个点）。
pub const TAB_STEP: f64 = std::f64::consts::FRAC_PI_2 / 128.0;
/// 表长。
pub const N_TAB: usize = 129;

/// **编译期生成**的 `(sin, cos)` 表（`const fn` + Taylor，`f64`）。
///
/// 只读共享 ⇒ 多线程安全；`static` 一份（每个类型/精度一份）。
pub static COS_SIN_TAB: [(f64, f64); N_TAB] = build_tab();

/// `const fn` 版 `sin`（`x ∈ [0, π/2]`，Taylor 到 `x^17`；供表生成用）。
const fn csin(x: f64) -> f64 {
    // x - x^3/3! + x^5/5! - …（到 x^25；x→π/2 处截断 ~1e-19）
    let x2 = x * x;
    let mut term = x;
    let mut acc = x;
    let mut k = 1u32;
    while k < 13 {
        let d = ((2 * k) * (2 * k + 1)) as f64;
        term = -term * x2 / d;
        acc += term;
        k += 1;
    }
    acc
}

/// `const fn` 版 `cos`（`x ∈ [0, π/2]`，Taylor 到 `x^24`）。
const fn ccos(x: f64) -> f64 {
    let x2 = x * x;
    let mut term = 1.0f64;
    let mut acc = 1.0f64;
    let mut k = 1u32;
    while k < 13 {
        let d = ((2 * k - 1) * (2 * k)) as f64;
        term = -term * x2 / d;
        acc += term;
        k += 1;
    }
    acc
}

const fn build_tab() -> [(f64, f64); N_TAB] {
    let mut t = [(0.0f64, 0.0f64); N_TAB];
    let mut i = 0;
    while i < N_TAB {
        let x = (i as f64) * TAB_STEP;
        t[i] = (csin(x), ccos(x));
        i += 1;
    }
    t
}

/// **(b) 表 + 2 项 Taylor 修正** 的 `(cos, sin)`：`x` 任意（自做 2π 归约）；
/// 精度 ≈ `d³/6`（`d ≤ TAB_STEP/2 ≈ 6.1e-3` ⇒ ~4e-8）。
pub fn fast_cos_sin(x: f64) -> (f64, f64) {
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut r = x % two_pi;
    if r < 0.0 {
        r += two_pi;
    }
    // 折到第一象限 + 象限号
    let (mut r, mut q) = (r, 0u32);
    if r > std::f64::consts::PI {
        r -= std::f64::consts::PI;
        q += 2;
    }
    if r > std::f64::consts::FRAC_PI_2 {
        r = std::f64::consts::PI - r;
        q += 1;
    }
    let idx = (r / TAB_STEP) as usize;
    let idx = idx.min(N_TAB - 2);
    let d = r - idx as f64 * TAB_STEP;
    let (s0, c0) = COS_SIN_TAB[idx];
    // sin(x0+d) ≈ s0 + d·c0 − d²/2·s0；cos(x0+d) ≈ c0 − d·s0 − d²/2·c0
    let (s, c) = (
        s0 + d * c0 - 0.5 * d * d * s0,
        c0 - d * s0 - 0.5 * d * d * c0,
    );
    match q & 3 {
        0 => (c, s),
        1 => (-c, s),
        2 => (-c, -s),
        _ => (c, -s),
    }
}

/// **(b) 表 + 多项式** 的 `exp(x)`：`2^i·exp(f)`（`f ∈ [−ln2/2, ln2/2]`，5 项 Taylor），
/// 指数部分用位运算缩放（无 `libm`）。
pub fn fast_exp(x: f64) -> f64 {
    if x < -700.0 {
        return 0.0;
    }
    if x > 700.0 {
        return f64::INFINITY;
    }
    let ln2 = std::f64::consts::LN_2;
    let t = x / ln2;
    let i = t.round();
    let f = x - i * ln2;
    // exp(f) Taylor（|f| ≤ 0.347 ⇒ 5 项已 ~1e-12）
    let mut term = 1.0f64;
    let mut acc = 1.0f64;
    let mut k = 1u32;
    while k <= 8 {
        term *= f / k as f64;
        acc += term;
        k += 1;
    }
    let scale = f64::from_bits(((1023i64 + i as i64) as u64) << 52);
    acc * scale
}

/// **(c) 等距递推**构造一条核（`σ` 固定、`ω` 固定）：`e^{−σ i Δt}` 逐次乘，
/// `cos/sin(ω i Δt)` 用旋转递推（~2 FMA/元素）。**误差随 `N` 累积**（实测界见诊断输出）。
pub fn recurrence_row(sig: f64, om: f64, dt: f64, n: usize, w: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut kc = vec![0.0f64; n];
    let mut ks = vec![0.0f64; n];
    let decay = (-sig * dt).exp(); // 一次 libm 调用（也可换 fast_exp）
    let (c1, s1) = ((om * dt).cos(), (om * dt).sin());
    let (mut e, mut c, mut s) = (1.0f64, 1.0f64, 0.0f64);
    for i in 0..n {
        let wi = w[i] * dt;
        kc[i] = e * c * wi;
        ks[i] = e * s * wi;
        // 递推下一步
        e *= decay;
        let (nc, ns) = (c * c1 - s * s1, s * c1 + c * s1);
        c = nc;
        s = ns;
    }
    (kc, ks)
}

/// **(a) 基线核**：`libm` 逐元素（与 `laplace::kernels64` 同式，供诊断对照）。
pub fn baseline_row(sig: f64, om: f64, dt: f64, n: usize, w: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut kc = vec![0.0f64; n];
    let mut ks = vec![0.0f64; n];
    for i in 0..n {
        let t = i as f64 * dt;
        let e = (-sig * t).exp() * w[i] * dt;
        kc[i] = e * (om * t).cos();
        ks[i] = e * (om * t).sin();
    }
    (kc, ks)
}

/// **(b) 表 + 多项式核**。
pub fn table_row(sig: f64, om: f64, dt: f64, n: usize, w: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut kc = vec![0.0f64; n];
    let mut ks = vec![0.0f64; n];
    for i in 0..n {
        let t = i as f64 * dt;
        let e = fast_exp(-sig * t) * w[i] * dt;
        let (c, s) = fast_cos_sin(om * t);
        kc[i] = e * c;
        ks[i] = e * s;
    }
    (kc, ks)
}

#[cfg(test)]
mod recon {
    use super::*;
    use std::time::Instant;

    /// 表大小与缓存占用（字节）。
    #[test]
    fn table_size_report() {
        let bytes = std::mem::size_of::<[(f64, f64); N_TAB]>();
        println!(
            "共享表：{N_TAB} 项 × (sin,cos) f64 = {bytes} B（{} KiB）；static 一份、const 生成、零运行时初始化",
            bytes / 1024
        );
        assert_eq!(bytes, N_TAB * 16);
    }

    /// **精度**：(b)/(c) 与 (a) 基线（libm）比：**绝对误差** + **掩码相对误差**
    /// （只统计 `|ref| > 1e-3·max|ref|` 的元素——核在衰减尾部会到 1e-9 量级，对那儿算"相对误差"
    /// 只会反映绝对误差，是**不公平指标**）+ 掩码 f32 ULP 差。
    #[test]
    fn accuracy_vs_libm() {
        let (jj_max, n) = (64usize, 1024usize);
        let dt = 1.0 / 256.0;
        let w = vec![1.0 / 3.0; n];
        let (mut abs_b, mut abs_c) = (0.0f64, 0.0f64);
        let (mut rel_b, mut rel_c) = (0.0f64, 0.0f64);
        let (mut ulp_b, mut ulp_c) = (0u32, 0u32);
        let mut peak = 0.0f64;
        for jj in 0..jj_max {
            let sig = 0.5 + 0.05 * jj as f64;
            let om = (jj % 16) as f64 * 0.25;
            let (br, _bs) = baseline_row(sig, om, dt, n, &w);
            let (tr, _ts) = table_row(sig, om, dt, n, &w);
            let (rr, _rs) = recurrence_row(sig, om, dt, n, &w);
            for v in &br {
                peak = peak.max(v.abs());
            }
            // 第二遍：掩码统计
            for i in 0..n {
                let a = (tr[i] - br[i]).abs();
                let b = (rr[i] - br[i]).abs();
                abs_b = abs_b.max(a);
                abs_c = abs_c.max(b);
                if br[i].abs() > 0.1 * peak {
                    rel_b = rel_b.max(a / br[i].abs());
                    rel_c = rel_c.max(b / br[i].abs());
                    ulp_b = ulp_b.max((br[i] as f32).to_bits().abs_diff((tr[i] as f32).to_bits()));
                    ulp_c = ulp_c.max((br[i] as f32).to_bits().abs_diff((rr[i] as f32).to_bits()));
                }
            }
        }
        println!("精度（对 libm 基线，J=64×N=1024 网格；核峰值 {peak:.3e}）：");
        println!(
            "  (b) 表+多项式：最大绝对误差 {abs_b:.3e}，掩码相对误差 {rel_b:.3e}，掩码 f32 ULP 差 {ulp_b}"
        );
        println!(
            "  (c) 等距递推  ：最大绝对误差 {abs_c:.3e}，掩码相对误差 {rel_c:.3e}，掩码 f32 ULP 差 {ulp_c}"
        );
        assert!(
            rel_b < 5e-6,
            "(b) 掩码相对误差应 <5e-6（实测界），实测 {rel_b:e}"
        );
        assert!(ulp_b <= 32, "(b) 掩码 f32 ULP 差应 ≤32，实测 {ulp_b}");
        assert!(rel_c < 1e-9, "(c) 掩码相对误差应 <1e-9，实测 {rel_c:e}");
    }

    /// **速度（同源口径）**：核构造三条路线，同进程交替 6 对丢首对。
    #[test]
    #[ignore = "recon：三路线核构造性能"]
    fn diag_kernel_construction_three_routes() {
        let (jmax, n) = (64usize, 1024usize);
        let dt = 1.0 / 256.0;
        let w = vec![1.0 / 3.0; n];
        let mut rows: Vec<(f64, f64, f64, i32, f64, String)> = Vec::new();
        for pair in 0..6 {
            let mut t = [0.0f64; 3];
            let mut iters = 0i32;
            let mut ms_max = 0.0f64;
            for step in 0..3 {
                let which = (pair + step) % 3;
                let mut cnt = 0i32;
                let mut mx = 0.0f64;
                let t0 = Instant::now();
                while t0.elapsed().as_secs_f64() < 0.05 {
                    let t1 = Instant::now();
                    for jj in 0..jmax {
                        let sig = 0.5 + 0.05 * jj as f64;
                        let om = (jj % 16) as f64 * 0.25;
                        match which {
                            0 => {
                                std::hint::black_box(baseline_row(sig, om, dt, n, &w));
                            }
                            1 => {
                                std::hint::black_box(table_row(sig, om, dt, n, &w));
                            }
                            _ => {
                                std::hint::black_box(recurrence_row(sig, om, dt, n, &w));
                            }
                        }
                    }
                    let d = t1.elapsed().as_secs_f64();
                    if d > mx {
                        mx = d;
                    }
                    cnt += 1;
                }
                t[which] = t0.elapsed().as_secs_f64() / cnt.max(1) as f64;
                iters = cnt;
                if mx > ms_max {
                    ms_max = mx;
                }
            }
            let l1 = std::fs::read_to_string("/proc/loadavg")
                .ok()
                .and_then(|s| s.split_whitespace().next().map(str::to_owned))
                .unwrap_or_default();
            rows.push((t[0], t[1], t[2], iters, ms_max * 1e3, l1));
        }
        let med = |i: usize| -> f64 {
            let mut v: Vec<f64> = rows[1..]
                .iter()
                .map(|r| match i {
                    0 => r.0,
                    1 => r.1,
                    _ => r.2,
                })
                .collect();
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let (a, b, c) = (med(0), med(1), med(2));
        for (i, r) in rows.iter().enumerate() {
            println!(
                "pair{} baseline={:.4}ms table={:.4}ms recur={:.4}ms iters={} ms_max={:.3} loadavg={}{}",
                i + 1,
                r.0 * 1e3,
                r.1 * 1e3,
                r.2 * 1e3,
                r.3,
                r.4,
                r.5,
                if i == 0 { " (dropped)" } else { "" }
            );
        }
        println!(
            "核构造中位（J=64×N=1024）：baseline {:.4} ms｜表+多项式 {:.4} ms（{:.2}×）｜递推 {:.4} ms（{:.2}×）",
            a * 1e3,
            b * 1e3,
            b / a,
            c * 1e3,
            c / a
        );
    }

    /// **收益边界**：核构造 vs 整条变换（小/大 `NB`）。
    #[test]
    #[ignore = "recon：小/大 NB 两情景（核占比与推算改后）"]
    fn diag_kernel_share_small_and_large_nb() {
        use crate::laplace::laplace_batch_f32;
        const M: usize = 64;
        const N: usize = 1024;
        let dt = 1.0 / 256.0;
        let sig: Vec<f64> = (0..M).map(|j| 0.5 + 0.05 * j as f64).collect();
        let om: Vec<f64> = (0..M).map(|j| (j % 16) as f64 * 0.25).collect();
        let w = vec![1.0 / 3.0; N];
        let time_kernel = |fast: bool| -> f64 {
            let mut ts = Vec::new();
            for _ in 0..7 {
                let t0 = Instant::now();
                for jj in 0..M {
                    if fast {
                        std::hint::black_box(table_row(sig[jj], om[jj], dt, N, &w));
                    } else {
                        std::hint::black_box(baseline_row(sig[jj], om[jj], dt, N, &w));
                    }
                }
                ts.push(t0.elapsed().as_secs_f64());
            }
            ts.sort_by(f64::total_cmp);
            ts[ts.len() / 2]
        };
        let old_k = time_kernel(false);
        let new_k = time_kernel(true);
        for (label, nb) in [("小 NB", 2usize), ("大 NB", 64usize)] {
            let f: Vec<f32> = vec![1.0f32; N * nb];
            let mut ts = Vec::new();
            for _ in 0..7 {
                let t0 = Instant::now();
                match nb {
                    2 => {
                        let _ = laplace_batch_f32::<M, N, 2>(&sig, &om, dt, &f);
                    }
                    _ => {
                        let _ = laplace_batch_f32::<M, N, 64>(&sig, &om, dt, &f);
                    }
                }
                ts.push(t0.elapsed().as_secs_f64());
            }
            ts.sort_by(f64::total_cmp);
            let total = ts[ts.len() / 2];
            let est = total - old_k + new_k;
            println!(
                "{label}（NB={nb}）：整条 {:.4} ms = 核 {:.4} ms（{:.1}%）+ 其余 {:.4} ms；推算改后 {:.4} ms（{:+.2}%）",
                total * 1e3,
                old_k * 1e3,
                old_k / total * 100.0,
                (total - old_k) * 1e3,
                est * 1e3,
                (est - total) / total * 100.0
            );
        }
        println!(
            "（注意：\"改后 ms\" 是**推算** = 整条 − 旧核 + 新核；核与整条各自独立测量，非同一次交替 A/B）"
        );
    }
}
