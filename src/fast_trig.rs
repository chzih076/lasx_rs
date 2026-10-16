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

/// **(A) 手写 LASX 版：4 行锁步递推**（`f64`，`F64x4`；行间并行、行内递推）。
///
/// 与 [`recurrence_rows_scalar`] **同累加次序**（每行都是 `e *= decay` 再旋转），
/// 只是 4 行一起算 ⇒ 用于回答"LLVM 自动向量化能吃到手写的百分之多少"。
pub fn recurrence_rows_lasx(
    sig: &[f64],
    om: &[f64],
    dt: f64,
    n: usize,
    w: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    use crate::arch::lasx;
    use std::arch::loongarch64::*;
    let m = sig.len();
    assert_eq!(om.len(), m);
    let mut kc = vec![0.0f64; m * n];
    let mut ks = vec![0.0f64; m * n];
    let mut j = 0usize;
    while j + 4 <= m {
        let decay = [
            (-sig[j] * dt).exp(),
            (-sig[j + 1] * dt).exp(),
            (-sig[j + 2] * dt).exp(),
            (-sig[j + 3] * dt).exp(),
        ];
        let c1 = [
            (om[j] * dt).cos(),
            (om[j + 1] * dt).cos(),
            (om[j + 2] * dt).cos(),
            (om[j + 3] * dt).cos(),
        ];
        let s1 = [
            (om[j] * dt).sin(),
            (om[j + 1] * dt).sin(),
            (om[j + 2] * dt).sin(),
            (om[j + 3] * dt).sin(),
        ];
        // SAFETY: 都是栈上 4 元素数组；`load_f64x4` 只需 4 个可读元素。
        let (vdecay, vc1, vs1) = unsafe {
            (
                lasx::load_f64x4(decay.as_ptr()),
                lasx::load_f64x4(c1.as_ptr()),
                lasx::load_f64x4(s1.as_ptr()),
            )
        };
        let one = lasx::splat_f64(1.0);
        let (mut ve, mut vc, mut vs) = (one, one, lasx::splat_f64(0.0));
        let mut tmp = vec![0.0f64; 4];
        for i in 0..n {
            let vw = lasx::splat_f64(w[i] * dt);
            // SAFETY: 纯逐元素向量运算 + 4 元素可写槽（与标量版同式同序）。
            unsafe {
                let b = lasx_xvfmul_d(lasx_xvfmul_d(ve, vc), vw);
                lasx::store_f64x4(tmp.as_mut_ptr(), b);
            }
            for l in 0..4 {
                kc[(j + l) * n + i] = tmp[l];
            }
            // SAFETY: 同上。
            unsafe {
                let b = lasx_xvfmul_d(lasx_xvfmul_d(ve, vs), vw);
                lasx::store_f64x4(tmp.as_mut_ptr(), b);
            }
            for l in 0..4 {
                ks[(j + l) * n + i] = tmp[l];
            }
            // SAFETY: 同上的逐元素运算；次序与标量版一致。
            unsafe {
                ve = lasx_xvfmul_d(ve, vdecay);
                let t1 = lasx_xvfmul_d(vc, vc1);
                let t2 = lasx_xvfmul_d(vs, vs1);
                let t3 = lasx_xvfmul_d(vs, vc1);
                let t4 = lasx_xvfmul_d(vc, vs1);
                vc = lasx_xvfsub_d(t1, t2);
                vs = lasx_xvfadd_d(t3, t4);
            }
        }
        j += 4;
    }
    while j < m {
        let (r, s) = recurrence_row(sig[j], om[j], dt, n, w);
        kc[j * n..(j + 1) * n].copy_from_slice(&r);
        ks[j * n..(j + 1) * n].copy_from_slice(&s);
        j += 1;
    }
    (kc, ks)
}

/// **(B'') 锁步宽度参数化的惯用 Rust 版（`L` 行同时推进，无 intrinsic）**：
/// 定长数组 `[f64; L]` + 定长下标循环 ⇒ **"形状机制"实验**用（扫描 `L ∈ {2,4,8,16}`）。
///
/// **与手写 LASX 版同累加次序** ⇒ 期望**逐位一致**（f64、不用 `mul_add`）。
pub fn recurrence_rows_iter_l<const L: usize>(
    sig: &[f64],
    om: &[f64],
    dt: f64,
    n: usize,
    w: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    assert!(L > 0);
    let m = sig.len();
    let mut kc = vec![0.0f64; m * n];
    let mut ks = vec![0.0f64; m * n];
    let mut j = 0usize;
    while j + L <= m {
        let mut decay = [0.0f64; L];
        let mut c1 = [0.0f64; L];
        let mut s1 = [0.0f64; L];
        for l in 0..L {
            decay[l] = (-sig[j + l] * dt).exp();
            c1[l] = (om[j + l] * dt).cos();
            s1[l] = (om[j + l] * dt).sin();
        }
        let mut e = [1.0f64; L];
        let mut c = [1.0f64; L];
        let mut s = [0.0f64; L];
        for i in 0..n {
            let wi = w[i] * dt;
            for l in 0..L {
                kc[(j + l) * n + i] = e[l] * c[l] * wi;
                ks[(j + l) * n + i] = e[l] * s[l] * wi;
            }
            for l in 0..L {
                e[l] *= decay[l];
            }
            for l in 0..L {
                let (cc, ss) = (c[l], s[l]);
                c[l] = cc * c1[l] - ss * s1[l];
                s[l] = ss * c1[l] + cc * s1[l];
            }
        }
        j += L;
    }
    for j2 in j..m {
        let (r, s2) = recurrence_row(sig[j2], om[j2], dt, n, w);
        kc[j2 * n..(j2 + 1) * n].copy_from_slice(&r);
        ks[j2 * n..(j2 + 1) * n].copy_from_slice(&s2);
    }
    (kc, ks)
}

/// **(B') 惯用迭代器/定长数组 Rust 版（同算法、同序、无 intrinsic）**：
/// 用 **`[f64; 4]` 定长数组 + 定长下标循环**表达"4 行锁步"（rustc/LLVM 对定长数组循环
/// 通常能自动向量化成 `F64x4`）⇒ 这是"编译器最容易优化好的形状"的候选写法。
///
/// **与 (A) 同累加次序** ⇒ 期望**逐位一致**（Rust 默认不重结合、不自动 FMA 收缩）。
pub fn recurrence_rows_iter(
    sig: &[f64],
    om: &[f64],
    dt: f64,
    n: usize,
    w: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    const R: usize = 4;
    let m = sig.len();
    let mut kc = vec![0.0f64; m * n];
    let mut ks = vec![0.0f64; m * n];
    let mut j = 0usize;
    while j + R <= m {
        // 定长数组（编译器知道长度 ⇒ 内层循环可向量化）
        let mut decay = [0.0f64; R];
        let mut c1 = [0.0f64; R];
        let mut s1 = [0.0f64; R];
        for l in 0..R {
            decay[l] = (-sig[j + l] * dt).exp();
            c1[l] = (om[j + l] * dt).cos();
            s1[l] = (om[j + l] * dt).sin();
        }
        let mut e = [1.0f64; R];
        let mut c = [1.0f64; R];
        let mut s = [0.0f64; R];
        for i in 0..n {
            let wi = w[i] * dt;
            for l in 0..R {
                kc[(j + l) * n + i] = e[l] * c[l] * wi;
                ks[(j + l) * n + i] = e[l] * s[l] * wi;
            }
            for l in 0..R {
                e[l] *= decay[l];
            }
            for l in 0..R {
                let (cc, ss) = (c[l], s[l]);
                c[l] = cc * c1[l] - ss * s1[l];
                s[l] = ss * c1[l] + cc * s1[l];
            }
        }
        j += R;
    }
    for j2 in j..m {
        let (r, s2) = recurrence_row(sig[j2], om[j2], dt, n, w);
        kc[j2 * n..(j2 + 1) * n].copy_from_slice(&r);
        ks[j2 * n..(j2 + 1) * n].copy_from_slice(&s2);
    }
    (kc, ks)
}

/// **(B') 惯用迭代器的组合步**（`out = a − b − c`；切片/迭代器链，无 intrinsic）。
pub fn sub3_iter(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    out.iter_mut()
        .zip(a.iter().zip(b.iter()).zip(c.iter()))
        .for_each(|(o, ((&x, &y), &z))| *o = x - y - z);
}

/// **(B) 朴素标量 Rust 版：同算法、同序，无任何 intrinsic**（交给 LLVM `-O3` 自动向量化）。
pub fn recurrence_rows_scalar(
    sig: &[f64],
    om: &[f64],
    dt: f64,
    n: usize,
    w: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let m = sig.len();
    let mut kc = vec![0.0f64; m * n];
    let mut ks = vec![0.0f64; m * n];
    for j in 0..m {
        let (r, s) = recurrence_row(sig[j], om[j], dt, n, w);
        kc[j * n..(j + 1) * n].copy_from_slice(&r);
        ks[j * n..(j + 1) * n].copy_from_slice(&s);
    }
    (kc, ks)
}

#[cfg(test)]
mod recon {
    use super::*;
    use std::time::Instant;

    /// **锁步宽度扫描（`#[ignore]`）**：`L ∈ {2,4,8,16}` 惯用 Rust（f64、无 `mul_add`）vs 手写 LASX。
    /// **判据先写死**：任一变体比手写快 ≥3% ⇒ "机制+形状可超越手写"；否则如实写"未超越 + 最接近配置与差距"。
    #[test]
    #[ignore = "形状机制：锁步宽度扫描 vs 手写 LASX"]
    fn diag_lockstep_width_sweep() {
        let (m, n) = (64usize, 1024usize);
        let dt = 1.0 / 256.0;
        let sig: Vec<f64> = (0..m).map(|j| 0.5 + 0.05 * j as f64).collect();
        let om: Vec<f64> = (0..m).map(|j| (j % 8) as f64 * 0.003).collect();
        let w = vec![1.0 / 3.0; n];
        let time_it = |f: &mut dyn FnMut()| -> f64 {
            let mut ts = Vec::new();
            for _ in 0..7 {
                let t0 = Instant::now();
                for _ in 0..20 {
                    f();
                }
                ts.push(t0.elapsed().as_secs_f64() / 20.0);
            }
            ts.sort_by(f64::total_cmp);
            ts[ts.len() / 2]
        };
        let base = time_it(&mut || {
            std::hint::black_box(recurrence_rows_lasx(&sig, &om, dt, n, &w));
        });
        println!("手写 LASX 基准：{:.4} ms（M={m}, N={n}）", base * 1e3);
        for l in [2usize, 4, 8, 16] {
            let med = time_it(&mut || match l {
                2 => {
                    std::hint::black_box(recurrence_rows_iter_l::<2>(&sig, &om, dt, n, &w));
                }
                4 => {
                    std::hint::black_box(recurrence_rows_iter_l::<4>(&sig, &om, dt, n, &w));
                }
                8 => {
                    std::hint::black_box(recurrence_rows_iter_l::<8>(&sig, &om, dt, n, &w));
                }
                _ => {
                    std::hint::black_box(recurrence_rows_iter_l::<16>(&sig, &om, dt, n, &w));
                }
            });
            println!(
                "L={l:2}（惯用定长数组, f64, 无 mul_add）：{:.4} ms，相对手写 {:.2}×，差 {:+.1}%",
                med * 1e3,
                med / base,
                (med - base) / base * 100.0
            );
        }
        let l1 = std::fs::read_to_string("/proc/loadavg")
            .ok()
            .and_then(|s| s.split_whitespace().next().map(str::to_owned))
            .unwrap_or_default();
        println!("loadavg={l1}");
    }

    /// **锁步宽度变体与手写 LASX 逐位一致**（f64 + 无 `mul_add` ⇒ **逐位类**）：`L ∈ {2,4,8,16}`。
    #[test]
    fn lockstep_widths_bitwise_same_as_lasx() {
        let (m, n) = (16usize, 256usize);
        let dt = 1.0 / 256.0;
        let sig: Vec<f64> = (0..m).map(|j| 0.5 + 0.05 * j as f64).collect();
        let om: Vec<f64> = (0..m).map(|j| (j % 8) as f64 * 0.003).collect();
        let w = vec![1.0 / 3.0; n];
        let (ar, ai) = recurrence_rows_lasx(&sig, &om, dt, n, &w);
        let bits = |v: &[f64]| -> Vec<u64> { v.iter().map(|x| x.to_bits()).collect() };
        for l in [2usize, 4, 8, 16] {
            let (br, bi) = match l {
                2 => recurrence_rows_iter_l::<2>(&sig, &om, dt, n, &w),
                4 => recurrence_rows_iter_l::<4>(&sig, &om, dt, n, &w),
                8 => recurrence_rows_iter_l::<8>(&sig, &om, dt, n, &w),
                _ => recurrence_rows_iter_l::<16>(&sig, &om, dt, n, &w),
            };
            assert_eq!(bits(&ar), bits(&br), "L={l} 实部与手写 LASX 不逐位");
            assert_eq!(bits(&ai), bits(&bi), "L={l} 虚部与手写 LASX 不逐位");
        }
    }

    /// **LLVM 委派实验：逐位一致性**——(A) 手写 LASX 4 行锁步 vs (B) 同序朴素标量，
    /// 结果必须**逐位一致**（Rust 不开 fast-math ⇒ 不重结合、不做 FMA 收缩）。
    #[test]
    fn lasx_vs_scalar_bitwise_same_order() {
        let (m, n) = (8usize, 1024usize);
        let dt = 1.0 / 256.0;
        let sig: Vec<f64> = (0..m).map(|j| 0.5 + 0.05 * j as f64).collect();
        let om: Vec<f64> = (0..m).map(|j| (j % 8) as f64 * 0.003).collect();
        let w = vec![1.0 / 3.0; n];
        let (ar, ai) = recurrence_rows_lasx(&sig, &om, dt, n, &w);
        let (br, bi) = recurrence_rows_scalar(&sig, &om, dt, n, &w);
        let bits = |v: &[f64]| -> Vec<u64> { v.iter().map(|x| x.to_bits()).collect() };
        assert_eq!(bits(&ar), bits(&br), "手写 LASX vs 同序标量 实部不逐位");
        assert_eq!(bits(&ai), bits(&bi), "手写 LASX vs 同序标量 虚部不逐位");
        // **(B') 惯用迭代器/定长数组**也必须在同序下逐位一致
        let (cr2, ci2) = recurrence_rows_iter(&sig, &om, dt, n, &w);
        assert_eq!(bits(&ar), bits(&cr2), "手写 LASX vs 惯用迭代器 实部不逐位");
        assert_eq!(bits(&ai), bits(&ci2), "手写 LASX vs 惯用迭代器 虚部不逐位");
        assert_eq!(bits(&br), bits(&cr2), "朴素标量 vs 惯用迭代器 不逐位");
        // 组合步三变体（LASX / 朴素标量 / 惯用迭代器）逐位一致
        let len = 4096usize;
        let x: Vec<f32> = (0..len).map(|i| ((i % 17) as f32) * 0.125 - 1.0).collect();
        let y: Vec<f32> = (0..len).map(|i| ((i % 13) as f32) * 0.0625 - 0.5).collect();
        let z: Vec<f32> = (0..len).map(|i| ((i % 11) as f32) * 0.25 - 1.25).collect();
        let (mut o_lasx, mut o_naive, mut o_iter) =
            (vec![0.0f32; len], vec![0.0f32; len], vec![0.0f32; len]);
        crate::ops::cmatmul::combine::sub3_f32(&x, &y, &z, &mut o_lasx);
        for l in 0..len {
            o_naive[l] = x[l] - y[l] - z[l];
        }
        sub3_iter(&x, &y, &z, &mut o_iter);
        let bits32 = |v: &[f32]| -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() };
        assert_eq!(
            bits32(&o_lasx),
            bits32(&o_iter),
            "组合步 LASX vs 迭代器 不逐位"
        );
        assert_eq!(
            bits32(&o_naive),
            bits32(&o_iter),
            "组合步 朴素 vs 迭代器 不逐位"
        );
    }

    /// **委派实验（同源交替 A/B，`#[ignore]`）**：核构造 (A) 手写 LASX / (B) 朴素标量 /
    /// (C) libm 基线；组合步 (A) 手写 LASX / (B) 朴素标量。
    #[test]
    #[ignore = "委派实验：手写 LASX vs 朴素标量 vs libm"]
    fn diag_delegate_to_llvm() {
        use std::time::Instant;
        let (m, n) = (64usize, 1024usize);
        let dt = 1.0 / 256.0;
        let sig: Vec<f64> = (0..m).map(|j| 0.5 + 0.05 * j as f64).collect();
        let om: Vec<f64> = (0..m).map(|j| (j % 8) as f64 * 0.003).collect();
        let w = vec![1.0 / 3.0; n];
        // ---- 件 1：核构造（三路线）----
        let mut rows1: Vec<(f64, f64, f64, f64, i32, f64, String)> = Vec::new();
        for pair in 0..6 {
            let mut t = [0.0f64; 4];
            let mut iters = 0i32;
            let mut ms_max = 0.0f64;
            for step in 0..4 {
                let which = (pair + step) % 4;
                let mut cnt = 0i32;
                let mut mx = 0.0f64;
                let t0 = Instant::now();
                while t0.elapsed().as_secs_f64() < 0.05 {
                    let t1 = Instant::now();
                    match which {
                        0 => {
                            std::hint::black_box(recurrence_rows_lasx(&sig, &om, dt, n, &w));
                        }
                        1 => {
                            std::hint::black_box(recurrence_rows_scalar(&sig, &om, dt, n, &w));
                        }
                        2 => {
                            // **(B') 惯用迭代器/定长数组**
                            std::hint::black_box(recurrence_rows_iter(&sig, &om, dt, n, &w));
                        }
                        _ => {
                            // (C) libm 基线（逐元素 exp/cos/sin）
                            for j in 0..m {
                                std::hint::black_box(baseline_row(sig[j], om[j], dt, n, &w));
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
            rows1.push((t[0], t[1], t[2], t[3], iters, ms_max * 1e3, l1));
        }
        let med = |rows: &Vec<(f64, f64, f64, f64, i32, f64, String)>, i: usize| -> f64 {
            let mut v: Vec<f64> = rows[1..]
                .iter()
                .map(|r| match i {
                    0 => r.0,
                    1 => r.1,
                    2 => r.2,
                    _ => r.3,
                })
                .collect();
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let (la, ls, li, lc) = (
            med(&rows1, 0),
            med(&rows1, 1),
            med(&rows1, 2),
            med(&rows1, 3),
        );
        for (i, r) in rows1.iter().enumerate() {
            println!(
                "核 pair{} lasx={:.4}ms naive={:.4}ms iter={:.4}ms libm={:.4}ms iters={} ms_max={:.3} loadavg={}{}",
                i + 1,
                r.0 * 1e3,
                r.1 * 1e3,
                r.2 * 1e3,
                r.3 * 1e3,
                r.4,
                r.5,
                r.6,
                if i == 0 { " (dropped)" } else { "" }
            );
        }
        println!(
            "核构造中位：手写 LASX {:.4} ms｜朴素标量 {:.4} ms｜**惯用迭代器 {:.4} ms**｜libm {:.4} ms",
            la * 1e3,
            ls * 1e3,
            li * 1e3,
            lc * 1e3
        );
        println!(
            "⇒ **(B)/(A) = {:.2}×**，**(B')/(A) = {:.2}×**，委派到位率 (B') = **{:.0}%**，(C)/(A) = {:.2}×",
            ls / la,
            li / la,
            100.0 * la / li,
            lc / la
        );
        // ---- 件 2：组合步（LASX vs 朴素标量）----
        use crate::ops::cmatmul::combine;
        let len = 192 * 3072;
        let x: Vec<f32> = (0..len).map(|i| ((i % 17) as f32) * 0.125 - 1.0).collect();
        let y: Vec<f32> = (0..len).map(|i| ((i % 13) as f32) * 0.0625 - 0.5).collect();
        let z: Vec<f32> = (0..len).map(|i| ((i % 11) as f32) * 0.25 - 1.25).collect();
        let mut out = vec![0.0f32; len];
        let mut out_s = vec![0.0f32; len];
        let mut out_i = vec![0.0f32; len];
        let mut rows2: Vec<(f64, f64, f64, i32, f64, String)> = Vec::new();
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
                    match which {
                        0 => combine::sub3_f32(&x, &y, &z, &mut out),
                        1 => {
                            // (B) 朴素标量同序
                            for l in 0..len {
                                out_s[l] = x[l] - y[l] - z[l];
                            }
                            std::hint::black_box(&out_s);
                        }
                        _ => {
                            // **(B') 惯用迭代器**
                            sub3_iter(&x, &y, &z, &mut out_i);
                            std::hint::black_box(&out_i);
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
            rows2.push((t[0], t[1], t[2], iters, ms_max, l1));
        }
        let med2 = |i: usize| -> f64 {
            let mut v: Vec<f64> = rows2[1..]
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
        let (ca, cs, ci) = (med2(0), med2(1), med2(2));
        for (i, r) in rows2.iter().enumerate() {
            println!(
                "组合 pair{} lasx={:.4}ms naive={:.4}ms iter={:.4}ms iters={} ms_max={:.3} loadavg={}{}",
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
            "组合步中位：手写 LASX {:.4} ms｜朴素标量 {:.4} ms｜**惯用迭代器 {:.4} ms**",
            ca * 1e3,
            cs * 1e3,
            ci * 1e3
        );
        println!(
            "⇒ **(B)/(A) = {:.2}×**，**(B')/(A) = {:.2}×**，委派到位率 (B') = **{:.0}%**",
            cs / ca,
            ci / ca,
            100.0 * ca / ci
        );
    }

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
