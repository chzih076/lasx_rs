//! int8 对称量化的**生产端与反量化**：`amax` / `absmax_rows`、`quantize_i8_*`、`dequantize_i8*`。
//!
//! 这一族是 `docs/ops.md` §12.3 **N3 批次**的基础（消费端是 `gemv_i8` 与 `dot_i8`）。
//! 设计记录见 `docs/dev.md` §21；本文件的注释只写**实现契约**。
//!
//! **LASX-only**：没有降级分支（与 N1 批次一致，见手册 Caveats 与 `docs/ops.md` §3.2）。
//!
//! # 数值契约（调用方必须照此实现参考版才逐位一致）
//!
//! | 量 | 定义 |
//! |---|---|
//! | 对称范围 | `qmax = 127`，`q ∈ [−127, 127]`（**不用 −128**，避免不对称） |
//! | `amax` | `max_i \|x_i\|`（与 `f32::max` 折叠逐位一致，含 NaN，见下） |
//! | `scale` | `amax / 127.0`（f32 除法，每张量/每行一次）；`amax == 0` ⇒ `scale = 0` 且输出全 0 |
//! | `q` | `round_ties_even(x · (1.0/scale)) as i8`（**不夹取**，见下） |
//! | 反量化 | `x̃ = (q as f32) · scale`（**一次乘法**、一次舍入） |
//!
//! 三条必须照抄的细节：
//!
//! 1. **用倒数乘、不用除法**：LASX **没有向量除法指令**，所以量化写成 `x · (1.0/scale)`。
//!    这与 `softmax_rows` 的 `e · (1/Σ)` 是同一条约定：参考实现若写 `x / scale`，
//!    舍入点不同，结果不保证逐位一致。
//! 2. **取整是 ties-to-even**：向量侧 `xvftintrne.w.s`、标量侧 `f32::round_ties_even()`。
//!    两者已实测一致（`/tmp` 探针：`0.5→0、1.5→2、2.5→2、−0.5→0、−1.5→−2、−2.5→−2`）。
//! 3. **不做夹取**，`|q| ≤ 127` 由 scale 的定义保证：`scale = amax/127` 且 `|x| ≤ amax`
//!    ⇒ `|x·(1/scale)| ≤ 127`（`scale` 的舍入只带来 ~2⁻²⁴ 相对误差，离 127.5 的取整门槛
//!    差着 5 个数量级，实测边界 `|x| = amax` 稳定落在 127）。省掉夹取既是少 2 条向量指令，
//!    也是**让含 NaN 的输入在两条路径上给出同一结果**（夹取用的是 maxNum 语义，
//!    会把 NaN 变成 ∓127，而标量侧的 `as i32` 给 0 —— 两条路径会不一致）。
//!
//! # NaN / ±inf
//!
//! **不在契约内**（与 `softmax_rows` 对 `+inf` 的处理一致）。实现上 `|x|` 用
//! `xvfmax.s(x, −x)`，已实测与 `f32::abs` 逐 lane 相同；归约用 `f32::max` 折叠，
//! **NaN 被忽略**——所以**全 NaN 输入 ⇒ `amax = 0.0` ⇒ `scale = 0` ⇒ 输出全 0**
//! （`scale == 0` 这条分支直接写 0，不走"乘倒数"）。含 ±inf 时 `scale` 为 inf，输出无意义。
//!
//! 稀疏 NaN（与有限值混在一起）也不在契约内；实现上两条路径**一致地**把它落到
//! `xvftintrne(±NaN)` / `NaN.round_ties_even() as i8` 的结果（测试里逐位对照），
//! 但这只是"两条路径一致"，不是"结果有意义"。
//!
//! # 实现形态（哪一段是向量、哪一段是标量）
//!
//! - `amax` / 量化：**向量主体 + 标量尾**，两条路径同一串运算；
//! - `dequantize`：**逐元素标量**（`(q as f32) · scale`）。i8 → i32 的加宽链
//!   （`xvaddwev/xvaddwod`）出来是 even/odd **交织** lane，要写回连续下标还得一次交织重排，
//!   而这一族只在残差/归一化的出口用；现在按"先正确再谈速度"取标量，**未做向量化**
//!   （也不声称它慢——没有实测就不写结论）。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（策略见 `docs/dev.md` §17）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]
use crate::arch::lasx;
use std::arch::loongarch64::*;

/// 对称量化的 `qmax`（`q ∈ [−qmax, qmax]`）。
const QMAX: f32 = 127.0;

/// `|x|`：`max(x, −x)`（与 `f32::abs` 逐 lane 相同，见模块头）。
#[inline]
unsafe fn abs_vec(v: m256) -> m256 {
    let z = lasx::zero_f32x8();
    unsafe { lasx_xvfmax_s(v, lasx_xvfsub_s(z, v)) }
}

/// 把 f32 标量广播成 256 位向量。
#[inline]
fn splat(v: f32) -> m256 {
    lasx::splat_f32(v)
}

/// 8 个 lane 按**下标升序**用 `f32::max` 折叠成一个标量（max 满足交换律，次序只为可复现）。
#[inline]
fn fold_max8(tmp: &[f32; 8]) -> f32 {
    let mut m = tmp[0];
    for &v in &tmp[1..] {
        m = m.max(v);
    }
    m
}

/// 最大绝对值（逐张量）：`max_i |x_i|`。空输入返回 `0.0`。
pub(crate) fn amax(x: &[f32]) -> f32 {
    let n = x.len();
    let mut m = splat(f32::NEG_INFINITY);
    let mut i = 0;
    while i + 8 <= n {
        let v = unsafe { lasx::load_f32x8(x.as_ptr().add(i)) };
        m = unsafe { lasx_xvfmax_s(m, abs_vec(v)) };
        i += 8;
    }
    let mut tmp = [0f32; 8];
    unsafe { lasx::store_f32x8(tmp.as_mut_ptr(), m) };
    let mut best = fold_max8(&tmp);
    while i < n {
        best = best.max(x[i].abs());
        i += 1;
    }
    if best == f32::NEG_INFINITY { 0.0 } else { best }
}

/// 逐行最大绝对值：`out[r] = max_j |x[r·cols + j]|`。
///
/// per-token 激活（`[token, hidden]`）与 per-channel 权重（`[out, in]`）**都是行主序按行归约**，
/// 所以共用这一条；语义差别在调用方，不在内核。
pub(crate) fn absmax_rows(x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    for r in 0..rows {
        out[r] = amax(&x[r * cols..r * cols + cols]);
    }
}

/// 由 `amax` 得到对称 `scale` 与它的倒数：`scale = amax/127`、`recip = 1/scale`。
///
/// `amax == 0`（全零输入）⇒ `(0.0, 0.0)`：量化时 `x·recip` 得 `±0`、取整为 0，
/// **不会**出现 `0 · inf = NaN`。`amax` 非有限（inf/NaN）不在契约内。
#[inline]
fn scale_from_amax(amax: f32) -> (f32, f32) {
    if amax > 0.0 && amax.is_finite() {
        let scale = amax / QMAX;
        (scale, 1.0 / scale)
    } else {
        (0.0, 0.0)
    }
}

/// 标量侧的"缩放 → ties-even 取整"（与向量路径同一串运算，**不夹取**）。
#[inline]
fn round_i8(v: f32) -> i8 {
    v.round_ties_even() as i8
}

/// 一段（整张量或一行）的量化：写 `q[0..len]`。
///
/// 向量主体 + 标量尾用同一串运算（`x·recip → round_ties_even`），故逐位一致。
/// `recip == 0`（全零/全 NaN 输入）⇒ 直接写 0：`NaN · 0 = NaN` 会让"乘倒数"这条路
/// 走进未定义区，显式分支既保证不产生 NaN、也与契约里"`scale = 0` ⇒ 输出全 0"对齐。
#[inline]
fn quantize_span(x: &[f32], q: &mut [i8], recip: f32) {
    if recip == 0.0 {
        q.fill(0);
        return;
    }
    let n = x.len();
    let rv = splat(recip);
    let mut tmp = [0i32; 8];
    let mut i = 0;
    while i + 8 <= n {
        let v = unsafe { lasx::load_f32x8(x.as_ptr().add(i)) };
        let scaled = unsafe { lasx_xvfmul_s(v, rv) };
        let qi = unsafe { lasx_xvftintrne_w_s(scaled) };
        unsafe { lasx_xvst(qi, tmp.as_mut_ptr() as *mut i8, 0) };
        for (k, &t) in tmp.iter().enumerate() {
            q[i + k] = t as i8;
        }
        i += 8;
    }
    while i < n {
        q[i] = round_i8(x[i] * recip);
        i += 1;
    }
}

/// 逐张量对称量化（`out` 与 `x` 等长），返回 `scale`。
pub(crate) fn quantize_i8_per_tensor(x: &[f32], out: &mut [i8]) -> f32 {
    let (scale, recip) = scale_from_amax(amax(x));
    quantize_span(x, out, recip);
    scale
}

/// 逐行对称量化：每行一个 `scale`（写进 `scales[0..rows]`）。
pub(crate) fn quantize_i8_per_row(
    x: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [i8],
    scales: &mut [f32],
) {
    debug_assert!(x.len() >= rows * cols && out.len() >= rows * cols && scales.len() >= rows);
    for (r, s) in scales.iter_mut().take(rows).enumerate() {
        let lo = r * cols;
        let hi = lo + cols;
        let (scale, recip) = scale_from_amax(amax(&x[lo..hi]));
        *s = scale;
        quantize_span(&x[lo..hi], &mut out[lo..hi], recip);
    }
}

/// 逐张量反量化：`out[i] = (q[i] as f32) · scale`（逐元素标量，见模块头）。
pub(crate) fn dequantize_i8(q: &[i8], scale: f32, out: &mut [f32]) {
    for (o, &v) in out.iter_mut().zip(q.iter()) {
        *o = (v as f32) * scale;
    }
}

/// 逐行反量化：每行用自己的 `scales[r]`。
pub(crate) fn dequantize_i8_per_row(
    q: &[i8],
    rows: usize,
    cols: usize,
    scales: &[f32],
    out: &mut [f32],
) {
    debug_assert!(q.len() >= rows * cols && out.len() >= rows * cols && scales.len() >= rows);
    for (r, &s) in scales.iter().take(rows).enumerate() {
        let lo = r * cols;
        let hi = lo + cols;
        dequantize_i8(&q[lo..hi], s, &mut out[lo..hi]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 标量参考实现：**照契约写**（`x·recip`、ties-even、不夹取、`recip==0` 直接 0），
    /// 不是"数学上等价"的写法。
    fn scalar_quantize(x: &[f32]) -> (Vec<i8>, f32) {
        let amax = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let (scale, recip) = scale_from_amax(amax);
        let q = if recip == 0.0 {
            vec![0i8; x.len()]
        } else {
            x.iter().map(|&v| round_i8(v * recip)).collect()
        };
        (q, scale)
    }

    fn lcg(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u32 << 31) as f32) * 2.0 - 1.0
        }
    }

    /// 覆盖向量尾/标量尾的所有规模（含 0、1、8 的倍数与非倍数）。
    const LENS: &[usize] = &[0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 33, 64, 127, 128, 1000];

    /// `amax` 与 `f32::max` 折叠**逐位**一致（含 NaN 分布）。
    #[test]
    fn test_amax_matches_scalar_bitwise() {
        let mut rnd = lcg(0x1234);
        for &n in LENS {
            let x: Vec<f32> = (0..n).map(|_| 100.0 * rnd()).collect();
            let want = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            assert_eq!(amax(&x).to_bits(), want.to_bits(), "n={n} 随机");

            // 稀疏 NaN / ±0 / 极小值：向量与标量的 NaN 语义已实测一致
            let mut y = x.clone();
            if n > 0 {
                y[n / 3] = f32::NAN;
                y[n - 1] = -0.0;
                y[0] = 1e-30;
            }
            let want = y.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            assert_eq!(amax(&y).to_bits(), want.to_bits(), "n={n} 含 NaN/±0");
        }
        // 全 NaN：归约忽略 NaN ⇒ amax = 0（⇒ scale = 0 ⇒ 量化输出全 0）。
        // 与标量参考（`fold(0.0, max)`）一致，且比"返回 NaN"更可测：向量/标量对所有输入同结果。
        let all_nan = [f32::NAN; 16];
        assert_eq!(amax(&all_nan), 0.0, "全 NaN ⇒ 忽略后无数据 ⇒ 0");
        let mut q = vec![7i8; 16];
        assert_eq!(quantize_i8_per_tensor(&all_nan, &mut q), 0.0);
        assert_eq!(
            q,
            vec![0i8; 16],
            "全 NaN ⇒ scale 0 ⇒ 输出全 0（不产生 NaN）"
        );
        // 空输入 → 0.0
        assert_eq!(amax(&[]), 0.0);
    }

    /// `absmax_rows` 与逐行标量一致。
    #[test]
    fn test_absmax_rows_matches_scalar() {
        let rows = 5usize;
        let cols = 13usize;
        let mut rnd = lcg(0xbeef);
        let x: Vec<f32> = (0..rows * cols).map(|_| 50.0 * rnd()).collect();
        let mut out = vec![0.0f32; rows];
        absmax_rows(&x, rows, cols, &mut out);
        for r in 0..rows {
            let want = x[r * cols..(r + 1) * cols]
                .iter()
                .fold(0.0f32, |m, &v| m.max(v.abs()));
            assert_eq!(out[r].to_bits(), want.to_bits(), "row {r}");
        }
    }

    /// 量化与标量参考**逐位一致**（i8 输出 + scale 的位型）。
    #[test]
    fn test_quantize_per_tensor_matches_scalar_bitwise() {
        let mut rnd = lcg(0x5eed);
        for &n in LENS {
            let x: Vec<f32> = (0..n).map(|_| 300.0 * rnd()).collect();
            let mut q = vec![0i8; n];
            let scale = quantize_i8_per_tensor(&x, &mut q);
            let (want_q, want_scale) = scalar_quantize(&x);
            assert_eq!(q, want_q, "n={n} i8 输出");
            assert_eq!(scale.to_bits(), want_scale.to_bits(), "n={n} scale");
        }
        // 退化：全零 → scale = 0、输出全 0（且不产生 NaN）
        let zeros = vec![0.0f32; 9];
        let mut q = vec![7i8; 9];
        let scale = quantize_i8_per_tensor(&zeros, &mut q);
        assert_eq!(scale, 0.0);
        assert_eq!(q, vec![0i8; 9]);
        // 退化：单个元素 / 极大极小
        for v in [1.0f32, -1.0, 1e30, -1e30, 1e-30, f32::MAX] {
            let x = [v, v * 0.5];
            let mut q = vec![0i8; 2];
            let scale = quantize_i8_per_tensor(&x, &mut q);
            let (want_q, _) = scalar_quantize(&x);
            assert_eq!(q, want_q, "x={v}");
            assert!(scale.is_finite());
        }
    }

    /// **动态范围断言**：满量程数据必须打到 ±120 以上。
    ///
    /// 这是那次真实踩坑的回归测试——漏除 `qmax`（`s = max|w|` 而不是 `max|w|/127`）会让
    /// `round(w/s) ∈ {−1,0,1}`，权重退化为三值，而指标"看起来合理"不会暴露它。
    #[test]
    fn test_quantize_uses_full_dynamic_range() {
        let mut rnd = lcg(0xc0de);
        let x: Vec<f32> = (0..512).map(|_| 10.0 * rnd()).collect();
        let mut q = vec![0i8; x.len()];
        let _ = quantize_i8_per_tensor(&x, &mut q);
        let peak = q.iter().map(|v| v.unsigned_abs()).max().unwrap();
        assert!(
            peak >= 120,
            "量化动态范围不足：max|q| = {peak}（漏除 qmax？）"
        );
        assert!(q.iter().all(|v| *v != i8::MIN), "不用 −128（对称范围）");

        // 边界：|x| = amax 必须落在 ±127（这正是"不做夹取也安全"的依据）
        let a = 3.75f32;
        let x = [a, a * 0.5, -a, 0.0, a * 0.99, -a * 0.5, a * 0.01, -0.0];
        let mut q = vec![0i8; 8];
        let scale = quantize_i8_per_tensor(&x, &mut q);
        assert_eq!(scale.to_bits(), (a / 127.0).to_bits());
        assert_eq!(q[0], 127, "amax 处应打到 +127");
        assert_eq!(q[2], -127, "−amax 处应打到 −127");
        let (want_q, _) = scalar_quantize(&x);
        assert_eq!(q, want_q, "边界样本也要与标量参考逐位一致");
    }

    /// 量化误差 ≤ `scale/2`（契约里 `q = round(x·recip)`，故 `|x·recip − q| ≤ 0.5`；
    /// 反量化时 `recip` 与 `scale` 的互逆各有 1 ulp 级误差，允许 1e-6 相对余量）。
    #[test]
    fn test_quantize_error_bounded_by_half_scale() {
        let mut rnd = lcg(0xf00d);
        let x: Vec<f32> = (0..999).map(|_| 7.0 * rnd()).collect();
        let mut q = vec![0i8; x.len()];
        let scale = quantize_i8_per_tensor(&x, &mut q);
        let mut worst = 0.0f32;
        for (&v, &qi) in x.iter().zip(q.iter()) {
            worst = worst.max(((qi as f32) * scale - v).abs());
        }
        assert!(
            worst <= 0.5 * scale * (1.0 + 1e-6),
            "worst={worst:e} > scale/2={:e}",
            0.5 * scale
        );
    }

    /// 逐行量化：每行 scale 与标量一致，且**行间互不影响**（不同幅度的行各自归一）。
    #[test]
    fn test_quantize_per_row_scales_are_independent() {
        let rows = 4usize;
        let cols = 10usize;
        // 每行幅度差 1000 倍：per-tensor 会把小行压成 0，per-row 不会
        let x: Vec<f32> = (0..rows)
            .flat_map(|r| {
                let a = 10f32.powi(r as i32 - 2);
                (0..cols).map(move |j| a * ((j as f32) / cols as f32 * 2.0 - 1.0))
            })
            .collect();
        let mut q = vec![0i8; rows * cols];
        let mut scales = vec![0f32; rows];
        quantize_i8_per_row(&x, rows, cols, &mut q, &mut scales);
        for r in 0..rows {
            let row = &x[r * cols..(r + 1) * cols];
            let (want_q, want_scale) = scalar_quantize(row);
            assert_eq!(&q[r * cols..(r + 1) * cols], &want_q[..], "row {r}");
            assert_eq!(scales[r].to_bits(), want_scale.to_bits(), "row {r} scale");
            let peak = want_q.iter().map(|v| v.unsigned_abs()).max().unwrap();
            assert!(peak >= 120, "row {r} 也应用满量程：peak={peak}");
        }
    }

    /// 反量化：逐位等于 `(q as f32)·scale`，且是 `scale` 的整数倍。
    #[test]
    fn test_dequantize_is_exact_multiple_of_scale() {
        let mut rnd = lcg(0xabcd);
        let x: Vec<f32> = (0..257).map(|_| 3.5 * rnd()).collect();
        let mut q = vec![0i8; x.len()];
        let scale = quantize_i8_per_tensor(&x, &mut q);
        let mut back = vec![0f32; x.len()];
        dequantize_i8(&q, scale, &mut back);
        for i in 0..x.len() {
            assert_eq!(
                back[i].to_bits(),
                ((q[i] as f32) * scale).to_bits(),
                "第 {i} 个元素不是 scale 的整数倍"
            );
            // 量化值就是整数（对称 int8 的定义）
            assert_eq!(q[i] as f32 % 1.0, 0.0);
        }
        // 逐行反量化与逐张量一致（同一行数据）
        let rows = 3usize;
        let cols = 4usize;
        let mut scales = vec![0f32; rows];
        let mut qr = vec![0i8; rows * cols];
        quantize_i8_per_row(&x[..rows * cols], rows, cols, &mut qr, &mut scales);
        let mut out = vec![0f32; rows * cols];
        dequantize_i8_per_row(&qr, rows, cols, &scales, &mut out);
        for (r, &s) in scales.iter().enumerate() {
            for j in 0..cols {
                let i = r * cols + j;
                assert_eq!(out[i].to_bits(), ((qr[i] as f32) * s).to_bits());
            }
        }
    }
}
