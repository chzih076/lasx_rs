//! `lasx_dot_f16` / `lasx_gemv_f16` —— f16 权重 × f32 激活的点积与矩阵-向量。
//!
//! **LASX-only**：没有降级分支，无 LASX 的 CPU 上会执行 LASX 指令（见手册 Caveats）。
//!
//! # 为什么这个算子的价值在**带宽**上
//!
//! decode 阶段每算一个输出元素都要读整行权重，所以 `gemv` 是纯带宽题：权重用 f16 存 =
//! 访存量减半。f16→f32 的转换是**精确**的（f16 的每个值 f32 都能表示），所以误差只来自
//! "权重本来是 f16"这件事本身（相对 ~2^-11 ≈ 4.9e-4），转换本身不欠账。
//!
//! **但大矩阵上它是"转换受限"而不是"带宽受限"**（2026-09-25 控制变量探针，见
//! `docs/dev.md` §20.7）：纯 f16 流能跑 15.8–16.4 GB/s，而加进 `cvt`（+2.2 ms）与
//! `permi_q`（+1.6 ms）之后掉到 4.3 GB/s。这两个的数量由契约决定（`cvt` 每 8 个元素一条是
//! ISA 上限；`permi_q` 用来把结果拼回自然序），所以 3.75–4.3 GB/s 是**结构代价**：
//! 想更快要动下面的累加次序契约，不是调参能解决的。
//!
//! # 累加次序（契约本身；设计见 `docs/dev.md` §20.7）
//!
//! **4 条独立累加链**，每 32 个元素一轮：
//!
//! ```text
//! (a0,a1) = cvt(a[i..i+16]);  (a2,a3) = cvt(a[i+16..i+32])
//! c0 = fma(a0, b[i..i+8],   c0)      // 元素 i+k     → lane k
//! c1 = fma(a1, b[i+8..i+16],  c1)
//! c2 = fma(a2, b[i+16..i+24], c2)
//! c3 = fma(a3, b[i+24..i+32], c3)
//! ```
//!
//! 不足 32 个时补一轮 16 元素（只用 `c0`/`c1`）；再不足走标量尾、**延续同一 lane**
//! （`t[j % 8] += …`，两步舍入）。最后**固定次序**合并：`s = (c0+c1) + (c2+c3)`（lane 内），
//! 再把 8 个 lane 按两两归约折叠（与 softmax/rms_norm 同一棵树）。
//!
//! 为什么是 4 条链而不是 1 条：单链时每 16 个元素只有两条**互相依赖**的 FMA，把 FMA 延迟
//! 整段暴露出来——实测 `1×2048` 从 565 ns 降到 190 ns（3.0×）。
//!
//! 为什么不用 `ops::dot` 那套"4 条链 + 每 256 元素落盘 f64"：那是为了压 f32 累加误差，
//! 而这里 f16 的量化误差（4.9e-4）比 f32 累加误差（n = 4096 时 ~1e-4 量级）不会更小，
//! 再套 f64 落盘是白付复杂度。**实测确认**：与 f64 参考的相对误差在 1e-6 量级（见单测），
//! 比 f16 量化误差小两个数量级 ⇒ 累加口径够用。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（策略见 `docs/dev.md` §17）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]
use crate::arch::lasx;
use std::arch::loongarch64::*;

/// f16 位型 → f32（**精确**转换；与硬件 `xvfcvtl_s_h` 逐位一致，测试穷举全部 65536 个位型）。
///
/// 手写而不是用 `f16` 类型：本库零依赖、只吃 u16 位型；而这段也就十几行，且**可穷举验证**
/// ——次正规那条分支（f16 的次正规最小值 2^-24 在 f32 里是正规数）是最容易写错的地方。
#[inline]
pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let man = (bits & 0x03ff) as u32;
    let out = if exp == 0 {
        if man == 0 {
            sign // ±0
        } else {
            // 次正规：值 = man · 2^-24，规格化到 f32（把最高位顶到 bit 10）
            let mut m = man;
            let mut e: i32 = -14;
            while m & 0x0400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (((e + 127) as u32) << 23) | ((m & 0x03ff) << 13)
        }
    } else if exp == 0x1f {
        // ±inf / NaN（NaN 的 payload 由硬件定，不在逐位契约内）
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(out)
}

/// f16 权重与 f32 向量的点积：`Σ f16_to_f32(a[i]) · b[i]`。
///
/// `a` 与 `b` 等长（调用方保证）；长度 0 返回 `0.0`。
///
/// # Safety
/// 无额外前提：循环只在 `i + 16 <= n` 时载入 16 个 `u16`，标量尾只在 `i < n` 时下标访问。
pub(crate) unsafe fn dot_f16(a: &[u16], b: &[f32]) -> f32 {
    let n = a.len();
    debug_assert_eq!(n, b.len(), "dot_f16: 两个输入长度必须相等");
    // 4 条独立累加链（32 元素/轮）：单链会被 FMA 延迟卡住——实测 1×2048 时
    // 每 16 元素只有两条**互相依赖**的 FMA，等于把 4 周期延迟暴露出来。
    let mut c0 = lasx::zero_f32x8();
    let mut c1 = lasx::zero_f32x8();
    let mut c2 = lasx::zero_f32x8();
    let mut c3 = lasx::zero_f32x8();
    let mut i = 0;
    while i + 32 <= n {
        let (a0, a1) = lasx::load_f16x16_as_f32x8x2(a.as_ptr().add(i));
        let (a2, a3) = lasx::load_f16x16_as_f32x8x2(a.as_ptr().add(i + 16));
        c0 = lasx_xvfmadd_s(a0, lasx::load_f32x8(b.as_ptr().add(i)), c0);
        c1 = lasx_xvfmadd_s(a1, lasx::load_f32x8(b.as_ptr().add(i + 8)), c1);
        c2 = lasx_xvfmadd_s(a2, lasx::load_f32x8(b.as_ptr().add(i + 16)), c2);
        c3 = lasx_xvfmadd_s(a3, lasx::load_f32x8(b.as_ptr().add(i + 24)), c3);
        i += 32;
    }
    if i + 16 <= n {
        let (a0, a1) = lasx::load_f16x16_as_f32x8x2(a.as_ptr().add(i));
        c0 = lasx_xvfmadd_s(a0, lasx::load_f32x8(b.as_ptr().add(i)), c0);
        c1 = lasx_xvfmadd_s(a1, lasx::load_f32x8(b.as_ptr().add(i + 8)), c1);
        i += 16;
    }
    // 归约：链内先按固定次序合并（(c0+c1) + (c2+c3)），再两两归约 8 个 lane
    let s = lasx_xvfadd_s(lasx_xvfadd_s(c0, c1), lasx_xvfadd_s(c2, c3));
    let mut t = [0f32; 8];
    lasx::store_f32x8(t.as_mut_ptr(), s);
    // 尾部：延续同一 lane（`i` 是 16 的倍数 ⇒ lane = j % 8）
    for j in i..n {
        t[j % 8] += f16_to_f32(a[j]) * b[j];
    }
    ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]))
}

/// f16 权重矩阵（`m × k` 行主序）× f32 向量：`y[r] = dot_f16(a[r*k..(r+1)*k], x)`。
///
/// 每行独立 ⇒ 逐行调用 [`dot_f16`]，因此 `y[r]` 与单独调 `dot_f16` **逐位相同**。
///
/// # Safety
/// `a.len() == m * k`、`x.len() == k`、`y.len() == m`（调用方保证；`api`/`_checked` 会先校验）。
pub(crate) unsafe fn gemv_f16(a: &[u16], x: &[f32], m: usize, k: usize, y: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(y.len(), m);
    for (r, out) in y.iter_mut().enumerate() {
        *out = dot_f16(std::slice::from_raw_parts(a.as_ptr().add(r * k), k), x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 标量模拟：**独立**写出同一串 op（32 元素块分 4 链、16 元素块用前 2 链、
    /// `(c0+c1)+(c2+c3)` 合并、尾部延续 lane `j%8`、固定次序两两归约）。
    fn emulate(a: &[u16], b: &[f32]) -> f32 {
        let n = a.len();
        let mut c = [[0f32; 8]; 4];
        let mut i = 0;
        while i + 32 <= n {
            for (chain, cs) in c.iter_mut().enumerate() {
                for (l, acc) in cs.iter_mut().enumerate() {
                    let j = i + chain * 8 + l;
                    *acc += f16_to_f32(a[j]) * b[j];
                }
            }
            i += 32;
        }
        if i + 16 <= n {
            for (chain, cs) in c.iter_mut().enumerate().take(2) {
                for (l, acc) in cs.iter_mut().enumerate() {
                    let j = i + chain * 8 + l;
                    *acc += f16_to_f32(a[j]) * b[j];
                }
            }
            i += 16;
        }
        let mut t = [0f32; 8];
        for (l, tl) in t.iter_mut().enumerate() {
            *tl = (c[0][l] + c[1][l]) + (c[2][l] + c[3][l]);
        }
        for j in i..n {
            t[j % 8] += f16_to_f32(a[j]) * b[j];
        }
        ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]))
    }

    fn run(a: &[u16], b: &[f32]) -> f32 {
        // SAFETY: 调用点保证等长。
        unsafe { dot_f16(a, b) }
    }

    /// **穷举全部 65536 个 f16 位型**：硬件转换 == 手写 `f16_to_f32`。
    ///
    /// 这是最关键的一条测试：它把"转换精确"从"抽几个值看看"变成**全覆盖**，
    /// 顺带钉死次正规（f16 次正规在 f32 里是正规数，最容易写错的那条分支）。
    #[test]
    fn test_conversion_exhaustive_all_bit_patterns() {
        // 一次测 16 个：用内核同一条原语（`load_f16x16_as_f32x8x2`）
        let mut buf = [0u16; 16];
        let mut checked = 0usize;
        let mut nan_patterns = 0usize;
        for base in (0..=0xffffu32).step_by(16) {
            for (l, slot) in buf.iter_mut().enumerate() {
                *slot = (base + l as u32) as u16;
            }
            // SAFETY: `buf` 有 16 个元素。
            let (lo, hi) = unsafe { lasx::load_f16x16_as_f32x8x2(buf.as_ptr()) };
            let mut got = [0f32; 16];
            // SAFETY: 各 8 个元素。
            unsafe {
                lasx::store_f32x8(got.as_mut_ptr(), lo);
                lasx::store_f32x8(got.as_mut_ptr().add(8), hi);
            }
            for (l, &bits) in buf.iter().enumerate() {
                let want = f16_to_f32(bits);
                if want.is_nan() {
                    // NaN 的 payload 由硬件决定：只要求两边都是 NaN
                    assert!(
                        got[l].is_nan(),
                        "bits={bits:#06x} 硬件应给 NaN，得到 {}",
                        got[l]
                    );
                    nan_patterns += 1;
                } else {
                    assert_eq!(
                        got[l].to_bits(),
                        want.to_bits(),
                        "bits={bits:#06x} 硬件={} 手写={}",
                        got[l],
                        want
                    );
                }
                checked += 1;
            }
        }
        assert_eq!(checked, 65536);
        assert_eq!(nan_patterns, 2046, "f16 的 NaN 位型数（exp=31 且 man≠0）");
    }

    /// 逐位一致：向量路径 vs 标量模拟，覆盖 16 的边界与多种长度。
    ///
    /// 随机用的是**任意 f16 位型**（含次正规与 `±inf`），所以会撞上 NaN：这时只要求两边都是
    /// NaN——`NaN` 的 payload 由硬件转换决定，**不在**逐位契约内（与 `docs/ops.md` §2.11
    /// 的措辞一致）。
    #[test]
    fn test_matches_scalar_emulation_bit_for_bit() {
        let mut seed = 12345u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed >> 8
        };
        let mut nan_cases = 0usize;
        for n in [0usize, 1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 1000] {
            // 随机 f16 位型（含次正规/正规/±0/inf/NaN），b 用普通 f32
            let a: Vec<u16> = (0..n).map(|_| next() as u16).collect();
            let b: Vec<f32> = (0..n).map(|i| (i as f32).mul_add(0.017, -1.3)).collect();
            let got = run(&a, &b);
            let want = emulate(&a, &b);
            if got.is_nan() || want.is_nan() {
                assert!(got.is_nan() && want.is_nan(), "n={n} 一边 NaN 一边不是");
                nan_cases += 1;
            } else {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "n={n}（got={got} want={want}）"
                );
            }
        }
        // 随机数据里必须真的出现过 NaN，否则上面那条分支等于没测
        assert!(nan_cases > 0, "随机数据里没出现 NaN，分支没被覆盖");
        // 全是"好看"的 f16（能精确表示），让数值断言也有意义
        let a: Vec<u16> = (0..100).map(|i| half_of((i as f32) * 0.25 - 5.0)).collect();
        let b: Vec<f32> = (0..100).map(|i| (i as f32).mul_add(0.11, 0.5)).collect();
        assert_eq!(run(&a, &b).to_bits(), emulate(&a, &b).to_bits());
    }

    /// 测试用：f32 → f16 位型（只用于构造**精确可表示**的小整数/分数；不是通用转换）。
    fn half_of(x: f32) -> u16 {
        // 简单实现：只支持 |x| < 65504 的、能用 f16 表示的常见值（测试里用的都是 k/4 与整数）
        let bits = x.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32;
        let man = bits & 0x007f_ffff;
        if exp == 0 && man == 0 {
            return sign;
        }
        let e16 = exp - 127 + 15;
        assert!(
            (1..=30).contains(&e16) && man & 0x1fff == 0,
            "half_of 只支持精确值：{x}"
        );
        sign | ((e16 as u16) << 10) | ((man >> 13) as u16)
    }

    /// 数值：与 f64 参考的相对误差（应当远小于 f16 的量化误差 4.9e-4）。
    #[test]
    fn test_accuracy_vs_f64_reference() {
        // 取一组"典型"权重：正态样子的 f16（用高精度值转成 f16 位型）
        let n = 4096usize;
        let mut a = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        let mut seed = 999u32;
        for i in 0..n {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let r = ((seed >> 9) as f32 / 4194304.0) - 1.0; // [-1,1)
            a.push(f32_to_f16_bits(r * 0.05));
            b.push((i as f32).mul_add(0.001, 0.1) * (1.0 - r));
        }
        let got = run(&a, &b);
        let want: f64 = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| f16_to_f32(x) as f64 * y as f64)
            .sum();
        let rel = ((got as f64 - want) / want).abs();
        assert!(rel < 1e-6, "相对误差 {rel}（got={got} want={want}）");
    }

    /// 测试用：把 f32 转成最近的 f16 位型（含次正规与溢出到 inf 的简单处理）。
    fn f32_to_f16_bits(x: f32) -> u16 {
        let bits = x.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32;
        let man = bits & 0x007f_ffff;
        if exp == 0xff {
            return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
        }
        let e = exp - 127 + 15;
        if e >= 0x1f {
            return sign | 0x7c00; // 溢出到 inf
        }
        if e <= 0 {
            // 次正规或 0：右移 (1 - e) 位并舍入
            if e < -10 {
                return sign;
            }
            let m = (man | 0x0080_0000) >> (1 - e);
            let half = 1u32 << (13 - e + 1 - 1);
            let rounded = (m + (half >> 1)) >> (13 - e);
            return sign | rounded as u16;
        }
        let rounded = (man + 0x1000) >> 13; // 就近舍入（不处理 ties-to-even，够测试用）
        sign | ((e as u16) << 10) | rounded as u16
    }

    /// `gemv_f16`：逐行与 `dot_f16` 逐位相同；形状退化情形不崩。
    #[test]
    fn test_gemv_matches_row_dots() {
        let (m, k) = (7usize, 33usize);
        let a: Vec<u16> = (0..m * k).map(|i| ((i * 37) as u16) & 0x7bff).collect();
        let x: Vec<f32> = (0..k).map(|i| (i as f32).mul_add(0.05, -0.7)).collect();
        let mut y = vec![0f32; m];
        // SAFETY: 长度自洽（m*k、k、m）。
        unsafe { gemv_f16(&a, &x, m, k, &mut y) };
        for r in 0..m {
            let row = &a[r * k..(r + 1) * k];
            // SAFETY: 等长。
            let want = unsafe { dot_f16(row, &x) };
            assert_eq!(y[r].to_bits(), want.to_bits(), "r={r}");
        }
        // k = 0 ⇒ 全 0；m = 0 ⇒ 什么也不写
        let mut y0 = vec![7f32; 3];
        // SAFETY: k=0 时 a 为空、x 为空。
        unsafe { gemv_f16(&[], &[], 3, 0, &mut y0) };
        assert!(y0.iter().all(|&v| v == 0.0));
        // SAFETY: m=0。
        unsafe { gemv_f16(&[], &[], 0, 4, &mut []) };
    }

    /// 退化：`n = 0` ⇒ `0.0`；`inf × 0` 是 NaN（定义域问题，写进契约）。
    #[test]
    fn test_degenerate_inputs() {
        assert_eq!(run(&[], &[]), 0.0);
        assert_eq!(run(&[0x3c00], &[0.0]), 0.0); // f16 1.0 × 0 = 0
        let inf = f32::INFINITY;
        assert_eq!(run(&[0x3c00], &[inf]), inf);
        // 乘积本身是 NaN 的情形：f16 的 +inf（0x7c00）× f32 的 0.0
        assert!(run(&[0x7c00], &[0.0]).is_nan());
        // 而 inf 与 0 落在**不同 lane**时只是 inf（没有 inf×0 这个乘法）
        assert!(run(&[0x3c00, 0x3c00], &[inf, 0.0]).is_infinite());
    }
}
