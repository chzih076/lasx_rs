//! **P4-b：谱检测的两条融合内核**（LASX 加速；**逐位类**）。
//!
//! # 语义（签名固定）
//! 1. [`energy_dc`]：`out[c] = Σ_r (x[r·cols+c] − mean[c])²`（**单遍、无中间缓冲**）。
//! 2. [`psd_pair`]：`out[j·n+c] = ((Σ_t a[j·k+t]·x[t·n+c])² + (Σ_t b[j·k+t]·x[t·n+c])²) · inv[c]`
//!    （**成对 epilogue**：两个 matmul 输出直接合成 psd，省掉 `re2`/`im2`/`psd` 三块中间缓冲）。
//!
//! # 等价类（**如实**）
//! 两个内核都是**纯标量、按参照同序累加**（无 FMA 收缩、无重结合）⇒ **与便携参照逐位一致**
//! （单测用 `to_bits()` 钉住）。**LASX FMA 版本轮未落地**：`lasx_xvfsub_s` 在 `arch/lasx.rs` 里
//! **核对不上**（只有 `load/store/splat/zero/max` 等封装），按"别凭记忆写 intrinsic"的规矩停下来，
//! 先用标量写通 + 单测 + 界 ⇒ **FMA 版与它的容差界留待下一轮（拿到正确封装名后）**。
//!
//! # 门控
//! **整个模块挂 `dsl-backend` feature**（与 `dsl_map` 同因：默认档无人使用 ⇒ `dead_code` 会被
//! `clippy -D warnings` 判死，AGENTS.md #49）。**不要用 `#[allow(dead_code)]` 掩盖。**

/// **内核 1**：`out[c] = Σ_r (x[r·cols+c] − mean[c])²`。
///
/// 累加序 = **按 `r` 顺序**（与便携参照一致）⇒ **逐位**。
///
/// # Panics
/// `x.len() != rows*cols`、`mean.len() != cols`、`out.len() != cols` ⇒ panic（前置条件）。
pub fn energy_dc(x: &[f32], rows: usize, cols: usize, mean: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), rows * cols, "energy_dc：x 长度应为 rows*cols");
    assert_eq!(mean.len(), cols, "energy_dc：mean 长度应为 cols");
    assert_eq!(out.len(), cols, "energy_dc：out 长度应为 cols");
    out.iter_mut().enumerate().for_each(|(c, slot)| {
        *slot = (0..rows).fold(0.0f32, |acc, r| {
            let d = x[r * cols + c] - mean[c];
            acc + d * d
        });
    });
}

/// **内核 2（成对 epilogue）**：
/// `out[j·n+c] = ((Σ_t a[j·k+t]·x[t·n+c])² + (Σ_t b[j·k+t]·x[t·n+c])²) · inv[c]`。
///
/// 运算顺序：先两个 `Σ`（按 `t` 顺序），再平方、相加、乘 `inv[c]` ⇒ 与便携参照**同序** ⇒ **逐位**。
///
/// # Panics
/// `a.len() != j*k`、`b.len() != j*k`、`x.len() != k*n`、`inv.len() != n`、`out.len() != j*n` ⇒ panic。
pub fn psd_pair(
    a: &[f32],
    b: &[f32],
    x: &[f32],
    j: usize,
    k: usize,
    n: usize,
    inv: &[f32],
    out: &mut [f32],
) {
    assert_eq!(a.len(), j * k, "psd_pair：a 长度应为 j*k");
    assert_eq!(b.len(), j * k, "psd_pair：b 长度应为 j*k");
    assert_eq!(x.len(), k * n, "psd_pair：x 长度应为 k*n");
    assert_eq!(inv.len(), n, "psd_pair：inv 长度应为 n");
    assert_eq!(out.len(), j * n, "psd_pair：out 长度应为 j*n");
    out.iter_mut().enumerate().for_each(|(idx, slot)| {
        let jj = idx / n;
        let c = idx % n;
        let sr = (0..k).fold(0.0f32, |acc, t| acc + a[jj * k + t] * x[t * n + c]);
        let si = (0..k).fold(0.0f32, |acc, t| acc + b[jj * k + t] * x[t * n + c]);
        let re = sr * sr;
        let im = si * si;
        *slot = (re + im) * inv[c];
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(len: usize, seed: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i % 23) as f32) * 0.0625 - 1.0 + seed)
            .collect()
    }

    /// **与便携参照逐位**（同序 ⇒ `to_bits` 相等）。
    #[test]
    fn kernels_bitwise_same_as_portable_reference() {
        // 内核 1
        let (rows, cols) = (16usize, 24usize);
        let x = data(rows * cols, 0.0);
        let mean = data(cols, 0.5);
        let mut got = vec![0.0f32; cols];
        energy_dc(&x, rows, cols, &mean, &mut got);
        let want: Vec<f32> = (0..cols)
            .map(|c| {
                (0..rows).fold(0.0f32, |acc, r| {
                    let d = x[r * cols + c] - mean[c];
                    acc + d * d
                })
            })
            .collect();
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        // 内核 2
        let (j, k, n) = (4usize, 6usize, 10usize);
        let (aa, bb, xx, inv) = (
            data(j * k, 0.0),
            data(j * k, 0.125),
            data(k * n, 0.25),
            data(n, 0.0625),
        );
        let mut got2 = vec![0.0f32; j * n];
        psd_pair(&aa, &bb, &xx, j, k, n, &inv, &mut got2);
        let want2: Vec<f32> = (0..j * n)
            .map(|idx| {
                let jj = idx / n;
                let c = idx % n;
                let sr = (0..k).fold(0.0f32, |acc, t| acc + aa[jj * k + t] * xx[t * n + c]);
                let si = (0..k).fold(0.0f32, |acc, t| acc + bb[jj * k + t] * xx[t * n + c]);
                ((sr * sr) + (si * si)) * inv[c]
            })
            .collect();
        assert_eq!(
            got2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want2.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    /// 前置条件：长度不符 ⇒ panic。
    #[test]
    #[should_panic(expected = "energy_dc：x 长度应为 rows*cols")]
    fn energy_dc_panics_on_bad_len() {
        let mut out = vec![0.0f32; 3];
        energy_dc(&[0.0f32; 5], 2, 3, &[0.0; 3], &mut out);
    }
}
