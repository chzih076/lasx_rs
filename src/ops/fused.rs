//! **P4-b：谱检测的两条融合内核**（LASX 加速；**逐位类**）。
//!
//! # 语义（签名固定）
//! 1. [`energy_dc`]：`out[c] = Σ_r (x[r·cols+c] − mean[c])²`（**单遍、无中间缓冲**）。
//! 2. [`psd_pair`]：`out[j·n+c] = ((Σ_t a[j·k+t]·x[t·n+c])² + (Σ_t b[j·k+t]·x[t·n+c])²) · inv[c]`
//!    （**成对 epilogue**：两个 matmul 输出直接合成 psd，省掉 `re2`/`im2`/`psd` 三块中间缓冲）。
//!
//! # 等价类（**逐位**，向量化后仍成立）
//! 向量化只把**不同列**放进不同 lane，**同列内的运算序列与顺序完全不变**（`sub`→`mul`→`add`，
//! **不用 FMA 收缩**、不重结合）⇒ 与便携参照 `to_bits()` 相等（单测钉住）。
//! **注意**：若将来改用 `xvfmadd` 融合乘加，会**静默离开逐位类** ⇒ 必须改声明为容差类并回填实测界。
//!
//! # 门控
//! **整个模块挂 `dsl-backend` feature**（与 `dsl_map` 同因：默认档无人使用 ⇒ `dead_code` 会被
//! `clippy -D warnings` 判死，AGENTS.md #49）。**不要用 `#[allow(dead_code)]` 掩盖。**
use crate::arch::lasx;
use std::arch::loongarch64::*;

/// **内核 1（向量化）**：`out[c] = Σ_r (x[r·cols+c] − mean[c])²`。
///
/// **每列一条 lane**（与 `dsl_map::reduce_sum_cols` 同款：`c` 越界检查后按 `r` 顺序累加），
/// 尾部列走标量同形实现。**无 FMA ⇒ 逐位**。
///
/// # Panics
/// `x.len() != rows*cols`、`mean.len() != cols`、`out.len() != cols` ⇒ panic（前置条件）。
pub fn energy_dc(x: &[f32], rows: usize, cols: usize, mean: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), rows * cols, "energy_dc：x 长度应为 rows*cols");
    assert_eq!(mean.len(), cols, "energy_dc：mean 长度应为 cols");
    assert_eq!(out.len(), cols, "energy_dc：out 长度应为 cols");
    let mut c = 0usize;
    while c + 8 <= cols {
        // SAFETY: `c + 8 <= cols == mean.len()`，`mean` 内偏移合法。
        let vmean = unsafe { lasx::load_f32x8(mean.as_ptr().add(c)) };
        let mut acc = lasx::zero_f32x8();
        for r in 0..rows {
            let off = r * cols + c;
            // SAFETY: `c + 8 <= cols` 且 `r < rows` ⇒ `off + 8 <= rows*cols = x.len()`。
            unsafe {
                let vx = lasx::load_f32x8(x.as_ptr().add(off));
                let d = lasx_xvfsub_s(vx, vmean);
                let dd = lasx_xvfmul_s(d, d);
                acc = lasx_xvfadd_s(acc, dd);
            }
        }
        // SAFETY: `c + 8 <= cols == out.len()`。
        unsafe { lasx::store_f32x8(out.as_mut_ptr().add(c), acc) };
        c += 8;
    }
    // 尾部列：与参照同形（f32 单累加器、按行序、无 FMA）。
    for cc in c..cols {
        out[cc] = (0..rows).fold(0.0f32, |acc, r| {
            let d = x[r * cols + cc] - mean[cc];
            acc + d * d
        });
    }
}

/// **内核 2（成对 epilogue，向量化）**：
/// `out[j·n+c] = ((Σ_t a[j·k+t]·x[t·n+c])² + (Σ_t b[j·k+t]·x[t·n+c])²) · inv[c]`。
///
/// **沿 `c` 分 lane**（8 路）；每个 lane 内 `t` 顺序累加、先平方再相加再乘 `inv[c]` ⇒ **逐位**。
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
    for jj in 0..j {
        let mut c = 0usize;
        while c + 8 <= n {
            let mut acc_r = lasx::zero_f32x8();
            let mut acc_i = lasx::zero_f32x8();
            for t in 0..k {
                let off = t * n + c;
                // SAFETY: `c + 8 <= n` 且 `t < k` ⇒ `off + 8 <= k*n = x.len()`；`a/b` 的标量索引在界内。
                unsafe {
                    let vx = lasx::load_f32x8(x.as_ptr().add(off));
                    let va = lasx::splat_f32(a[jj * k + t]);
                    let vb = lasx::splat_f32(b[jj * k + t]);
                    acc_r = lasx_xvfadd_s(acc_r, lasx_xvfmul_s(va, vx));
                    acc_i = lasx_xvfadd_s(acc_i, lasx_xvfmul_s(vb, vx));
                }
            }
            // SAFETY: `c + 8 <= n == inv.len()`，`out` 内 `jj*n + c + 8 <= j*n`。
            unsafe {
                let sr = lasx_xvfmul_s(acc_r, acc_r);
                let si = lasx_xvfmul_s(acc_i, acc_i);
                let s = lasx_xvfadd_s(sr, si);
                let vinv = lasx::load_f32x8(inv.as_ptr().add(c));
                lasx::store_f32x8(out.as_mut_ptr().add(jj * n + c), lasx_xvfmul_s(s, vinv));
            }
            c += 8;
        }
        // 尾部列：与参照同形。
        for cc in c..n {
            let sr = (0..k).fold(0.0f32, |acc, t| acc + a[jj * k + t] * x[t * n + cc]);
            let si = (0..k).fold(0.0f32, |acc, t| acc + b[jj * k + t] * x[t * n + cc]);
            out[jj * n + cc] = ((sr * sr) + (si * si)) * inv[cc];
        }
    }
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
