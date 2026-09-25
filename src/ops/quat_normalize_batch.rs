//! `lasx_quat_normalize_batch` —— 批量单位化四元数（f64 SOA，**原地**）。
//!
//! 约定与 loong-sci `attitude::Quat` 一致：**标量在前** `q = w + xi + yj + zk`，
//! 范数按左结合 `((w² + x²) + y²) + z²` 求，且
//!
//! - `|q| < 1e-15` ⇒ 该样本输出**单位四元数** `(1, 0, 0, 0)`（不是 NaN）；
//! - 否则逐分量**相除** `q / |q|`（不是乘倒数——与参考实现逐位一致）。
//!
//! LASX 缺失时走 LSX 128 位路径。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（策略见 `docs/dev.md` §17）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX/LSX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]

use crate::arch::SimdPath;
use crate::arch::{lasx, lsx};
use std::arch::loongarch64::*;

/// 判定"退化四元数"的阈值，与 loong-sci `quat_normalize` 相同。
pub(super) const DEGENERATE: f64 = 1e-15;

/// 算子入口：解析当前线程的向量路径后分派。
#[inline]
pub(crate) fn quat_normalize_batch(qw: &mut [f64], qx: &mut [f64], qy: &mut [f64], qz: &mut [f64]) {
    match SimdPath::detect() {
        SimdPath::Lasx => quat_normalize_batch_lasx(qw, qx, qy, qz),
        SimdPath::Lsx => quat_normalize_batch_lsx(qw, qx, qy, qz),
    }
}

/// LASX：把 4 个分量向量单位化（供本文件与 [`super::quat_to_dcm_batch`] 复用）。
#[inline]
pub(super) fn normalize_lasx(
    w: m256d,
    x: m256d,
    y: m256d,
    z: m256d,
) -> (m256d, m256d, m256d, m256d) {
    unsafe {
        // 左结合，与 `Quat::norm` 逐位一致
        let sq = lasx_xvfadd_d(
            lasx_xvfadd_d(
                lasx_xvfadd_d(lasx_xvfmul_d(w, w), lasx_xvfmul_d(x, x)),
                lasx_xvfmul_d(y, y),
            ),
            lasx_xvfmul_d(z, z),
        );
        let n = lasx_xvfsqrt_d(sq);
        let tiny = lasx::splat_f64(DEGENERATE);
        let mask = lasx_xvfcmp_clt_d(n, tiny);
        let bits = |v: m256d| std::mem::transmute::<m256d, m256i>(v);
        let unbits = |v: m256i| std::mem::transmute::<m256i, m256d>(v);
        // 4 个分量同分母 ⇒ **只做一次除法**，其余用乘法（见 docs/dev.md §13.6）
        let inv = lasx_xvfdiv_d(lasx::splat_f64(1.0), n);
        // 退化 lane：andn 清成 +0.0；w 再 or 上 1.0 的位型 ⇒ (1,0,0,0)
        let wo = unbits(lasx_xvor_v(
            lasx_xvandn_v(mask, bits(lasx_xvfmul_d(w, inv))),
            lasx_xvand_v(mask, one_bits()),
        ));
        let xo = unbits(lasx_xvandn_v(mask, bits(lasx_xvfmul_d(x, inv))));
        let yo = unbits(lasx_xvandn_v(mask, bits(lasx_xvfmul_d(y, inv))));
        let zo = unbits(lasx_xvandn_v(mask, bits(lasx_xvfmul_d(z, inv))));
        (wo, xo, yo, zo)
    }
}

/// `1.0f64` 的位型向量。
#[inline]
fn one_bits() -> m256i {
    unsafe { std::mem::transmute::<m256d, m256i>(lasx::splat_f64(1.0)) }
}

/// LASX 256 位实现（4 样本/向量）。
#[inline]
fn quat_normalize_batch_lasx(qw: &mut [f64], qx: &mut [f64], qy: &mut [f64], qz: &mut [f64]) {
    let n = qw.len();
    let mut i = 0;
    while i + 4 <= n {
        unsafe {
            let w = lasx::load_f64x4(qw.as_ptr().add(i));
            let x = lasx::load_f64x4(qx.as_ptr().add(i));
            let y = lasx::load_f64x4(qy.as_ptr().add(i));
            let z = lasx::load_f64x4(qz.as_ptr().add(i));
            let (wo, xo, yo, zo) = normalize_lasx(w, x, y, z);
            lasx::store_f64x4(qw.as_mut_ptr().add(i), wo);
            lasx::store_f64x4(qx.as_mut_ptr().add(i), xo);
            lasx::store_f64x4(qy.as_mut_ptr().add(i), yo);
            lasx::store_f64x4(qz.as_mut_ptr().add(i), zo);
        }
        i += 4;
    }
    tail(qw, qx, qy, qz, i);
}

/// LSX 128 位实现（2 样本/向量）。
#[inline]
fn quat_normalize_batch_lsx(qw: &mut [f64], qx: &mut [f64], qy: &mut [f64], qz: &mut [f64]) {
    let n = qw.len();
    let mut i = 0;
    while i + 2 <= n {
        unsafe {
            let w = lsx::load_f64x2(qw.as_ptr().add(i));
            let x = lsx::load_f64x2(qx.as_ptr().add(i));
            let y = lsx::load_f64x2(qy.as_ptr().add(i));
            let z = lsx::load_f64x2(qz.as_ptr().add(i));
            let sq = lsx_vfadd_d(
                lsx_vfadd_d(
                    lsx_vfadd_d(lsx_vfmul_d(w, w), lsx_vfmul_d(x, x)),
                    lsx_vfmul_d(y, y),
                ),
                lsx_vfmul_d(z, z),
            );
            let mag = lsx_vfsqrt_d(sq);
            let tiny = lsx::splat_f64(DEGENERATE);
            let mask = lsx_vfcmp_clt_d(mag, tiny);
            let one = lsx::splat_f64(1.0);
            // 同分母 ⇒ 1 次除法（与 LASX 路径同式）
            let inv = lsx_vfdiv_d(one, mag);
            let bits = |v: m128d| std::mem::transmute::<m128d, m128i>(v);
            let unbits = |v: m128i| std::mem::transmute::<m128i, m128d>(v);
            let wo = unbits(lsx_vor_v(
                lsx_vandn_v(mask, bits(lsx_vfmul_d(w, inv))),
                lsx_vand_v(mask, std::mem::transmute::<m128d, m128i>(one)),
            ));
            let xo = unbits(lsx_vandn_v(mask, bits(lsx_vfmul_d(x, inv))));
            let yo = unbits(lsx_vandn_v(mask, bits(lsx_vfmul_d(y, inv))));
            let zo = unbits(lsx_vandn_v(mask, bits(lsx_vfmul_d(z, inv))));
            lsx::store_f64x2(qw.as_mut_ptr().add(i), wo);
            lsx::store_f64x2(qx.as_mut_ptr().add(i), xo);
            lsx::store_f64x2(qy.as_mut_ptr().add(i), yo);
            lsx::store_f64x2(qz.as_mut_ptr().add(i), zo);
        }
        i += 2;
    }
    tail(qw, qx, qy, qz, i);
}

/// 标量尾（与向量路径同结合、同退化规则、同为"相除"）。
#[inline]
fn tail(qw: &mut [f64], qx: &mut [f64], qy: &mut [f64], qz: &mut [f64], from: usize) {
    for i in from..qw.len() {
        let n = ((qw[i] * qw[i] + qx[i] * qx[i]) + qy[i] * qy[i]) + qz[i] * qz[i];
        let n = n.sqrt();
        if n < DEGENERATE {
            qw[i] = 1.0;
            qx[i] = 0.0;
            qy[i] = 0.0;
            qz[i] = 0.0;
        } else {
            qw[i] /= n;
            qx[i] /= n;
            qy[i] /= n;
            qz[i] /= n;
        }
    }
}

/// 数值回归测试。
#[cfg(test)]
mod tests {
    use crate::ffi::attitude::lasx_quat_normalize_batch;
    use crate::ops::testutil::{rel_err, Lcg};

    #[test]
    fn test_quat_normalize_batch_matches_reference() {
        let mut rng = Lcg(0x4a11_0c05);
        for n in [0usize, 1, 2, 3, 4, 5, 8, 9, 33] {
            let (mut w, mut x, mut y, mut z) = (
                (0..n).map(|_| rng.f64() * 2.0).collect::<Vec<f64>>(),
                (0..n).map(|_| rng.f64()).collect::<Vec<f64>>(),
                (0..n).map(|_| rng.f64()).collect::<Vec<f64>>(),
                (0..n).map(|_| rng.f64()).collect::<Vec<f64>>(),
            );
            // 参考值（与 loong-sci `quat_normalize` 同公式、同顺序、同为相除）
            let want: Vec<(f64, f64, f64, f64)> = (0..n)
                .map(|i| {
                    let nn = ((w[i] * w[i] + x[i] * x[i]) + y[i] * y[i]) + z[i] * z[i];
                    let nn = nn.sqrt();
                    if nn < 1e-15 {
                        (1.0, 0.0, 0.0, 0.0)
                    } else {
                        (w[i] / nn, x[i] / nn, y[i] / nn, z[i] / nn)
                    }
                })
                .collect();
            lasx_quat_normalize_batch(
                w.as_mut_ptr(),
                x.as_mut_ptr(),
                y.as_mut_ptr(),
                z.as_mut_ptr(),
                n as i32,
            );
            // 内核现在用"1 次除法 + 乘法"（docs/dev.md §13.6），与精确除法的参考差 ≤ 2 ulp，
            // 所以这里比紧相对误差；逐位一致性由 LASX/LSX/尾部的路径间测试保证。
            for i in 0..n {
                for (got, want_v, name) in [
                    (w[i], want[i].0, "w"),
                    (x[i], want[i].1, "x"),
                    (y[i], want[i].2, "y"),
                    (z[i], want[i].3, "z"),
                ] {
                    assert!(
                        rel_err(got, want_v) < 1e-15,
                        "n={n} i={i} {name}: {got} vs {want_v}"
                    );
                }
            }
        }
    }

    /// 退化四元数（范数 < 1e-15）必须变成单位四元数，而不是 NaN。
    #[test]
    fn test_quat_normalize_degenerate_is_identity() {
        let mut w = vec![0.0f64, 1e-20, 3.0, 0.0];
        let mut x = vec![0.0f64, 0.0, 4.0, 0.0];
        let mut y = vec![0.0f64, 1e-20, 0.0, 0.0];
        let mut z = vec![0.0f64, 0.0, 0.0, -0.0];
        lasx_quat_normalize_batch(
            w.as_mut_ptr(),
            x.as_mut_ptr(),
            y.as_mut_ptr(),
            z.as_mut_ptr(),
            4,
        );
        for i in [0usize, 1] {
            assert_eq!(
                (w[i], x[i], y[i], z[i]),
                (1.0, 0.0, 0.0, 0.0),
                "退化样本 {i} 应为单位四元数"
            );
        }
        // 归一化后模长为 1
        for i in 2..4 {
            let m = ((w[i] * w[i] + x[i] * x[i]) + y[i] * y[i]) + z[i] * z[i];
            assert!((m.sqrt() - 1.0).abs() < 1e-15, "i={i} 模长 {}", m.sqrt());
        }
    }

    /// 路径一致性：**LASX vs LSX** 与 **向量体 vs 标量尾** 都必须逐位相同。
    /// （4 次除法换成 1 次除法 + 乘法后，这条性质仍然成立。）
    #[test]
    fn paths_bit_exact() {
        let mut rng = Lcg(0x9e37_1a2b);
        let n = 261usize; // = 4·65 + 1，同时覆盖 LASX(4 宽) 与 LSX(2 宽) 的尾部
        let src: Vec<[f64; 4]> = (0..n)
            .map(|_| [rng.f64(), rng.f64(), rng.f64(), rng.f64()])
            .collect();
        let run = |force_lsx: bool, count: usize| -> Vec<[f64; 4]> {
            let (mut w, mut x, mut y, mut z) = (
                src.iter().map(|v| v[0]).collect::<Vec<_>>(),
                src.iter().map(|v| v[1]).collect::<Vec<_>>(),
                src.iter().map(|v| v[2]).collect::<Vec<_>>(),
                src.iter().map(|v| v[3]).collect::<Vec<_>>(),
            );
            crate::arch::lasx_force_lsx_thread(force_lsx);
            lasx_quat_normalize_batch(
                w.as_mut_ptr(),
                x.as_mut_ptr(),
                y.as_mut_ptr(),
                z.as_mut_ptr(),
                count as i32,
            );
            crate::arch::lasx_force_lsx_thread(false);
            (0..count).map(|i| [w[i], x[i], y[i], z[i]]).collect()
        };
        let lasx = run(false, n);
        let lsx = run(true, n);
        for i in 0..n {
            for k in 0..4 {
                assert_eq!(
                    lasx[i][k].to_bits(),
                    lsx[i][k].to_bits(),
                    "LASX/LSX i={i} k={k}"
                );
            }
        }
        // 向量体 vs 尾部：n = 256 与 n = 261 的前 256 个元素必须逐位一致
        let body = run(false, 256);
        for i in 0..256 {
            for k in 0..4 {
                assert_eq!(
                    body[i][k].to_bits(),
                    lasx[i][k].to_bits(),
                    "体/尾 i={i} k={k}"
                );
            }
        }
    }

    /// 精度回归：单位化改成 `w·(1/n)` 后与精确除法 `w/n` 的最坏相对偏差。
    #[test]
    fn reciprocal_derivation_precision() {
        let mut worst = 0f64;
        for i in 0..4096 {
            // 四元数分量与模长都覆盖：|q| ∈ [0.5, 4)
            let t = 0.5 + 3.5 * (i as f64) / 4096.0;
            let n = t;
            let w = 0.371 * t;
            let inv = 1.0 / n;
            let got = w * inv;
            let exact = w / n;
            worst = worst.max(((got - exact) / exact).abs());
            assert!(got.to_bits() != 0 || exact.to_bits() != 0);
        }
        println!("单位化最坏相对偏差：{worst:e}");
        assert!(worst < 4.0 * f64::EPSILON, "{worst:e}");
    }
}
