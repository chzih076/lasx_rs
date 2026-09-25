//! `lasx_quat_to_dcm_batch` —— 批量四元数 → 3×3 方向余弦阵（f64 SOA，行主序输出）。
//!
//! 约定与 loong-sci `attitude::quat_rotate` 里的 R 一致（**体坐标 → 惯性坐标**，
//! 标量在前的单位四元数）：
//!
//! ```text
//! R = [ 1−2(y²+z²)   2(xy−wz)    2(xz+wy) ]
//!     [ 2(xy+wz)     1−2(x²+z²)  2(yz−wx) ]
//!     [ 2(xz−wy)     2(yz+wx)    1−2(x²+y²) ]
//! ```
//!
//! 输入四元数**先按 [`super::quat_normalize_batch`] 的规则单位化**（含退化 → 单位四元数），
//! 因此传未归一化的四元数也安全。输出 `m0..m8` 行主序，可直接喂给
//! [`super::mat3_mul_vec3_batch`]。
//!
//! LASX 缺失时走 LSX 128 位路径。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（策略见 `docs/dev.md` §17）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX/LSX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]

use crate::arch::SimdPath;
use crate::arch::{lasx, lsx};
use std::arch::loongarch64::*;

use super::quat_normalize_batch::normalize_lasx;

/// 算子入口：解析当前线程的向量路径后分派。
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn quat_to_dcm_batch(
    qw: &[f64],
    qx: &[f64],
    qy: &[f64],
    qz: &[f64],
    m: [&mut [f64]; 9],
) {
    match SimdPath::detect() {
        SimdPath::Lasx => quat_to_dcm_batch_lasx(qw, qx, qy, qz, m),
        SimdPath::Lsx => quat_to_dcm_batch_lsx(qw, qx, qy, qz, m),
    }
}

/// 由**已单位化**的 4 个分量向量构造 R 的 9 个元素（行主序）。
///
/// 供本文件与 [`super::quat_rotate_batch`] 共用，避免两处公式漂移。
#[inline]
pub(super) fn dcm_entries_lasx(w: m256d, x: m256d, y: m256d, z: m256d) -> [m256d; 9] {
    unsafe {
        let two = lasx::splat_f64(2.0);
        let one = lasx::splat_f64(1.0);
        let m2 = |v: m256d| lasx_xvfmul_d(two, v);
        [
            lasx_xvfsub_d(
                one,
                m2(lasx_xvfadd_d(lasx_xvfmul_d(y, y), lasx_xvfmul_d(z, z))),
            ),
            m2(lasx_xvfsub_d(lasx_xvfmul_d(x, y), lasx_xvfmul_d(w, z))),
            m2(lasx_xvfadd_d(lasx_xvfmul_d(x, z), lasx_xvfmul_d(w, y))),
            m2(lasx_xvfadd_d(lasx_xvfmul_d(x, y), lasx_xvfmul_d(w, z))),
            lasx_xvfsub_d(
                one,
                m2(lasx_xvfadd_d(lasx_xvfmul_d(x, x), lasx_xvfmul_d(z, z))),
            ),
            m2(lasx_xvfsub_d(lasx_xvfmul_d(y, z), lasx_xvfmul_d(w, x))),
            m2(lasx_xvfsub_d(lasx_xvfmul_d(x, z), lasx_xvfmul_d(w, y))),
            m2(lasx_xvfadd_d(lasx_xvfmul_d(y, z), lasx_xvfmul_d(w, x))),
            lasx_xvfsub_d(
                one,
                m2(lasx_xvfadd_d(lasx_xvfmul_d(x, x), lasx_xvfmul_d(y, y))),
            ),
        ]
    }
}

/// LSX 版 [`dcm_entries_lasx`]。
#[inline]
pub(super) fn dcm_entries_lsx(w: m128d, x: m128d, y: m128d, z: m128d) -> [m128d; 9] {
    unsafe {
        let two = lsx::splat_f64(2.0);
        let one = lsx::splat_f64(1.0);
        let m2 = |v: m128d| lsx_vfmul_d(two, v);
        [
            lsx_vfsub_d(one, m2(lsx_vfadd_d(lsx_vfmul_d(y, y), lsx_vfmul_d(z, z)))),
            m2(lsx_vfsub_d(lsx_vfmul_d(x, y), lsx_vfmul_d(w, z))),
            m2(lsx_vfadd_d(lsx_vfmul_d(x, z), lsx_vfmul_d(w, y))),
            m2(lsx_vfadd_d(lsx_vfmul_d(x, y), lsx_vfmul_d(w, z))),
            lsx_vfsub_d(one, m2(lsx_vfadd_d(lsx_vfmul_d(x, x), lsx_vfmul_d(z, z)))),
            m2(lsx_vfsub_d(lsx_vfmul_d(y, z), lsx_vfmul_d(w, x))),
            m2(lsx_vfsub_d(lsx_vfmul_d(x, z), lsx_vfmul_d(w, y))),
            m2(lsx_vfadd_d(lsx_vfmul_d(y, z), lsx_vfmul_d(w, x))),
            lsx_vfsub_d(one, m2(lsx_vfadd_d(lsx_vfmul_d(x, x), lsx_vfmul_d(y, y)))),
        ]
    }
}

/// LASX 256 位实现（4 样本/向量）。
#[inline]
fn quat_to_dcm_batch_lasx(qw: &[f64], qx: &[f64], qy: &[f64], qz: &[f64], m: [&mut [f64]; 9]) {
    let n = qw.len();
    let mut i = 0;
    while i + 4 <= n {
        unsafe {
            let (w, x, y, z) = normalize_lasx(
                lasx::load_f64x4(qw.as_ptr().add(i)),
                lasx::load_f64x4(qx.as_ptr().add(i)),
                lasx::load_f64x4(qy.as_ptr().add(i)),
                lasx::load_f64x4(qz.as_ptr().add(i)),
            );
            let r = dcm_entries_lasx(w, x, y, z);
            let mut k = 0;
            while k < 9 {
                lasx::store_f64x4(m[k].as_mut_ptr().add(i), r[k]);
                k += 1;
            }
        }
        i += 4;
    }
    tail(qw, qx, qy, qz, m, i);
}

/// LSX 128 位实现（2 样本/向量）。
#[inline]
fn quat_to_dcm_batch_lsx(qw: &[f64], qx: &[f64], qy: &[f64], qz: &[f64], m: [&mut [f64]; 9]) {
    let n = qw.len();
    let mut i = 0;
    while i + 2 <= n {
        unsafe {
            let (w, x, y, z) = normalize_lsx(
                lsx::load_f64x2(qw.as_ptr().add(i)),
                lsx::load_f64x2(qx.as_ptr().add(i)),
                lsx::load_f64x2(qy.as_ptr().add(i)),
                lsx::load_f64x2(qz.as_ptr().add(i)),
            );
            let r = dcm_entries_lsx(w, x, y, z);
            let mut k = 0;
            while k < 9 {
                lsx::store_f64x2(m[k].as_mut_ptr().add(i), r[k]);
                k += 1;
            }
        }
        i += 2;
    }
    tail(qw, qx, qy, qz, m, i);
}

/// 单位化的 LSX 版（与 [`normalize_lasx`] 同规则）。
#[inline]
fn normalize_lsx(w: m128d, x: m128d, y: m128d, z: m128d) -> (m128d, m128d, m128d, m128d) {
    unsafe {
        let sq = lsx_vfadd_d(
            lsx_vfadd_d(
                lsx_vfadd_d(lsx_vfmul_d(w, w), lsx_vfmul_d(x, x)),
                lsx_vfmul_d(y, y),
            ),
            lsx_vfmul_d(z, z),
        );
        let n = lsx_vfsqrt_d(sq);
        let mask = lsx_vfcmp_clt_d(n, lsx::splat_f64(super::quat_normalize_batch::DEGENERATE));
        let bits = |v: m128d| std::mem::transmute::<m128d, m128i>(v);
        let unbits = |v: m128i| std::mem::transmute::<m128i, m128d>(v);
        // 同分母 ⇒ 1 次除法（与 LASX/标量路径同式，见 docs/dev.md §13.6）
        let inv = lsx_vfdiv_d(lsx::splat_f64(1.0), n);
        let wo = unbits(lsx_vor_v(
            lsx_vandn_v(mask, bits(lsx_vfmul_d(w, inv))),
            lsx_vand_v(
                mask,
                std::mem::transmute::<m128d, m128i>(lsx::splat_f64(1.0)),
            ),
        ));
        let xo = unbits(lsx_vandn_v(mask, bits(lsx_vfmul_d(x, inv))));
        let yo = unbits(lsx_vandn_v(mask, bits(lsx_vfmul_d(y, inv))));
        let zo = unbits(lsx_vandn_v(mask, bits(lsx_vfmul_d(z, inv))));
        (wo, xo, yo, zo)
    }
}

/// 标量尾（与向量路径同公式、同结合次序）。
#[inline]
fn tail(qw: &[f64], qx: &[f64], qy: &[f64], qz: &[f64], m: [&mut [f64]; 9], from: usize) {
    let out = m;
    for i in from..qw.len() {
        let n = (((qw[i] * qw[i] + qx[i] * qx[i]) + qy[i] * qy[i]) + qz[i] * qz[i]).sqrt();
        let (w, x, y, z) = if n < super::quat_normalize_batch::DEGENERATE {
            (1.0, 0.0, 0.0, 0.0)
        } else {
            // 与向量路径同式：1 次除法 + 乘法
            let inv = 1.0 / n;
            (qw[i] * inv, qx[i] * inv, qy[i] * inv, qz[i] * inv)
        };
        let r = [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ];
        for k in 0..9 {
            out[k][i] = r[k];
        }
    }
}

/// 数值回归测试。
#[cfg(test)]
mod tests {
    use crate::ffi::attitude::lasx_quat_to_dcm_batch;
    use crate::ops::testutil::{rel_err, states, Lcg};

    /// 用 R 旋转向量必须与 `quat_rotate` 的公式逐位一致。
    #[test]
    fn test_quat_to_dcm_matches_rotation_formula() {
        let n = 21;
        let (rx, ry, rz) = states(n);
        // 造一组非平凡四元数（用位置分量凑，保证不是单位四元数，顺带测内部归一化）
        let qw: Vec<f64> = (0..n).map(|i| 0.3 + 0.001 * i as f64).collect();
        let mut m: Vec<Vec<f64>> = (0..9).map(|_| vec![0.0; n]).collect();
        lasx_quat_to_dcm_batch(
            qw.as_ptr(),
            rx.as_ptr(),
            ry.as_ptr(),
            rz.as_ptr(),
            m[0].as_mut_ptr(),
            m[1].as_mut_ptr(),
            m[2].as_mut_ptr(),
            m[3].as_mut_ptr(),
            m[4].as_mut_ptr(),
            m[5].as_mut_ptr(),
            m[6].as_mut_ptr(),
            m[7].as_mut_ptr(),
            m[8].as_mut_ptr(),
            n as i32,
        );
        for i in 0..n {
            // 参考：与 quat_rotate 同公式（先单位化，再按 R 乘）
            let nn = ((qw[i] * qw[i] + rx[i] * rx[i]) + ry[i] * ry[i]) + rz[i] * rz[i];
            let nn = nn.sqrt();
            let (w, x, y, z) = if nn < 1e-15 {
                (1.0, 0.0, 0.0, 0.0)
            } else {
                (qw[i] / nn, rx[i] / nn, ry[i] / nn, rz[i] / nn)
            };
            let r = [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ];
            // 与精确除法参考比紧相对误差（内核用"1 次除法 + 乘法"，见 docs/dev.md §13.6）
            for k in 0..9 {
                assert!(
                    rel_err(m[k][i], r[k]) < 1e-15,
                    "i={i} k={k}: {} vs {}",
                    m[k][i],
                    r[k]
                );
            }
            // 正交性：R·Rᵀ ≈ I
            for r0 in 0..3 {
                for c0 in 0..3 {
                    let dot: f64 = (0..3).map(|t| m[r0 * 3 + t][i] * m[c0 * 3 + t][i]).sum();
                    let want = if r0 == c0 { 1.0 } else { 0.0 };
                    assert!(
                        rel_err(dot, want) < 1e-12,
                        "i={i} 正交性 ({r0},{c0}) = {dot}"
                    );
                }
            }
        }
    }

    /// 路径一致性：LASX vs LSX、向量体 vs 标量尾（4 次除法 → 1 次后的守护）。
    #[test]
    fn paths_bit_exact() {
        const N: usize = 261;
        let mut rng = Lcg(0x2c4d_3e5f);
        let q: Vec<[f64; 4]> = (0..N)
            .map(|_| [rng.f64(), rng.f64(), rng.f64(), rng.f64()])
            .collect();
        let run = |force_lsx: bool, count: usize| -> Vec<[f64; 9]> {
            let (qw, qx, qy, qz) = (
                q.iter().map(|v| v[0]).collect::<Vec<_>>(),
                q.iter().map(|v| v[1]).collect::<Vec<_>>(),
                q.iter().map(|v| v[2]).collect::<Vec<_>>(),
                q.iter().map(|v| v[3]).collect::<Vec<_>>(),
            );
            let mut m: Vec<Vec<f64>> = (0..9).map(|_| vec![0.0; count]).collect();
            let mp: [*mut f64; 9] = std::array::from_fn(|k| m[k].as_mut_ptr());
            crate::arch::lasx_force_lsx_thread(force_lsx);
            lasx_quat_to_dcm_batch(
                qw.as_ptr(),
                qx.as_ptr(),
                qy.as_ptr(),
                qz.as_ptr(),
                mp[0],
                mp[1],
                mp[2],
                mp[3],
                mp[4],
                mp[5],
                mp[6],
                mp[7],
                mp[8],
                count as i32,
            );
            crate::arch::lasx_force_lsx_thread(false);
            (0..count)
                .map(|i| std::array::from_fn(|k| m[k][i]))
                .collect()
        };
        let lasx = run(false, N);
        let lsx = run(true, N);
        for i in 0..N {
            for k in 0..9 {
                assert_eq!(
                    lasx[i][k].to_bits(),
                    lsx[i][k].to_bits(),
                    "LASX/LSX i={i} k={k}"
                );
            }
        }
        let body = run(false, 256);
        for i in 0..256 {
            for k in 0..9 {
                assert_eq!(
                    body[i][k].to_bits(),
                    lasx[i][k].to_bits(),
                    "体/尾 i={i} k={k}"
                );
            }
        }
    }
}
