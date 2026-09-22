//! `lasx_quat_rotate_batch` —— 批量用四元数旋转三维向量（f64 SOA）。
//!
//! 语义与 loong-sci `attitude::quat_rotate` 一致：**输入四元数先单位化**（含退化 →
//! 单位四元数，规则见 [`super::quat_normalize_batch`]），再用旋转矩阵
//! `R(q)` 乘向量：`o = R·v`（**体坐标 → 惯性坐标**；惯性转体请先取共轭）。
//!
//! `R` 的构造与 [`super::quat_to_dcm_batch`] **共用同一份实现**，两处不会漂移；
//! 每行分量按左结合 `((r0·vx + r1·vy) + r2·vz)` 求和，三条路径一致 ⇒ 逐位一致。
//!
//! LASX 缺失时走 LSX 128 位路径。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（策略见 `docs/dev.md` §17）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX/LSX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]

use crate::arch::SimdPath;
use crate::arch::{lasx, lsx};
use std::arch::loongarch64::*;

/// 算子入口：解析当前线程的向量路径后分派。
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn quat_rotate_batch(
    qw: &[f64],
    qx: &[f64],
    qy: &[f64],
    qz: &[f64],
    vx: &[f64],
    vy: &[f64],
    vz: &[f64],
    ox: &mut [f64],
    oy: &mut [f64],
    oz: &mut [f64],
) {
    match SimdPath::detect() {
        SimdPath::Lasx => quat_rotate_batch_lasx(qw, qx, qy, qz, vx, vy, vz, ox, oy, oz),
        SimdPath::Lsx => quat_rotate_batch_lsx(qw, qx, qy, qz, vx, vy, vz, ox, oy, oz),
    }
}

/// LASX 256 位实现（4 样本/向量）。
#[inline]
#[allow(clippy::too_many_arguments)]
fn quat_rotate_batch_lasx(
    qw: &[f64],
    qx: &[f64],
    qy: &[f64],
    qz: &[f64],
    vx: &[f64],
    vy: &[f64],
    vz: &[f64],
    ox: &mut [f64],
    oy: &mut [f64],
    oz: &mut [f64],
) {
    let n = qw.len();
    let mut i = 0;
    while i + 4 <= n {
        unsafe {
            let (w, x, y, z) = super::quat_normalize_batch::normalize_lasx(
                lasx::load_f64x4(qw.as_ptr().add(i)),
                lasx::load_f64x4(qx.as_ptr().add(i)),
                lasx::load_f64x4(qy.as_ptr().add(i)),
                lasx::load_f64x4(qz.as_ptr().add(i)),
            );
            let r = super::quat_to_dcm_batch::dcm_entries_lasx(w, x, y, z);
            let (vvx, vvy, vvz) = (
                lasx::load_f64x4(vx.as_ptr().add(i)),
                lasx::load_f64x4(vy.as_ptr().add(i)),
                lasx::load_f64x4(vz.as_ptr().add(i)),
            );
            for (row, v) in [(0usize, vvx), (1, vvy), (2, vvz)] {
                let a = lasx_xvfmul_d(r[row * 3], vvx);
                let b = lasx_xvfmul_d(r[row * 3 + 1], vvy);
                let c = lasx_xvfmul_d(r[row * 3 + 2], vvz);
                let _ = v;
                let out = lasx_xvfadd_d(lasx_xvfadd_d(a, b), c);
                match row {
                    0 => lasx::store_f64x4(ox.as_mut_ptr().add(i), out),
                    1 => lasx::store_f64x4(oy.as_mut_ptr().add(i), out),
                    _ => lasx::store_f64x4(oz.as_mut_ptr().add(i), out),
                }
            }
        }
        i += 4;
    }
    tail(qw, qx, qy, qz, vx, vy, vz, ox, oy, oz, i);
}

/// LSX 128 位实现（2 样本/向量）。
#[inline]
#[allow(clippy::too_many_arguments)]
fn quat_rotate_batch_lsx(
    qw: &[f64],
    qx: &[f64],
    qy: &[f64],
    qz: &[f64],
    vx: &[f64],
    vy: &[f64],
    vz: &[f64],
    ox: &mut [f64],
    oy: &mut [f64],
    oz: &mut [f64],
) {
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
            let r = super::quat_to_dcm_batch::dcm_entries_lsx(w, x, y, z);
            let (vvx, vvy, vvz) = (
                lsx::load_f64x2(vx.as_ptr().add(i)),
                lsx::load_f64x2(vy.as_ptr().add(i)),
                lsx::load_f64x2(vz.as_ptr().add(i)),
            );
            let mut row = 0;
            while row < 3 {
                let a = lsx_vfmul_d(r[row * 3], vvx);
                let b = lsx_vfmul_d(r[row * 3 + 1], vvy);
                let c = lsx_vfmul_d(r[row * 3 + 2], vvz);
                let out = lsx_vfadd_d(lsx_vfadd_d(a, b), c);
                match row {
                    0 => lsx::store_f64x2(ox.as_mut_ptr().add(i), out),
                    1 => lsx::store_f64x2(oy.as_mut_ptr().add(i), out),
                    _ => lsx::store_f64x2(oz.as_mut_ptr().add(i), out),
                }
                row += 1;
            }
        }
        i += 2;
    }
    tail(qw, qx, qy, qz, vx, vy, vz, ox, oy, oz, i);
}

/// 单位化的 LSX 版（与 [`super::quat_normalize_batch`] 同规则）。
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
#[allow(clippy::too_many_arguments)]
fn tail(
    qw: &[f64],
    qx: &[f64],
    qy: &[f64],
    qz: &[f64],
    vx: &[f64],
    vy: &[f64],
    vz: &[f64],
    ox: &mut [f64],
    oy: &mut [f64],
    oz: &mut [f64],
    from: usize,
) {
    for i in from..qw.len() {
        let n = (((qw[i] * qw[i] + qx[i] * qx[i]) + qy[i] * qy[i]) + qz[i] * qz[i]).sqrt();
        let (w, x, y, z) = if n < super::quat_normalize_batch::DEGENERATE {
            (1.0, 0.0, 0.0, 0.0)
        } else {
            {
                let inv = 1.0 / n;
                (qw[i] * inv, qx[i] * inv, qy[i] * inv, qz[i] * inv)
            }
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
        let (px, py, pz) = (vx[i], vy[i], vz[i]);
        ox[i] = (r[0] * px + r[1] * py) + r[2] * pz;
        oy[i] = (r[3] * px + r[4] * py) + r[5] * pz;
        oz[i] = (r[6] * px + r[7] * py) + r[8] * pz;
    }
}

/// 数值回归测试。
#[cfg(test)]
mod tests {
    use crate::ffi::attitude::lasx_quat_rotate_batch;
    use crate::ops::testutil::{rel_err, Lcg};

    /// 与 loong-sci `quat_rotate` 的标量公式逐位一致。
    #[test]
    fn test_quat_rotate_batch_bit_exact() {
        let mut rng = Lcg(0x5150_5150);
        for n in [0usize, 1, 2, 3, 4, 5, 8, 9, 33] {
            let mk = |rng: &mut Lcg| (0..n).map(|_| rng.f64()).collect::<Vec<f64>>();
            let (qw, qx, qy, qz) = (mk(&mut rng), mk(&mut rng), mk(&mut rng), mk(&mut rng));
            let (vx, vy, vz) = (mk(&mut rng), mk(&mut rng), mk(&mut rng));
            let (mut ox, mut oy, mut oz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            lasx_quat_rotate_batch(
                qw.as_ptr(),
                qx.as_ptr(),
                qy.as_ptr(),
                qz.as_ptr(),
                vx.as_ptr(),
                vy.as_ptr(),
                vz.as_ptr(),
                ox.as_mut_ptr(),
                oy.as_mut_ptr(),
                oz.as_mut_ptr(),
                n as i32,
            );
            for i in 0..n {
                let nn = (((qw[i] * qw[i] + qx[i] * qx[i]) + qy[i] * qy[i]) + qz[i] * qz[i]).sqrt();
                let (w, x, y, z) = if nn < 1e-15 {
                    (1.0, 0.0, 0.0, 0.0)
                } else {
                    {
                        let inv = 1.0 / nn;
                        (qw[i] * inv, qx[i] * inv, qy[i] * inv, qz[i] * inv)
                    }
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
                let wx = (r[0] * vx[i] + r[1] * vy[i]) + r[2] * vz[i];
                let wy = (r[3] * vx[i] + r[4] * vy[i]) + r[5] * vz[i];
                let wz = (r[6] * vx[i] + r[7] * vy[i]) + r[8] * vz[i];
                assert_eq!(ox[i].to_bits(), wx.to_bits(), "n={n} i={i} ox");
                assert_eq!(oy[i].to_bits(), wy.to_bits(), "n={n} i={i} oy");
                assert_eq!(oz[i].to_bits(), wz.to_bits(), "n={n} i={i} oz");
            }
        }
    }

    /// 绕 z 轴 90° 的四元数把 x̂ 转到 ŷ；旋转保长。
    #[test]
    fn test_quat_rotate_geometry() {
        let n = 5;
        let s = (std::f64::consts::FRAC_PI_4).sin();
        let c = (std::f64::consts::FRAC_PI_4).cos();
        let qw = vec![c; n];
        let qx = vec![0.0; n];
        let qy = vec![0.0; n];
        let qz = vec![s; n];
        let (vx, vy, vz) = (vec![1.0; n], vec![0.0; n], vec![0.0; n]);
        let (mut ox, mut oy, mut oz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        lasx_quat_rotate_batch(
            qw.as_ptr(),
            qx.as_ptr(),
            qy.as_ptr(),
            qz.as_ptr(),
            vx.as_ptr(),
            vy.as_ptr(),
            vz.as_ptr(),
            ox.as_mut_ptr(),
            oy.as_mut_ptr(),
            oz.as_mut_ptr(),
            n as i32,
        );
        for i in 0..n {
            assert!(rel_err(ox[i], 0.0) < 1e-15 && rel_err(oy[i], 1.0) < 1e-15 && oz[i] == 0.0);
            let m = (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]).sqrt();
            let mo = (ox[i] * ox[i] + (oy[i] * oy[i] + oz[i] * oz[i])).sqrt();
            assert!(rel_err(mo, m) < 1e-15, "旋转应保长：{mo} vs {m}");
        }
    }

    /// 路径一致性：LASX vs LSX、向量体 vs 标量尾（本轮 4 次除法 → 1 次后的守护）。
    #[test]
    fn paths_bit_exact() {
        const N: usize = 261;
        let mut rng = Lcg(0x517e_2b3c);
        let q: Vec<[f64; 4]> = (0..N)
            .map(|_| [rng.f64(), rng.f64(), rng.f64(), rng.f64()])
            .collect();
        let v: Vec<[f64; 3]> = (0..N).map(|_| [rng.f64(), rng.f64(), rng.f64()]).collect();
        let run = |force_lsx: bool, count: usize| -> Vec<[f64; 3]> {
            let (qw, qx, qy, qz) = (
                q.iter().map(|v| v[0]).collect::<Vec<_>>(),
                q.iter().map(|v| v[1]).collect::<Vec<_>>(),
                q.iter().map(|v| v[2]).collect::<Vec<_>>(),
                q.iter().map(|v| v[3]).collect::<Vec<_>>(),
            );
            let (vx, vy, vz) = (
                v.iter().map(|v| v[0]).collect::<Vec<_>>(),
                v.iter().map(|v| v[1]).collect::<Vec<_>>(),
                v.iter().map(|v| v[2]).collect::<Vec<_>>(),
            );
            let (mut ox, mut oy, mut oz) = (vec![0.0; count], vec![0.0; count], vec![0.0; count]);
            crate::arch::lasx_force_lsx_thread(force_lsx);
            lasx_quat_rotate_batch(
                qw.as_ptr(),
                qx.as_ptr(),
                qy.as_ptr(),
                qz.as_ptr(),
                vx.as_ptr(),
                vy.as_ptr(),
                vz.as_ptr(),
                ox.as_mut_ptr(),
                oy.as_mut_ptr(),
                oz.as_mut_ptr(),
                count as i32,
            );
            crate::arch::lasx_force_lsx_thread(false);
            (0..count).map(|i| [ox[i], oy[i], oz[i]]).collect()
        };
        let lasx = run(false, N);
        let lsx = run(true, N);
        for i in 0..N {
            for k in 0..3 {
                assert_eq!(
                    lasx[i][k].to_bits(),
                    lsx[i][k].to_bits(),
                    "LASX/LSX i={i} k={k}"
                );
            }
        }
        let body = run(false, 256);
        for i in 0..256 {
            for k in 0..3 {
                assert_eq!(
                    body[i][k].to_bits(),
                    lasx[i][k].to_bits(),
                    "体/尾 i={i} k={k}"
                );
            }
        }
    }
}
