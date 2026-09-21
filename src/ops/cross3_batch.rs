//! `lasx_cross3_batch` —— 批量三维叉积 `o = a × b`（f64 SOA）。
//!
//! 右手系，分量公式与 loong-sci `orbit::cross3` 一致：
//! `(ay·bz − az·by, az·bx − ax·bz, ax·by − ay·bx)`。
//!
//! LASX 缺失时走 LSX 128 位路径；三条路径（LASX / LSX / 标量尾）**同结合次序**，
//! 因此任意长度、任意切块都逐位一致。
//!
//! **允许原地**：三个输出可与任意输入别名（向量路径先取完本轮的 a/b 再写回）。

use crate::arch::SimdPath;
use crate::arch::{lasx, lsx};
use std::arch::loongarch64::*;

/// 算子入口：解析当前线程的向量路径后分派。
#[inline]
pub(crate) fn cross3_batch(
    ax: &[f64],
    ay: &[f64],
    az: &[f64],
    bx: &[f64],
    by: &[f64],
    bz: &[f64],
    ox: &mut [f64],
    oy: &mut [f64],
    oz: &mut [f64],
) {
    match SimdPath::detect() {
        SimdPath::Lasx => cross3_batch_lasx(ax, ay, az, bx, by, bz, ox, oy, oz),
        SimdPath::Lsx => cross3_batch_lsx(ax, ay, az, bx, by, bz, ox, oy, oz),
    }
}

/// LASX 256 位实现（4 样本/向量）。
#[inline]
#[allow(clippy::too_many_arguments)]
fn cross3_batch_lasx(
    ax: &[f64],
    ay: &[f64],
    az: &[f64],
    bx: &[f64],
    by: &[f64],
    bz: &[f64],
    ox: &mut [f64],
    oy: &mut [f64],
    oz: &mut [f64],
) {
    let n = ax.len();
    let mut i = 0;
    while i + 4 <= n {
        unsafe {
            let (axv, ayv, azv) = (
                lasx::load_f64x4(ax.as_ptr().add(i)),
                lasx::load_f64x4(ay.as_ptr().add(i)),
                lasx::load_f64x4(az.as_ptr().add(i)),
            );
            let (bxv, byv, bzv) = (
                lasx::load_f64x4(bx.as_ptr().add(i)),
                lasx::load_f64x4(by.as_ptr().add(i)),
                lasx::load_f64x4(bz.as_ptr().add(i)),
            );
            let vx = lasx_xvfsub_d(lasx_xvfmul_d(ayv, bzv), lasx_xvfmul_d(azv, byv));
            let vy = lasx_xvfsub_d(lasx_xvfmul_d(azv, bxv), lasx_xvfmul_d(axv, bzv));
            let vz = lasx_xvfsub_d(lasx_xvfmul_d(axv, byv), lasx_xvfmul_d(ayv, bxv));
            lasx::store_f64x4(ox.as_mut_ptr().add(i), vx);
            lasx::store_f64x4(oy.as_mut_ptr().add(i), vy);
            lasx::store_f64x4(oz.as_mut_ptr().add(i), vz);
        }
        i += 4;
    }
    for j in i..n {
        ox[j] = ay[j] * bz[j] - az[j] * by[j];
        oy[j] = az[j] * bx[j] - ax[j] * bz[j];
        oz[j] = ax[j] * by[j] - ay[j] * bx[j];
    }
}

/// LSX 128 位实现（2 样本/向量）。
#[inline]
#[allow(clippy::too_many_arguments)]
fn cross3_batch_lsx(
    ax: &[f64],
    ay: &[f64],
    az: &[f64],
    bx: &[f64],
    by: &[f64],
    bz: &[f64],
    ox: &mut [f64],
    oy: &mut [f64],
    oz: &mut [f64],
) {
    let n = ax.len();
    let mut i = 0;
    while i + 2 <= n {
        unsafe {
            let (axv, ayv, azv) = (
                lsx::load_f64x2(ax.as_ptr().add(i)),
                lsx::load_f64x2(ay.as_ptr().add(i)),
                lsx::load_f64x2(az.as_ptr().add(i)),
            );
            let (bxv, byv, bzv) = (
                lsx::load_f64x2(bx.as_ptr().add(i)),
                lsx::load_f64x2(by.as_ptr().add(i)),
                lsx::load_f64x2(bz.as_ptr().add(i)),
            );
            let vx = lsx_vfsub_d(lsx_vfmul_d(ayv, bzv), lsx_vfmul_d(azv, byv));
            let vy = lsx_vfsub_d(lsx_vfmul_d(azv, bxv), lsx_vfmul_d(axv, bzv));
            let vz = lsx_vfsub_d(lsx_vfmul_d(axv, byv), lsx_vfmul_d(ayv, bxv));
            lsx::store_f64x2(ox.as_mut_ptr().add(i), vx);
            lsx::store_f64x2(oy.as_mut_ptr().add(i), vy);
            lsx::store_f64x2(oz.as_mut_ptr().add(i), vz);
        }
        i += 2;
    }
    for j in i..n {
        ox[j] = ay[j] * bz[j] - az[j] * by[j];
        oy[j] = az[j] * bx[j] - ax[j] * bz[j];
        oz[j] = ax[j] * by[j] - ay[j] * bx[j];
    }
}

/// 数值回归测试：对照独立标量参考，要求**逐位一致**。
#[cfg(test)]
mod tests {
    use crate::ffi::attitude::lasx_cross3_batch;
    use crate::ops::testutil::Lcg;

    type Six = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

    fn data(n: usize) -> Six {
        let mut rng = Lcg(0x0c05_5a11);
        let mk = |rng: &mut Lcg| (0..n).map(|_| rng.f64() * 100.0).collect::<Vec<f64>>();
        let (a, b, c) = (mk(&mut rng), mk(&mut rng), mk(&mut rng));
        let (d, e, f) = (mk(&mut rng), mk(&mut rng), mk(&mut rng));
        (a, b, c, d, e, f)
    }

    #[test]
    fn test_cross3_batch_bit_exact() {
        for n in [0usize, 1, 2, 3, 4, 5, 7, 8, 9, 33, 64, 65] {
            let (ax, ay, az, bx, by, bz) = data(n);
            let (mut ox, mut oy, mut oz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            lasx_cross3_batch(
                ax.as_ptr(),
                ay.as_ptr(),
                az.as_ptr(),
                bx.as_ptr(),
                by.as_ptr(),
                bz.as_ptr(),
                ox.as_mut_ptr(),
                oy.as_mut_ptr(),
                oz.as_mut_ptr(),
                n as i32,
            );
            for i in 0..n {
                let (wx, wy, wz) = (
                    ay[i] * bz[i] - az[i] * by[i],
                    az[i] * bx[i] - ax[i] * bz[i],
                    ax[i] * by[i] - ay[i] * bx[i],
                );
                assert_eq!(ox[i].to_bits(), wx.to_bits(), "n={n} i={i} ox");
                assert_eq!(oy[i].to_bits(), wy.to_bits(), "n={n} i={i} oy");
                assert_eq!(oz[i].to_bits(), wz.to_bits(), "n={n} i={i} oz");
            }
        }
    }

    /// 叉积的基本性质：`a × a = 0`；原地（输出别名到输入）与非原地逐位一致。
    #[test]
    fn test_cross3_properties() {
        let n = 37;
        let (ax, ay, az, bx, by, bz) = data(n);
        let (mut ox, mut oy, mut oz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);

        // a × a = 0
        lasx_cross3_batch(
            ax.as_ptr(),
            ay.as_ptr(),
            az.as_ptr(),
            ax.as_ptr(),
            ay.as_ptr(),
            az.as_ptr(),
            ox.as_mut_ptr(),
            oy.as_mut_ptr(),
            oz.as_mut_ptr(),
            n as i32,
        );
        assert!(ox.iter().all(|&v| v == 0.0) && oy.iter().all(|&v| v == 0.0));

        // 非原地参考
        let (mut sx, mut sy, mut sz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        lasx_cross3_batch(
            ax.as_ptr(),
            ay.as_ptr(),
            az.as_ptr(),
            bx.as_ptr(),
            by.as_ptr(),
            bz.as_ptr(),
            sx.as_mut_ptr(),
            sy.as_mut_ptr(),
            sz.as_mut_ptr(),
            n as i32,
        );

        // 原地：输出与 b 同一缓冲
        let (mut cx, mut cy, mut cz) = (bx.clone(), by.clone(), bz.clone());
        lasx_cross3_batch(
            ax.as_ptr(),
            ay.as_ptr(),
            az.as_ptr(),
            cx.as_ptr(),
            cy.as_ptr(),
            cz.as_ptr(),
            cx.as_mut_ptr(),
            cy.as_mut_ptr(),
            cz.as_mut_ptr(),
            n as i32,
        );
        assert_eq!(cx, sx, "原地 ox 与非原地不一致");
        assert_eq!(cy, sy, "原地 oy 与非原地不一致");
        assert_eq!(cz, sz, "原地 oz 与非原地不一致");
    }
}
