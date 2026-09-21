//! `lasx_vec3_add_scaled_batch` —— 批量缩放加 `o[i] = a[i] + s·b[i]`（f64 SOA，三分量独立）。
//!
//! LASX 缺失时走 LSX 128 位路径。
use crate::arch::SimdPath;
use crate::arch::{lasx, lsx};
use std::arch::loongarch64::*;

/// 算子入口：解析当前线程的向量路径后分派。
#[inline]
pub(crate) fn vec3_add_scaled_batch(
    ax: &[f64],
    ay: &[f64],
    az: &[f64],
    bx: &[f64],
    by: &[f64],
    bz: &[f64],
    s: f64,
    ox: &mut [f64],
    oy: &mut [f64],
    oz: &mut [f64],
) {
    match SimdPath::detect() {
        SimdPath::Lasx => vec3_add_scaled_batch_lasx(ax, ay, az, bx, by, bz, s, ox, oy, oz),
        SimdPath::Lsx => vec3_add_scaled_batch_lsx(ax, ay, az, bx, by, bz, s, ox, oy, oz),
    }
}

/// LASX 256 位实现。
#[inline]
fn vec3_add_scaled_batch_lasx(
    ax: &[f64],
    ay: &[f64],
    az: &[f64],
    bx: &[f64],
    by: &[f64],
    bz: &[f64],
    s: f64,
    ox: &mut [f64],
    oy: &mut [f64],
    oz: &mut [f64],
) {
    let n = ax.len();
    let vs = lasx::splat_f64(s);
    let mut i = 0;
    while i + 4 <= n {
        let vax = unsafe { lasx::load_f64x4(ax.as_ptr().add(i)) };
        let vay = unsafe { lasx::load_f64x4(ay.as_ptr().add(i)) };
        let vaz = unsafe { lasx::load_f64x4(az.as_ptr().add(i)) };
        let vbx = unsafe { lasx::load_f64x4(bx.as_ptr().add(i)) };
        let vby = unsafe { lasx::load_f64x4(by.as_ptr().add(i)) };
        let vbz = unsafe { lasx::load_f64x4(bz.as_ptr().add(i)) };
        // o = a + s·b（单次 FMA，误差 ≤ 标量 mul+add 两舍入）
        unsafe {
            lasx::store_f64x4(ox.as_mut_ptr().add(i), lasx_xvfmadd_d(vs, vbx, vax));
            lasx::store_f64x4(oy.as_mut_ptr().add(i), lasx_xvfmadd_d(vs, vby, vay));
            lasx::store_f64x4(oz.as_mut_ptr().add(i), lasx_xvfmadd_d(vs, vbz, vaz));
        }
        i += 4;
    }
    for j in i..n {
        // 与向量路径同式：单次 FMA（s·b + a），保证任一分块逐位一致
        ox[j] = f64::mul_add(s, bx[j], ax[j]);
        oy[j] = f64::mul_add(s, by[j], ay[j]);
        oz[j] = f64::mul_add(s, bz[j], az[j]);
    }
}

/// LSX 128 位实现（2 样本/向量）。
#[inline]
fn vec3_add_scaled_batch_lsx(
    ax: &[f64],
    ay: &[f64],
    az: &[f64],
    bx: &[f64],
    by: &[f64],
    bz: &[f64],
    s: f64,
    ox: &mut [f64],
    oy: &mut [f64],
    oz: &mut [f64],
) {
    let n = ax.len();
    let vs = lsx::splat_f64(s);
    let mut i = 0;
    while i + 2 <= n {
        let vax = unsafe { lsx::load_f64x2(ax.as_ptr().add(i)) };
        let vay = unsafe { lsx::load_f64x2(ay.as_ptr().add(i)) };
        let vaz = unsafe { lsx::load_f64x2(az.as_ptr().add(i)) };
        let vbx = unsafe { lsx::load_f64x2(bx.as_ptr().add(i)) };
        let vby = unsafe { lsx::load_f64x2(by.as_ptr().add(i)) };
        let vbz = unsafe { lsx::load_f64x2(bz.as_ptr().add(i)) };
        unsafe {
            lsx::store_f64x2(ox.as_mut_ptr().add(i), lsx_vfmadd_d(vs, vbx, vax));
            lsx::store_f64x2(oy.as_mut_ptr().add(i), lsx_vfmadd_d(vs, vby, vay));
            lsx::store_f64x2(oz.as_mut_ptr().add(i), lsx_vfmadd_d(vs, vbz, vaz));
        }
        i += 2;
    }
    for j in i..n {
        ox[j] = f64::mul_add(s, bx[j], ax[j]);
        oy[j] = f64::mul_add(s, by[j], ay[j]);
        oz[j] = f64::mul_add(s, bz[j], az[j]);
    }
}

/// 数值回归测试：对照独立参考实现。
#[cfg(test)]
mod tests {
    use crate::ffi::batch::lasx_vec3_add_scaled_batch;
    use crate::ops::testutil::{rel_err, states};

    fn scalar_add_scaled(a: &[f64], b: &[f64], s: f64) -> Vec<f64> {
        a.iter().zip(b).map(|(&x, &y)| x + s * y).collect()
    }
    #[test]
    fn test_vec3_add_scaled_batch_matches_scalar() {
        let (x, y, z) = states(37);
        let (b, c, d) = states(37);
        for s in [0.0, 0.5, 2.0, -1.3, 1.0] {
            let (mut ox, mut oy, mut oz) = (vec![0.0; 37], vec![0.0; 37], vec![0.0; 37]);
            lasx_vec3_add_scaled_batch(
                x.as_ptr(),
                y.as_ptr(),
                z.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                d.as_ptr(),
                s,
                ox.as_mut_ptr(),
                oy.as_mut_ptr(),
                oz.as_mut_ptr(),
                37,
            );
            let wx = scalar_add_scaled(&x, &b, s);
            let wy = scalar_add_scaled(&y, &c, s);
            let wz = scalar_add_scaled(&z, &d, s);
            for i in 0..37 {
                assert!(
                    rel_err(ox[i], wx[i]) < 1e-9
                        && rel_err(oy[i], wy[i]) < 1e-9
                        && rel_err(oz[i], wz[i]) < 1e-9,
                    "s={s} i={i}"
                );
            }
        }
    }
}
