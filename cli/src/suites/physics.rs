use crate::data::{states, velocities, Soa6, J2, MU, RE};
use crate::report::{row3, Row};
use crate::scalar_ref::*;
use crate::timing::{time_mode, timeit, Mode};
use lasx_rs::*;
use std::hint::black_box;

pub fn j2(rows: &mut Vec<Row>) {
    for &n in &[1usize << 16, 1 << 20] {
        let (rx, ry, rz) = states(n);
        let (mut ax, mut ay, mut az) = (vec![0f64; n], vec![0f64; n], vec![0f64; n]);
        let lasx = time_mode(Mode::Lasx, || {
            lasx_j2_accel_batch(
                rx.as_ptr(),
                ry.as_ptr(),
                rz.as_ptr(),
                MU,
                J2,
                RE,
                ax.as_mut_ptr(),
                ay.as_mut_ptr(),
                az.as_mut_ptr(),
                n as i32,
            );
            let _ = black_box(ax[0]);
        });
        let lsx = time_mode(Mode::Lsx, || {
            lasx_j2_accel_batch(
                rx.as_ptr(),
                ry.as_ptr(),
                rz.as_ptr(),
                MU,
                J2,
                RE,
                ax.as_mut_ptr(),
                ay.as_mut_ptr(),
                az.as_mut_ptr(),
                n as i32,
            );
            let _ = black_box(ax[0]);
        });
        let scalar = timeit(|| {
            scalar_j2_accel(&rx, &ry, &rz, MU, J2, RE, &mut ax, &mut ay, &mut az);
            let _ = black_box(ax[0]);
        });
        row3(
            "lasx_j2_accel_batch f64",
            n.to_string(),
            n as f64,
            "sample/s",
            lasx,
            Some(lsx),
            scalar,
            rows,
        );
    }
}

pub fn ballistic(rows: &mut Vec<Row>) {
    for &n in &[1usize << 16, 1 << 20] {
        let zero = vec![0f32; n];
        let mut x = zero.clone();
        let mut y = zero.clone();
        let mut z = zero.clone();
        let vx: Vec<f32> = (0..n).map(|i| 300.0 + i as f32 * 10.0).collect();
        let vy: Vec<f32> = (0..n).map(|i| 40.0 - i as f32 * 2.0).collect();
        let vz = zero.clone();
        let mut vxw = vx.clone();
        let mut vyw = vy.clone();
        let mut vzw = vz.clone();
        let k = vec![1e-5f32; n];
        let lasx = time_mode(Mode::Lasx, || {
            lasx_ballistic_step(
                x.as_mut_ptr(),
                y.as_mut_ptr(),
                z.as_mut_ptr(),
                vxw.as_mut_ptr(),
                vyw.as_mut_ptr(),
                vzw.as_mut_ptr(),
                k.as_ptr(),
                n as i32,
                0.005,
                9.81,
            );
            let _ = black_box(x[0]);
        });
        let lsx = time_mode(Mode::Lsx, || {
            lasx_ballistic_step(
                x.as_mut_ptr(),
                y.as_mut_ptr(),
                z.as_mut_ptr(),
                vxw.as_mut_ptr(),
                vyw.as_mut_ptr(),
                vzw.as_mut_ptr(),
                k.as_ptr(),
                n as i32,
                0.005,
                9.81,
            );
            let _ = black_box(x[0]);
        });
        let mut vxs = vx.clone();
        let mut vys = vy.clone();
        let mut vzs = vz.clone();
        let scalar = timeit(|| {
            scalar_ballistic(
                &mut x, &mut y, &mut z, &mut vxs, &mut vys, &mut vzs, &k, 0.005, 9.81,
            );
            let _ = black_box(x[0]);
        });
        row3(
            "lasx_ballistic_step f32",
            n.to_string(),
            n as f64,
            "shot/s",
            lasx,
            Some(lsx),
            scalar,
            rows,
        );
    }
}

pub fn rk4(rows: &mut Vec<Row>) {
    for &n in &[1usize << 14, 1 << 18] {
        let (rx, ry, rz) = states(n);
        let (vx, vy, vz) = velocities(n);
        let st = Soa6::new(rx, ry, rz, vx, vy, vz);
        let mut single = st.clone();
        let lasx = time_mode(Mode::Lasx, || {
            single.step(false);
            let _ = black_box(single.rx[0]);
        });
        // 注意：本内核无 LSX 向量路径，force_lsx 时整体退化为标量
        let lsx = time_mode(Mode::Lsx, || {
            single.step(true);
            let _ = black_box(single.rx[0]);
        });
        let w = st.clone();
        let scalar = timeit(|| {
            let mut s = w.clone();
            s.step_scalar();
            let _ = black_box(s.rx[0]);
        });
        row3(
            "lasx_rk4_j2_step_batch f64",
            n.to_string(),
            n as f64,
            "sample/s",
            lasx,
            Some(lsx),
            scalar,
            rows,
        );
    }
}
