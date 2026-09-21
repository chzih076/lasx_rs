use crate::data::{states, velocities, AlignedBuf, Lcg};
use crate::report::{row3, Row};
use crate::scalar_ref::*;
use crate::timing::{time_mode, timeit, Mode};
use lasx_rs::*;
use std::hint::black_box;

pub fn norm3(rows: &mut Vec<Row>) {
    for &n in &[1usize << 16, 1 << 20] {
        let (xs, ys, zs) = states(n);
        let mut out = AlignedBuf::new(n);
        let lasx = time_mode(Mode::Lasx, || {
            lasx_norm3_batch(
                xs.as_ptr(),
                ys.as_ptr(),
                zs.as_ptr(),
                out.as_mut_ptr(),
                n as i32,
            );
            let _ = black_box(out[0]);
        });
        let lsx = time_mode(Mode::Lsx, || {
            lasx_norm3_batch(
                xs.as_ptr(),
                ys.as_ptr(),
                zs.as_ptr(),
                out.as_mut_ptr(),
                n as i32,
            );
            let _ = black_box(out[0]);
        });
        let scalar = timeit(|| {
            scalar_norm3(&xs, &ys, &zs, &mut out);
            let _ = black_box(out[0]);
        });
        row3(
            "lasx_norm3_batch f64",
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

pub fn vec3(rows: &mut Vec<Row>) {
    for &n in &[1usize << 16, 1 << 20] {
        let (ax, ay, az) = states(n);
        let (bx, by, bz) = velocities(n);
        let (mut ox, mut oy, mut oz) = (vec![0f64; n], vec![0f64; n], vec![0f64; n]);
        let lasx = time_mode(Mode::Lasx, || {
            lasx_vec3_add_scaled_batch(
                ax.as_ptr(),
                ay.as_ptr(),
                az.as_ptr(),
                bx.as_ptr(),
                by.as_ptr(),
                bz.as_ptr(),
                1.5,
                ox.as_mut_ptr(),
                oy.as_mut_ptr(),
                oz.as_mut_ptr(),
                n as i32,
            );
            let _ = black_box(ox[0]);
        });
        let lsx = time_mode(Mode::Lsx, || {
            lasx_vec3_add_scaled_batch(
                ax.as_ptr(),
                ay.as_ptr(),
                az.as_ptr(),
                bx.as_ptr(),
                by.as_ptr(),
                bz.as_ptr(),
                1.5,
                ox.as_mut_ptr(),
                oy.as_mut_ptr(),
                oz.as_mut_ptr(),
                n as i32,
            );
            let _ = black_box(ox[0]);
        });
        let scalar = timeit(|| {
            scalar_vec3_add_scaled(&ax, &ay, &az, &bx, &by, &bz, 1.5, &mut ox, &mut oy, &mut oz);
            let _ = black_box(ox[0]);
        });
        row3(
            "lasx_vec3_add_scaled_batch f64",
            n.to_string(),
            6.0 * n as f64,
            "elem/s",
            lasx,
            Some(lsx),
            scalar,
            rows,
        );
    }
}

pub fn distance2d(rows: &mut Vec<Row>) {
    for &n in &[1usize << 16, 1 << 20] {
        let mut rng = Lcg::new(0xd157 ^ n as u64);
        let xs = AlignedBuf::fill_with(n, |_| 100.0 * rng.f32());
        let ys = AlignedBuf::fill_with(n, |_| 100.0 * rng.f32());
        let mut out = AlignedBuf::new(n);
        let lasx = time_mode(Mode::Lasx, || {
            lasx_batch_distance2d(
                1.0,
                2.0,
                xs.as_ptr(),
                ys.as_ptr(),
                out.as_mut_ptr(),
                n as i32,
            );
            let _ = black_box(out[0]);
        });
        let lsx = time_mode(Mode::Lsx, || {
            lasx_batch_distance2d(
                1.0,
                2.0,
                xs.as_ptr(),
                ys.as_ptr(),
                out.as_mut_ptr(),
                n as i32,
            );
            let _ = black_box(out[0]);
        });
        let scalar = timeit(|| {
            scalar_batch_distance2d(1.0, 2.0, &xs, &ys, &mut out);
            let _ = black_box(out[0]);
        });
        row3(
            "lasx_batch_distance2d f32",
            n.to_string(),
            n as f64,
            "point/s",
            lasx,
            Some(lsx),
            scalar,
            rows,
        );
    }
}
