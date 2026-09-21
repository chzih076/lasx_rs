use crate::data::{AlignedBuf, Lcg};
use crate::report::{row3, Row};
use crate::scalar_ref::*;
use crate::timing::{time_mode, timeit, Mode};
use lasx_rs::*;
use std::hint::black_box;

pub fn dot(rows: &mut Vec<Row>) {
    for &n in &[8usize, 16, 24, 32, 48, 64, 256, 4096, 1 << 16, 1 << 20] {
        let mut rng = Lcg::new(n as u64);
        let a = AlignedBuf::fill_with(n, |_| rng.f32());
        let b = AlignedBuf::fill_with(n, |_| rng.f32());
        let lasx = time_mode(Mode::Lasx, || {
            let _ = black_box(lasx_dot(a.as_ptr(), b.as_ptr(), n as i32));
        });
        let lsx = time_mode(Mode::Lsx, || {
            let _ = black_box(lasx_dot(a.as_ptr(), b.as_ptr(), n as i32));
        });
        let scalar = timeit(|| {
            let _ = black_box(scalar_dot(&a, &b));
        });
        row3(
            "lasx_dot f32",
            n.to_string(),
            n as f64,
            "elem/s",
            lasx,
            Some(lsx),
            scalar,
            rows,
        );
    }
}

pub fn sum(rows: &mut Vec<Row>) {
    for &n in &[1usize << 16, 1 << 22] {
        let mut rng = Lcg::new(0x5eed ^ n as u64);
        let x = AlignedBuf::fill_with(n, |_| rng.f32());
        let lasx = time_mode(Mode::Lasx, || {
            let _ = black_box(lasx_sum(x.as_ptr(), n as i32));
        });
        let scalar = timeit(|| {
            let _ = black_box(scalar_sum(&x));
        });
        row3(
            "lasx_sum f32(LASX-only)",
            n.to_string(),
            n as f64,
            "elem/s",
            lasx,
            None,
            scalar,
            rows,
        );
    }
}

pub fn axpy(rows: &mut Vec<Row>) {
    for &n in &[1usize << 16, 1 << 22] {
        let mut rng = Lcg::new(0xa9_1e ^ n as u64);
        let x = AlignedBuf::fill_with(n, |_| rng.f32());
        let mut y = AlignedBuf::fill_with(n, |_| rng.f32());
        let lasx = time_mode(Mode::Lasx, || {
            lasx_axpy(0.5, x.as_ptr(), y.as_mut_ptr(), n as i32);
            let _ = black_box(y[0]);
        });
        let mut y2 = y.clone();
        let scalar = timeit(|| {
            scalar_axpy(0.5, &x, &mut y2);
            let _ = black_box(y2[0]);
        });
        row3(
            "lasx_axpy f32(LASX-only)",
            n.to_string(),
            2.0 * n as f64,
            "elem/s",
            lasx,
            None,
            scalar,
            rows,
        );
    }
}

pub fn dot_f64(rows: &mut Vec<Row>) {
    for &n in &[1usize << 15, 1 << 21] {
        let mut rng = Lcg::new(0xd07f64 ^ n as u64);
        let a = AlignedBuf::fill_with(n, |_| rng.f64());
        let b = AlignedBuf::fill_with(n, |_| rng.f64());
        let lasx = time_mode(Mode::Lasx, || {
            let _ = black_box(lasx_dot_f64(a.as_ptr(), b.as_ptr(), n as i32));
        });
        let scalar = timeit(|| {
            let _ = black_box(scalar_dot_f64(&a, &b));
        });
        row3(
            "lasx_dot_f64(LASX-only)",
            n.to_string(),
            n as f64,
            "elem/s",
            lasx,
            None,
            scalar,
            rows,
        );
    }
}
