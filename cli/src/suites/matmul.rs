use crate::data::AlignedBuf;
use crate::data::Lcg;
use crate::report::{row3, Row};
use crate::scalar_ref::*;
use crate::timing::{time_mode, timeit, Mode};
use lasx_rs::*;
use std::hint::black_box;

pub fn matmul(rows: &mut Vec<Row>) {
    for &(m, k, n) in &[
        (64usize, 64usize, 64usize),
        (128, 128, 128),
        (256, 256, 256),
    ] {
        let mut rng = Lcg::new((m * 1000 + k * 10 + n) as u64);
        let a = AlignedBuf::fill_with(m * k, |_| rng.f32());
        let b = AlignedBuf::fill_with(k * n, |_| rng.f32());
        let mut c = AlignedBuf::new(m * n);
        let tag = format!("{m}×{k}×{n}");
        let lasx = time_mode(Mode::Lasx, || {
            lasx_matmul(
                m as i32,
                k as i32,
                n as i32,
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
            );
            let _ = black_box(c[0]);
        });
        let scalar = timeit(|| {
            scalar_matmul(m, k, n, &a, &b, &mut c);
            let _ = black_box(c[0]);
        });
        row3(
            "lasx_matmul f32(LASX-only)",
            tag,
            2.0 * (m * k * n) as f64,
            "FLOP/s",
            lasx,
            None,
            scalar,
            rows,
        );
    }

    for &(m, k, n) in &[(64usize, 64usize, 64usize), (128, 128, 128)] {
        let mut rng = Lcg::new((m * 7 + k * 13 + n) as u64);
        let a = AlignedBuf::fill_with(m * k, |_| rng.f64());
        let b = AlignedBuf::fill_with(k * n, |_| rng.f64());
        let mut c = AlignedBuf::new(m * n);
        let tag = format!("{m}×{k}×{n}");
        let lasx = time_mode(Mode::Lasx, || {
            lasx_matmul_f64(
                m as i32,
                k as i32,
                n as i32,
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
            );
            let _ = black_box(c[0]);
        });
        let scalar = timeit(|| {
            scalar_matmul_f64(m, k, n, &a, &b, &mut c);
            let _ = black_box(c[0]);
        });
        row3(
            "lasx_matmul_f64(LASX-only)",
            tag,
            2.0 * (m * k * n) as f64,
            "FLOP/s",
            lasx,
            None,
            scalar,
            rows,
        );
    }
}
