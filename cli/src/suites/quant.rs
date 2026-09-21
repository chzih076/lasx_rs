use crate::data::{AlignedBuf, Lcg};
use crate::report::{row3, Row};
use crate::scalar_ref::*;
use crate::timing::{time_mode, timeit, Mode};
use lasx_rs::*;
use std::hint::black_box;

/// int8 量化点积（LASX-only）。
pub fn dot_i8(rows: &mut Vec<Row>) {
    for &n in &[1usize << 16, 1 << 22] {
        let mut rng = Lcg::new(0x9a17 ^ n as u64);
        let a = AlignedBuf::fill_with(n, |_| rng.i8());
        let b = AlignedBuf::fill_with(n, |_| rng.i8());
        let lasx = time_mode(Mode::Lasx, || {
            let _ = black_box(lasx_dot_i8(a.as_ptr(), b.as_ptr(), n as i32));
        });
        let scalar = timeit(|| {
            let _ = black_box(scalar_dot_i8(&a, &b));
        });
        row3(
            "lasx_dot_i8(LASX-only)",
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

/// Q4 量化点积（LASX-only）。
pub fn dot_q4(rows: &mut Vec<Row>) {
    for &n in &[1usize << 16, 1 << 22] {
        let mut rng = Lcg::new(0x9a17 ^ n as u64);
        let qa = AlignedBuf::fill_with(n, |_| rng.u8());
        let qb = AlignedBuf::fill_with(n, |_| rng.u8());
        let groups = n.div_ceil(32);
        let sa = AlignedBuf::fill_with(groups, |_| 0.01 + 0.001 * rng.f32().abs());
        let sb = AlignedBuf::fill_with(groups, |_| 0.01 + 0.001 * rng.f32().abs());
        let lasx = time_mode(Mode::Lasx, || {
            let _ = black_box(lasx_dot_q4(
                qa.as_ptr(),
                sa.as_ptr(),
                qb.as_ptr(),
                sb.as_ptr(),
                n as i32,
            ));
        });
        let scalar = timeit(|| {
            let _ = black_box(scalar_dot_q4(&qa, &sa, &qb, &sb));
        });
        row3(
            "lasx_dot_q4(LASX-only)",
            n.to_string(),
            2.0 * n as f64,
            "nibble/s",
            lasx,
            None,
            scalar,
            rows,
        );
    }
}
