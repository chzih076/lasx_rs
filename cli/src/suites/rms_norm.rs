//! `rms_norm`：行内 RMSNorm 的带宽实测。
//!
//! 与 `softmax_rows` 同族：吃带宽（读 `x` + 写 `out`，权重 `w` 每行复用、通常驻留缓存），
//! 所以口径同样是**字节吞吐**。「标量」列是朴素 Rust 行循环，**只作性能参照**（`sqrt`/除法
//! 的实现不同，逐位一致性由 `src/ops/rms_norm.rs` 的单测守）。

use crate::data::{AlignedBuf, Lcg};
use crate::report::{row3, Row};
use crate::timing::{time_mode, timeit, Mode};
use lasx_rs::ffi::nn::lasx_rms_norm;
use std::hint::black_box;

fn scalar_rms_norm(x: &[f32], w: &[f32], eps: f32, rows: usize, cols: usize, out: &mut [f32]) {
    for i in 0..rows {
        let (xr, o) = (
            &x[i * cols..(i + 1) * cols],
            &mut out[i * cols..(i + 1) * cols],
        );
        let mut s = 0f32;
        for v in xr {
            s += v * v;
        }
        let inv = 1.0 / (s / cols as f32 + eps).sqrt();
        for j in 0..cols {
            o[j] = xr[j] * inv * w[j];
        }
    }
}

pub fn rms_norm(rows: &mut Vec<Row>) {
    for &(n_rows, n_cols) in &[
        (128usize, 128usize),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (32, 4096),
        (4096, 128),
    ] {
        let mut rng = Lcg::new((n_rows * 1000 + n_cols) as u64);
        let x = AlignedBuf::fill_with(n_rows * n_cols, |_| rng.f32() * 8.0 - 4.0);
        let w = AlignedBuf::fill_with(n_cols, |_| rng.f32() + 0.5);
        let mut out = AlignedBuf::new(n_rows * n_cols);
        let tag = format!("{n_rows}×{n_cols}");

        let lasx = time_mode(Mode::Lasx, || {
            lasx_rms_norm(
                x.as_ptr(),
                w.as_ptr(),
                out.as_mut_ptr(),
                n_rows as i32,
                n_cols as i32,
                1e-5,
            );
            let _ = black_box(out[0]);
        });
        let scalar = timeit(|| {
            scalar_rms_norm(
                x.as_slice(),
                w.as_slice(),
                1e-5,
                n_rows,
                n_cols,
                out.as_mut_slice(),
            );
            let _ = black_box(out[0]);
        });
        // 工作量口径：读 x + 写 out = 2 条流 × n × 4 字节（w 每行复用，不计入）
        let bytes = 2.0 * (n_rows * n_cols) as f64 * 4.0;
        row3(
            "lasx_rms_norm(LASX-only)",
            tag,
            bytes,
            "B/s",
            lasx,
            None,
            scalar,
            rows,
        );
    }
}
