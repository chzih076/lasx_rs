//! `softmax_rows`：行内 softmax 的带宽实测。
//!
//! 这个算子**吃带宽**：读 `x`（可选再读 `mask`）、写 `out`，算力只有"减 max + exp + 乘"。
//! 所以口径用**字节吞吐**而不是 FLOP/s，参照是 `docs/ops.md` §12.2 记的那组数
//! （llama.cpp 的 `SOFT_MAX` 典型 2.4–4 GB/s、可达 ~25 GB/s）。
//!
//! 「标量」列是一条朴素的 Rust 行循环（`f32::exp`）——**只作性能参照**，不与算子逐位比较
//! （`exp` 的实现不同，逐位一致性由 `src/ops/softmax_rows.rs` 的单测守）。

use crate::data::{AlignedBuf, Lcg};
use crate::report::{row3, Row};
use crate::timing::{time_mode, timeit, Mode};
use lasx_rs::ffi::nn::lasx_softmax_rows;
use std::hint::black_box;

/// 朴素参照：逐行"减 max → exp → 求和 → 归一"。
fn scalar_softmax_rows(x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    for i in 0..rows {
        let (xr, o) = (
            &x[i * cols..(i + 1) * cols],
            &mut out[i * cols..(i + 1) * cols],
        );
        let m = xr.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for j in 0..cols {
            o[j] = (xr[j] - m).exp();
            sum += o[j];
        }
        let inv = 1.0 / sum;
        for v in o.iter_mut() {
            *v *= inv;
        }
    }
}

pub fn softmax(rows: &mut Vec<Row>) {
    // 形状按 attention 的实际形态取：方阵（自注意力）、长行（长上下文）、扁矩阵（多层批量）。
    for &(n_rows, n_cols) in &[
        (128usize, 128usize),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (32, 4096),
        (4096, 128),
    ] {
        let mut rng = Lcg::new((n_rows * 1000 + n_cols) as u64);
        let x = AlignedBuf::fill_with(n_rows * n_cols, |_| rng.f32() * 8.0);
        let mut out = AlignedBuf::new(n_rows * n_cols);
        let tag = format!("{n_rows}×{n_cols}");

        let lasx = time_mode(Mode::Lasx, || {
            lasx_softmax_rows(
                x.as_ptr(),
                std::ptr::null(),
                out.as_mut_ptr(),
                n_rows as i32,
                n_cols as i32,
                1.0,
            );
            let _ = black_box(out[0]);
        });
        let scalar = timeit(|| {
            scalar_softmax_rows(x.as_slice(), n_rows, n_cols, out.as_mut_slice());
            let _ = black_box(out[0]);
        });
        // 工作量口径：读 x + 写 out = 2 条流 × n × 4 字节
        let bytes = 2.0 * (n_rows * n_cols) as f64 * 4.0;
        row3(
            "lasx_softmax_rows(LASX-only)",
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
