//! N3（int8 推理）：`lasx_gemv_i8` 与量化生产端（`lasx_amax` / `lasx_quantize_i8_*`）的带宽实测。
//!
//! 口径与 §7 一致（**单线程、绑核**）。两组行的比较意义：
//!
//! - `lasx_gemv_i8`：工作量按**权重字节**算（int8 是 1 B/元素），并给出同形状 f32
//!   `lasx_matmul`（`n = 1`，即 f32 的 GEMV 等价路径）的对照行——**同样的数学乘加，
//!   权重表示不同**：4 B/元素 vs 1 B/元素。这一对行回答"int8 到底买到了多少"；
//! - `lasx_quantize_i8_*`：读 f32 激活、写 int8 + scale，工作量按**读写字节**算；
//!   形状取 118M 判别式模型一层的激活（`192 × 768` 与 `192 × 3072`）。
//!   per-tensor 与 per-row 的差就是 **per-token 量化的额外开销**（多一遍 amax 与每行 scale）。
//!
//! 标量参照：GEMV 用 `scalar_dot_i8` 逐行（口径与内核一致的整数点积），
//! 量化用本文件的 `scalar_quantize_rows`（照 `docs/ops.md` §2.12 的契约写：乘倒数、
//! ties-to-even、不夹取），所以"向量 vs 标量"列是同一份契约下的两条实现。

use crate::data::{AlignedBuf, Lcg};
use crate::report::{row3, Row};
use crate::scalar_ref::scalar_dot_i8;
use crate::timing::{time_mode, timeit, Mode};
use lasx_rs::ffi::quant::{
    lasx_absmax_rows, lasx_amax, lasx_gemv_i8, lasx_matmul_i8, lasx_quantize_i8_per_row,
    lasx_quantize_i8_per_tensor,
};
use lasx_rs::lasx_matmul;
use std::hint::black_box;

/// 标量 GEMV：逐行 `scalar_dot_i8` + 出口一次乘法（与内核同一出口口径）。
fn scalar_gemv_i8(w: &[i8], sw: &[f32], x: &[i8], sx: f32, m: usize, k: usize, y: &mut [f32]) {
    for r in 0..m {
        let acc = scalar_dot_i8(&w[r * k..(r + 1) * k], x);
        y[r] = (acc as f32) * (sw[r] * sx);
    }
}

/// 标量量化（逐行；`rows == 1` 即逐张量）：照 `docs/ops.md` §2.12 的契约写。
fn scalar_quantize_rows(x: &[f32], rows: usize, cols: usize, q: &mut [i8], scales: &mut [f32]) {
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let amax = row.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        let scale = if amax > 0.0 && amax.is_finite() {
            amax / 127.0
        } else {
            0.0
        };
        let recip = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        scales[r] = scale;
        for (j, &v) in row.iter().enumerate() {
            q[r * cols + j] = if recip == 0.0 {
                0
            } else {
                (v * recip).round_ties_even() as i8
            };
        }
    }
}

/// `lasx_gemv_i8`：int8 权重 × int8 激活，形状取 118M 编码器的真实层形状。
pub fn int8_gemv(rows: &mut Vec<Row>) {
    // (m, k)：Q/K/V/O 投影 768×768；FFN 上/下 3072×768 与 768×3072；
    // 末尾两行是"单行点积"（GEMV 的核心循环）与 L1 驻留的小形状。
    for &(m, k) in &[
        (768usize, 768usize),
        (768, 3072),
        (3072, 768),
        (1, 3072),
        (64, 768),
    ] {
        let mut rng = Lcg::new((m * 131 + k) as u64);
        let w = AlignedBuf::fill_with(m * k, |_| rng.i8());
        let sw = AlignedBuf::fill_with(m, |_| 0.001 + 0.0005 * rng.f32().abs());
        let x = AlignedBuf::fill_with(k, |_| rng.i8());
        let sx = 0.002f32;
        let mut y = AlignedBuf::new(m);
        let tag = format!("{m}×{k}");

        let lasx = time_mode(Mode::Lasx, || {
            lasx_gemv_i8(
                w.as_ptr(),
                sw.as_ptr(),
                x.as_ptr(),
                sx,
                y.as_mut_ptr(),
                m as i32,
                k as i32,
            );
            let _ = black_box(y[0]);
        });
        let scalar = timeit(|| {
            scalar_gemv_i8(
                w.as_slice(),
                sw.as_slice(),
                x.as_slice(),
                sx,
                m,
                k,
                y.as_mut_slice(),
            );
            let _ = black_box(y[0]);
        });
        // 口径 = 权重字节（1 B/元素）+ 激活 + 输出
        let bytes = (m * k + k + m * 4) as f64;
        row3(
            "lasx_gemv_i8(LASX-only)",
            tag.clone(),
            bytes,
            "B/s",
            lasx,
            None,
            scalar,
            rows,
        );

        // 对照：同形状的 f32 权重路径（`lasx_matmul` 当 GEMV 用，n = 1）
        let wf = AlignedBuf::fill_with(m * k, |_| rng.f32() * 2.0 - 1.0);
        let xf = AlignedBuf::fill_with(k, |_| rng.f32() * 2.0 - 1.0);
        let mut yf = AlignedBuf::new(m);
        let lasx_f32 = time_mode(Mode::Lasx, || {
            lasx_matmul(
                m as i32,
                k as i32,
                1,
                wf.as_ptr(),
                xf.as_ptr(),
                yf.as_mut_ptr(),
            );
            let _ = black_box(yf[0]);
        });
        let scalar_f32 = timeit(|| {
            crate::scalar_ref::scalar_matmul(m, k, 1, &wf, &xf, &mut yf);
            let _ = black_box(yf[0]);
        });
        let bytes_f32 = (m * k * 4 + k * 4 + m * 4) as f64;
        row3(
            "lasx_matmul f32 同形状(n=1)",
            tag,
            bytes_f32,
            "B/s",
            lasx_f32,
            None,
            scalar_f32,
            rows,
        );
    }
}

/// `lasx_matmul_i8`：批量（prefill 形态）与"逐 token 跑 GEMV"的对照。
///
/// 这两行是 §21.6 那笔账的实测：批量让权重在 L2 里被 `tokens` 个 token 复用，
/// 把限制从**带宽**换成**算力**。f32 对照用**真正的 GEMM**（`lasx_matmul`，n = tokens），
/// 不用 n=1 的打包路径——否则会重复 §7.9 第 2 条那个"对照选错"的坑。
pub fn int8_matmul(rows: &mut Vec<Row>) {
    let tokens = 192usize;
    for &(k, n) in &[(768usize, 768usize), (768, 3072), (3072, 768)] {
        let mut rng = Lcg::new((k * 7 + n) as u64);
        let x = AlignedBuf::fill_with(tokens * k, |_| rng.i8());
        let w = AlignedBuf::fill_with(n * k, |_| rng.i8());
        let sw = AlignedBuf::fill_with(n, |_| 0.001 + 0.0005 * rng.f32().abs());
        let sx = AlignedBuf::fill_with(tokens, |_| 0.002 + 0.0005 * rng.f32().abs());
        let mut y = AlignedBuf::new(tokens * n);
        let tag = format!("{tokens}×{k}×{n}");

        let lasx = time_mode(Mode::Lasx, || {
            lasx_matmul_i8(
                x.as_ptr(),
                w.as_ptr(),
                sw.as_ptr(),
                sx.as_ptr(),
                y.as_mut_ptr(),
                tokens as i32,
                k as i32,
                n as i32,
            );
            let _ = black_box(y[0]);
        });
        // 今天的路径：逐 token 调 `gemv_i8`（权重被重读 tokens 次）
        let gemv_loop = timeit(|| {
            for t in 0..tokens {
                lasx_gemv_i8(
                    w.as_ptr(),
                    sw.as_ptr(),
                    unsafe { x.as_ptr().add(t * k) },
                    sx[t],
                    unsafe { y.as_mut_ptr().add(t * n) },
                    n as i32,
                    k as i32,
                );
            }
            let _ = black_box(y[0]);
        });
        // 口径 = 权重字节（1 B/元素）+ 激活 + 输出（批量的核心指标是"权重读了几遍"）
        let bytes = (n * k + tokens * k + tokens * n * 4) as f64;
        row3(
            "lasx_matmul_i8(LASX-only)",
            tag.clone(),
            bytes,
            "B/s",
            lasx,
            None,
            gemv_loop,
            rows,
        );

        // f32 对照：真正的 GEMM（`lasx_matmul`: C[m×n] = A[m×k]·B[k×n]）
        let xf = AlignedBuf::fill_with(tokens * k, |_| rng.f32() * 2.0 - 1.0);
        let wf = AlignedBuf::fill_with(k * n, |_| rng.f32() * 2.0 - 1.0);
        let mut yf = AlignedBuf::new(tokens * n);
        let lasx_f32 = time_mode(Mode::Lasx, || {
            lasx_matmul(
                tokens as i32,
                k as i32,
                n as i32,
                xf.as_ptr(),
                wf.as_ptr(),
                yf.as_mut_ptr(),
            );
            let _ = black_box(yf[0]);
        });
        let bytes_f32 = ((n * k + tokens * k) * 4 + tokens * n * 4) as f64;
        row3(
            "lasx_matmul f32 GEMM(n=192)",
            tag,
            bytes_f32,
            "B/s",
            lasx_f32,
            None,
            lasx_f32,
            rows,
        );
    }
}

/// 量化生产端：`lasx_amax` / `lasx_quantize_i8_per_tensor` / `lasx_quantize_i8_per_row`。
///
/// 形状是"一层激活"：`192 token × hidden`（`hidden = 768` 注意力、`3072` FFN）。
pub fn int8_quant(rows: &mut Vec<Row>) {
    for &(r_len, c_len) in &[(192usize, 768usize), (192, 3072), (1, 768), (768, 768)] {
        let mut rng = Lcg::new((r_len * 17 + c_len) as u64);
        let x = AlignedBuf::fill_with(r_len * c_len, |_| 8.0 * rng.f32() - 4.0);
        let n = r_len * c_len;
        let mut q = AlignedBuf::fill_with(n, |_| 0i8);
        let mut scales = AlignedBuf::fill_with(r_len, |_| 0f32);
        let mut amax_out = AlignedBuf::fill_with(r_len, |_| 0f32);
        let tag = format!("{r_len}×{c_len}");

        // 读 f32 + 写 int8 + 写 scale（per-row）；per-tensor 只有 1 个 scale
        let bytes_row = (n * 4 + n + r_len * 4) as f64;
        let lasx_row = time_mode(Mode::Lasx, || {
            lasx_quantize_i8_per_row(
                x.as_ptr(),
                q.as_mut_ptr(),
                scales.as_mut_ptr(),
                r_len as i32,
                c_len as i32,
            );
            let _ = black_box(q[0]);
        });
        let scalar_row = timeit(|| {
            scalar_quantize_rows(
                x.as_slice(),
                r_len,
                c_len,
                q.as_mut_slice(),
                scales.as_mut_slice(),
            );
            let _ = black_box(q[0]);
        });
        row3(
            "lasx_quantize_i8_per_row",
            tag.clone(),
            bytes_row,
            "B/s",
            lasx_row,
            None,
            scalar_row,
            rows,
        );

        let lasx_tensor = time_mode(Mode::Lasx, || {
            let _ = black_box(lasx_quantize_i8_per_tensor(
                x.as_ptr(),
                q.as_mut_ptr(),
                n as i32,
            ));
            let _ = black_box(q[0]);
        });
        let scalar_tensor = timeit(|| {
            scalar_quantize_rows(x.as_slice(), 1, n, q.as_mut_slice(), scales.as_mut_slice());
            let _ = black_box(q[0]);
        });
        let bytes_tensor = (n * 4 + n + 4) as f64;
        row3(
            "lasx_quantize_i8_per_tensor",
            tag.clone(),
            bytes_tensor,
            "B/s",
            lasx_tensor,
            None,
            scalar_tensor,
            rows,
        );

        // 逐行 amax（per-token scale 的那一半工作）与整张量 amax
        let lasx_absmax = time_mode(Mode::Lasx, || {
            lasx_absmax_rows(
                x.as_ptr(),
                r_len as i32,
                c_len as i32,
                amax_out.as_mut_ptr(),
            );
            let _ = black_box(amax_out[0]);
        });
        let scalar_absmax = timeit(|| {
            for r in 0..r_len {
                let row = &x[r * c_len..(r + 1) * c_len];
                amax_out[r] = row.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            }
            let _ = black_box(amax_out[0]);
        });
        row3(
            "lasx_absmax_rows",
            tag.clone(),
            (n * 4 + r_len * 4) as f64,
            "B/s",
            lasx_absmax,
            None,
            scalar_absmax,
            rows,
        );

        let lasx_amax = time_mode(Mode::Lasx, || {
            let _ = black_box(lasx_amax(x.as_ptr(), n as i32));
        });
        let scalar_amax = timeit(|| {
            let _ = black_box(x.iter().fold(0.0f32, |a, &b| a.max(b.abs())));
        });
        row3(
            "lasx_amax",
            tag,
            (n * 4) as f64,
            "B/s",
            lasx_amax,
            None,
            scalar_amax,
            rows,
        );
    }
}
