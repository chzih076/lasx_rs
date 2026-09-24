//! `rope`：旋转位置编码的带宽实测（两种配对 mode 分开报）。
//!
//! 访存口径：读 `x` + 写 `out`（2 条流）+ 读 `cos`/`sin` 各半条（表是 `x` 的一半大小），
//! 合计 **3 条流**。NeoX 走 LASX（8 对/向量），GptJ 走标量（原因见 `ops::rope` 模块文档：
//! 相邻配对的自然序反交错要跨 128 位 lane），所以两行的对比本身就是结论。

use crate::data::{AlignedBuf, Lcg};
use crate::report::{row3, Row};
use crate::timing::{time_mode, timeit, Mode};
use lasx_rs::ffi::nn::lasx_rope;
use std::hint::black_box;

/// 标量参照：与内核同式的朴素循环（`fma` 语义用 `mul_add` 对上）。
///
/// 参数打包成结构体只是为了过 clippy 的 `too_many_arguments`（8 个）——这几个参数本来就
/// 同属一份"形状描述"，打包比 `#[allow]` 更干净。
struct Shape<'a> {
    cos: &'a [f32],
    sin: &'a [f32],
    rows: usize,
    cols: usize,
    n_dims: usize,
    neox: bool,
}

fn scalar_rope(x: &[f32], a: &Shape<'_>, out: &mut [f32]) {
    let (cos, sin, rows, cols, n_dims, neox) = (a.cos, a.sin, a.rows, a.cols, a.n_dims, a.neox);
    let half = n_dims / 2;
    out.copy_from_slice(x);
    for r in 0..rows {
        for i in 0..half {
            let (i0, i1) = if neox {
                (r * cols + i, r * cols + i + half)
            } else {
                (r * cols + 2 * i, r * cols + 2 * i + 1)
            };
            let (x0, x1) = (x[i0], x[i1]);
            let (c, s) = (cos[r * half + i], sin[r * half + i]);
            out[i0] = x0.mul_add(c, -(x1 * s));
            out[i1] = x1.mul_add(c, x0 * s);
        }
    }
}

pub fn rope(rows_out: &mut Vec<Row>) {
    // 典型推理形状：注意力头的 q/k（head_dim 128）、一次 decode 一行、一次 prefill 多行
    for &(n_rows, n_cols, n_dims) in &[
        (1usize, 128usize, 128usize),
        (32, 128, 128),
        (512, 128, 128),
        (4096, 128, 128),
        (8, 4096, 128), // 宽 head（部分旋转：只转前 128 列）
    ] {
        let mut rng = Lcg::new((n_rows * 7919 + n_cols) as u64);
        let x = AlignedBuf::fill_with(n_rows * n_cols, |_| rng.f32() * 4.0 - 2.0);
        let half = n_dims / 2;
        // 表用真实角度（值本身不影响速度，但贴近实际：cos/sin 在 [-1,1]）
        let (cos, sin) = lasx_rs::api::rope_tables(
            &(0..n_rows).map(|r| r as f32).collect::<Vec<f32>>(),
            n_dims,
            10_000.0,
        )
        .expect("建表");
        let mut out = AlignedBuf::new(n_rows * n_cols);
        let tag = format!("{n_rows}×{n_cols}(d={n_dims})");

        for (name, neox, mode_i) in [
            ("lasx_rope·NeoX(LASX)", true, 0i32),
            ("lasx_rope·GptJ(标量)", false, 1),
        ] {
            let lasx = time_mode(Mode::Lasx, || {
                lasx_rope(
                    x.as_ptr(),
                    cos.as_ptr(),
                    sin.as_ptr(),
                    out.as_mut_ptr(),
                    n_rows as i32,
                    n_cols as i32,
                    n_dims as i32,
                    mode_i,
                );
                let _ = black_box(out[0]);
            });
            let shape = Shape {
                cos: cos.as_slice(),
                sin: sin.as_slice(),
                rows: n_rows,
                cols: n_cols,
                n_dims,
                neox,
            };
            let scalar = timeit(|| {
                scalar_rope(x.as_slice(), &shape, out.as_mut_slice());
                let _ = black_box(out[0]);
            });
            // 3 条流：x + out + (cos+sin)/2
            let bytes = 2.5 * (n_rows * n_cols) as f64 * 4.0;
            row3(
                name,
                tag.clone(),
                bytes,
                "B/s",
                lasx,
                None,
                scalar,
                rows_out,
            );
        }
        assert_eq!(half, n_dims / 2);
    }
}
