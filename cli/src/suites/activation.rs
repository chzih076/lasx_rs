//! `silu` / `gelu_quick`：逐元素激活的带宽实测。
//!
//! 与 `rms_norm` 同为**两遍**（读 `x` + 写 `out`），所以口径同样是字节吞吐。
//! 「标量」列是朴素 Rust 循环，**只作性能参照**：它用的是 `std::exp`，与内核的 6 次多项式
//! 不是同一条式子，逐位一致性由 `src/ops/{silu,gelu_quick}.rs` 的单测守。

use crate::data::{AlignedBuf, Lcg};
use crate::report::{row3, Row};
use crate::timing::{time_mode, timeit, Mode};
use lasx_rs::ffi::nn::{lasx_gelu_quick, lasx_silu};
use std::hint::black_box;

/// 标量参照：`silu` 与 `gelu_quick` 只差一个系数，合成一个函数少一份重复。
fn scalar_act(x: &[f32], out: &mut [f32], c: f32) {
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        *o = v / (1.0 + (-(c * v)).exp());
    }
}

pub fn activation(rows: &mut Vec<Row>) {
    for &n in &[1024usize, 65_536, 1_048_576, 8_388_608] {
        let mut rng = Lcg::new(n as u64);
        let x = AlignedBuf::fill_with(n, |_| rng.f32() * 16.0 - 8.0);
        let mut out = AlignedBuf::new(n);
        let tag = format!("n={n}");

        for (name, kernel, c) in [
            (
                "lasx_silu(LASX-only)",
                lasx_silu as extern "C" fn(*const f32, *mut f32, i32),
                1.0f32,
            ),
            ("lasx_gelu_quick(LASX-only)", lasx_gelu_quick, 1.702),
        ] {
            let lasx = time_mode(Mode::Lasx, || {
                // `x`/`out` 各 n 个元素、长度一致（裸符号是 safe fn，内部自己取切片）。
                kernel(x.as_ptr(), out.as_mut_ptr(), n as i32);
                let _ = black_box(out[0]);
            });
            let scalar = timeit(|| {
                scalar_act(x.as_slice(), out.as_mut_slice(), c);
                let _ = black_box(out[0]);
            });
            // 工作量口径：读 x + 写 out = 2 条流 × n × 4 字节
            let bytes = 2.0 * n as f64 * 4.0;
            row3(name, tag.clone(), bytes, "B/s", lasx, None, scalar, rows);
        }
    }
}
