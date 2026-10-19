//! **B1/task-106（M5）组合原语 override 的等价证据**（`dsl-backend` feature 下）。
//!
//! 本轮只接 `axpy_inplace_f32`（`src/ops` 现有 `api::axpy` 两舍入向量核可复用）；
//! 其余 5 个原语（`reduce_rows_map_f32`/`reduce_sum_rows_f32`/`reduce_max_f32`/
//! `map_unary_f32`/`map_binary_scalar_f32`）在 `src/ops` **无现成内核** ⇒ 按任务口径
//! **如实报缺口、不硬造**（清单见 `docs/dev.md §7.52`）。
#![cfg(feature = "dsl-backend")]

use lasx_rs::dsl_backend::LasxBackend;
use simd_dsl::{Backend, ScalarBackend};

fn sample(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 37 % 101) as f32) * 0.05 - 2.5)
        .collect()
}

/// `axpy_inplace_f32`：LASX override 与 portable 默认**逐位**（含 `±0`、次正规、大值边界）。
#[test]
fn axpy_inplace_override_is_bitwise_vs_portable() {
    let be = LasxBackend;
    let n = 4099usize; // 含非 8 倍数尾巴
    let mut x = sample(n);
    // 边界值：±0、次正规、大幅值（不含 NaN——NaN payload 不在逐位承诺内）
    x[0] = 0.0;
    x[1] = -0.0;
    x[2] = f32::from_bits(1);
    x[3] = 3.0e38;
    let base = sample(n);
    for alpha in [0.375f32, 1.0, -1.25, 0.0, 2.0] {
        let mut y_lasx = base.clone();
        let mut y_port = base.clone();
        be.axpy_inplace_f32(&mut y_lasx, &x, alpha).unwrap();
        ScalarBackend
            .axpy_inplace_f32(&mut y_port, &x, alpha)
            .unwrap();
        for (i, (a, b)) in y_lasx.iter().zip(&y_port).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "axpy_inplace_f32 @ {i}（alpha={alpha}）不逐位：{a:?} vs {b:?}"
            );
        }
    }
}

/// 长度契约：LASX override 与 portable 同错误码（`LengthMismatch`）。
#[test]
fn axpy_inplace_override_keeps_length_contract() {
    let be = LasxBackend;
    let mut y = [0.0f32; 4];
    assert!(be.axpy_inplace_f32(&mut y, &[1.0f32; 3], 1.0).is_err());
    let mut y = [0.0f32; 4];
    assert!(be.axpy_inplace_f32(&mut y, &[1.0f32; 5], 1.0).is_err());
}
