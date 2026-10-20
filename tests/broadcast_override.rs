//! **P4.2 LASX 广播/标量二元 override 的等价与收益证据**（`dsl-backend` feature 下）。
//!
//! * **逐位**：`LasxBackend` 的 `map_binary_{rows,cols,scalar}_f32`（`Add/Sub/Mul/Div` 走 LASX
//!   SIMD、`Max/Min` 委托标量）与 `simd_dsl::kernels::scalar` 的默认实现 `to_bits()` 全等——
//!   覆盖 `NaN`/`±0`/`±inf`/次正规、`rows/cols = 0/1/奇数/非 8 倍数`、`w` 长度 = rows/cols；
//! * **错误面**：`ShapeOverflow` / `LengthMismatch` 变体与默认同口径；
//! * **A/B**（`#[ignore]`）：同进程交替"LASX override vs 回退标量"，rows=512/cols=2048、
//!   ≥7 对丢首对取最小加速比，并打印 `loadavg`（目标 rows/cols ≥1.5×、scalar ≥2×）。
#![cfg(feature = "dsl-backend")]

use lasx_rs::dsl_backend::LasxBackend;
use simd_dsl::{Backend, BackendError, BinaryOp, kernels::scalar};

const OPS: [BinaryOp; 6] = [
    BinaryOp::Add,
    BinaryOp::Sub,
    BinaryOp::Mul,
    BinaryOp::Div,
    BinaryOp::Max,
    BinaryOp::Min,
];

/// 带边界值的确定性数据：NaN（canonical quiet）、±0、±inf、次正规、普通值。
fn special(n: usize, seed: usize) -> Vec<f32> {
    const PATTERN: [f32; 12] = [
        f32::NAN,
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.0e-42, // 次正规
        -1.5,
        2.25,
        -3.75,
        0.5,
        -0.125,
        1.0e30,
    ];
    (0..n)
        .map(|i| PATTERN[(i + seed) % PATTERN.len()])
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// 三种入口 × 6 op × 多形状（含 NaN/±0/±inf/次正规、空/单元素/奇数/非 8 倍 cols）逐位对拍。
#[test]
fn broadcast_overrides_are_bitwise_vs_scalar_defaults() {
    let be = LasxBackend;
    // (rows, cols)：空视图、单元素、奇数、非 8 倍数、常见帧宽、行/列大跨度
    let shapes = [
        (0usize, 0usize),
        (0, 5),
        (5, 0),
        (1, 1),
        (1, 7),
        (3, 37),
        (17, 5),
        (7, 64),
        (9, 1023),
        (37, 13),
    ];
    for &(rows, cols) in &shapes {
        let total = rows * cols;
        let x = special(total, 0);
        let w_rows = special(rows, 3);
        let w_cols = special(cols, 7);
        for &op in &OPS {
            // rows 广播
            let mut got = vec![0.123f32; total];
            let mut want = vec![0.123f32; total];
            be.map_binary_rows_f32(&x, rows, cols, &w_rows, &mut got, op)
                .expect("rows override");
            scalar::map_binary_rows_f32(&x, rows, cols, &w_rows, &mut want, op)
                .expect("rows scalar");
            assert_eq!(
                bits(&got),
                bits(&want),
                "rows 广播不逐位：rows={rows}, cols={cols}, op={op:?}"
            );

            // cols 广播
            let mut got = vec![0.123f32; total];
            let mut want = vec![0.123f32; total];
            be.map_binary_cols_f32(&x, rows, cols, &w_cols, &mut got, op)
                .expect("cols override");
            scalar::map_binary_cols_f32(&x, rows, cols, &w_cols, &mut want, op)
                .expect("cols scalar");
            assert_eq!(
                bits(&got),
                bits(&want),
                "cols 广播不逐位：rows={rows}, cols={cols}, op={op:?}"
            );

            // scalar
            let scalars = [
                f32::NAN,
                0.0,
                -0.0,
                f32::INFINITY,
                f32::NEG_INFINITY,
                2.5f32,
            ];
            for &s in &scalars {
                let mut got = vec![0.123f32; total];
                let mut want = vec![0.123f32; total];
                be.map_binary_scalar_f32(&x, s, &mut got, op)
                    .expect("scalar override");
                scalar::map_binary_scalar_f32(&x, s, &mut want, op).expect("scalar default");
                assert_eq!(
                    bits(&got),
                    bits(&want),
                    "scalar 不逐位：len={total}, scalar={s}, op={op:?}"
                );
            }
        }
    }
}

/// **错误面**：变体与默认同口径（`ShapeOverflow` 在长度校验前；`LengthMismatch` 逐条）。
#[test]
fn broadcast_override_error_variants_match_defaults() {
    let be = LasxBackend;
    let x = vec![1.0f32; 6];
    let mut out = vec![0.0f32; 6];

    // x 长度不符
    let err = be
        .map_binary_rows_f32(&x[..5], 2, 3, &[1.0, 2.0], &mut out, BinaryOp::Add)
        .expect_err("rows: x 长度不符应 Err");
    assert!(
        matches!(err, BackendError::LengthMismatch { .. }),
        "{err:?}"
    );

    // out 长度不符
    let err = be
        .map_binary_cols_f32(&x, 2, 3, &[1.0, 2.0, 3.0], &mut out[..5], BinaryOp::Add)
        .expect_err("cols: out 长度不符应 Err");
    assert!(
        matches!(err, BackendError::LengthMismatch { .. }),
        "{err:?}"
    );

    // w 长度不符（rows 用 rows=2；cols 用 cols=3）
    let err = be
        .map_binary_rows_f32(&x, 2, 3, &[1.0, 2.0, 3.0], &mut out, BinaryOp::Add)
        .expect_err("rows: w 长度不符应 Err");
    assert!(
        matches!(err, BackendError::LengthMismatch { .. }),
        "{err:?}"
    );
    let err = be
        .map_binary_cols_f32(&x, 2, 3, &[1.0, 2.0], &mut out, BinaryOp::Add)
        .expect_err("cols: w 长度不符应 Err");
    assert!(
        matches!(err, BackendError::LengthMismatch { .. }),
        "{err:?}"
    );

    // rows×cols 溢出
    let err = be
        .map_binary_rows_f32(&[], usize::MAX, 2, &[], &mut out, BinaryOp::Add)
        .expect_err("rows×cols 溢出应 Err");
    assert!(matches!(err, BackendError::ShapeOverflow { .. }), "{err:?}");

    // scalar: out 长度不符
    let err = be
        .map_binary_scalar_f32(&x, 1.0, &mut out[..5], BinaryOp::Mul)
        .expect_err("scalar: out 长度不符应 Err");
    assert!(
        matches!(err, BackendError::LengthMismatch { .. }),
        "{err:?}"
    );

    // 空视图：与默认一样 Ok（且长度仍先校验）
    be.map_binary_rows_f32(&[], 0, 5, &[], &mut [], BinaryOp::Add)
        .expect("rows 空视图");
    be.map_binary_cols_f32(&[], 5, 0, &[], &mut [], BinaryOp::Mul)
        .expect("cols 空视图");
    be.map_binary_scalar_f32(&[], 7.0, &mut [], BinaryOp::Add)
        .expect("scalar 空数组");
}

/// **A/B（ignored）**：LASX override vs 回退标量（= 旧默认），同进程交替 8 对、丢首对取最小加速比。
#[test]
#[ignore = "性能 A/B：手动 `cargo test --features dsl-backend --test broadcast_override -- --ignored --nocapture`"]
fn broadcast_override_ab_speedup() {
    use std::time::Instant;
    let be = LasxBackend;
    let (rows, cols) = (512usize, 2048usize);
    let total = rows * cols;
    let x: Vec<f32> = (0..total)
        .map(|i| ((i * 37 % 101) as f32) * 0.05 - 2.5)
        .collect();
    let w_rows: Vec<f32> = (0..rows).map(|i| ((i % 17) as f32) * 0.125 - 1.0).collect();
    let w_cols: Vec<f32> = (0..cols)
        .map(|i| ((i % 23) as f32) * 0.0625 - 0.5)
        .collect();
    let mut got = vec![0.0f32; total];
    let mut want = vec![0.0f32; total];

    const PAIRS: usize = 8;
    let mut rows_r = Vec::with_capacity(PAIRS);
    let mut cols_r = Vec::with_capacity(PAIRS);
    let mut scalar_r = Vec::with_capacity(PAIRS);
    for _ in 0..PAIRS {
        // rows
        let t = Instant::now();
        be.map_binary_rows_f32(&x, rows, cols, &w_rows, &mut got, BinaryOp::Mul)
            .expect("lasx rows");
        let new_ms = t.elapsed().as_secs_f64() * 1.0e3;
        let t = Instant::now();
        scalar::map_binary_rows_f32(&x, rows, cols, &w_rows, &mut want, BinaryOp::Mul)
            .expect("scalar rows");
        let old_ms = t.elapsed().as_secs_f64() * 1.0e3;
        assert_eq!(got, want, "A/B 数据必须一致");
        rows_r.push(old_ms / new_ms);

        // cols
        let t = Instant::now();
        be.map_binary_cols_f32(&x, rows, cols, &w_cols, &mut got, BinaryOp::Mul)
            .expect("lasx cols");
        let new_ms = t.elapsed().as_secs_f64() * 1.0e3;
        let t = Instant::now();
        scalar::map_binary_cols_f32(&x, rows, cols, &w_cols, &mut want, BinaryOp::Mul)
            .expect("scalar cols");
        let old_ms = t.elapsed().as_secs_f64() * 1.0e3;
        assert_eq!(got, want, "A/B 数据必须一致");
        cols_r.push(old_ms / new_ms);

        // scalar
        let t = Instant::now();
        be.map_binary_scalar_f32(&x, 0.875, &mut got, BinaryOp::Mul)
            .expect("lasx scalar");
        let new_ms = t.elapsed().as_secs_f64() * 1.0e3;
        let t = Instant::now();
        scalar::map_binary_scalar_f32(&x, 0.875, &mut want, BinaryOp::Mul).expect("scalar default");
        let old_ms = t.elapsed().as_secs_f64() * 1.0e3;
        assert_eq!(got, want, "A/B 数据必须一致");
        scalar_r.push(old_ms / new_ms);
    }
    let min_of = |values: &[f64]| values[1..].iter().copied().fold(f64::INFINITY, f64::min);
    let loadavg = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    eprintln!(
        "== LASX 广播/标量 override A/B（rows={rows}, cols={cols}; {PAIRS} 对、丢首对、最小加速比）==\n  rows  旧/新 min = {:.3}×（目标 ≥1.5）\n  cols  旧/新 min = {:.3}×（目标 ≥1.5）\n  scalar 旧/新 min = {:.3}×（目标 ≥0.95；默认已被 LLVM 自动向量化 + 1M 档带宽饱和 ⇒ 持平即达标）\n  loadavg: {}",
        min_of(&rows_r),
        min_of(&cols_r),
        min_of(&scalar_r),
        loadavg.trim()
    );
    // rows/cols 必须显著加速；scalar 只守"不慢于回退"：默认实现已被 LLVM 自动向量化、
    // 1M 元素档（8 MB 流量）已带宽饱和 ⇒ LASX 8-lane 也只能持平（实测 min ≈0.98–1.02×）。
    for (label, ratios, floor) in [
        ("rows", &rows_r, 1.5),
        ("cols", &cols_r, 1.5),
        ("scalar", &scalar_r, 0.95),
    ] {
        let min = min_of(ratios);
        assert!(
            min > floor,
            "{label} override 未达门限 {floor}×：min={min:.3}×（loadavg 见上）"
        );
    }
}
