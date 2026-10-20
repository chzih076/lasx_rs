//! **B1/task-106 + B1b/task-108（M5）组合原语 override 的等价证据**（`dsl-backend` feature 下）。
//!
//! * B1：`axpy_inplace_f32`（`api::axpy` 两舍入向量核）——**逐位**；
//! * B1b：`map_unary_f32`（`Exp`/`Sqrt` 向量化；其余委托标量）与 `reduce_rows_map_f32`
//!   （`RowFuse` 的 `Sum`/`SumSq` × `Add/Sub/Mul/Div` 向量路径；未验证组合委托标量）。
//!   本文件的后两个测试覆盖**全矩阵**与**奇数 cols（3/13）**，并打印实测 rel 界。
//!
//! `reduce_sum_rows_f32` 已由 `src/ops/dsl_map.rs` 接上（8-lane 顺序 fold，与默认实现**逐位**，本轮起有 override）；
//! **P4.3 起 `max/min` 轴归约 4 个方向 + 整片 `max/min` 也已 override**（逐位；轴归约按独立轴并行，
//! 整片不并行——全 NaN 轴的 payload 位模式随比较次序变化，见 `dsl_backend.rs` 注释）；
//! 其余原语（`map_binary_scalar_f32`）仍缺 `src/ops` 现成内核。
#![cfg(feature = "dsl-backend")]

use lasx_rs::dsl_backend::LasxBackend;
use simd_dsl::{Backend, BinaryOp, RowEpilogue, RowFuse, RowReduce, ScalarBackend, UnaryOp};

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

// ==================== B1b：map_unary_f32 ====================

/// 相对误差（以标量参照的绝对值为分母；分母为 0 时退化为绝对误差）。
fn max_rel_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = (x - y).abs();
            let m = y.abs();
            if m > 0.0 { d / m } else { d }
        })
        .fold(0.0f32, f32::max)
}

/// `map_unary_f32`：`Exp` 容差类（LASX 多项式 vs libm，打印实测 rel 界）；
/// `Sqrt` 与其余委托变体**逐位**。
#[test]
fn map_unary_override_is_bitwise_except_exp() {
    let be = LasxBackend;
    let n = 4099usize; // 非 8 倍数 ⇒ 覆盖尾块
    // 小幅度有符号数据（exp 不溢出；sin/cos/erf/sigmoid/rcp/abs 定义域安全）。
    let xs: Vec<f32> = (0..n)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.02)
        .collect();
    // 正数数据（ln/sqrt 定义域安全）。
    let xp: Vec<f32> = (0..n)
        .map(|i| ((i * 53 % 97) as f32) * 0.02 + 0.05)
        .collect();

    for (op, x) in [
        (UnaryOp::Exp, &xs),
        (UnaryOp::Sin, &xs),
        (UnaryOp::Cos, &xs),
        (UnaryOp::Erf, &xs),
        (UnaryOp::Sigmoid, &xs),
        (UnaryOp::Rcp, &xs),
        (UnaryOp::Abs, &xs),
        (UnaryOp::Sqrt, &xp),
        (UnaryOp::Ln, &xp),
    ] {
        let mut out_lasx = vec![0.0f32; n];
        let mut out_port = vec![0.0f32; n];
        be.map_unary_f32(x, &mut out_lasx, op).unwrap();
        ScalarBackend.map_unary_f32(x, &mut out_port, op).unwrap();
        if op == UnaryOp::Exp {
            let rel = max_rel_diff(&out_lasx, &out_port);
            println!("map_unary Exp 实测 max_rel = {rel:.3e}（vs libm exp）");
            assert!(
                rel <= 5e-7,
                "map_unary Exp 超出声明容差界：实测 max_rel={rel:.3e} > 5e-7"
            );
        } else {
            for (i, (a, b)) in out_lasx.iter().zip(&out_port).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "map_unary {op:?} @ {i} 不逐位：{a:?} vs {b:?}"
                );
            }
        }
    }
}

/// 长度契约与别名口径（`LengthMismatch` 同 portable）。
#[test]
fn map_unary_override_keeps_length_contract() {
    let be = LasxBackend;
    let x = [1.0f32; 5];
    let mut out = [0.0f32; 4];
    assert!(be.map_unary_f32(&x, &mut out, UnaryOp::Exp).is_err());
    let x = [1.0f32; 5];
    let mut out = [0.0f32; 6];
    assert!(be.map_unary_f32(&x, &mut out, UnaryOp::Sqrt).is_err());
}

// ==================== B1b：reduce_rows_map_f32 ====================

/// `reduce_rows_map_f32` **全矩阵**：`RowReduce{Sum,SumSq,Max}` × 6 个 `RowEpilogue` ×
/// 6 个 `BinaryOp` × cols∈{3,13,16,64}。
///
/// **口径**：`ScalarBackend` 是**顺序单累加器**下限臂（`reference/baseline.rs`），而 portable
/// 契约默认是 `kernels::scalar::reduce_rows_map_f32`（**8 lane 多累加器**）——本 override 的等价
/// 对象是**后者**：断言与它 **`to_bits` 逐位**；再对 `ScalarBackend` 顺序版打印/断言容差 rel 界
/// （`RowReduce::Sum/SumSq` = 容差类；`Max` = 逐位类，两臂都应逐位）。
#[test]
fn reduce_rows_map_override_matrix_is_bitwise() {
    let be = LasxBackend;
    let rows = 7usize;
    let reduces = [RowReduce::Sum, RowReduce::SumSq, RowReduce::Max];
    let epilogues = [
        RowEpilogue::Raw,
        RowEpilogue::Mean,
        RowEpilogue::MeanPlusEps,
        RowEpilogue::SqrtMeanPlusEps,
        RowEpilogue::InvSqrtMeanPlusEps,
        RowEpilogue::Inv,
    ];
    let maps = [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::Max,
        BinaryOp::Min,
    ];
    for cols in [3usize, 13, 16, 64] {
        let n = rows * cols;
        let x: Vec<f32> = (0..n)
            .map(|i| ((i * 29 % 89) as f32) * 0.031 + 0.05)
            .collect();
        for reduce in reduces {
            for epilogue in epilogues {
                for map in maps {
                    let plan = RowFuse {
                        reduce,
                        epilogue,
                        map,
                        eps: 1e-6,
                    };
                    let mut out_lasx = vec![0.0f32; n];
                    let mut out_default = vec![0.0f32; n];
                    let mut out_seq = vec![0.0f32; n];
                    be.reduce_rows_map_f32(&x, rows, cols, &mut out_lasx, plan)
                        .unwrap();
                    // portable 契约默认（多累加器）= 本 override 的等价对象。
                    simd_dsl::kernels::scalar::reduce_rows_map_f32(
                        &x,
                        rows,
                        cols,
                        &mut out_default,
                        plan,
                    )
                    .unwrap();
                    ScalarBackend
                        .reduce_rows_map_f32(&x, rows, cols, &mut out_seq, plan)
                        .unwrap();
                    for (i, (a, b)) in out_lasx.iter().zip(&out_default).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "reduce_rows_map vs portable 默认 @ {i}（cols={cols}, \
                             reduce={reduce:?}, epilogue={epilogue:?}, map={map:?}）不逐位：\
                             {a:?} vs {b:?}"
                        );
                    }
                    // 与顺序下限臂：`Max` 逐位（逐位类）。
                    if reduce == RowReduce::Max {
                        for (i, (a, b)) in out_lasx.iter().zip(&out_seq).enumerate() {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "reduce_rows_map Max 与顺序臂不逐位 @ {i}（cols={cols}, \
                                 epilogue={epilogue:?}, map={map:?}）：{a:?} vs {b:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    // `Sum`/`SumSq` 的容差类：比较**行统计量**本身（`Raw` + `Mul` ⇒ `stat = out/x`，避开
    // `Sub` 的灾难性相消），多累加器（本 override ≡ portable 默认）vs 顺序下限臂。
    let mut worst_tol_rel = 0.0f32;
    let mut worst_tol_desc = String::new();
    for cols in [3usize, 13, 16, 64, 1000] {
        let n = rows * cols;
        let x: Vec<f32> = (0..n)
            .map(|i| ((i * 29 % 89) as f32) * 0.031 + 0.05)
            .collect();
        for reduce in [RowReduce::Sum, RowReduce::SumSq] {
            let plan = RowFuse {
                reduce,
                epilogue: RowEpilogue::Raw,
                map: BinaryOp::Mul,
                eps: 0.0,
            };
            let mut out_lasx = vec![0.0f32; n];
            let mut out_seq = vec![0.0f32; n];
            be.reduce_rows_map_f32(&x, rows, cols, &mut out_lasx, plan)
                .unwrap();
            ScalarBackend
                .reduce_rows_map_f32(&x, rows, cols, &mut out_seq, plan)
                .unwrap();
            for r in 0..rows {
                // `out[r,0] = x[r,0] * stat_r`，`x[r,0] ≥ 0.05` ⇒ 可安全反解 stat。
                let stat_lasx = out_lasx[r * cols] / x[r * cols];
                let stat_seq = out_seq[r * cols] / x[r * cols];
                let rel = max_rel_diff(&[stat_lasx], &[stat_seq]);
                if rel > worst_tol_rel {
                    worst_tol_rel = rel;
                    worst_tol_desc = format!("cols={cols}, reduce={reduce:?}");
                }
            }
        }
    }
    println!(
        "reduce_rows_map 多累加器 vs 顺序臂（行统计量）实测 max_rel = {worst_tol_rel:.3e}（{worst_tol_desc}）"
    );
    assert!(
        worst_tol_rel <= 2e-6,
        "reduce_rows_map 容差类超出声明界：实测 max_rel={worst_tol_rel:.3e} > 2e-6"
    );
}

/// `reduce_rows_map_f32` 契约：长度不符/溢出与 portable 同错误码。
#[test]
fn reduce_rows_map_override_keeps_error_contract() {
    let be = LasxBackend;
    let plan = RowFuse {
        reduce: RowReduce::Sum,
        epilogue: RowEpilogue::Raw,
        map: BinaryOp::Sub,
        eps: 0.0,
    };
    let mut out = [0.0f32; 12];
    assert!(
        be.reduce_rows_map_f32(&[0.0f32; 11], 3, 4, &mut out, plan)
            .is_err()
    );
    let mut out = [0.0f32; 11];
    assert!(
        be.reduce_rows_map_f32(&[0.0f32; 12], 3, 4, &mut out, plan)
            .is_err()
    );
    // rows×cols 溢出：两臂都应 `ShapeOverflow`（而不是 panic）。
    let mut out = [0.0f32; 1];
    assert!(
        be.reduce_rows_map_f32(&[0.0f32; 1], usize::MAX, 2, &mut out, plan)
            .is_err()
    );
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// **P4.3：`max/min` 轴归约 + 整片归约的 6 个 override 与 portable 默认逐位一致**。
///
/// 覆盖：NaN（含**全 NaN 轴** ⇒ NaN payload 的位模式也要相等）、±0、±inf、次正规、单元素、
/// 奇数长度、`cols=1`/`rows=1`。错误面：长度不符 / `rows×cols` 溢出 / 归约轴为 0。
#[test]
fn axis_extreme_overrides_are_bitwise_vs_portable() {
    let be = LasxBackend;
    let portable = ScalarBackend;
    let mut base = sample(64 * 41); // 容纳下面最大形状（64×41 = 2624）
    // 边界值（NaN payload 不同：静默 NaN 与 signaling 形态各取一个）
    base[0] = f32::NAN;
    base[1] = f32::from_bits(0x7fc0_1234); // 非标准 payload
    base[2] = -0.0;
    base[3] = 0.0;
    base[4] = f32::INFINITY;
    base[5] = f32::NEG_INFINITY;
    base[6] = f32::from_bits(1); // 次正规

    for &(rows, cols) in &[(1usize, 1usize), (1, 37), (37, 1), (37, 70), (64, 41)] {
        let x = &base[..rows * cols];
        let mut got = vec![0.0f32; cols.max(rows)];
        let mut want = vec![0.0f32; cols.max(rows)];
        macro_rules! check_axis {
            ($m:ident, $len:expr) => {
                got[..$len].fill(0.0);
                want[..$len].fill(0.0);
                be.$m(x, rows, cols, &mut got[..$len]).expect("Lasx");
                portable
                    .$m(x, rows, cols, &mut want[..$len])
                    .expect("portable");
                assert_eq!(
                    bits(&got[..$len]),
                    bits(&want[..$len]),
                    "{} {rows}×{cols} 不逐位",
                    stringify!($m)
                );
            };
        }
        check_axis!(reduce_max_cols_f32, cols);
        check_axis!(reduce_min_cols_f32, cols);
        check_axis!(reduce_max_rows_f32, rows);
        check_axis!(reduce_min_rows_f32, rows);
    }

    // **全 NaN 轴**：结果 NaN，且 payload 位模式逐位一致（这是"不向量化"的直接理由）。
    let all_nan = vec![f32::from_bits(0x7fc0_0001); 8 * 4];
    let mut got = vec![0.0f32; 4];
    let mut want = vec![0.0f32; 4];
    be.reduce_max_cols_f32(&all_nan, 8, 4, &mut got).unwrap();
    portable
        .reduce_max_cols_f32(&all_nan, 8, 4, &mut want)
        .unwrap();
    assert_eq!(bits(&got), bits(&want), "全 NaN 轴 payload 不逐位");
    // 行方向：输出长度 = rows = 8。
    let (mut got_rows, mut want_rows) = (vec![0.0f32; 8], vec![0.0f32; 8]);
    be.reduce_min_rows_f32(&all_nan, 8, 4, &mut got_rows)
        .unwrap();
    portable
        .reduce_min_rows_f32(&all_nan, 8, 4, &mut want_rows)
        .unwrap();
    assert_eq!(
        bits(&got_rows),
        bits(&want_rows),
        "全 NaN 行 payload 不逐位"
    );

    // **整片 max/min**：n = 0（幺元）/1/2/37/超长（含边界值）逐位一致。
    for n in [0usize, 1, 2, 37, 1000] {
        let x = &base[..n.min(base.len())];
        assert_eq!(
            be.reduce_max_f32(x).unwrap().to_bits(),
            portable.reduce_max_f32(x).unwrap().to_bits(),
            "整片 max n={n} 不逐位"
        );
        assert_eq!(
            be.reduce_min_f32(x).unwrap().to_bits(),
            portable.reduce_min_f32(x).unwrap().to_bits(),
            "整片 min n={n} 不逐位"
        );
    }

    // 错误面：形状不符 / 溢出 / 归约轴为 0（错误变体与 op 名一致）
    assert!(matches!(
        be.reduce_max_cols_f32(&[0.0; 6], 2, 4, &mut [0.0; 4]),
        Err(simd_dsl::BackendError::LengthMismatch {
            op: "reduce_max_cols_f32",
            what: "x",
            ..
        })
    ));
    assert!(matches!(
        be.reduce_min_rows_f32(&[0.0; 8], 2, 4, &mut [0.0; 3]),
        Err(simd_dsl::BackendError::LengthMismatch {
            op: "reduce_min_rows_f32",
            what: "out",
            ..
        })
    ));
    assert!(matches!(
        be.reduce_max_cols_f32(&[], 0, 4, &mut [0.0; 4]),
        Err(simd_dsl::BackendError::LengthMismatch {
            op: "reduce_max_cols_f32",
            ..
        })
    ));
    assert!(matches!(
        be.reduce_min_cols_f32(&[], usize::MAX, 2, &mut [0.0; 2]),
        Err(simd_dsl::BackendError::ShapeOverflow {
            op: "reduce_min_cols_f32",
            ..
        })
    ));
}
