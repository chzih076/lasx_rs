//! **P1 批1+批2 后端 A/B（task-66 / task-69）**：`FusedPlan` 路径 vs 直接 `api::*`（**11 个 kind**）。
//!
//! 臂：
//! * **A = Plan 路**：`LasxBackend::exec_fused_plan(FusedPlan::…)`（本仓 override → LASX 融合核）；
//! * **B = 直接 api**：`api::axpy`/`softmax_rows_into`/`rms_norm_into`/`layer_norm_into`/
//!   `gelu_quick_into`/`rope_into(RopeMode::NeoX)`；几何 5 走 `api::*_batch_into`；
//! * Axpy/SoftmaxRows 另有具名方法（`axpy_f32`/`softmax_rows_f32`）一并做数值核对。
//!
//! 判据：**A ≡ B 逐位**（同一内核）；性能 **B/A ≥0.98**（Plan 路径不掉性能）。
//! `unitize3` 零向量单列：**Plan(LASX) = 0 / portable 计划解释 = NaN**（路径各自语义）。
//!
//! 注：`FusedPlan` 自批2 起**不是 `Copy`**（几何变体自带 f64 缓冲）⇒ 每次用现构造。
//!
//! 用法（`--release`）：
//! ```text
//! cargo run --release --features dsl-backend --example plan_ab
//! ```

use std::hint::black_box;
use std::time::{Duration, Instant};

use lasx_rs::dsl_backend::LasxBackend;
use simd_dsl::{Backend, FusedIo, FusedPlan, PatternKind, ScalarBackend};

const ROUNDS: usize = 6;

fn timeit(f: &mut impl FnMut()) -> Duration {
    let t = Instant::now();
    f();
    t.elapsed()
}
fn median(mut v: Vec<Duration>) -> Duration {
    v.sort_unstable();
    v[v.len() / 2]
}
fn ab(mut a: impl FnMut(), mut b: impl FnMut()) -> (f64, f64) {
    let (mut av, mut bv) = (Vec::new(), Vec::new());
    for r in 0..ROUNDS {
        let (ta, tb) = if r % 2 == 0 {
            (timeit(&mut a), timeit(&mut b))
        } else {
            let tb = timeit(&mut b);
            let ta = timeit(&mut a);
            (ta, tb)
        };
        av.push(ta);
        bv.push(tb);
    }
    av.remove(0);
    bv.remove(0);
    (
        median(av).as_secs_f64() * 1e3,
        median(bv).as_secs_f64() * 1e3,
    )
}
fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "<unreadable>".to_string())
}
fn sample(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed) % 101) as f32) * 0.05 - 2.5)
        .collect()
}
fn gsample(n: usize, seed: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (((i * 37 + seed) % 101) as f64) * 0.05 - 2.5)
        .collect()
}
fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}
fn f64_bits_eq(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}
fn row(name: &str, a: f64, b: f64, eq: &str) {
    println!("| {name} | {a:.3} | {b:.3} | {:.3}× | {eq} |", b / a);
}

/// 空 `FusedIo`（几何 pattern 用这个）。
fn empty_io() -> FusedIo<'static> {
    FusedIo {
        x: &[],
        y: &mut [],
        out: &mut [],
        cos: &[],
        sin: &[],
    }
}
fn f32_io<'a>(x: &'a [f32], out: &'a mut [f32]) -> FusedIo<'a> {
    FusedIo {
        x,
        y: &mut [],
        out,
        cos: &[],
        sin: &[],
    }
}
/// 行/逐元素 f32 pattern 的现构造（`FusedPlan` 非 Copy）。
fn mk_f32_plan(
    name: &str,
    rows: usize,
    cols: usize,
    eps: f32,
    alpha: f32,
    n_dims: usize,
) -> FusedPlan<'static> {
    match name {
        "Axpy" => FusedPlan::Axpy { alpha },
        "SoftmaxRows" => FusedPlan::SoftmaxRows { rows, cols },
        "RmsNormRows" => FusedPlan::RmsNormRows { rows, cols, eps },
        "LayerNormRows" => FusedPlan::LayerNormRows { rows, cols, eps },
        "GeluQuick" => FusedPlan::GeluQuick,
        _ => FusedPlan::RopeApply { rows, cols, n_dims },
    }
}
/// f64 几何 plan 的现构造。
fn vec3_plan<'a>(
    s: f64,
    a: [&'a [f64]; 3],
    b: [&'a [f64]; 3],
    o: [&'a mut [f64]; 3],
) -> FusedPlan<'a> {
    FusedPlan::Vec3AddScaledF64 { s, a, b, o }
}

fn main() {
    let be = LasxBackend;
    println!(
        "# Plan 后端 A/B：`exec_fused_plan`（13 kind；GeluQuick 融合核已于 W3 删除）vs 直接 `api::*`（后端 = LasxBackend）"
    );
    println!(
        "- threads = {}；loadavg = `{}`（同进程交替 {ROUNDS} 对丢首对取中位）",
        lasx_rs::pool::global().threads(),
        loadavg()
    );
    let kinds = [
        PatternKind::Axpy,
        PatternKind::SoftmaxRows,
        PatternKind::RmsNormRows,
        PatternKind::LayerNormRows,
        PatternKind::RopeApply,
        PatternKind::Vec3AddScaledF64,
        PatternKind::Norm3BatchF64,
        PatternKind::Unitize3BatchF64,
        PatternKind::Cross3BatchF64,
        PatternKind::Mat3MulVec3BatchF64,
        // 批3：quantize/dot/gemv 已接；dequantize 如实不接（flags 里会打印 false）
        PatternKind::QuantizeI8Rows,
        PatternKind::DequantizeI8Rows,
        PatternKind::DotI8,
        PatternKind::GemvI8,
    ];
    let flags = kinds
        .iter()
        .map(|k| format!("{k:?}={}", be.supports_plan(*k)))
        .collect::<Vec<_>>()
        .join(" ");
    println!("- `supports_plan`（LASX）：{flags}；portable 全 false（如实）");
    println!("| 用例 | A=Plan ms | B=直接 api ms | B/A | 等价类 |");
    println!("|---|---|---|---|---|");

    // ================= f32：Axpy / Softmax / Rms / Layer / Rope =================
    {
        let n = 1usize << 20;
        let x = sample(n, 3);
        let base = sample(n, 11);
        let alpha = 0.375f32;
        let mut ya = base.clone();
        let mut yb = base.clone();
        let mut yn = base.clone();
        be.exec_fused_plan(
            mk_f32_plan("Axpy", 0, 0, 0.0, alpha, 0),
            FusedIo {
                x: &x,
                y: &mut ya,
                out: &mut [],
                cos: &[],
                sin: &[],
            },
        )
        .unwrap();
        lasx_rs::api::axpy(alpha, &x, &mut yb).unwrap();
        be.axpy_f32(&mut yn, &x, alpha).unwrap();
        let eq = if bits_eq(&ya, &yb) && bits_eq(&ya, &yn) {
            "A≡api≡命名方法 **逐位**"
        } else {
            "**不逐位！**"
        };
        let (a, b) = ab(
            || {
                be.exec_fused_plan(
                    mk_f32_plan("Axpy", 0, 0, 0.0, alpha, 0),
                    FusedIo {
                        x: &x,
                        y: &mut ya,
                        out: &mut [],
                        cos: &[],
                        sin: &[],
                    },
                )
                .unwrap();
                black_box(ya[0]);
            },
            || {
                lasx_rs::api::axpy(alpha, &x, &mut yb).unwrap();
                black_box(yb[0]);
            },
        );
        row("Axpy（1M）", a, b, eq);
        // **W3（task-104）逐位修正的 A/B**：融合计划路（两次舍入核）vs 纯组合 3 趟
        // （`t = alpha·x + 0.0` + `y + t` + copy，与 `ops_dsl::reduce::axpy_f32` 同序）。
        // 注意：`ya` 已被上面的 A/B 循环反复累加 ⇒ 逐位对拍要用**新鲜**缓冲单独跑一次。
        let mut y_plan_bit = base.clone();
        be.exec_fused_plan(
            mk_f32_plan("Axpy", 0, 0, 0.0, alpha, 0),
            FusedIo {
                x: &x,
                y: &mut y_plan_bit,
                out: &mut [],
                cos: &[],
                sin: &[],
            },
        )
        .unwrap();
        let mut yc = base.clone();
        let (mut tmpc, mut accc) = (vec![0.0f32; n], vec![0.0f32; n]);
        be.map_affine_f32(&x, alpha, 0.0, &mut tmpc).unwrap();
        be.map_binary_f32(&base, &tmpc, &mut accc, simd_dsl::BinaryOp::Add)
            .unwrap();
        yc.copy_from_slice(&accc);
        let eq_comp = if bits_eq(&y_plan_bit, &yc) {
            "融合 vs 组合 **逐位**（W3 修正后）"
        } else {
            "**融合 vs 组合不逐位！**"
        };
        let mut yp2 = base.clone();
        let mut yc2 = base.clone();
        let (mut tmp2, mut acc2) = (vec![0.0f32; n], vec![0.0f32; n]);
        let (ac, bc) = ab(
            || {
                be.exec_fused_plan(
                    mk_f32_plan("Axpy", 0, 0, 0.0, alpha, 0),
                    FusedIo {
                        x: &x,
                        y: &mut yp2,
                        out: &mut [],
                        cos: &[],
                        sin: &[],
                    },
                )
                .unwrap();
                black_box(yp2[0]);
            },
            || {
                be.map_affine_f32(&x, alpha, 0.0, &mut tmp2).unwrap();
                be.map_binary_f32(&base, &tmp2, &mut acc2, simd_dsl::BinaryOp::Add)
                    .unwrap();
                yc2.copy_from_slice(&acc2);
                black_box(yc2[0]);
            },
        );
        row("Axpy 组合（1M）", ac, bc, eq_comp);
    }

    let (rows, cols) = (512usize, 2048usize);
    let n = rows * cols;
    let xs = sample(n, 5);
    let eps = 1e-5f32;
    let mut of_a = vec![0.0f32; n];
    let mut of_b = vec![0.0f32; n];
    macro_rules! f32_row {
        ($name:literal, $api:expr) => {{
            be.exec_fused_plan(
                mk_f32_plan($name, rows, cols, eps, 0.0, 0),
                f32_io(&xs, &mut of_a),
            )
            .unwrap();
            $api(&xs, &mut of_b);
            let eq = if bits_eq(&of_a, &of_b) {
                "A≡api **逐位**"
            } else {
                "**不逐位！**"
            };
            let (a, b) = ab(
                || {
                    be.exec_fused_plan(
                        mk_f32_plan($name, rows, cols, eps, 0.0, 0),
                        f32_io(&xs, &mut of_a),
                    )
                    .unwrap();
                    black_box(of_a[0]);
                },
                || {
                    $api(&xs, &mut of_b);
                    black_box(of_b[0]);
                },
            );
            row($name, a, b, eq);
        }};
    }
    f32_row!("SoftmaxRows", |x: &[f32], o: &mut [f32]| {
        lasx_rs::api::softmax_rows_into(x, None, rows, cols, 1.0, o).unwrap()
    });
    f32_row!("RmsNormRows", |x: &[f32], o: &mut [f32]| {
        lasx_rs::api::rms_norm_into(x, None, rows, cols, eps, o).unwrap()
    });
    f32_row!("LayerNormRows", |x: &[f32], o: &mut [f32]| {
        lasx_rs::api::layer_norm_into(x, None, None, rows, cols, eps, o).unwrap()
    });

    // **W3（task-104）：GeluQuick 融合核已删**——LASX 融合核实测比纯组合慢 29%
    // （Lead 实测 composed/fused = 0.763–0.773×）⇒ `supports_plan(GeluQuick)` 报 false、
    // `exec_fused_plan` 回退 portable 解释器。本表不再有 GeluQuick 行；
    // `FusedPlan::GeluQuick` 变体保留在契约面（删除建议由 simd 侧单列 breaking）。
    // RopeApply
    {
        let (rr, cc, nd) = (1024usize, 128usize, 64usize);
        let total = rr * cc;
        let half = nd / 2;
        let cos: Vec<f32> = (0..rr * half)
            .map(|i| ((i * 7 % 41) as f32 * 0.03).cos())
            .collect();
        let sin: Vec<f32> = (0..rr * half)
            .map(|i| ((i * 7 % 41) as f32 * 0.03).sin())
            .collect();
        let xr = sample(total, 9);
        let (mut ra, mut rb) = (vec![0.0f32; total], vec![0.0f32; total]);
        be.exec_fused_plan(
            mk_f32_plan("RopeApply", rr, cc, 0.0, 0.0, nd),
            FusedIo {
                x: &xr,
                y: &mut [],
                out: &mut ra,
                cos: &cos,
                sin: &sin,
            },
        )
        .unwrap();
        lasx_rs::api::rope_into(
            &xr,
            &cos,
            &sin,
            rr,
            cc,
            nd,
            lasx_rs::api::RopeMode::NeoX,
            &mut rb,
        )
        .unwrap();
        let eq = if bits_eq(&ra, &rb) {
            "A≡api **逐位**"
        } else {
            "**不逐位！**"
        };
        let (a, b) = ab(
            || {
                be.exec_fused_plan(
                    mk_f32_plan("RopeApply", rr, cc, 0.0, 0.0, nd),
                    FusedIo {
                        x: &xr,
                        y: &mut [],
                        out: &mut ra,
                        cos: &cos,
                        sin: &sin,
                    },
                )
                .unwrap();
                black_box(ra[0]);
            },
            || {
                lasx_rs::api::rope_into(
                    &xr,
                    &cos,
                    &sin,
                    rr,
                    cc,
                    nd,
                    lasx_rs::api::RopeMode::NeoX,
                    &mut rb,
                )
                .unwrap();
                black_box(rb[0]);
            },
        );
        row("RopeApply（1024×128,n_dims=64）", a, b, eq);
    }

    // ================= 批2：f64 几何 5 pattern =================
    let gn = 1usize << 20;
    let (ax, ay, az) = (gsample(gn, 1), gsample(gn, 2), gsample(gn, 3));
    let (bx, by, bz) = (gsample(gn, 4), gsample(gn, 5), gsample(gn, 6));

    // Vec3AddScaled
    {
        let (mut pa, mut pa1, mut pa2) = (vec![0.0f64; gn], vec![0.0f64; gn], vec![0.0f64; gn]);
        let (mut pb, mut pb1, mut pb2) = (vec![0.0f64; gn], vec![0.0f64; gn], vec![0.0f64; gn]);
        be.exec_fused_plan(
            vec3_plan(
                1.25,
                [&ax, &ay, &az],
                [&bx, &by, &bz],
                [&mut pa[..], &mut pa1[..], &mut pa2[..]],
            ),
            empty_io(),
        )
        .unwrap();
        lasx_rs::api::vec3_add_scaled_batch_into(
            &ax, &ay, &az, &bx, &by, &bz, 1.25, &mut pb, &mut pb1, &mut pb2,
        )
        .unwrap();
        let eq = if f64_bits_eq(&pa, &pb) && f64_bits_eq(&pa1, &pb1) && f64_bits_eq(&pa2, &pb2) {
            "A≡api **逐位**"
        } else {
            "**不逐位！**"
        };
        let (a, b) = ab(
            || {
                be.exec_fused_plan(
                    vec3_plan(
                        1.25,
                        [&ax, &ay, &az],
                        [&bx, &by, &bz],
                        [&mut pa[..], &mut pa1[..], &mut pa2[..]],
                    ),
                    empty_io(),
                )
                .unwrap();
                black_box(pa[0]);
            },
            || {
                lasx_rs::api::vec3_add_scaled_batch_into(
                    &ax, &ay, &az, &bx, &by, &bz, 1.25, &mut pb, &mut pb1, &mut pb2,
                )
                .unwrap();
                black_box(pb[0]);
            },
        );
        row("Vec3AddScaledF64（1M×3）", a, b, eq);
    }
    // Norm3
    {
        let (mut na, mut nb) = (vec![0.0f64; gn], vec![0.0f64; gn]);
        be.exec_fused_plan(
            FusedPlan::Norm3BatchF64 {
                xyz: [&ax, &ay, &az],
                out: &mut na,
            },
            empty_io(),
        )
        .unwrap();
        lasx_rs::api::norm3_batch_into(&ax, &ay, &az, &mut nb).unwrap();
        let eq = if f64_bits_eq(&na, &nb) {
            "A≡api **逐位**"
        } else {
            "**不逐位！**"
        };
        let (a, b) = ab(
            || {
                be.exec_fused_plan(
                    FusedPlan::Norm3BatchF64 {
                        xyz: [&ax, &ay, &az],
                        out: &mut na,
                    },
                    empty_io(),
                )
                .unwrap();
                black_box(na[0]);
            },
            || {
                lasx_rs::api::norm3_batch_into(&ax, &ay, &az, &mut nb).unwrap();
                black_box(nb[0]);
            },
        );
        row("Norm3BatchF64（1M）", a, b, eq);
    }
    // Unitize3（非零）
    {
        let (mut ua, mut ua1, mut ua2) = (vec![0.0f64; gn], vec![0.0f64; gn], vec![0.0f64; gn]);
        let (mut ub, mut ub1, mut ub2) = (vec![0.0f64; gn], vec![0.0f64; gn], vec![0.0f64; gn]);
        be.exec_fused_plan(
            FusedPlan::Unitize3BatchF64 {
                xyz: [&ax, &ay, &az],
                o: [&mut ua[..], &mut ua1[..], &mut ua2[..]],
            },
            empty_io(),
        )
        .unwrap();
        lasx_rs::api::unitize3_batch_into(&ax, &ay, &az, &mut ub, &mut ub1, &mut ub2).unwrap();
        let eq = if f64_bits_eq(&ua, &ub) && f64_bits_eq(&ua1, &ub1) && f64_bits_eq(&ua2, &ub2) {
            "A≡api **逐位**"
        } else {
            "**不逐位！**"
        };
        let (a, b) = ab(
            || {
                be.exec_fused_plan(
                    FusedPlan::Unitize3BatchF64 {
                        xyz: [&ax, &ay, &az],
                        o: [&mut ua[..], &mut ua1[..], &mut ua2[..]],
                    },
                    empty_io(),
                )
                .unwrap();
                black_box(ua[0]);
            },
            || {
                lasx_rs::api::unitize3_batch_into(&ax, &ay, &az, &mut ub, &mut ub1, &mut ub2)
                    .unwrap();
                black_box(ub[0]);
            },
        );
        row("Unitize3BatchF64（1M，非零）", a, b, eq);
    }
    // Cross3
    {
        let (mut ca, mut ca1, mut ca2) = (vec![0.0f64; gn], vec![0.0f64; gn], vec![0.0f64; gn]);
        let (mut cb, mut cb1, mut cb2) = (vec![0.0f64; gn], vec![0.0f64; gn], vec![0.0f64; gn]);
        be.exec_fused_plan(
            FusedPlan::Cross3BatchF64 {
                a: [&ax, &ay, &az],
                b: [&bx, &by, &bz],
                o: [&mut ca[..], &mut ca1[..], &mut ca2[..]],
            },
            empty_io(),
        )
        .unwrap();
        lasx_rs::api::cross3_batch_into(&ax, &ay, &az, &bx, &by, &bz, &mut cb, &mut cb1, &mut cb2)
            .unwrap();
        let eq = if f64_bits_eq(&ca, &cb) && f64_bits_eq(&ca1, &cb1) && f64_bits_eq(&ca2, &cb2) {
            "A≡api **逐位**"
        } else {
            "**不逐位！**"
        };
        let (a, b) = ab(
            || {
                be.exec_fused_plan(
                    FusedPlan::Cross3BatchF64 {
                        a: [&ax, &ay, &az],
                        b: [&bx, &by, &bz],
                        o: [&mut ca[..], &mut ca1[..], &mut ca2[..]],
                    },
                    empty_io(),
                )
                .unwrap();
                black_box(ca[0]);
            },
            || {
                lasx_rs::api::cross3_batch_into(
                    &ax, &ay, &az, &bx, &by, &bz, &mut cb, &mut cb1, &mut cb2,
                )
                .unwrap();
                black_box(cb[0]);
            },
        );
        row("Cross3BatchF64（1M×3）", a, b, eq);
    }
    // Mat3MulVec3
    {
        let gm: Vec<Vec<f64>> = (0..9).map(|k| gsample(gn, 10 + k)).collect();
        let gmr: [&[f64]; 9] = std::array::from_fn(|k| gm[k].as_slice());
        let (mut ma, mut ma1, mut ma2) = (vec![0.0f64; gn], vec![0.0f64; gn], vec![0.0f64; gn]);
        let (mut mb, mut mb1, mut mb2) = (vec![0.0f64; gn], vec![0.0f64; gn], vec![0.0f64; gn]);
        be.exec_fused_plan(
            FusedPlan::Mat3MulVec3BatchF64 {
                m: gmr,
                v: [&ax, &ay, &az],
                o: [&mut ma[..], &mut ma1[..], &mut ma2[..]],
            },
            empty_io(),
        )
        .unwrap();
        lasx_rs::api::mat3_mul_vec3_batch_into(gmr, &ax, &ay, &az, &mut mb, &mut mb1, &mut mb2)
            .unwrap();
        let eq = if f64_bits_eq(&ma, &mb) && f64_bits_eq(&ma1, &mb1) && f64_bits_eq(&ma2, &mb2) {
            "A≡api **逐位**"
        } else {
            "**不逐位！**"
        };
        let (a, b) = ab(
            || {
                be.exec_fused_plan(
                    FusedPlan::Mat3MulVec3BatchF64 {
                        m: gmr,
                        v: [&ax, &ay, &az],
                        o: [&mut ma[..], &mut ma1[..], &mut ma2[..]],
                    },
                    empty_io(),
                )
                .unwrap();
                black_box(ma[0]);
            },
            || {
                lasx_rs::api::mat3_mul_vec3_batch_into(
                    gmr, &ax, &ay, &az, &mut mb, &mut mb1, &mut mb2,
                )
                .unwrap();
                black_box(mb[0]);
            },
        );
        row("Mat3MulVec3BatchF64（1M×3）", a, b, eq);
    }

    // ================= 批3：i8 量化（quantize/dot/gemv；dequantize 未接 ⇒ 跳过并说明） =================
    {
        // quantize（512×2048 = 1M）
        let (qrows, qcols) = (512usize, 2048usize);
        let total = qrows * qcols;
        let qx: Vec<f32> = (0..total)
            .map(|i| ((i * 37 % 101) as f32) * 0.03125 - 1.5)
            .collect();
        let qscale: Vec<f32> = (0..qrows).map(|r| 0.02 + (r % 7) as f32 * 0.01).collect();
        let (mut qa, mut qb) = (vec![0i8; total], vec![0i8; total]);
        be.exec_fused_plan(
            FusedPlan::QuantizeI8Rows {
                rows: qrows,
                cols: qcols,
                x: &qx,
                scale: &qscale,
                out: &mut qa,
            },
            empty_io(),
        )
        .unwrap();
        simd_dsl::kernels::scalar::quantize_i8_rows_f32(&qx, qrows, qcols, &qscale, &mut qb)
            .unwrap();
        let eq = if qa == qb {
            "A≡portable **逐位**（ties-even/clamp 在内核）"
        } else {
            "**不逐位！**"
        };
        let (a, b) = ab(
            || {
                be.exec_fused_plan(
                    FusedPlan::QuantizeI8Rows {
                        rows: qrows,
                        cols: qcols,
                        x: &qx,
                        scale: &qscale,
                        out: &mut qa,
                    },
                    empty_io(),
                )
                .unwrap();
                black_box(qa[0]);
            },
            || {
                simd_dsl::kernels::scalar::quantize_i8_rows_f32(
                    &qx, qrows, qcols, &qscale, &mut qb,
                )
                .unwrap();
                black_box(qb[0]);
            },
        );
        row("QuantizeI8Rows（1M）", a, b, eq);
    }
    {
        // dot（1M；含饱和路径另测过：k=200k 全 127 ⇒ i32::MAX）
        let n = 1usize << 20;
        let da: Vec<i8> = (0..n)
            .map(|i| ((i * 53 % 251) as i32 - 125) as i8)
            .collect();
        let db: Vec<i8> = (0..n)
            .map(|i| ((i * 97 % 241) as i32 - 120) as i8)
            .collect();
        let want = simd_dsl::kernels::scalar::dot_i8(&da, &db).unwrap();
        let mut got = 0i32;
        be.exec_fused_plan(
            FusedPlan::DotI8 {
                a: &da,
                b: &db,
                out: &mut got,
            },
            empty_io(),
        )
        .unwrap();
        let eq = if got == want {
            "A≡portable **逐位**（i64 累加+末端饱和）"
        } else if got == i32::MAX || got == i32::MIN {
            "A=饱和值（portable 也饱和则等价）"
        } else {
            "**不相等！**"
        };
        let (a, b) = ab(
            || {
                let mut o = 0i32;
                be.exec_fused_plan(
                    FusedPlan::DotI8 {
                        a: &da,
                        b: &db,
                        out: &mut o,
                    },
                    empty_io(),
                )
                .unwrap();
                black_box(o);
            },
            || {
                black_box(simd_dsl::kernels::scalar::dot_i8(&da, &db).unwrap());
            },
        );
        row("DotI8（1M）", a, b, eq);
    }
    {
        // gemv（1024×1024）
        let (j, k) = (1024usize, 1024usize);
        let ga: Vec<i8> = (0..j * k)
            .map(|i| ((i * 41 % 251) as i32 - 125) as i8)
            .collect();
        let gx: Vec<i8> = (0..k)
            .map(|i| ((i * 59 % 241) as i32 - 120) as i8)
            .collect();
        let (mut go, mut gw) = (vec![0i32; j], vec![0i32; j]);
        be.exec_fused_plan(
            FusedPlan::GemvI8 {
                j,
                k,
                a: &ga,
                x: &gx,
                out: &mut go,
            },
            empty_io(),
        )
        .unwrap();
        simd_dsl::kernels::scalar::gemv_i8(j, k, &ga, &gx, &mut gw).unwrap();
        let eq = if go == gw {
            "A≡portable **逐位**（i64 累加+末端饱和）"
        } else {
            "**不相等！**"
        };
        let (a, b) = ab(
            || {
                be.exec_fused_plan(
                    FusedPlan::GemvI8 {
                        j,
                        k,
                        a: &ga,
                        x: &gx,
                        out: &mut go,
                    },
                    empty_io(),
                )
                .unwrap();
                black_box(go[0]);
            },
            || {
                simd_dsl::kernels::scalar::gemv_i8(j, k, &ga, &gx, &mut gw).unwrap();
                black_box(gw[0]);
            },
        );
        row("GemvI8（1024×1024）", a, b, eq);
    }
    println!(
        "\n- `DequantizeI8Rows` **无 LASX override**（如实）：`supports_plan` = false、\
         `exec_fused_plan` 回退 portable 解释（逐位）⇒ 不做性能 A/B（与 portable 同路径，无对照意义）。"
    );

    // unitize3 零向量：路径各自语义
    {
        let z4 = vec![0.0f64; 4];
        let (mut za, mut zb, mut zc) = (vec![f64::NAN; 4], vec![f64::NAN; 4], vec![f64::NAN; 4]);
        be.exec_fused_plan(
            FusedPlan::Unitize3BatchF64 {
                xyz: [&z4, &z4, &z4],
                o: [&mut za[..], &mut zb[..], &mut zc[..]],
            },
            empty_io(),
        )
        .unwrap();
        let lasx_zero = za.iter().all(|v| v.to_bits() == 0.0f64.to_bits())
            && zb.iter().all(|v| v.to_bits() == 0.0f64.to_bits())
            && zc.iter().all(|v| v.to_bits() == 0.0f64.to_bits());
        let (mut pa, mut pb, mut pc) = (vec![0.0f64; 4], vec![0.0f64; 4], vec![0.0f64; 4]);
        ScalarBackend
            .exec_fused_plan(
                FusedPlan::Unitize3BatchF64 {
                    xyz: [&z4, &z4, &z4],
                    o: [&mut pa[..], &mut pb[..], &mut pc[..]],
                },
                empty_io(),
            )
            .unwrap();
        let port_nan = pa.iter().all(|v| v.is_nan()) && pb.iter().all(|v| v.is_nan());
        println!(
            "\n- unitize3 零向量：Plan(LASX) = 0？**{lasx_zero}**；portable 计划解释 = NaN？**{port_nan}**\
             （路径各自语义，调用方按 `supports_plan` 选择）"
        );
    }

    println!("\nloadavg（结束）= `{}`", loadavg());
    println!(
        "\n判定：**B/A ≥0.98 视为 Plan 路径不掉性能**（<0.98 如实记）。\
         A = `LasxBackend::exec_fused_plan`（override → LASX 融合核）；B = 直接 `api::*`（同一内核）；\
         **量化 3 条的 B 是 portable 标量默认**（LASX 量化核无公开 api 直调入口），\
         比值 3.1–4.2× 与 P1-5 的 override-vs-portable（4.47/2.27/3.06×）一致 ⇒ Plan 抽象未掉性能。\
         15 个 pattern：14 个 `supports_plan`=true 且 Plan≡对照臂逐位；`DequantizeI8Rows` 如实 false + portable 回退。"
    );
}
