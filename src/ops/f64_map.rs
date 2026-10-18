//! `f64_map` —— 给 `simd_dsl::Backend` 的 **f64 逐元素族**提供 LASX 内核（P1-4）。
//!
//! # 覆盖（`Backend` 的三个 f64 逐元素方法）
//!
//! | 方法 | 语义 | 本文件入口 |
//! |---|---|---|
//! | `map_binary_f64` | `outᵢ = xᵢ op yᵢ`（`Add`/`Sub`/`Mul`/`Div`） | [`map_binary`] |
//! | `map_ternary_f64` | `outᵢ = xᵢ − yᵢ − zᵢ`（当前唯一变体 `SubSub`） | [`map_ternary`] |
//! | `map_affine_f64` | `outᵢ = a·xᵢ + b` | [`map_affine`] |
//!
//! # 等价类（**逐条写死**，与 `simd_dsl::kernels::scalar` 参照比）
//!
//! * **全部逐位类**（`map_binary` 的 Add/Sub/Mul/Div、`map_ternary`、`map_affine`）：
//!   向量指令（`xvfadd_d`/`xvfsub_d`/`xvfmul_d`/`xvfdiv_d`）与标量 IEEE 运算**逐位同结果**；
//!   **`map_affine` 故意不融 FMA**（先 `a·x` 再 `+b`，与标量参照 `a * v + b` 同序）⇒ 也是逐位，
//!   不像 f32 的 `map_affine` 那样因 FMA 降为容差类（用两条向量指令换"逐位"，值得）。
//! * `Max`/`Min` **不在本文件实现**：Rust `f64::max/min` 的 NaN 语义（另一侧是数就返回它）与
//!   LASX `xvfmax_d` 的 IEEE 语义不同 ⇒ 由 `LasxBackend` 委托标量参照（见 `dsl_backend.rs`）。
//!
//! # 实现要点
//!
//! * **向量宽度 4**（LASX 256 位 = 4 × f64），主循环 **4 路展开**（16 元素/迭代）摊薄访存；
//! * **尾部标量兜底**（`n % 16` / `n % 4`），与标量参照同一表达式 ⇒ 尾部也逐位；
//! * **不依赖 `simd_dsl`**（本仓内核保持独立，与 `dsl_map.rs` 同口径）⇒ 本地 `BinOpF64` 枚举分派。
//! * **别名**：`out` 不得与任一输入重叠（契约要求；本文件只 `debug_assert` 基地址，安全 Rust 由
//!   借用检查器守——与 `dsl_map.rs` 同口径）。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（分层策略：内核模块在函数级统一写 SAFETY，不逐块重复）。
#![allow(clippy::undocumented_unsafe_blocks)]
use crate::arch::lasx;
use std::arch::loongarch64::*;

/// 本地逐元素二元运算种类（与 `simd_dsl::BinaryOp` 的 `Add/Sub/Mul/Div` 一一对应）。
///
/// `Max`/`Min` **不在此枚举里**：NaN 语义不同 ⇒ 由 `LasxBackend` 委托标量（见模块文档）。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum BinOpF64 {
    /// `x + y`
    Add,
    /// `x − y`
    Sub,
    /// `x · y`
    Mul,
    /// `x / y`
    Div,
}

/// 标量兜底（尾部与"小规模直接走标量"同式，保证逐位）。
#[inline]
fn bin_scalar(op: BinOpF64, a: f64, b: f64) -> f64 {
    match op {
        BinOpF64::Add => a + b,
        BinOpF64::Sub => a - b,
        BinOpF64::Mul => a * b,
        BinOpF64::Div => a / b,
    }
}

/// 一条向量指令（4 个 f64 lane）。纯寄存器操作，无内存前提。
#[inline]
fn bin_vec(op: BinOpF64, a: lasx::F64x4, b: lasx::F64x4) -> lasx::F64x4 {
    // SAFETY: 纯寄存器操作（无内存访问、无前提）。
    unsafe {
        match op {
            BinOpF64::Add => lasx_xvfadd_d(a, b),
            BinOpF64::Sub => lasx_xvfsub_d(a, b),
            BinOpF64::Mul => lasx_xvfmul_d(a, b),
            BinOpF64::Div => lasx_xvfdiv_d(a, b),
        }
    }
}

/// `outᵢ = xᵢ op yᵢ`（`Add`/`Sub`/`Mul`/`Div`）。**逐位类**。
///
/// # Safety
/// 调用方保证 `x.len() == y.len() == out.len()` 且 `out` 不与 `x`/`y` 重叠。
#[inline]
pub(crate) fn map_binary(x: &[f64], y: &[f64], out: &mut [f64], op: BinOpF64) {
    let n = x.len();
    debug_assert_eq!(y.len(), n, "map_binary_f64：y 必须与 x 等长");
    debug_assert_eq!(out.len(), n, "map_binary_f64：out 必须与 x 等长");
    debug_assert!(
        !std::ptr::eq(x.as_ptr(), out.as_ptr()) && !std::ptr::eq(y.as_ptr(), out.as_ptr()),
        "map_binary_f64：out 不得与输入重叠（契约）"
    );
    let mut i = 0;
    // 4 路展开：逐元素、无归约 ⇒ 展开不改数值（按构造逐位安全），只为摊薄循环与访存开销。
    while i + 16 <= n {
        // SAFETY: `i + 16 <= n` ⇒ 四组 4-lane 访存全在切片内；非别名由调用方保证。
        unsafe {
            let px = x.as_ptr().add(i);
            let py = y.as_ptr().add(i);
            let po = out.as_mut_ptr().add(i);
            for u in [0usize, 4, 8, 12] {
                let vx = lasx::load_f64x4(px.add(u));
                let vy = lasx::load_f64x4(py.add(u));
                lasx::store_f64x4(po.add(u), bin_vec(op, vx, vy));
            }
        }
        i += 16;
    }
    while i + 4 <= n {
        // SAFETY: `i + 4 <= n` ⇒ 访存落在切片内；非别名由调用方保证。
        unsafe {
            let vx = lasx::load_f64x4(x.as_ptr().add(i));
            let vy = lasx::load_f64x4(y.as_ptr().add(i));
            lasx::store_f64x4(out.as_mut_ptr().add(i), bin_vec(op, vx, vy));
        }
        i += 4;
    }
    for j in i..n {
        out[j] = bin_scalar(op, x[j], y[j]);
    }
}

/// `outᵢ = xᵢ − yᵢ − zᵢ`（`TernaryOp::SubSub`）。**逐位类**。
///
/// # Safety
/// 调用方保证三输入与输出等长、且 `out` 不与任一输入重叠。
#[inline]
pub(crate) fn map_ternary(x: &[f64], y: &[f64], z: &[f64], out: &mut [f64]) {
    let n = x.len();
    debug_assert_eq!(y.len(), n, "map_ternary_f64：y 必须与 x 等长");
    debug_assert_eq!(z.len(), n, "map_ternary_f64：z 必须与 x 等长");
    debug_assert_eq!(out.len(), n, "map_ternary_f64：out 必须与 x 等长");
    debug_assert!(
        !std::ptr::eq(x.as_ptr(), out.as_ptr())
            && !std::ptr::eq(y.as_ptr(), out.as_ptr())
            && !std::ptr::eq(z.as_ptr(), out.as_ptr()),
        "map_ternary_f64：out 不得与输入重叠（契约）"
    );
    let mut i = 0;
    while i + 16 <= n {
        // SAFETY: 同 `map_binary`（四组 4-lane 访存都在切片内）。
        unsafe {
            let px = x.as_ptr().add(i);
            let py = y.as_ptr().add(i);
            let pz = z.as_ptr().add(i);
            let po = out.as_mut_ptr().add(i);
            for u in [0usize, 4, 8, 12] {
                let vx = lasx::load_f64x4(px.add(u));
                let vy = lasx::load_f64x4(py.add(u));
                let vz = lasx::load_f64x4(pz.add(u));
                // 与标量参照同序：`(x − y) − z`（两次减法，不重排）。
                let d = lasx_xvfsub_d(vx, vy);
                lasx::store_f64x4(po.add(u), lasx_xvfsub_d(d, vz));
            }
        }
        i += 16;
    }
    while i + 4 <= n {
        // SAFETY: `i + 4 <= n`。
        unsafe {
            let vx = lasx::load_f64x4(x.as_ptr().add(i));
            let vy = lasx::load_f64x4(y.as_ptr().add(i));
            let vz = lasx::load_f64x4(z.as_ptr().add(i));
            let d = lasx_xvfsub_d(vx, vy);
            lasx::store_f64x4(out.as_mut_ptr().add(i), lasx_xvfsub_d(d, vz));
        }
        i += 4;
    }
    for j in i..n {
        out[j] = (x[j] - y[j]) - z[j];
    }
}

/// `outᵢ = a·xᵢ + b`。**逐位类**（**故意不融 FMA**：先乘后加，与标量参照同序）。
///
/// # Safety
/// 调用方保证 `x.len() == out.len()` 且 `out` 不与 `x` 重叠。
#[inline]
pub(crate) fn map_affine(a: f64, b: f64, x: &[f64], out: &mut [f64]) {
    let n = x.len();
    debug_assert_eq!(out.len(), n, "map_affine_f64：out 必须与 x 等长");
    debug_assert!(
        !std::ptr::eq(x.as_ptr(), out.as_ptr()),
        "map_affine_f64：out 不得与输入重叠（契约）"
    );
    let va = lasx::splat_f64(a);
    let vb = lasx::splat_f64(b);
    let mut i = 0;
    while i + 16 <= n {
        // SAFETY: 四组 4-lane 访存全在切片内；非别名由调用方保证。
        unsafe {
            let px = x.as_ptr().add(i);
            let po = out.as_mut_ptr().add(i);
            for u in [0usize, 4, 8, 12] {
                let v = lasx::load_f64x4(px.add(u));
                // **非 FMA**：`a·x` 再 `+b`（与 `scalar::map_affine_f64` 的 `a * v + b` 同序 ⇒ 逐位）。
                let prod = lasx_xvfmul_d(va, v);
                lasx::store_f64x4(po.add(u), lasx_xvfadd_d(prod, vb));
            }
        }
        i += 16;
    }
    while i + 4 <= n {
        // SAFETY: `i + 4 <= n`。
        unsafe {
            let v = lasx::load_f64x4(x.as_ptr().add(i));
            let prod = lasx_xvfmul_d(va, v);
            lasx::store_f64x4(out.as_mut_ptr().add(i), lasx_xvfadd_d(prod, vb));
        }
        i += 4;
    }
    for j in i..n {
        out[j] = a * x[j] + b;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 多形状（含非 4 的倍数、小于一个向量、空）：**逐位** vs 标量参照（本文件按设计**不依赖**
    /// `simd_dsl` ⇒ 参照在测试里独立写一遍）。
    #[test]
    fn all_ops_are_bitwise_same_as_scalar_reference() {
        for &n in &[0usize, 1, 3, 4, 5, 15, 16, 17, 33, 1029] {
            let x: Vec<f64> = (0..n)
                .map(|i| ((i * 37 % 101) as f64) * 0.1 - 5.05)
                .collect();
            let y: Vec<f64> = (0..n).map(|i| ((i * 53 % 97) as f64) * 0.1 + 1.3).collect();
            let z: Vec<f64> = (0..n).map(|i| ((i * 11 % 89) as f64) * 0.1 - 0.7).collect();
            let bits = |v: &[f64]| v.iter().map(|q| q.to_bits()).collect::<Vec<_>>();
            for op in [BinOpF64::Add, BinOpF64::Sub, BinOpF64::Mul, BinOpF64::Div] {
                let mut got = vec![f64::NAN; n];
                map_binary(&x, &y, &mut got, op);
                // 参照独立写一遍（不调 `kernels::scalar`：本文件按设计不依赖 simd_dsl）。
                let want: Vec<f64> = x
                    .iter()
                    .zip(&y)
                    .map(|(a, b)| match op {
                        BinOpF64::Add => a + b,
                        BinOpF64::Sub => a - b,
                        BinOpF64::Mul => a * b,
                        BinOpF64::Div => a / b,
                    })
                    .collect();
                assert_eq!(bits(&got), bits(&want), "n={n} op={op:?} 不逐位");
            }
            let mut got = vec![f64::NAN; n];
            map_ternary(&x, &y, &z, &mut got);
            let want: Vec<f64> = x
                .iter()
                .zip(&y)
                .zip(&z)
                .map(|((a, b), c)| a - b - c)
                .collect();
            assert_eq!(bits(&got), bits(&want), "n={n} ternary 不逐位");
            let mut got = vec![f64::NAN; n];
            map_affine(0.3, -1.7, &x, &mut got);
            let want: Vec<f64> = x.iter().map(|v| 0.3 * v - 1.7).collect();
            assert_eq!(bits(&got), bits(&want), "n={n} affine 不逐位");
        }
    }
}
