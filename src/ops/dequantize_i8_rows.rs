//! `lasx_dequantize_i8_rows` —— i8 行反量化 `out[r,c] = x[r,c] as f32 · scale[r]`（task-180）。
//!
//! 契约（`simd_dsl` 的 `FusedPlan::DequantizeI8Rows`，见 `src/contract/patterns.tsv`）：
//! 行主序、**每行一个 f32 scale**、逐元素单次乘法。
//!
//! **等价类 = 逐位**：i8→f32 是精确转换；乘法与标量同为一次 IEEE 乘法（scale 的 NaN/±inf
//! 也按同一硬件语义）⇒ `to_bits()` 全等（测试含 ±127/-128/±0/次正规/NaN/±inf scale）。
//!
//! # 为什么需要 lane 重排（纯化前被判"不做"的更正）
//!
//! LASX 没有 i8→f32 直转：要用 `xvsllwil_h_b`（每 128 位 lane 各自拓宽低 8 字节）→
//! `xvsllwil_w_h`（每 lane 低 4 个 i16）两级拓宽。原判据担心 128-bit lane 语义会得到
//! `[b0..b3,b16..b19]` 的乱序；实际可用 **两次重叠的 32B load + `xvpermi_q`/`xvpermi_d`**
//! 把顺序拼自然：每 16 个元素约 2 load + 2×`sllwil_h_b` + 2×`sllwil_w_h` + 1 `permi_d`
//! + 2 `permi_q` + 2 `ffint` + 2 `mul` + 2 store（13 条/16 元素）。
//! 实测（`/tmp/deq_probe.rs`，LoongArch 3.4 load）：**1M 元素 2.2×、8M 1.24×**
//! （vs 标量循环，6 对丢首对中位）⇒ 超 1.2× 门槛，故接（2026-11 复核）。
//!
//! LSX-only CPU **不接**（128 位 lane 更窄，重排成本占比更高）：走同式标量循环。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（分层策略：内核模块在函数级统一写 SAFETY，
// 不逐块重复；策略全文见 `docs/dev.md` §2.1）：
#![allow(clippy::undocumented_unsafe_blocks)]
use crate::arch::SimdPath;
use std::arch::loongarch64::*;

/// 算子入口：LASX 走向量路径，其余（LSX）走同式标量（见模块文档）。
///
/// **调用方保证**：`x/out` 长度 = `rows*cols`、`scale` 长度 = `rows`（adapter 侧
/// `FusedPlan::validate` 已校验）；这里只做 debug 断言。
#[inline]
pub fn dequantize_i8_rows(x: &[i8], rows: usize, cols: usize, scale: &[f32], out: &mut [f32]) {
    debug_assert_eq!(
        x.len(),
        rows * cols,
        "dequantize_i8_rows：x 长度应为 rows*cols"
    );
    debug_assert_eq!(
        out.len(),
        rows * cols,
        "dequantize_i8_rows：out 长度应为 rows*cols"
    );
    debug_assert_eq!(scale.len(), rows, "dequantize_i8_rows：scale 长度应为 rows");
    match SimdPath::detect() {
        SimdPath::Lasx => dequantize_i8_rows_lasx(x, rows, cols, scale, out),
        SimdPath::Lsx => dequantize_i8_rows_scalar(x, rows, cols, scale, out),
    }
}

/// 标量同式（LSX 兜底 + 测试参照）：`out = q as f32 * scale[r]`。
#[inline]
fn dequantize_i8_rows_scalar(x: &[i8], rows: usize, cols: usize, scale: &[f32], out: &mut [f32]) {
    debug_assert_eq!(scale.len(), rows);
    for (r, &s) in scale.iter().enumerate() {
        let base = r * cols;
        for c in 0..cols {
            out[base + c] = x[base + c] as f32 * s;
        }
    }
}

/// LASX 路径：16 元素/迭代（2 个 f32x8），行尾标量。
#[inline]
fn dequantize_i8_rows_lasx(x: &[i8], rows: usize, cols: usize, scale: &[f32], out: &mut [f32]) {
    debug_assert_eq!(scale.len(), rows);
    for (r, &s) in scale.iter().enumerate() {
        // SAFETY: 纯寄存器操作；splat 的位型与标量 `s` 相同（`to_ne_bytes` 保位）。
        let sv: m256 =
            unsafe { std::mem::transmute(lasx_xvreplgr2vr_w(i32::from_ne_bytes(s.to_ne_bytes()))) };
        let base = r * cols;
        let mut c = 0usize;
        // 主循环读 `[base+c, base+c+40)`（两次 32B 读，第二次从 +8 起）⇒ 先留 40 元素余量。
        while c + 40 <= cols {
            // SAFETY: 上面的余量保证两次读都在本行内、本行在 `x` 内；`out` 等长且与 `x` 不重叠；
            // intrinsic 为寄存器/任意对齐访存。
            unsafe {
                let p = x.as_ptr().add(base + c);
                let v0: m256i = lasx_xvld(p, 0); // b0..b31
                let v1: m256i = lasx_xvld(p.add(8), 0); // b8..b39
                // 每 128 位 lane 各自拓宽低 8 字节：h0=[b0..b7,b16..b23]、h1=[b8..b15,b24..b31]
                let h0: m256i = lasx_xvsllwil_h_b::<0>(v0);
                let h1: m256i = lasx_xvsllwil_h_b::<0>(v1);
                // 取两个 lane0 拼自然序：h=[b0..b7 | b8..b15]（i16）
                let h: m256i = lasx_xvpermi_q::<0x02>(h0, h1);
                let w0: m256i = lasx_xvsllwil_w_h::<0>(h); // [b0..b3 | b8..b11]
                // 64 位 lane 交换 [in1,in0,in3,in2] ⇒ 第二组 i32（高半字移到低半）。
                let hs: m256i = lasx_xvpermi_d::<0xb1>(h);
                let w1: m256i = lasx_xvsllwil_w_h::<0>(hs); // [b4..b7 | b12..b15]
                let n0: m256i = lasx_xvpermi_q::<0x02>(w0, w1); // b0..b7
                let n1: m256i = lasx_xvpermi_q::<0x13>(w0, w1); // b8..b15
                // i32 → f32 是精确转换；乘行标量与标量路径同一次 IEEE 乘法 ⇒ 逐位。
                let f0: m256 = lasx_xvffint_s_w(n0);
                let f1: m256 = lasx_xvffint_s_w(n1);
                let o0 = lasx_xvfmul_s(f0, sv);
                let o1 = lasx_xvfmul_s(f1, sv);
                lasx_xvst(
                    std::mem::transmute(o0),
                    out.as_mut_ptr().add(base + c) as *mut i8,
                    0,
                );
                lasx_xvst(
                    std::mem::transmute(o1),
                    out.as_mut_ptr().add(base + c + 8) as *mut i8,
                    0,
                );
            }
            c += 16;
        }
        while c < cols {
            out[base + c] = x[base + c] as f32 * s;
            c += 1;
        }
    }
}

/// 数值回归：**逐位**对照标量同式（含 NaN/±inf/±0 scale 与所有 i8 值）。
#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_ref(x: &[i8], rows: usize, cols: usize, scale: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        dequantize_i8_rows_scalar(x, rows, cols, scale, &mut out);
        out
    }

    #[test]
    fn test_dequantize_i8_rows_bitwise_vs_scalar() {
        // 覆盖：行尾（cols 非 16 倍数）、全 i8 值域、单行/多行
        for &(rows, cols) in &[(1usize, 0usize), (1, 1), (3, 17), (7, 37), (2, 40), (5, 41)] {
            let x: Vec<i8> = (0..rows * cols)
                .map(|i| (((i * 53 + 7) % 251) as i32 - 128) as i8)
                .collect();
            let scale: Vec<f32> = (0..rows).map(|r| 0.001 * (r as f32 + 1.0)).collect();
            let mut out = vec![0.0f32; rows * cols];
            dequantize_i8_rows(&x, rows, cols, &scale, &mut out);
            let want = scalar_ref(&x, rows, cols, &scale);
            for i in 0..out.len() {
                assert_eq!(
                    out[i].to_bits(),
                    want[i].to_bits(),
                    "regular r={rows} c={cols} i={i}"
                );
            }
        }
        // 全 i8 值域 × 边界 scale
        let cols = 256usize;
        let x: Vec<i8> = (0..cols).map(|i| (i as i32 - 128) as i8).collect();
        let scale = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
            0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
        ];
        for (r, &s) in scale.iter().enumerate() {
            let scale = vec![s];
            let mut out = vec![0.0f32; cols];
            dequantize_i8_rows(&x, 1, cols, &scale, &mut out);
            let want = scalar_ref(&x, 1, cols, &scale);
            for i in 0..cols {
                if want[i].is_nan() {
                    assert!(out[i].is_nan(), "edge scale {s:?} i={i} 期望 NaN");
                } else {
                    assert_eq!(
                        out[i].to_bits(),
                        want[i].to_bits(),
                        "edge scale {s:?} i={i}"
                    );
                }
            }
            let _ = r;
        }
    }
}
