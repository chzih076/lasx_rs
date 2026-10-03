//! `lasx_gather_rows` —— 按 id 取行（嵌入表查找）：`out[i] = table[ids[i]]`。
//!
//! **这个算子不用 LASX 手写，是有意的**：它不做算术，只做行拷贝（典型行宽 384 个 f32 = 1.5 KB），
//! 而 [`slice::copy_from_slice`] 就是 `memcpy`——LLVM 会把它编成向量化的块拷贝，**比手写
//! LASX 循环更快也更容易验证**（无需处理列尾、无需位精确论证：拷贝没有舍入）。
//! 所以本模块没有 `F32x8`，也没有降级分支（它不依赖任何 ISA 扩展）。
//!
//! 存在理由：外部契约（ONNX 图上 `Gather` 34 处：词/段/位置嵌入），
//! 以及 `docs/platform.md` §4.2 对"怎么取"的决策：库侧给两种口径——
//!
//! - [`gather_rows`]：f32 表，纯行拷贝；
//! - [`gather_rows_i8`]：int8 表 + **每行**一个 scale，取行时就地反量化
//!   （`out[i][j] = table[ids[i]][j] as f32 * scales[ids[i]]`）。
//!
//! **"嵌入表存 f32 还是 int8"不是本库的决定**（见 §4.2）：这里只保证两种口径都正确、
//! 边界的责任清楚。id 越界在 `api`/`ffi` 层被拒（`docs/ops.md` §2.17）。
//!
//! # 数值契约
//!
//! ```text
//! out[i * row_len + j] = table[ids[i] * row_len + j]                        // f32 口径：逐位拷贝
//! out[i * row_len + j] = table[ids[i] * row_len + j] as f32 * scales[ids[i]]  // int8 口径
//! ```
//!
//! 两条都**没有累加**，所以不存在结合次序问题：给定 `table`/`ids`/`scales`，输出是确定的
//! 逐元素函数（int8 口径是"一次转换 + 一次乘法"，两次舍入都没有）。

/// f32 表的按 id 取行（行主序，`row_len` 个 f32 一行）。
///
/// # Panics
/// 调用方（`api`/`ffi`）已保证：`ids` 里每个 id 都 `< table.len() / row_len`、
/// `out.len() == ids.len() * row_len`；这里只做 debug 断言。
pub(crate) fn gather_rows(table: &[f32], ids: &[usize], row_len: usize, out: &mut [f32]) {
    debug_assert!(row_len > 0);
    debug_assert_eq!(out.len(), ids.len() * row_len);
    debug_assert_eq!(table.len() % row_len, 0);
    for (i, &id) in ids.iter().enumerate() {
        debug_assert!(id * row_len + row_len <= table.len());
        out[i * row_len..(i + 1) * row_len]
            .copy_from_slice(&table[id * row_len..id * row_len + row_len]);
    }
}

/// int8 表 + **每行** scale 的按 id 取行并反量化。
///
/// `scales` 长度 = 表行数（每行一个 scale，`docs/ops.md` §2.12 的 per-row 约定）。
///
/// # Panics
/// 同 [`gather_rows`]，外加 `scales.len() == table.len() / row_len`；这里只做 debug 断言。
pub(crate) fn gather_rows_i8(
    table: &[i8],
    scales: &[f32],
    ids: &[usize],
    row_len: usize,
    out: &mut [f32],
) {
    debug_assert!(row_len > 0);
    debug_assert_eq!(out.len(), ids.len() * row_len);
    debug_assert_eq!(table.len() % row_len, 0);
    debug_assert_eq!(scales.len(), table.len() / row_len);
    for (i, &id) in ids.iter().enumerate() {
        debug_assert!(id * row_len + row_len <= table.len());
        let s = scales[id];
        let src = &table[id * row_len..id * row_len + row_len];
        let dst = &mut out[i * row_len..(i + 1) * row_len];
        for (d, &q) in dst.iter_mut().zip(src) {
            *d = q as f32 * s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f32 口径：与"逐元素按 id 索引"的参考**逐位一致**，覆盖重复 id、逆序 id、
    /// 行宽非 8 的倍数、`ids` 为空、单行。
    #[test]
    fn test_gather_rows_matches_indexing_bit_for_bit() {
        let rows = 7usize;
        for &row_len in &[1usize, 3, 8, 17, 384] {
            let mut rng = crate::ops::testutil::Lcg((row_len * 7) as u64 ^ 0x6a11);
            let table: Vec<f32> = (0..rows * row_len)
                .map(|_| rng.f64() as f32 - 0.5)
                .collect();
            for ids in [
                vec![],
                vec![0usize],
                vec![6],
                vec![3, 3, 3],
                vec![6, 0, 5, 1],
                (0..rows).rev().collect::<Vec<_>>(),
            ] {
                let mut got = vec![0f32; ids.len() * row_len];
                gather_rows(&table, &ids, row_len, &mut got);
                for (i, &id) in ids.iter().enumerate() {
                    for j in 0..row_len {
                        assert_eq!(
                            got[i * row_len + j].to_bits(),
                            table[id * row_len + j].to_bits(),
                            "row_len={row_len} id={id} i={i} j={j}"
                        );
                    }
                }
            }
        }
    }

    /// int8 口径：`q as f32 * s`（一次转换 + 一次乘法，无累加 ⇒ 无结合次序问题）。
    #[test]
    fn test_gather_rows_i8_dequantizes_per_row() {
        let (rows, row_len) = (5usize, 9usize);
        let table: Vec<i8> = (0..rows * row_len)
            .map(|i| ((i as i32 * 37) % 255 - 127) as i8)
            .collect();
        let scales: Vec<f32> = vec![0.5, 0.25, 1.0, 0.125, 2.0];
        let ids = vec![4usize, 0, 2, 4];
        let mut got = vec![0f32; ids.len() * row_len];
        gather_rows_i8(&table, &scales, &ids, row_len, &mut got);
        for (i, &id) in ids.iter().enumerate() {
            for j in 0..row_len {
                let want = table[id * row_len + j] as f32 * scales[id];
                assert_eq!(
                    got[i * row_len + j].to_bits(),
                    want.to_bits(),
                    "id={id} i={i} j={j}"
                );
            }
        }
        // 负零/符号：`0i8 as f32 * (-1.0)` 是 `-0.0`（`f32` 乘法保符号）——逐位断言它
        let t0 = [0i8];
        let s0 = [-1.0f32];
        let mut o = [9f32];
        gather_rows_i8(&t0, &s0, &[0], 1, &mut o);
        assert_eq!(o[0].to_bits(), (-0.0f32).to_bits());
    }

    /// 表尾行（`id = 最后一行`）恰好贴边：这是最容易写出"少拷/多拷一个元素"的位置。
    #[test]
    fn test_gather_rows_last_row_edge() {
        let row_len = 4usize;
        let table: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let mut got = [0f32; 4];
        gather_rows(&table, &[2], row_len, &mut got);
        assert_eq!(got, [8.0, 9.0, 10.0, 11.0]);
    }
}
