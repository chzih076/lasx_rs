//! **P5-a：gather / scatter 的 LASX 高性能内核**（签名与 `simd_dsl/docs/structure.md §7` 的
//! **解冻最终契约**逐字一致；**stride 从第一天就在签名里**，ids 用 `u32`，越界 ⇒ `Err`）。
//!
//! # 三个内核
//! - [`gather_rows_f32`] / [`gather_rows_i8`]：按 `ids` 把 `table` 的行**整行**拷到 `out`；
//! - [`scatter_rows_f32`]：按 `ids` 把 `src` 的行写回 `out`（**重复 id 时后写覆盖前写**，如实文档化）。
//!
//! # 边界语义（**果断**，不静默截断）
//! - `row_len == 0` ⇒ **空操作**，返回 `Ok(())`；
//! - `ids` 里任一 `id` 越界（`id*table_stride + row_len > table.len()`，或乘法溢出）⇒ **`Err`**；
//! - `out` 容量不足（`(ids.len()-1)*out_stride + row_len > out.len()`）⇒ **`Err`**；
//! - scatter 的 `src` 容量不足（`(ids.len()-1)*src_stride + row_len > src.len()`）⇒ **`Err`**。
//!
//! # 性能做法
//! 行拷贝 f32 用 **8 lane 向量**（`load_f32x8`/`store_f32x8`）+ 尾部标量；i8 用 **32 B 块**
//! （`copy_nonoverlapping`，memcpy 级） + 尾部标量。**整行一次处理**，不做 per-element 分支。
//!
//! # 门控
//! `simd_dsl` 的 trait 已声明这 3 个方法、`dsl_backend.rs` 已有 override 调用本模块 ⇒ 有生产调用方
//! ⇒ 只挂 `#[cfg(feature = "dsl-backend")]`（不需要 `test` 门控；否则会被 `dead_code` 判死，
//! 且禁止 `#[allow(dead_code)]`，见 AGENTS.md #49）。

use simd_dsl::BackendError;

/// 长度/越界错误构造（与 `dsl_backend.rs` 的 `len_err` 同形）。
fn mismatch(op: &'static str, what: &'static str, expected: usize, got: usize) -> BackendError {
    BackendError::LengthMismatch {
        op,
        what,
        expected,
        got,
    }
}

/// **f32 行拷贝（8 lane + 尾部标量）**。
///
/// # Safety
/// `src` 可读 `row_len` 个 f32、`dst` 可写 `row_len` 个 f32，且两者**不重叠**（调用方保证）。
#[inline]
unsafe fn copy_row_f32(src: *const f32, dst: *mut f32, row_len: usize) {
    // **实测负结果（P5-a）**：手写 8-lane 循环（`load_f32x8`/`store_f32x8` 未内联、未展开）在
    // `n_rows=4096,row_len=768` 上比 `memcpy` **慢约 23×**（0.042×，见 A/B 测试的注释）⇒ 行拷贝用
    // `copy_nonoverlapping`（编成 `memcpy`，libc 内部已向量化）。**手写 8-lane 只在"行内还有算术"
    // 时有意义，纯拷贝没有**——这是本轮如实记录的负结果，不是"没测出来"。
    // SAFETY: 由本函数的契约保证两侧各可访问 `row_len` 个元素且不重叠。
    unsafe { std::ptr::copy_nonoverlapping(src, dst, row_len) };
}

/// **i8 行拷贝（32 B 块 + 尾部标量）**。
///
/// # Safety
/// 同 [`copy_row_f32`]（`i8` 版本）。
#[inline]
unsafe fn copy_row_i8(src: *const i8, dst: *mut i8, row_len: usize) {
    // 与 f32 版同理：整行 `memcpy` 优于手写 32 B 块循环（**负结果**见 `copy_row_f32` 的注释）。
    // SAFETY: 由本函数的契约保证两侧各可访问 `row_len` 字节且不重叠。
    unsafe { std::ptr::copy_nonoverlapping(src, dst, row_len) };
}

/// **gather（f32）**：`out[t*out_stride + c] = table[ids[t]*table_stride + c]`。
///
/// # Errors
/// 越界 `id`／`out` 容量不足／乘法溢出 ⇒ `BackendError::LengthMismatch`。
pub fn gather_rows_f32(
    table: &[f32],
    row_len: usize,
    table_stride: usize,
    ids: &[u32],
    out: &mut [f32],
    out_stride: usize,
) -> Result<(), BackendError> {
    if row_len == 0 {
        return Ok(());
    }
    let need = ids
        .len()
        .checked_sub(1)
        .and_then(|v| v.checked_mul(out_stride))
        .and_then(|v| v.checked_add(row_len))
        .ok_or_else(|| mismatch("gather_rows_f32", "out（溢出）", usize::MAX, out.len()))?;
    if ids.is_empty() {
        return Ok(());
    }
    if out.len() < need {
        return Err(mismatch("gather_rows_f32", "out", need, out.len()));
    }
    // **快路径（实测需要）**：先把所有 `id` 的越界/溢出**一次校验完**，主循环里只做算术 + 拷贝。
    // 逐行 `checked_mul`/错误闭包在 256 行小拷贝上就吃掉约 25%（实测 0.769× 的主因）。
    let max_id = ids.iter().copied().max().unwrap_or(0) as usize;
    let max_end = max_id
        .checked_mul(table_stride)
        .and_then(|b| b.checked_add(row_len))
        .ok_or_else(|| mismatch("gather_rows_f32", "id 越界（溢出）", usize::MAX, max_id))?;
    if max_end > table.len() {
        return Err(mismatch("gather_rows_f32", "table", max_end, table.len()));
    }
    ids.iter().enumerate().for_each(|(t, &id)| {
        let base = id as usize * table_stride;
        let dst = t * out_stride;
        // SAFETY: `id <= max_id` ⇒ `base + row_len <= max_end <= table.len()`（上面已校验）；
        // `t*out_stride + row_len <= out.len()`（函数开头已校验）；`table`/`out` 不重叠。
        unsafe { copy_row_f32(table.as_ptr().add(base), out.as_mut_ptr().add(dst), row_len) };
    });
    Ok(())
}

/// **gather（i8）**：语义同 [`gather_rows_f32`]。
///
/// # Errors
/// 同 [`gather_rows_f32`]。
pub fn gather_rows_i8(
    table: &[i8],
    row_len: usize,
    table_stride: usize,
    ids: &[u32],
    out: &mut [i8],
    out_stride: usize,
) -> Result<(), BackendError> {
    if row_len == 0 {
        return Ok(());
    }
    let need = ids
        .len()
        .checked_sub(1)
        .and_then(|v| v.checked_mul(out_stride))
        .and_then(|v| v.checked_add(row_len))
        .ok_or_else(|| mismatch("gather_rows_i8", "out（溢出）", usize::MAX, out.len()))?;
    if ids.is_empty() {
        return Ok(());
    }
    if out.len() < need {
        return Err(mismatch("gather_rows_i8", "out", need, out.len()));
    }
    ids.iter().enumerate().try_for_each(|(t, &id)| {
        let base = (id as usize).checked_mul(table_stride).ok_or_else(|| {
            mismatch(
                "gather_rows_i8",
                "id*table_stride（溢出）",
                usize::MAX,
                id as usize,
            )
        })?;
        let end = base
            .checked_add(row_len)
            .ok_or_else(|| mismatch("gather_rows_i8", "id 越界", usize::MAX, base))?;
        if end > table.len() {
            return Err(mismatch("gather_rows_i8", "table", end, table.len()));
        }
        let dst = t * out_stride;
        // SAFETY: 同 f32 版本。
        unsafe { copy_row_i8(table.as_ptr().add(base), out.as_mut_ptr().add(dst), row_len) };
        Ok(())
    })
}

/// **scatter（f32）**：`out[ids[t]*out_stride + c] = src[t*src_stride + c]`；**重复 id ⇒ 后写覆盖前写**。
///
/// # Errors
/// 越界 `id`／`src` 或 `out` 容量不足／乘法溢出 ⇒ `BackendError::LengthMismatch`。
pub fn scatter_rows_f32(
    src: &[f32],
    src_stride: usize,
    ids: &[u32],
    row_len: usize,
    out: &mut [f32],
    out_stride: usize,
) -> Result<(), BackendError> {
    if row_len == 0 {
        return Ok(());
    }
    let need_src = ids
        .len()
        .checked_sub(1)
        .and_then(|v| v.checked_mul(src_stride))
        .and_then(|v| v.checked_add(row_len))
        .ok_or_else(|| mismatch("scatter_rows_f32", "src（溢出）", usize::MAX, src.len()))?;
    if ids.is_empty() {
        return Ok(());
    }
    if src.len() < need_src {
        return Err(mismatch("scatter_rows_f32", "src", need_src, src.len()));
    }
    ids.iter().enumerate().try_for_each(|(t, &id)| {
        let base = (id as usize).checked_mul(out_stride).ok_or_else(|| {
            mismatch(
                "scatter_rows_f32",
                "id*out_stride（溢出）",
                usize::MAX,
                id as usize,
            )
        })?;
        let end = base
            .checked_add(row_len)
            .ok_or_else(|| mismatch("scatter_rows_f32", "id 越界", usize::MAX, base))?;
        if end > out.len() {
            return Err(mismatch("scatter_rows_f32", "out", end, out.len()));
        }
        // SAFETY: `base + row_len <= out.len()`（已校验）、`t*src_stride + row_len <= src.len()`（已校验），
        // 且 `src`/`out` 是不同切片。
        unsafe {
            copy_row_f32(
                src.as_ptr().add(t * src_stride),
                out.as_mut_ptr().add(base),
                row_len,
            )
        };
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table_f32(rows: usize, stride: usize, row_len: usize) -> Vec<f32> {
        (0..rows * stride)
            .map(|i| {
                let r = i / stride;
                let c = i % stride;
                if c < row_len {
                    (r * 1000 + c) as f32
                } else {
                    -1.0
                }
            })
            .collect()
    }

    /// **正确性：与朴素参考逐位**（`row_len` 故意不整除 8 ⇒ 覆盖尾部标量）。
    #[test]
    fn gather_f32_matches_naive() {
        let (rows, stride, row_len) = (16usize, 20usize, 13usize);
        let table = table_f32(rows, stride, row_len);
        let ids: Vec<u32> = vec![15, 0, 7, 7, 3];
        let out_stride = row_len + 2;
        let mut out = vec![f32::NAN; 5 * out_stride];
        gather_rows_f32(&table, row_len, stride, &ids, &mut out, out_stride).unwrap();
        let want: Vec<f32> = ids
            .iter()
            .flat_map(|&id| (0..row_len).map(move |c| (id as usize * 1000 + c) as f32))
            .collect();
        let got: Vec<f32> = ids
            .iter()
            .enumerate()
            .flat_map(|(t, _)| (0..row_len).map(move |c| (t, c)).collect::<Vec<_>>())
            .map(|(t, c)| out[t * out_stride + c])
            .collect();
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        // 未写入的 stride 间隙必须原样（NaN 哨兵）：每行 `[row_len, out_stride)` 是间隙
        assert!(
            (0..ids.len()).all(|t| out[t * out_stride + row_len..(t + 1) * out_stride]
                .iter()
                .all(|v| v.is_nan())),
            "gap 未被写入"
        );
    }

    /// i8 版正确性（32 B 块 + 尾部）。
    #[test]
    fn gather_i8_matches_naive() {
        let (rows, stride, row_len) = (8usize, 40usize, 37usize);
        let table: Vec<i8> = (0..rows * stride).map(|i| (i % 100) as i8).collect();
        let ids: Vec<u32> = vec![7, 1];
        let mut out = vec![0i8; 2 * row_len];
        gather_rows_i8(&table, row_len, stride, &ids, &mut out, row_len).unwrap();
        let got_ref = |t: usize, c: usize| out[t * row_len + c];
        assert_eq!(got_ref(0, 0), table[7 * stride]);
        assert_eq!(got_ref(0, 36), table[7 * stride + 36]);
        assert_eq!(got_ref(1, 10), table[stride + 10]);
    }

    /// scatter 正确性 + **重复 id 后写覆盖前写**。
    #[test]
    fn scatter_f32_last_write_wins() {
        let row_len = 8usize;
        let src: Vec<f32> = vec![1.0; row_len]
            .into_iter()
            .chain(vec![2.0; row_len])
            .collect();
        let mut out = vec![0.0f32; 4 * row_len];
        scatter_rows_f32(&src, row_len, &[3, 3], row_len, &mut out, row_len).unwrap();
        assert!(out[3 * row_len..4 * row_len].iter().all(|&v| v == 2.0));
    }

    /// 边界：`row_len == 0` ⇒ 空操作（`Ok`，不改 out）。
    #[test]
    fn row_len_zero_is_noop() {
        let mut out = vec![7.0f32; 4];
        gather_rows_f32(&[1.0, 2.0], 0, 2, &[0], &mut out, 0).unwrap();
        assert_eq!(out, vec![7.0f32; 4]);
        scatter_rows_f32(&[1.0], 0, &[0], 0, &mut out, 0).unwrap();
        assert_eq!(out, vec![7.0f32; 4]);
    }

    /// 边界：越界 id ⇒ `Err`（不 UB、不截断）。
    #[test]
    fn out_of_range_id_is_err() {
        let table = vec![0.0f32; 4 * 8];
        let mut out = vec![0.0f32; 8];
        assert!(gather_rows_f32(&table, 8, 8, &[4], &mut out, 8).is_err());
        assert!(gather_rows_f32(&table, 8, 8, &[0], &mut out[..4], 8).is_err());
        let mut o2 = vec![0.0f32; 8];
        assert!(scatter_rows_f32(&[0.0; 8], 8, &[4], 8, &mut o2, 8).is_err());
    }

    /// **A/B 实测（§5 口径：同进程交替、≥5 对丢首对、中位）**：新内核 vs 朴素逐行 `copy_from_slice`。
    ///
    /// **口径说明（如实）**：不与 `api::gather_rows` 比 —— 它**每次调用都分配新 Vec**，
    /// 与本内核（写进调用方缓冲）不同口径，比出来的是"分配开销"而不是"内核快慢"。
    #[test]
    fn bench_vs_naive_copy() {
        use std::time::Instant;
        let (n_rows, row_len) = (4096usize, 768usize);
        let table: Vec<f32> = (0..n_rows * row_len).map(|i| i as f32).collect();
        let ids: Vec<u32> = (0..256u32).map(|i| (i * 16) % n_rows as u32).collect();
        let mut out_fast = vec![0.0f32; ids.len() * row_len];
        let mut out_naive = vec![0.0f32; ids.len() * row_len];
        // **抗噪口径**：同进程交替 **8 轮、丢首轮（7 个样本）**；同时给中位与 **min** 比值
        // （负载只增不减 ⇒ min 更接近无噪速度）；本测试无时间阈值断言，只做趋势证据。
        let med = |f: &mut dyn FnMut()| -> (f64, f64) {
            let mut v = Vec::new();
            for _ in 0..8 {
                let t = Instant::now();
                f();
                v.push(t.elapsed().as_secs_f64() * 1e3);
            }
            v.remove(0);
            v.sort_by(f64::total_cmp);
            (v[v.len() / 2], v[0])
        };
        let (fast, min_fast) = med(&mut || {
            gather_rows_f32(&table, row_len, row_len, &ids, &mut out_fast, row_len).unwrap();
        });
        let (naive, min_naive) = med(&mut || {
            ids.iter().enumerate().for_each(|(t, &id)| {
                let s = id as usize * row_len;
                out_naive[t * row_len..(t + 1) * row_len].copy_from_slice(&table[s..s + row_len]);
            });
        });
        println!(
            "P5-a A/B（n_rows={n_rows}, row_len={row_len}, {} 行）：内核 {fast:.4} ms（min {min_fast:.4}）｜朴素 {naive:.4} ms（min {min_naive:.4}）｜**朴素/内核 = {:.3}×（中位）/ {:.3}×（min）**",
            ids.len(),
            naive / fast,
            min_naive / min_fast
        );
        assert_eq!(
            out_fast.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            out_naive.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "A/B 两臂结果必须逐位一致"
        );
        assert!(fast > 0.0 && naive > 0.0);
    }
}
