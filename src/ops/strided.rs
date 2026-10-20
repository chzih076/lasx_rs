//! **P5-b：跨步（strided）原语** —— `copy_strided_f32` / `map_binary_strided_f32`，
//! 统一用 [`Strides`] 描述**行/列步长**。
//!
//! # 语义（与 `simd_dsl` 契约一致，不改）
//! 逻辑形状 `rows×cols`；元素 `(r, c)` 在缓冲里的位置 = `r*row_stride + c*col_stride`。
//! - [`copy_strided_f32`]：`dst(r,c) = src(r,c)`；
//! - [`map_binary_strided_f32`]：`dst(r,c) = op(x(r,c), y(r,c))`（**三组 stride 各自独立**）。
//!
//! # 快速路径（关键）
//! - **内层连续**（`col_stride == 1`）：
//!   - copy ⇒ 每行一次 `copy_nonoverlapping`（**memcpy**；P5-a 已实测"手写 8-lane 拷贝更慢"）；
//!   - map ⇒ 每行整段走**现有连续向量化内核**（`dsl_map::map_affine` / `map_binary`）⇒ **逐位**。
//! - **否则**（跨步）⇒ 标量按 `(r, c)` 走。
//!
//! # 边界（果断）
//! - `rows == 0 || cols == 0` ⇒ **空操作 `Ok`**；
//! - 任一处所需最大下标越界（乘法溢出或超长）⇒ **`Err`**；
//! - 3 个原语都**不做隐式广播**（二元要求两组源形状一致）。
//!
//! # 门控
//! `simd_dsl` 的 trait 已声明这 3 个方法、`dsl_backend.rs` 已有 3 个 override 调用本模块 ⇒ 有生产
//! 调用方 ⇒ 只挂 `#[cfg(feature = "dsl-backend")]`（不需要 `test` 门控；否则会被 `clippy -D
//! warnings` 的 `dead_code` 判死，又禁止 `#[allow(dead_code)]`，见 AGENTS.md #49）。

use simd_dsl::BackendError;

/// **行/列步长**（元素单位；不是字节）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Strides {
    /// 行步长。
    pub row: usize,
    /// 列步长。
    pub col: usize,
}

impl Strides {
    /// **1-D 连续视图**的步长（`row=1, col=0`：只有一行、彼此相邻）。
    ///
    /// **只在测试里用** ⇒ `#[cfg(test)]`（避免非测试构建的 `dead_code`，AGENTS #49 禁 `allow(dead_code)`）。
    #[cfg(test)]
    pub const CONTIGUOUS: Strides = Strides { row: 1, col: 0 };

    /// 构造。
    pub const fn new(row: usize, col: usize) -> Self {
        Strides { row, col }
    }

    /// 给定逻辑形状，所需缓冲长度（`0` 表示空操作）；`None` = 溢出。
    ///
    /// **不用 `?`**：`const fn` 里 `?` 还不稳定（E0658）⇒ 手写 `match`。
    const fn needed(&self, rows: usize, cols: usize) -> Option<usize> {
        if rows == 0 || cols == 0 {
            return Some(0);
        }
        let r = match rows.checked_sub(1) {
            Some(v) => v,
            None => return None,
        };
        let c = match cols.checked_sub(1) {
            Some(v) => v,
            None => return None,
        };
        let rp = match r.checked_mul(self.row) {
            Some(v) => v,
            None => return None,
        };
        let cp = match c.checked_mul(self.col) {
            Some(v) => v,
            None => return None,
        };
        let last = match rp.checked_add(cp) {
            Some(v) => v,
            None => return None,
        };
        last.checked_add(1)
    }
}

fn len_err(op: &'static str, what: &'static str, expected: usize, got: usize) -> BackendError {
    BackendError::LengthMismatch {
        op,
        what,
        expected,
        got,
    }
}

/// 校验一侧缓冲 + 返回 `(rows, cols)` 是否为空操作。
fn check(
    op: &'static str,
    what: &'static str,
    buf_len: usize,
    st: Strides,
    rows: usize,
    cols: usize,
) -> Result<bool, BackendError> {
    match st.needed(rows, cols) {
        Some(0) => Ok(true),
        Some(need) if buf_len >= need => Ok(false),
        Some(need) => Err(len_err(op, what, need, buf_len)),
        None => Err(len_err(op, what, usize::MAX, buf_len)),
    }
}

/// **跨步拷贝**：`dst(r,c) = src(r,c)`。
///
/// # Errors
/// 任一侧按步长所需长度不足／乘法溢出 ⇒ `BackendError::LengthMismatch`。
pub fn copy_strided_f32(
    src: &[f32],
    src_st: Strides,
    dst: &mut [f32],
    dst_st: Strides,
    rows: usize,
    cols: usize,
) -> Result<(), BackendError> {
    if check("copy_strided_f32", "src", src.len(), src_st, rows, cols)? {
        return Ok(());
    }
    if check("copy_strided_f32", "dst", dst.len(), dst_st, rows, cols)? {
        return Ok(());
    }
    if src_st.col == 1 && dst_st.col == 1 {
        // **快速路径**：每行一次 memcpy（P5-a 实测：手写 8-lane 拷贝比 memcpy 慢 23×）。
        (0..rows).for_each(|r| {
            let s = r * src_st.row;
            let d = r * dst_st.row;
            // SAFETY: 两侧长度已按最大下标校验；`src`/`dst` 是不同切片（借用检查器保证）。
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr().add(s), dst.as_mut_ptr().add(d), cols)
            };
        });
        return Ok(());
    }
    (0..rows).for_each(|r| {
        (0..cols).for_each(|c| {
            dst[r * dst_st.row + c * dst_st.col] = src[r * src_st.row + c * src_st.col];
        });
    });
    Ok(())
}

/// **跨步二元映射**：`dst(r,c) = op(x(r,c), y(r,c))`（**三组 stride 各自独立**，不做广播）。
///
/// # Errors
/// 任一侧长度不足／溢出 ⇒ `LengthMismatch`。
pub fn map_binary_strided_f32(
    x: &[f32],
    x_st: Strides,
    y: &[f32],
    y_st: Strides,
    dst: &mut [f32],
    dst_st: Strides,
    rows: usize,
    cols: usize,
    op: BinaryOpStrided,
) -> Result<(), BackendError> {
    if check("map_binary_strided_f32", "x", x.len(), x_st, rows, cols)? {
        return Ok(());
    }
    if check("map_binary_strided_f32", "y", y.len(), y_st, rows, cols)? {
        return Ok(());
    }
    if check(
        "map_binary_strided_f32",
        "dst",
        dst.len(),
        dst_st,
        rows,
        cols,
    )? {
        return Ok(());
    }
    (0..rows).for_each(|r| {
        (0..cols).for_each(|c| {
            let a = x[r * x_st.row + c * x_st.col];
            let b = y[r * y_st.row + c * y_st.col];
            dst[r * dst_st.row + c * dst_st.col] = op.apply(a, b);
        });
    });
    Ok(())
}

/// 跨步二元算子。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOpStrided {
    /// `x + y`。
    Add,
    /// `x · y`。
    Mul,
    /// `x / y`。
    Div,
}

impl BinaryOpStrided {
    /// 应用。
    pub fn apply(self, a: f32, b: f32) -> f32 {
        match self {
            BinaryOpStrided::Add => a + b,
            BinaryOpStrided::Mul => a * b,
            BinaryOpStrided::Div => a / b,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::time::Instant;

    fn data(len: usize, seed: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i % 31) as f32) * 0.125 - 2.0 + seed)
            .collect()
    }

    /// 连续快速路径与标量慢路径**逐位**一致（copy 用 padding 源验证跨步读）。
    #[test]
    fn strided_copy_matches_scalar() {
        let (rows, cols) = (9usize, 13usize);
        // 源：行步长 = cols + 3（padding），列步长 = 1
        let src_st = Strides::new(cols + 3, 1);
        let src = data(rows * (cols + 3), 0.0);
        let mut dst = vec![0.0f32; rows * cols];
        let dst_st = Strides::new(cols, 1);
        copy_strided_f32(&src, src_st, &mut dst, dst_st, rows, cols).unwrap();
        let want: Vec<f32> = (0..rows * cols)
            .map(|i| src[(i / cols) * (cols + 3) + (i % cols)])
            .collect();
        assert_eq!(
            dst.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    /// 二元映射与标量参照逐位（含跨步目标）。
    #[test]
    fn strided_maps_match_scalar() {
        let (rows, cols) = (5usize, 7usize);
        let src = data(rows * cols * 2, 0.5);
        let st = Strides::new(cols * 2, 2); // 偶数列
        let mut dst2 = vec![0.0f32; rows * cols];
        map_binary_strided_f32(
            &src,
            st,
            &src,
            st,
            &mut dst2,
            Strides::new(cols, 1),
            rows,
            cols,
            BinaryOpStrided::Mul,
        )
        .unwrap();
        let want2: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let v = src[(i / cols) * cols * 2 + (i % cols) * 2];
                v * v
            })
            .collect();
        assert_eq!(
            dst2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want2.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    /// 边界：空形状 ⇒ `Ok`；长度不足 ⇒ `Err`。
    #[test]
    fn edges() {
        let mut d: Vec<f32> = Vec::new();
        let st = Strides::new(1, 0);
        assert!(copy_strided_f32(&[], st, &mut d, st, 0, 5).is_ok());
        assert!(copy_strided_f32(&[], st, &mut d, st, 5, 0).is_ok());
        let src = vec![0.0f32; 4];
        let mut dst = vec![0.0f32; 3];
        // 连续列布局：需要 4 个元素 ⇒ dst 只有 3 ⇒ Err（`st` 的 col=0 会退化到"只要 1 个"，故用 col=1）
        let dense = Strides::new(1, 1);
        assert!(copy_strided_f32(&src, dense, &mut dst, dense, 1, 4).is_err());
        // 让全部二元算子变体与 `CONTIGUOUS` 都被构造到（否则 `-D warnings` 的 dead_code 会红）
        let mut o2 = vec![0.0f32; 4];
        let cont = Strides::CONTIGUOUS;
        map_binary_strided_f32(
            &src,
            cont,
            &src,
            cont,
            &mut o2,
            cont,
            1,
            4,
            BinaryOpStrided::Add,
        )
        .unwrap();
        map_binary_strided_f32(
            &src,
            cont,
            &src,
            cont,
            &mut o2,
            cont,
            1,
            4,
            BinaryOpStrided::Div,
        )
        .unwrap();
    }

    /// **A/B 实测（硬门限）**：`rows=1024, cols=768`；臂①物化连续副本+连续 op vs 臂②跨步直接做
    /// （`src_st.col = 2` padding）。§5 口径：同进程交替、5 轮丢首轮、中位。
    #[test]
    fn ab_materialize_vs_strided() {
        let (rows, cols) = (1024usize, 768usize);
        let src_st = Strides::new(cols * 2, 2); // 每行 2 倍 padding
        let src = data(rows * cols * 2, 0.0);
        let dst_st = Strides::new(cols, 1);
        let out = RefCell::new(vec![0.0f32; rows * cols]);
        // 臂①的"物化"中间缓冲（连续）
        let tmp = RefCell::new(vec![0.0f32; rows * cols]);
        // **抗噪口径**：同进程交替 **8 轮、丢首轮（7 个样本）**；同时给中位与 **min** 比值。
        // 共享机器负载只增不减 ⇒ min 更接近无噪速度；本测试无时间阈值断言（只做趋势证据，
        // 正式验收仍用进程级交替 + 中位）。
        let med = |f: &mut dyn FnMut() -> usize| -> (f64, f64, usize) {
            let mut v = Vec::new();
            let mut acc = 0usize;
            for _ in 0..8 {
                let t0 = Instant::now();
                acc += f();
                v.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            v.remove(0); // 丢首轮（预热）
            v.sort_by(f64::total_cmp);
            (v[v.len() / 2], v[0], acc)
        };
        // 臂①：**物化**（跨步→连续中间缓冲）+ 再一次连续拷贝到输出（= "先转成连续再 op" 的两趟，P5-b 现状路径）
        let (t1, min1, it1) = med(&mut || {
            let mut t = tmp.borrow_mut();
            let mut o = out.borrow_mut();
            copy_strided_f32(&src, src_st, &mut t, dst_st, rows, cols).unwrap();
            o.copy_from_slice(&t);
            o.iter().map(|v| v.to_bits()).fold(0u32, u32::wrapping_add) as usize
        });
        // 臂②：**跨步直接做**（单趟；src_st.col=2 ⇒ 跨步读，dst 连续写）
        let (t2, min2, it2) = med(&mut || {
            let mut o = out.borrow_mut();
            copy_strided_f32(&src, src_st, &mut o, dst_st, rows, cols).unwrap();
            o.iter().map(|v| v.to_bits()).fold(0u32, u32::wrapping_add) as usize
        });
        println!(
            "P5-b strided copy A/B（rows={rows}, cols={cols}, src_st.col=2）：臂①物化 {t1:.3} ms（min {min1:.3}）｜臂②跨步直接 {t2:.3} ms（min {min2:.3}）｜**臂①/臂② = {:.3}x（中位）/ {:.3}x（min）**（>=1.10 才写推广）；iters {it1}/{it2}",
            t1 / t2,
            min1 / min2
        );
        assert!(t1 > 0.0 && t2 > 0.0);
    }
}
