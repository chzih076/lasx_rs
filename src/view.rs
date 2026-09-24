//! 矩阵视图：把"形状 + 跨距"和数据放在一起，**构造时校验一次**，之后所有算子只认这一个对象。
//!
//! # 这一层解决什么问题
//!
//! 库里的 C ABI 和大部分算子长这样：
//!
//! ```text
//! lasx_matmul(m, k, n, a, b, c)     // m、k、n 是三个裸数字
//! ```
//!
//! 三个数字谁是谁、跟 `a`/`b`/`c` 的长度对不对得上，全靠调用方记。本仓库里**真的踩过**
//! 这些坑：把对齐后的列数 `n32` 当成行跨距传进微内核（整个矩阵行错位）、把"既是循环上界
//! 又是 B 的行跨距"的那一个参数交换掉（结果全错）、基准里 `(m, k, n)` 顺序写反（读数没意义）。
//!
//! 换成视图之后，"长度和形状对不对"在**构造那一刻**就查完了，而且报错直接说清是哪个矩阵、
//! 期望多少、实际多少：
//!
//! ```
//! use lasx_rs::view::MatRef;
//!
//! // 4 行 3 列，数据按行连着放：第 0 行 [1,2,3]、第 1 行 [4,5,6]……
//! let data: Vec<f32> = (1..=12).map(|v| v as f32).collect();
//! let m = MatRef::row_major(&data, 4, 3).unwrap();
//! assert_eq!((m.rows(), m.cols()), (4, 3));
//! assert_eq!(m.get(1, 2), 6.0);
//! assert_eq!(m.row(1).to_vec(), vec![4.0, 5.0, 6.0]);
//!
//! // 长度不对 → 立刻报错说清期望与实际，而不是等算子里越界
//! let e = MatRef::<f32>::row_major(&data, 3, 5).unwrap_err();
//! assert!(e.to_string().contains("15"), "{e}");
//! assert!(e.to_string().contains("12"), "{e}");
//! ```
//!
//! # 两种布局，以及零拷贝互换
//!
//! ```
//! # use lasx_rs::view::MatRef;
//! # let data: Vec<f32> = (1..=12).map(|v| v as f32).collect();
//! // 行主序（C 习惯）：一行的元素挨着放
//! let r = MatRef::row_major(&data, 4, 3).unwrap();
//! assert_eq!(r.row(0).to_vec(), vec![1.0, 2.0, 3.0]);
//!
//! // 列主序（Fortran/BLAS 习惯，也就是"已经预先转置好的 B"）：一列的元素挨着放
//! let c = MatRef::col_major(&data, 3, 4).unwrap();
//! assert_eq!(c.row(0).to_vec(), vec![1.0, 4.0, 7.0, 10.0]);
//!
//! // 同一块数据，两种视角零拷贝互换：转置
//! let t = r.transpose();
//! assert_eq!((t.rows(), t.cols()), (3, 4));
//! assert_eq!(t.row(0).to_vec(), c.row(0).to_vec());
//! // 逐行遍历转置视角 == 逐列遍历原矩阵（取列不必另造一个"列视图"类型）
//! let cols: Vec<Vec<f32>> = t.iter_rows().map(|row| row.to_vec()).collect();
//! assert_eq!(cols[2], vec![3.0, 6.0, 9.0, 12.0]);
//! ```
//!
//! # 哪些视图能直接下给向量内核
//!
//! 现有内核要的是**连续行主序**（行挨着行、列挨着列）。所以：
//!
//! - [`MatRef::as_row_major_contiguous`] 能拿到 `Some(&[T])` 就直接交给内核；
//! - 拿到 `None` 说明布局不是它要的，此时**不要猜**：要么用
//!   [`MatRef::to_row_major_vec`] 明确付一次复制的代价，要么用 [`crate::plan::MatmulPlan`]
//!   —— 它在**构造时**就把任意布局的 B 归拢成内核要的样子，之后每次调用都不再碰原矩阵。

use crate::api::Error;

/// 只读矩阵视图：借用一段 `&[T]`，附上行数、列数与两个跨距。
///
/// 行列下标都是 0 基。取第 `i` 行用 [`MatRef::row`]，**和底层是行主序还是列主序无关**：
/// 布局只决定"同一行里下一个元素在哪"，这件事由视图负责。
#[derive(Clone, Copy, Debug)]
pub struct MatRef<'a, T> {
    data: &'a [T],
    rows: usize,
    cols: usize,
    /// 相邻两**行**在 `data` 里相隔多少个元素（行主序 = `cols`，列主序 = 1）
    row_stride: usize,
    /// 同一行里相邻两**列**相隔多少个元素（行主序 = 1，列主序 = `rows`）
    col_stride: usize,
}

/// 可写矩阵视图。只提供行主序（内核的输出都是行主序）；需要别的布局时先在
/// [`MatRef`] 上用 [`MatRef::to_row_major_vec`] 转过来。
#[derive(Debug)]
pub struct MatMut<'a, T> {
    data: &'a mut [T],
    rows: usize,
    cols: usize,
}

/// 矩阵的一行。[`MatRef::row`] 的返回值。
///
/// 单独有这个类型，是因为**列主序矩阵的一行在内存里并不连续**：
/// 与其返回一个骗人的 `&[T]`，不如返回一个"按下标取数"的小视图。
/// 连续时用 [`RowRef::as_slice`] 拿回切片（`copy_from_slice` 之类的批量操作就能用了）。
#[derive(Clone, Copy, Debug)]
pub struct RowRef<'a, T> {
    data: &'a [T],
    start: usize,
    len: usize,
    stride: usize,
}

impl<'a, T> MatRef<'a, T> {
    /// 行主序视图：`data` 是 `rows × cols` 个元素，**一行挨着一行**放。
    ///
    /// # Errors
    /// `data.len() != rows × cols`（[`Error::Shape`]），或 `rows × cols` 溢出
    /// `usize`（[`Error::Overflow`]）。
    pub fn row_major(data: &'a [T], rows: usize, cols: usize) -> Result<Self, Error> {
        let want = checked_area("MatRef::row_major", rows, cols)?;
        expect_area("MatRef::row_major", data.len(), want)?;
        Ok(MatRef {
            data,
            rows,
            cols,
            row_stride: cols,
            col_stride: 1,
        })
    }

    /// 列主序视图：`data` 是 `rows × cols` 个元素，**一列挨着一列**放。
    ///
    /// 这是 Fortran/BLAS 的习惯，也是"权重已预先转置好"的表示法。
    ///
    /// # Errors
    /// 同 [`MatRef::row_major`]。
    pub fn col_major(data: &'a [T], rows: usize, cols: usize) -> Result<Self, Error> {
        let want = checked_area("MatRef::col_major", rows, cols)?;
        expect_area("MatRef::col_major", data.len(), want)?;
        Ok(MatRef {
            data,
            rows,
            cols,
            row_stride: 1,
            col_stride: rows,
        })
    }

    /// 带行跨距的行主序视图：每行占 `row_stride` 个元素，只用每行前 `cols` 个。
    ///
    /// 用来**零拷贝切出大矩阵里的一块**（每行后面带 padding，或两个矩阵按行交错存放）。
    /// 末尾允许有 padding：只要容得下最后一行的 `cols` 个元素就行。
    ///
    /// # Errors
    /// `row_stride < cols`（行会重叠）或 `data` 容不下 `rows` 行。
    pub fn row_major_strided(
        data: &'a [T],
        rows: usize,
        cols: usize,
        row_stride: usize,
    ) -> Result<Self, Error> {
        if row_stride < cols {
            return Err(Error::Shape {
                op: "MatRef::row_major_strided",
                what: "row_stride（不能小于 cols）",
                expected: cols,
                got: row_stride,
            });
        }
        let need = if rows == 0 {
            0
        } else {
            (rows - 1)
                .checked_mul(row_stride)
                .and_then(|v| v.checked_add(cols))
                .ok_or(Error::Overflow {
                    op: "MatRef::row_major_strided",
                    what: "(rows-1)×row_stride+cols",
                })?
        };
        if data.len() < need {
            return Err(Error::Shape {
                op: "MatRef::row_major_strided",
                what: "data",
                expected: need,
                got: data.len(),
            });
        }
        Ok(MatRef {
            data,
            rows,
            cols,
            row_stride,
            col_stride: 1,
        })
    }

    /// 行数。
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// 列数。
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// 是否是空矩阵（`rows == 0` 或 `cols == 0`）。
    pub fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }

    /// 相邻两行在底层数据里相隔多少个元素（只用于诊断/与内核对齐口径）。
    pub fn row_stride(&self) -> usize {
        self.row_stride
    }

    /// 同一行里相邻两列相隔多少个元素。
    pub fn col_stride(&self) -> usize {
        self.col_stride
    }

    /// 转置视角（**零拷贝**）：行列互换，数据一个字节都不动。
    ///
    /// 行主序矩阵转置后就是同一块数据的列主序视角，反之亦然。
    /// `m.transpose().row(j)` 就是 `m` 的第 `j` 列。
    pub fn transpose(&self) -> MatRef<'a, T> {
        MatRef {
            data: self.data,
            rows: self.cols,
            cols: self.rows,
            row_stride: self.col_stride,
            col_stride: self.row_stride,
        }
    }

    /// 取第 `i` 行，长度正好 `cols`（用 [`RowRef::get`] 按列取数，
    /// 或用 [`RowRef::as_slice`] 在布局连续时拿整行切片）。
    ///
    /// # Panics
    /// `i >= rows` 时 panic（越界取行是调用方的逻辑错误）。不想 panic 用
    /// [`MatRef::row_checked`]。
    pub fn row(&self, i: usize) -> RowRef<'a, T> {
        assert!(i < self.rows, "行下标 {i} 越界（共 {} 行）", self.rows);
        RowRef {
            data: self.data,
            start: i * self.row_stride,
            len: self.cols,
            stride: self.col_stride,
        }
    }

    /// 取第 `i` 行，越界返回 `None`。
    pub fn row_checked(&self, i: usize) -> Option<RowRef<'a, T>> {
        if i < self.rows {
            Some(self.row(i))
        } else {
            None
        }
    }

    /// 按行迭代（每项是 [`RowRef`]）。
    pub fn iter_rows(self) -> impl Iterator<Item = RowRef<'a, T>> {
        (0..self.rows).map(move |i| self.row(i))
    }

    /// 取第 `i` 行第 `j` 列的元素。
    ///
    /// # Panics
    /// 下标越界时 panic。
    pub fn get(&self, i: usize, j: usize) -> T
    where
        T: Copy,
    {
        assert!(
            i < self.rows && j < self.cols,
            "下标 ({i},{j}) 越界（{}×{}）",
            self.rows,
            self.cols
        );
        self.data[i * self.row_stride + j * self.col_stride]
    }

    /// 这个视图正好是**连续行主序**时，返回背后的整段切片；否则 `None`。
    ///
    /// "连续行主序" = 行挨着行（`row_stride == cols`）且列挨着列（`col_stride == 1`；
    /// 只有一列时列跨距没有意义，不看它）。现有向量内核都要这个布局，所以这里是
    /// "能不能直接下给内核"的判据。
    pub fn as_row_major_contiguous(&self) -> Option<&'a [T]> {
        let rows_packed = self.row_stride == self.cols;
        let cols_packed = self.col_stride == 1 || self.cols <= 1;
        if rows_packed && cols_packed {
            Some(&self.data[..self.rows * self.cols])
        } else {
            None
        }
    }

    /// 不是连续行主序时**明确复制**一份连续行主序的数据出来。
    ///
    /// 名字里带 `_vec` 就是为了让调用方看见这一次分配 + 一次搬运。
    pub fn to_row_major_vec(&self) -> Vec<T>
    where
        T: Copy,
    {
        let mut out = Vec::with_capacity(self.rows * self.cols);
        for r in self.iter_rows() {
            out.extend(r.iter());
        }
        out
    }
}

impl<'a, T> RowRef<'a, T> {
    /// 这一行有多少列。
    pub fn len(&self) -> usize {
        self.len
    }

    /// 这一行是否为空（`cols == 0`）。
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 取本行第 `j` 个元素。
    ///
    /// # Panics
    /// `j >= len()` 时 panic。
    pub fn get(&self, j: usize) -> T
    where
        T: Copy,
    {
        assert!(j < self.len, "列下标 {j} 越界（共 {} 列）", self.len);
        self.data[self.start + j * self.stride]
    }

    /// 这一行在内存里连续时返回切片（列主序矩阵的行不连续，会得到 `None`）。
    pub fn as_slice(&self) -> Option<&'a [T]> {
        if self.stride == 1 {
            Some(&self.data[self.start..self.start + self.len])
        } else {
            None
        }
    }

    /// 按列迭代取数（不连续布局也能用，只是逐个取）。
    pub fn iter(self) -> impl Iterator<Item = T> + 'a
    where
        T: Copy,
    {
        (0..self.len).map(move |j| self.get(j))
    }

    /// 收集成 `Vec`（不连续布局时最直接的"落地"办法）。
    pub fn to_vec(self) -> Vec<T>
    where
        T: Copy,
    {
        match self.as_slice() {
            Some(s) => s.to_vec(),
            None => self.iter().collect(),
        }
    }
}

impl<'a, T> MatMut<'a, T> {
    /// 行主序可写视图：`data` 是 `rows × cols` 个元素，一行挨着一行放。
    ///
    /// # Errors
    /// 同 [`MatRef::row_major`]。
    pub fn row_major(data: &'a mut [T], rows: usize, cols: usize) -> Result<Self, Error> {
        let want = checked_area("MatMut::row_major", rows, cols)?;
        expect_area("MatMut::row_major", data.len(), want)?;
        Ok(MatMut { data, rows, cols })
    }

    /// 行数。
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// 列数。
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// 取第 `i` 行（长度正好 `cols`）。
    ///
    /// # Panics
    /// `i >= rows` 时 panic。
    pub fn row_mut(&mut self, i: usize) -> &mut [T] {
        assert!(i < self.rows, "行下标 {i} 越界（共 {} 行）", self.rows);
        let start = i * self.cols;
        &mut self.data[start..start + self.cols]
    }

    /// 整个矩阵的连续切片（行主序，长度 `rows × cols`）。
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
    }

    /// 把每个元素设成 `value`。
    pub fn fill(&mut self, value: T)
    where
        T: Copy,
    {
        self.data.fill(value);
    }
}

/// `rows × cols`，溢出则报错。
fn checked_area(op: &'static str, rows: usize, cols: usize) -> Result<usize, Error> {
    rows.checked_mul(cols).ok_or(Error::Overflow {
        op,
        what: "rows×cols",
    })
}

/// 长度必须正好等于 `want`。
fn expect_area(op: &'static str, got: usize, want: usize) -> Result<(), Error> {
    if got == want {
        Ok(())
    } else {
        Err(Error::Shape {
            op,
            what: "data",
            expected: want,
            got,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1..=12 的行主序 4×3 矩阵。
    fn data() -> Vec<f32> {
        (1..=12).map(|v| v as f32).collect()
    }

    #[test]
    fn row_major_and_col_major_agree_on_transpose() {
        let d = data();
        let r = MatRef::row_major(&d, 4, 3).unwrap();
        let c = MatRef::col_major(&d, 3, 4).unwrap();
        let t = r.transpose();
        assert_eq!((t.rows(), t.cols()), (3, 4));
        for i in 0..3 {
            assert_eq!(t.row(i).to_vec(), c.row(i).to_vec(), "第 {i} 行");
        }
        // 转置出来的行不连续，原矩阵的行连续
        assert!(t.row(0).as_slice().is_none());
        assert!(r.row(0).as_slice().is_some());
    }

    #[test]
    fn get_and_row_agree_with_index_math() {
        let d = data();
        let r = MatRef::row_major(&d, 4, 3).unwrap();
        let c = r.transpose(); // 同一块数据的列主序 3×4 视角
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(c.get(i, j), r.get(j, i));
            }
        }
    }

    #[test]
    fn shape_error_reports_expected_and_got() {
        let d = [0f32; 10];
        match MatRef::row_major(&d, 3, 4).unwrap_err() {
            Error::Shape {
                op,
                what,
                expected,
                got,
            } => assert_eq!(
                (op, what, expected, got),
                ("MatRef::row_major", "data", 12, 10)
            ),
            other => panic!("期望 Shape 错误，得到 {other:?}"),
        }
    }

    #[test]
    fn strided_view_selects_rows_without_copying() {
        // 每行 5 个元素、只取前 3 个（后 2 个是 padding）
        let d: Vec<i32> = (1..=20).collect();
        let m = MatRef::row_major_strided(&d, 4, 3, 5).unwrap();
        assert_eq!(m.row(0).to_vec(), vec![1, 2, 3]);
        assert_eq!(m.row(1).to_vec(), vec![6, 7, 8]);
        assert!(
            m.as_row_major_contiguous().is_none(),
            "带跨距就不是连续行主序"
        );
        assert_eq!(
            m.to_row_major_vec(),
            vec![1, 2, 3, 6, 7, 8, 11, 12, 13, 16, 17, 18]
        );

        // 跨距小于列数（行会重叠）与容不下 rows 行，都要报错
        assert!(MatRef::row_major_strided(&d, 4, 3, 2).is_err());
        assert!(MatRef::row_major_strided(&d, 5, 3, 5).is_err());
        // 但末尾允许有 padding：4 行 × 跨距 5，只需 3 + 3*5 = 18 个元素
        assert!(MatRef::row_major_strided(&d[..18], 4, 3, 5).is_ok());
    }

    #[test]
    fn contiguous_detection() {
        let d = data();
        assert_eq!(
            MatRef::row_major(&d, 4, 3)
                .unwrap()
                .as_row_major_contiguous()
                .unwrap()
                .len(),
            12
        );
        assert!(MatRef::col_major(&d, 4, 3)
            .unwrap()
            .as_row_major_contiguous()
            .is_none());
        // 单列矩阵：行主序与列主序是同一件事，都算连续
        let col = MatRef::col_major(&d[..4], 4, 1).unwrap();
        assert!(col.as_row_major_contiguous().is_some());
    }

    #[test]
    fn zero_rows_is_ok_and_iterates_nothing() {
        let m = MatRef::<f32>::row_major(&[], 0, 7).unwrap();
        assert!(m.is_empty());
        assert_eq!(m.iter_rows().count(), 0);
        assert!(m.row_checked(0).is_none());
    }

    #[test]
    fn mat_mut_writes_rows() {
        let mut d = data();
        {
            let mut m = MatMut::row_major(&mut d, 4, 3).unwrap();
            assert_eq!((m.rows(), m.cols()), (4, 3));
            m.row_mut(2).copy_from_slice(&[0.0, 0.0, 0.0]);
            m.fill(1.0);
        }
        assert_eq!(d, vec![1.0; 12]);
        assert!(MatMut::row_major(&mut d[..], 3, 5).is_err());
    }

    #[test]
    fn row_out_of_range_panics_with_message() {
        let d = data();
        let m = MatRef::row_major(&d, 4, 3).unwrap();
        let e = std::panic::catch_unwind(|| m.row(4));
        assert!(e.is_err(), "越界取行应当 panic");
    }
}
