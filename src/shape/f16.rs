//! f16 权重（`u16` 位型）的形状视图：`F16Vec` / `F16Mat`。
//!
//! f16 在本库里以 **`u16` 位型**进出（零依赖，不引入 half crate，见 `docs/ops.md` §2.11），
//! 所以视图的元素类型是 `u16`，由类型名 `F16*` 承担"这是 f16"的语义。
//!
//! | 视图 | 形状 | 对应内核 |
//! |---|---|---|
//! | [`F16Vec<N>`](F16Vec) | `N` 个 f16（一行权重） | `lasx_dot_f16` |
//! | [`F16Mat<N, K>`](F16Mat) | `N × K` 行主序（`N` 行、每行 `K` 个 f16） | `lasx_gemv_f16` |
//!
//! `F16Mat` 的行主序约定**决定了公式怎么写**：`y[N] = w[N, K] * x[K]`——左边是权重矩阵、
//! 右边是 f32 向量，收缩维 `K` 在两侧同名。反过来写（`x[K] * w[N, K]`）需要 `[K, N]`
//! 布局，本库的 `gemv_f16` 不提供，`gemv!` 会**在编译期报错**并说明原因。
//!
//! # 谁守什么
//!
//! | 契约 | 由什么保证 |
//! |---|---|
//! | `N`/`K` 与数据长度一致 | **类型系统** + 构造时一次校验（`F16Mat::new`） |
//! | `y[r] == dot_f16(w[r, :], x)` | **同一个后端函数**（`ops::dot_f16`），测试逐位对照 |
//! | 池化路径与单线程逐位一致 | 每行独立、切块不影响任何一行（测试逐位对照） |

use crate::api::Error;
use crate::shape::vec::{DotPair, VecBuf, VecRef};

/// f16 权重**向量**（`N` 个 `u16` 位型）：`dot!` 的左操作数。
pub struct F16Vec<'a, const N: usize> {
    data: &'a [u16],
}

impl<'a, const N: usize> F16Vec<'a, N> {
    /// 把一个 `u16` 位型切片包成 `N` 长的 f16 向量视图。
    ///
    /// # Errors
    /// `data.len() != N`（[`Error::Shape`]）。
    pub fn new(data: &'a [u16]) -> Result<Self, Error> {
        if data.len() == N {
            Ok(F16Vec { data })
        } else {
            Err(Error::Shape {
                op: "F16Vec::new",
                what: "data",
                expected: N,
                got: data.len(),
            })
        }
    }

    /// 元素个数（编译期已知）。
    pub fn len(&self) -> usize {
        N
    }

    /// `N == 0`。
    pub fn is_empty(&self) -> bool {
        N == 0
    }

    /// `u16` 位型切片。
    pub fn as_slice(&self) -> &'a [u16] {
        self.data
    }
}

/// f16 权重**矩阵**（`N × K` 行主序，`N` 行、每行 `K` 个 `u16` 位型）。
///
/// `N` 是输出长度（`y` 的元素个数），`K` 是收缩维（`x` 的元素个数）——与
/// `lasx_gemv_f16(a, x, y, m = N, k = K)` 的参数顺序一致。
pub struct F16Mat<'a, const N: usize, const K: usize> {
    data: &'a [u16],
}

impl<'a, const N: usize, const K: usize> F16Mat<'a, N, K> {
    /// 构造时只查长度；形状对不对是类型的事。
    ///
    /// # Errors
    /// `data.len() != N × K`（[`Error::Shape`]），或 `N × K` 溢出 `usize`
    /// （[`Error::Overflow`]，只有常量取到极端值才可能）。
    pub fn new(data: &'a [u16]) -> Result<Self, Error> {
        let want = N.checked_mul(K).ok_or(Error::Overflow {
            op: "F16Mat::new",
            what: "N×K",
        })?;
        if data.len() == want {
            Ok(F16Mat { data })
        } else {
            Err(Error::Shape {
                op: "F16Mat::new",
                what: "data",
                expected: want,
                got: data.len(),
            })
        }
    }

    /// 行数 `N`（输出长度，编译期已知）。
    pub fn rows(&self) -> usize {
        N
    }

    /// 列数 `K`（收缩维，编译期已知）。
    pub fn cols(&self) -> usize {
        K
    }

    /// 连续行主序切片。
    pub fn as_slice(&self) -> &'a [u16] {
        self.data
    }

    /// 第 `row` 行（`K` 个 f16）——与 `lasx_dot_f16` 的一行等长。
    ///
    /// # Panics
    /// `row >= N`。
    pub fn row(&self, row: usize) -> F16Vec<'_, K> {
        assert!(row < N, "行号 {row} 越界（N = {N}）");
        F16Vec {
            data: &self.data[row * K..(row + 1) * K],
        }
    }

    /// `y = W · x`（单线程，分配输出）。
    pub fn gemv(&self, x: &VecRef<'_, f32, K>) -> VecBuf<f32, N> {
        let mut out = VecBuf::<f32, N>::new();
        self.gemv_into(x, &mut out);
        out
    }

    /// `y = W · x`，写进调用方的缓冲（不分配）。
    pub fn gemv_into(&self, x: &VecRef<'_, f32, K>, out: &mut VecBuf<f32, N>) {
        // SAFETY: `gemv_f16` 的前提是 `a.len() == m·k`、`x.len() == k`、`y.len() == m`。
        // 这里 `m = N`、`k = K`：`self.data` 的长度由 `F16Mat::new` 校验为 `N×K`，
        // `x.as_slice()` 的长度由 `VecRef<f32, K>` 保证为 `K`，`out` 是 `VecBuf<f32, N>`。
        unsafe {
            crate::ops::dot_f16::gemv_f16(self.data, x.as_slice(), N, K, out.as_mut_slice());
        }
    }

    /// `y = W · x`，用常驻池按行块切分（多核；**与单线程逐位一致**）。
    ///
    /// 线程数取池的大小；中小形状建议给物理核数并留一个空位（`docs/dev.md` §20.7）。
    ///
    /// # Errors
    /// 形状/溢出错误（[`crate::parallel::gemv_f16`] 会再校验一次，口径与 `api` 一致）。
    pub fn gemv_pooled(
        &self,
        pool: &crate::pool::WorkerPool,
        x: &VecRef<'_, f32, K>,
        out: &mut VecBuf<f32, N>,
    ) -> Result<(), Error> {
        crate::parallel::gemv_f16(pool, N, K, self.data, x.as_slice(), out.as_mut_slice())
    }
}

impl<'r1, 'r2, 'v1, const N: usize> DotPair for (&'r1 F16Vec<'v1, N>, &'r2 VecRef<'_, f32, N>) {
    type Out = f32;

    fn dot_pair(self) -> f32 {
        // SAFETY: 两侧长度都是 `N`（`F16Vec<N>` 与 `VecRef<f32, N>` 各自保证）。
        unsafe { crate::ops::dot_f16::dot_f16(self.0.as_slice(), self.1.as_slice()) }
    }
}

impl<'r1, 'r2, 'v2, const N: usize> DotPair for (&'r1 VecRef<'_, f32, N>, &'r2 F16Vec<'v2, N>) {
    type Out = f32;

    fn dot_pair(self) -> f32 {
        (self.1, self.0).dot_pair()
    }
}

impl<const N: usize> std::fmt::Debug for F16Vec<'_, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "F16Vec<{N}>")
    }
}

impl<const N: usize, const K: usize> std::fmt::Debug for F16Mat<'_, N, K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "F16Mat<{N}×{K}>")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::vec::VecBuf;

    /// 0x3c00 = 1.0、0x4000 = 2.0、0x3800 = 0.5（f16 精确可表示，便于手算期望值）。
    const ONE: u16 = 0x3c00;
    const TWO: u16 = 0x4000;
    const HALF: u16 = 0x3800;

    /// 长度不符 ⇒ `Error::Shape`；`N×K` 极端取值 ⇒ `Error::Overflow`。
    #[test]
    fn test_len_errors() {
        match F16Mat::<2, 3>::new(&[ONE; 5]) {
            Err(Error::Shape {
                op,
                what,
                expected,
                got,
            }) => assert_eq!((op, what, expected, got), ("F16Mat::new", "data", 6, 5)),
            other => panic!("期望 Shape 错误，得到 {other:?}"),
        }
        match F16Vec::<4>::new(&[]) {
            Err(Error::Shape { op, expected, .. }) => {
                assert_eq!((op, expected), ("F16Vec::new", 4));
            }
            other => panic!("期望 Shape 错误，得到 {other:?}"),
        }
        assert!(matches!(
            F16Mat::<{ usize::MAX }, 2>::new(&[]),
            Err(Error::Overflow { op, what }) if op == "F16Mat::new" && what == "N×K"
        ));
    }

    /// `gemv` 与 `api::gemv_f16` **逐位一致**；`gemv_into` 与 `gemv` 一致；
    /// 每行等于 `dot_f16`（逐位）。
    #[test]
    fn test_gemv_matches_api_bit_for_bit() {
        const N: usize = 5;
        const K: usize = 33; // 跨 16/32 边界，走标量尾
        let w: Vec<u16> = (0..N * K)
            .map(|i| match i % 3 {
                0 => ONE,
                1 => TWO,
                _ => HALF,
            })
            .collect();
        let x: Vec<f32> = (0..K).map(|i| (i as f32).mul_add(0.125, -1.5)).collect();

        let w_view = F16Mat::<N, K>::new(&w).unwrap();
        let x_view = VecRef::<f32, K>::new(&x).unwrap();

        let got = w_view.gemv(&x_view);
        let want = crate::api::gemv_f16(N, K, &w, &x).unwrap();
        for r in 0..N {
            assert_eq!(got.as_slice()[r].to_bits(), want[r].to_bits(), "行 {r}");
        }

        let mut into = VecBuf::<f32, N>::new();
        w_view.gemv_into(&x_view, &mut into);
        assert_eq!(into.as_slice(), got.as_slice());

        // 逐行与 dot_f16 一致
        for r in 0..N {
            let d = crate::api::dot_f16(w_view.row(r).as_slice(), &x).unwrap();
            assert_eq!(
                got.as_slice()[r].to_bits(),
                d.to_bits(),
                "行 {r} 与 dot_f16 不一致"
            );
        }
    }

    /// 池化 `gemv_pooled` 与单线程**逐位一致**（切块不影响任何一行）。
    fn check_pooled<const N: usize, const K: usize>(pool: &crate::pool::WorkerPool) {
        let w: Vec<u16> = (0..N * K)
            .map(|i| if i % 2 == 0 { ONE } else { HALF })
            .collect();
        let x: Vec<f32> = (0..K).map(|i| (i as f32).mul_add(0.03, 0.7)).collect();
        let wv = F16Mat::<N, K>::new(&w).unwrap();
        let xv = VecRef::<f32, K>::new(&x).unwrap();
        let want = crate::api::gemv_f16(N, K, &w, &x).unwrap();
        let mut got = VecBuf::<f32, N>::new();
        wv.gemv_pooled(pool, &xv, &mut got).unwrap();
        for r in 0..N {
            assert_eq!(
                got.as_slice()[r].to_bits(),
                want[r].to_bits(),
                "{N}×{K} 行 {r}"
            );
        }
    }

    #[test]
    fn test_gemv_pooled_matches_serial() {
        let pool = crate::pool::WorkerPool::new(4);
        check_pooled::<1, 1>(&pool); // m = 1：池内原地串行
        check_pooled::<3, 40>(&pool); // 低于 MIN_PARALLEL_LEN
        check_pooled::<64, 96>(&pool); // 真正按行块切 + 行尾
        check_pooled::<129, 130>(&pool); // 两维都不整除
    }

    /// 退化档：`N = 0`（空输出）与 `K = 0`（空内积 ⇒ 全 0）。
    #[test]
    fn test_gemv_degenerate() {
        let pool = crate::pool::WorkerPool::new(2);
        // K = 0：y 全 0（与 api 同口径）
        let w0 = F16Mat::<3, 0>::new(&[]).unwrap();
        let x0 = VecRef::<f32, 0>::new(&[]).unwrap();
        let y0 = w0.gemv(&x0);
        assert_eq!(y0.as_slice(), &[0.0; 3]);
        assert!(F16Mat::<0, 4>::new(&[])
            .unwrap()
            .gemv(&VecRef::<f32, 4>::new(&[1.0; 4]).unwrap())
            .is_empty());
        // 池化路径同口径
        let mut y_pooled = VecBuf::<f32, 3>::new();
        w0.gemv_pooled(&pool, &x0, &mut y_pooled).unwrap();
        assert_eq!(y_pooled.as_slice(), &[0.0; 3]);
        let mut y = [1.0f32; 3];
        crate::parallel::gemv_f16(&pool, 3, 0, &[], &[], &mut y).unwrap();
        assert_eq!(y, [0.0; 3]);
        assert!(crate::parallel::gemv_f16(&pool, 0, 4, &[], &[1.0; 4], &mut []).is_ok());
    }

    /// `dot!` 的两个方向都必须认，且与 `api::dot_f16` 逐位一致。
    #[test]
    fn test_f16_dot_both_orders() {
        const N: usize = 40;
        let w: Vec<u16> = (0..N).map(|i| if i % 2 == 0 { ONE } else { TWO }).collect();
        let x: Vec<f32> = (0..N).map(|i| (i as f32) * 0.25).collect();
        let (wf, xv) = (
            F16Vec::<N>::new(&w).unwrap(),
            VecRef::<f32, N>::new(&x).unwrap(),
        );
        let want = crate::api::dot_f16(&w, &x).unwrap();
        assert_eq!(
            crate::shape::vec::dot_of((&wf, &xv)).to_bits(),
            want.to_bits()
        );
        assert_eq!(
            crate::shape::vec::dot_of((&xv, &wf)).to_bits(),
            want.to_bits()
        );
        assert_eq!(wf.len(), N);
        assert!(!wf.is_empty());
        assert_eq!(wf.as_slice().len(), N);
    }

    /// `F16Mat::row` 越界 panic（消息里带行号与 `N`）。
    #[test]
    #[should_panic(expected = "行号 2 越界")]
    fn test_row_out_of_range_panics() {
        let w = [ONE; 6];
        let m = F16Mat::<2, 3>::new(&w).unwrap();
        let _ = m.row(2);
    }

    /// `F16Mat` 的 `rows/cols/as_slice` 与类型参数一致。
    #[test]
    fn test_shape_accessors() {
        let w = vec![ONE; 12];
        let m = F16Mat::<3, 4>::new(&w).unwrap();
        assert_eq!((m.rows(), m.cols()), (3, 4));
        assert_eq!(m.as_slice().len(), 12);
        assert_eq!(m.row(2).as_slice().len(), 4);
        // 空的 f16 数学：N = 0 或 K = 0
        assert!(F16Mat::<0, 4>::new(&[]).unwrap().as_slice().is_empty());
        assert!(F16Mat::<3, 0>::new(&[]).unwrap().as_slice().is_empty());
    }

    /// `gemv` 的输出缓冲是 32 字节对齐的（`VecBuf` 用 `AlignedVec`）。
    #[test]
    fn test_output_buffer_is_aligned() {
        let w = vec![ONE; 8];
        let x = vec![1.0f32; 4];
        let m = F16Mat::<2, 4>::new(&w).unwrap();
        let y = m.gemv(&VecRef::<f32, 4>::new(&x).unwrap());
        assert_eq!(y.as_slice().as_ptr() as usize % crate::aligned::ALIGN, 0);
    }
}
