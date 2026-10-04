//! 形状层的**向量视图**：`VecRef`（借用）与 `VecBuf`（拥有），一维、长度进类型。
//!
//! 与二维的 [`Mat`](crate::shape::Mat) / [`MatBuf`](crate::shape::MatBuf) 同构：
//! **构造时校验一次长度**，之后所有下标运算都不可能再失败——所以点积直接返回
//! `f32`/`f64`，而不是 `Result`。f16 权重的向量/矩阵视图在
//! [`crate::shape::f16`]（`F16Vec` / `F16Mat`）。
//!
//! # 谁守什么
//!
//! | 契约 | 由什么保证 |
//! |---|---|
//! | 元素个数 == `N` | **类型系统**：`VecRef::new` 校验一次，之后 `N` 是类型参数 |
//! | 点积两侧长度一致 | **类型系统**：`Dot` 的 impl 把两侧的 `N` 绑成同一个常量 |
//! | 点积两侧 dtype 受支持 | **类型系统**（impl 集合）；不支持时 rustc 用 `on_unimplemented` 给提示 |
//! | 输出对齐到 32 字节 | [`AlignedVec`]（与 `api::*` 的输出同一条约定） |
//!
//! # 例子
//!
//! ```
//! use lasx_rs::shape::{F16Mat, VecBuf, VecRef};
//! use lasx_rs::{dot, gemv};
//!
//! const K: usize = 8;
//! const N: usize = 2;
//!
//! let x = VecRef::<f32, K>::new(&[1.0; K])?;
//! let y = VecRef::<f32, K>::new(&[2.0; K])?;
//! assert_eq!(dot!(x[K] * y[K]), 16.0);          // f32 · f32
//!
//! let w_bits: Vec<u16> = [0x3c00u16; N * K].to_vec(); // 0x3c00 = 1.0
//! let w = F16Mat::<N, K>::new(&w_bits)?;
//! let out: VecBuf<f32, N> = gemv!(w[N, K] * x[K]);
//! assert_eq!(out.as_slice(), &[8.0, 8.0]);      // f16 权重 · f32 向量
//! # Ok::<(), lasx_rs::api::Error>(())
//! ```

use crate::aligned::AlignedVec;
use crate::api::{Error, expect_len};

/// 一维向量的**借用**视图：长度 `N` 在类型里。
///
/// 构造时只查长度；形状对不对是类型的事。
pub struct VecRef<'a, T, const N: usize> {
    data: &'a [T],
}

impl<'a, T: Copy, const N: usize> VecRef<'a, T, N> {
    /// 把一个切片包成 `N` 长的向量视图。
    ///
    /// # Errors
    /// `data.len() != N`（[`Error::Shape`]）。
    pub fn new(data: &'a [T]) -> Result<Self, Error> {
        expect_len("VecRef::new", "data", data.len(), N)?;
        Ok(VecRef { data })
    }

    /// 元素个数（编译期已知）。
    pub fn len(&self) -> usize {
        N
    }

    /// `N == 0`。
    pub fn is_empty(&self) -> bool {
        N == 0
    }

    /// 连续切片。
    pub fn as_slice(&self) -> &'a [T] {
        self.data
    }
}

/// 一维向量的**拥有**视图：长度 `N` 在类型里，缓冲 32 字节对齐（[`AlignedVec`]）。
pub struct VecBuf<T, const N: usize> {
    data: AlignedVec<T>,
}

impl<T: Copy + Default, const N: usize> VecBuf<T, N> {
    /// 分配 `N` 个元素的零值缓冲（不失败：长度是编译期常量）。
    pub fn new() -> Self {
        VecBuf {
            data: AlignedVec::new(N),
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

    /// 连续切片。
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    /// 连续可变切片。
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }

    /// 借出等长的 [`VecRef`]（把它当输入用时）。
    pub fn as_vec_ref(&self) -> VecRef<'_, T, N> {
        VecRef {
            data: self.data.as_slice(),
        }
    }
}

impl<T: Copy + Default, const N: usize> Default for VecBuf<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> std::fmt::Debug for VecRef<'_, T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VecRef<{}>", N)
    }
}

impl<T, const N: usize> std::fmt::Debug for VecBuf<T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VecBuf<{}>", N)
    }
}

/// 点积：`a · b`。**按两侧的 dtype 分派**，长度由同一常量 `N` 绑定。
///
/// 实现落在**"两个操作数"这一对**上（而不是 `Dot<Rhs>` 那样按接收者分派）：只有这样，
/// 不支持的组合（如 `f16 · f16`）才会走到下面这条 `on_unimplemented` 提示，而不是
/// 变成"第二个实参类型不符"。
///
/// | 左侧 | 右侧 | 结果 | 后端 |
/// |---|---|---|---|
/// | `&VecRef<f32, N>` | `&VecRef<f32, N>` | `f32` | `crate::ops::dot` |
/// | `&VecRef<f64, N>` | `&VecRef<f64, N>` | `f64` | `crate::ops::dot_f64` |
/// | `&F16Vec<N>` | `&VecRef<f32, N>` | `f32` | `crate::ops::dot_f16` |
/// | `&VecRef<f32, N>` | `&F16Vec<N>` | `f32` | 同上（反序也认） |
#[diagnostic::on_unimplemented(
    message = "这两个操作数不支持点积（`dot!`）",
    label = "`dot!` 只认受支持的 dtype 组合",
    note = "支持：`VecRef<f32>·VecRef<f32>`、`VecRef<f64>·VecRef<f64>`、`F16Vec·VecRef<f32>`（两个方向）；f16·f16 或 f32·f64 请先显式转换"
)]
pub trait DotPair {
    /// 点积的结果类型（f32/f64）。
    type Out;

    /// 算 `self.0 · self.1`。长度与 dtype 都由类型定死，**不可能失败**。
    fn dot_pair(self) -> Self::Out;
}

/// `dot!(a[K] * b[K])` 的落点：把 impl 的解析交给 trait 求解。
///
/// 参数是**一个元组**而不是两个泛型实参：两个实参时 rustc 会从第一个实参定下类型、再把
/// 第二个当作待定的推断变量去凑唯一那个 impl，于是"不支持的组合"（如 `F16Vec · F16Vec`）
/// 报成**第二个实参类型不符**；元组让整对类型先定下来，缺 impl 才会触发
/// [`DotPair`] 上的 `on_unimplemented` 提示。
pub fn dot_of<Pair: DotPair>(pair: Pair) -> Pair::Out {
    pair.dot_pair()
}

impl<'r1, 'r2, 'v1, 'v2, const N: usize> DotPair
    for (&'r1 VecRef<'v1, f32, N>, &'r2 VecRef<'v2, f32, N>)
{
    type Out = f32;

    fn dot_pair(self) -> f32 {
        crate::ops::dot::dot(self.0.as_slice(), self.1.as_slice())
    }
}

impl<'r1, 'r2, 'v1, 'v2, const N: usize> DotPair
    for (&'r1 VecRef<'v1, f64, N>, &'r2 VecRef<'v2, f64, N>)
{
    type Out = f64;

    fn dot_pair(self) -> f64 {
        crate::ops::dot_f64::dot_f64(self.0.as_slice(), self.1.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 长度不符 ⇒ `Error::Shape`，算子名/参数名/期望/实际四项都要对（错误口径与 `api` 一致）。
    #[test]
    fn test_len_mismatch_is_shape_error() {
        let data = [1.0f32; 3];
        match VecRef::<f32, 4>::new(&data) {
            Err(Error::Shape {
                op,
                what,
                expected,
                got,
            }) => assert_eq!((op, what, expected, got), ("VecRef::new", "data", 4, 3)),
            other => panic!("期望 Shape 错误，得到 {other:?}"),
        }
        assert!(VecRef::<f32, 3>::new(&data).is_ok());
        assert!(VecRef::<f32, 0>::new(&[]).is_ok());
    }

    /// f32 / f64 点积与 `api::dot` / `api::dot_f64` **逐位一致**（同一条实现）。
    #[test]
    fn test_dot_matches_api_bit_for_bit() {
        let a32: Vec<f32> = (0..33).map(|i| (i as f32).mul_add(0.25, -3.0)).collect();
        let b32: Vec<f32> = (0..33).map(|i| (i as f32).mul_add(-0.5, 7.0)).collect();
        let (ra, rb) = (
            VecRef::<f32, 33>::new(&a32).unwrap(),
            VecRef::<f32, 33>::new(&b32).unwrap(),
        );
        let want = crate::api::dot(&a32, &b32).unwrap();
        assert_eq!(dot_of((&ra, &rb)).to_bits(), want.to_bits());
        // 反序：点积对称，结果必须逐位相同
        assert_eq!(dot_of((&rb, &ra)).to_bits(), want.to_bits());

        let a64: Vec<f64> = (0..40).map(|i| (i as f64).mul_add(0.125, -1.5)).collect();
        let b64: Vec<f64> = (0..40).map(|i| (i as f64).mul_add(-0.25, 2.5)).collect();
        let (ra64, rb64) = (
            VecRef::<f64, 40>::new(&a64).unwrap(),
            VecRef::<f64, 40>::new(&b64).unwrap(),
        );
        let want64 = crate::api::dot_f64(&a64, &b64).unwrap();
        assert_eq!(dot_of((&ra64, &rb64)).to_bits(), want64.to_bits());

        // n = 0：空内积是 0
        let (z1, z2) = (
            VecRef::<f32, 0>::new(&[]).unwrap(),
            VecRef::<f32, 0>::new(&[]).unwrap(),
        );
        assert_eq!(dot_of((&z1, &z2)), 0.0);
    }

    /// `VecBuf` 的缓冲是 32 字节对齐（与 `api` 的输出同一条约定），且长度/切片自洽。
    #[test]
    fn test_vec_buf_is_aligned() {
        let mut b = VecBuf::<f32, 5>::new();
        assert_eq!(b.len(), 5);
        assert!(!b.is_empty());
        assert_eq!(b.as_slice().len(), 5);
        assert_eq!(b.as_mut_slice().len(), 5);
        assert_eq!(
            b.as_mut_slice().as_ptr() as usize % crate::aligned::ALIGN,
            0
        );
        b.as_mut_slice()[2] = 1.5;
        assert_eq!(b.as_vec_ref().as_slice()[2], 1.5);
        assert!(VecBuf::<f64, 0>::new().is_empty());
    }
}
