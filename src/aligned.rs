//! 对齐缓冲区：LASX 是 **32 字节**访存，首地址不落在 32 字节边界时，每次
//! 32 字节的 `xvld/xvst` 都会跨 64 字节缓存行。实测在 L1 驻留规模上差
//! **1.09×–1.56×**（见 `docs/dev.md` §7.5）。
//!
//! 普通 `Vec<T>` 只保证 `align_of::<T>()`（`f32` 为 4、`f64` 为 8），而
//! `malloc`/Dart FFI 的缓冲区**只有约一半**落在 32 字节边界上——是抽签，不能假设。
//! 需要稳定拿到对齐缓冲时用这里的 [`AlignedVec`]。
//!
//! ```
//! use lasx_rs::aligned::AlignedVec;
//!
//! let a = AlignedVec::fill_with(4, |i| i as f32);
//! assert_eq!(a.as_ptr() as usize % lasx_rs::aligned::ALIGN, 0);
//! assert_eq!(&a[..], &[0.0, 1.0, 2.0, 3.0]);
//! ```

/// [`AlignedVec`] 首地址的对齐字节数。
///
/// 取 64：既是 LASX 要求的 [`crate::ffi::memory::LASX_ALIGN`]（32）的倍数，
/// 也让首地址落在缓存行边界上。
pub const ALIGN: usize = 64;

/// 首地址保证 [`ALIGN`] 字节对齐、长度为 `len` 的缓冲区。
///
/// 通过 [`Deref`](std::ops::Deref) 可直接当 `&[T]` 用（索引、`len()`、迭代、
/// 传给接收切片的函数都成立）；[`as_ptr`](Self::as_ptr) 给出保证对齐的裸指针，
/// 可交给 C ABI。
pub struct AlignedVec<T> {
    buf: Vec<T>,
    /// 从 `buf` 起点到对齐窗口的偏移（元素数）。
    off: usize,
    len: usize,
}

impl<T: Copy + Default> AlignedVec<T> {
    /// 分配 `len` 个元素的零值缓冲，保证首地址 [`ALIGN`] 字节对齐。
    pub fn new(len: usize) -> Self {
        let esz = std::mem::size_of::<T>().max(1);
        // 多分配一个对齐单位 + 1 个元素，保证无论 buf 起点如何都能切出对齐窗口
        let extra = ALIGN / esz + 1;
        let buf = vec![T::default(); len + extra];
        let misalign = buf.as_ptr() as usize % ALIGN;
        let off = if misalign == 0 {
            0
        } else {
            (ALIGN - misalign).div_ceil(esz)
        };
        let v = AlignedVec { buf, off, len };
        debug_assert_eq!(v.as_ptr() as usize % ALIGN, 0);
        v
    }

    /// 分配并用 `f(i)` 填充（下标从 0 起，形状与 `(0..len).map(f)` 一致）。
    pub fn fill_with(len: usize, mut f: impl FnMut(usize) -> T) -> Self {
        let mut v = Self::new(len);
        for (i, x) in AlignedVec::as_mut_slice(&mut v).iter_mut().enumerate() {
            *x = f(i);
        }
        v
    }
}

impl<T> AlignedVec<T> {
    /// 元素个数。
    pub fn len(&self) -> usize {
        self.len
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 底层切片视图。
    pub fn as_slice(&self) -> &[T] {
        &self.buf[self.off..self.off + self.len]
    }

    /// 底层可变切片视图。
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.buf[self.off..self.off + self.len]
    }

    /// 首地址（保证 [`ALIGN`] 字节对齐）。
    pub fn as_ptr(&self) -> *const T {
        AlignedVec::as_slice(self).as_ptr()
    }

    /// 可写首地址（保证 [`ALIGN`] 字节对齐）。
    pub fn as_mut_ptr(&mut self) -> *mut T {
        AlignedVec::as_mut_slice(self).as_mut_ptr()
    }
}

impl<T> std::ops::Deref for AlignedVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        AlignedVec::as_slice(self)
    }
}

impl<T> std::ops::DerefMut for AlignedVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        AlignedVec::as_mut_slice(self)
    }
}

impl<T: Copy + Default> Clone for AlignedVec<T> {
    /// 克隆后**重新计算对齐偏移**——直接复用原 `off` 会因新分配地址不同而失准。
    fn clone(&self) -> Self {
        let mut v = Self::new(self.len);
        AlignedVec::as_mut_slice(&mut v).copy_from_slice(AlignedVec::as_slice(self));
        v
    }
}

impl<T> std::fmt::Debug for AlignedVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedVec")
            .field("len", &self.len)
            .field("align", &ALIGN)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment_holds_for_various_types_and_lengths() {
        macro_rules! check {
            ($t:ty) => {
                for &n in &[0usize, 1, 3, 7, 8, 63, 64, 65, 1024] {
                    let v: AlignedVec<$t> = AlignedVec::new(n);
                    assert_eq!(v.len(), n, "len 不符");
                    assert_eq!(
                        v.as_ptr() as usize % ALIGN,
                        0,
                        "{} n={n}: 首地址未对齐",
                        stringify!($t)
                    );
                    assert_eq!(v.as_slice().len(), n);
                }
            };
        }
        check!(f32);
        check!(f64);
        check!(i8);
        check!(u8);
        check!(i32);
    }

    #[test]
    fn test_fill_with_uses_index_and_values_are_addressable() {
        let v = AlignedVec::fill_with(5, |i| i as f64 * 2.0);
        assert_eq!(&v[..], &[0.0, 2.0, 4.0, 6.0, 8.0]);
        // Deref 到切片：len / 迭代 / 切片传参都成立
        assert_eq!(v.len(), 5);
        assert_eq!(v.iter().sum::<f64>(), 20.0);
        assert_eq!(&v[1..3], &[2.0, 4.0]);
    }

    #[test]
    fn test_mutation_through_deref_and_as_mut_slice() {
        let mut v = AlignedVec::<f32>::new(4);
        v[0] = 1.0;
        v.as_mut_slice()[3] = 4.0;
        assert_eq!(&v[..], &[1.0, 0.0, 0.0, 4.0]);
    }

    #[test]
    fn test_clone_realigns_and_copies() {
        let v = AlignedVec::fill_with(9, |i| i as i8);
        let c = v.clone();
        assert_eq!(c.as_ptr() as usize % ALIGN, 0, "克隆体也必须对齐");
        assert_eq!(&c[..], &v[..]);
        // 两个缓冲区相互独立
        assert_ne!(c.as_ptr(), v.as_ptr());
    }

    #[test]
    fn test_empty_is_derefable() {
        let v = AlignedVec::<f64>::new(0);
        assert!(v.is_empty());
        assert_eq!(v.as_slice(), &[] as &[f64]);
        assert_eq!(v.as_ptr() as usize % ALIGN, 0);
    }
}
