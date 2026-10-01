//! 二分量的批量视图（f32，SOA）：给 `batch_distance2d` 这类"输入两条、输出一条"的算子用。

use crate::api::{expect_len, Error};
use crate::shape::VecBuf;

/// 二分量**借用**视图：x/y 两条等长数组，样本数 `N` 在类型里。
pub struct V2Ref<'a, T, const N: usize> {
    x: &'a [T],
    y: &'a [T],
}

impl<'a, T: Copy, const N: usize> V2Ref<'a, T, N> {
    /// 把两条分量数组包成 `N` 样本的视图。
    ///
    /// # Errors
    /// 任一条长度不等于 `N`（[`Error::Shape`]，`what` 点名是哪一条分量）。
    pub fn new(x: &'a [T], y: &'a [T]) -> Result<Self, Error> {
        expect_len("V2Ref::new", "x 分量", x.len(), N)?;
        expect_len("V2Ref::new", "y 分量", y.len(), N)?;
        Ok(V2Ref { x, y })
    }

    /// 样本数（编译期已知）。
    pub fn len(&self) -> usize {
        N
    }

    /// `N == 0`。
    pub fn is_empty(&self) -> bool {
        N == 0
    }

    /// 两条分量（顺序固定：x、y）。
    pub fn components(&self) -> [&'a [T]; 2] {
        [self.x, self.y]
    }

    /// 第 `i` 个样本的 `(x, y)`。
    ///
    /// # Panics
    /// `i >= N`。
    pub fn at(&self, i: usize) -> (T, T) {
        (self.x[i], self.y[i])
    }
}

impl<const N: usize> V2Ref<'_, f32, N> {
    /// 到定点 `(px, py)` 的平面距离 `out[i] = √((x-px)² + (y-py)²)`（LASX 8 点并行）。
    pub fn distance2d_into(&self, px: f32, py: f32, out: &mut VecBuf<f32, N>) {
        crate::ops::batch_distance2d::batch_distance2d(px, py, self.x, self.y, out.as_mut_slice());
    }
}

impl<T, const N: usize> std::fmt::Debug for V2Ref<'_, T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("V2Ref")
            .field("N", &N)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_len_mismatch_reports_which_component() {
        for (x, y, expect) in [
            (vec![0.0f32; 3], vec![0.0f32; 4], "x 分量"),
            (vec![0.0f32; 4], vec![0.0f32; 3], "y 分量"),
        ] {
            match V2Ref::<f32, 4>::new(&x, &y) {
                Err(Error::Shape {
                    what,
                    expected,
                    got,
                    ..
                }) => {
                    assert_eq!(what, expect);
                    assert_eq!((expected, got), (4, 3));
                }
                other => panic!("应报长度错误，实际 {other:?}"),
            }
        }
    }

    /// 与直接调内核逐位一致（N 不是 8 的倍数，覆盖向量尾 + 标量尾）。
    #[test]
    fn test_matches_direct_kernel_bitwise() {
        const N: usize = 21;
        let mut lcg = crate::ops::testutil::Lcg(0xabc);
        let x: Vec<f32> = (0..N).map(|_| 100.0 * lcg.f64() as f32).collect();
        let y: Vec<f32> = (0..N).map(|_| 100.0 * lcg.f64() as f32).collect();
        let v = V2Ref::<f32, N>::new(&x, &y).unwrap();
        let mut got = VecBuf::<f32, N>::new();
        v.distance2d_into(1.5, -2.5, &mut got);
        let mut want = vec![0.0f32; N];
        crate::ops::batch_distance2d::batch_distance2d(1.5, -2.5, &x, &y, &mut want);
        assert_eq!(
            got.as_slice()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}
