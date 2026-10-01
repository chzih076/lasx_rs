//! 三分量批量视图：`V3Ref`（借用）/ `V3Buf`（拥有）。
//!
//! 覆盖库里所有"逐样本三分量"的批量算子；方法名统一是 `*_into`（结果写进 `V3Buf` 或
//! `VecBuf`），与 [`crate::shape::Mat::matmul_into`] 的命名一致。

use crate::aligned::AlignedVec;
use crate::api::{expect_len, Error};
use crate::shape::VecBuf;

/// 三分量**借用**视图：x/y/z 三条数组，样本数 `N` 在类型里。
pub struct V3Ref<'a, T, const N: usize> {
    x: &'a [T],
    y: &'a [T],
    z: &'a [T],
}

impl<'a, T: Copy, const N: usize> V3Ref<'a, T, N> {
    /// 把三条分量数组包成 `N` 样本的视图。
    ///
    /// # Errors
    /// 任一条长度不等于 `N`（[`Error::Shape`]，`what` 点名是哪一条分量）。
    pub fn new(x: &'a [T], y: &'a [T], z: &'a [T]) -> Result<Self, Error> {
        expect_len("V3Ref::new", "x 分量", x.len(), N)?;
        expect_len("V3Ref::new", "y 分量", y.len(), N)?;
        expect_len("V3Ref::new", "z 分量", z.len(), N)?;
        Ok(V3Ref { x, y, z })
    }

    /// 样本数（编译期已知）。
    pub fn len(&self) -> usize {
        N
    }

    /// `N == 0`。
    pub fn is_empty(&self) -> bool {
        N == 0
    }

    /// 三条分量（顺序固定：x、y、z）。
    pub fn components(&self) -> [&'a [T]; 3] {
        [self.x, self.y, self.z]
    }

    /// 第 `i` 个样本的 `(x, y, z)`。
    ///
    /// # Panics
    /// `i >= N`（与普通切片下标一致的越界 panic）。
    pub fn at(&self, i: usize) -> (T, T, T) {
        (self.x[i], self.y[i], self.z[i])
    }
}

/// 三分量**拥有**视图：三条 32 字节对齐的缓冲。
pub struct V3Buf<T, const N: usize> {
    x: AlignedVec<T>,
    y: AlignedVec<T>,
    z: AlignedVec<T>,
}

impl<T: Copy + Default, const N: usize> V3Buf<T, N> {
    /// 分配三条 `N` 长的零值缓冲（不失败：长度是编译期常量）。
    pub fn new() -> Self {
        V3Buf {
            x: AlignedVec::new(N),
            y: AlignedVec::new(N),
            z: AlignedVec::new(N),
        }
    }

    /// 样本数（编译期已知）。
    pub fn len(&self) -> usize {
        N
    }

    /// `N == 0`。
    pub fn is_empty(&self) -> bool {
        N == 0
    }

    /// 三条可变分量（顺序固定：x、y、z）。三条来自不同字段，借用检查器能证明互不重叠。
    pub fn components_mut(&mut self) -> [&mut [T]; 3] {
        [
            self.x.as_mut_slice(),
            self.y.as_mut_slice(),
            self.z.as_mut_slice(),
        ]
    }

    /// 借出等长的只读视图（把它当输入用时）。
    pub fn as_ref(&self) -> V3Ref<'_, T, N> {
        V3Ref {
            x: self.x.as_slice(),
            y: self.y.as_slice(),
            z: self.z.as_slice(),
        }
    }

    /// 第 `i` 个样本的 `(x, y, z)`。
    ///
    /// # Panics
    /// `i >= N`。
    pub fn at(&self, i: usize) -> (T, T, T) {
        (
            self.x.as_slice()[i],
            self.y.as_slice()[i],
            self.z.as_slice()[i],
        )
    }
}

impl<T: Copy + Default, const N: usize> Default for V3Buf<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> std::fmt::Debug for V3Ref<'_, T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("V3Ref")
            .field("N", &N)
            .finish_non_exhaustive()
    }
}

impl<T, const N: usize> std::fmt::Debug for V3Buf<T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("V3Buf")
            .field("N", &N)
            .finish_non_exhaustive()
    }
}

// ============================ f64 算子（逐样本三分量） ============================

impl<'a, const N: usize> V3Ref<'a, f64, N> {
    /// 模长 `out[i] = √(x² + y² + z²)`（LASX 4 样本/向量，LSX 2 样本，标量尾）。
    pub fn norm3_into(&self, out: &mut VecBuf<f64, N>) {
        crate::ops::norm3_batch::norm3_batch(self.x, self.y, self.z, out.as_mut_slice());
    }

    /// 单位化 `o = v / |v|`（`|v|` 与 [`Self::norm3_into`] 同结合；零向量→原样）。
    pub fn unitize3_into(&self, out: &mut V3Buf<f64, N>) {
        let [ox, oy, oz] = out.components_mut();
        crate::ops::unitize3_batch::unitize3_batch(self.x, self.y, self.z, ox, oy, oz);
    }

    /// 叉积 `o = a × b`。
    pub fn cross_into(&self, b: &V3Ref<'_, f64, N>, out: &mut V3Buf<f64, N>) {
        let [ox, oy, oz] = out.components_mut();
        crate::ops::cross3_batch::cross3_batch(self.x, self.y, self.z, b.x, b.y, b.z, ox, oy, oz);
    }

    /// 缩放加 `o = a + s·b`。
    pub fn add_scaled_into(&self, b: &V3Ref<'_, f64, N>, s: f64, out: &mut V3Buf<f64, N>) {
        let [ox, oy, oz] = out.components_mut();
        crate::ops::vec3_add_scaled_batch::vec3_add_scaled_batch(
            self.x, self.y, self.z, b.x, b.y, b.z, s, ox, oy, oz,
        );
    }

    /// 中心项 + J2 加速度 `o = a(r; mu, j2, re)`。
    pub fn j2_accel_into(&self, mu: f64, j2: f64, re: f64, out: &mut V3Buf<f64, N>) {
        let [ox, oy, oz] = out.components_mut();
        crate::ops::j2_accel_batch::j2_accel_batch(self.x, self.y, self.z, mu, j2, re, ox, oy, oz);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 长度不符 ⇒ `Error::Shape`，且 `what` 必须点名**是哪一条分量**（三条分别测）。
    #[test]
    fn test_component_len_mismatch_reports_which() {
        let ok: &[f64] = &[0.0; 4];
        let short: &[f64] = &[0.0; 3];
        for (bad, expect_what) in [
            ((short, ok, ok), "x 分量"),
            ((ok, short, ok), "y 分量"),
            ((ok, ok, short), "z 分量"),
        ] {
            match V3Ref::<f64, 4>::new(bad.0, bad.1, bad.2) {
                Err(Error::Shape {
                    op,
                    what,
                    expected,
                    got,
                }) => {
                    assert_eq!(op, "V3Ref::new");
                    assert_eq!(what, expect_what);
                    assert_eq!((expected, got), (4, 3));
                }
                other => panic!("应报长度错误，实际 {other:?}"),
            }
        }
    }

    /// 与直接调内核**逐位一致**（这一层只做视图与转发，不改数值）。
    #[test]
    fn test_matches_direct_kernel_bitwise() {
        const N: usize = 37; // 不是 4 的倍数：覆盖向量尾 + 标量尾
        let mut lcg = crate::ops::testutil::Lcg(0x5eed);
        let x: Vec<f64> = (0..N).map(|_| lcg.f64()).collect();
        let y: Vec<f64> = (0..N).map(|_| lcg.f64()).collect();
        let z: Vec<f64> = (0..N).map(|_| lcg.f64()).collect();
        let bx: Vec<f64> = (0..N).map(|_| lcg.f64()).collect();
        let by: Vec<f64> = (0..N).map(|_| lcg.f64()).collect();
        let bz: Vec<f64> = (0..N).map(|_| lcg.f64()).collect();
        let a = V3Ref::<f64, N>::new(&x, &y, &z).unwrap();
        let b = V3Ref::<f64, N>::new(&bx, &by, &bz).unwrap();

        // norm3
        let mut got = VecBuf::<f64, N>::new();
        a.norm3_into(&mut got);
        let mut want = vec![0.0f64; N];
        crate::ops::norm3_batch::norm3_batch(&x, &y, &z, &mut want);
        assert_eq!(bits(got.as_slice()), bits(&want), "norm3");

        // unitize3 / cross3 / add_scaled / j2_accel：逐个对照
        let mut got3 = V3Buf::<f64, N>::new();
        a.unitize3_into(&mut got3);
        let (mut wx, mut wy, mut wz) = (vec![0.0; N], vec![0.0; N], vec![0.0; N]);
        crate::ops::unitize3_batch::unitize3_batch(&x, &y, &z, &mut wx, &mut wy, &mut wz);
        assert_eq!(bits3(&got3), bits3s(&[&wx, &wy, &wz]), "unitize3");

        a.cross_into(&b, &mut got3);
        crate::ops::cross3_batch::cross3_batch(
            &x, &y, &z, &bx, &by, &bz, &mut wx, &mut wy, &mut wz,
        );
        assert_eq!(bits3(&got3), bits3s(&[&wx, &wy, &wz]), "cross3");

        a.add_scaled_into(&b, 2.5, &mut got3);
        crate::ops::vec3_add_scaled_batch::vec3_add_scaled_batch(
            &x, &y, &z, &bx, &by, &bz, 2.5, &mut wx, &mut wy, &mut wz,
        );
        assert_eq!(bits3(&got3), bits3s(&[&wx, &wy, &wz]), "add_scaled");

        a.j2_accel_into(3.986e14, 1.0826e-3, 6.378e6, &mut got3);
        crate::ops::j2_accel_batch::j2_accel_batch(
            &x, &y, &z, 3.986e14, 1.0826e-3, 6.378e6, &mut wx, &mut wy, &mut wz,
        );
        assert_eq!(bits3(&got3), bits3s(&[&wx, &wy, &wz]), "j2_accel");
    }

    /// `N = 0` / `N = 1` 不炸，且 `N = 0` 时内核不被调用（输出保持零）。
    #[test]
    fn test_degenerate_sizes() {
        let empty = V3Ref::<f64, 0>::new(&[], &[], &[]).unwrap();
        let mut out = V3Buf::<f64, 0>::new();
        empty.norm3_into(&mut VecBuf::new());
        empty.cross_into(&empty, &mut out);
        assert!(empty.is_empty());

        let one = V3Ref::<f64, 1>::new(&[3.0], &[4.0], &[0.0]).unwrap();
        let mut mag = VecBuf::<f64, 1>::new();
        one.norm3_into(&mut mag);
        assert_eq!(mag.as_slice(), &[5.0]);
    }

    /// `at()` 与切片访问一致。
    #[test]
    fn test_at_matches_components() {
        let x = [1.0f64, 2.0];
        let y = [3.0f64, 4.0];
        let z = [5.0f64, 6.0];
        let v = V3Ref::<f64, 2>::new(&x, &y, &z).unwrap();
        assert_eq!(v.at(1), (2.0, 4.0, 6.0));
        assert_eq!(v.components()[1], &y[..]);
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn bits3(v: &V3Buf<f64, 37>) -> [Vec<u64>; 3] {
        let [x, y, z] = [
            v.as_ref().components()[0],
            v.as_ref().components()[1],
            v.as_ref().components()[2],
        ];
        [bits(x), bits(y), bits(z)]
    }

    fn bits3s(v: &[&Vec<f64>; 3]) -> [Vec<u64>; 3] {
        [bits(v[0]), bits(v[1]), bits(v[2])]
    }
}
