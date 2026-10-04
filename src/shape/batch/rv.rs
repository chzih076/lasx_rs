//! 位置 + 速度的批量状态视图（6 条分量），覆盖两条"原地推进一步"的内核。
//!
//! | 元素类型 | 方法 | 内核 |
//! |---|---|---|
//! | `f64` | [`RvMut::rk4_j2_step`] | `rk4_j2_step_batch`（中心项 + J2，单片 RK4） |
//! | `f32` | [`RvMut::ballistic_step`] | `ballistic_step`（需要额外的阻力系数数组 `k`） |

use crate::api::{Error, expect_len};
use crate::shape::VecRef;
use crate::shape::batch::V3Ref;

/// 批量状态**独占**视图：`rx, ry, rz, vx, vy, vz` 六条等长数组（原地更新）。
pub struct RvMut<'a, T, const N: usize> {
    rx: &'a mut [T],
    ry: &'a mut [T],
    rz: &'a mut [T],
    vx: &'a mut [T],
    vy: &'a mut [T],
    vz: &'a mut [T],
}

impl<'a, T: Copy, const N: usize> RvMut<'a, T, N> {
    /// 把六条分量数组包成 `N` 样本的状态视图。
    ///
    /// # Errors
    /// 任一条长度不等于 `N`（[`Error::Shape`]，`what` 点名是哪一条）。
    ///
    /// # Panics
    /// 不会 panic：长度全部校验后才构造。
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rx: &'a mut [T],
        ry: &'a mut [T],
        rz: &'a mut [T],
        vx: &'a mut [T],
        vy: &'a mut [T],
        vz: &'a mut [T],
    ) -> Result<Self, Error> {
        expect_len("RvMut::new", "rx 位置分量", rx.len(), N)?;
        expect_len("RvMut::new", "ry 位置分量", ry.len(), N)?;
        expect_len("RvMut::new", "rz 位置分量", rz.len(), N)?;
        expect_len("RvMut::new", "vx 速度分量", vx.len(), N)?;
        expect_len("RvMut::new", "vy 速度分量", vy.len(), N)?;
        expect_len("RvMut::new", "vz 速度分量", vz.len(), N)?;
        Ok(RvMut {
            rx,
            ry,
            rz,
            vx,
            vy,
            vz,
        })
    }

    /// 样本数（编译期已知）。
    pub fn len(&self) -> usize {
        N
    }

    /// `N == 0`。
    pub fn is_empty(&self) -> bool {
        N == 0
    }

    /// 位置三分量的只读视图。
    pub fn position(&self) -> V3Ref<'_, T, N> {
        V3Ref::new(self.rx, self.ry, self.rz).expect("长度已在构造时校验")
    }

    /// 速度三分量的只读视图。
    pub fn velocity(&self) -> V3Ref<'_, T, N> {
        V3Ref::new(self.vx, self.vy, self.vz).expect("长度已在构造时校验")
    }

    /// 六条分量的只读视图（顺序：rx, ry, rz, vx, vy, vz）。
    pub fn components(&self) -> [&[T]; 6] {
        [self.rx, self.ry, self.rz, self.vx, self.vy, self.vz]
    }
}

impl<const N: usize> RvMut<'_, f64, N> {
    /// 推进一步（中心项 + J2），原地更新位置与速度。
    pub fn rk4_j2_step(&mut self, mu: f64, j2: f64, re: f64, dt: f64) {
        crate::ops::rk4_j2_step_batch::rk4_j2_step_batch(
            self.rx, self.ry, self.rz, self.vx, self.vy, self.vz, mu, j2, re, dt,
        );
    }
}

impl<const N: usize> RvMut<'_, f32, N> {
    /// 推进一步（弹道：重力 + 阻力），原地更新位置与速度。
    ///
    /// `k` 是逐样本的阻力系数数组（同样 `N` 个样本，长度在构造时校验）。
    pub fn ballistic_step(&mut self, k: &VecRef<'_, f32, N>, dt: f32, g: f32) {
        crate::ops::ballistic_step::ballistic_step(
            self.rx,
            self.ry,
            self.rz,
            self.vx,
            self.vy,
            self.vz,
            k.as_slice(),
            dt,
            g,
        );
    }
}

impl<T, const N: usize> std::fmt::Debug for RvMut<'_, T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RvMut")
            .field("N", &N)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 长度不符 ⇒ `Error::Shape` 且点名分量（六条逐个测）。
    #[test]
    fn test_len_mismatch_reports_which_component() {
        const N: usize = 3;
        let expect = [
            "rx 位置分量",
            "ry 位置分量",
            "rz 位置分量",
            "vx 速度分量",
            "vy 速度分量",
            "vz 速度分量",
        ];
        for (bad, expect_what) in expect.iter().enumerate() {
            // 六条独立 `Vec`：解构成 6 个绑定再传（同一数组上 `&mut v[0], &mut v[1]` 会被
            // 借用检查器拒掉，`let [a, b, …] = &mut v` 才是标准写法）
            let mut v: [Vec<f64>; 6] =
                std::array::from_fn(|i| vec![0.0; if i == bad { N - 1 } else { N }]);
            let [r0, r1, r2, r3, r4, r5] = &mut v;
            match RvMut::<f64, N>::new(r0, r1, r2, r3, r4, r5) {
                Err(Error::Shape {
                    what,
                    expected,
                    got,
                    ..
                }) => {
                    assert_eq!(&what, expect_what, "第 {bad} 条");
                    assert_eq!((expected, got), (N, N - 1));
                }
                other => panic!("应报长度错误，实际 {other:?}"),
            }
        }
    }

    /// 与直接调内核**逐位一致**（f64 RK4 与 f32 弹道两条都要对）。
    #[test]
    fn test_matches_direct_kernel_bitwise() {
        const N: usize = 33; // 覆盖 4 对齐尾 + 标量尾
        for seed in [0x1234u64, 0xfeed] {
            // f64：RK4（中心项 + J2）。`state64` 返回 6 条独立数组，解构成 6 个绑定
            // （`Vec<Vec<_>>` 上写 `&mut v[0], &mut v[1]` 会撞借用检查）。
            let [mut rx, mut ry, mut rz, mut vx, mut vy, mut vz] = state64(N, seed);
            let [want_rx, want_ry, want_rz, want_vx, want_vy, want_vz] = {
                let [mut wrx, mut wry, mut wrz, mut wvx, mut wvy, mut wvz] = state64(N, seed);
                crate::ops::rk4_j2_step_batch::rk4_j2_step_batch(
                    &mut wrx, &mut wry, &mut wrz, &mut wvx, &mut wvy, &mut wvz, 3.986e14,
                    1.0826e-3, 6.378e6, 0.5,
                );
                [wrx, wry, wrz, wvx, wvy, wvz]
            };
            {
                let mut rv =
                    RvMut::<f64, N>::new(&mut rx, &mut ry, &mut rz, &mut vx, &mut vy, &mut vz)
                        .unwrap();
                rv.rk4_j2_step(3.986e14, 1.0826e-3, 6.378e6, 0.5);
            }
            for (got, want, name) in [
                (&rx, &want_rx, "RK4 rx"),
                (&ry, &want_ry, "RK4 ry"),
                (&rz, &want_rz, "RK4 rz"),
                (&vx, &want_vx, "RK4 vx"),
                (&vy, &want_vy, "RK4 vy"),
                (&vz, &want_vz, "RK4 vz"),
            ] {
                assert_eq!(bits(got), bits(want), "{name}");
            }

            // f32：弹道（重力 + 阻力 + k）
            let [
                mut rx32,
                mut ry32,
                mut rz32,
                mut vx32,
                mut vy32,
                mut vz32,
                k,
            ] = state32(N, seed);
            let [
                want_rx32,
                want_ry32,
                want_rz32,
                want_vx32,
                want_vy32,
                want_vz32,
                _,
            ] = {
                let [mut wrx, mut wry, mut wrz, mut wvx, mut wvy, mut wvz, kk] = state32(N, seed);
                crate::ops::ballistic_step::ballistic_step(
                    &mut wrx, &mut wry, &mut wrz, &mut wvx, &mut wvy, &mut wvz, &kk, 0.01, 9.8,
                );
                [wrx, wry, wrz, wvx, wvy, wvz, kk]
            };
            {
                let kref = VecRef::<f32, N>::new(&k).unwrap();
                let mut rv = RvMut::<f32, N>::new(
                    &mut rx32, &mut ry32, &mut rz32, &mut vx32, &mut vy32, &mut vz32,
                )
                .unwrap();
                rv.ballistic_step(&kref, 0.01, 9.8);
            }
            for (got, want, name) in [
                (&rx32, &want_rx32, "弹道 rx"),
                (&ry32, &want_ry32, "弹道 ry"),
                (&rz32, &want_rz32, "弹道 rz"),
                (&vx32, &want_vx32, "弹道 vx"),
                (&vy32, &want_vy32, "弹道 vy"),
                (&vz32, &want_vz32, "弹道 vz"),
            ] {
                assert_eq!(bits32(got), bits32(want), "{name}");
            }
        }
    }

    /// 视图访问器与直接下标一致。
    #[test]
    fn test_accessors_match() {
        let [mut rx, mut ry, mut rz, mut vx, mut vy, mut vz] = state64(4, 7);
        let expect_pos = (rx[2], ry[2], rz[2]);
        let expect_vz = vz.clone();
        let rv =
            RvMut::<f64, 4>::new(&mut rx, &mut ry, &mut rz, &mut vx, &mut vy, &mut vz).unwrap();
        assert_eq!(rv.position().at(2), expect_pos, "位置视图");
        assert_eq!(rv.velocity().components()[2], &expect_vz[..], "速度视图");
        assert_eq!(rv.len(), 4);
        assert!(!rv.is_empty());
    }

    /// 六条独立数组（顺序 rx, ry, rz, vx, vy, vz）：返回数组而不是 `Vec<Vec<_>>`，
    /// 这样调用方能用数组解构拿到 6 个独立绑定（`&mut v[0], &mut v[1]` 在 `Vec` 上会撞借用）。
    fn state64(n: usize, seed: u64) -> [Vec<f64>; 6] {
        let mut lcg = crate::ops::testutil::Lcg(seed);
        std::array::from_fn(|i| {
            let scale = if i < 3 { 7.0e6 } else { 7.5e3 };
            (0..n).map(|_| scale * lcg.f64()).collect()
        })
    }

    /// 七条独立数组（顺序 rx…vz, k）。
    fn state32(n: usize, seed: u64) -> [Vec<f32>; 7] {
        let mut lcg = crate::ops::testutil::Lcg(seed);
        std::array::from_fn(|_| (0..n).map(|_| lcg.f64() as f32).collect())
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn bits32(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }
}
