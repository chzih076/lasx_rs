//! 批量视图的**运行期长度**孪生：`V3Dyn` / `V3DynMut` / `RvDynMut` / `V2Dyn`。
//!
//! 与 [`super::v3`] 等的分工，同 [`crate::shape::Mat`] 与 [`crate::shape::MatDyn`] 的分工：
//!
//! | | 长度在哪 | 适合什么 |
//! |---|---|---|
//! | [`V3Ref`](super::V3Ref) / [`RvMut`](super::RvMut) | **类型里**（const 泛型 `N`） | 形状编译期已知 |
//! | 本模块（`*Dyn`） | **运行时字段** | 批量大小是运行时参数（下游库里的常态） |
//!
//! 两个口径的**契约是同一条**：所有分量数组等长。区别只是"等长"校验的时机与粒度——
//! `N` 版本校验 `== N`，`*Dyn` 版本校验"三条/六条彼此相等"（以第一条为基准）。
//! 输出数组长度在调用点再校验一次（[`Error::Shape`]），因为输出是独立构造的。
//!
//! # 为什么需要它
//!
//! 内核只信第一条数组的长度（`let n = xs.len();`），而转发层原本只有 `debug_assert_eq!`
//! ——**发布构建里长度不一致就是 UB**。这里把"等长"变成构造时的一次显式校验：
//! 不一致就拿不到视图（[`V3Dyn::new`] 返回 `Err`），也就调不出内核。

use crate::api::{expect_len, Error};

/// 三分量运行期长度**借用**视图。
pub struct V3Dyn<'a, T> {
    x: &'a [T],
    y: &'a [T],
    z: &'a [T],
}

/// 三分量运行期长度**独占**视图（输出/原地用）。
pub struct V3DynMut<'a, T> {
    x: &'a mut [T],
    y: &'a mut [T],
    z: &'a mut [T],
}

impl<T> V3Dyn<'_, T> {
    /// 样本数（运行期）。
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// 样本数为 0。
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }
}

impl<'a, T: Copy> V3Dyn<'a, T> {
    /// 三条分量必须等长（以 `x` 为基准）。
    ///
    /// # Errors
    /// `y`/`z` 与 `x` 长度不一致（[`Error::Shape`]，`what` 点名是哪一条）。
    pub fn new(x: &'a [T], y: &'a [T], z: &'a [T]) -> Result<Self, Error> {
        expect_len("V3Dyn::new", "y 分量", y.len(), x.len())?;
        expect_len("V3Dyn::new", "z 分量", z.len(), x.len())?;
        Ok(V3Dyn { x, y, z })
    }

    /// 三条分量（顺序固定：x、y、z）。
    pub fn components(&self) -> [&'a [T]; 3] {
        [self.x, self.y, self.z]
    }

    /// 第 `i` 个样本的 `(x, y, z)`。
    ///
    /// # Panics
    /// `i >= len()`。
    pub fn at(&self, i: usize) -> (T, T, T) {
        (self.x[i], self.y[i], self.z[i])
    }
}

impl<T> V3DynMut<'_, T> {
    /// 样本数（运行期）。
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// 样本数为 0。
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }
}

impl<'a, T: Copy> V3DynMut<'a, T> {
    /// 三条分量必须等长（以 `x` 为基准）。
    ///
    /// # Errors
    /// `y`/`z` 与 `x` 长度不一致。
    pub fn new(x: &'a mut [T], y: &'a mut [T], z: &'a mut [T]) -> Result<Self, Error> {
        expect_len("V3DynMut::new", "y 分量", y.len(), x.len())?;
        expect_len("V3DynMut::new", "z 分量", z.len(), x.len())?;
        Ok(V3DynMut { x, y, z })
    }

    /// 借出等长的只读视图。
    pub fn as_ref(&self) -> V3Dyn<'_, T> {
        V3Dyn {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }

    /// 三条可变分量（顺序固定：x、y、z）。
    pub fn components_mut(&mut self) -> [&mut [T]; 3] {
        [self.x, self.y, self.z]
    }
}

impl<T> std::fmt::Debug for V3Dyn<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("V3Dyn").field("n", &self.len()).finish()
    }
}

impl<T> std::fmt::Debug for V3DynMut<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("V3DynMut").field("n", &self.len()).finish()
    }
}

/// 输出三分量必须与输入等长（输出是独立构造的，所以在这里再校验一次）。
fn expect_out3(op: &'static str, out: &V3DynMut<'_, impl Sized>, want: usize) -> Result<(), Error> {
    expect_len(op, "输出分量", out.len(), want)
}

impl V3Dyn<'_, f64> {
    /// 模长 `out[i] = √(x² + y² + z²)`。
    ///
    /// # Errors
    /// `out.len() != self.len()`。
    pub fn norm3_into(&self, out: &mut [f64]) -> Result<(), Error> {
        expect_len("V3Dyn::norm3_into", "out", out.len(), self.len())?;
        crate::ops::norm3_batch::norm3_batch(self.x, self.y, self.z, out);
        Ok(())
    }

    /// 单位化 `o = v / |v|`。
    ///
    /// # Errors
    /// 输出分量数与输入不一致。
    pub fn unitize3_into(&self, out: &mut V3DynMut<'_, f64>) -> Result<(), Error> {
        expect_out3("V3Dyn::unitize3_into", out, self.len())?;
        let [ox, oy, oz] = out.components_mut();
        crate::ops::unitize3_batch::unitize3_batch(self.x, self.y, self.z, ox, oy, oz);
        Ok(())
    }

    /// 叉积 `o = a × b`（`b` 必须与 `self` 等长）。
    ///
    /// # Errors
    /// `b` 或输出分量数与 `self` 不一致。
    pub fn cross_into(&self, b: &V3Dyn<'_, f64>, out: &mut V3DynMut<'_, f64>) -> Result<(), Error> {
        expect_len("V3Dyn::cross_into", "b", b.len(), self.len())?;
        expect_out3("V3Dyn::cross_into", out, self.len())?;
        let [ox, oy, oz] = out.components_mut();
        crate::ops::cross3_batch::cross3_batch(self.x, self.y, self.z, b.x, b.y, b.z, ox, oy, oz);
        Ok(())
    }

    /// 缩放加 `o = a + s·b`。
    ///
    /// # Errors
    /// `b` 或输出分量数与 `self` 不一致。
    pub fn add_scaled_into(
        &self,
        b: &V3Dyn<'_, f64>,
        s: f64,
        out: &mut V3DynMut<'_, f64>,
    ) -> Result<(), Error> {
        expect_len("V3Dyn::add_scaled_into", "b", b.len(), self.len())?;
        expect_out3("V3Dyn::add_scaled_into", out, self.len())?;
        let [ox, oy, oz] = out.components_mut();
        crate::ops::vec3_add_scaled_batch::vec3_add_scaled_batch(
            self.x, self.y, self.z, b.x, b.y, b.z, s, ox, oy, oz,
        );
        Ok(())
    }

    /// 中心项 + J2 加速度 `o = a(r; mu, j2, re)`。
    ///
    /// # Errors
    /// 输出分量数与输入不一致。
    pub fn j2_accel_into(
        &self,
        mu: f64,
        j2: f64,
        re: f64,
        out: &mut V3DynMut<'_, f64>,
    ) -> Result<(), Error> {
        expect_out3("V3Dyn::j2_accel_into", out, self.len())?;
        let [ox, oy, oz] = out.components_mut();
        crate::ops::j2_accel_batch::j2_accel_batch(self.x, self.y, self.z, mu, j2, re, ox, oy, oz);
        Ok(())
    }
}

/// 位置 + 速度的运行期长度**独占**视图（6 条，原地更新）。
pub struct RvDynMut<'a, T> {
    rx: &'a mut [T],
    ry: &'a mut [T],
    rz: &'a mut [T],
    vx: &'a mut [T],
    vy: &'a mut [T],
    vz: &'a mut [T],
}

impl<T> RvDynMut<'_, T> {
    /// 样本数（运行期）。
    pub fn len(&self) -> usize {
        self.rx.len()
    }

    /// 样本数为 0。
    pub fn is_empty(&self) -> bool {
        self.rx.is_empty()
    }
}

impl<'a, T: Copy> RvDynMut<'a, T> {
    /// 六条必须等长（以 `rx` 为基准）。
    ///
    /// # Errors
    /// 其余任一条与 `rx` 长度不一致（[`Error::Shape`]，`what` 点名是哪一条）。
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rx: &'a mut [T],
        ry: &'a mut [T],
        rz: &'a mut [T],
        vx: &'a mut [T],
        vy: &'a mut [T],
        vz: &'a mut [T],
    ) -> Result<Self, Error> {
        let n = rx.len();
        expect_len("RvDynMut::new", "ry 位置分量", ry.len(), n)?;
        expect_len("RvDynMut::new", "rz 位置分量", rz.len(), n)?;
        expect_len("RvDynMut::new", "vx 速度分量", vx.len(), n)?;
        expect_len("RvDynMut::new", "vy 速度分量", vy.len(), n)?;
        expect_len("RvDynMut::new", "vz 速度分量", vz.len(), n)?;
        Ok(RvDynMut {
            rx,
            ry,
            rz,
            vx,
            vy,
            vz,
        })
    }

    /// 位置三分量的只读视图。
    pub fn position(&self) -> V3Dyn<'_, T> {
        V3Dyn {
            x: self.rx,
            y: self.ry,
            z: self.rz,
        }
    }

    /// 速度三分量的只读视图。
    pub fn velocity(&self) -> V3Dyn<'_, T> {
        V3Dyn {
            x: self.vx,
            y: self.vy,
            z: self.vz,
        }
    }

    /// 六条分量的只读切片（顺序：rx, ry, rz, vx, vy, vz）。
    pub fn components(&self) -> [&[T]; 6] {
        [self.rx, self.ry, self.rz, self.vx, self.vy, self.vz]
    }
}

impl RvDynMut<'_, f64> {
    /// 推进一步（中心项 + J2），原地更新位置与速度。
    pub fn rk4_j2_step(&mut self, mu: f64, j2: f64, re: f64, dt: f64) {
        crate::ops::rk4_j2_step_batch::rk4_j2_step_batch(
            self.rx, self.ry, self.rz, self.vx, self.vy, self.vz, mu, j2, re, dt,
        );
    }
}

impl RvDynMut<'_, f32> {
    /// 推进一步（弹道：重力 + 阻力），原地更新位置与速度。
    ///
    /// # Errors
    /// `k.len() != self.len()`。
    pub fn ballistic_step(&mut self, k: &[f32], dt: f32, g: f32) -> Result<(), Error> {
        expect_len("RvDynMut::ballistic_step", "k", k.len(), self.len())?;
        crate::ops::ballistic_step::ballistic_step(
            self.rx, self.ry, self.rz, self.vx, self.vy, self.vz, k, dt, g,
        );
        Ok(())
    }
}

impl<T> std::fmt::Debug for RvDynMut<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RvDynMut").field("n", &self.len()).finish()
    }
}

/// 二分量运行期长度**借用**视图（f32）。
pub struct V2Dyn<'a, T> {
    x: &'a [T],
    y: &'a [T],
}

impl<T> V2Dyn<'_, T> {
    /// 样本数（运行期）。
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// 样本数为 0。
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }
}

impl<'a, T: Copy> V2Dyn<'a, T> {
    /// 两条分量必须等长（以 `x` 为基准）。
    ///
    /// # Errors
    /// `y` 与 `x` 长度不一致。
    pub fn new(x: &'a [T], y: &'a [T]) -> Result<Self, Error> {
        expect_len("V2Dyn::new", "y 分量", y.len(), x.len())?;
        Ok(V2Dyn { x, y })
    }

    /// 两条分量（顺序固定：x、y）。
    pub fn components(&self) -> [&'a [T]; 2] {
        [self.x, self.y]
    }
}

impl V2Dyn<'_, f32> {
    /// 到定点 `(px, py)` 的平面距离。
    ///
    /// # Errors
    /// `out.len() != self.len()`。
    pub fn distance2d_into(&self, px: f32, py: f32, out: &mut [f32]) -> Result<(), Error> {
        expect_len("V2Dyn::distance2d_into", "out", out.len(), self.len())?;
        crate::ops::batch_distance2d::batch_distance2d(px, py, self.x, self.y, out);
        Ok(())
    }
}

impl<T> std::fmt::Debug for V2Dyn<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("V2Dyn").field("n", &self.len()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 分量长度不一致 ⇒ `Err`，且点名是哪一条（以第一条为基准）。
    #[test]
    fn test_len_mismatch_is_error() {
        let x = [0.0f64; 5];
        let short = [0.0f64; 4];
        match V3Dyn::new(&x, &short, &x) {
            Err(Error::Shape {
                op,
                what,
                expected,
                got,
            }) => {
                assert_eq!(op, "V3Dyn::new");
                assert_eq!(what, "y 分量");
                assert_eq!((expected, got), (5, 4));
            }
            other => panic!("应报长度错误，实际 {other:?}"),
        }
        match V3Dyn::new(&x, &x, &short) {
            Err(Error::Shape { what, .. }) => assert_eq!(what, "z 分量"),
            other => panic!("应报长度错误，实际 {other:?}"),
        }
        // 输出长度独立校验
        let v = V3Dyn::new(&x, &x, &x).unwrap();
        let mut out = vec![0.0f64; 3];
        match v.norm3_into(&mut out) {
            Err(Error::Shape {
                op,
                what,
                expected,
                got,
            }) => {
                assert_eq!(op, "V3Dyn::norm3_into");
                assert_eq!(what, "out");
                assert_eq!((expected, got), (5, 3));
            }
            other => panic!("应报输出长度错误，实际 {other:?}"),
        }
    }

    /// 与直接调内核**逐位一致**（运行期长度 37：覆盖向量尾 + 标量尾）。
    #[test]
    fn test_matches_direct_kernel_bitwise() {
        let n = 37usize;
        let mut lcg = crate::ops::testutil::Lcg(0xd1ce);
        let mk = |lcg: &mut crate::ops::testutil::Lcg| -> Vec<f64> {
            (0..n).map(|_| lcg.f64()).collect()
        };
        let (x, y, z) = (mk(&mut lcg), mk(&mut lcg), mk(&mut lcg));
        let (bx, by, bz) = (mk(&mut lcg), mk(&mut lcg), mk(&mut lcg));
        let a = V3Dyn::new(&x, &y, &z).unwrap();
        let b = V3Dyn::new(&bx, &by, &bz).unwrap();

        let mut got = vec![0.0f64; n];
        a.norm3_into(&mut got).unwrap();
        let mut want = vec![0.0f64; n];
        crate::ops::norm3_batch::norm3_batch(&x, &y, &z, &mut want);
        assert_eq!(bits(&got), bits(&want), "norm3");

        let (mut ox, mut oy, mut oz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let mut out = V3DynMut::new(&mut ox, &mut oy, &mut oz).unwrap();
        a.cross_into(&b, &mut out).unwrap();
        let (mut wx, mut wy, mut wz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        crate::ops::cross3_batch::cross3_batch(
            &x, &y, &z, &bx, &by, &bz, &mut wx, &mut wy, &mut wz,
        );
        assert_eq!(
            (bits(&ox), bits(&oy), bits(&oz)),
            (bits(&wx), bits(&wy), bits(&wz)),
            "cross3"
        );

        let mut out = V3DynMut::new(&mut ox, &mut oy, &mut oz).unwrap();
        a.add_scaled_into(&b, -1.75, &mut out).unwrap();
        crate::ops::vec3_add_scaled_batch::vec3_add_scaled_batch(
            &x, &y, &z, &bx, &by, &bz, -1.75, &mut wx, &mut wy, &mut wz,
        );
        assert_eq!(
            (bits(&ox), bits(&oy), bits(&oz)),
            (bits(&wx), bits(&wy), bits(&wz)),
            "add_scaled"
        );

        let mut out = V3DynMut::new(&mut ox, &mut oy, &mut oz).unwrap();
        a.j2_accel_into(3.986e14, 1.0826e-3, 6.378e6, &mut out)
            .unwrap();
        crate::ops::j2_accel_batch::j2_accel_batch(
            &x, &y, &z, 3.986e14, 1.0826e-3, 6.378e6, &mut wx, &mut wy, &mut wz,
        );
        assert_eq!(
            (bits(&ox), bits(&oy), bits(&oz)),
            (bits(&wx), bits(&wy), bits(&wz)),
            "j2_accel"
        );

        let mut out = V3DynMut::new(&mut ox, &mut oy, &mut oz).unwrap();
        a.unitize3_into(&mut out).unwrap();
        crate::ops::unitize3_batch::unitize3_batch(&x, &y, &z, &mut wx, &mut wy, &mut wz);
        assert_eq!(
            (bits(&ox), bits(&oy), bits(&oz)),
            (bits(&wx), bits(&wy), bits(&wz)),
            "unitize3"
        );
    }

    /// RK4 / 弹道 / 2D 距离三条运行期路径与直接调内核逐位一致。
    #[test]
    fn test_rv_and_v2_match_direct_bitwise() {
        let n = 33usize;
        let mut lcg = crate::ops::testutil::Lcg(0x5a17);
        let mut mk = |scale: f64| -> Vec<f64> { (0..n).map(|_| scale * lcg.f64()).collect() };
        let [r0, r1, r2, v0, v1, v2] = [
            mk(7.0e6),
            mk(7.0e6),
            mk(7.0e6),
            mk(7.5e3),
            mk(7.5e3),
            mk(7.5e3),
        ];
        let [mut rx, mut ry, mut rz, mut vx, mut vy, mut vz] = [
            r0.clone(),
            r1.clone(),
            r2.clone(),
            v0.clone(),
            v1.clone(),
            v2.clone(),
        ];
        let [mut wx, mut wy, mut wz, mut ux, mut uy, mut uz] = [r0, r1, r2, v0, v1, v2];
        {
            let mut rv =
                RvDynMut::new(&mut rx, &mut ry, &mut rz, &mut vx, &mut vy, &mut vz).unwrap();
            rv.rk4_j2_step(3.986e14, 1.0826e-3, 6.378e6, 1.0);
        }
        crate::ops::rk4_j2_step_batch::rk4_j2_step_batch(
            &mut wx, &mut wy, &mut wz, &mut ux, &mut uy, &mut uz, 3.986e14, 1.0826e-3, 6.378e6, 1.0,
        );
        for (g, w, name) in [
            (&rx, &wx, "rk4 rx"),
            (&ry, &wy, "rk4 ry"),
            (&rz, &wz, "rk4 rz"),
            (&vx, &ux, "rk4 vx"),
            (&vy, &uy, "rk4 vy"),
            (&vz, &uz, "rk4 vz"),
        ] {
            assert_eq!(bits(g), bits(w), "{name}");
        }

        // f32 弹道
        let mut lcg32 = crate::ops::testutil::Lcg(0xbeef);
        let mut mk32 = |_i: usize| -> Vec<f32> { (0..n).map(|_| lcg32.f64() as f32).collect() };
        let [mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, k] = [
            mk32(0),
            mk32(1),
            mk32(2),
            mk32(3),
            mk32(4),
            mk32(5),
            mk32(6),
        ];
        let [mut b0, mut b1, mut b2, mut b3, mut b4, mut b5] = [
            a0.clone(),
            a1.clone(),
            a2.clone(),
            a3.clone(),
            a4.clone(),
            a5.clone(),
        ];
        {
            let mut rv =
                RvDynMut::new(&mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut a5).unwrap();
            rv.ballistic_step(&k, 0.01, 9.8).unwrap();
        }
        crate::ops::ballistic_step::ballistic_step(
            &mut b0, &mut b1, &mut b2, &mut b3, &mut b4, &mut b5, &k, 0.01, 9.8,
        );
        for (g, w, name) in [
            (&a0, &b0, "ballistic x"),
            (&a1, &b1, "ballistic y"),
            (&a2, &b2, "ballistic z"),
            (&a3, &b3, "ballistic vx"),
            (&a4, &b4, "ballistic vy"),
            (&a5, &b5, "ballistic vz"),
        ] {
            assert_eq!(bits32(g), bits32(w), "{name}");
        }
        // k 长度不符 ⇒ Err
        let mut rv = RvDynMut::new(&mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut a5).unwrap();
        match rv.ballistic_step(&k[..n - 1], 0.01, 9.8) {
            Err(Error::Shape { what, .. }) => assert_eq!(what, "k"),
            other => panic!("应报长度错误，实际 {other:?}"),
        }

        // 2D 距离
        let xs: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let ys: Vec<f32> = (0..n).map(|i| -(i as f32) * 0.25).collect();
        let v2 = V2Dyn::new(&xs, &ys).unwrap();
        let mut got = vec![0.0f32; n];
        v2.distance2d_into(1.0, -1.0, &mut got).unwrap();
        let mut want = vec![0.0f32; n];
        crate::ops::batch_distance2d::batch_distance2d(1.0, -1.0, &xs, &ys, &mut want);
        assert_eq!(bits32(&got), bits32(&want), "distance2d");
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn bits32(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }
}
