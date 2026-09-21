//! 基准输入：确定性伪随机数、轨道状态与 SOA 容器。

use crate::scalar_ref::scalar_rk4_batch;
use lasx_rs::{lasx_force_lsx_thread, lasx_rk4_j2_step_batch};

/// 地球引力常数 μ（m³/s²）。
pub const MU: f64 = 3.986_004_418e14;
/// J2 带谐系数。
pub const J2: f64 = 1.082_626_68e-3;
/// 地球赤道半径 Re（m）。
pub const RE: f64 = 6.378_137e6;

pub struct Lcg(pub u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Lcg(seed
            .wrapping_mul(2862933555777941757)
            .wrapping_add(3037000493))
    }
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    pub fn f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
    pub fn f64(&mut self) -> f64 {
        (self.next_u32() as f64 / u32::MAX as f64) * 2.0 - 1.0
    }
    pub fn u8(&mut self) -> u8 {
        (self.next_u32() >> 24) as u8
    }
    pub fn i8(&mut self) -> i8 {
        (self.next_u32() >> 24) as i8
    }
}

/// 伪随机轨道状态（LEO/GEO/椭圆/高轨混合），与 `src/lib.rs` 测试同构
pub fn states(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for i in 0..n {
        let a = 6.8e6 + (i as f64) * 1.7e6;
        let e = 0.05 + 0.3 * (i as f64) / n as f64;
        let th = (i as f64) * 2.39996;
        let r = a * (1.0 - e * e) / (1.0 + e * (th * 1.7).cos());
        let ph = (i as f64) * 1.131;
        x.push(r * th.cos() * ph.cos());
        y.push(r * th.sin() * ph.cos());
        z.push(r * ph.sin());
    }
    (x, y, z)
}

/// 轨道速度（量级 ~7.5 km/s，方向随样本变化）
pub fn velocities(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut vx = Vec::with_capacity(n);
    let mut vy = Vec::with_capacity(n);
    let mut vz = Vec::with_capacity(n);
    for i in 0..n {
        let ph = (i as f64) * 0.713;
        let v = 7.4e3 + 200.0 * ph.sin();
        vx.push(v * ph.cos());
        vy.push(v * ph.sin());
        vz.push(0.05 * v * (ph * 2.0).sin());
    }
    (vx, vy, vz)
}

#[derive(Clone)]
pub struct Soa6 {
    pub rx: Vec<f64>,
    pub ry: Vec<f64>,
    pub rz: Vec<f64>,
    pub vx: Vec<f64>,
    pub vy: Vec<f64>,
    pub vz: Vec<f64>,
}

impl Soa6 {
    pub fn new(
        rx: Vec<f64>,
        ry: Vec<f64>,
        rz: Vec<f64>,
        vx: Vec<f64>,
        vy: Vec<f64>,
        vz: Vec<f64>,
    ) -> Self {
        Soa6 {
            rx,
            ry,
            rz,
            vx,
            vy,
            vz,
        }
    }

    pub fn len(&self) -> usize {
        self.rx.len()
    }

    /// 单次 RK4 步；`force_lsx` 走降级路径
    pub fn step(&mut self, force_lsx: bool) {
        let n = self.rx.len();
        lasx_force_lsx_thread(force_lsx);
        lasx_rk4_j2_step_batch(
            self.rx.as_mut_ptr(),
            self.ry.as_mut_ptr(),
            self.rz.as_mut_ptr(),
            self.vx.as_mut_ptr(),
            self.vy.as_mut_ptr(),
            self.vz.as_mut_ptr(),
            MU,
            J2,
            RE,
            10.0,
            n as i32,
        );
        lasx_force_lsx_thread(false);
    }

    pub fn step_scalar(&mut self) {
        scalar_rk4_batch(
            &mut self.rx,
            &mut self.ry,
            &mut self.rz,
            &mut self.vx,
            &mut self.vy,
            &mut self.vz,
            MU,
            J2,
            RE,
            10.0,
        );
    }
}

/// 64 字节对齐的缓冲区包装。
///
/// LASX 是 256 位访问：若起始地址不落在 32 字节边界上，32 字节的 load/store 会
/// **跨 64 字节缓存行**，实测 `lasx_dot` 因此慢约 38%（n=4096：365 ns → 264 ns）。
/// 普通 `Vec<T>` 只保证 `align_of::<T>()`（f32 为 4、f64 为 8），所以基准里显式
/// 对齐，以免把"调用方缓冲区没对齐"的代价记到内核头上。
///
/// FFI 侧同理：C/Dart 调用者若想让 LASX 跑满，应传入 32 字节对齐的指针
/// （本库的 `lasx_alloc` 已保证）。
pub struct AlignedBuf<T> {
    buf: Vec<T>,
    off: usize,
    len: usize,
}

impl<T: Copy + Default> AlignedBuf<T> {
    const ALIGN: usize = 64;

    /// 分配 `len` 个元素，保证首地址 64 字节对齐。
    pub fn new(len: usize) -> Self {
        let esz = std::mem::size_of::<T>().max(1);
        let extra = Self::ALIGN / esz + 1;
        let buf = vec![T::default(); len + extra];
        let misalign = buf.as_ptr() as usize % Self::ALIGN;
        let off = if misalign == 0 {
            0
        } else {
            (Self::ALIGN - misalign).div_ceil(esz)
        };
        let b = AlignedBuf { buf, off, len };
        debug_assert_eq!(b.as_slice().as_ptr() as usize % Self::ALIGN, 0);
        b
    }

    /// 分配并用 `f(i)` 填充（下标从 0 起，形状与 `(0..len).map(f)` 一致）。
    pub fn fill_with(len: usize, mut f: impl FnMut(usize) -> T) -> Self {
        let mut b = Self::new(len);
        for (i, x) in AlignedBuf::as_mut_slice(&mut b).iter_mut().enumerate() {
            *x = f(i);
        }
        b
    }
}

impl<T> AlignedBuf<T> {
    /// 底层切片视图。
    pub fn as_slice(&self) -> &[T] {
        &self.buf[self.off..self.off + self.len]
    }

    /// 底层可变切片视图。
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.buf[self.off..self.off + self.len]
    }

    /// 首地址（保证 64 字节对齐）。
    pub fn as_ptr(&self) -> *const T {
        AlignedBuf::as_slice(self).as_ptr()
    }

    /// 可写首地址（保证 64 字节对齐）。
    pub fn as_mut_ptr(&mut self) -> *mut T {
        AlignedBuf::as_mut_slice(self).as_mut_ptr()
    }
}

impl<T> std::ops::Deref for AlignedBuf<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        AlignedBuf::as_slice(self)
    }
}

impl<T> std::ops::DerefMut for AlignedBuf<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        AlignedBuf::as_mut_slice(self)
    }
}

impl<T: Copy + Default> Clone for AlignedBuf<T> {
    /// 克隆后**重新计算对齐偏移**——直接复用原 `off` 会因新分配地址不同而失准。
    fn clone(&self) -> Self {
        let mut b = Self::new(self.len);
        b.as_mut_slice().copy_from_slice(self.as_slice());
        b
    }
}
