//! 测试共用的夹具与度量（仅在 `cfg(test)` 下编译）。
//!
//! 这里的参考实现是**独立**于算子内部标量兜底的另一套写法，故意不复用，
//! 避免"自己证明自己"（见手册 §7.3）。

use crate::lasx_force_lsx_thread;

/// 伪随机轨道状态（LEO/GEO/椭圆/高轨混合），避免巧合。
pub fn states(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for i in 0..n {
        let a = 6.8e6 + (i as f64) * 1.7e6; // 500km .. 12万 km
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

/// 相对误差（`|a−b| / max(1,|b|)`）：对 ~1e7 量级轨道状态用相对度量。
pub fn rel_err(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1.0)
}

/// 相对误差（归约类内核用；对接近 0 的和设分母下限 1）。
pub fn rel(got: f64, want: f64) -> f64 {
    (got - want).abs() / want.abs().max(1.0)
}

/// 线性同余伪随机数发生器（确定性，便于复现）。
pub struct Lcg(pub u64);

impl Lcg {
    /// 均匀分布于 `[-1, 1)` 的伪随机 `f64`。
    pub fn f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
    }
}

/// f64 精确累加参考：`C = A·B`（行主序）。
pub fn reference<T: Copy + Into<f64>>(m: usize, k: usize, n: usize, a: &[T], b: &[T]) -> Vec<f64> {
    (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k)
                .map(|p| a[i * k + p].into() * b[p * n + j].into())
                .sum()
        })
        .collect()
}

/// 归约类内核的测试规模集合（含块边界与质数规模）。
pub const NS: &[usize] = &[
    0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511,
    512, 1000, 4095, 4096, 65536,
];

/// 确定性 f32 输入对。
pub fn data(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut s = 0x243f_6a88_85a3_08d3u64;
    let mut next = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as f64 / (1u64 << 31) as f64) as f32 * 2.0 - 1.0
    };
    let a: Vec<f32> = (0..n).map(|_| next()).collect();
    let b: Vec<f32> = (0..n).map(|_| next()).collect();
    (a, b)
}

/// 同一份输入分别走 LASX 与强制 LSX，两路都必须贴近 f64 参考。
pub fn both_paths<F: Fn() -> f32>(f: F) -> (f64, f64) {
    lasx_force_lsx_thread(false);
    let lasx = f() as f64;
    lasx_force_lsx_thread(true);
    let lsx = f() as f64;
    lasx_force_lsx_thread(false);
    (lasx, lsx)
}
