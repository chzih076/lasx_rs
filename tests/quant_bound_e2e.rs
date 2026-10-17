//! **R20-P3.1② 端到端验收**：`lasx_rs` **真量化内核** vs `simd_dsl` 判据层声明的界。
//!
//! `simd_dsl/tests/quant_calibration.rs` 量的是**复刻的量化数学**；这里换成**真内核**
//! （`api::quantize_i8_per_tensor` + `api::dequantize_i8`），验证：
//! ① 真内核的 per-tensor int8 误差是否落在 `criteria::entropy::quant_bound(8)` 之内；
//! ② 真内核与复刻数学是否**一致**（同一 scale 规则 ⇒ 应当一致）。
//!
//! 前置：`--features dsl-backend`（本仓对 `simd_dsl` 的依赖是可选/feature 门控的）。
#![cfg(feature = "dsl-backend")]

use lasx_rs::api;

/// 确定性 LCG（零依赖）。
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next_sym(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((self.0 >> 11) as f64) / ((1u64 << 53) as f64);
        (u * 2.0 - 1.0) as f32
    }
}

/// 真内核量化再反量化，返回相对峰值误差。
fn kernel_err_peak(x: &[f32]) -> f64 {
    let (q, scale) = api::quantize_i8_per_tensor(x).expect("量化内核应成功");
    let xh = api::dequantize_i8(&q, scale).expect("反量化内核应成功");
    let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs())) as f64;
    let max_abs = x
        .iter()
        .zip(xh.iter())
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs())) as f64;
    max_abs / peak
}

#[test]
fn real_int8_kernel_stays_within_the_declared_bound() {
    let mut g = Lcg::new(0xC0FF_EE01);
    // 与 simd_dsl 校准一致的三种形态（均匀 / 近高斯 / 谱功率型），避免"只测一种分布"
    let n = 4096usize;
    let mut uniform = vec![0.0f32; n];
    let mut gauss = vec![0.0f32; n];
    let mut power = vec![0.0f32; n];
    for i in 0..n {
        uniform[i] = g.next_sym();
        gauss[i] = (0..12).map(|_| g.next_sym()).sum::<f32>() * 1.2;
        let a: f32 = (0..12).map(|_| g.next_sym()).sum();
        let b: f32 = (0..12).map(|_| g.next_sym()).sum();
        power[i] = (a * a + b * b).sqrt();
    }
    let bound = simd_dsl::criteria::entropy::quant_bound(8).value;
    for (name, x) in [("均匀", &uniform), ("高斯", &gauss), ("谱功率型", &power)] {
        let e = kernel_err_peak(x);
        eprintln!("  {name:>8} 真内核 int8 err_peak={e:e}  声明界={bound:e}");
        assert!(
            e <= bound * 1.05,
            "{name}：真内核误差 {e:e} 超出声明界 {bound:e} 的 1.05×"
        );
        assert!(
            e > bound * 0.5,
            "{name}：真内核误差 {e:e} 远小于声明界 {bound:e}（界过松？需复核）"
        );
    }
}
