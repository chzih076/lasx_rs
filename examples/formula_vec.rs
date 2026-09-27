//! 公式 DSL 的**一维形态**：`dot!` 与 `gemv!`（`matmul!` 之外的部分）。
//!
//! `cargo run --release --example formula_vec`
//!
//! 三条路径都与对应的 `api` 调用**逐位一致**（示例里内联断言）。

use lasx_rs::shape::{F16Mat, F16Vec, VecBuf, VecRef};
use lasx_rs::{dot, gemv};

fn main() -> Result<(), lasx_rs::api::Error> {
    // ---- ① f32 / f64 点积：`VecRef` 的长度进类型，形状错了是编译错误 ----
    const K: usize = 33; // 跨 16/32 边界，会走标量的尾部
    const N: usize = 4;

    let a32: Vec<f32> = (0..K).map(|i| (i as f32).mul_add(0.25, -3.0)).collect();
    let b32: Vec<f32> = (0..K).map(|i| (i as f32).mul_add(-0.5, 7.0)).collect();
    let a = VecRef::<f32, K>::new(&a32)?;
    let b = VecRef::<f32, K>::new(&b32)?;
    let s32: f32 = dot!(a[K] * b[K]);
    assert_eq!(s32.to_bits(), lasx_rs::api::dot(&a32, &b32)?.to_bits());

    let a64: Vec<f64> = (0..K).map(|i| (i as f64).mul_add(0.125, -1.5)).collect();
    let b64: Vec<f64> = (0..K).map(|i| (i as f64).mul_add(-0.25, 2.5)).collect();
    let a64v = VecRef::<f64, K>::new(&a64)?;
    let b64v = VecRef::<f64, K>::new(&b64)?;
    let s64: f64 = dot!(a64v[K] * b64v[K]);
    assert_eq!(s64.to_bits(), lasx_rs::api::dot_f64(&a64, &b64)?.to_bits());

    // ---- ② f16 权重 · f32 向量：权重用 u16 位型，两个方向都认 ----
    let w_bits: Vec<u16> = (0..N * K).map(|i| 0x3c00 + (i as u16 % 5)).collect();
    let wv = F16Vec::<K>::new(&w_bits[..K])?;
    let s16: f32 = dot!(wv[K] * a[K]);
    assert_eq!(
        s16.to_bits(),
        lasx_rs::api::dot_f16(&w_bits[..K], &a32)?.to_bits()
    );
    let s16_rev: f32 = dot!(a[K] * wv[K]); // 反序：同一个 impl 的另一侧
    assert_eq!(s16_rev.to_bits(), s16.to_bits());

    // ---- ③ f16 GEMV：`y[N] = w[N, K] * x[K]`（分配 / 写进缓冲 / 池化）----
    let w = F16Mat::<N, K>::new(&w_bits)?;
    let y: VecBuf<f32, N> = gemv!(w[N, K] * a[K]);
    let want = lasx_rs::api::gemv_f16(N, K, &w_bits, &a32)?;
    for r in 0..N {
        assert_eq!(y.as_slice()[r].to_bits(), want[r].to_bits(), "行 {r}");
    }

    // 写进已有缓冲（零分配）
    let mut y2 = VecBuf::<f32, N>::new();
    gemv!(y2[N] = w[N, K] * a[K]);
    assert_eq!(y2.as_slice(), y.as_slice());

    // 池化：与单线程逐位一致（切块不影响任何一行）
    let pool = lasx_rs::pool::WorkerPool::new(4);
    let mut y3 = VecBuf::<f32, N>::new();
    w.gemv_pooled(&pool, &a, &mut y3)?;
    assert_eq!(y3.as_slice(), y.as_slice());

    println!(
        "dot f32 = {s32:.4} | dot f64 = {s64:.4} | dot f16·f32 = {s16:.4}\n\
         gemv 前两行 = {:?}（单线程 == 池化，逐位）",
        &y.as_slice()[..2]
    );
    Ok(())
}
