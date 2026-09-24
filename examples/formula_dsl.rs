//! 公式 DSL 的最小用例（也是 `-Zunpretty=expanded` 的观察对象）。
//!
//! `cargo run --release --example formula_dsl`

use lasx_rs::matmul;
use lasx_rs::shape::{Mat, MatBuf, MatBufDyn, MatDyn};

fn main() -> Result<(), lasx_rs::api::Error> {
    const M: usize = 64;
    const K: usize = 256;
    const N: usize = 256;

    let weights: Vec<f32> = (0..K * N).map(|i| (i % 13) as f32).collect();
    let batch: Vec<f32> = (0..M * K).map(|i| (i % 7) as f32).collect();

    let w = Mat::<f32, { K }, { N }>::new(&weights)?;
    let x = Mat::<f32, { M }, { K }>::new(&batch)?;

    // 全静态
    let mut y = MatBuf::<f32, { M }, { N }>::new();
    matmul!(y[M, N] = x[M, K] * w[K, N]);

    // DYN（行下标小写 = 运行期值）
    let xd = MatDyn::<f32, { K }>::new(&batch)?;
    let mut yd = MatBufDyn::<f32, { N }>::with_rows(M);
    matmul!(yd[m, N] = xd[m, K] * w[K, N]);

    assert_eq!(y.as_slice(), yd.as_slice());
    println!("两种形态逐位一致：y[0] = {:.3}", y.as_slice()[0]);
    Ok(())
}
