//! "怎么调用"的样子：权重固定、反复算 `A·B`。
//!
//! 这里用**运行期形状**的接口（[`lasx_rs::api`] / [`lasx_rs::plan`] / [`lasx_rs::view`]）；
//! 形状编译期已知时看 `examples/formula_dsl.rs`（`shape` 层 + `matmul!` 公式 DSL）。
//!
//! `cargo run --release --example plan_reuse`

use lasx_rs::api;
use lasx_rs::plan::MatmulPlan;
use lasx_rs::view::MatRef;
use std::sync::Arc;

fn main() -> Result<(), api::Error> {
    let (m, k, n) = (64usize, 256usize, 256usize);
    let w: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32 - 6.0) * 0.25).collect();
    let x: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();

    // ① 只算一次：形状三个数字给清楚，输出自己分配
    let y1 = api::matmul(m, k, n, &x, &w)?;

    // ② 同一个权重算很多次：先打包一次，之后每次只要 A
    let plan = MatmulPlan::from_row_major(&w, k, n)?;
    let y2 = plan.run(&x)?;
    assert_eq!(&y1[..], &y2[..]); // 逐位一致

    // ③ 权重本来就是列主序（预先转置好的）：不用先转成行主序
    let mut wt = vec![0f32; k * n];
    for p in 0..k {
        for j in 0..n {
            wt[j * k + p] = w[p * n + j];
        }
    }
    let plan_t = MatmulPlan::new(&MatRef::col_major(&wt, k, n)?)?;
    let y3 = plan_t.run(&x)?;
    assert_eq!(&y2[..], &y3[..]);

    // ④ 输出写进自己的缓冲（每帧不再分配）+ 多线程共享同一份打包好的权重
    let plan = Arc::new(plan);
    let mut y4 = vec![0f32; m * n];
    std::thread::scope(|s| {
        let (top, bottom) = y4.split_at_mut((m / 2) * n);
        for (rows, out) in [(0..m / 2, top), (m / 2..m, bottom)] {
            let plan = Arc::clone(&plan);
            let x = &x[rows.start * k..rows.end * k];
            s.spawn(move || plan.run_into(x, out).unwrap());
        }
    });
    assert_eq!(&y2[..], &y4[..]);

    println!(
        "① {:.1}  ② {:.1}  ③ {:.1}  ④ {:.1}",
        y1[0], y2[0], y3[0], y4[0]
    );
    println!("plan: {plan:?}");
    Ok(())
}
