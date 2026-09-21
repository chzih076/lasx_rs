//! 库内常驻线程池的用法示例：`y += 2·x`，切成 `threads` 段并行。
//!
//! 运行：`cargo run --release --example pool_axpy`
//!
//! 要点：
//! - 池**建一次、跨调用复用**——每步新建线程的代价摊不掉（见 `docs/perf-report.md` §13.3）；
//! - 走 `rlib` 直接调用，**不经过 C ABI**，因此池不引入任何新导出符号；
//! - 小数据（< [`lasx_rs::pool::MIN_PARALLEL_LEN`]）会自动原地串行，调用方无需分支。

use lasx_rs::aligned::AlignedVec;
use lasx_rs::pool::WorkerPool;

fn main() {
    let n = 1 << 22; // 4 Mi 元素 = 16 MiB，足够压到内存带宽
    let alpha = 2.0f32;

    let mut x = AlignedVec::<f32>::fill_with(n, |i| (i % 97) as f32 - 48.0);
    let mut y = AlignedVec::<f32>::fill_with(n, |_| 0.0);

    // 注意：多数组接口要求 `&mut [T]`——池必须能把各块安全地**独占**交给不同线程，
    // 只读入参也得先 `as_mut_slice()`（这是"独占切分"的代价，也是它不需要 unsafe 的原因）。
    let mut pool = WorkerPool::new(12);
    pool.for_each_chunks_mut([x.as_mut_slice(), y.as_mut_slice()], |[x, y]| {
        let m = x.len() as i32;
        lasx_rs::lasx_axpy(alpha, x.as_ptr(), y.as_mut_ptr(), m);
    });

    // 串行结果逐位对照
    let mut serial = AlignedVec::<f32>::fill_with(n, |_| 0.0);
    lasx_rs::lasx_axpy(alpha, x.as_ptr(), serial.as_mut_ptr(), n as i32);

    let bad = (0..n)
        .filter(|&i| y[i].to_bits() != serial[i].to_bits())
        .count();
    assert_eq!(bad, 0, "并行结果与串行不一致：{bad} 个元素不同");
    println!(
        "池 {} 线程，n = {n}：y += {alpha}·x 与串行逐位一致",
        pool.threads()
    );
}
