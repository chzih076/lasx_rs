//! 计时装置：自适应重复 + 取中位数，以及两种向量口径。

use lasx_rs::lasx_force_lsx_thread;
use std::time::{Duration, Instant};

pub fn median(mut v: Vec<Duration>) -> Duration {
    v.sort_unstable();
    v[v.len() / 2]
}
/// 自适应重复：先估单次耗时，凑到 ~25ms 一件，重复 5 件取中位数
pub fn timeit<F: FnMut()>(mut f: F) -> Duration {
    f();
    f();
    let t = Instant::now();
    f();
    let single = t.elapsed().max(Duration::from_nanos(200));
    let reps =
        (Duration::from_millis(25).as_nanos() / single.as_nanos()).clamp(1, 1_000_000) as u32;
    let mut samples = Vec::with_capacity(5);
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        samples.push(t.elapsed() / reps);
    }
    median(samples)
}
#[derive(Clone, Copy, PartialEq)]
pub enum Mode {
    Lasx,
    Lsx,
}
pub fn time_mode<F: FnMut()>(mode: Mode, f: F) -> Duration {
    lasx_force_lsx_thread(mode == Mode::Lsx);
    let d = timeit(f);
    lasx_force_lsx_thread(false); // 钩子是线程级的，用完必须复位
    d
}
pub fn fmt_t(d: Duration) -> String {
    let us = d.as_secs_f64() * 1e6;
    if us < 1.0 {
        format!("{:.0} ns", us * 1000.0)
    } else if us < 1000.0 {
        format!("{us:.1} µs")
    } else if us < 1e6 {
        format!("{:.2} ms", us / 1e3)
    } else {
        format!("{:.3} s", us / 1e6)
    }
}
pub fn fmt_thr(v: f64) -> String {
    if v >= 1e9 {
        format!("{:.2} G/s", v / 1e9)
    } else if v >= 1e6 {
        format!("{:.1} M/s", v / 1e6)
    } else if v >= 1e3 {
        format!("{:.1} K/s", v / 1e3)
    } else {
        format!("{v:.0} /s")
    }
}
