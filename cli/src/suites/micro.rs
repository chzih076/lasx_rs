//! 微基准：纯寄存器 FMA 吞吐与多线程扩展性。

use crate::data::{states, velocities, Soa6, J2, MU, RE};
use crate::timing::{fmt_t, timeit};
use lasx_rs::lasx_force_lsx_thread;
use lasx_rs::*;
use std::arch::loongarch64::*;
use std::hint::black_box;
use std::time::Duration;

/// 一个 SOA 样本块：位置与速度各三分量。
type Soa6Chunk<'a> = (
    &'a mut [f64],
    &'a mut [f64],
    &'a mut [f64],
    &'a mut [f64],
    &'a mut [f64],
    &'a mut [f64],
);

/// FMA 吞吐微基准的迭代次数。
const FMA_ITERS: usize = 20_000_000;

/// 把 n 个样本按线程数切成对齐到 4 样本（32 字节）的块并并行推进一步
pub fn rk4_parallel(buf: &mut Soa6, threads: usize) {
    let n = buf.len();
    let chunk = n.div_ceil(threads).next_multiple_of(4).max(4);

    let rxv: Vec<&mut [f64]> = buf.rx.chunks_mut(chunk).collect();
    let ryv: Vec<&mut [f64]> = buf.ry.chunks_mut(chunk).collect();
    let rzv: Vec<&mut [f64]> = buf.rz.chunks_mut(chunk).collect();
    let vxv: Vec<&mut [f64]> = buf.vx.chunks_mut(chunk).collect();
    let vyv: Vec<&mut [f64]> = buf.vy.chunks_mut(chunk).collect();
    let vzv: Vec<&mut [f64]> = buf.vz.chunks_mut(chunk).collect();

    let parts: Vec<Soa6Chunk<'_>> = rxv
        .into_iter()
        .zip(ryv)
        .zip(rzv)
        .zip(vxv)
        .zip(vyv)
        .zip(vzv)
        .map(|(((((rx, ry), rz), vx), vy), vz)| (rx, ry, rz, vx, vy, vz))
        .collect();

    std::thread::scope(|s| {
        for (rx, ry, rz, vx, vy, vz) in parts {
            s.spawn(move || {
                let m = rx.len() as i32;
                lasx_force_lsx_thread(false);
                lasx_rk4_j2_step_batch(
                    rx.as_mut_ptr(),
                    ry.as_mut_ptr(),
                    rz.as_mut_ptr(),
                    vx.as_mut_ptr(),
                    vy.as_mut_ptr(),
                    vz.as_mut_ptr(),
                    MU,
                    J2,
                    RE,
                    10.0,
                    m,
                );
            });
        }
    });
}

pub fn thread_scaling() {
    let n = 1 << 18;
    let (rx, ry, rz) = states(n);
    let (vx, vy, vz) = velocities(n);
    let base = Soa6::new(rx, ry, rz, vx, vy, vz);

    let t1 = {
        let mut b = base.clone();
        timeit(|| {
            b.step(false);
            let _ = black_box(b.rx[0]);
        })
    };

    println!();
    println!("## 多线程扩展性：`lasx_rk4_j2_step_batch`（n = 2^18 = {n}，原地单步）");
    println!();
    println!("| 线程数 | 单步耗时 | 相对 1 线程 | 扩展效率 |");
    println!("|---|---|---|---|");
    println!("| 1 | {} | 1.00× | 100% |", fmt_t(t1));

    for &th in &[2usize, 4, 6, 8, 12, 24] {
        let mut b = base.clone();
        let d = timeit(|| {
            rk4_parallel(&mut b, th);
            let _ = black_box(b.rx[0]);
        });
        let sp = t1.as_secs_f64() / d.as_secs_f64();
        println!(
            "| {th} | {} | {sp:.2}× | {:.0}% |",
            fmt_t(d),
            100.0 * sp / th as f64
        );
    }

    let ov = spawn_overhead(24);
    println!();
    println!(
        "> 参考：24 次线程创建/回收的固定开销 ≈ {}（上表 24 线程行含此开销）",
        fmt_t(ov)
    );
}

#[inline(never)]
pub fn fma_peak_lasx() -> f64 {
    unsafe {
        let a: m256 = std::mem::transmute(lasx_xvreplgr2vr_w(0x3f80_0000));
        let b: m256 = std::mem::transmute(lasx_xvreplgr2vr_w(0x3f00_0000));
        let (mut v0, mut v1, mut v2, mut v3) = (a, a, a, a);
        let (mut v4, mut v5, mut v6, mut v7) = (a, a, a, a);
        for _ in 0..FMA_ITERS {
            v0 = lasx_xvfmadd_s(a, b, v0);
            v1 = lasx_xvfmadd_s(a, b, v1);
            v2 = lasx_xvfmadd_s(a, b, v2);
            v3 = lasx_xvfmadd_s(a, b, v3);
            v4 = lasx_xvfmadd_s(a, b, v4);
            v5 = lasx_xvfmadd_s(a, b, v5);
            v6 = lasx_xvfmadd_s(a, b, v6);
            v7 = lasx_xvfmadd_s(a, b, v7);
        }
        let s = lasx_xvfadd_s(
            lasx_xvfadd_s(v0, v1),
            lasx_xvfadd_s(
                lasx_xvfadd_s(v2, v3),
                lasx_xvfadd_s(lasx_xvfadd_s(v4, v5), lasx_xvfadd_s(v6, v7)),
            ),
        );
        let mut tmp = [0f32; 8];
        lasx_xvst(
            std::mem::transmute::<m256, m256i>(s),
            tmp.as_mut_ptr() as *mut i8,
            0,
        );
        black_box(tmp.iter().sum::<f32>()) as f64
    }
}

#[inline(never)]
pub fn fma_peak_lsx() -> f64 {
    unsafe {
        let a: m128 = std::mem::transmute(lsx_vreplgr2vr_w(0x3f80_0000));
        let b: m128 = std::mem::transmute(lsx_vreplgr2vr_w(0x3f00_0000));
        let (mut v0, mut v1, mut v2, mut v3) = (a, a, a, a);
        let (mut v4, mut v5, mut v6, mut v7) = (a, a, a, a);
        for _ in 0..FMA_ITERS {
            v0 = lsx_vfmadd_s(a, b, v0);
            v1 = lsx_vfmadd_s(a, b, v1);
            v2 = lsx_vfmadd_s(a, b, v2);
            v3 = lsx_vfmadd_s(a, b, v3);
            v4 = lsx_vfmadd_s(a, b, v4);
            v5 = lsx_vfmadd_s(a, b, v5);
            v6 = lsx_vfmadd_s(a, b, v6);
            v7 = lsx_vfmadd_s(a, b, v7);
        }
        let s = lsx_vfadd_s(
            lsx_vfadd_s(v0, v1),
            lsx_vfadd_s(
                lsx_vfadd_s(v2, v3),
                lsx_vfadd_s(lsx_vfadd_s(v4, v5), lsx_vfadd_s(v6, v7)),
            ),
        );
        let mut tmp = [0f32; 4];
        lsx_vst(
            std::mem::transmute::<m128, m128i>(s),
            tmp.as_mut_ptr() as *mut i8,
            0,
        );
        black_box(tmp.iter().sum::<f32>()) as f64
    }
}

pub fn fma_peak() {
    // FLOP = 累加器数 × 通道数 × 迭代数 × 2（乘 + 加）
    let flops_lasx = (8 * 8 * FMA_ITERS * 2) as f64;
    let flops_lsx = (8 * 4 * FMA_ITERS * 2) as f64;

    let t_lasx = timeit(|| {
        let _ = black_box(fma_peak_lasx());
    });
    let t_lsx = timeit(|| {
        let _ = black_box(fma_peak_lsx());
    });

    let g_lasx = flops_lasx / t_lasx.as_secs_f64() / 1e9;
    let g_lsx = flops_lsx / t_lsx.as_secs_f64() / 1e9;
    println!();
    println!("## 纯寄存器 FMA 吞吐（无内存访问，8 条独立累加链）");
    println!();
    println!("| 路径 | 宽度 | FLOP/s | 相对峰值 |");
    println!("|---|---|---|---|");
    println!("| LASX | 256 位（8×f32） | {g_lasx:.2} GFLOP/s | 1.00× |");
    println!(
        "| LSX | 128 位（4×f32） | {g_lsx:.2} GFLOP/s | {:.2}× |",
        g_lasx / g_lsx
    );
    println!();
    println!(
        "> LASX/LSX 吞吐比 = **{:.2}×**（若 LA664 的 256 位 FMA 为满宽实现应接近 2.00×）",
        g_lasx / g_lsx
    );
}

/// 线程创建/回收的固定开销（用于校正 24 线程扩展比）
pub fn spawn_overhead(threads: usize) -> Duration {
    timeit(|| {
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    let _ = black_box(1u32);
                });
            }
        });
    })
}
