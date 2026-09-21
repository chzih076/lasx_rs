//! **大集**：工作集远超缓存（>> L3 32 MB）时，各内存型内核离"机器搬运上限"有多远。
//!
//! 这组不用"相对标量"口径——规模一大，标量列与向量列都会被内存带宽压平，比值失去意义。
//! 改用**带宽占比**：
//!
//! ```text
//!   实际 GB/s = 这次调用必须搬运的字节数 / 时间
//!   锚点       = 同规模、同读写比例的"纯搬运"实测 GB/s
//!   占比       = 实际 / 锚点
//! ```
//!
//! 锚点分四种：本机单线程流式只有几 GB/s、多线程才到十几 GB/s，读写比也有影响，
//! 所以必须分开取——单线程只读 / 单线程拷贝 / 12 线程只读 / 12 线程拷贝。
//!
//! 占比接近 100% ⇒ 已贴住搬运上限，只能靠"少搬字节"再优化；
//! 明显偏低 ⇒ 还有别的东西在拖（访存形态、指令发射、归约串行、线程扩展），逐个查。

use crate::data::{states, AlignedBuf, Lcg, Soa6, J2, MU, RE};
use crate::timing::{fmt_t, timeit};
use lasx_rs::pool::WorkerPool;
use lasx_rs::*;
use std::hint::black_box;
use std::sync::Mutex;
use std::time::Duration;

/// 锚点：`(单流只读, 双流只读, 单线程拷贝, 双流字节读, N 线程只读, N 线程拷贝, 6 流 3读3写)`，单位 GB/s。
fn anchors(n: usize, threads: usize) -> (f64, f64, f64, f64, f64, f64, f64) {
    let src = AlignedBuf::<f64>::fill_with(n, |i| i as f64);
    let other = AlignedBuf::<f64>::fill_with(n, |i| (i % 7) as f64);
    let mut dst = AlignedBuf::<f64>::new(n);
    let mut src_mut = src.clone();
    let copy_bytes = 2.0 * n as f64 * 8.0;
    let read_bytes = n as f64 * 8.0;

    // 单线程拷贝（读+写）
    let t = timeit(|| {
        dst.copy_from_slice(&src);
        let _ = black_box(dst[0]);
    });
    let c1 = copy_bytes / t.as_secs_f64() / 1e9;

    // 单线程单流只读（4 路独立累加，否则 LLVM 可能只生成标量依赖链）
    let t = timeit(|| {
        let mut acc = [0.0f64; 4];
        for ch in src.as_chunks::<4>().0 {
            acc[0] += ch[0];
            acc[1] += ch[1];
            acc[2] += ch[2];
            acc[3] += ch[3];
        }
        let _ = black_box(acc);
    });
    let r1 = read_bytes / t.as_secs_f64() / 1e9;

    // 单线程**双流**只读：dot 这类内核要同时读两个数组，DRAM 对多流交替的带宽低于单流
    let t = timeit(|| {
        let mut acc = [0.0f64; 4];
        for (ca, cb) in src.as_chunks::<4>().0.iter().zip(other.as_chunks::<4>().0) {
            acc[0] += ca[0] + cb[0];
            acc[1] += ca[1] + cb[1];
            acc[2] += ca[2] + cb[2];
            acc[3] += ca[3] + cb[3];
        }
        let _ = black_box(acc);
    });
    let r2 = (2.0 * read_bytes) / t.as_secs_f64() / 1e9;

    // 单线程**双流字节读**：量化内核（dot_i8/dot_q4）的真实形态（每元素 1 字节，
    // 两条流）。用 u64 异或折叠：每 8 字节一次载入、无依赖链，纯测访存。
    // 数组字节数与量化内核一致（每流 n 字节）
    let (ba, bb) = (vec![0xA5u8; n], vec![0x5Au8; n]);
    let t = timeit(|| {
        let mut x = 0u64;
        for (ca, cb) in ba.as_chunks::<8>().0.iter().zip(bb.as_chunks::<8>().0) {
            x ^= u64::from_le_bytes(*ca) ^ u64::from_le_bytes(*cb);
        }
        let _ = black_box(x);
    });
    let rb = (2.0 * n as f64) / t.as_secs_f64() / 1e9;

    // 单线程**6 流**（3 读 3 写）：`j2_accel_batch` 这类"3 进 3 出"内核的真实形态。
    // 一个循环体里同时碰 3 个源、3 个目的，测的是"多流 + 读写混合"的合成上限。
    let (mut oa, mut ob, mut oc) = (
        AlignedBuf::<f64>::new(n),
        AlignedBuf::<f64>::new(n),
        AlignedBuf::<f64>::new(n),
    );
    let t = timeit(|| {
        let (s0, s1, s2) = (
            src.as_chunks::<4>().0,
            other.as_chunks::<4>().0,
            src_mut.as_chunks::<4>().0,
        );
        let (d0, d1, d2) = (
            oa.as_chunks_mut::<4>().0,
            ob.as_chunks_mut::<4>().0,
            oc.as_chunks_mut::<4>().0,
        );
        for i in 0..d0.len() {
            d0[i].copy_from_slice(&s0[i]);
            d1[i].copy_from_slice(&s1[i]);
            d2[i].copy_from_slice(&s2[i]);
        }
        let _ = black_box(d0[0]);
    });
    let s6 = (6.0 * n as f64 * 8.0) / t.as_secs_f64() / 1e9;

    if threads <= 1 {
        return (r1, r2, c1, rb, r1, c1, s6);
    }
    let mut pool = WorkerPool::new(threads);
    let t = timeit(|| {
        pool.for_each_chunks_mut([dst.as_mut_slice(), src_mut.as_mut_slice()], |[d, s]| {
            d.copy_from_slice(s)
        });
        let _ = black_box(dst[0]);
    });
    let cn = copy_bytes / t.as_secs_f64() / 1e9;
    let t = timeit(|| {
        let acc = Mutex::new(0.0f64);
        pool.for_each_chunk_mut(src_mut.as_mut_slice(), |chunk| {
            let s: f64 = chunk.iter().sum();
            *acc.lock().unwrap() += s;
        });
        let _ = black_box(*acc.lock().unwrap());
    });
    let rn = read_bytes / t.as_secs_f64() / 1e9;
    (r1, r2, c1, rb, rn, cn, s6)
}

/// 大集分析入口。
pub fn run() {
    let ns64 = [1usize << 22, 1 << 24];
    let ns32 = [1usize << 23, 1 << 25];

    println!("## 大集：工作集远超 L3（32 MB）时的带宽占比");
    println!();
    println!("### 锚点（同规模纯搬运实测）");
    println!();
    println!("| 规模 | 单流只读 | 双流只读 | 双流字节读 | **6 流 3读3写** | 单线程拷贝 | 12 线程只读 | 12 线程拷贝 | 字节/f64 |");
    println!("|---|---|---|---|---|---|---|---|---|");
    let mut anchor = std::collections::HashMap::new();
    for &n in ns64.iter().chain(ns32.iter()) {
        let a = anchors(n, 12);
        anchor.insert(n, a);
        println!(
            "| {n} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | **{:.0}%** |",
            a.0,
            a.1,
            a.3,
            a.6,
            a.2,
            a.4,
            a.5,
            100.0 * a.3 / a.1
        );
    }
    println!();
    println!("### 各内核");
    println!();
    println!("| 内核 | n | 搬运 MB | 实际 GB/s | 锚点 GB/s | 占比 |");
    println!("|---|---|---|---|---|---|");

    // `write_frac`：搬运字节里"写"占的比例（0=全读，0.5=读写各半）；`mt`：用 12 线程锚点
    let push = |kernel: &str, n: usize, bytes: f64, write_frac: f64, mt: bool, t: Duration| {
        let gbs = bytes / t.as_secs_f64() / 1e9;
        let a = anchor[&n];
        // 锚点要同"footprint"：每元素 1 字节的内核（dot_i8/dot_q4）用同规模的**字节**双流读锚点，
        // 否则会拿 8 倍 footprint 的 f64 锚点去比，口径不一致（本表早期版本就犯过这个错）。
        let is_byte = kernel.starts_with("dot_i8") || kernel.starts_with("dot_q4");
        let (read, copy) = if kernel.starts_with("j2_accel_batch") {
            // "3 进 3 出"内核直接用同形态锚点，不再用"读+拷贝"内插
            (a.6, a.6)
        } else if is_byte {
            (a.3, a.3)
        } else if mt {
            (a.4, a.5)
        } else {
            (a.0, a.2)
        };
        let want = read + (copy - read) * write_frac.clamp(0.0, 1.0);
        println!(
            "| {kernel} | {n} | {:.1} | {gbs:.1} | {want:.1} | **{:.0}%** |",
            bytes / 1e6,
            100.0 * gbs / want
        );
    };

    for &n in &ns64 {
        let (x, y, z) = states(n);
        let mut out = AlignedBuf::<f64>::new(n);
        let mut y2 = AlignedBuf::<f64>::new(n);
        let mut z2 = AlignedBuf::<f64>::new(n);

        let t = timeit(|| {
            lasx_norm3_batch(
                x.as_ptr(),
                y.as_ptr(),
                z.as_ptr(),
                out.as_mut_ptr(),
                n as i32,
            );
            let _ = black_box(out[0]);
        });
        push("norm3_batch f64", n, 4.0 * n as f64 * 8.0, 0.25, false, t);

        let t = timeit(|| {
            lasx_vec3_add_scaled_batch(
                x.as_ptr(),
                y.as_ptr(),
                z.as_ptr(),
                x.as_ptr(),
                y.as_ptr(),
                z.as_ptr(),
                0.5,
                out.as_mut_ptr(),
                y2.as_mut_ptr(),
                z2.as_mut_ptr(),
                n as i32,
            );
            let _ = black_box(out[0]);
        });
        push(
            "vec3_add_scaled f64",
            n,
            9.0 * n as f64 * 8.0,
            0.333,
            false,
            t,
        );

        let t = timeit(|| {
            lasx_j2_accel_batch(
                x.as_ptr(),
                y.as_ptr(),
                z.as_ptr(),
                MU,
                J2,
                RE,
                out.as_mut_ptr(),
                y2.as_mut_ptr(),
                z2.as_mut_ptr(),
                n as i32,
            );
            let _ = black_box(out[0]);
        });
        push("j2_accel_batch f64", n, 6.0 * n as f64 * 8.0, 0.5, false, t);

        let mut soa = Soa6::new(
            x.clone(),
            y.clone(),
            z.clone(),
            x.clone(),
            y.clone(),
            z.clone(),
        );
        let t = timeit(|| {
            soa.step(false);
            let _ = black_box(soa.rx[0]);
        });
        push("rk4_j2_step f64", n, 12.0 * n as f64 * 8.0, 0.5, false, t);

        let t = timeit(|| {
            let _ = black_box(lasx_dot_f64(x.as_ptr(), y.as_ptr(), n as i32));
        });
        push("dot_f64", n, 2.0 * n as f64 * 8.0, 0.0, false, t);

        let mut pool = WorkerPool::new(12);
        let mut soa2 = Soa6::new(
            x.clone(),
            y.clone(),
            z.clone(),
            x.clone(),
            y.clone(),
            z.clone(),
        );
        let t = timeit(|| {
            soa2.step_pooled(&mut pool);
            let _ = black_box(soa2.rx[0]);
        });
        push(
            "rk4_j2_step f64 ×12 线程",
            n,
            12.0 * n as f64 * 8.0,
            0.5,
            true,
            t,
        );
    }

    for &n in &ns32 {
        let mut rng = Lcg::new(0x1a2b);
        let a = AlignedBuf::<f32>::fill_with(n, |_| rng.f32());
        let b = AlignedBuf::<f32>::fill_with(n, |_| rng.f32());
        let mut c = AlignedBuf::<f32>::new(n);

        let t = timeit(|| {
            let _ = black_box(lasx_dot(a.as_ptr(), b.as_ptr(), n as i32));
        });
        push("dot f32", n, 2.0 * n as f64 * 4.0, 0.0, false, t);

        let t = timeit(|| {
            let _ = black_box(lasx_sum(a.as_ptr(), n as i32));
        });
        push("sum f32", n, n as f64 * 4.0, 0.0, false, t);

        c.copy_from_slice(&a);
        let t = timeit(|| {
            lasx_axpy(2.0, a.as_ptr(), c.as_mut_ptr(), n as i32);
            let _ = black_box(c[0]);
        });
        push("axpy f32", n, 3.0 * n as f64 * 4.0, 0.333, false, t);

        let ai: AlignedBuf<i8> = AlignedBuf::fill_with(n, |i| (i % 127) as i8);
        let bi: AlignedBuf<i8> = AlignedBuf::fill_with(n, |i| (i % 89) as i8);
        let t = timeit(|| {
            let _ = black_box(lasx_dot_i8(ai.as_ptr(), bi.as_ptr(), n as i32));
        });
        push("dot_i8", n, 2.0 * n as f64, 0.0, false, t);

        let qa: AlignedBuf<u8> = AlignedBuf::fill_with(n, |i| (i % 251) as u8);
        let qb: AlignedBuf<u8> = AlignedBuf::fill_with(n, |i| (i % 241) as u8);
        let sa = AlignedBuf::<f32>::fill_with(n.div_ceil(32), |_| 0.01);
        let sb = AlignedBuf::<f32>::fill_with(n.div_ceil(32), |_| 0.02);
        let t = timeit(|| {
            let _ = black_box(lasx_dot_q4(
                qa.as_ptr(),
                sa.as_ptr(),
                qb.as_ptr(),
                sb.as_ptr(),
                n as i32,
            ));
        });
        push("dot_q4", n, 2.0 * n as f64, 0.0, false, t);
    }

    println!();
    println!("### 矩阵乘（算力型：报 GFLOP/s 与\"必要搬运\"的 GB/s）");
    println!();
    println!("| 形状 | 时间 | GFLOP/s | 必要搬运 MB | 搬运 GB/s |");
    println!("|---|---|---|---|---|");
    for &s in &[1024usize, 2048] {
        let mut rng = Lcg::new(0x5eed);
        let a = AlignedBuf::<f32>::fill_with(s * s, |_| rng.f32());
        let b = AlignedBuf::<f32>::fill_with(s * s, |_| rng.f32());
        let mut c = AlignedBuf::<f32>::new(s * s);
        let t = timeit(|| {
            lasx_matmul(
                s as i32,
                s as i32,
                s as i32,
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
            );
            let _ = black_box(c[0]);
        });
        let bytes = 3.0 * (s * s) as f64 * 4.0;
        let flop = 2.0 * (s as f64).powi(3);
        println!(
            "| {s}³ | {} | **{:.1}** | {:.1} | {:.1} |",
            fmt_t(t),
            flop / t.as_secs_f64() / 1e9,
            bytes / 1e6,
            bytes / t.as_secs_f64() / 1e9
        );
    }
}
