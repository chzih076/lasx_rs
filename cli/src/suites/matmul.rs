use crate::data::AlignedBuf;
use crate::data::Lcg;
use crate::report::{Row, row3};
use crate::scalar_ref::*;
use crate::timing::{Mode, time_mode, timeit};
use lasx_rs::api;
use lasx_rs::*;
use std::hint::black_box;

pub fn matmul(rows: &mut Vec<Row>) {
    // A/B 臂模式：`-- matmul a` / `-- matmul b` 只跑融合对照的一段（供**进程级交替**用，
    // 不渲染大表）。见 [`fusion_ab`]。
    if let Some(arm) = std::env::args().nth(2) {
        fusion_ab(&arm);
        return;
    }
    // 256³ 以下打包/流式基本持平，384³ 起打包明显更优。
    for &(m, k, n) in &[
        (64usize, 64usize, 64usize),
        (128, 128, 128),
        (256, 256, 256),
        (384, 384, 384),
        (512, 512, 512),
    ] {
        let mut rng = Lcg::new((m * 1000 + k * 10 + n) as u64);
        let a = AlignedBuf::fill_with(m * k, |_| rng.f32());
        let b = AlignedBuf::fill_with(k * n, |_| rng.f32());
        let mut c = AlignedBuf::new(m * n);
        let tag = format!("{m}×{k}×{n}");
        let lasx = time_mode(Mode::Lasx, || {
            lasx_matmul(
                m as i32,
                k as i32,
                n as i32,
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
            );
            let _ = black_box(c[0]);
        });
        let scalar = timeit(|| {
            scalar_matmul(m, k, n, &a, &b, &mut c);
            let _ = black_box(c[0]);
        });
        row3(
            "lasx_matmul f32(LASX-only)",
            tag,
            2.0 * (m * k * n) as f64,
            "FLOP/s",
            lasx,
            None,
            scalar,
            rows,
        );
    }

    // f64 的打包阈值是 k ≥ 192 / work ≥ 8M：64³、128³ 都在阈值下（走流式），
    // 加 256³、512³ 才能看到打包 + k 分块的收益。
    for &(m, k, n) in &[
        (64usize, 64usize, 64usize),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
    ] {
        let mut rng = Lcg::new((m * 7 + k * 13 + n) as u64);
        let a = AlignedBuf::fill_with(m * k, |_| rng.f64());
        let b = AlignedBuf::fill_with(k * n, |_| rng.f64());
        let mut c = AlignedBuf::new(m * n);
        let tag = format!("{m}×{k}×{n}");
        let lasx = time_mode(Mode::Lasx, || {
            lasx_matmul_f64(
                m as i32,
                k as i32,
                n as i32,
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
            );
            let _ = black_box(c[0]);
        });
        let scalar = timeit(|| {
            scalar_matmul_f64(m, k, n, &a, &b, &mut c);
            let _ = black_box(c[0]);
        });
        row3(
            "lasx_matmul_f64(LASX-only)",
            tag,
            2.0 * (m * k * n) as f64,
            "FLOP/s",
            lasx,
            None,
            scalar,
            rows,
        );
    }
}

/// `C = alpha·A·B + beta·C` 的 A/B 臂——**融合（输出那一遍里缩放）vs 朴素两遍**。
///
/// 用法（`a` = 融合臂，`b` = 两遍臂；**两臂必须是两个进程**，交替跑）：
///
/// ```text
/// target/release/lasx_bench matmul a
/// target/release/lasx_bench matmul b
/// ```
///
/// 每行是机器可读的 `AB arm=<a|b> case=<m>x<k>x<n> alpha=<…> beta=<…> us=<…> loadavg=<…> same=…`，
/// 其中 `us` 是 [`timeit`] 的口径（预热 2 次、自适应重复到 ~25 ms/样品、5 取样取中位数），
/// `loadavg` 是该进程启动时的 `/proc/loadavg` 第一项，`same` 是"与未融合参考
/// `beta.mul_add(C_old, alpha * acc)` 逐位一致"的结果。
///
/// **纪律**（`docs/dev.md` §6.4）：两臂交替起进程、≥3 轮、丢弃首轮、取中位数并报极差；
/// `load > 3` 的轮次只能当趋势、不得当验收。
///
/// 两遍臂就是"先把 `A·B` 算进独立缓冲，再整块做 `C = beta·C_old + alpha·acc`"——
/// 这正是融合要取代的那一遍；融合臂调 [`api::matmul_scaled_into`]（被测的公开入口）。
///
/// 五条 case 覆盖：宽薄打包（`k = 64`，`beta = 0` 与 `beta = 1`）、极宽列块（`n = 19147`）、
/// 方阵 + k 分块（`k = 512 > 384`，此时 `beta != 0` 无法融合、退回二遍法）。
pub fn fusion_ab(arm: &str) {
    let fused = match arm {
        "a" => true,
        "b" => false,
        other => {
            println!("（臂名只能是 a（融合）或 b（两遍），收到 `{other}`）");
            return;
        }
    };
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    let load = load.split_whitespace().next().unwrap_or("?").to_string();
    for &(m, k, n, alpha, beta) in &[
        (256usize, 64usize, 4096usize, 1.5f32, 0.0f32),
        (100, 64, 19147, 1.5, 0.0),
        (256, 64, 4096, 1.5, 1.0),
        (512, 512, 512, 1.5, 0.0),
        (512, 512, 512, 1.5, 1.0),
    ] {
        let mut rng = Lcg::new((m as u64) * 31 + (k as u64) * 7 + n as u64);
        let a = AlignedBuf::fill_with(m * k, |_| rng.f32());
        let b = AlignedBuf::fill_with(k * n, |_| rng.f32());
        // `beta != 0` 的原地累加：两臂都从同一份 C 出发（计时里会被反复改写，数值会漂，
        // 但两臂漂的是同一个量级，不影响时间口径）。
        let mut c = AlignedBuf::fill_with(m * n, |_| 0.25);
        let mut acc = AlignedBuf::new(m * n);

        // 正确性：先与"未融合参考"逐位比（`acc` 走同一条 `lasx_matmul`）。
        let want = {
            let mut want = AlignedBuf::<f32>::fill_with(m * n, |_| 0.25);
            lasx_matmul(
                m as i32,
                k as i32,
                n as i32,
                a.as_ptr(),
                b.as_ptr(),
                acc.as_mut_ptr(),
            );
            let (ws, acs) = (want.as_mut_slice(), acc.as_slice());
            for i in 0..m * n {
                ws[i] = beta.mul_add(ws[i], alpha * acs[i]);
            }
            if fused {
                let mut got = AlignedBuf::<f32>::fill_with(m * n, |_| 0.25);
                api::matmul_scaled_into(
                    m,
                    k,
                    n,
                    alpha,
                    beta,
                    a.as_slice(),
                    b.as_slice(),
                    got.as_mut_slice(),
                )
                .unwrap();
                let gs = got.as_slice();
                assert!(
                    gs.iter()
                        .zip(want.iter())
                        .all(|(x, y)| x.to_bits() == y.to_bits()),
                    "{m}×{k}×{n} alpha={alpha} beta={beta}: 融合与未融合参考不逐位一致"
                );
            }
            want
        };
        // 计时从同一份 C_old 出发（两臂都从参考结果开始，避免"谁先跑谁占便宜"）。
        c.copy_from_slice(&want);

        // 索引一律走 `as_slice()`/`as_mut_slice()` 取到的切片：`AlignedVec` 的
        // `Deref` 索引在每个元素上多一层调用，写成 `c[i]` 实测能把 elementwise 那遍
        // 拖慢 ~10×（3.7 ms vs 0.37 ms）——那会让"两遍法"被人为变慢，属于测量错误。
        let a_s = a.as_slice();
        let b_s = b.as_slice();
        let cs = c.as_mut_slice();
        let acs = acc.as_mut_slice();
        let t = if fused {
            timeit(|| {
                api::matmul_scaled_into(m, k, n, alpha, beta, a_s, b_s, cs).unwrap();
                let _ = black_box(cs[0]);
            })
        } else {
            timeit(|| {
                lasx_matmul(
                    m as i32,
                    k as i32,
                    n as i32,
                    a_s.as_ptr(),
                    b_s.as_ptr(),
                    acs.as_mut_ptr(),
                );
                for i in 0..m * n {
                    cs[i] = beta.mul_add(cs[i], alpha * acs[i]);
                }
                let _ = black_box(cs[0]);
            })
        };
        if fused {
            println!(
                "AB arm=a case={m}x{k}x{n} alpha={alpha} beta={beta} us={:.2} loadavg={} same=yes",
                t.as_secs_f64() * 1e6,
                load
            );
        } else {
            // 拆解：两遍法的时间花在哪一遍（`mm` = `lasx_matmul` 写 acc，`ep` = elementwise）。
            let mm = timeit(|| {
                lasx_matmul(
                    m as i32,
                    k as i32,
                    n as i32,
                    a_s.as_ptr(),
                    b_s.as_ptr(),
                    acs.as_mut_ptr(),
                );
                let _ = black_box(acs[0]);
            });
            let ep = timeit(|| {
                for i in 0..m * n {
                    cs[i] = beta.mul_add(cs[i], alpha * acs[i]);
                }
                let _ = black_box(cs[0]);
            });
            println!(
                "AB arm=b case={m}x{k}x{n} alpha={alpha} beta={beta} us={:.2} loadavg={} same=yes mm_us={:.2} ep_us={:.2}",
                t.as_secs_f64() * 1e6,
                load,
                mm.as_secs_f64() * 1e6,
                ep.as_secs_f64() * 1e6
            );
        }
    }
}
