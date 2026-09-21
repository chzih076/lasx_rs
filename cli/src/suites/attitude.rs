//! 批量姿态/几何内核的基准：LASX 与标量对照。
//!
//! 这组算子的**单元素算力/字节比很低**（叉积 9 flop 读 48 B 写 24 B），规模一大就受
//! 内存带宽限制，向量化拿不到多少好处；规模在缓存里时才是 SIMD 的主场。
//! 因此每行都跑两个规模：4 Ki（L1/L2 驻留）与 256 Ki（DRAM 流式）。

use crate::data::{AlignedBuf, Lcg};
use crate::report::{row3, Row};
use crate::timing::{time_mode, timeit, Mode};
use lasx_rs::*;
use std::hint::black_box;

/// 规模：缓存驻留 + 流式各一档。
const NS: [usize; 2] = [4096, 1 << 18];

/// 造一组 f64 SOA 数据。
fn bufs(n: usize, seed: u64, count: usize) -> Vec<AlignedBuf<f64>> {
    let mut rng = Lcg::new(seed);
    (0..count)
        .map(|_| AlignedBuf::fill_with(n, |_| rng.f64()))
        .collect()
}

pub fn attitude(rows: &mut Vec<Row>) {
    for n in NS {
        let b = bufs(n, 0xa770 ^ n as u64, 12);
        let p = |k: usize| b[k].as_ptr();
        // 输出缓冲：预先算好裸指针，闭包只捕获指针（避免与 black_box 的读取争借用）
        let o = bufs(n, 0, 9);
        let om: [*mut f64; 9] = std::array::from_fn(|k| o[k].as_ptr() as *mut f64);
        let read = |k: usize| unsafe { *om[k] };
        let ni = n as i32;

        // ---- 叉积：9 flop / 6 读 3 写 ----
        let lasx = time_mode(Mode::Lasx, || {
            lasx_cross3_batch(p(0), p(1), p(2), p(3), p(4), p(5), om[0], om[1], om[2], ni);
            let _ = black_box(read(0));
        });
        let scalar = timeit(|| {
            for i in 0..n {
                let (ax, ay, az) = (b[0][i], b[1][i], b[2][i]);
                let (bx, by, bz) = (b[3][i], b[4][i], b[5][i]);
                unsafe {
                    *om[0].add(i) = ay * bz - az * by;
                    *om[1].add(i) = az * bx - ax * bz;
                    *om[2].add(i) = ax * by - ay * bx;
                }
            }
            let _ = black_box(read(0));
        });
        row3(
            "lasx_cross3_batch",
            format!("n={n}"),
            9.0 * n as f64,
            "flop/s",
            lasx,
            None,
            scalar,
            rows,
        );

        // ---- 单位化：约 11 flop + sqrt + div / 3 读 3 写 ----
        let lasx = time_mode(Mode::Lasx, || {
            lasx_unitize3_batch(p(0), p(1), p(2), om[3], om[4], om[5], ni);
            let _ = black_box(read(3));
        });
        let scalar = timeit(|| {
            for i in 0..n {
                let m = (b[0][i] * b[0][i] + (b[1][i] * b[1][i] + b[2][i] * b[2][i])).sqrt();
                let inv = if m == 0.0 { 0.0 } else { 1.0 / m };
                unsafe {
                    *om[3].add(i) = b[0][i] * inv;
                    *om[4].add(i) = b[1][i] * inv;
                    *om[5].add(i) = b[2][i] * inv;
                }
            }
            let _ = black_box(read(3));
        });
        row3(
            "lasx_unitize3_batch",
            format!("n={n}"),
            11.0 * n as f64,
            "flop/s",
            lasx,
            None,
            scalar,
            rows,
        );

        // ---- quat_rotate：约 40 flop + sqrt + div / 7 读 3 写 ----
        let q = bufs(n, 0x9a71 ^ n as u64, 4);
        let qp = |k: usize| q[k].as_ptr();
        let (qw, qx, qy, qz) = (q[0].as_ptr(), q[1].as_ptr(), q[2].as_ptr(), q[3].as_ptr());
        let lasx = time_mode(Mode::Lasx, || {
            lasx_quat_rotate_batch(qw, qx, qy, qz, p(0), p(1), p(2), om[6], om[7], om[8], ni);
            let _ = black_box(read(6));
        });
        let scalar = timeit(|| {
            for i in 0..n {
                let nn = (((q[0][i] * q[0][i] + q[1][i] * q[1][i]) + q[2][i] * q[2][i])
                    + q[3][i] * q[3][i])
                    .sqrt();
                let (w, x, y, z) = if nn < 1e-15 {
                    (1.0, 0.0, 0.0, 0.0)
                } else {
                    (q[0][i] / nn, q[1][i] / nn, q[2][i] / nn, q[3][i] / nn)
                };
                let r = [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - w * z),
                    2.0 * (x * z + w * y),
                    2.0 * (x * y + w * z),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - w * x),
                    2.0 * (x * z - w * y),
                    2.0 * (y * z + w * x),
                    1.0 - 2.0 * (x * x + y * y),
                ];
                let (px, py, pz) = (b[0][i], b[1][i], b[2][i]);
                unsafe {
                    *om[6].add(i) = (r[0] * px + r[1] * py) + r[2] * pz;
                    *om[7].add(i) = (r[3] * px + r[4] * py) + r[5] * pz;
                    *om[8].add(i) = (r[6] * px + r[7] * py) + r[8] * pz;
                }
            }
            let _ = black_box(read(6));
        });
        row3(
            "lasx_quat_rotate_batch",
            format!("n={n}"),
            40.0 * n as f64,
            "flop/s",
            lasx,
            None,
            scalar,
            rows,
        );
        let _ = (qp, qw);
    }
}
