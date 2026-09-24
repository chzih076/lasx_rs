//! `dot_f16` / `gemv_f16`：f16 权重点积与矩阵-向量的带宽实测。
//!
//! 这个算子的价值**全在带宽上**：decode 时每算一个输出元素要读整行权重，f16 存权重让访存
//! 量减半。所以口径按"权重字节数"算（`m×k×2`），并同时给出 f32 权重 `matmul`/`gemv` 的
//! 对照行——**同一条 `k` 下 f16 应当接近 f32 的 2 倍吞吐**才算拿到好处。
//!
//! 标量参照用同一个累加口径（16 元素块 + lane = j%8 + 固定次序归约），所以列间可比。

use crate::data::{AlignedBuf, Lcg};
use crate::report::{row3, Row};
use crate::timing::{time_mode, timeit, Mode};
use lasx_rs::ffi::nn::{lasx_dot_f16, lasx_gemv_f16};
use std::hint::black_box;

/// 标量参照：f16 → f32（f32 的位运算转换）后按同一 lane 口径累加。
fn scalar_gemv(a: &[u16], x: &[f32], m: usize, k: usize, y: &mut [f32]) {
    for r in 0..m {
        let row = &a[r * k..(r + 1) * k];
        let mut t = [0f32; 8];
        for (j, (&h, &xb)) in row.iter().zip(x).enumerate() {
            t[j % 8] += half_to_f32(h) * xb;
        }
        y[r] = ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]));
    }
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let man = (bits & 0x03ff) as u32;
    if exp == 0 {
        if man == 0 {
            return f32::from_bits(sign);
        }
        let (mut m, mut e) = (man, -14i32);
        while m & 0x0400 == 0 {
            m <<= 1;
            e -= 1;
        }
        return f32::from_bits(sign | (((e + 127) as u32) << 23) | ((m & 0x03ff) << 13));
    }
    f32::from_bits(sign | ((exp + 127 - 15) << 23) | (man << 13))
}

/// 把 f32 权重"压缩"成 f16 位型（只为造数据；值本身不影响速度）。
fn f32_to_half(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x007f_ffff;
    let e = exp - 127 + 15;
    if e <= 0 {
        return sign; // 太小 → 0（造数据够用）
    }
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    sign | ((e as u16) << 10) | ((man >> 13) as u16)
}

pub fn f16_gemv(rows: &mut Vec<Row>) {
    // decode 的典型形状：k = hidden（2048/4096），m = 词表或 FFN 的行数
    for &(m, k) in &[
        (1usize, 2048usize),
        (64, 2048),
        (1024, 2048),
        (4096, 4096),
        (32, 8192),
    ] {
        let mut rng = Lcg::new((m * 31 + k) as u64);
        let a: Vec<u16> = (0..m * k)
            .map(|_| f32_to_half(rng.f32() * 2.0 - 1.0))
            .collect();
        let x = AlignedBuf::fill_with(k, |_| rng.f32() * 2.0 - 1.0);
        let mut y = AlignedBuf::new(m);
        let tag = format!("{m}×{k}");

        let lasx = time_mode(Mode::Lasx, || {
            lasx_gemv_f16(a.as_ptr(), x.as_ptr(), y.as_mut_ptr(), m as i32, k as i32);
            let _ = black_box(y[0]);
        });
        let scalar = timeit(|| {
            scalar_gemv(&a, x.as_slice(), m, k, y.as_mut_slice());
            let _ = black_box(y[0]);
        });
        // 口径 = 权重字节（f16 是 2 B/元素）+ 输出 + 输入（后两项在小 m 时也可忽略）
        let bytes = (m * k * 2 + k * 4 + m * 4) as f64;
        row3(
            "lasx_gemv_f16(LASX-only)",
            tag,
            bytes,
            "B/s",
            lasx,
            None,
            scalar,
            rows,
        );
    }
    // 单行点积（gemv 的核心循环）
    for &k in &[1024usize, 4096, 16_384] {
        let mut rng = Lcg::new(k as u64);
        let a: Vec<u16> = (0..k).map(|_| f32_to_half(rng.f32() * 2.0 - 1.0)).collect();
        let x = AlignedBuf::fill_with(k, |_| rng.f32() * 2.0 - 1.0);
        let lasx = time_mode(Mode::Lasx, || {
            let _ = black_box(lasx_dot_f16(a.as_ptr(), x.as_ptr(), k as i32));
        });
        let mut y = AlignedBuf::new(1);
        let scalar = timeit(|| {
            scalar_gemv(&a, x.as_slice(), 1, k, y.as_mut_slice());
            let _ = black_box(y[0]);
        });
        let bytes = (k * 2 + k * 4) as f64;
        row3(
            "lasx_dot_f16(LASX-only)",
            format!("n={k}"),
            bytes,
            "B/s",
            lasx,
            None,
            scalar,
            rows,
        );
    }
}
