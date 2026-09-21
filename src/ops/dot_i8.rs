//! `lasx_dot_i8` —— int8 量化点积：i16 乘积 → 拓宽 i32 累加，每 1024 字节落盘 i64。
//!
//!
use crate::arch::lasx;
use std::arch::loongarch64::*;

/// LASX-only：本内核没有 `has_lasx()` 降级分支，直接执行 256 位实现。
/// 在无 LASX 的 CPU 上会执行 LASX 指令（见手册 Caveats）。
#[inline]
pub(crate) fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    let n = a.len();
    let mut s_acc: i64 = 0;
    let mut i = 0;
    let z = lasx::zero_i32x8();
    let mut acc0 = z;
    let mut acc1 = z;
    let mut acc2 = z;
    let mut acc3 = z;
    #[inline(always)]
    unsafe fn step(
        a: *const i8,
        b: *const i8,
        i: usize,
        z: m256i,
        acc0: &mut m256i,
        acc1: &mut m256i,
        acc2: &mut m256i,
        acc3: &mut m256i,
    ) {
        let va = lasx_xvld(a.add(i), 0);
        let vb = lasx_xvld(b.add(i), 0);
        let lo16 = lasx_xvmulwev_h_b(va, vb);
        let hi16 = lasx_xvmulwod_h_b(va, vb);
        // 拓宽到 i32 再累加（不能链式用 xvaddwev/xvaddwod 当累加器）
        *acc0 = lasx_xvadd_w(*acc0, lasx_xvaddwev_w_h(lo16, z));
        *acc1 = lasx_xvadd_w(*acc1, lasx_xvaddwod_w_h(lo16, z));
        *acc2 = lasx_xvadd_w(*acc2, lasx_xvaddwev_w_h(hi16, z));
        *acc3 = lasx_xvadd_w(*acc3, lasx_xvaddwod_w_h(hi16, z));
    }

    // 主循环：每 1024 字节（32 个向量）才落盘一次
    let mut tmp = [0i32; 8];
    while i + 1024 <= n {
        for _ in 0..32 {
            unsafe {
                step(
                    a.as_ptr(),
                    b.as_ptr(),
                    i,
                    z,
                    &mut acc0,
                    &mut acc1,
                    &mut acc2,
                    &mut acc3,
                )
            };
            i += 32;
        }
        // 合并成两条再落盘，减少一半 store 与标量加
        let s01 = unsafe { lasx_xvadd_w(acc0, acc1) };
        let s23 = unsafe { lasx_xvadd_w(acc2, acc3) };
        unsafe {
            lasx_xvst(s01, tmp.as_mut_ptr() as *mut i8, 0);
        }
        for &v in &tmp {
            s_acc += v as i64;
        }
        unsafe {
            lasx_xvst(s23, tmp.as_mut_ptr() as *mut i8, 0);
        }
        for &v in &tmp {
            s_acc += v as i64;
        }
        acc0 = z;
        acc1 = z;
        acc2 = z;
        acc3 = z;
    }

    // 余下不足 1024 字节：按 32 字节块处理，收尾统一落盘
    while i + 32 <= n {
        unsafe {
            step(
                a.as_ptr(),
                b.as_ptr(),
                i,
                z,
                &mut acc0,
                &mut acc1,
                &mut acc2,
                &mut acc3,
            )
        };
        i += 32;
    }
    let s01 = unsafe { lasx_xvadd_w(acc0, acc1) };
    let s23 = unsafe { lasx_xvadd_w(acc2, acc3) };
    unsafe {
        lasx_xvst(s01, tmp.as_mut_ptr() as *mut i8, 0);
    }
    for &v in &tmp {
        s_acc += v as i64;
    }
    unsafe {
        lasx_xvst(s23, tmp.as_mut_ptr() as *mut i8, 0);
    }
    for &v in &tmp {
        s_acc += v as i64;
    }

    for j in i..n {
        s_acc += (a[j] as i64) * (b[j] as i64);
    }
    s_acc as i32
}

/// 数值回归测试：对照独立参考实现。
#[cfg(test)]
mod tests {
    use crate::ffi::quant::lasx_dot_i8;

    fn exact_i64(a: &[i8], b: &[i8]) -> i32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| x as i64 * y as i64)
            .sum::<i64>() as i32
    }
    /// 与 i64 精确参考逐位相等；覆盖块边界、i32 回绕与 (−128)² 的 i16 溢出边界
    #[test]
    fn test_dot_i8_matches_exact_i64_reference() {
        let mut s = 0x1234_5678_9abc_def0u64;
        let mut rnd = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as i32 as i8
        };

        for &n in &[
            0usize, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 1023, 1024,
            1025, 1055, 2047, 4095,
        ] {
            // 随机
            let a: Vec<i8> = (0..n).map(|_| rnd()).collect();
            let b: Vec<i8> = (0..n).map(|_| rnd()).collect();
            assert_eq!(
                lasx_dot_i8(a.as_ptr(), b.as_ptr(), n as i32),
                exact_i64(&a, &b),
                "n={n} 随机"
            );

            // 常数填充：−128 触发 (−128)·(−128)=16384 的边界
            for &fill in &[-128i8, 127, -1, 0, 1] {
                let a = vec![fill; n];
                let b = vec![fill; n];
                assert_eq!(
                    lasx_dot_i8(a.as_ptr(), b.as_ptr(), n as i32),
                    exact_i64(&a, &b),
                    "n={n} fill={fill}"
                );
            }

            // 交替 ±极值：相邻乘积一正一负
            let a: Vec<i8> = (0..n)
                .map(|k| if k % 2 == 0 { -128 } else { 127 })
                .collect();
            let b: Vec<i8> = (0..n)
                .map(|k| if k % 3 == 0 { -128 } else { 127 })
                .collect();
            assert_eq!(
                lasx_dot_i8(a.as_ptr(), b.as_ptr(), n as i32),
                exact_i64(&a, &b),
                "n={n} 交替极值"
            );
        }

        // 大 n：真值超出 i32，验证最终截断与精确参考 mod 2^32 一致
        let n = 300_000usize;
        let a = vec![-128i8; n];
        let b = vec![-128i8; n];
        assert_eq!(
            lasx_dot_i8(a.as_ptr(), b.as_ptr(), n as i32),
            exact_i64(&a, &b),
            "n={n} 全 −128（i32 回绕）"
        );
    }
}
