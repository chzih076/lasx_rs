//! 矩阵乘内核的 C ABI 导出（行主序，B 不转置）。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（分层策略：内核模块在函数级统一写 SAFETY，不逐块重复）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX/LSX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]

/// f32 矩阵乘 `C[m×n] = A[m×k]·B[k×n]`。
///
/// C 签名：`void lasx_matmul(int m, int k, int n, const float *a, const float *b, float *c)`
#[unsafe(no_mangle)]
pub extern "C" fn lasx_matmul(m: i32, k: i32, n: i32, a: *const f32, b: *const f32, c: *mut f32) {
    let (m, k, n) = (m as usize, k as usize, n as usize);
    // SAFETY: FFI 约定——`a` 为 m×k、`b` 为 k×n 可读，`c` 为 m×n 可写。
    let a = unsafe { std::slice::from_raw_parts(a, m * k) };
    let b = unsafe { std::slice::from_raw_parts(b, k * n) };
    let c = unsafe { std::slice::from_raw_parts_mut(c, m * n) };
    crate::ops::matmul::matmul_f32(m, k, n, a, b, c);
}

/// f64 矩阵乘 `C[m×n] = A[m×k]·B[k×n]`。
///
/// C 签名：`void lasx_matmul_f64(int m, int k, int n, const double *a, const double *b, double *c)`
#[unsafe(no_mangle)]
pub extern "C" fn lasx_matmul_f64(
    m: i32,
    k: i32,
    n: i32,
    a: *const f64,
    b: *const f64,
    c: *mut f64,
) {
    let (m, k, n) = (m as usize, k as usize, n as usize);
    // SAFETY: FFI 约定——`a` 为 m×k、`b` 为 k×n 可读，`c` 为 m×n 可写。
    let a = unsafe { std::slice::from_raw_parts(a, m * k) };
    let b = unsafe { std::slice::from_raw_parts(b, k * n) };
    let c = unsafe { std::slice::from_raw_parts_mut(c, m * n) };
    crate::ops::matmul_f64::matmul_f64(m, k, n, a, b, c);
}

/// 融合缩放矩阵乘 `C = alpha·(A·B) + beta·C`（原地，行主序）。
///
/// C 签名：`void lasx_matmul_scaled(int m, int k, int n, float alpha, const float *a,
/// const float *b, float beta, float *c)`
///
/// 语义与数值口径与 [`crate::api::matmul_scaled_into`] **完全一致**（C 入口只是它的裸指针版）：
///
/// - `alpha = 1, beta = 0` 时与 [`lasx_matmul`] **逐位一致**（同一条内核，收尾不多一次乘法）；
/// - `beta == 0` 时**不读 `c`**（CBLAS 语义：`C` 无需在入参里有效）；
/// - 与"先算 `A·B` 再整体做 `beta·C_old + alpha·acc`"**逐位一致**；
/// - `beta != 0` 且 `k > 384`（打包路径要 k 分块、部分和会覆盖 `C_old`）退回与朴素二遍法
///   同构的实现——不假装更快。口径与实测见 `docs/dev.md` §19.15。
///
/// 需要错误通道时用 `crate::ffi::checked::lasx_matmul_scaled_checked`。
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn lasx_matmul_scaled(
    m: i32,
    k: i32,
    n: i32,
    alpha: f32,
    a: *const f32,
    b: *const f32,
    beta: f32,
    c: *mut f32,
) {
    let (m, k, n) = (m as usize, k as usize, n as usize);
    // SAFETY: FFI 约定——`a` 为 m×k、`b` 为 k×n 可读，`c` 为 m×n 可读写
    // （`beta != 0` 时读旧值，`beta == 0` 时不读）。
    let a = unsafe { std::slice::from_raw_parts(a, m * k) };
    let b = unsafe { std::slice::from_raw_parts(b, k * n) };
    let c = unsafe { std::slice::from_raw_parts_mut(c, m * n) };
    crate::ops::matmul::matmul_f32_scaled(m, k, n, alpha, beta, a, b, c);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api;
    use crate::ffi::checked::lasx_matmul_scaled_checked;
    use crate::ffi::status::LasxStatus;
    use crate::ops::testutil::Lcg;

    /// C ABI 与安全 Rust API **逐位一致**（"追加符号"不能引入第二套数值），并且
    /// C 入口在 `beta = 0` 时同样**不读 `c`**（NaN 探针）。
    ///
    /// 形状覆盖三条路径与边界：打包（单 k 块）、列块（极宽 n）、打包 + k 分块（`k > 384`，
    /// `beta != 0` 走二遍法回退）、行/列尾。
    #[test]
    fn test_matmul_scaled_c_abi_matches_api_bitwise() {
        let mut rng = Lcg(0x5ca1_ed03);
        for &(m, k, n) in &[
            (1usize, 1usize, 1usize),
            (5, 7, 40),
            (33, 65, 31),
            (32, 64, 96),
            (100, 64, 4096),
            (100, 448, 1024),
            (256, 512, 512),
        ] {
            let a: Vec<f32> = (0..m * k).map(|_| rng.f64() as f32).collect();
            let b: Vec<f32> = (0..k * n).map(|_| rng.f64() as f32).collect();
            let c0: Vec<f32> = (0..m * n).map(|_| rng.f64() as f32).collect();
            for &(alpha, beta) in &[
                (1.0f32, 0.0f32),
                (1.5, 0.0),
                (-0.25, 0.0),
                (1.0, 1.0),
                (1.5, 0.5),
                (0.75, -1.25),
            ] {
                let mut want = c0.clone();
                api::matmul_scaled_into(m, k, n, alpha, beta, &a, &b, &mut want).unwrap();
                let mut got = c0.clone();
                lasx_matmul_scaled(
                    m as i32,
                    k as i32,
                    n as i32,
                    alpha,
                    a.as_ptr(),
                    b.as_ptr(),
                    beta,
                    got.as_mut_ptr(),
                );
                for i in 0..m * n {
                    assert_eq!(
                        got[i].to_bits(),
                        want[i].to_bits(),
                        "C ABI vs api：{m}×{k}×{n} α={alpha} β={beta} @ {i}"
                    );
                }
            }
            // `beta = 0` 不读 c：NaN 不许从"不该读的旧值"传播进输出
            let mut got = vec![f32::NAN; m * n];
            lasx_matmul_scaled(
                m as i32,
                k as i32,
                n as i32,
                1.5,
                a.as_ptr(),
                b.as_ptr(),
                0.0,
                got.as_mut_ptr(),
            );
            assert!(
                got.iter().all(|v| v.is_finite()),
                "{m}×{k}×{n}：C 入口在 beta = 0 时读了 c"
            );
        }
    }

    /// `_checked` 的状态码路径：成功写 `Ok` 且逐位等于 api；负维度、空指针（长度非 0）
    /// 被拦，失败时**不触碰输出**；`status = NULL` 仍执行校验。
    ///
    /// 尺寸溢出（`SizeOverflow`）在 C 入口**不可达**：`i32` 维度的乘积 ≤ `(2^31-1)^2 ≈ 4.6e18`
    /// < `usize::MAX ≈ 1.8e19`（64 位），所以这里不造这个状态。
    #[test]
    fn test_matmul_scaled_checked_status_paths() {
        let (m, k, n) = (2usize, 3usize, 4usize);
        let a = vec![1.0f32; m * k];
        let b = vec![2.0f32; k * n];

        let mut want = vec![7.5f32; m * n];
        api::matmul_scaled_into(m, k, n, 1.5, 0.5, &a, &b, &mut want).unwrap();

        // 成功路径
        let mut c = vec![7.5f32; m * n];
        let mut st = 999;
        lasx_matmul_scaled_checked(
            m as i32,
            k as i32,
            n as i32,
            1.5,
            a.as_ptr(),
            b.as_ptr(),
            0.5,
            c.as_mut_ptr(),
            &mut st,
        );
        assert_eq!(st, LasxStatus::Ok as i32);
        for i in 0..m * n {
            assert_eq!(c[i].to_bits(), want[i].to_bits(), "i={i}");
        }

        // 负维度 → NegativeLength，输出原样
        let mut c2 = vec![7.5f32; m * n];
        let mut st = 999;
        lasx_matmul_scaled_checked(
            -1,
            k as i32,
            n as i32,
            1.5,
            a.as_ptr(),
            b.as_ptr(),
            0.5,
            c2.as_mut_ptr(),
            &mut st,
        );
        assert_eq!(st, LasxStatus::NegativeLength as i32);
        assert!(c2.iter().all(|&v| v == 7.5), "失败时不该触碰输出");

        // 输入空指针（长度非 0）→ NullPointer
        let mut st = 999;
        lasx_matmul_scaled_checked(
            m as i32,
            k as i32,
            n as i32,
            1.5,
            std::ptr::null(),
            b.as_ptr(),
            0.5,
            c2.as_mut_ptr(),
            &mut st,
        );
        assert_eq!(st, LasxStatus::NullPointer as i32);

        // 输出空指针 → NullPointer
        let mut st = 999;
        lasx_matmul_scaled_checked(
            m as i32,
            k as i32,
            n as i32,
            1.5,
            a.as_ptr(),
            b.as_ptr(),
            0.5,
            std::ptr::null_mut(),
            &mut st,
        );
        assert_eq!(st, LasxStatus::NullPointer as i32);

        // `status = NULL`：不关心原因，但校验仍执行（不 deref 空 status）
        lasx_matmul_scaled_checked(
            -1,
            k as i32,
            n as i32,
            1.5,
            a.as_ptr(),
            b.as_ptr(),
            0.5,
            c2.as_mut_ptr(),
            std::ptr::null_mut(),
        );

        // 空形状（m = n = 0）：合法，写 Ok，不碰任何缓冲
        let mut st = 999;
        lasx_matmul_scaled_checked(
            0,
            k as i32,
            0,
            1.5,
            a.as_ptr(),
            b.as_ptr(),
            0.5,
            std::ptr::null_mut(),
            &mut st,
        );
        assert_eq!(st, LasxStatus::Ok as i32);
    }
}
