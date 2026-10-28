//! **P5-b：NT-GEMM（`b` 行主序、按 `j` 逐行点积）的 LASX 内核** —— **逐位硬约束**。
//!
//! # 语义（签名与 `simd_dsl` 契约一致，不许改）
//! `matmul_bt_f32(m, k, n, a, b, out)`：`out[i*n + j] = Σ_t a[i*k + t] · b[j*k + t]`。
//! 形状：`a: m×k`、`b: n×k`（**同样 k 方向连续** ⇒ 点积两侧都是顺序访存，**不需要转置 `b`**）、`out: m×n`。
//!
//! # 为什么逐位
//! 累加**严格按 `t` 顺序**、**无多累加器、无重排、无 FMA 收缩** ⇒ 与"朴素三重循环（`i→j→t`）"
//! **逐位一致**（单测用 `to_bits()` 钉住）。这是本轮硬约束。
//!
//! # 性能：当前仍是标量（**如实**）
//! 逐位要求"无重排、无 FMA 收缩"⇒ 只能用 8-lane `mul`+`add`（非 FMA）。当前内核**仍是标量**实现：
//! 8-lane 非 FMA 版本**未做**（封装名已具备：`lasx_xvfmul_s`/`lasx_xvfadd_s`，见 `ops/axpy.rs`），
//! 性能结论待后续补测；当前只兑现"少一次物化转置"的结构收益。
//!
//! # 门控
//! `simd_dsl` 的 trait 已声明该方法、`dsl_backend.rs` 已有 `matmul_bt_f32` override 调用本模块
//! ⇒ 有生产调用方 ⇒ 只挂 `#[cfg(feature = "dsl-backend")]`（不需要 `test` 门控；否则会被
//! `clippy -D warnings` 的 `dead_code` 判死，且禁止 `#[allow(dead_code)]`，见 AGENTS.md #49）。

use crate::api::Error;

fn mismatch(op: &'static str, what: &'static str, expected: usize, got: usize) -> Error {
    Error::Shape {
        op,
        what,
        expected,
        got,
    }
}

/// **NT-GEMM**：`out[i*n+j] = Σ_t a[i*k+t]·b[j*k+t]`（**逐位类**）。
///
/// # Errors
/// `a.len() != m*k`、`b.len() != n*k`、`out.len() != m*n` 或乘积溢出 ⇒ `BackendError::LengthMismatch`。
pub fn matmul_bt_f32(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
) -> Result<(), Error> {
    let need_a = m.checked_mul(k).ok_or(Error::Overflow {
        op: "matmul_bt_f32",
        what: "m*k",
    })?;
    let need_b = n.checked_mul(k).ok_or(Error::Overflow {
        op: "matmul_bt_f32",
        what: "n*k",
    })?;
    let need_o = m.checked_mul(n).ok_or(Error::Overflow {
        op: "matmul_bt_f32",
        what: "m*n",
    })?;
    if a.len() != need_a {
        return Err(mismatch("matmul_bt_f32", "a", need_a, a.len()));
    }
    if b.len() != need_b {
        return Err(mismatch("matmul_bt_f32", "b", need_b, b.len()));
    }
    if out.len() != need_o {
        return Err(mismatch("matmul_bt_f32", "out", need_o, out.len()));
    }
    (0..m).for_each(|i| {
        (0..n).for_each(|j| {
            out[i * n + j] = (0..k).fold(0.0f32, |acc, t| acc + a[i * k + t] * b[j * k + t]);
        });
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::time::Instant;

    fn data(len: usize, seed: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i % 29) as f32) * 0.0625 - 1.0 + seed)
            .collect()
    }

    /// **逐位**：与朴素三重循环（`i→j→t`）逐位一致（`to_bits`），`k` 故意不整除 8。
    #[test]
    fn bitwise_same_as_naive() {
        let (m, k, n) = (7usize, 37usize, 11usize);
        let a = data(m * k, 0.0);
        let b = data(n * k, 0.125);
        let mut got = vec![0.0f32; m * n];
        matmul_bt_f32(m, k, n, &a, &b, &mut got).unwrap();
        let want: Vec<f32> = (0..m * n)
            .map(|idx| {
                let (i, j) = (idx / n, idx % n);
                (0..k).fold(0.0f32, |acc, t| acc + a[i * k + t] * b[j * k + t])
            })
            .collect();
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "NT 内核必须与朴素三重循环逐位一致（本轮硬约束）"
        );
    }

    /// 形状不合 ⇒ `Err`（不 panic、不截断）。
    #[test]
    fn shape_errors() {
        let mut out = vec![0.0f32; 6];
        assert!(matmul_bt_f32(2, 3, 3, &[0.0; 6], &[0.0; 9], &mut out).is_ok());
        assert!(matmul_bt_f32(2, 3, 3, &[0.0; 5], &[0.0; 9], &mut out).is_err());
        assert!(matmul_bt_f32(2, 3, 3, &[0.0; 6], &[0.0; 8], &mut out).is_err());
        assert!(matmul_bt_f32(2, 3, 3, &[0.0; 6], &[0.0; 9], &mut out[..5]).is_err());
    }

    /// **三臂 A/B 实测（性能探针，不进 CI）**。
    ///
    /// 臂：① 物化转置 `bᵀ`（`n×k→k×n`）+ 行点积（≈现状"物化转置 + matmul"的数据移动量）
    /// ② **NT 内核**（本文件）｜③ 分块转置（32 行一块）+ 行点积
    ///
    /// **为什么 `#[ignore]`**：①/③ 是**朴素三重循环**（2048×128×2048 下每臂已 ~0.5 G MAC），
    /// 多轮在共享 CI 机器上纯属拖时间；且本测试**无时间阈值断言**（只有 `>0` 合理性）。
    /// 数值等价由**非 ignore** 的 [`bitwise_same_as_naive`]（小形状、逐位）覆盖。
    /// 手动跑：`cargo test --release -p lasx_rs --lib three_arm_ab -- --ignored --nocapture`。
    ///
    /// **口径（抗噪）**：同进程交替 **8 轮、丢首轮（7 个样本）**，同时打印中位与 **min** 比值
    /// （负载只增不减 ⇒ min 更接近无噪速度；正式采纳结论仍看进程级交替 + 中位）。
    #[ignore = "性能探针（无时间阈值断言，朴素臂很重）：cargo test --release -p lasx_rs --lib three_arm_ab -- --ignored --nocapture"]
    #[test]
    fn three_arm_ab() {
        let shapes = [(1024usize, 64usize, 1024usize), (2048, 128, 2048)];
        for (m, k, n) in shapes {
            let a = data(m * k, 0.0);
            let b = data(n * k, 0.25);
            let out = RefCell::new(vec![0.0f32; m * n]);
            let bt = RefCell::new(vec![0.0f32; k * n]);
            let med = |f: &mut dyn FnMut()| -> (f64, f64) {
                let mut v = Vec::new();
                for _ in 0..8 {
                    let t0 = Instant::now();
                    f();
                    v.push(t0.elapsed().as_secs_f64() * 1e3);
                }
                v.remove(0); // 丢首轮（预热）
                v.sort_by(f64::total_cmp);
                (v[v.len() / 2], v[0])
            };
            // ① 物化转置 + 行点积
            let (t1, min1) = med(&mut || {
                let mut bt = bt.borrow_mut();
                let mut o = out.borrow_mut();
                (0..n).for_each(|j| {
                    (0..k).for_each(|t| bt[t * n + j] = b[j * k + t]);
                });
                (0..m).for_each(|i| {
                    (0..n).for_each(|j| {
                        o[i * n + j] =
                            (0..k).fold(0.0f32, |acc, t| acc + a[i * k + t] * bt[t * n + j]);
                    });
                });
            });
            // ② NT 内核
            let (t2, min2) = med(&mut || {
                matmul_bt_f32(m, k, n, &a, &b, &mut out.borrow_mut()).unwrap();
            });
            // ③ 分块转置（32 行一块）+ 行点积
            let (t3, min3) = med(&mut || {
                let mut bt = bt.borrow_mut();
                let mut o = out.borrow_mut();
                let mut j0 = 0usize;
                while j0 < n {
                    let j1 = (j0 + 32).min(n);
                    (0..n).for_each(|j| {
                        (0..k).for_each(|t| bt[t * n + j] = b[j * k + t]);
                    });
                    (0..m).for_each(|i| {
                        (j0..j1).for_each(|j| {
                            o[i * n + j] =
                                (0..k).fold(0.0f32, |acc, t| acc + a[i * k + t] * bt[t * n + j]);
                        });
                    });
                    j0 = j1;
                }
            });
            println!(
                "P5-b 三臂 m={m},k={k},n={n}：① 物化转置+点积 {t1:.3} ms（min {min1:.3}）｜② NT 内核 {t2:.3} ms（min {min2:.3}）｜③ 分块转置+点积 {t3:.3} ms（min {min3:.3}）⇒ **①/② = {:.3}×（中位）/ {:.3}×（min）**（≥1.10 才采纳）、③/② = {:.3}×（中位）",
                t1 / t2,
                min1 / min2,
                t3 / t2
            );
            assert!(t1 > 0.0 && t2 > 0.0 && t3 > 0.0);
            // 数值等价由非 ignore 的 `bitwise_same_as_naive`（小形状、逐位）覆盖；本探针只测时间。
        }
    }
}
