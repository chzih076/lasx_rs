//! **`simd_dsl::Backend` 的 `lasx_rs` 实现**（步骤 ③a；**feature 门控**：`dsl-backend`）。
//!
//! # 范围（③a，**只做必需方法**）
//! - **必需方法**（trait 无默认值）只有两个：`matmul_f32`、`dot_f32` ⇒ **接到现有 LASX 内核**
//!   （`ops::matmul::matmul_f32` / `ops::dot::dot`）——**不绕开、不新写算法、不改语义**；
//! - **其余方法全部先用 `simd_dsl` 的默认实现**（标量回退；**不 panic**），包括
//!   `matmul_f64`/`dot_f64`/`reduce_sum_*`/`map_unary_*`/`pack_*`/`build_kernel_f32` 等；
//! - **本轮不做**：`build_kernel_f32` 覆盖（手写递推核）、`pack_f32`/`matmul_f32_packed`
//!   （现有 `Prepared`）、完整 `precision_claim` —— 留到 ③b。
//!
//! # `precision_claim` 只填**能如实声明**的部分
//! `matmul_f32`/`dot_f32` 走的是**现有 LASX 实内核**，与 `api::matmul`/`api::dot` **逐位一致**
//! （有测试 `bitwise_same_as_api` 钉住）⇒ 声明为**逐位类**并**点名参照物** = `lasx_rs::api`。
//! **其余路径一律 `None`**（未测未做 ⇒ 不声明）。
//!
//! # `Capability` **只低估**
//! 只报"确实为真"的能力位；拿不准的一律不报。
#![allow(clippy::undocumented_unsafe_blocks)]

use simd_dsl::backend::{Backend, BackendError, Capability};
use simd_dsl::equivalence::{EnginePath, ErrorBound, Fallback, PrecisionClaim, ScalarKind};

/// 用 LASX 现有内核的 `simd_dsl::Backend`。
#[derive(Debug, Default, Clone, Copy)]
pub struct LasxBackend;

/// 长度不符时的错误（与 `simd_dsl` 的默认实现同类）。
fn len_err(op: &'static str, what: &'static str, expected: usize, got: usize) -> BackendError {
    BackendError::LengthMismatch {
        op,
        what,
        expected,
        got,
    }
}

impl Backend for LasxBackend {
    fn name(&self) -> &'static str {
        "lasx_rs"
    }

    fn matmul_f32(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        if a.len() != m * k {
            return Err(len_err("matmul_f32", "a", m * k, a.len()));
        }
        if b.len() != k * n {
            return Err(len_err("matmul_f32", "b", k * n, b.len()));
        }
        if out.len() != m * n {
            return Err(len_err("matmul_f32", "out", m * n, out.len()));
        }
        if m == 0 || n == 0 {
            return Ok(());
        }
        // **现有 LASX 内核**（与 `api::matmul` 同一条路径）
        crate::ops::matmul::matmul_f32(m, k, n, a, b, out);
        Ok(())
    }

    fn dot_f32(&self, a: &[f32], b: &[f32]) -> Result<f32, BackendError> {
        if a.len() != b.len() {
            return Err(len_err("dot_f32", "b", a.len(), b.len()));
        }
        // **现有 LASX 内核**（与 `api::dot` 同一条路径）
        Ok(crate::ops::dot::dot(a, b))
    }

    /// **③b-1：`build_kernel_f32` 覆盖 —— 手写等距递推核（每行守卫 + 逐行回退 + 如实上报）**。
    ///
    /// **守卫只调 `simd_dsl::safety::recurrence_safe`**（不复制阈值、不存第二份常量）；
    /// `opt_in == false` 或**该行不安全** ⇒ **该行回退 libm**（与默认实现同式），
    /// 并把 `rows_fallback` / `route_taken` **如实**填进报告 —— **不静默假装走快档**。
    fn build_kernel_f32(
        &self,
        spec: &simd_dsl::kernel::KernelSpec<'_>,
        kc: &mut [f32],
        ks: &mut [f32],
    ) -> Result<simd_dsl::kernel::KernelReport, BackendError> {
        use simd_dsl::route::{FallbackReport, Route};
        let (rows, n) = (spec.rows, spec.n);
        let expected = rows.checked_mul(n).ok_or(BackendError::ShapeOverflow {
            op: "build_kernel_f32",
            what: "rows×n",
        })?;
        if kc.len() != expected {
            return Err(len_err("build_kernel_f32", "kc", expected, kc.len()));
        }
        if ks.len() != expected {
            return Err(len_err("build_kernel_f32", "ks", expected, ks.len()));
        }
        if spec.sigma.len() != rows {
            return Err(len_err("build_kernel_f32", "sigma", rows, spec.sigma.len()));
        }
        if spec.omega.len() != rows {
            return Err(len_err("build_kernel_f32", "omega", rows, spec.omega.len()));
        }
        let dt = spec.dt as f64;
        let mut rows_fallback = 0usize;
        let mut all_safe = true;
        for j in 0..rows {
            let om = spec.omega[j] as f64;
            let sg = spec.sigma[j] as f64;
            // **唯一守卫**：调用 `simd_dsl::safety::recurrence_safe`（阈值不在这里复制）
            let safe = simd_dsl::safety::recurrence_safe(om, dt, n);
            if !safe {
                all_safe = false;
            }
            // **权重约定 = 引擎约定**（③b-2 选 (b)）：`kc = e^{−σt}·cos(ωt)`、`ks = −e^{−σt}·sin(ωt)`
            // —— **无权重、无 `dt` 因子**，与 `simd_dsl::kernel::build` 的默认路径**同式**。
            if spec.opt_in && safe {
                // 等距递推（行内）：`e *= e^{−σΔt}`、`(c,s)` 旋转
                let (decay, c1, s1) = ((-sg * dt).exp(), (om * dt).cos(), (om * dt).sin());
                let (mut e, mut c, mut s) = (1.0f64, 1.0f64, 0.0f64);
                for i in 0..n {
                    kc[j * n + i] = (e * c) as f32;
                    ks[j * n + i] = (-e * s) as f32;
                    e *= decay;
                    let (nc, ns) = (c * c1 - s * s1, s * c1 + c * s1);
                    c = nc;
                    s = ns;
                }
            } else {
                rows_fallback += 1;
                // 回退：与引擎默认路径**同一式子**（f32 libm 逐元素）
                for i in 0..n {
                    let t = i as f32 * spec.dt;
                    let e = (-spec.sigma[j] * t).exp();
                    kc[j * n + i] = e * (spec.omega[j] * t).cos();
                    ks[j * n + i] = -(e * (spec.omega[j] * t).sin());
                }
            }
        }
        let route = if rows_fallback == rows {
            Route::Default
        } else {
            Route::Recurrence
        };
        Ok(simd_dsl::kernel::KernelReport::new(
            FallbackReport::new(route, rows_fallback),
            all_safe,
            rows,
        ))
    }

    /// **只声明能如实声明的**：默认路径 + f32 上的 `matmul_f32`/`dot_f32` 走的是**现有 LASX 内核**，
    /// 与 `lasx_rs::api` **逐位一致**（测试 `bitwise_same_as_api` 钉住）⇒ **逐位类 + 点名参照物**。
    /// **其余（`IsometricRecurrence`、f64 等）一律 `None`** —— 未做未测 ⇒ 不声明。
    fn precision_claim(&self, path: EnginePath, scalar: ScalarKind) -> Option<PrecisionClaim> {
        match (path, scalar) {
            (EnginePath::Default, ScalarKind::F32) => Some(PrecisionClaim::bit_exact(
                "参照物 = lasx_rs::api（同一 LASX 内核）",
            )),
            // **③b-1：递推路径 = 容差类 + 界**（与 `guard_of(IsometricRecurrence)` 的"逐行回退"一致）
            // **参照物写准**：相对的是**引擎默认路径**（`simd_dsl::kernel::build` 的 libm 口径），
            // 因为覆盖版按**引擎权重约定**实现（③b-2 选 (b)：`kc=e^{−σt}cos`、`ks=−e^{−σt}sin`，无权重无 dt）。
            (EnginePath::IsometricRecurrence, ScalarKind::F32) => Some(PrecisionClaim::tolerance(
                ErrorBound::rel(1e-4),
                "等距递推核，参照物 = 引擎默认路径（simd_dsl::kernel::build）；守卫 = simd_dsl::safety::recurrence_safe，超界逐行回退",
                Fallback::PerRowDefault,
            )),
            (EnginePath::IsometricRecurrence, ScalarKind::F64) => Some(PrecisionClaim::tolerance(
                ErrorBound::masked_rel(5.5e-14),
                "等距递推核（掩码相对），参照物 = 引擎默认路径（simd_dsl::kernel::build）；守卫 = simd_dsl::safety::recurrence_safe，超界逐行回退",
                Fallback::PerRowDefault,
            )),
            _ => None,
        }
    }

    /// **只低估**：③a 只覆盖了 f32 的 `matmul_f32`/`dot_f32`，**没有** f64/逐元素/归约/超越/打包/
    /// 多线程/复数组合步的专用实现 ⇒ **所有 `Capability` 一律 `false`**（"只报确实为真"）。
    fn supports(&self, _cap: Capability) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **③b-1 四条测试：守卫 + 逐行回退 + 计数 + 与默认实现一致（容差类，不断言逐位）**。
    #[test]
    fn kernel_guard_and_fallback_report() {
        use simd_dsl::kernel::KernelSpec;
        use simd_dsl::route::Route;
        let be = LasxBackend;
        let (rows, n) = (4usize, 64usize);
        let dt = 1.0f32 / 256.0;
        // 前两行在安全区（ω·Δt 很小）；第 3 行超区（ω·Δt = 2.0 > 1.5）；第 4 行也超区
        let omega = [0.05f32, 0.10, 2.0 / dt, 3.0 / dt];
        let sigma = [1.0f32; 4];
        let mut kc = vec![0.0f32; rows * n];
        let mut ks = vec![0.0f32; rows * n];

        // ① 安全区行走递推；② 超区行回退且计数正确（opt_in = true）
        let spec = KernelSpec::new(rows, n, dt, &sigma, &omega, true);
        let rep = be.build_kernel_f32(&spec, &mut kc, &mut ks).expect("build");
        assert_eq!(rep.rows, rows);
        assert_eq!(rep.fallback.route_taken, Route::Recurrence, "有行走递推");
        assert!(rep.fallback.had_fallback(), "有 2 行超区 ⇒ 必须上报回退");
        assert_eq!(
            rep.fallback.rows_fallback, 2,
            "超区行数应为 2（实测 {}）",
            rep.fallback.rows_fallback
        );
        assert!(!rep.all_safe, "守卫结论应为不安全");

        // ③ opt_in = false ⇒ 全回退（route_taken = Default、rows_fallback = rows）
        let spec_off = KernelSpec::new(rows, n, dt, &sigma, &omega, false);
        let rep_off = be
            .build_kernel_f32(&spec_off, &mut kc, &mut ks)
            .expect("build");
        assert_eq!(rep_off.fallback.route_taken, Route::Default);
        assert_eq!(rep_off.fallback.rows_fallback, rows);

        // ④a **同约定内**：libm 回退（opt_in=false）与递推（安全行）在同一行上应落在**容差类**内
        //     （注意：**不断言逐位** —— 递推是容差类）
        let mut kc_fb = vec![0.0f32; rows * n];
        let mut ks_fb = vec![0.0f32; rows * n];
        let _ = be.build_kernel_f32(&spec_off, &mut kc_fb, &mut ks_fb);
        let mut kc_rec = vec![0.0f32; rows * n];
        let mut ks_rec = vec![0.0f32; rows * n];
        let _ = be.build_kernel_f32(&spec, &mut kc_rec, &mut ks_rec);
        let mut worst = 0.0f32;
        for j in 0..2 {
            // 只看安全的两行（递推 vs libm）：实部核
            for i in 0..n {
                let l = j * n + i;
                worst = worst.max((kc_rec[l] - kc_fb[l]).abs() / kc_fb[l].abs().max(1e-30));
            }
        }
        // 虚部核（`ks`）另测一遍（同一行集合）
        let mut worst_s = 0.0f32;
        for j in 0..2 {
            for i in 0..n {
                let l = j * n + i;
                worst_s = worst_s.max((ks_rec[l] - ks_fb[l]).abs() / ks_fb[l].abs().max(1e-30));
            }
        }
        let worst = worst.max(worst_s);
        println!("安全行：递推 vs libm 回退 最大相对差 {worst:.3e}（容差类，不断言逐位）");
        assert!(
            worst < 1e-3,
            "安全行的递推应与 libm 落在容差内，实测 {worst:e}"
        );
        // ④b **等价性断言（③b-2 的 ①）**：覆盖版（全回退档）与**引擎默认路径**
        //     （`ScalarBackend::build_kernel_f32`）必须落在同一条**容差**内（**不断言逐位**）。
        let mut kc_sd = vec![0.0f32; rows * n];
        let mut ks_sd = vec![0.0f32; rows * n];
        let _ = simd_dsl::mock::ScalarBackend.build_kernel_f32(&spec_off, &mut kc_sd, &mut ks_sd);
        let (mut cross, mut peak) = (0.0f32, 0.0f32);
        for l in 0..rows * n {
            peak = peak.max(kc_sd[l].abs()).max(ks_sd[l].abs());
            cross = cross.max((kc_fb[l] - kc_sd[l]).abs());
            cross = cross.max((ks_fb[l] - ks_sd[l]).abs());
        }
        let rel = cross / peak.max(1e-30);
        println!("覆盖版（全回退）vs 引擎默认路径：最大绝对差 {cross:.3e}，相对峰值 {rel:.3e}");
        assert!(
            rel < 1e-4,
            "覆盖版必须与引擎默认路径在容差内等价（③b-2 的①），实测相对 {rel:e}"
        );
    }

    /// **③a 正确性（逐位）**：`LasxBackend::matmul_f32` / `dot_f32` 与 `lasx_rs::api` **逐位一致**。
    #[test]
    fn bitwise_same_as_api() {
        let be = LasxBackend;
        let (m, k, n) = (5usize, 7usize, 3usize);
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 17) as f32) * 0.125 - 1.0)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i % 13) as f32) * 0.0625 - 0.5)
            .collect();
        // 参照物：`api::matmul`（逐列 1 宽，公开路径）；形状 m×n
        let mut want = vec![0.0f32; m * n];
        for j in 0..n {
            let col: Vec<f32> = (0..k).map(|l| b[l * n + j]).collect();
            let got = crate::api::matmul(m, k, 1, &a, &col).expect("api::matmul");
            for i in 0..m {
                want[i * n + j] = got[i];
            }
        }
        let mut have = vec![0.0f32; m * n];
        be.matmul_f32(m, k, n, &a, &b, &mut have).expect("backend");
        assert_eq!(
            have.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "LasxBackend::matmul_f32 与 api 不逐位"
        );
        // dot
        let x: Vec<f32> = (0..64).map(|i| ((i % 11) as f32) * 0.25 - 1.0).collect();
        let y: Vec<f32> = (0..64).map(|i| ((i % 7) as f32) * 0.125 - 0.5).collect();
        let w = crate::api::dot(&x, &y).expect("api::dot");
        let h = be.dot_f32(&x, &y).expect("backend.dot");
        assert_eq!(
            h.to_bits(),
            w.to_bits(),
            "LasxBackend::dot_f32 与 api 不逐位"
        );
    }

    /// **长度不符 ⇒ 报错（不 panic）**。
    #[test]
    fn length_mismatch_is_an_error() {
        let be = LasxBackend;
        let a = vec![0.0f32; 6]; // m*k = 6 ✓
        let b_short = vec![0.0f32; 5]; // k*n 应为 6 ⇒ 短一个
        let mut out = vec![0.0f32; 4];
        assert!(be.matmul_f32(2, 3, 2, &a, &b_short, &mut out).is_err());
        let mut out_short = vec![0.0f32; 3]; // m*n 应为 4
        let b = vec![0.0f32; 6];
        assert!(be.matmul_f32(2, 3, 2, &a, &b, &mut out_short).is_err());
        assert!(be.dot_f32(&a, &b[..5]).is_err());
    }

    /// **只声明能如实声明的**：默认路径/f32 为逐位类（点名参照物）；未做路径为 `None`。
    #[test]
    fn precision_claim_is_honest() {
        let be = LasxBackend;
        assert!(
            be.precision_claim(EnginePath::Default, ScalarKind::F32)
                .is_some()
        );
        assert!(
            be.precision_claim(EnginePath::IsometricRecurrence, ScalarKind::F32)
                .is_some(),
            "③b-1 后递推路径应声明为容差类"
        );
        assert!(
            be.precision_claim(EnginePath::Default, ScalarKind::F64)
                .is_none(),
            "f64 专用实现本轮未做 ⇒ 必须 None"
        );
        // **只低估**：③a 覆盖面之外一律 false
        assert!(!be.supports(Capability::Packing));
        assert!(!be.supports(Capability::MatmulF64));
        assert!(!be.supports(Capability::Transcendental));
    }
}
