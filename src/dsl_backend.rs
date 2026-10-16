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
use simd_dsl::equivalence::{EnginePath, PrecisionClaim, ScalarKind};

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

    /// **只声明能如实声明的**：默认路径 + f32 上的 `matmul_f32`/`dot_f32` 走的是**现有 LASX 内核**，
    /// 与 `lasx_rs::api` **逐位一致**（测试 `bitwise_same_as_api` 钉住）⇒ **逐位类 + 点名参照物**。
    /// **其余（`IsometricRecurrence`、f64 等）一律 `None`** —— 未做未测 ⇒ 不声明。
    fn precision_claim(&self, path: EnginePath, scalar: ScalarKind) -> Option<PrecisionClaim> {
        match (path, scalar) {
            (EnginePath::Default, ScalarKind::F32) => Some(PrecisionClaim::bit_exact(
                "参照物 = lasx_rs::api（同一 LASX 内核）",
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
                .is_none(),
            "递推核本轮未做 ⇒ 必须 None"
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
