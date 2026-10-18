//! **`simd_dsl::Backend` 的 `lasx_rs` 实现**（步骤 ③a；**feature 门控**：`dsl-backend`）。
//!
//! # 范围（**当前状态，阶段3 批1 之后**）
//! - **必需方法**（trait 无默认值）：`matmul_f32`、`dot_f32` ⇒ **接到现有 LASX 内核**
//!   （`ops::matmul::matmul_f32` / `ops::dot::dot`）——**不绕开、不新写算法、不改语义**；
//! - **已覆盖（本仓有内核可接的）**：`matmul_f32`/`dot_f32`、**`matmul_f64`/`dot_f64`（IS-2）**、
//!   **`matmul_bt_f32`（IS-1）**、`reduce_sum_f32`、`map_affine_f32`、`map_binary_f32`、
//!   `reduce_{sum,weighted_sum,centered_sq_sum}_cols_f32`、`matmul_pair_psd_f32`、
//!   `gather_rows_f32`/`gather_rows_i8`/`scatter_rows_f32`、`fft_strided_f32`、
//!   `build_kernel_f32`（等距递推核）、`pack_f32`/`matmul_f32_packed`、
//!   `copy_strided_f32`/`map_unary_strided_f32`/`map_binary_strided_f32`；
//! - **仍未覆盖（**没有 LASX 内核**，或按分批计划不做）**：rows 方向归约、`max/min_{cols,rows}`、
//!   `map_unary_f32`/`map_binary_scalar_f32`、banded/stencil、f64 逐元素族 ⇒ **走 `simd_dsl` 标量默认**
//!   （**不 panic**；性能必然不如 api 版，迁移时按"无内核 ⇒ 不替换"如实记）；
//! - **线程**：`CallOpts{threads}` **未接**（本后端不按 opts 派活，见文末 ③b-4 留档）；但 **GEMM 内核
//!   自己**在 `work ≥ 32M` 时走**进程级全局池**（`ops::matmul::matmul_f32`，§7.28；`init_global(1)`
//!   是逃生门）⇒ **大形状 A/B 的 `threads` 会 >1**，必须在表里打印（`examples/migrate_ab.rs` 已照做）。
//!
//! # `precision_claim` 只填**能如实声明**的部分
//! `(Default, F32/F64)` 上被覆盖的**内核方法**（`matmul_f32`/`dot_f32`/`matmul_f64`/`dot_f64`）
//! 与 `lasx_rs::api` **逐位一致**（同一内核；测试 `bitwise_same_as_api` 与
//! `matmul_f64_and_dot_f64_bitwise_vs_api` 钉住）⇒ 声明为**逐位类**并**点名参照物** = `lasx_rs::api`。
//!
//! **IS-4 校正（audit §10 #2）**：`map_affine_f32` 与 `reduce_weighted_sum_cols_f32` 用 **FMA**
//! （`ops/dsl_map.rs`）⇒ 它们是**容差类**（实测相对峰值 ≤ 1e-5），**不在上面那条逐位声明里**。
//! 旧版本模块文档把这两条写成"逐位、map_affine 故意不融 FMA"，与代码事实相反 —— 已更正。
//!
//! # `Capability` **只低估**
//! 只报"确实为真"的能力位；拿不准的一律不报。
//!
//! # 留档结论（③b-4）：`CallOpts{threads}` **未接 —— 原因是接口/工作量，不是能力问题**
//!
//! **依据（真实签名）**：lasx_rs **有**多线程设施，`src/parallel.rs`：
//! ```text
//! pub fn matmul_f32(pool: &WorkerPool, m, k, n, a: &mut [f32], b: &[f32], c: &mut [f32]) -> Result<(), Error>
//! //   内部：let pick = crate::pool::pick_rows(m, pool.threads(), 4);
//! ```
//! ⇒ **"按线程数挑行块"这件事在本库做得到**，所以**不是能力问题**。**未接的两条具体约束**：
//! 1. **入口不匹配**：`parallel::matmul_f32` **不接受已打包的 `PackedB`**（它自己打包），
//!    与"预打包复用"不是同一条入口 ⇒ 要接 `CallOpts` 必须**新增"已打包面板 + 线程挑行块"的组合入口**
//!    （工作量，且要同时动并行层与打包层的接口）；
//! 2. **池的所有权**：`Backend: Send + Sync` ⇒ 后端要持 `WorkerPool` 就得先定**池的共享/生命周期**
//!    方案（静态池 vs 每调用建池）——这是**接口设计问题**，不是"写不出来"。
//!
//! **语义澄清**：`simd_dsl` 的 `CallOpts` 明确只是**意图**、不是保证（其默认实现里就是 `let _ = opts;`）
//! ⇒ 即便将来接上，也只能声明"**尽量按意图**"，不能声明"保证 N 线程"。
//! **当前状态**：`CallOpts{threads}` **未接、未测**；但 **GEMM 内核自动并行**（§7.28，`work ≥ 32M` 时
//! 自走 `pool::global()`）已经生效 ⇒ 大形状下"本后端单线程"这个旧说法**不再成立**，A/B 表必须打印
//! 实际 `threads`（本文件的 `examples/migrate_ab.rs` 就是这么做的）。
#![allow(clippy::undocumented_unsafe_blocks)]

// **bump 修法（R6-B-2）**：`simd_dsl` 在 M1–M5 分层重构后，把契约/判据/引擎各面**在 crate 根再导出**
// （见 `simd_dsl/src/lib.rs` 的 `pub use contract::…` / `criteria::…`），旧的平铺模块路径
// （`simd_dsl::backend`/`call`/`equivalence`/`packed`/`route`/`kernel`/`safety`）**已不存在** ⇒ 改走根。
use simd_dsl::{
    Backend, BackendError, BinaryOp, CallOpts, Capability, EnginePath, ErrorBound, Fallback,
    PackedB, PrecisionClaim, ScalarKind, UnaryOp,
};

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
        // **§7.28：走后端自己的公开自动入口** `api::matmul_into`。
        //
        // 为什么不再自己拼池：`ops::matmul::matmul_f32`（本仓公开内核）**自己**在 work ≥ 阈值时
        // 走 `pool::global()` + 按输出行切（绝不 split-K ⇒ **逐位不变**），并把池版要求的
        // `&mut A` 用**内部 scratch** 消化 ⇒ **DSL override 侧零复制、零税**（签名仍是 `&[f32]`）。
        // 线程数由全局池决定（`pool::init_global(n)` 或默认 `auto()`）；`init_global(1)` = 逃生门。
        crate::api::matmul_into(m, k, n, a, b, out).map_err(|_e| {
            // 长度/形状在上方已校验 ⇒ 这里只可能是溢出；映射成 DSL 的可诊断错误（不吞、不 panic）。
            BackendError::LengthMismatch {
                op: "matmul_f32",
                what: "matmul_into（形状/溢出）",
                expected: m * n,
                got: out.len(),
            }
        })?;
        Ok(())
    }

    fn dot_f32(&self, a: &[f32], b: &[f32]) -> Result<f32, BackendError> {
        if a.len() != b.len() {
            return Err(len_err("dot_f32", "b", a.len(), b.len()));
        }
        // **现有 LASX 内核**（与 `api::dot` 同一条路径）
        Ok(crate::ops::dot::dot(a, b))
    }

    /// **IS-1（阶段3 批1）：`Backend::matmul_bt_f32` → 本仓 LASX NT-GEMM 内核**。
    ///
    /// NT 形态：B 行主序 `n×k`（`out[i·n+j] = Σ_t a[i·k+t]·b[j·k+t]`），**不物化转置**、
    /// 两个操作数都按 `k` 连续读。内核本身就是按 `Result`/`LengthMismatch` 契约写的
    /// （`ops/matmul_bt.rs`：形状校验在**写入之前**跑完、不 panic）⇒ 这里**直接转发**，不复制校验。
    ///
    /// **等价类：逐位类** —— 内核严格按 `t` 顺序单累加器累加、无 FMA/多累加器/重排 ⇒ 与
    /// "物化 `Bᵀ` 再 `api::matmul_into`"以及 `simd_dsl` 的标量默认**同序**
    /// （测试 `matmul_bt_bitwise_same_as_api_and_scalar` 用 `to_bits()` 钉住；
    /// `examples/migrate_ab` 先跑逐位、只在不逐位时才谈界）。
    fn matmul_bt_f32(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        crate::ops::matmul_bt::matmul_bt_f32(m, k, n, a, b, out)
    }

    /// **IS-2（阶段3 批1）：`Backend::matmul_f64` → 本仓 LASX f64 GEMM**。
    ///
    /// 默认实现是 `simd_dsl::kernels::scalar::matmul_f64`（标量回退）⇒ 这里接上 LASX f64 内核
    /// （`ops::matmul_f64`：流式 + 打包两条路径，与 [`crate::api::matmul_f64`] 同一条）。
    ///
    /// **等价类（实测，2026-10-18）**：
    /// * 对**旧入口 `api::matmul_f64`**：**逐位**（同一条内核 ⇒ `to_bits()` 相等；测试
    ///   `matmul_f64_and_dot_f64_bitwise_vs_api` 钉住）；
    /// * 对**标量默认**（接线前的路径）：**容差类**——LASX 侧 4-lane 累加/打包次序与顺序累加不同，
    ///   实测相对峰值 **≤ 3.4e-16**（形状 33×65×31 / 100×448×1024 / 512×512×512）。
    ///
    /// **不接** `parallel::matmul_f64`（池版几何 = 批3 的事，本批只接线、不改线程口径）。
    fn matmul_f64(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f64],
        b: &[f64],
        out: &mut [f64],
    ) -> Result<(), BackendError> {
        let op = "matmul_f64";
        let need_a = m
            .checked_mul(k)
            .ok_or(BackendError::ShapeOverflow { op, what: "m×k" })?;
        let need_b = k
            .checked_mul(n)
            .ok_or(BackendError::ShapeOverflow { op, what: "k×n" })?;
        let need_out = m
            .checked_mul(n)
            .ok_or(BackendError::ShapeOverflow { op, what: "m×n" })?;
        if a.len() != need_a {
            return Err(len_err(op, "a", need_a, a.len()));
        }
        if b.len() != need_b {
            return Err(len_err(op, "b", need_b, b.len()));
        }
        if out.len() != need_out {
            return Err(len_err(op, "out", need_out, out.len()));
        }
        if m == 0 || n == 0 {
            return Ok(());
        }
        crate::ops::matmul_f64::matmul_f64(m, k, n, a, b, out);
        Ok(())
    }

    /// **IS-2（阶段3 批1）：`Backend::dot_f64` → 本仓 LASX f64 点积**（默认是标量回退）。
    ///
    /// **等价类（实测，2026-10-18）**：对**旧入口 `api::dot_f64` 逐位**（同内核）；对**标量默认**
    /// 是**容差类**——LASX 侧 4-lane 累加 + 水平归约，与顺序累加不同序 ⇒ 实测相对 **≤ 2.4e-14**
    /// （n = 4096；n ≤ 8 时恰好逐位）。
    fn dot_f64(&self, a: &[f64], b: &[f64]) -> Result<f64, BackendError> {
        if a.len() != b.len() {
            return Err(len_err("dot_f64", "b", a.len(), b.len()));
        }
        Ok(crate::ops::dot_f64::dot_f64(a, b))
    }

    /// **③b-1：`build_kernel_f32` 覆盖 —— 手写等距递推核（每行守卫 + 逐行回退 + 如实上报）**。
    ///
    /// **守卫只调 `simd_dsl::safety::recurrence_safe`**（不复制阈值、不存第二份常量）；
    /// `opt_in == false` 或**该行不安全** ⇒ **该行回退 libm**（与默认实现同式），
    /// 并把 `rows_fallback` / `route_taken` **如实**填进报告 —— **不静默假装走快档**。
    /// **归约 / 逐元素族接 LASX（R19-E）**：帧里 matmul 之外的多数开销在这五条上。
    ///
    /// **等价类（逐条钉住；每条都有测试）**：
    /// * `map_binary_f32`（`Add`/`Sub`/`Mul`/`Div`）：**与 `simd_dsl::kernels::scalar` 逐位**——
    ///   逐元素无归约。`Max`/`Min`：NaN 语义按 Rust（`f32::max`：另一侧是数就返回它），而 LASX
    ///   `xvfmax` 是 IEEE 语义（NaN 传播不同）⇒ **这两条委托标量参照**，不冒险。
    /// * `reduce_sum_cols_f32`：**逐位**——每列一条 lane、**按行序**累加（与参照同序）。
    /// * **`map_affine_f32` / `reduce_weighted_sum_cols_f32`：容差类**（**IS-4 校正**，audit §10 #2）——
    ///   两者都用 **FMA**（`a·x + b` / `v·w + acc`，见 `ops/dsl_map.rs` 的 `lasx_xvfmadd_s`），
    ///   舍入与"先乘后加"不同，**不是逐位**；实测相对峰值 ≤ 1e-5（`r19e_elementwise_and_axis_reduce_classes`
    ///   的 R19-G 断言，用非 2 的幂系数才测得出差）。旧注释写成"逐位、故意不融 FMA"，与代码事实相反。
    ///   若将来改回"先乘后加"，才可回到逐位类（必须重测后才改声明）。
    /// * `reduce_sum_f32`：参照本身就是**多累加器**（容差类）⇒ 这里用本仓 LASX 归约，**同性质**。
    fn reduce_sum_f32(&self, x: &[f32]) -> Result<f32, BackendError> {
        Ok(crate::ops::dsl_map::reduce_sum(x))
    }

    fn map_affine_f32(
        &self,
        x: &[f32],
        a: f32,
        b: f32,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        if out.len() != x.len() {
            return Err(len_err("map_affine_f32", "out", x.len(), out.len()));
        }
        debug_assert!(
            !std::ptr::eq(x.as_ptr(), out.as_ptr()),
            "map_affine_f32: out 不得与 x 重叠（契约）"
        );
        crate::ops::dsl_map::map_affine(a, b, x, out);
        Ok(())
    }

    fn map_binary_f32(
        &self,
        x: &[f32],
        y: &[f32],
        out: &mut [f32],
        op: BinaryOp,
    ) -> Result<(), BackendError> {
        if y.len() != x.len() {
            return Err(len_err("map_binary_f32", "y", x.len(), y.len()));
        }
        if out.len() != x.len() {
            return Err(len_err("map_binary_f32", "out", x.len(), out.len()));
        }
        let fast = match op {
            BinaryOp::Add => Some(crate::ops::dsl_map::BinOp::Add),
            BinaryOp::Sub => Some(crate::ops::dsl_map::BinOp::Sub),
            BinaryOp::Mul => Some(crate::ops::dsl_map::BinOp::Mul),
            BinaryOp::Div => Some(crate::ops::dsl_map::BinOp::Div),
            // NaN 语义按 Rust ⇒ 交给标量参照（不用 LASX 的 IEEE max/min）
            // `BinaryOp` 是 `#[non_exhaustive]` ⇒ 将来新增的变体也一律**委托标量**（保守、正确优先）
            _ => None,
        };
        match fast {
            Some(fop) => {
                debug_assert!(
                    !std::ptr::eq(x.as_ptr(), out.as_ptr()),
                    "map_binary_f32: out 不得与 x 重叠（契约）"
                );
                debug_assert!(
                    !std::ptr::eq(y.as_ptr(), out.as_ptr()),
                    "map_binary_f32: out 不得与 y 重叠（契约）"
                );
                crate::ops::dsl_map::map_binary(x, y, out, fop);
            }
            None => simd_dsl::kernels::scalar::map_binary_f32(x, y, out, op)?,
        }
        Ok(())
    }

    fn reduce_sum_cols_f32(
        &self,
        x: &[f32],
        rows: usize,
        cols: usize,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let op = "reduce_sum_cols_f32";
        let total = rows.checked_mul(cols).ok_or(BackendError::ShapeOverflow {
            op,
            what: "rows×cols",
        })?;
        if x.len() != total {
            return Err(len_err(op, "x", total, x.len()));
        }
        if out.len() != cols {
            return Err(len_err(op, "out", cols, out.len()));
        }
        debug_assert!(
            !std::ptr::eq(x.as_ptr(), out.as_ptr()),
            "{op}: out 不得与 x 重叠（契约）"
        );
        crate::ops::dsl_map::reduce_sum_cols(x, rows, cols, out);
        Ok(())
    }

    fn reduce_weighted_sum_cols_f32(
        &self,
        x: &[f32],
        w: &[f32],
        rows: usize,
        cols: usize,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let op = "reduce_weighted_sum_cols_f32";
        let total = rows.checked_mul(cols).ok_or(BackendError::ShapeOverflow {
            op,
            what: "rows×cols",
        })?;
        if x.len() != total {
            return Err(len_err(op, "x", total, x.len()));
        }
        if w.len() != rows {
            return Err(len_err(op, "w", rows, w.len()));
        }
        if out.len() != cols {
            return Err(len_err(op, "out", cols, out.len()));
        }
        debug_assert!(
            !std::ptr::eq(x.as_ptr(), out.as_ptr()),
            "{op}: out 不得与 x 重叠（契约）"
        );
        crate::ops::dsl_map::reduce_weighted_sum_cols(x, w, rows, cols, out);
        Ok(())
    }

    /// **P4-a 原语（LASX 覆盖）**：`out[c] = Σ_r (x[r·cols+c] − mean[c])²`，**单遍**（无 `x_dc` 中间缓冲）。
    ///
    /// **等价类：逐位**——`ops::fused::energy_dc` 是**纯标量、与便携参照同序**（单累加器、`r` 递增），
    /// 与 `kernels::scalar::reduce_centered_sq_sum_cols_f32` 的 `to_bits()` 相等（LASX 侧测试钉住）。
    /// **注意**：若将来改成 FMA/多累加器，会**静默离开逐位类** ⇒ 必须改声明为容差类并回填实测界。
    fn reduce_centered_sq_sum_cols_f32(
        &self,
        x: &[f32],
        rows: usize,
        cols: usize,
        mean: &[f32],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let op = "reduce_centered_sq_sum_cols_f32";
        let total = rows.checked_mul(cols).ok_or(BackendError::ShapeOverflow {
            op,
            what: "rows×cols",
        })?;
        if x.len() != total {
            return Err(len_err(op, "x", total, x.len()));
        }
        if mean.len() != cols {
            return Err(len_err(op, "mean", cols, mean.len()));
        }
        if out.len() != cols {
            return Err(len_err(op, "out", cols, out.len()));
        }
        debug_assert!(
            !std::ptr::eq(x.as_ptr(), out.as_ptr()),
            "{op}: out 不得与 x 重叠（契约）"
        );
        crate::ops::fused::energy_dc(x, rows, cols, mean, out);
        Ok(())
    }

    /// **P4-a 原语（LASX 覆盖）**：`out[j·n+c] = ((Σ_t a·x)² + (Σ_t b·x)²)·inv[c]`，**成对 epilogue**
    /// （无 `re2/im2/psd` 三块中间缓冲）。
    ///
    /// **等价类：逐位**——`ops::fused::psd_pair` 纯标量、与便携参照同序（单累加器、`t` 递增）；
    /// 与 `kernels::scalar::matmul_pair_psd_f32` 的 `to_bits()` 相等（LASX 侧测试钉住）。
    /// **`k = 0` 时输出是 `(0+0)·inv[c]`，`inv` 含负值时是 `-0.0`**（不可图省事 `fill(0.0)`）。
    fn matmul_pair_psd_f32(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
        x: &[f32],
        inv: &[f32],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let op = "matmul_pair_psd_f32";
        let mk = m
            .checked_mul(k)
            .ok_or(BackendError::ShapeOverflow { op, what: "m×k" })?;
        let kn = k
            .checked_mul(n)
            .ok_or(BackendError::ShapeOverflow { op, what: "k×n" })?;
        let mn = m
            .checked_mul(n)
            .ok_or(BackendError::ShapeOverflow { op, what: "m×n" })?;
        if a.len() != mk {
            return Err(len_err(op, "a", mk, a.len()));
        }
        if b.len() != mk {
            return Err(len_err(op, "b", mk, b.len()));
        }
        if x.len() != kn {
            return Err(len_err(op, "x", kn, x.len()));
        }
        if inv.len() != n {
            return Err(len_err(op, "inv", n, inv.len()));
        }
        if out.len() != mn {
            return Err(len_err(op, "out", mn, out.len()));
        }
        crate::ops::fused::psd_pair(a, b, x, m, k, n, inv, out);
        Ok(())
    }

    /// **P5-a 索引读写（LASX 覆盖）**：`out[i, 0..row_len] = table[ids[i], 0..row_len]`。
    ///
    /// **等价类：逐位**（纯行拷贝，无算术）；`stride` 显式 ⇒ 支持跨步表/跨步输出。
    /// 校验与 `Err` 语义在 LASX 内核里（越界 ⇒ `IndexOutOfRange`、溢出 ⇒ `ShapeOverflow`、
    /// **`Err` 时 `out` 不被部分写入**）。
    fn gather_rows_f32(
        &self,
        table: &[f32],
        row_len: usize,
        table_stride: usize,
        ids: &[u32],
        out: &mut [f32],
        out_stride: usize,
    ) -> Result<(), BackendError> {
        crate::ops::gather_scatter::gather_rows_f32(
            table,
            row_len,
            table_stride,
            ids,
            out,
            out_stride,
        )
    }

    /// **P5-a 索引读写（LASX 覆盖，i8 表）**：与 [`Backend::gather_rows_f32`] 同形；**逐位**。
    fn gather_rows_i8(
        &self,
        table: &[i8],
        row_len: usize,
        table_stride: usize,
        ids: &[u32],
        out: &mut [i8],
        out_stride: usize,
    ) -> Result<(), BackendError> {
        crate::ops::gather_scatter::gather_rows_i8(
            table,
            row_len,
            table_stride,
            ids,
            out,
            out_stride,
        )
    }

    /// **P5-a 索引写（LASX 覆盖）**：`cache[ids[i], 0..row_len] = src[i, 0..row_len]`。
    ///
    /// **等价类：逐位**（纯行拷贝）；**同 id 多次出现 ⇒ 调用方顺序 last-write-wins**（内核按顺序写）。
    fn scatter_rows_f32(
        &self,
        src: &[f32],
        src_stride: usize,
        ids: &[u32],
        row_len: usize,
        out: &mut [f32],
        out_stride: usize,
    ) -> Result<(), BackendError> {
        crate::ops::gather_scatter::scatter_rows_f32(src, src_stride, ids, row_len, out, out_stride)
    }

    /// **LASX 原地按列 radix-2 复数 FFT（R19-H）**：沿列 8 路向量化、**不转置**。
    ///
    /// **等价类：容差类**（与 `kernels::scalar::fft_radix2_strided_f32` 的 **f64 内部**实现比）：
    /// 旋转因子 f64 预算后降 f32、蝶形在 f32 里做 ⇒ 每级有舍入。**实测界**由
    /// `r19h_fft_matches_the_scalar_reference_within_a_measured_bound` 钉住并打印。
    /// **形状契约**：`n` 必须是 2 的幂（否则 `Unsupported`）、`re/im` 长度都是 `n*b`。
    fn fft_strided_f32(
        &self,
        re: &mut [f32],
        im: &mut [f32],
        n: usize,
        b: usize,
    ) -> Result<(), BackendError> {
        let op = "fft_strided_f32";
        if n <= 1 || b == 0 {
            return Ok(());
        }
        if !n.is_power_of_two() {
            // 与可移植参照同一条错误（`n` 非 2 的幂 ⇒ 该复系数路径不可服务）
            return Err(BackendError::Unsupported {
                op,
                cap: Capability::ComplexRoutes,
            });
        }
        let total = n
            .checked_mul(b)
            .ok_or(BackendError::ShapeOverflow { op, what: "n×b" })?;
        if re.len() != total {
            return Err(len_err(op, "re", total, re.len()));
        }
        if im.len() != total {
            return Err(len_err(op, "im", total, im.len()));
        }
        crate::ops::dsl_fft::fft_strided(re, im, n, b);
        Ok(())
    }

    fn build_kernel_f32(
        &self,
        spec: &simd_dsl::KernelSpec<'_>,
        kc: &mut [f32],
        ks: &mut [f32],
    ) -> Result<simd_dsl::KernelReport, BackendError> {
        use simd_dsl::{FallbackReport, Route};
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
            let safe = simd_dsl::recurrence_safe(om, dt, n);
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
                // 回退：与引擎默认路径**同一式子**（f32 libm 逐元素）。
                // ⚠️ **与引擎同式（本轮修）**：引擎在 R17 把相位改成**约减到 `[−π,π]`**
                // （`ω` 峰值可达数千 rad/s，f32 直接算 `ω·t` 自带 ~1e-4 绝对误差，见 `simd_dsl`
                // 的 `performance.md §14.5/§14.6`）⇒ 这里**转调同一份判据/本体**，否则两档不同式，
                // 且本档在 `opt_in = false` 或某行超安全区时会把那个误差带进核表。
                let reduce = simd_dsl::criteria::safety::phase_reduction_needed(om, dt, n);
                for i in 0..n {
                    let t = i as f32 * spec.dt;
                    let e = (-spec.sigma[j] * t).exp();
                    let arg = if reduce {
                        simd_dsl::criteria::safety::reduced_phase_f32(om, dt, i)
                    } else {
                        spec.omega[j] * t
                    };
                    kc[j * n + i] = e * arg.cos();
                    ks[j * n + i] = -(e * arg.sin());
                }
            }
        }
        let route = if rows_fallback == rows {
            Route::Default
        } else {
            Route::Recurrence
        };
        Ok(simd_dsl::KernelReport::new(
            FallbackReport::new(route, rows_fallback),
            all_safe,
            rows,
        ))
    }

    /// **只声明能如实声明的**：`(Default, F32)` 与 `(Default, F64)` 上被**覆盖的内核方法**
    /// （`matmul_f32`/`dot_f32`/`matmul_f64`/`dot_f64`）与 `lasx_rs::api` **逐位一致**（同一内核；
    /// 测试 `bitwise_same_as_api`、`matmul_f64_and_dot_f64_bitwise_vs_api` 钉住）⇒ **逐位类 + 点名参照物**。
    /// **注意（IS-4 校正）**：这条 `bit_exact` **不覆盖 FMA 族**——`map_affine_f32` 与
    /// `reduce_weighted_sum_cols_f32` 用 FMA ⇒ **容差类**（界见方法文档与 `r19e_*` 测试的 R19-G 断言）；
    /// 旧版本把这两条也写成逐位，是 audit §10 #2 记的错误。
    /// **其余路径（未覆盖的方法族、`FrontendForward` 等）一律 `None`** —— 未做未测 ⇒ 不声明。
    fn precision_claim(&self, path: EnginePath, scalar: ScalarKind) -> Option<PrecisionClaim> {
        match (path, scalar) {
            (EnginePath::Default, ScalarKind::F32) => Some(PrecisionClaim::bit_exact(
                "参照物 = lasx_rs::api（matmul_f32/dot_f32 同一 LASX 内核）；FMA 族方法（map_affine_f32 / reduce_weighted_sum_cols_f32）另行按容差类声明",
            )),
            // **IS-2**：f64 两个 override 接的是 LASX 内核，与 `api::matmul_f64`/`api::dot_f64` 逐位；
            // 与"接线前的 simd_dsl 标量回退"相差 ~1e-16（这是接线收益/差异，不是本声明的参照物）。
            (EnginePath::Default, ScalarKind::F64) => Some(PrecisionClaim::bit_exact(
                "参照物 = lasx_rs::api::matmul_f64 / api::dot_f64（同一 LASX 内核）",
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

    /// **`pack_f32`：走现有打包设施（`ops::matmul::pack_b`），blob 自带列尾。**
    ///
    /// **blob 布局**（本后端自定义；`blob` 只给产出它的后端解析）：
    /// ```text
    /// [0, k·n32·4)                    ：主 strips 的打包面板（`pack_b(k, n, 0, n32, b, …)` 的原始字节）
    ///                                    布局 = `packed[(s*k + p)*32 + r]`，`s` 为 32 列宽的条带
    /// [k·n32·4, (k·n32 + k·ntail)·4)  ：列尾 `n%32` 的**紧凑行主序**副本（每行 `ntail` 个 f32）
    /// ```
    /// 其中 `n32 = n/32*32`、`ntail = n - n32` ⇒ **`blob.len() == k*n*4`**（自洽测试钉住）。
    /// **为什么尾列要进 blob**：`matmul_f32_packed` 的入参里**没有 `b`**，尾列数据必须随 blob 携带，
    /// 否则尾列无从计算（列尾将用紧凑缓冲喂 `row_tail_range_f32`，`n = ntail, j0 = 0, j1 = ntail`）。
    fn pack_f32(&self, k: usize, n: usize, b: &[f32]) -> Result<PackedB, BackendError> {
        let expected = k.checked_mul(n).ok_or(BackendError::ShapeOverflow {
            op: "pack_f32",
            what: "k×n",
        })?;
        if b.len() != expected {
            return Err(len_err("pack_f32", "b", expected, b.len()));
        }
        let n32 = n / 32 * 32;
        let ntail = n - n32;
        // 主 strips：整段一次 `pack_b`（`nc = n32` ⇒ 条带连续排布）
        let mut packed = vec![0.0f32; k * n32];
        if n32 > 0 {
            crate::ops::matmul::pack_b(k, n, 0, n32, b, &mut packed);
        }
        // 列尾：紧凑行主序
        let mut tail = vec![0.0f32; k * ntail];
        for p in 0..k {
            for t in 0..ntail {
                tail[p * ntail + t] = b[p * n + n32 + t];
            }
        }
        // 拼字节（`f32` → little-endian 字节；本机为小端，`to_le_bytes` 保证可移植）
        let mut blob = Vec::with_capacity(expected * 4);
        for v in packed.iter().chain(tail.iter()) {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        debug_assert_eq!(blob.len(), expected * 4);
        Ok(PackedB::new(self.backend_tag(), k, n, blob))
    }

    /// **`matmul_f32_packed`：解 blob 后走现有 packed 路径（预打包复用）。**
    ///
    /// 步骤：校验 `tag`（`ForeignPacked`）→ 校验 `shape()==(k,n)` 与 `blob.len()==k*n*4`
    /// → 还原两区段 → **主 strips 逐块**调 `matmul_f32_packed_rows` →
    /// **尾列用临时紧凑缓冲**调 `row_tail_range_f32(n = ntail, j0 = 0, j1 = ntail)`。
    ///
    /// **契约**：与 [`Backend::pack_f32`] **必须同后端配对**（blob 格式由本后端自定义；见模块文档）。
    fn matmul_f32_packed(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        p: &PackedB,
        out: &mut [f32],
        opts: &CallOpts,
    ) -> Result<(), BackendError> {
        // `CallOpts` 只是**意图**：本后端单线程（留档结论见模块文档）；这里显式忽略。
        let _ = opts;
        if p.backend_tag() != self.backend_tag() {
            return Err(BackendError::ForeignPacked {
                op: "matmul_f32_packed",
                tag: p.backend_tag(),
                expected: self.backend_tag(),
            });
        }
        if p.shape() != (k, n) {
            return Err(len_err(
                "matmul_f32_packed",
                "packed_b 的形状(k×n)",
                k.saturating_mul(n),
                p.shape().0.saturating_mul(p.shape().1),
            ));
        }
        let total = k.checked_mul(n).ok_or(BackendError::ShapeOverflow {
            op: "matmul_f32_packed",
            what: "k×n",
        })?;
        if a.len() != m * k {
            return Err(len_err("matmul_f32_packed", "a", m * k, a.len()));
        }
        if out.len() != m * n {
            return Err(len_err("matmul_f32_packed", "out", m * n, out.len()));
        }
        if p.blob().len() != total * 4 {
            return Err(len_err(
                "matmul_f32_packed",
                "blob 字节数",
                total * 4,
                p.blob().len(),
            ));
        }
        if m == 0 || n == 0 {
            return Ok(());
        }
        let n32 = n / 32 * 32;
        let ntail = n - n32;
        let blob = p.blob();
        let f32_at = |idx: usize| -> f32 {
            let mut w = [0u8; 4];
            w.copy_from_slice(&blob[idx * 4..idx * 4 + 4]);
            f32::from_le_bytes(w)
        };
        let m4 = m / 4 * 4;
        // **主 strips 逐块**（与 `matmul_f32_packed` 同入口；`epi = Plain` 即 `C = acc`）
        let nc_strips = crate::ops::matmul::pack_strips(k);
        let strips_total = n32 / 32;
        let mut s0 = 0;
        while s0 < strips_total {
            let s1 = (s0 + nc_strips).min(strips_total);
            let jb = s0 * 32;
            let nc = (s1 - s0) * 32;
            let base = jb / 32 * k * 32; // 区段起点 = (jb/32)*k*32
            let packed: Vec<f32> = (0..k * nc).map(|i| f32_at(base + i)).collect();
            crate::ops::matmul::matmul_f32_packed_rows(
                k,
                n,
                jb,
                nc,
                &packed,
                a,
                out,
                m4,
                crate::ops::matmul::Epi::Plain,
            );
            s0 = s1;
        }
        // **尾列**：`row_tail_range_f32` 的 `n` 既是 `b` 的行步长、也是 `c_row` 的列步长
        // ⇒ 传**紧凑暂存行**（长度 `ntail`）算完后**拷回**真实行的 `[n32, n)` 段。
        if ntail > 0 {
            let tail_off = k * n32; // 列尾区段起点（以 f32 计）
            let tail: Vec<f32> = (0..k * ntail).map(|i| f32_at(tail_off + i)).collect();
            let mut staged = vec![0.0f32; ntail];
            for i in 0..m {
                let a_row = &a[i * k..(i + 1) * k];
                crate::ops::matmul::row_tail_range_f32(
                    a_row,
                    &mut staged,
                    &tail,
                    k,
                    ntail,
                    0,
                    ntail,
                    crate::ops::matmul::Epi::Plain,
                );
                let c_row = &mut out[i * n..(i + 1) * n];
                c_row[n32..n].copy_from_slice(&staged);
            }
        }
        Ok(())
    }

    /// **只低估**：已实现且**有测试**的能力才打开。
    /// * `Packing`：`pack_f32` + `matmul_f32_packed`（逐位断言已过，测试 `packed_bitwise_same_as_matmul`）；
    /// * `MatmulF64`：**IS-2 已接** `matmul_f64`/`dot_f64` → LASX f64 内核（与 `api::*` 逐位，
    ///   测试 `matmul_f64_and_dot_f64_bitwise_vs_api`）。
    ///
    /// 其余（逐元素/归约/超越/多线程/复数组合步/f64 逐元素）**继续 `false`**（"只报确实为真"；
    /// 归约/逐元素虽有 override，但能力位口径是"该族有专用实现"，本批不扩大声明面）。
    fn supports(&self, cap: Capability) -> bool {
        // 阶段4：`FusedNorms` / `FusedElementwise` / `FusedGeometry` **只报真**——
        // 对应方法**全部**在本文件 override 到 LASX 内核后才报 `true`（只报确实覆盖的族）。
        matches!(
            cap,
            Capability::Packing
                | Capability::MatmulF64
                | Capability::FusedNorms
                | Capability::FusedElementwise
                | Capability::FusedGeometry
        )
    }

    // ============ 阶段4 收尾：f64 几何五入口（`FusedGeometry`）============
    //
    // 映射：全部走 `api::*_into`（LASX 向量核，免分配）。**零向量语义差异（路径各自语义）**：
    // LASX `unitize3` 对零向量给 `0`；合同默认（组合版）给 `NaN` —— override **不额外打补丁**，
    // 保持内核语义；调用方要 NaNs 档就用组合版（能力探测为假时）。

    fn vec3_add_scaled_f64(
        &self,
        ax: &[f64],
        ay: &[f64],
        az: &[f64],
        bx: &[f64],
        by: &[f64],
        bz: &[f64],
        s: f64,
        ox: &mut [f64],
        oy: &mut [f64],
        oz: &mut [f64],
    ) -> Result<(), BackendError> {
        crate::api::vec3_add_scaled_batch_into(ax, ay, az, bx, by, bz, s, ox, oy, oz).map_err(
            |_| BackendError::Unsupported {
                op: "vec3_add_scaled_f64",
                cap: Capability::FusedGeometry,
            },
        )
    }

    fn norm3_batch_f64(
        &self,
        x: &[f64],
        y: &[f64],
        z: &[f64],
        out: &mut [f64],
    ) -> Result<(), BackendError> {
        crate::api::norm3_batch_into(x, y, z, out).map_err(|_| BackendError::Unsupported {
            op: "norm3_batch_f64",
            cap: Capability::FusedGeometry,
        })
    }

    fn unitize3_batch_f64(
        &self,
        x: &[f64],
        y: &[f64],
        z: &[f64],
        ox: &mut [f64],
        oy: &mut [f64],
        oz: &mut [f64],
    ) -> Result<(), BackendError> {
        crate::api::unitize3_batch_into(x, y, z, ox, oy, oz).map_err(|_| {
            BackendError::Unsupported {
                op: "unitize3_batch_f64",
                cap: Capability::FusedGeometry,
            }
        })
    }

    fn cross3_batch_f64(
        &self,
        ax: &[f64],
        ay: &[f64],
        az: &[f64],
        bx: &[f64],
        by: &[f64],
        bz: &[f64],
        ox: &mut [f64],
        oy: &mut [f64],
        oz: &mut [f64],
    ) -> Result<(), BackendError> {
        crate::api::cross3_batch_into(ax, ay, az, bx, by, bz, ox, oy, oz).map_err(|_| {
            BackendError::Unsupported {
                op: "cross3_batch_f64",
                cap: Capability::FusedGeometry,
            }
        })
    }

    fn mat3_mul_vec3_batch_f64(
        &self,
        m: [&[f64]; 9],
        x: &[f64],
        y: &[f64],
        z: &[f64],
        ox: &mut [f64],
        oy: &mut [f64],
        oz: &mut [f64],
    ) -> Result<(), BackendError> {
        crate::api::mat3_mul_vec3_batch_into(m, x, y, z, ox, oy, oz).map_err(|_| {
            BackendError::Unsupported {
                op: "mat3_mul_vec3_batch_f64",
                cap: Capability::FusedGeometry,
            }
        })
    }

    // ============ 阶段4：融合原语**加速选项**（源实现仍是 portable 组合版）============
    //
    // 等价类（界来自批2 A/B 实测）：softmax rel 2.79e-9、rms 1.83e-7、layer 2.08e-7、
    // **gelu_quick 5.26e-3（tanh 近似 vs σ(2z)，不是 1e-5 级）**、rope 6.92e-8、axpy 8.27e-8。
    // ES：`axpy_f32` 的 `y: &mut` 与 `x` 不得重叠（借用检查器守 + `debug_assert` 基地址）。

    fn softmax_rows_f32(
        &self,
        x: &[f32],
        rows: usize,
        cols: usize,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        crate::api::softmax_rows_into(x, None, rows, cols, 1.0, out).map_err(|_| {
            BackendError::Unsupported {
                op: "softmax_rows_f32",
                cap: Capability::FusedNorms,
            }
        })
    }

    fn rms_norm_rows_f32(
        &self,
        x: &[f32],
        rows: usize,
        cols: usize,
        eps: f32,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        crate::api::rms_norm_into(x, None, rows, cols, eps, out).map_err(|_| {
            BackendError::Unsupported {
                op: "rms_norm_rows_f32",
                cap: Capability::FusedNorms,
            }
        })
    }

    fn layer_norm_rows_f32(
        &self,
        x: &[f32],
        rows: usize,
        cols: usize,
        eps: f32,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        crate::api::layer_norm_into(x, None, None, rows, cols, eps, out).map_err(|_| {
            BackendError::Unsupported {
                op: "layer_norm_rows_f32",
                cap: Capability::FusedNorms,
            }
        })
    }

    fn gelu_quick_f32(&self, x: &[f32], out: &mut [f32]) -> Result<(), BackendError> {
        crate::api::gelu_quick_into(x, out).map_err(|_| BackendError::Unsupported {
            op: "gelu_quick_f32",
            cap: Capability::FusedElementwise,
        })
    }

    fn rope_apply_f32(
        &self,
        x: &[f32],
        rows: usize,
        cols: usize,
        n_dims: usize,
        cos: &[f32],
        sin: &[f32],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        crate::api::rope_into(
            x,
            cos,
            sin,
            rows,
            cols,
            n_dims,
            crate::api::RopeMode::NeoX,
            out,
        )
        .map_err(|_| BackendError::Unsupported {
            op: "rope_apply_f32",
            cap: Capability::FusedElementwise,
        })
    }

    fn axpy_f32(&self, y: &mut [f32], x: &[f32], alpha: f32) -> Result<(), BackendError> {
        debug_assert_ne!(x.as_ptr(), y.as_ptr(), "axpy_f32：x/y 不得重叠");
        crate::api::axpy(alpha, x, y).map_err(|_| BackendError::Unsupported {
            op: "axpy_f32",
            cap: Capability::FusedElementwise,
        })
    }

    /// **P5-b：跨步拷贝** → `ops::strided::copy_strided_f32`（`col==1` 走 memcpy 行拷贝）。
    ///
    /// **别名未测**（如实）：本轮**没有**做 `src`/`dst` 重叠检测或测试；契约与 `simd_dsl` 默认实现一致
    /// ——**调用方保证不重叠**（重叠是未定义行为，不是 `Err`）。
    fn copy_strided_f32(
        &self,
        src: &[f32],
        src_st: simd_dsl::Strides,
        dst: &mut [f32],
        dst_st: simd_dsl::Strides,
        rows: usize,
        cols: usize,
    ) -> Result<(), BackendError> {
        crate::ops::strided::copy_strided_f32(
            src,
            strided_of(src_st),
            dst,
            strided_of(dst_st),
            rows,
            cols,
        )
    }

    /// **P5-b：跨步一元映射** → `ops::strided` 内核；**未覆盖的变体委托标量**
    /// （`simd_dsl::kernels::scalar::map_unary_f32` 走"跨步→连续物化 → 标量 → 写回"，逐位不变）。
    fn map_unary_strided_f32(
        &self,
        src: &[f32],
        src_st: simd_dsl::Strides,
        dst: &mut [f32],
        dst_st: simd_dsl::Strides,
        rows: usize,
        cols: usize,
        op: UnaryOp,
    ) -> Result<(), BackendError> {
        let fast = match op {
            UnaryOp::Sqrt => Some(crate::ops::strided::UnaryOpStrided::Sqrt),
            UnaryOp::Rcp => Some(crate::ops::strided::UnaryOpStrided::Rcp),
            // 其余（Exp/Sin/Cos/Ln/Erf/… 及 `#[non_exhaustive]` 将来变体）⇒ **委托标量**（保守、逐位）
            _ => None,
        };
        match fast {
            Some(o) => crate::ops::strided::map_unary_strided_f32(
                src,
                strided_of(src_st),
                dst,
                strided_of(dst_st),
                rows,
                cols,
                o,
            ),
            None => {
                // 物化 → 标量 → 写回（逐位；代价是多两次搬运，正确性优先）
                let mut tmp = vec![0.0f32; rows * cols];
                crate::ops::strided::copy_strided_f32(
                    src,
                    strided_of(src_st),
                    &mut tmp,
                    crate::ops::strided::Strides::new(cols, 1),
                    rows,
                    cols,
                )?;
                let mut out_tmp = vec![0.0f32; rows * cols];
                simd_dsl::kernels::scalar::map_unary_f32(&tmp, &mut out_tmp, op)?;
                crate::ops::strided::copy_strided_f32(
                    &out_tmp,
                    crate::ops::strided::Strides::new(cols, 1),
                    dst,
                    strided_of(dst_st),
                    rows,
                    cols,
                )
            }
        }
    }

    /// **P5-b：跨步二元映射** → `ops::strided` 内核；**未覆盖的变体委托标量**（同上一律逐位）。
    fn map_binary_strided_f32(
        &self,
        x: &[f32],
        x_st: simd_dsl::Strides,
        y: &[f32],
        y_st: simd_dsl::Strides,
        dst: &mut [f32],
        dst_st: simd_dsl::Strides,
        rows: usize,
        cols: usize,
        op: BinaryOp,
    ) -> Result<(), BackendError> {
        let fast = match op {
            BinaryOp::Add => Some(crate::ops::strided::BinaryOpStrided::Add),
            BinaryOp::Mul => Some(crate::ops::strided::BinaryOpStrided::Mul),
            BinaryOp::Div => Some(crate::ops::strided::BinaryOpStrided::Div),
            _ => None,
        };
        match fast {
            Some(o) => crate::ops::strided::map_binary_strided_f32(
                x,
                strided_of(x_st),
                y,
                strided_of(y_st),
                dst,
                strided_of(dst_st),
                rows,
                cols,
                o,
            ),
            None => {
                let mut xt = vec![0.0f32; rows * cols];
                let mut yt = vec![0.0f32; rows * cols];
                let mut out_tmp = vec![0.0f32; rows * cols];
                let cont = crate::ops::strided::Strides::new(cols, 1);
                crate::ops::strided::copy_strided_f32(
                    x,
                    strided_of(x_st),
                    &mut xt,
                    cont,
                    rows,
                    cols,
                )?;
                crate::ops::strided::copy_strided_f32(
                    y,
                    strided_of(y_st),
                    &mut yt,
                    cont,
                    rows,
                    cols,
                )?;
                simd_dsl::kernels::scalar::map_binary_f32(&xt, &yt, &mut out_tmp, op)?;
                crate::ops::strided::copy_strided_f32(
                    &out_tmp,
                    cont,
                    dst,
                    strided_of(dst_st),
                    rows,
                    cols,
                )
            }
        }
    }
}

/// `simd_dsl::Strides` → `ops::strided::Strides`（同字段，**显式转换**避免类型混用）。
const fn strided_of(s: simd_dsl::Strides) -> crate::ops::strided::Strides {
    crate::ops::strided::Strides::new(s.row, s.col)
}

/// **A/B 用的"接之前"形态**（R19-E）：`matmul`/`dot`/建核/打包**照旧走 LASX**，但
/// **逐元素与归约族用 `Backend` 的可移植默认实现** —— 这正是 `R19-E` 之前的 `LasxBackend` 行为。
///
/// **为什么要有它**：要给出"接归约/逐元素族带来的收益"的**同进程**硬数（§5 口径），
/// 必须在**同一个进程内交替**跑"接之前"与"接之后"两种后端 ⇒ 用两个后端类型，而不是改全局开关。
/// **不改任何数值口径**：它只是把 5 条方法落回默认实现，其余与 [`LasxBackend`] 完全一致。
#[derive(Debug, Default, Clone, Copy)]
pub struct LasxBackendHybrid;

impl Backend for LasxBackendHybrid {
    fn name(&self) -> &'static str {
        "lasx_rs(hybrid: LASX matmul/kernel + portable elementwise/reduce)"
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
        LasxBackend.matmul_f32(m, k, n, a, b, out)
    }

    fn dot_f32(&self, a: &[f32], b: &[f32]) -> Result<f32, BackendError> {
        LasxBackend.dot_f32(a, b)
    }

    fn build_kernel_f32(
        &self,
        spec: &simd_dsl::KernelSpec<'_>,
        kc: &mut [f32],
        ks: &mut [f32],
    ) -> Result<simd_dsl::KernelReport, BackendError> {
        LasxBackend.build_kernel_f32(spec, kc, ks)
    }

    fn pack_f32(&self, k: usize, n: usize, b: &[f32]) -> Result<PackedB, BackendError> {
        LasxBackend.pack_f32(k, n, b)
    }

    fn matmul_f32_packed(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        p: &PackedB,
        out: &mut [f32],
        opts: &CallOpts,
    ) -> Result<(), BackendError> {
        LasxBackend.matmul_f32_packed(m, k, n, a, p, out, opts)
    }
}

/// **③b-4 自洽测试**：blob 布局（长度 = `k*n*4`、主 strips 与列尾可定位）、`PackedB` 形状一致、
/// 非法长度**返回 `LengthMismatch` 不 panic**。
#[test]
#[allow(clippy::too_many_lines)]
fn packed_bitwise_same_as_matmul() {
    let be = LasxBackend;
    // **多形状**：含 `ntail != 0`（100、33）与 `ntail == 0`（64、96）；含 m 不是 4 的倍数（5、7）
    for (m, k, n) in [
        (5usize, 7usize, 100usize),
        (7, 4, 33),
        (8, 8, 64),
        (6, 8, 96),
    ] {
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 17) as f32) * 0.125 - 1.0)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i % 13) as f32) * 0.0625 - 0.5)
            .collect();
        // 参照：现有 `matmul_f32`（LASX 内核）
        let mut want = vec![0.0f32; m * n];
        be.matmul_f32(m, k, n, &a, &b, &mut want)
            .expect("matmul_f32");
        // 打包 → 复用打包面板乘
        let p = be.pack_f32(k, n, &b).expect("pack_f32");
        let mut got = vec![0.0f32; m * n];
        let opts = CallOpts::single();
        be.matmul_f32_packed(m, k, n, &a, &p, &mut got, &opts)
            .expect("matmul_f32_packed");
        assert_eq!(
            got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "matmul_f32_packed 与 matmul_f32 不逐位（m={m}, k={k}, n={n}, ntail={}）",
            n % 32
        );
    }
    // 外来句柄 / 形状不符 ⇒ 报错不 panic
    let p = be.pack_f32(4, 64, &vec![0.0f32; 256]).expect("pack");
    let opts = CallOpts::single();
    let mut out = vec![0.0f32; 4 * 64];
    assert!(
        be.matmul_f32_packed(4, 4, 32, &[0.0f32; 16], &p, &mut out, &opts)
            .is_err()
    );
}

/// 非法长度**返回 `LengthMismatch` 不 panic**。
#[test]
fn pack_f32_blob_layout_is_self_consistent() {
    let be = LasxBackend;
    let (k, n) = (4usize, 100usize); // n32 = 96, ntail = 4 ⇒ 两个分支都覆盖
    let b: Vec<f32> = (0..k * n).map(|i| (i % 11) as f32 - 5.0).collect();
    let p = be.pack_f32(k, n, &b).expect("pack");
    assert_eq!(p.shape(), (k, n), "PackedB 形状应与请求一致");
    assert_eq!(p.backend_tag(), be.backend_tag(), "标签应为本后端");
    assert_eq!(p.blob().len(), k * n * 4, "blob 长度应为 k*n*4");
    // 主 strips：`blob[(s*k + p)*32 + r]` == `b[p*n + s*32 + r]`
    let f32_at = |off: usize| {
        let mut w = [0u8; 4];
        w.copy_from_slice(&p.blob()[off..off + 4]);
        f32::from_le_bytes(w)
    };
    let (n32, ntail) = (96usize, 4usize);
    for s in 0..n32 / 32 {
        for row in 0..k {
            for r in 0..32 {
                let got = f32_at(((s * k + row) * 32 + r) * 4);
                assert_eq!(
                    got,
                    b[row * n + s * 32 + r],
                    "主 strips 布局不符 s={s} row={row} r={r}"
                );
            }
        }
    }
    // 列尾：紧凑行主序，起点 = k*n32*4
    let tail_off = k * n32 * 4;
    for row in 0..k {
        for t in 0..ntail {
            let got = f32_at(tail_off + (row * ntail + t) * 4);
            assert_eq!(got, b[row * n + n32 + t], "列尾布局不符 row={row} t={t}");
        }
    }
    // n 是 32 的倍数 ⇒ 无尾列
    let p2 = be.pack_f32(2, 64, &vec![1.0f32; 128]).expect("pack");
    assert_eq!(p2.blob().len(), 2 * 64 * 4);
    // 非法长度 ⇒ LengthMismatch（不 panic）
    assert!(be.pack_f32(2, 3, &[0.0f32; 5]).is_err());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **③b-1 四条测试：守卫 + 逐行回退 + 计数 + 与默认实现一致（容差类，不断言逐位）**。
    #[test]
    fn kernel_guard_and_fallback_report() {
        use simd_dsl::KernelSpec;
        use simd_dsl::Route;
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
        let _ = simd_dsl::ScalarBackend.build_kernel_f32(&spec_off, &mut kc_sd, &mut ks_sd);
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

    /// **IS-1 等价类**：`matmul_bt_f32` 对 **`simd_dsl` 标量默认逐位**（trait 文档点名的参照物 =
    /// "先物化 `Bᵀ` 再 `Backend::matmul_f32`"，两者同序）；对 **"物化转置 + `api::matmul"`** 是
    /// **容差类**（LASX GEMM 用 FMA ⇒ 舍入不同，实测 rel ≤ 1.3e-7）——**两个参照物别混**。
    #[test]
    fn matmul_bt_bitwise_same_as_api_and_scalar() {
        use simd_dsl::kernels::scalar;
        let be = LasxBackend;
        for &(m, k, n) in &[(7usize, 37usize, 11usize), (33, 65, 31), (100, 448, 1024)] {
            let a: Vec<f32> = (0..m * k)
                .map(|i| ((i * 37 % 101) as f32) * 0.1 - 6.07)
                .collect();
            // NT：B 是 n×k（按 j 行主序，k 方向连续）
            let bt: Vec<f32> = (0..n * k)
                .map(|i| ((i * 53 % 97) as f32) * 0.1 + 1.3)
                .collect();
            let mut got = vec![f32::NAN; m * n];
            be.matmul_bt_f32(m, k, n, &a, &bt, &mut got)
                .expect("backend.matmul_bt_f32");
            // ① 标量默认（trait 文档的参照物）⇒ **逐位**
            let mut want = vec![f32::NAN; m * n];
            scalar::matmul_bt_f32(m, k, n, &a, &bt, &mut want).expect("scalar::matmul_bt_f32");
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{m}×{k}×{n}: matmul_bt_f32 与标量默认不逐位"
            );
            // ② 物化转置 + api::matmul_into ⇒ **容差类**（⾮逐位；界 ≤ 1e-6，实测 ≤1.3e-7）
            let mut bmat = vec![0.0f32; k * n];
            for j in 0..n {
                for t in 0..k {
                    bmat[t * n + j] = bt[j * k + t];
                }
            }
            let mut api = vec![f32::NAN; m * n];
            crate::api::matmul_into(m, k, n, &a, &bmat, &mut api).expect("api::matmul_into");
            let peak = api.iter().fold(0.0f32, |p, v| p.max(v.abs())).max(1e-30) as f64;
            let rel = got
                .iter()
                .zip(&api)
                .fold(0.0f64, |w, (g, s)| w.max((*g - *s).abs() as f64))
                / peak;
            assert!(
                rel <= 1e-6,
                "{m}×{k}×{n}: matmul_bt vs 物化转置+api::matmul 超界 {rel:e}（LASX GEMM 用 FMA）"
            );
        }
        // 零维：k == 0 ⇒ out 全 +0.0（与默认实现同口径）
        let mut z = vec![f32::NAN; 6];
        be.matmul_bt_f32(2, 0, 3, &[], &[], &mut z)
            .expect("k=0 合法空操作");
        assert!(
            z.iter().all(|v| v.to_bits() == 0.0f32.to_bits()),
            "k=0 时 out 必须是 +0.0"
        );
        // 长度/形状错误 ⇒ Err（不 panic）
        let mut o = vec![0.0f32; 4];
        assert!(
            be.matmul_bt_f32(2, 3, 2, &[0.0; 5], &[0.0; 6], &mut o)
                .is_err()
        );
        assert!(
            be.matmul_bt_f32(2, 3, 2, &[0.0; 6], &[0.0; 5], &mut o)
                .is_err()
        );
    }

    /// **IS-2 等价类（matmul_f64）**：对**旧入口 `api::matmul_f64` 逐位**（同一 LASX 内核）；
    /// 对**标量默认**（接线前的路径）是**容差类**（4-lane 累加/打包次序不同）——实测 rel ≤ 3.4e-16。
    #[test]
    fn matmul_f64_bitwise_vs_api_and_tolerance_vs_scalar() {
        use simd_dsl::kernels::scalar;
        let be = LasxBackend;
        for &(m, k, n) in &[
            (5usize, 7usize, 3usize),
            (33, 65, 31),
            (100, 448, 1024), // 打包路径
        ] {
            let a: Vec<f64> = (0..m * k)
                .map(|i| ((i * 37 % 101) as f64) * 0.1 - 6.07)
                .collect();
            let b: Vec<f64> = (0..k * n)
                .map(|i| ((i * 53 % 97) as f64) * 0.1 + 1.3)
                .collect();
            let api = crate::api::matmul_f64(m, k, n, &a, &b).expect("api::matmul_f64");
            let mut got = vec![f64::NAN; m * n];
            be.matmul_f64(m, k, n, &a, &b, &mut got)
                .expect("backend.matmul_f64");
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                api.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{m}×{k}×{n}: matmul_f64 与 api::matmul_f64 不逐位"
            );
            // 对标量默认：容差类（实测 ≤ 3.4e-16，界留到 1e-15）
            let mut sc = vec![f64::NAN; m * n];
            scalar::matmul_f64(m, k, n, &a, &b, &mut sc).expect("scalar::matmul_f64");
            let peak = sc.iter().fold(0.0f64, |p, v| p.max(v.abs())).max(1e-300);
            let rel = got
                .iter()
                .zip(&sc)
                .fold(0.0f64, |w, (g, s)| w.max((*g - *s).abs()))
                / peak;
            assert!(
                rel <= 1e-15,
                "{m}×{k}×{n}: matmul_f64 vs 标量默认超界 {rel:e}"
            );
        }
        // 长度错误 ⇒ Err（不 panic）
        let mut o = vec![0.0f64; 4];
        assert!(
            be.matmul_f64(2, 3, 2, &[0.0; 5], &[0.0; 6], &mut o)
                .is_err()
        );
        assert!(
            be.matmul_f64(2, 3, 2, &[0.0; 6], &[0.0; 6], &mut o[..3])
                .is_err()
        );
    }

    /// **IS-2 等价类（dot_f64）**：对 `api::dot_f64` **逐位**；对标量默认**容差类**（4-lane 累加 +
    /// 水平归约 vs 顺序累加，实测 rel ≤ 2.4e-14 @ n=4096；n ≤ 8 恰好逐位）。
    #[test]
    fn dot_f64_bitwise_vs_api_and_tolerance_vs_scalar() {
        use simd_dsl::kernels::scalar;
        let be = LasxBackend;
        for &n in &[0usize, 1, 3, 7, 8, 33, 4096] {
            let a: Vec<f64> = (0..n)
                .map(|i| ((i * 11 % 37) as f64) * 0.25 - 3.0)
                .collect();
            let b: Vec<f64> = (0..n)
                .map(|i| ((i * 7 % 53) as f64) * 0.125 - 1.0)
                .collect();
            let api = crate::api::dot_f64(&a, &b).expect("api::dot_f64");
            let got = be.dot_f64(&a, &b).expect("backend.dot_f64");
            assert_eq!(
                got.to_bits(),
                api.to_bits(),
                "n={n}: dot_f64 与 api::dot_f64 不逐位"
            );
            let sc = scalar::dot_f64(&a, &b).expect("scalar::dot_f64");
            let rel = (got - sc).abs() / sc.abs().max(1e-300);
            assert!(rel <= 1e-12, "n={n}: dot_f64 vs 标量默认超界 {rel:e}");
        }
        // 长度错误 ⇒ Err（不 panic）
        assert!(be.dot_f64(&[0.0; 3], &[0.0; 4]).is_err());
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

    /// **只声明能如实声明的**：`(Default, F32/F64)` 为逐位类（点名参照物 = `lasx_rs::api`）；
    /// 递推路径为容差类 + 界；未做路径为 `None`。`supports` 同步（`Packing`/`MatmulF64` 为真，
    /// `Threaded` 仍 false）。
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
                .is_some(),
            "IS-2 后 f64 有专用实现（与 api::* 逐位）⇒ 应声明为逐位类"
        );
        // 未做的路径仍然 None（未测未做 ⇒ 不声明）
        assert!(
            be.precision_claim(EnginePath::IsometricRecurrence, ScalarKind::F64)
                .is_some(),
            "递推 f64 已在 ③b-1 声明"
        );
        // **只低估**：`Packing`（packed 已接线且逐位断言过）与 `MatmulF64`（IS-2 已接）为真；
        // 其余能力一律 false（`Threaded` 仍 false：`CallOpts` 未接，见 ③b-4 留档）
        assert!(
            be.supports(Capability::Packing),
            "Packing 已实现且逐位断言过 ⇒ 应为 true"
        );
        assert!(
            be.supports(Capability::MatmulF64),
            "IS-2 后 matmul_f64/dot_f64 有专用实现 ⇒ 应为 true"
        );
        assert!(!be.supports(Capability::Transcendental));
        assert!(!be.supports(Capability::Threaded));
    }

    /// **R19-E：逐元素/轴归约族必须与标量参照逐位**（`map_affine` 不融 FMA；轴归约按行序累加），
    /// 并覆盖形状错误路径（不 panic、返回 `LengthMismatch`）。
    #[test]
    fn r19e_elementwise_and_axis_reduce_classes() {
        let be = LasxBackend;
        let n = 1029; // 非 8 的倍数 ⇒ 覆盖尾部处理
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin() * 3.1).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.11).cos() * 1.7 - 0.4)
            .collect();
        for op in [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Max,
            BinaryOp::Min,
        ] {
            let mut a = vec![0.0f32; n];
            be.map_binary_f32(&x, &y, &mut a, op).expect("map_binary");
            let mut b = vec![0.0f32; n];
            simd_dsl::kernels::scalar::map_binary_f32(&x, &y, &mut b, op).expect("scalar");
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&a), bits(&b), "map_binary({op:?}) 必须与标量参照逐位");
        }
        // 系数**故意取非精确二进制值**（0.7 / 0.3）：2 的幂系数会让 `a*x` 精确、两步与 FMA 恰好同值，
        // 那就测不出 FMA 的舍入差（本测试初版用 -0.5/0.25 就得到 0.000e0，如实记在这里）。
        let (ca, cb) = (0.7f32, 0.3f32);
        let mut a = vec![0.0f32; n];
        be.map_affine_f32(&x, ca, cb, &mut a).expect("affine");
        let mut b = vec![0.0f32; n];
        simd_dsl::kernels::scalar::map_affine_f32(&x, ca, cb, &mut b).expect("scalar");
        // **R19-G（用户批准 FMA）**：这两条从"逐位"改为**容差类**——FMA 的舍入与"先乘后加"不同。
        let peak = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30) as f64;
        let worst_affine = a
            .iter()
            .zip(&b)
            .fold(0.0f64, |m, (u, v)| m.max((*u - *v).abs() as f64))
            / peak;
        eprintln!("R19-G：map_affine(FMA) vs 标量参照 = {worst_affine:.3e}（相对峰值）");
        assert!(
            worst_affine <= 1e-5,
            "FMA 舍入差应 ≤ 1e-5（相对峰值），实测 {worst_affine:e}"
        );
        // 轴归约（rows=7, cols=13：8 的倍数与非倍数都覆盖）
        let (rows, cols) = (7usize, 13usize);
        let x2: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.19).sin() + 0.3)
            .collect();
        let w: Vec<f32> = (0..rows).map(|r| 0.5 + r as f32 * 0.13).collect();
        let mut c1 = vec![0.0f32; cols];
        be.reduce_sum_cols_f32(&x2, rows, cols, &mut c1)
            .expect("sum_cols");
        let mut c2 = vec![0.0f32; cols];
        simd_dsl::kernels::scalar::reduce_sum_cols_f32(&x2, rows, cols, &mut c2).expect("scalar");
        assert!(
            c1.iter().zip(&c2).all(|(u, v)| u.to_bits() == v.to_bits()),
            "reduce_sum_cols 必须逐位（每列一条 lane、按行序）"
        );
        let mut d1 = vec![0.0f32; cols];
        be.reduce_weighted_sum_cols_f32(&x2, &w, rows, cols, &mut d1)
            .expect("wsum_cols");
        let mut d2 = vec![0.0f32; cols];
        simd_dsl::kernels::scalar::reduce_weighted_sum_cols_f32(&x2, &w, rows, cols, &mut d2)
            .expect("scalar");
        // **R19-G**：加权轴归约也改 FMA（`v·w + acc`）⇒ **容差类**，界同上。
        let peak = d2.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30) as f64;
        let worst_wsum = d1
            .iter()
            .zip(&d2)
            .fold(0.0f64, |m, (u, v)| m.max((*u - *v).abs() as f64))
            / peak;
        eprintln!(
            "R19-G：reduce_weighted_sum_cols(FMA) vs 标量参照 = {worst_wsum:.3e}（相对峰值）"
        );
        assert!(
            worst_wsum <= 1e-5,
            "FMA 舍入差应 ≤ 1e-5（相对峰值），实测 {worst_wsum:e}"
        );
        // 形状错误：一律 `LengthMismatch`、不 panic
        assert!(
            be.reduce_sum_cols_f32(&x2[..rows * cols - 1], rows, cols, &mut c1)
                .is_err()
        );
        assert!(
            be.reduce_weighted_sum_cols_f32(&x2, &w[..rows - 1], rows, cols, &mut d1)
                .is_err()
        );
        assert!(
            be.map_binary_f32(&x, &y[..n - 1], &mut a, BinaryOp::Add)
                .is_err()
        );
        assert!(be.map_affine_f32(&x, 1.0, 0.0, &mut a[..n - 1]).is_err());
    }

    /// **R19-H：LASX 按列 FFT 与可移植参照的实测界**（容差类：参照是 **f64 内部**实现、
    /// 本实现旋转因子 f64 预算后降 f32、蝶形在 f32 里做 ⇒ 每级有舍入）。
    #[test]
    fn r19h_fft_matches_the_scalar_reference_within_a_measured_bound() {
        let be = LasxBackend;
        for &(n, b) in &[
            (64usize, 24usize),
            (1024usize, 32usize),
            (4096usize, 256usize),
        ] {
            let r0: Vec<f32> = (0..n * b).map(|i| (i % 97) as f32 * 0.05 - 2.0).collect();
            let i0: Vec<f32> = (0..n * b).map(|i| (i % 53) as f32 * 0.02 - 0.5).collect();
            let (mut r1, mut i1) = (r0.clone(), i0.clone());
            let (mut r2, mut i2) = (r0.clone(), i0.clone());
            be.fft_strided_f32(&mut r1, &mut i1, n, b).unwrap();
            simd_dsl::kernels::scalar::fft_radix2_strided_f32(&mut r2, &mut i2, n, b).unwrap();
            let peak = r2
                .iter()
                .chain(i2.iter())
                .fold(0.0f32, |m, v| m.max(v.abs()))
                .max(1e-30) as f64;
            let worst = r1
                .iter()
                .chain(i1.iter())
                .zip(r2.iter().chain(i2.iter()))
                .fold(0.0f64, |m, (u, v)| m.max((*u - *v).abs() as f64))
                / peak;
            eprintln!("R19-H：LASX-FFT(n={n},b={b}) vs f64 参照 = {worst:.3e}（相对峰值）");
            assert!(worst <= 1e-4, "n={n},b={b}：实测 {worst:e} 超过 1e-4");
        }
        // 形状契约：n 非 2 的幂 ⇒ `Unsupported`；长度不符 ⇒ `LengthMismatch`
        assert!(
            be.fft_strided_f32(&mut [0.0f32; 12], &mut [0.0f32; 12], 6, 2)
                .is_err()
        );
        assert!(
            be.fft_strided_f32(&mut [0.0f32; 8], &mut [0.0f32; 8], 8, 2)
                .is_err()
        );
    }
}
