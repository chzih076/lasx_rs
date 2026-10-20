//! **`simd_dsl::Backend` 的 `lasx_rs` 实现**（步骤 ③a；**feature 门控**：`dsl-backend`）。
//!
//! # 范围（**当前状态，截至 B1b 组合原语补齐**）
//! - **必需方法**（trait 无默认值）：`matmul_f32`、`dot_f32` ⇒ **接到现有 LASX 内核**
//!   （`ops::matmul::matmul_f32` / `ops::dot::dot`）——**不绕开、不新写算法、不改语义**；
//! - **已覆盖（本仓有内核可接的）**：`matmul_f32`/`dot_f32`、**`matmul_f64`/`dot_f64`（IS-2）**、
//!   **`matmul_bt_f32`（IS-1）**、`reduce_sum_f32`、`map_affine_f32`、`map_binary_f32`、
//!   **`map_binary_{rows,cols,scalar}_f32`（P4.2 广播三入口：`ops/broadcast.rs`；`Add/Sub/Mul/Div`
//!   逐位、`Max/Min` 委托标量）**、
//!   **`map_binary_f64`/`map_affine_f64`（P1-4，LASX f64 逐元素）**、
//!   **`quantize_i8_rows_f32`/`dot_i8`/`gemv_i8`（P1-5，LASX i8 内核；`dequantize_i8_rows_f32`
//!   未接 ⇒ `Capability::Quant` 仍 `false`）**、
//!   `reduce_{sum,weighted_sum,centered_sq_sum}_cols_f32`、`matmul_pair_psd_f32`、
//!   `gather_rows_f32`/`gather_rows_i8`/`scatter_rows_f32`、`fft_strided_f32`、
//!   `build_kernel_f32`（等距递推核）、`pack_f32`/`matmul_f32_packed`、
//!   `copy_strided_f32`/`map_binary_strided_f32`、
//!   **`map_unary_f32`/`reduce_rows_map_f32`/`axpy_inplace_f32`（B1b 组合原语；内核在 `ops/prims.rs` 与 `ops/axpy.rs`）**、
//!   **`reduce_sum_rows_f32`（P4.2：LASX 8-lane 行和 + 行块自动并行；与默认实现**逐位**）**；
//! - **仍未覆盖（**没有 LASX 内核**，或按分批计划不做）**：rows 方向归约的其余项
//!   （`reduce_max_f32` 等）、`max/min_{cols,rows}`、banded/stencil、
//!   **f64 逐元素的其他族（如 f64 轴归约、`map_unary_f64`）** ⇒ **走 `simd_dsl` 标量默认**
//!   （**不 panic**；性能必然不如向量版，迁移时按"无内核 ⇒ 不替换"如实记）；
//! - **线程**：`CallOpts{threads}` **未接**（本后端不按 opts 派活，见上文 ③b-4 留档）；但 **GEMM 内核
//!   自己**在 `work ≥ 32M` 时走**进程级全局池**（`ops::matmul::matmul_f32`，§7.28；`init_global(1)`
//!   是逃生门）⇒ **大形状 A/B 的 `threads` 会 >1**，必须在表里打印（`examples/migrate_ab.rs` 已照做）。
//!
//! # `precision_claim` 只填**能如实声明**的部分
//! `(Default, F32/F64)` 上被覆盖的**内核方法**（`matmul_f32`/`dot_f32`/`matmul_f64`/`dot_f64`）
//! 与 `lasx_rs::api` **逐位一致**（同一内核；测试 `bitwise_same_as_api` 与
//! `matmul_f64_and_dot_f64_bitwise_vs_api` 钉住）⇒ 声明为**逐位类**并**点名参照物** = `lasx_rs::api`。
//!
//! **IS-4 校正（审计复核记录）**：`map_affine_f32` 与 `reduce_weighted_sum_cols_f32` 用 **FMA**
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
    FusedIo, FusedPlan, PackedB, PatternKind, PrecisionClaim, ReduceMapOp, RowFuse, ScalarKind,
    UnaryOp,
};

/// 用 LASX 现有内核的 `simd_dsl::Backend`。
#[derive(Debug, Default, Clone, Copy)]
pub struct LasxBackend;

/// **仅测试钩子**：`LASX_NO_PLAN` 已设置 ⇒ `supports_plan`/`FusedNorms`/`FusedElementwise`
/// 报 false，让 fsync 探针能在**同一后端**上测得生产组合路径（性能 A/B 用）。
/// 默认（未设置）行为逐字不变；本函数不参与任何生产分支的语义。
fn no_plan_hook() -> bool {
    std::env::var_os("LASX_NO_PLAN").is_some()
}

/// 长度不符时的错误（与 `simd_dsl` 的默认实现同类）。
fn len_err(op: &'static str, what: &'static str, expected: usize, got: usize) -> BackendError {
    BackendError::LengthMismatch {
        op,
        what,
        expected,
        got,
    }
}

/// **轴 `max/min` 归约的公共校验**（P4.3；与 `simd_dsl::kernels::scalar` 的默认实现**逐条同口径**）：
/// 先 `rows×cols` 溢出、再 `x`/`out` 长度、最后"归约轴长度为 0"（错误信息逐字一致）。
fn axis_extreme_check(
    op: &'static str,
    x: &[f32],
    rows: usize,
    cols: usize,
    out: &[f32],
    out_len: usize,
    axis_len: usize,
) -> Result<(), BackendError> {
    let total = rows.checked_mul(cols).ok_or(BackendError::ShapeOverflow {
        op,
        what: "rows×cols",
    })?;
    if x.len() != total {
        return Err(len_err(op, "x", total, x.len()));
    }
    if out.len() != out_len {
        return Err(len_err(op, "out", out_len, out.len()));
    }
    if axis_len == 0 {
        return Err(BackendError::LengthMismatch {
            op,
            what: "rows/cols（max/min 归约至少需要一个元素）",
            expected: 1,
            got: 0,
        });
    }
    Ok(())
}

/// `BinaryOp` → LASX 逐元素 `BinOp`（**只映射逐位安全的 4 条**；`Max/Min` 与 `#[non_exhaustive]`
/// 的未来变体返回 `None` ⇒ 调用方委托 `simd_dsl::kernels::scalar`——Rust `f32::max/min` 的 NaN
/// 语义 ≠ LASX `xvfmax/xvfmin` 的 IEEE 语义，不能向量化还声称逐位）。
fn lasx_binop(op: BinaryOp) -> Option<crate::ops::dsl_map::BinOp> {
    match op {
        BinaryOp::Add => Some(crate::ops::dsl_map::BinOp::Add),
        BinaryOp::Sub => Some(crate::ops::dsl_map::BinOp::Sub),
        BinaryOp::Mul => Some(crate::ops::dsl_map::BinOp::Mul),
        BinaryOp::Div => Some(crate::ops::dsl_map::BinOp::Div),
        _ => None,
    }
}

/// **广播二元的公共校验**（`map_binary_{rows,cols}_f32`；与 `simd_dsl::kernels::scalar` 的默认
/// 实现**逐条同口径**）：先 `rows×cols` 溢出、再 `x`/`out` 等长、最后 `w` 长度（`w_what` 与默认
/// 实现的措辞一致）。空形状不在这里早退——默认实现也是先校验再空循环，严格保持错误面不变。
#[allow(clippy::too_many_arguments)] // 形状 4 个 + 三块缓冲 + 长度/措辞 = 与默认实现同口径的校验参数
fn binary_broadcast_check(
    op: &'static str,
    x: &[f32],
    rows: usize,
    cols: usize,
    w: &[f32],
    out: &[f32],
    w_len: usize,
    w_what: &'static str,
) -> Result<(), BackendError> {
    let total = rows.checked_mul(cols).ok_or(BackendError::ShapeOverflow {
        op,
        what: "rows×cols",
    })?;
    if x.len() != total {
        return Err(len_err(op, "x", total, x.len()));
    }
    if out.len() != total {
        return Err(len_err(op, "out", total, out.len()));
    }
    if w.len() != w_len {
        return Err(len_err(op, w_what, w_len, w.len()));
    }
    Ok(())
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
    /// **未接** `parallel::matmul_f64`：当前 override 走单线程 `ops::matmul_f64`；池版 f64 GEMM
    /// 未做，与 `CallOpts{threads}` 未接的状态一致（见上文 ③b-4 留档）。
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
    /// * **`map_affine_f32` / `reduce_weighted_sum_cols_f32`：容差类**（**IS-4 校正**，审计复核记录）——
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

    /// **P4.2 广播入口（rows 方向）**：`out[r·cols+c] = op(x[r·cols+c], w[r])`（`w.len() == rows`）。
    ///
    /// `Add`/`Sub`/`Mul`/`Div` 走 [`crate::ops::broadcast::map_binary_rows`]：**每行 `w[r]` 只读一次**
    /// 并 splat 进 SIMD、行内连续扫 `cols`（4 路展开 + 8-lane 尾 + 标量尾）——逐元素无重排
    /// ⇒ 与 `kernels::scalar` 默认**逐位**。`Max`/`Min`/未来变体委托标量参照（NaN 语义不同）。
    fn map_binary_rows_f32(
        &self,
        x: &[f32],
        rows: usize,
        cols: usize,
        w: &[f32],
        out: &mut [f32],
        op: BinaryOp,
    ) -> Result<(), BackendError> {
        let name = "map_binary_rows_f32";
        binary_broadcast_check(name, x, rows, cols, w, out, rows, "w（广播轴 = rows）")?;
        match lasx_binop(op) {
            Some(fop) => {
                debug_assert!(
                    !std::ptr::eq(x.as_ptr(), out.as_ptr()),
                    "map_binary_rows_f32: out 不得与 x 重叠（契约）"
                );
                crate::ops::broadcast::map_binary_rows(x, rows, cols, w, out, fop);
                Ok(())
            }
            None => simd_dsl::kernels::scalar::map_binary_rows_f32(x, rows, cols, w, out, op),
        }
    }

    /// **P4.2 广播入口（cols 方向）**：`out[r·cols+c] = op(x[r·cols+c], w[c])`（`w.len() == cols`）。
    ///
    /// `Add`/`Sub`/`Mul`/`Div` 走 [`crate::ops::broadcast::map_binary_cols`]：**每行**用连续 `w` 向量
    /// 直接调用逐元素 SIMD 核（`ops::dsl_map::map_binary`）⇒ 与默认**逐位**；`Max/Min` 委托标量。
    fn map_binary_cols_f32(
        &self,
        x: &[f32],
        rows: usize,
        cols: usize,
        w: &[f32],
        out: &mut [f32],
        op: BinaryOp,
    ) -> Result<(), BackendError> {
        let name = "map_binary_cols_f32";
        binary_broadcast_check(name, x, rows, cols, w, out, cols, "w（广播轴 = cols）")?;
        match lasx_binop(op) {
            Some(fop) => {
                debug_assert!(
                    !std::ptr::eq(x.as_ptr(), out.as_ptr()),
                    "map_binary_cols_f32: out 不得与 x 重叠（契约）"
                );
                crate::ops::broadcast::map_binary_cols(x, rows, cols, w, out, fop);
                Ok(())
            }
            None => simd_dsl::kernels::scalar::map_binary_cols_f32(x, rows, cols, w, out, op),
        }
    }

    /// **P4.2 标量二元入口**：`out[i] = op(x[i], scalar)`。
    ///
    /// `Add`/`Sub`/`Mul`/`Div` 走 [`crate::ops::broadcast::map_binary_scalar`]（整段 SIMD，标量提前
    /// splat；与默认**逐位**）；`Max`/`Min` 委托标量参照（NaN 语义不同）。
    fn map_binary_scalar_f32(
        &self,
        x: &[f32],
        scalar: f32,
        out: &mut [f32],
        op: BinaryOp,
    ) -> Result<(), BackendError> {
        let name = "map_binary_scalar_f32";
        if out.len() != x.len() {
            return Err(len_err(name, "out", x.len(), out.len()));
        }
        match lasx_binop(op) {
            Some(fop) => {
                debug_assert!(
                    !std::ptr::eq(x.as_ptr(), out.as_ptr()),
                    "map_binary_scalar_f32: out 不得与 x 重叠（契约）"
                );
                crate::ops::broadcast::map_binary_scalar(x, scalar, out, fop);
                Ok(())
            }
            None => simd_dsl::kernels::scalar::map_binary_scalar_f32(x, scalar, out, op),
        }
    }

    /// **P1-4：f64 逐元素 LASX 覆盖**（4 f64/lane + 尾部标量；**逐位类**）。
    ///
    /// `Add`/`Sub`/`Mul`/`Div` 由 `ops::f64_map::map_binary` 向量化；**`Max`/`Min` 仍委托标量**：
    /// Rust `f64::max` 的 NaN 语义（另一侧是数就返回它）≠ LASX `xvfmax_d` 的 IEEE 语义。
    /// **等价类：逐位**（向量指令与标量 IEEE 运算逐位同结果；测试 `f64_map_is_bitwise_vs_scalar`）。
    fn map_binary_f64(
        &self,
        x: &[f64],
        y: &[f64],
        out: &mut [f64],
        op: BinaryOp,
    ) -> Result<(), BackendError> {
        if y.len() != x.len() {
            return Err(len_err("map_binary_f64", "y", x.len(), y.len()));
        }
        if out.len() != x.len() {
            return Err(len_err("map_binary_f64", "out", x.len(), out.len()));
        }
        let fast = match op {
            BinaryOp::Add => Some(crate::ops::f64_map::BinOpF64::Add),
            BinaryOp::Sub => Some(crate::ops::f64_map::BinOpF64::Sub),
            BinaryOp::Mul => Some(crate::ops::f64_map::BinOpF64::Mul),
            BinaryOp::Div => Some(crate::ops::f64_map::BinOpF64::Div),
            // NaN 语义按 Rust ⇒ 委托标量（同 f32 的口径）
            _ => None,
        };
        match fast {
            Some(fop) => {
                debug_assert!(
                    !std::ptr::eq(x.as_ptr(), out.as_ptr()),
                    "map_binary_f64: out 不得与 x 重叠（契约）"
                );
                debug_assert!(
                    !std::ptr::eq(y.as_ptr(), out.as_ptr()),
                    "map_binary_f64: out 不得与 y 重叠（契约）"
                );
                crate::ops::f64_map::map_binary(x, y, out, fop);
            }
            None => simd_dsl::kernels::scalar::map_binary_f64(x, y, out, op)?,
        }
        Ok(())
    }

    /// **P1-4：`map_affine_f64` 的 LASX 覆盖**（`a·x + b`，**故意不融 FMA** ⇒ **逐位类**）。
    ///
    /// 用两条向量指令（`xvfmul_d` + `xvfadd_d`）换"与标量参照 `a * v + b` 同序"；
    /// 若改单条 `xvfmadd_d` 会降到容差类（f32 版本就是这样）——本批优先逐位。
    fn map_affine_f64(
        &self,
        x: &[f64],
        a: f64,
        b: f64,
        out: &mut [f64],
    ) -> Result<(), BackendError> {
        if out.len() != x.len() {
            return Err(len_err("map_affine_f64", "out", x.len(), out.len()));
        }
        debug_assert!(
            !std::ptr::eq(x.as_ptr(), out.as_ptr()),
            "map_affine_f64: out 不得与 x 重叠（契约）"
        );
        crate::ops::f64_map::map_affine(a, b, x, out);
        Ok(())
    }

    // ============ P1-5：i8 量化族 → **P1 批3 起迁入 `exec_fused_plan`**（`FusedPlan` 自带 i8/i32 缓冲）============
    //
    // 批3（`fd37e6b`）删除了 `quantize_i8_rows_f32` / `dequantize_i8_rows_f32` / `dot_i8` / `gemv_i8`
    // 四个具名方法（`Backend` 66→62），改成 `FusedPlan::{QuantizeI8Rows,DequantizeI8Rows,DotI8,GemvI8}`。
    // **本后端映射**（Plan 臂在 `exec_fused_plan` 里实现）：
    // * `QuantizeI8Rows` → `ops::quant_i8::quantize_i8_rows_contract`（LASX 向量核；ties-even +
    //   先舍入后 clamp（内核用"浮点钳位端点为整数 ⇒ 与先取整再钳位等价"论证 + 边界单测钉住）、
    //   `NaN⇒0`、`±inf⇒±端点`）；
    // * `DotI8` → `ops::dot_i8::dot_i8_saturating`（i64 精确累加 + **末端一次**饱和到 i32）；
    // * `GemvI8` → `ops::gemv_i8::gemv_i8_i32`（逐行复用饱和 dot；**不是**中途 `saturating_add`）；
    // * **`DequantizeI8Rows` 未接 LASX 核**（如实）：LASX 拓宽类指令是 128-bit lane 语义
    //   （`xvsllwil_h_b` 取每个半区低 8 字节 ⇒ 链到 w 后 lane 顺序 `[b0..b3,b16..b19]`），
    //   拼成连续 8 字节块需 4 次 unaligned load + `xvpermi_q`，收益不足 ⇒
    //   `supports_plan(DequantizeI8Rows) = false`、`exec_fused_plan` 该臂**回退 portable 解释**
    //   （`scalar::dequantize_i8_rows_f32`）——**不虚报**。
    // ⇒ `Capability::Quant` 语义由 `supports_plan` 探测；本后端**不报**该位（只低估）。

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

    /// **行和 override（P4.2）**：`out[r] = Σ_c x[r·cols + c]`。
    ///
    /// **等价类 = 逐位**：本仓内核（`ops::dsl_map::reduce_sum_rows`）用 LASX 8-lane 向量加法做
    /// **与默认实现完全相同**的 per-lane 累加，水平归约走**顺序 fold**（不是两两树）、尾部顺序补加
    /// ⇒ 与 `simd_dsl::kernels::scalar::reduce_sum_rows_f32` 逐位一致（测试 `to_bits` 钉住）。
    ///
    /// **自动并行**：`work ≥ PAR_MIN_SUM_ROWS_ELEMS`（**4M 元素**，实测定：8K–3.1M 中性、
    /// 4.2M 起 8.5×/2.6×/2.3×）且池线程 > 1 时按行块并行
    /// （`parallel::reduce_sum_rows`；线程数由 `pool::global()`/`init_global(n)` 控制，默认 auto）。
    /// 行块切分**不改变**任何行的归约次序 ⇒ 并行仍是**逐位类**。
    ///
    /// 错误面与默认实现一致（`rows×cols` 溢出 ⇒ `ShapeOverflow`；长度不符 ⇒ `LengthMismatch`）。
    fn reduce_sum_rows_f32(
        &self,
        x: &[f32],
        rows: usize,
        cols: usize,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let op = "reduce_sum_rows_f32";
        let total = rows.checked_mul(cols).ok_or(BackendError::ShapeOverflow {
            op,
            what: "rows×cols",
        })?;
        if x.len() != total {
            return Err(len_err(op, "x", total, x.len()));
        }
        if out.len() != rows {
            return Err(len_err(op, "out", rows, out.len()));
        }
        debug_assert!(
            !std::ptr::eq(x.as_ptr(), out.as_ptr()),
            "{op}: out 不得与 x 重叠（契约）"
        );
        crate::ops::dsl_map::reduce_sum_rows(x, rows, cols, out);
        Ok(())
    }

    // ================= P4.3：`max/min` 轴归约 + 整片归约 override =================
    //
    // **等价类 = 逐位**：默认实现是"初值取该轴首元素，之后逐元素 `f32::max/min` fold"；
    // 本仓内核**逐字复制**该 fold（`ops::dsl_map`），并行只切**与归约轴无关的另一维** ⇒ 每输出的
    // 元素次序不变 ⇒ `to_bits()` 相等（`tests/prims_override.rs` 钉住，含 NaN/±0/全 NaN 轴）。
    // **不向量化**：`f32::max` 对"全 NaN 轴"返回的 NaN payload 依赖比较次序 ⇒ 换结合次序会破逐位。
    // **错误面**：`axis_extreme_check` 与默认实现逐条同口径（溢出/长度/归约轴为 0）。

    /// 轴 `max`（沿 `rows`）；P4.3 override。
    fn reduce_max_cols_f32(
        &self,
        x: &[f32],
        rows: usize,
        cols: usize,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let op = "reduce_max_cols_f32";
        axis_extreme_check(op, x, rows, cols, out, cols, rows)?;
        debug_assert!(
            !std::ptr::eq(x.as_ptr(), out.as_ptr()),
            "{op}: out 不得与 x 重叠（契约）"
        );
        crate::ops::dsl_map::reduce_max_cols(x, rows, cols, out);
        Ok(())
    }

    /// 轴 `min`（沿 `rows`）；P4.3 override。
    fn reduce_min_cols_f32(
        &self,
        x: &[f32],
        rows: usize,
        cols: usize,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let op = "reduce_min_cols_f32";
        axis_extreme_check(op, x, rows, cols, out, cols, rows)?;
        debug_assert!(
            !std::ptr::eq(x.as_ptr(), out.as_ptr()),
            "{op}: out 不得与 x 重叠（契约）"
        );
        crate::ops::dsl_map::reduce_min_cols(x, rows, cols, out);
        Ok(())
    }

    /// 轴 `max`（沿 `cols`）；P4.3 override。
    fn reduce_max_rows_f32(
        &self,
        x: &[f32],
        rows: usize,
        cols: usize,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let op = "reduce_max_rows_f32";
        axis_extreme_check(op, x, rows, cols, out, rows, cols)?;
        debug_assert!(
            !std::ptr::eq(x.as_ptr(), out.as_ptr()),
            "{op}: out 不得与 x 重叠（契约）"
        );
        crate::ops::dsl_map::reduce_max_rows(x, rows, cols, out);
        Ok(())
    }

    /// 轴 `min`（沿 `cols`）；P4.3 override。
    fn reduce_min_rows_f32(
        &self,
        x: &[f32],
        rows: usize,
        cols: usize,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let op = "reduce_min_rows_f32";
        axis_extreme_check(op, x, rows, cols, out, rows, cols)?;
        debug_assert!(
            !std::ptr::eq(x.as_ptr(), out.as_ptr()),
            "{op}: out 不得与 x 重叠（契约）"
        );
        crate::ops::dsl_map::reduce_min_rows(x, rows, cols, out);
        Ok(())
    }

    /// **整片 `max`**（P4.3 override）：**不并行**——整片归约没有"独立轴"可切，树形归约会让
    /// "全 NaN"时的 NaN payload / ±0 位模式随次序变化 ⇒ 保**逐位**就保持顺序 fold（与默认同一代码）。
    /// 空输入 = 幺元 `NEG_INFINITY`（与默认一致，永远 `Ok`）。
    fn reduce_max_f32(&self, x: &[f32]) -> Result<f32, BackendError> {
        Ok(crate::ops::dsl_map::max_global(x))
    }

    /// **整片 `min`**（P4.3 override）：语义/不并行理由同 [`Backend::reduce_max_f32`]；
    /// 空输入 = 幺元 `POSITIVE_INFINITY`。
    fn reduce_min_f32(&self, x: &[f32]) -> Result<f32, BackendError> {
        Ok(crate::ops::dsl_map::min_global(x))
    }

    /// **行 absmax override（P4.4）**：`out[r] = max_c |x[r·cols + c]|`。
    ///
    /// **映射到 1 趟融合核** `ops::quant_i8::absmax_rows`（每行一次 `amax`；与 `api::absmax_rows`
    /// 同一个内核，`api` 只是多分配一次输出）⇒ **零拷贝写 `out`**、无中间缓冲。
    ///
    /// **等价类 = 逐位**（与契约默认实现 / 旧 3 趟组合都逐位）：内核语义 = 先 `abs` 再
    /// `f32::max` fold（初值 `NEG_INFINITY`；全 NaN 行归一成 `0.0`；`±0 ⇒ +0`；`±inf` 照常），
    /// 与默认实现同序 ⇒ `to_bits()` 相等（`dsl_backend` 测试对拍钉住，含 NaN/±0/±inf/次正规）。
    ///
    /// **错误面与默认同口径**：`rows×cols` 溢出 ⇒ `ShapeOverflow`；`x`/`out` 长度不符 ⇒
    /// `LengthMismatch`（`out.len()` 必须 == `rows`）；空形状 ⇒ `out` 全 0（合法空操作）。
    fn reduce_absmax_rows_f32(
        &self,
        x: &[f32],
        rows: usize,
        cols: usize,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let op = "reduce_absmax_rows_f32";
        let total = rows.checked_mul(cols).ok_or(BackendError::ShapeOverflow {
            op,
            what: "rows×cols",
        })?;
        if x.len() != total {
            return Err(len_err(op, "x", total, x.len()));
        }
        if out.len() != rows {
            return Err(len_err(op, "out", rows, out.len()));
        }
        debug_assert!(
            !std::ptr::eq(x.as_ptr(), out.as_ptr()),
            "{op}: out 不得与 x 重叠（契约）"
        );
        if rows == 0 || cols == 0 {
            out.fill(0.0);
            return Ok(());
        }
        crate::ops::quant_i8::absmax_rows(x, rows, cols, out);
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
                // 的 `docs/measurements/longson-batches.md` §14.6）⇒ 这里**转调同一份判据/本体**，否则两档不同式，
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
    /// 旧版本把这两条也写成逐位，是 审计复核记录 记的错误。
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
    ///   测试 `matmul_f64_and_dot_f64_bitwise_vs_api`）；
    /// * **`F64Elementwise`（P1-4 新增）**：`map_binary_f64`/`map_affine_f64`
    ///   已 override 到 `ops::f64_map`（LASX 4 f64/lane；与 `kernels::scalar` **逐位**，
    ///   测试 `f64_map_is_bitwise_vs_scalar`）。
    ///
    // ============ P0-2：单趟 map + 行和（`map_rows_sum_f32`）============
    //
    // 契约语义：`out_mapped = f(x, shift[r])`、`out_rowsum[r] = Σ_c out_mapped`（**一次扫描**）。
    // **等价类：容差**（契约文档自己也这么写）：映射值用 LASX `exp_vec`（与 `softmax_rows` 同串），
    // 与 portable 的 libm `exp` 非逐位；行和是 per-lane 累加 + 固定次序水平求和（portable 是单累加器）。
    // 实测界与 A/B 见 `examples/map_rowsum_ab.rs`：portable/LASX = 4.745×/3.357×/3.087×/2.557×/2.293×
    // （5 组全部 ≥1.10 门槛，loadavg≈7.5）⇒ **保留此 override**。
    fn map_rows_sum_f32(
        &self,
        x: &[f32],
        rows: usize,
        cols: usize,
        shift: &[f32],
        map: ReduceMapOp,
        out_mapped: &mut [f32],
        out_rowsum: &mut [f32],
    ) -> Result<(), BackendError> {
        let op = "map_rows_sum_f32";
        let total = rows.checked_mul(cols).ok_or(BackendError::ShapeOverflow {
            op,
            what: "rows×cols",
        })?;
        if x.len() != total {
            return Err(len_err(op, "x", total, x.len()));
        }
        if out_mapped.len() != total {
            return Err(len_err(op, "out_mapped", total, out_mapped.len()));
        }
        if out_rowsum.len() != rows {
            return Err(len_err(op, "out_rowsum", rows, out_rowsum.len()));
        }
        match map {
            ReduceMapOp::Exp => {
                if !shift.is_empty() {
                    return Err(len_err(op, "shift（Exp 不接受位移）", 0, shift.len()));
                }
            }
            ReduceMapOp::ExpShiftedRow => {
                if shift.len() != rows {
                    return Err(len_err(op, "shift", rows, shift.len()));
                }
            }
        }
        debug_assert!(
            !std::ptr::eq(x.as_ptr(), out_mapped.as_ptr()),
            "{op}: out_mapped 不得与 x 重叠（契约）"
        );
        debug_assert!(
            !std::ptr::eq(x.as_ptr(), out_rowsum.as_ptr()),
            "{op}: out_rowsum 不得与 x 重叠（契约）"
        );
        crate::ops::map_rows_sum::map_rows_sum_f32(
            x, rows, cols, shift, map, out_mapped, out_rowsum,
        );
        Ok(())
    }

    /// 其余（归约/超越/多线程/复数组合步）**继续 `false`**（"只报确实为真"；
    /// 归约/逐元素虽有 override，但能力位口径是"该族有专用实现"，本批不扩大声明面）。
    ///
    /// **`Capability::Quant` 自 P1 批3 起不报**：4 个具名方法已被 `FusedPlan` 取代（批3 删除），
    /// 该位语义（"4 方法全 override"）不再成立；量化能力由 `supports_plan` 的 4 个 kind 承载
    /// （本后端 `QuantizeI8Rows`/`DotI8`/`GemvI8` = true；`DequantizeI8Rows` = false，见 Plan 注释）。
    fn supports(&self, cap: Capability) -> bool {
        // 阶段4：`FusedNorms` / `FusedElementwise` **只报真**——对应方法全部 override 到 LASX 内核才报。
        // **`FusedGeometry`（批2）/ `Quant`（批3）不再报**：对应具名方法已被 `FusedPlan` 取代；
        // 融合能力由 `supports_plan` 的 kind 承载（本后端几何 5 kind 全真、量化 3/4 真）。
        //
        // **测试钩子（env `LASX_NO_PLAN=1`）**：只把 `FusedNorms`/`FusedElementwise` 报 false，
        // 供基准探针在同一后端上测**生产组合路径**；默认构建/运行行为不变（见 `no_plan_hook`）。
        if no_plan_hook() && matches!(cap, Capability::FusedNorms | Capability::FusedElementwise) {
            return false;
        }
        matches!(
            cap,
            Capability::Packing
                | Capability::MatmulF64
                | Capability::F64Elementwise
                | Capability::FusedNorms
                | Capability::FusedElementwise
                // **Threaded（引擎×后端双重并行修复）**：LASX 的 matmul 在 work ≥ `PAR_MIN_WORK` 时
                // 自己经 `pool::global()` 并行 ⇒ 引擎 `run_matmul_f32` 见到本能力位后**不再按行切块**，
                // 避免"引擎行块 × 后端池"嵌套（实测 1024³：嵌套 21.4ms vs 只后端 6.3ms，3.4× 差）。
                | Capability::Threaded
        )
    }

    // ============ 阶段4 收尾：f64 几何五入口（`FusedGeometry`）→ **P1 批2 起迁入 `exec_fused_plan`** ============
    //
    // b89a32f 删除了 `vec3_add_scaled_f64` / `norm3_batch_f64` / `unitize3_batch_f64` /
    // `cross3_batch_f64` / `mat3_mul_vec3_batch_f64` 五个具名方法（`Backend` 71→66）；
    // 它们的 PLAN overrides 全部走 `api::*_into`（LASX 向量核，免分配）：
    // `FusedPlan::{Vec3AddScaledF64, Norm3BatchF64, Unitize3BatchF64, Cross3BatchF64, Mat3MulVec3BatchF64}`。
    // **零向量语义差异（路径各自语义）**：LASX `unitize3` 对零向量给 `0`；契约默认（组合版）给 `NaN` ——
    // PLAN override **不额外打补丁**、保持内核语义；调用方要 `NaN` 档就用 portable 计划解释。
    // ⇒ `Capability::FusedGeometry`（语义 = "5 个具名方法全 override"）在本后端**不再报 true**
    //   （方法已不存在；能力改由 `supports_plan` 的 5 个 `PatternKind` 承载）。

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

    fn axpy_f32(&self, y: &mut [f32], x: &[f32], alpha: f32) -> Result<(), BackendError> {
        debug_assert_ne!(x.as_ptr(), y.as_ptr(), "axpy_f32：x/y 不得重叠");
        crate::api::axpy(alpha, x, y).map_err(|_| BackendError::Unsupported {
            op: "axpy_f32",
            cap: Capability::FusedElementwise,
        })
    }

    /// **组合原语 override（B1/task-106 M5）**：`y := alpha·x + y`（就地，LASX 1 趟向量）。
    ///
    /// 这是 `axpy!` 宏（B2）与组合回退的落点：默认实现走 `kernels::scalar::axpy_inplace_f32`
    /// （逐元素标量）；本 override 映射到 `api::axpy`（LASX 两舍入向量核）。
    /// **等价类：逐位**（同一 `t = alpha*x + 0.0; y += t` 两次舍入序列，含 ±0 语义；
    /// `tests/prims_override.rs` 用 `to_bits` 对拍；W3 已把该核从 FMA 改成两次舍入）。
    fn axpy_inplace_f32(&self, y: &mut [f32], x: &[f32], alpha: f32) -> Result<(), BackendError> {
        debug_assert_ne!(x.as_ptr(), y.as_ptr(), "axpy_inplace_f32：x/y 不得重叠");
        crate::api::axpy(alpha, x, y).map_err(|_| BackendError::Unsupported {
            op: "axpy_inplace_f32",
            cap: Capability::FusedElementwise,
        })
    }

    /// **组合原语 override（B1b/task-108 M5 续）**：逐元素一元——`Exp` 走 LASX 8-lane 多项式、
    /// `Sqrt` 走 `xvfsqrt_s`（IEEE ⇒ 逐位），**其余变体整片委托
    /// `kernels::scalar::map_unary_f32`**（逐位不变）。
    ///
    /// **等价类**：`Sqrt` 逐位；`Exp` 容差类（实测 rel 界见 `tests/prims_override.rs` 与
    /// `docs/dev.md §7.52`）；其余（`Sin/Cos/Ln/Erf/Sigmoid/Rcp/Abs`）委托 ⇒ 逐位。
    /// **意义**：`fuse!`/宏组合路径里的 `map_unary!(Exp)`（softmax 回退、`exp_log_fold` 等）
    /// 不再回落 portable 标量 libm。
    fn map_unary_f32(&self, x: &[f32], out: &mut [f32], op: UnaryOp) -> Result<(), BackendError> {
        crate::ops::prims::map_unary(x, out, op)
    }

    /// **组合原语 override（B1b/task-108 M5 续）**：窄三元组融合 `RowFuse`
    /// （`fuse!` 宏的落点，rms/layer/softmax 的生产组合路径）。
    ///
    /// **支持**：`RowReduce::{Sum,SumSq}` × `RowEpilogue`（6 变体）× `BinaryOp::{Add,Sub,Mul,Div}`；
    /// 行内 8-lane 累加与 `kernels::scalar::reduce_rows_map_f32` **同序** ⇒ **逐位**。
    /// **未验证组合**（`Max` 归约、`Max/Min` map）**整片委托标量参照**（`NaN`/`±0` 语义优先，
    /// 不为优化让路）；契约错误码与 portable 同口径（`LengthMismatch`/`ShapeOverflow`）。
    fn reduce_rows_map_f32(
        &self,
        x: &[f32],
        rows: usize,
        cols: usize,
        out: &mut [f32],
        plan: RowFuse,
    ) -> Result<(), BackendError> {
        crate::ops::prims::reduce_rows_map(x, rows, cols, out, plan)
    }

    // ============ P3 Plan：`supports_plan` / `exec_fused_plan`（task-63 + P1 批1 task-66）============
    //
    // **语义映射（契约 `9af59f7` 定死，不是猜；先读 `simd_dsl` 的 `backend.rs` + `tests/fused_plan.rs`）**：
    // * `Axpy { alpha }`：`y := alpha·x + y`（就地；`io.x` 只读、`io.y` 输出；`y.len()==x.len()`）；
    // * `SoftmaxRows { rows, cols }`：行 softmax（`io.x`/`io.out`，`len==rows·cols`）；
    // * `RmsNormRows { rows, cols, eps }`：`out = x/√(mean(x²)+eps)`（`io.x`/`io.out`）；
    // * `LayerNormRows { rows, cols, eps }`：`out = (x−mean)/√(var+eps)`（`io.x`/`io.out`）；
    // * `GeluQuick`：GELU tanh 近似（等长 `io.x`/`io.out`）——**W3 起不接**（见下）；
    // * `RopeApply { rows, cols, n_dims }`：半旋转（`io.x`/`io.cos`/`io.sin`/`io.out`；
    //   `cos`/`sin` 长度 = `rows·(n_dims/2)`、`n_dims` 偶数且 ≤ cols）。
    // **本后端映射**（走 LASX 融合核）：Axpy → `api::axpy`；SoftmaxRows → `api::softmax_rows_into`；
    // RmsNormRows → `api::rms_norm_into`；LayerNormRows → `api::layer_norm_into`；
    // RopeApply → `api::rope_into(RopeMode::NeoX)`。
    // **`GeluQuick` 融合核已删（W3）**：实测 LASX 融合核比纯组合**慢 29%**（composed/fused =
    // 0.763–0.773×）⇒ `supports_plan` 报 false、`exec_fused_plan` 回退 portable 解释；
    // `FusedPlan::GeluQuick` 变体保留（契约面不删，死亡方式：simd 侧后续单独 breaking 建议）。
    // **长度/形状校验与契约默认解释器同口径**（错 ⇒ `LengthMismatch`/`ShapeOverflow`，空 ⇒ `Ok`）。
    // 数值：Plan 路 ≡ 直接 `api::*` **逐位**（同一内核）；与 portable 计划解释比 = 既有容差类
    // （softmax 2.79e-9、rms 1.83e-7、layer 2.08e-7、rope 6.92e-8）；**axpy 自 W3 起逐位**
    // （两次舍入核修正，原 FMA 单次舍入差 1 ulp）。

    fn supports_plan(&self, kind: PatternKind) -> bool {
        // **f32 的 5 个（Axpy/Softmax/Rms/Layer/Rope）+ 批2 的 5 个 f64 几何
        // + 批3 的 quantize/dot/gemv：全部 true**；
        // **`GeluQuick`/`DequantizeI8Rows`/`Norm2BatchF64` 明确 false**（不虚报）：
        // GeluQuick = 融合核实测更慢（W3 删除）；Dequantize = 无高效连续 i8 块拓宽核；
        // Norm2BatchF64 = 无 f64 norm2 融合核。三者 `exec_fused_plan` 对应臂都回退 portable 解释。
        // **测试钩子**：`LASX_NO_PLAN` 已设置 ⇒ 一律 false（基准探针测生产组合路径用）。
        if no_plan_hook() {
            return false;
        }
        match kind {
            PatternKind::Axpy
            | PatternKind::SoftmaxRows
            | PatternKind::RmsNormRows
            | PatternKind::LayerNormRows
            | PatternKind::RopeApply
            | PatternKind::Vec3AddScaledF64
            | PatternKind::Norm3BatchF64
            | PatternKind::Unitize3BatchF64
            | PatternKind::Cross3BatchF64
            | PatternKind::Mat3MulVec3BatchF64
            | PatternKind::QuantizeI8Rows
            | PatternKind::DotI8
            | PatternKind::GemvI8
            | PatternKind::GeluQuick => true,
            PatternKind::DequantizeI8Rows | PatternKind::Norm2BatchF64 => false,
        }
    }

    fn exec_fused_plan(&self, plan: FusedPlan<'_>, io: FusedIo<'_>) -> Result<(), BackendError> {
        // **共享校验（task-80）**：长度/形状/溢出/空/RoPE n_dims/几何数组/量化切片，
        // 全部由 `FusedPlan::validate` 一处实现（与 portable 解释器**同口径、同错误码**）；
        // 各臂不再重复校验，只保留"内核不能吃空输入"的执行守卫（不是校验、不产生错误码）。
        plan.validate(&io)?;
        match plan {
            FusedPlan::Axpy { alpha } => {
                // Plan 语义：y := alpha·x + y（就地）；`out` 不参与（契约为空）。
                if io.x.is_empty() {
                    return Ok(());
                }
                debug_assert_ne!(
                    io.x.as_ptr(),
                    io.y.as_ptr(),
                    "exec_fused_plan(Axpy)：x/y 不得重叠"
                );
                crate::api::axpy(alpha, io.x, io.y).map_err(|_| BackendError::Unsupported {
                    op: "exec_fused_plan(Axpy)",
                    cap: Capability::FusedElementwise,
                })
            }
            // **GeluQuick（task-130）：走 1 趟成熟核 `api::gelu_quick_into`**。
            //
            // **与 W3 删掉的旧融合核的区别**：旧核是"为融合另写的一份 tanh-GELU 实现"，
            // 实测比纯组合**慢 29%** ⇒ 被删；现在**不新增任何核**，只把 `FusedPlan` 的
            // plan 通道接到**已经在比赛里证明快 10.7×** 的 `api::gelu_quick_into`
            // （`src/api.rs:719`：一次读 x、一次写 out 的 1 趟实现）。
            // ⇒ 这不是"复活旧核"，是"让 plan 通道用上成熟核"。
            FusedPlan::SoftmaxRows { rows, cols } => {
                let op = "exec_fused_plan(SoftmaxRows)";
                if io.x.is_empty() {
                    return Ok(());
                }
                crate::api::softmax_rows_into(io.x, None, rows, cols, 1.0, io.out).map_err(|_| {
                    BackendError::Unsupported {
                        op,
                        cap: Capability::FusedNorms,
                    }
                })
            }
            FusedPlan::RmsNormRows { rows, cols, eps } => {
                let op = "exec_fused_plan(RmsNormRows)";
                if io.x.is_empty() {
                    return Ok(());
                }
                crate::api::rms_norm_into(io.x, None, rows, cols, eps, io.out).map_err(|_| {
                    BackendError::Unsupported {
                        op,
                        cap: Capability::FusedNorms,
                    }
                })
            }
            FusedPlan::LayerNormRows { rows, cols, eps } => {
                let op = "exec_fused_plan(LayerNormRows)";
                if io.x.is_empty() {
                    return Ok(());
                }
                crate::api::layer_norm_into(io.x, None, None, rows, cols, eps, io.out).map_err(
                    |_| BackendError::Unsupported {
                        op,
                        cap: Capability::FusedNorms,
                    },
                )
            }
            // **task-130：GeluQuick 改走成熟 1 趟核**（与 W3 删掉的旧融合核无关——旧核是
            // "为融合另写一份 tanh-GELU"，实测比组合慢 29% 才被删；现在**不新增核**，只把
            // plan 通道接到比赛里快 10.7× 的 `api::gelu_quick_into`（`src/api.rs:719`））。
            // `supports_plan(GeluQuick)` 已改 true ⇒ 调用方会走这条；`LASX_NO_PLAN=1` 时
            // `supports_plan` 一律 false，仍可强制退回组合路径。
            FusedPlan::GeluQuick => {
                if io.x.is_empty() {
                    return Ok(());
                }
                crate::api::gelu_quick_into(io.x, io.out).map_err(|_| BackendError::Unsupported {
                    op: "exec_fused_plan(GeluQuick)",
                    cap: Capability::FusedElementwise,
                })
            }
            FusedPlan::RopeApply { rows, cols, n_dims } => {
                let op = "exec_fused_plan(RopeApply)";
                if io.x.is_empty() {
                    return Ok(());
                }
                crate::api::rope_into(
                    io.x,
                    io.cos,
                    io.sin,
                    rows,
                    cols,
                    n_dims,
                    crate::api::RopeMode::NeoX,
                    io.out,
                )
                .map_err(|_| BackendError::Unsupported {
                    op,
                    cap: Capability::FusedElementwise,
                })
            }
            // ---- P1 批2（task-69）：f64 几何 5 pattern。缓冲在 plan 值里（`FusedPlan<'a>`），
            //      `io` 不参与（契约：几何 pattern 传空 `FusedIo`）。校验与契约 `fused_geo_len` 同口径：
            //      n = 第一个输入的长度，其余（含所有输出）必须等长；n==0 ⇒ Ok。
            //      映射：全部 → `api::*_into`（LASX 向量核，免分配）。
            //      **零向量**：Unitize3 走 LASX 内核 ⇒ `0`（契约默认解释 = 组合版 `0/0 = NaN`，
            //      "路径各自语义"由调用方按 `supports_plan` 选择）。
            FusedPlan::Vec3AddScaledF64 { s, a, b, o } => {
                let op = "exec_fused_plan(Vec3AddScaledF64)";
                let [ax, ay, az] = a;
                let [bx, by, bz] = b;
                let [ox, oy, oz] = o;
                if ax.is_empty() {
                    return Ok(());
                }
                crate::api::vec3_add_scaled_batch_into(ax, ay, az, bx, by, bz, s, ox, oy, oz)
                    .map_err(|_| BackendError::Unsupported {
                        op,
                        cap: Capability::FusedGeometry,
                    })
            }
            FusedPlan::Norm3BatchF64 { xyz, out } => {
                let op = "exec_fused_plan(Norm3BatchF64)";
                let [x, y, z] = xyz;
                if x.is_empty() {
                    return Ok(());
                }
                crate::api::norm3_batch_into(x, y, z, out).map_err(|_| BackendError::Unsupported {
                    op,
                    cap: Capability::FusedGeometry,
                })
            }
            FusedPlan::Unitize3BatchF64 { xyz, o } => {
                let op = "exec_fused_plan(Unitize3BatchF64)";
                let [x, y, z] = xyz;
                let [ox, oy, oz] = o;
                if x.is_empty() {
                    return Ok(());
                }
                crate::api::unitize3_batch_into(x, y, z, ox, oy, oz).map_err(|_| {
                    BackendError::Unsupported {
                        op,
                        cap: Capability::FusedGeometry,
                    }
                })
            }
            FusedPlan::Cross3BatchF64 { a, b, o } => {
                let op = "exec_fused_plan(Cross3BatchF64)";
                let [ax, ay, az] = a;
                let [bx, by, bz] = b;
                let [ox, oy, oz] = o;
                if ax.is_empty() {
                    return Ok(());
                }
                crate::api::cross3_batch_into(ax, ay, az, bx, by, bz, ox, oy, oz).map_err(|_| {
                    BackendError::Unsupported {
                        op,
                        cap: Capability::FusedGeometry,
                    }
                })
            }
            FusedPlan::Mat3MulVec3BatchF64 { m, v, o } => {
                let op = "exec_fused_plan(Mat3MulVec3BatchF64)";
                let [x, y, z] = v;
                let [ox, oy, oz] = o;
                if x.is_empty() {
                    return Ok(());
                }
                // **mat3 走 Rust 层 `_into` helper**（`api::mat3_mul_vec3_batch_into`，免分配；
                // 与 lasx 侧 `api::mat3_mul_vec3_batch` 数值逐位）。
                crate::api::mat3_mul_vec3_batch_into(m, x, y, z, ox, oy, oz).map_err(|_| {
                    BackendError::Unsupported {
                        op,
                        cap: Capability::FusedGeometry,
                    }
                })
            }
            // ---- P1 批3（task-70）：i8 量化族 4 pattern。缓冲（i8/i32）在 plan 值里，`io` 不参与。
            //      语义红线：quantize = ties-even + 先舍入后 clamp（NaN⇒0、±inf⇒±端点）；
            //      dot/gemv = **i64 精确累加 + 末端一次饱和**（不是中途 saturating_add）。
            FusedPlan::QuantizeI8Rows {
                rows,
                cols,
                x,
                scale,
                out,
            } => {
                if x.is_empty() {
                    return Ok(());
                }
                // SAFETY: `validate` 已校验三个长度；`x: &[f32]` 与 `out: &mut [i8]` 类型不同 ⇒
                // 安全 Rust 下不可能重叠（别名契约由类型系统 + 借用检查器守）。
                unsafe {
                    crate::ops::quant_i8::quantize_i8_rows_contract(x, rows, cols, scale, out)
                };
                Ok(())
            }
            FusedPlan::DequantizeI8Rows {
                rows,
                cols,
                x,
                scale,
                out,
            } => {
                // **未接 LASX 核 ⇒ 回退 portable 解释**（与契约默认同序列、逐位；不虚报能力）。
                simd_dsl::kernels::scalar::dequantize_i8_rows_f32(x, rows, cols, scale, out)
            }
            FusedPlan::DotI8 { a, b, out } => {
                *out = crate::ops::dot_i8::dot_i8_saturating(a, b);
                Ok(())
            }
            FusedPlan::GemvI8 { j, k, a, x, out } => {
                crate::ops::gemv_i8::gemv_i8_i32(j, k, a, x, out);
                Ok(())
            }
            // **`Norm2BatchF64`（simd 批4 新增 kind）**：LASX 没有 f64 norm2 融合核 ⇒
            // `supports_plan` 报 false、这里**回退契约默认解释的组合序列**（与 portable 解释器
            // 同表达式：`(x²+y²).sqrt()`，f64 精确运算 ⇒ 逐位一致）。
            FusedPlan::Norm2BatchF64 { xy, out } => {
                let [x, y] = xy;
                if x.is_empty() {
                    return Ok(());
                }
                for i in 0..x.len() {
                    out[i] = (x[i] * x[i] + y[i] * y[i]).sqrt();
                }
                Ok(())
            }
        }
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
/// **为什么要有它**：要给出"接归约/逐元素族带来的收益"的**同进程**硬数（`docs/dev.md` §6.4 口径），
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
        // **只低估**：`Packing`（packed 已接线且逐位断言过）、`MatmulF64`（IS-2 已接）与
        // `F64Elementwise`（P1-4 已接）为真；
        // 其余能力一律 false（`Threaded` 仍 false：`CallOpts` 未接，见 ③b-4 留档）
        assert!(
            be.supports(Capability::Packing),
            "Packing 已实现且逐位断言过 ⇒ 应为 true"
        );
        assert!(
            be.supports(Capability::MatmulF64),
            "IS-2 后 matmul_f64/dot_f64 有专用实现 ⇒ 应为 true"
        );
        assert!(
            be.supports(Capability::F64Elementwise),
            "P1-4 后 map_*_f64 有 LASX 向量内核且逐位断言过 ⇒ 应为 true"
        );
        assert!(!be.supports(Capability::Transcendental));
        // `Threaded` = "后端会在内部并行"：LASX matmul 在 work ≥ PAR_MIN_WORK 时走全局池 ⇒ 报真，
        // 让引擎不再外层按行切块（避免双重并行）。见 `supports` 的注释。
        assert!(be.supports(Capability::Threaded));
    }

    /// **P1-4：f64 逐元素 LASX 覆盖 vs `kernels::scalar` 参照**——多形状（含非 4 倍数/尾部/空）
    /// × 全部 op ⇒ **逐位**（`to_bits()`）；`Max`/`Min` 走标量委托（同样逐位）与形状错误路径。
    #[test]
    fn f64_map_is_bitwise_vs_scalar() {
        use simd_dsl::kernels::scalar;
        let be = LasxBackend;
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        for &n in &[0usize, 1, 3, 4, 5, 15, 16, 17, 33, 1029] {
            let x: Vec<f64> = (0..n)
                .map(|i| ((i * 37 % 101) as f64) * 0.1 - 5.05)
                .collect();
            let y: Vec<f64> = (0..n).map(|i| ((i * 53 % 97) as f64) * 0.1 + 1.3).collect();
            for op in [
                BinaryOp::Add,
                BinaryOp::Sub,
                BinaryOp::Mul,
                BinaryOp::Div,
                BinaryOp::Max,
                BinaryOp::Min,
            ] {
                let mut got = vec![f64::NAN; n];
                be.map_binary_f64(&x, &y, &mut got, op).expect("backend");
                let mut want = vec![f64::NAN; n];
                scalar::map_binary_f64(&x, &y, &mut want, op).expect("scalar");
                assert_eq!(
                    bits(&got),
                    bits(&want),
                    "map_binary_f64 n={n} op={op:?} 不逐位"
                );
            }
            let mut got = vec![f64::NAN; n];
            be.map_affine_f64(&x, 0.3, -1.7, &mut got).expect("backend");
            let mut want = vec![f64::NAN; n];
            scalar::map_affine_f64(&x, 0.3, -1.7, &mut want).expect("scalar");
            assert_eq!(bits(&got), bits(&want), "map_affine_f64 n={n} 不逐位");
        }
        // 形状错误 ⇒ LengthMismatch（不 panic）
        let v = vec![1.0f64; 6];
        let mut o = vec![0.0f64; 6];
        assert!(
            be.map_binary_f64(&v, &v[..5], &mut o, BinaryOp::Add)
                .is_err()
        );
        assert!(
            be.map_binary_f64(&v, &v, &mut o[..5], BinaryOp::Add)
                .is_err()
        );
        assert!(be.map_affine_f64(&v, 1.0, 0.0, &mut o[..5]).is_err());
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

    /// **P3 Plan（task-63）→ 批1（task-66）→ 批2（task-69）→ W3（task-104）**：**13 个 kind**
    /// 的语义映射与数值——Plan 路 ≡ 直接 `api::*` **逐位**（同一内核）；vs portable 计划解释 =
    /// 既有容差类/几何零向量"路径各自语义"（LASX `unitize3` = 0、契约默认 = NaN）；
    /// portable 后端 `supports_plan` 全 false；`GeluQuick`/`Dequantize`/`Norm2` 如实 false
    /// 且回退 portable 解释；长度/形状错 ⇒ `Err`（不 panic）。
    #[test]
    fn plan_overrides_map_to_fused_kernels_and_stay_bitwise() {
        use simd_dsl::{FusedIo, FusedPlan, PatternKind, ScalarBackend};
        let be = LasxBackend;
        let kinds = [
            PatternKind::Axpy,
            PatternKind::SoftmaxRows,
            PatternKind::RmsNormRows,
            PatternKind::LayerNormRows,
            PatternKind::RopeApply,
            PatternKind::Vec3AddScaledF64,
            PatternKind::Norm3BatchF64,
            PatternKind::Unitize3BatchF64,
            PatternKind::Cross3BatchF64,
            PatternKind::Mat3MulVec3BatchF64,
            // 批3：quantize/dot/gemv 已接；**DequantizeI8Rows 明确不接**（见下）
            PatternKind::QuantizeI8Rows,
            PatternKind::DotI8,
            PatternKind::GemvI8,
        ];
        // ① 能力位：按 PatternKind 粒度、只低估——本后端 13 个接；portable 全 false。
        for k in kinds {
            assert!(be.supports_plan(k), "LASX 应报 {k:?} = true");
            assert!(!ScalarBackend.supports_plan(k), "portable 不应报 {k:?}");
        }
        // **task-130：GeluQuick 改接成熟 1 趟核 `api::gelu_quick_into` ⇒ 如实 true。**
        // （W3 删的是"为融合另写的一份"慢核；此处的核是既有的、比赛里快 10.7× 的实现。）
        assert!(be.supports_plan(PatternKind::GeluQuick));
        // 等价性是**容差类**（tanh 近似实现次序不同 ⇒ 非逐位）；界见 `api::gelu_quick_into`
        // 注释与比赛量测（量级 7.6e-3）。
        let mut gelu_out = [0.0f32; 3];
        let gelu_x = [0.5f32, -1.0, 2.0];
        be.exec_fused_plan(FusedPlan::GeluQuick, io(&gelu_x, &mut [], &mut gelu_out))
            .expect("GeluQuick 计划应执行成功");
        // **DequantizeI8Rows 如实不报**（无 LASX 核；exec 走 portable 回退，见下）
        assert!(!be.supports_plan(PatternKind::DequantizeI8Rows));
        /// 空 `FusedIo` 构造（`cos`/`sin` 只 RopeApply 用）。
        fn io<'a>(x: &'a [f32], y: &'a mut [f32], out: &'a mut [f32]) -> FusedIo<'a> {
            FusedIo {
                x,
                y,
                out,
                cos: &[],
                sin: &[],
            }
        }
        let max_d = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
        };

        // ② Axpy：Plan 路 ≡ 命名方法 ≡ 直接 api（逐位）
        let n = 257usize;
        let x: Vec<f32> = (0..n).map(|i| (i % 37) as f32 * 0.125 - 2.0).collect();
        let base: Vec<f32> = (0..n).map(|i| (i % 53) as f32 * 0.0625 + 1.0).collect();
        let alpha = 0.375f32;
        let mut y_plan = base.clone();
        be.exec_fused_plan(FusedPlan::Axpy { alpha }, io(&x, &mut y_plan, &mut []))
            .unwrap();
        let mut y_named = base.clone();
        be.axpy_f32(&mut y_named, &x, alpha).unwrap();
        let mut y_api = base.clone();
        crate::api::axpy(alpha, &x, &mut y_api).unwrap();
        for (a, b) in y_plan.iter().zip(&y_named) {
            assert_eq!(a.to_bits(), b.to_bits(), "Axpy：Plan 路与命名方法不逐位");
        }
        for (a, b) in y_plan.iter().zip(&y_api) {
            assert_eq!(a.to_bits(), b.to_bits(), "Axpy：Plan 路与直接 api 不逐位");
        }
        let mut y_port = base.clone();
        ScalarBackend
            .exec_fused_plan(FusedPlan::Axpy { alpha }, io(&x, &mut y_port, &mut []))
            .unwrap();
        eprintln!(
            "Plan Axpy：LASX vs portable 计划解释 max|d| = {:.3e}",
            max_d(&y_plan, &y_port)
        );

        // ③ 行 pattern：Softmax / RmsNorm / LayerNorm —— Plan 路 ≡ 直接 api（逐位）
        let (rows, cols) = (16usize, 129usize);
        let m = rows * cols;
        let xs: Vec<f32> = (0..m).map(|i| (i % 71) as f32 * 0.05 - 1.8).collect();
        let eps = 1e-5f32;
        for name in ["SoftmaxRows", "RmsNormRows", "LayerNormRows"] {
            // `FusedPlan` 批2 起**不是 `Copy`**（几何变体自带 f64 缓冲）⇒ 每次用现构造。
            let mk = || match name {
                "SoftmaxRows" => FusedPlan::SoftmaxRows { rows, cols },
                "RmsNormRows" => FusedPlan::RmsNormRows { rows, cols, eps },
                _ => FusedPlan::LayerNormRows { rows, cols, eps },
            };
            let mut out_plan = vec![0.0f32; m];
            be.exec_fused_plan(mk(), io(&xs, &mut [], &mut out_plan))
                .unwrap();
            let mut out_api = vec![0.0f32; m];
            match name {
                "SoftmaxRows" => {
                    crate::api::softmax_rows_into(&xs, None, rows, cols, 1.0, &mut out_api).unwrap()
                }
                "RmsNormRows" => {
                    crate::api::rms_norm_into(&xs, None, rows, cols, eps, &mut out_api).unwrap()
                }
                _ => crate::api::layer_norm_into(&xs, None, None, rows, cols, eps, &mut out_api)
                    .unwrap(),
            }
            for (a, b) in out_plan.iter().zip(&out_api) {
                assert_eq!(a.to_bits(), b.to_bits(), "{name}：Plan 路与直接 api 不逐位");
            }
            let mut out_port = vec![0.0f32; m];
            ScalarBackend
                .exec_fused_plan(mk(), io(&xs, &mut [], &mut out_port))
                .unwrap();
            let d = max_d(&out_plan, &out_port);
            eprintln!("Plan {name}：LASX vs portable 计划解释 max|d| = {d:.3e}");
            assert!(d <= 1e-5, "{name} 计划路与 portable 超界 {d:e}");
        }

        // ④ GeluQuick（**task-130：改接成熟 1 趟核 `api::gelu_quick_into`**）：
        //    能力位 true；与 portable 解释的等价性是**容差类**（tanh 近似实现次序不同，非逐位），
        //    这里打印实测 `max abs/rel`（参照物 = `ScalarBackend::exec_fused_plan` 的 portable 解释）。
        let xg: Vec<f32> = (0..257).map(|i| (i % 43) as f32 * 0.125 - 2.5).collect();
        let mut g_plan = vec![0.0f32; xg.len()];
        be.exec_fused_plan(FusedPlan::GeluQuick, io(&xg, &mut [], &mut g_plan))
            .unwrap();
        let mut g_port = vec![0.0f32; xg.len()];
        ScalarBackend
            .exec_fused_plan(FusedPlan::GeluQuick, io(&xg, &mut [], &mut g_port))
            .unwrap();
        let (mut max_abs, mut max_rel) = (0.0f64, 0.0f64);
        for (a, b) in g_plan.iter().zip(&g_port) {
            let (a, b) = (*a as f64, *b as f64);
            max_abs = max_abs.max((a - b).abs());
            if b.abs() > 1e-6 {
                max_rel = max_rel.max((a - b).abs() / b.abs());
            }
        }
        println!(
            "GeluQuick 融合 vs portable：max_abs = {max_abs:.3e}｜max_rel = {max_rel:.3e}（257 点）"
        );
        // **判据用 abs 不用 rel**：GELU 在 0 附近输出趋 0 ⇒ rel 会爆（本实测 max_rel 1.3 出现在
        // |portable| 极小处，属数学上无意义的相对误差）。**真正的界是 max_abs**；
        // 参照物 = portable 解释（引擎侧另一实现 ⇒ 本来就是**容差类**，非逐位）。
        assert!(
            max_abs < 5.0e-2,
            "GeluQuick vs portable 的 max_abs 应 < 5e-2，实测 {max_abs:.3e}"
        );

        // ⑤ RopeApply：Plan 路 ≡ 直接 api（RopeMode::NeoX）逐位；vs portable 容差
        let (rows, cols, n_dims) = (8usize, 64usize, 32usize);
        let total = rows * cols;
        let half = n_dims / 2;
        let cos: Vec<f32> = (0..rows * half)
            .map(|i| ((i * 7 % 41) as f32 * 0.03).cos())
            .collect();
        let sin: Vec<f32> = (0..rows * half)
            .map(|i| ((i * 7 % 41) as f32 * 0.03).sin())
            .collect();
        let xr: Vec<f32> = (0..total).map(|i| (i % 61) as f32 * 0.1 - 3.0).collect();
        let rope = FusedPlan::RopeApply { rows, cols, n_dims };
        let mut r_plan = vec![0.0f32; total];
        be.exec_fused_plan(
            rope,
            FusedIo {
                x: &xr,
                y: &mut [],
                out: &mut r_plan,
                cos: &cos,
                sin: &sin,
            },
        )
        .unwrap();
        let mut r_api = vec![0.0f32; total];
        crate::api::rope_into(
            &xr,
            &cos,
            &sin,
            rows,
            cols,
            n_dims,
            crate::api::RopeMode::NeoX,
            &mut r_api,
        )
        .unwrap();
        for (a, b) in r_plan.iter().zip(&r_api) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "RopeApply：Plan 路与直接 api 不逐位"
            );
        }
        let mut r_port = vec![0.0f32; total];
        ScalarBackend
            .exec_fused_plan(
                // `FusedPlan` 非 Copy ⇒ 现构造（同一字段值）。
                FusedPlan::RopeApply { rows, cols, n_dims },
                FusedIo {
                    x: &xr,
                    y: &mut [],
                    out: &mut r_port,
                    cos: &cos,
                    sin: &sin,
                },
            )
            .unwrap();
        let d_rope = max_d(&r_plan, &r_port);
        eprintln!("Plan RopeApply：LASX vs portable 计划解释 max|d| = {d_rope:.3e}");
        assert!(
            d_rope <= 1e-5,
            "RopeApply 计划路与 portable 超界 {d_rope:e}"
        );

        // ⑥ 批2：f64 几何 5 pattern —— Plan 路 ≡ 直接 api **逐位**；unitize3 零向量 = LASX `0`
        let gn = 257usize;
        let g = |seed: usize| -> Vec<f64> {
            (0..gn)
                .map(|i| ((i * 37 + seed) % 101) as f64 * 0.0625 - 3.0)
                .collect()
        };
        let (gax, gay, gaz) = (g(1), g(2), g(3));
        let (gbx, gby, gbz) = (g(4), g(5), g(6));
        let empty_io = || io(&[], &mut [], &mut []);
        // vec3_add_scaled
        let (mut o1x, mut o1y, mut o1z) = (vec![0.0f64; gn], vec![0.0f64; gn], vec![0.0f64; gn]);
        let (mut a1x, mut a1y, mut a1z) = (vec![0.0f64; gn], vec![0.0f64; gn], vec![0.0f64; gn]);
        be.exec_fused_plan(
            FusedPlan::Vec3AddScaledF64 {
                s: 1.25,
                a: [&gax, &gay, &gaz],
                b: [&gbx, &gby, &gbz],
                o: [&mut o1x[..], &mut o1y[..], &mut o1z[..]],
            },
            empty_io(),
        )
        .unwrap();
        crate::api::vec3_add_scaled_batch_into(
            &gax, &gay, &gaz, &gbx, &gby, &gbz, 1.25, &mut a1x, &mut a1y, &mut a1z,
        )
        .unwrap();
        for (p, q) in [(&o1x, &a1x), (&o1y, &a1y), (&o1z, &a1z)] {
            for (u, v) in p.iter().zip(q) {
                assert_eq!(
                    u.to_bits(),
                    v.to_bits(),
                    "Vec3AddScaled：Plan 路与 api 不逐位"
                );
            }
        }
        // norm3
        let (mut n1, mut n2) = (vec![0.0f64; gn], vec![0.0f64; gn]);
        be.exec_fused_plan(
            FusedPlan::Norm3BatchF64 {
                xyz: [&gax, &gay, &gaz],
                out: &mut n1,
            },
            empty_io(),
        )
        .unwrap();
        crate::api::norm3_batch_into(&gax, &gay, &gaz, &mut n2).unwrap();
        for (u, v) in n1.iter().zip(&n2) {
            assert_eq!(u.to_bits(), v.to_bits(), "Norm3Batch：Plan 路与 api 不逐位");
        }
        // cross3
        let (mut c1x, mut c1y, mut c1z) = (vec![0.0f64; gn], vec![0.0f64; gn], vec![0.0f64; gn]);
        let (mut c2x, mut c2y, mut c2z) = (vec![0.0f64; gn], vec![0.0f64; gn], vec![0.0f64; gn]);
        be.exec_fused_plan(
            FusedPlan::Cross3BatchF64 {
                a: [&gax, &gay, &gaz],
                b: [&gbx, &gby, &gbz],
                o: [&mut c1x[..], &mut c1y[..], &mut c1z[..]],
            },
            empty_io(),
        )
        .unwrap();
        crate::api::cross3_batch_into(
            &gax, &gay, &gaz, &gbx, &gby, &gbz, &mut c2x, &mut c2y, &mut c2z,
        )
        .unwrap();
        for (p, q) in [(&c1x, &c2x), (&c1y, &c2y), (&c1z, &c2z)] {
            for (u, v) in p.iter().zip(q) {
                assert_eq!(
                    u.to_bits(),
                    v.to_bits(),
                    "Cross3Batch：Plan 路与 api 不逐位"
                );
            }
        }
        // mat3（走 Rust 层 `_into` helper）
        let gm: Vec<Vec<f64>> = (0..9).map(|k| g(10 + k)).collect();
        let gmr: [&[f64]; 9] = std::array::from_fn(|k| gm[k].as_slice());
        let (mut m1x, mut m1y, mut m1z) = (vec![0.0f64; gn], vec![0.0f64; gn], vec![0.0f64; gn]);
        let (mut m2x, mut m2y, mut m2z) = (vec![0.0f64; gn], vec![0.0f64; gn], vec![0.0f64; gn]);
        be.exec_fused_plan(
            FusedPlan::Mat3MulVec3BatchF64 {
                m: gmr,
                v: [&gax, &gay, &gaz],
                o: [&mut m1x[..], &mut m1y[..], &mut m1z[..]],
            },
            empty_io(),
        )
        .unwrap();
        crate::api::mat3_mul_vec3_batch_into(gmr, &gax, &gay, &gaz, &mut m2x, &mut m2y, &mut m2z)
            .unwrap();
        for (p, q) in [(&m1x, &m2x), (&m1y, &m2y), (&m1z, &m2z)] {
            for (u, v) in p.iter().zip(q) {
                assert_eq!(
                    u.to_bits(),
                    v.to_bits(),
                    "Mat3MulVec3Batch：Plan 路与 api 不逐位"
                );
            }
        }
        // unitize3：非零 ≡ api 逐位；**零向量 = LASX 0 / portable 计划解释 = NaN**（路径各自语义）
        let (mut u1x, mut u1y, mut u1z) = (vec![0.0f64; gn], vec![0.0f64; gn], vec![0.0f64; gn]);
        let (mut u2x, mut u2y, mut u2z) = (vec![0.0f64; gn], vec![0.0f64; gn], vec![0.0f64; gn]);
        be.exec_fused_plan(
            FusedPlan::Unitize3BatchF64 {
                xyz: [&gax, &gay, &gaz],
                o: [&mut u1x[..], &mut u1y[..], &mut u1z[..]],
            },
            empty_io(),
        )
        .unwrap();
        crate::api::unitize3_batch_into(&gax, &gay, &gaz, &mut u2x, &mut u2y, &mut u2z).unwrap();
        for (p, q) in [(&u1x, &u2x), (&u1y, &u2y), (&u1z, &u2z)] {
            for (u, v) in p.iter().zip(q) {
                assert_eq!(
                    u.to_bits(),
                    v.to_bits(),
                    "Unitize3Batch：Plan 路与 api 不逐位"
                );
            }
        }
        let z4 = vec![0.0f64; 4];
        let (mut zx1, mut zy1, mut zz1) = (vec![f64::NAN; 4], vec![f64::NAN; 4], vec![f64::NAN; 4]);
        be.exec_fused_plan(
            FusedPlan::Unitize3BatchF64 {
                xyz: [&z4, &z4, &z4],
                o: [&mut zx1[..], &mut zy1[..], &mut zz1[..]],
            },
            empty_io(),
        )
        .unwrap();
        for arr in [&zx1, &zy1, &zz1] {
            assert!(
                arr.iter().all(|v| v.to_bits() == 0.0f64.to_bits()),
                "LASX unitize3 零向量必须是 0（内核语义）"
            );
        }
        let (mut zx2, mut zy2, mut zz2) = (vec![0.0f64; 4], vec![0.0f64; 4], vec![0.0f64; 4]);
        ScalarBackend
            .exec_fused_plan(
                FusedPlan::Unitize3BatchF64 {
                    xyz: [&z4, &z4, &z4],
                    o: [&mut zx2[..], &mut zy2[..], &mut zz2[..]],
                },
                empty_io(),
            )
            .unwrap();
        for arr in [&zx2, &zy2, &zz2] {
            assert!(
                arr.iter().all(|v| v.is_nan()),
                "契约默认解释的零向量必须是 NaN（组合版 0/0）"
            );
        }
        // 几何长度错 ⇒ Err；空批量 ⇒ Ok
        assert!(matches!(
            be.exec_fused_plan(
                FusedPlan::Norm3BatchF64 {
                    xyz: [&gax, &gay, &gaz[..gn - 1]],
                    out: &mut n1,
                },
                empty_io(),
            ),
            Err(BackendError::LengthMismatch { .. })
        ));
        let empty = vec![0.0f64; 0];
        let (mut e0, mut e1, mut e2) = (vec![0.0f64; 0], vec![0.0f64; 0], vec![0.0f64; 0]);
        be.exec_fused_plan(
            FusedPlan::Vec3AddScaledF64 {
                s: 1.0,
                a: [&empty, &empty, &empty],
                b: [&empty, &empty, &empty],
                o: [&mut e0[..], &mut e1[..], &mut e2[..]],
            },
            empty_io(),
        )
        .unwrap();

        // ⑥b 批3：i8 量化 4 pattern —— quantize/dot/gemv：Plan 路 ≡ portable 标量（**逐位**）；
        //      dequantize：本后端**未接** ⇒ Plan 臂回退 portable 解释（**逐位**；`supports_plan` = false）。
        let (qrows, qcols) = (3usize, 257usize);
        let qn = qrows * qcols;
        // ties-even / clamp / NaN / ±inf 边界样本（其余补 0.75）
        let mut qx: Vec<f32> = vec![
            0.5,
            -0.5,
            1.5,
            -1.5,
            2.5,
            -2.5,
            127.5,
            -127.5,
            128.5,
            -128.5,
            129.5,
            -129.5,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
        ];
        qx.resize(qn, 0.75);
        let qscale: Vec<f32> = (0..qrows).map(|r| 0.5 + r as f32).collect();
        let (mut q_plan, mut q_port) = (vec![0i8; qn], vec![0i8; qn]);
        be.exec_fused_plan(
            FusedPlan::QuantizeI8Rows {
                rows: qrows,
                cols: qcols,
                x: &qx,
                scale: &qscale,
                out: &mut q_plan,
            },
            empty_io(),
        )
        .unwrap();
        simd_dsl::kernels::scalar::quantize_i8_rows_f32(&qx, qrows, qcols, &qscale, &mut q_port)
            .unwrap();
        assert_eq!(
            q_plan, q_port,
            "QuantizeI8Rows：Plan 路与 portable 标量不逐位（ties-even/clamp/NaN/±inf 边界）"
        );
        // dequantize：Plan 回退 ≡ portable（逐位）
        let (mut dq_plan, mut dq_port) = (vec![0.0f32; qn], vec![0.0f32; qn]);
        be.exec_fused_plan(
            FusedPlan::DequantizeI8Rows {
                rows: qrows,
                cols: qcols,
                x: &q_plan,
                scale: &qscale,
                out: &mut dq_plan,
            },
            empty_io(),
        )
        .unwrap();
        simd_dsl::kernels::scalar::dequantize_i8_rows_f32(
            &q_plan,
            qrows,
            qcols,
            &qscale,
            &mut dq_port,
        )
        .unwrap();
        for (a, b) in dq_plan.iter().zip(&dq_port) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "DequantizeI8Rows 回退与 portable 不逐位"
            );
        }
        // dot：普通形状 + 正/负饱和（k 足够大 ⇒ i32::MAX/MIN）
        for &dn in &[0usize, 1, 31, 32, 33, 1029] {
            let da: Vec<i8> = (0..dn)
                .map(|i| ((i * 53 % 251) as i32 - 125) as i8)
                .collect();
            let db: Vec<i8> = (0..dn)
                .map(|i| ((i * 97 % 241) as i32 - 120) as i8)
                .collect();
            let mut got = 0i32;
            be.exec_fused_plan(
                FusedPlan::DotI8 {
                    a: &da,
                    b: &db,
                    out: &mut got,
                },
                empty_io(),
            )
            .unwrap();
            assert_eq!(
                got,
                simd_dsl::kernels::scalar::dot_i8(&da, &db).unwrap(),
                "DotI8：Plan 路与 portable 不逐位（n={dn}）"
            );
        }
        let sat_n = 200_000usize;
        let sat_a = vec![127i8; sat_n];
        let sat_b = vec![127i8; sat_n];
        let mut got = 0i32;
        be.exec_fused_plan(
            FusedPlan::DotI8 {
                a: &sat_a,
                b: &sat_b,
                out: &mut got,
            },
            empty_io(),
        )
        .unwrap();
        assert_eq!(
            got,
            i32::MAX,
            "DotI8 正饱和必须是 i32::MAX（i64 累加 + 末端一次饱和）"
        );
        // gemv：普通 + 饱和
        for &(j, k) in &[(0usize, 5usize), (4, 31), (3, 1024)] {
            let ga: Vec<i8> = (0..j * k)
                .map(|i| ((i * 41 % 251) as i32 - 125) as i8)
                .collect();
            let gx: Vec<i8> = (0..k)
                .map(|i| ((i * 59 % 241) as i32 - 120) as i8)
                .collect();
            let (mut go, mut gwant) = (vec![0i32; j], vec![0i32; j]);
            be.exec_fused_plan(
                FusedPlan::GemvI8 {
                    j,
                    k,
                    a: &ga,
                    x: &gx,
                    out: &mut go,
                },
                empty_io(),
            )
            .unwrap();
            simd_dsl::kernels::scalar::gemv_i8(j, k, &ga, &gx, &mut gwant).unwrap();
            assert_eq!(go, gwant, "GemvI8：Plan 路与 portable 不逐位（{j}×{k}）");
        }
        // 量化错误路径 ⇒ Err（不 panic）
        assert!(matches!(
            be.exec_fused_plan(
                FusedPlan::QuantizeI8Rows {
                    rows: 2,
                    cols: 3,
                    x: &[0.0; 5],
                    scale: &[1.0; 2],
                    out: &mut [0i8; 6],
                },
                empty_io(),
            ),
            Err(BackendError::LengthMismatch { .. })
        ));
        assert!(matches!(
            be.exec_fused_plan(
                FusedPlan::DotI8 {
                    a: &[1i8; 3],
                    b: &[1i8; 2],
                    out: &mut 0i32,
                },
                empty_io(),
            ),
            Err(BackendError::LengthMismatch { .. })
        ));
        assert!(matches!(
            be.exec_fused_plan(
                FusedPlan::GemvI8 {
                    j: 2,
                    k: 3,
                    a: &[1i8; 5],
                    x: &[1i8; 3],
                    out: &mut [0i32; 2],
                },
                empty_io(),
            ),
            Err(BackendError::LengthMismatch { .. })
        ));

        // ⑦ 形状/长度错 ⇒ Err（不 panic）
        assert!(matches!(
            be.exec_fused_plan(
                FusedPlan::Axpy { alpha: 1.0 },
                io(&[1.0, 2.0, 3.0], &mut [0.0; 2], &mut []),
            ),
            Err(BackendError::LengthMismatch { .. })
        ));
        assert!(matches!(
            be.exec_fused_plan(
                FusedPlan::SoftmaxRows { rows: 2, cols: 3 },
                io(&[1.0, 2.0], &mut [], &mut [0.0; 6]),
            ),
            Err(BackendError::LengthMismatch { .. })
        ));
        assert!(matches!(
            be.exec_fused_plan(
                FusedPlan::RmsNormRows {
                    rows: 2,
                    cols: 3,
                    eps: 1e-5
                },
                io(&[1.0; 6], &mut [], &mut [0.0; 5]),
            ),
            Err(BackendError::LengthMismatch { .. })
        ));
        assert!(matches!(
            be.exec_fused_plan(FusedPlan::GeluQuick, io(&[1.0; 4], &mut [], &mut [0.0; 3])),
            Err(BackendError::LengthMismatch { .. })
        ));
        // Rope：n_dims 奇数 / n_dims > cols / cos 长度错
        for bad in [
            FusedPlan::RopeApply {
                rows: 2,
                cols: 8,
                n_dims: 3,
            },
            FusedPlan::RopeApply {
                rows: 2,
                cols: 8,
                n_dims: 10,
            },
        ] {
            assert!(matches!(
                be.exec_fused_plan(
                    bad,
                    FusedIo {
                        x: &[1.0; 16],
                        y: &mut [],
                        out: &mut [0.0; 16],
                        cos: &[1.0],
                        sin: &[1.0],
                    },
                ),
                Err(BackendError::LengthMismatch { .. })
            ));
        }
        assert!(matches!(
            be.exec_fused_plan(
                FusedPlan::RopeApply {
                    rows: 2,
                    cols: 8,
                    n_dims: 4
                },
                FusedIo {
                    x: &[1.0; 16],
                    y: &mut [],
                    out: &mut [0.0; 16],
                    cos: &[1.0; 3],
                    sin: &[1.0; 4],
                },
            ),
            Err(BackendError::LengthMismatch { .. })
        ));
        // 溢出：rows×cols 超过 usize
        assert!(matches!(
            be.exec_fused_plan(
                FusedPlan::SoftmaxRows {
                    rows: usize::MAX / 2 + 1,
                    cols: 3
                },
                io(&[], &mut [], &mut []),
            ),
            Err(BackendError::ShapeOverflow { .. })
        ));
    }

    /// **P4.2：`reduce_sum_rows_f32` override** —— 与默认实现（`simd_dsl` 标量参照）**逐位**，
    /// 且错误面完全一致。
    ///
    /// 逐位依据：内核用 8-lane 向量加法做与参照**相同**的 per-lane 累加，水平归约用**顺序 fold**
    /// （非两两树）、尾部顺序补加 ⇒ 每条输出的运算次序一致（注释见 `ops::dsl_map::row_sum_bitwise`）。
    #[test]
    fn sum_rows_override_bitwise_vs_default_and_error_surface() {
        let be = LasxBackend;
        // 覆盖：8 的倍数/非倍数行宽、奇数行、单行、空形状。
        for &(rows, cols) in &[
            (0usize, 0usize),
            (0, 5),
            (3, 0),
            (1, 1),
            (7, 8),
            (37, 70), // 尾列 + 尾行块
            (64, 41), // 非 8 倍列宽
            (128, 8),
            (256, 1024),
            (2048, 2048), // 4M 元素 = 阈值 ⇒ 真正走自动并行（全局池线程 >1 时）
        ] {
            let n = rows * cols;
            let x: Vec<f32> = (0..n)
                .map(|i| ((i * 37 % 101) as f32) * 0.1 - 6.07)
                .collect();
            let mut got = vec![f32::NAN; rows];
            let mut want = vec![0.0f32; rows];
            let _ = be.reduce_sum_rows_f32(&x, rows, cols, &mut got);
            simd_dsl::kernels::scalar::reduce_sum_rows_f32(&x, rows, cols, &mut want)
                .expect("scalar 参照");
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "reduce_sum_rows_f32 {rows}×{cols} 与默认实现不逐位"
            );
            // 与"纯顺序"参照是容差类（多累加器 vs 单累加器），只校验在这里不夸张（实测界见测试消息）。
            if n > 0 {
                let mut seq = vec![0.0f32; rows];
                simd_dsl::kernels::scalar::reduce_sum_rows_f32_sequential(&x, rows, cols, &mut seq)
                    .expect("顺序参照");
                let max_rel = got
                    .iter()
                    .zip(&seq)
                    .map(|(a, b)| {
                        let d = (a - b).abs();
                        let s = b.abs().max(1e-12);
                        d / s
                    })
                    .fold(0.0f32, f32::max);
                // 实测界打印（默认实现=8 累加器 vs 纯顺序=1 累加器 ⇒ **容差类**；仅记录用）。
                eprintln!("[P4.2] {rows}×{cols} vs 顺序参照 max_rel={max_rel:.3e}");
                assert!(max_rel <= 1e-5, "与顺序参照 max_rel={max_rel:.3e} 超出预期");
            }
        }

        // 错误面：长度不符 ⇒ LengthMismatch（op 名与默认实现一致）
        assert!(matches!(
            be.reduce_sum_rows_f32(&[0.0; 6], 2, 4, &mut [0.0; 2]),
            Err(BackendError::LengthMismatch {
                op: "reduce_sum_rows_f32",
                what: "x",
                ..
            })
        ));
        assert!(matches!(
            be.reduce_sum_rows_f32(&[0.0; 8], 2, 4, &mut [0.0; 3]),
            Err(BackendError::LengthMismatch {
                op: "reduce_sum_rows_f32",
                what: "out",
                ..
            })
        ));
        // 溢出 ⇒ ShapeOverflow
        assert!(matches!(
            be.reduce_sum_rows_f32(&[], usize::MAX / 2 + 1, 3, &mut []),
            Err(BackendError::ShapeOverflow {
                op: "reduce_sum_rows_f32",
                ..
            })
        ));
        // 空输入：合法空操作
        be.reduce_sum_rows_f32(&[], 0, 0, &mut [])
            .expect("空输入合法");
    }

    /// **P4.4：`reduce_absmax_rows_f32` override** —— 与契约默认实现 / 旧 3 趟组合 /
    /// LASX `api::absmax_rows` **逐位一致**（含 NaN/±0/±inf/次正规/空/单行），错误面同口径。
    ///
    /// 旧 3 趟组合 = `map_affine_f32(neg) → map_binary_f32(Max) → reduce_max_rows_f32`（后端调用序）。
    #[test]
    fn absmax_rows_override_bitwise_and_error_surface() {
        let be = LasxBackend;
        let pass = |rows: usize, cols: usize, x: &[f32]| -> Vec<f32> {
            let n = rows * cols;
            let mut got = vec![f32::NAN; rows];
            be.reduce_absmax_rows_f32(x, rows, cols, &mut got).unwrap();
            // 旧 3 趟组合（与旧 ops_dsl 生产路径的后端调用完全相同）。
            // **边角例外（已文档化，P4.4）**：旧组合在"全 NaN 行"得 NaN（fold 初值是 NaN）、
            // 在"−0 行"可能得 −0；新的 1 趟核与 LASX `amax` 同口径（NaN 被忽略 ⇒ 0.0；abs ⇒ +0）
            // ⇒ 这两类边角**不承诺与旧组合逐位**，只对"无 NaN、无 −0"的普通数据比较旧臂。
            let edge = x
                .iter()
                .any(|v| v.is_nan() || (v.is_sign_negative() && *v == 0.0));
            if rows > 0 && cols > 0 && !edge {
                let (mut neg, mut abs) = (vec![0.0f32; n], vec![0.0f32; n]);
                let mut old = vec![0.0f32; rows];
                be.map_affine_f32(x, -1.0, 0.0, &mut neg).unwrap();
                be.map_binary_f32(x, &neg, &mut abs, simd_dsl::BinaryOp::Max)
                    .unwrap();
                be.reduce_max_rows_f32(&abs, rows, cols, &mut old).unwrap();
                assert_eq!(
                    got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    old.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "{rows}×{cols}：override 与旧 3 趟不逐位"
                );
            }
            // 与 LASX 1 趟核（api::absmax_rows）逐位
            let api = crate::api::absmax_rows(x, rows, cols).expect("api::absmax_rows");
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                api.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{rows}×{cols}：override 与 api 核不逐位"
            );
            got
        };

        // 普通值 + 特殊值
        for &(rows, cols) in &[
            (0usize, 0usize),
            (0, 3),
            (3, 0),
            (1, 1),
            (1, 37),
            (37, 0),
            (37, 70),
            (64, 41),
        ] {
            let x: Vec<f32> = (0..rows * cols)
                .map(|i| ((i * 29 % 71) as f32) * 0.37 - 12.0)
                .collect();
            let got = pass(rows, cols, &x);
            assert_eq!(got.len(), rows);
        }
        let specials = [
            vec![f32::NAN, f32::NAN],                    // 全 NaN 行 ⇒ 0.0
            vec![-0.0, -0.0],                            // ±0 ⇒ +0
            vec![f32::NAN, -7.0],                        // NaN 被忽略
            vec![f32::INFINITY, f32::NEG_INFINITY, 3.0], // ±inf
            vec![f32::from_bits(1), 0.5],                // 次正规
        ];
        for x in specials {
            let got = pass(1, x.len(), &x);
            assert!(got[0].is_sign_positive(), "absmax 结果应为 +0/正值");
        }
        pass(0, 3, &[]);
        pass(3, 0, &[]);

        // 错误面与默认同口径
        assert!(matches!(
            be.reduce_absmax_rows_f32(&[0.0; 6], 2, 4, &mut [0.0; 2]),
            Err(BackendError::LengthMismatch {
                op: "reduce_absmax_rows_f32",
                what: "x",
                ..
            })
        ));
        assert!(matches!(
            be.reduce_absmax_rows_f32(&[0.0; 8], 2, 4, &mut [0.0; 3]),
            Err(BackendError::LengthMismatch {
                op: "reduce_absmax_rows_f32",
                what: "out",
                ..
            })
        ));
        assert!(matches!(
            be.reduce_absmax_rows_f32(&[], usize::MAX / 2 + 1, 3, &mut []),
            Err(BackendError::ShapeOverflow {
                op: "reduce_absmax_rows_f32",
                ..
            })
        ));
        be.reduce_absmax_rows_f32(&[], 0, 0, &mut [])
            .expect("空输入合法");
    }

    /// **P4.4 A/B：旧 3 趟组合 vs 单趟 override vs LASX `api::absmax_rows`（性能探针，不进 CI）**。
    ///
    /// **为什么 `#[ignore]`**：本测试纯测量（**没有任何时间阈值断言**），却要在 16M 元素 × 3 形状上
    /// 扫 3 臂多轮——放在 CI 里既拖时间、又让"机器负载"混进 CI 日志。证据价值保留：
    /// 手动跑 `cargo test --release -p lasx_rs --lib absmax_rows_single_pass_ab -- --ignored --nocapture`。
    ///
    /// **口径（抗噪）**：同进程交替 **8 对（丢首对 ⇒ 7 对）**，同时打印**中位比值**与 **min 比值**。
    /// 为什么 min：共享机器负载**只增不减** ⇒ 每臂的 min 更接近"无噪速度"；正式验收仍用
    /// 进程级交替 + 中位（本探针只是趋势证据，不据此下"采纳/不采纳"结论）。
    #[ignore = "性能探针（无断言、16M×3 形状）：cargo test --release -p lasx_rs --lib absmax_rows_single_pass_ab -- --ignored --nocapture"]
    #[test]
    fn absmax_rows_single_pass_ab() {
        use std::time::{Duration, Instant};
        let be = LasxBackend;
        let loadavg = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
        eprintln!(
            "absmax_rows A/B（LasxBackend；单线程）loadavg={}",
            loadavg.trim()
        );
        eprintln!(
            "| rows×cols | 3-pass ms(med) | 1-pass ms(med) | api ms(med) | 3pass/1pass(med) | 1pass/api(med) | 3pass/1pass(min) | 1pass/api(min) |"
        );
        eprintln!("|---|---|---|---|---|---|---|---|");
        for &(rows, cols) in &[(1024usize, 1024usize), (4096, 1024), (16384, 1024)] {
            let n = rows * cols;
            let x: Vec<f32> = (0..n)
                .map(|i| ((i * 29 % 71) as f32) * 0.37 - 12.0)
                .collect();
            let (mut old_out, mut new_out) = (vec![0.0f32; rows], vec![0.0f32; rows]);
            let (mut neg, mut abs) = (vec![0.0f32; n], vec![0.0f32; n]);
            let three_pass = |old_out: &mut [f32], neg: &mut [f32], abs: &mut [f32]| {
                be.map_affine_f32(&x, -1.0, 0.0, neg).unwrap();
                be.map_binary_f32(&x, neg, abs, simd_dsl::BinaryOp::Max)
                    .unwrap();
                be.reduce_max_rows_f32(abs, rows, cols, old_out).unwrap();
            };
            let one_pass = |new_out: &mut [f32]| {
                be.reduce_absmax_rows_f32(&x, rows, cols, new_out).unwrap();
            };
            let api_core = || {
                let v = crate::api::absmax_rows(&x, rows, cols).unwrap();
                std::hint::black_box(&v[0]);
            };
            let (mut t_old, mut t_new, mut t_api) = (Vec::new(), Vec::new(), Vec::new());
            for round in 0..8 {
                let timeit = |f: &mut dyn FnMut()| -> Duration {
                    let t = Instant::now();
                    f();
                    t.elapsed()
                };
                let mut f_old = || three_pass(&mut old_out, &mut neg, &mut abs);
                let mut f_new = || one_pass(&mut new_out);
                let mut f_api = api_core;
                let order = round % 3;
                if order == 0 {
                    t_old.push(timeit(&mut f_old));
                    t_new.push(timeit(&mut f_new));
                    t_api.push(timeit(&mut f_api));
                } else if order == 1 {
                    t_new.push(timeit(&mut f_new));
                    t_api.push(timeit(&mut f_api));
                    t_old.push(timeit(&mut f_old));
                } else {
                    t_api.push(timeit(&mut f_api));
                    t_old.push(timeit(&mut f_old));
                    t_new.push(timeit(&mut f_new));
                }
            }
            // 丢首对（预热）：三臂同时丢，保持配对。
            t_old.remove(0);
            t_new.remove(0);
            t_api.remove(0);
            let median = |v: &mut Vec<Duration>| {
                v.sort_unstable();
                v[v.len() / 2]
            };
            let min = |v: &mut Vec<Duration>| {
                v.sort_unstable();
                v[0]
            };
            let (m_old, m_new, m_api) =
                (median(&mut t_old), median(&mut t_new), median(&mut t_api));
            let (n_old, n_new, n_api) = (min(&mut t_old), min(&mut t_new), min(&mut t_api));
            eprintln!(
                "| {rows}×{cols} | {:.3} | {:.3} | {:.3} | **{:.2}×** | **{:.2}×** | **{:.2}×** | **{:.2}×** |",
                m_old.as_secs_f64() * 1e3,
                m_new.as_secs_f64() * 1e3,
                m_api.as_secs_f64() * 1e3,
                m_old.as_secs_f64() / m_new.as_secs_f64(),
                m_new.as_secs_f64() / m_api.as_secs_f64(),
                n_old.as_secs_f64() / n_new.as_secs_f64(),
                n_new.as_secs_f64() / n_api.as_secs_f64()
            );
        }
    }
}
