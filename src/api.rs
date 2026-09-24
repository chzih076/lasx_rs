//! Rust 原生 API：**切片进出、错误上抛、输出对齐**。
//!
//! # 与另外两层的分工
//!
//! | 层 | 给谁用 | 形态 |
//! |---|---|---|
//! | [`crate::ffi`]（`lasx_*` 符号） | C / Dart 等 FFI 调用方 | 裸指针 + `i32` 长度，零校验（**误用即 UB**） |
//! | **`api`（本模块）** | **Rust 调用方** | **`&[T]` / `&mut [T]` 进出，形状不符返回 [`Error`]，输出是 32 字节对齐的 [`AlignedVec`]** |
//! | [`crate::parallel`] | Rust，要多核 | 在本层之外再叠常驻线程池 |
//!
//! ```
//! use lasx_rs::api;
//!
//! let a = [1.0f32, 2.0, 3.0];
//! let b = [1.0f32, 1.0, 1.0];
//! assert_eq!(api::dot(&a, &b).unwrap(), 6.0);
//! assert_eq!(api::sum(&a), 6.0);
//!
//! // 形状不对是 Err，不是 UB、也不是未定义行为
//! assert!(api::dot(&a, &b[..2]).is_err());
//!
//! // 需要输出的内核直接返回对齐缓冲
//! let c = api::matmul(2, 2, 2, &[1.0, 0.0, 0.0, 1.0], &[1.0, 2.0, 3.0, 4.0]).unwrap();
//! assert_eq!(&c[..], &[1.0, 2.0, 3.0, 4.0]);
//! ```
//!
//! # 为什么输出是 [`AlignedVec`] 而不是 `Vec`
//!
//! LASX 是 32 字节访存，缓冲区是否对齐直接影响性能（L1 驻留规模上实测 1.1–1.56×），
//! 而 `Vec<T>` 只保证 `align_of::<T>()`（f32 只有 4 字节）。本层新分配的输出落在
//! 对齐窗口内，调用方不必再操心，也不会因为"忘了 `posix_memalign`"而白丢性能。
//!
//! # 不会失败的操作不套 `Result`
//!
//! [`sum`] 对任何切片都有定义（空切片求和为 0），所以直接返回 `f32`；其余内核的形状
//! 约束是真实存在的，一律返回 `Result`。已经自己校验过形状、且要榨掉最后几 ns 的调用方，
//! 仍然可以用 [`crate::ffi`] 的裸指针版本。
//!
//! # 错误类型
//!
//! [`Error`] 带上**算子名、参数名、期望值与实际值**，实现 [`std::error::Error`]，
//! 因此 `?` 与 `Box<dyn Error>` 都能直接用。

use crate::aligned::AlignedVec;
// `RopeMode` 定义在私有的 `ops::rope` 里（内核的家），在这里**重新导出**成公开名字：
// `ops` 整体是私有的（内核不该成为公开 API），但类型本身可以是公开可达的。
pub use crate::ops::rope::RopeMode;

/// `api` 层可能返回的错误：形状不符、形状相乘溢出、物理常数非法。
///
/// 每个变体都带 `op`（算子名）与 `what`（参数名），消息可直接给最终用户看。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// 参与同一运算的数组长度（或形状）与要求不符。
    Shape {
        /// 算子名，如 `"matmul"`。
        op: &'static str,
        /// 参数名，如 `"b"`。
        what: &'static str,
        /// 要求的长度/元素个数。
        expected: usize,
        /// 实际长度/元素个数。
        got: usize,
    },
    /// 形状相乘溢出（`m×k`、`k×n`、`m×n` 之一超出 `usize`）。
    Overflow {
        /// 算子名。
        op: &'static str,
        /// 溢出的是哪个乘积，如 `"m×k"`。
        what: &'static str,
    },
    /// 常数不是有限值（NaN 或 ±∞）。
    NotFinite {
        /// 算子名。
        op: &'static str,
        /// 参数名，如 `"dt"`。
        what: &'static str,
        /// 实际取值。
        value: f64,
    },
    /// 常数必须为正，实际不是。
    NotPositive {
        /// 算子名。
        op: &'static str,
        /// 参数名，如 `"mu"`。
        what: &'static str,
        /// 实际取值。
        value: f64,
    },
    /// 某个参数取值不合法，但不是长度/有限性/正负号这三类（例如"必须为偶数"）。
    ///
    /// 加这个变体是因为 `rope` 的 `n_dims` 既不是长度不符也不是大小越界，而是一个**约束**；
    /// 硬塞进 `Shape` 会给出误导的消息（"长度应为 X，实际 Y" 说不清"必须是偶数"）。
    /// `Error` 是 `#[non_exhaustive]` 的，新增变体不破坏下游的 `match`。
    BadValue {
        /// 算子名。
        op: &'static str,
        /// 参数名，如 `"n_dims"`。
        what: &'static str,
        /// 约束说明（直接进错误消息）。
        requirement: &'static str,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Shape {
                op,
                what,
                expected,
                got,
            } => write!(f, "{op}: 参数 {what} 的长度应为 {expected}，实际 {got}"),
            Error::Overflow { op, what } => write!(f, "{op}: {what} 溢出"),
            Error::NotFinite { op, what, value } => {
                write!(f, "{op}: 常数 {what} 必须是有限数，得到 {value}")
            }
            Error::NotPositive { op, what, value } => {
                write!(f, "{op}: 常数 {what} 必须为正，得到 {value}")
            }
            Error::BadValue {
                op,
                what,
                requirement,
            } => write!(f, "{op}: 参数 {what} 必须{requirement}"),
        }
    }
}

impl std::error::Error for Error {}

type Result<T> = std::result::Result<T, Error>;

/// 长度必须等于 `expected`。
///
/// `pub(crate)`：`parallel` 层复用同一套错误与消息口径，免得两条路径的报错风格不一致。
pub(crate) fn expect_len(
    op: &'static str,
    what: &'static str,
    got: usize,
    expected: usize,
) -> Result<()> {
    if got == expected {
        Ok(())
    } else {
        Err(Error::Shape {
            op,
            what,
            expected,
            got,
        })
    }
}

/// 长度必须至少为 `min`（`dot_q4` 的 scale 数组按约定是"≥ 组数"）。
fn expect_at_least(op: &'static str, what: &'static str, got: usize, min: usize) -> Result<()> {
    if got >= min {
        Ok(())
    } else {
        Err(Error::Shape {
            op,
            what,
            expected: min,
            got,
        })
    }
}

/// 一组数组必须等长（SOA 内核的形状约束），返回公共长度。
fn expect_same_len(op: &'static str, lens: &[(&'static str, usize)]) -> Result<usize> {
    let n = lens[0].1;
    for &(what, got) in &lens[1..] {
        if got != n {
            return Err(Error::Shape {
                op,
                what,
                expected: n,
                got,
            });
        }
    }
    Ok(n)
}

/// 形状相乘溢出检查；`pub(crate)` 理由同 [`expect_len`]。
pub(crate) fn checked_mul(
    op: &'static str,
    what: &'static str,
    a: usize,
    b: usize,
) -> Result<usize> {
    a.checked_mul(b).ok_or(Error::Overflow { op, what })
}

fn expect_finite(op: &'static str, what: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(Error::NotFinite { op, what, value })
    }
}

fn expect_positive(op: &'static str, what: &'static str, value: f64) -> Result<()> {
    if value > 0.0 {
        Ok(())
    } else {
        Err(Error::NotPositive { op, what, value })
    }
}

/* ==================== 归约 ==================== */

/// `Σ x[i]`（f32）。空切片返回 `0.0`，任何切片都有定义，故不返回 `Result`。
#[inline]
pub fn sum(x: &[f32]) -> f32 {
    crate::ops::sum::sum(x)
}

/// `Σ a[i]·b[i]`（f32）。
///
/// # Errors
/// `a` 与 `b` 长度不一致时返回 [`Error::Shape`]。
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> Result<f32> {
    expect_len("dot", "b", b.len(), a.len())?;
    Ok(crate::ops::dot::dot(a, b))
}

/// `Σ a[i]·b[i]`（f64）。
///
/// # Errors
/// `a` 与 `b` 长度不一致时返回 [`Error::Shape`]。
#[inline]
pub fn dot_f64(a: &[f64], b: &[f64]) -> Result<f64> {
    expect_len("dot_f64", "b", b.len(), a.len())?;
    Ok(crate::ops::dot_f64::dot_f64(a, b))
}

/// int8 量化点积（结果是精确的整数和，不溢出）。
///
/// # Errors
/// `a` 与 `b` 长度不一致时返回 [`Error::Shape`]。
#[inline]
pub fn dot_i8(a: &[i8], b: &[i8]) -> Result<i32> {
    expect_len("dot_i8", "b", b.len(), a.len())?;
    Ok(crate::ops::dot_i8::dot_i8(a, b))
}

/// Q4 量化点积：`qa`/`qb` 每字节 2 个无符号 nibble，`sa`/`sb` 每 32 字节一组 scale。
///
/// # Errors
/// - `qa` 与 `qb` 长度不一致；
/// - `sa`/`sb` 长度不足 `ceil(qa.len()/32)`。
pub fn dot_q4(qa: &[u8], sa: &[f32], qb: &[u8], sb: &[f32]) -> Result<f64> {
    let groups = qa.len().div_ceil(32);
    expect_len("dot_q4", "qb", qb.len(), qa.len())?;
    expect_at_least("dot_q4", "sa", sa.len(), groups)?;
    expect_at_least("dot_q4", "sb", sb.len(), groups)?;
    Ok(crate::ops::dot_q4::dot_q4(qa, sa, qb, sb))
}

/// `y[i] += alpha·x[i]`（原地）。
///
/// # Errors
/// `x` 与 `y` 长度不一致时返回 [`Error::Shape`]。
#[inline]
pub fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) -> Result<()> {
    expect_len("axpy", "y", y.len(), x.len())?;
    crate::ops::axpy::axpy(alpha, x, y);
    Ok(())
}

/* ==================== NN：softmax ==================== */

/// 行内 softmax：`out[i][j] = exp(scale·x[i][j] + mask[i][j] − max) / Σ`（`rows × cols` 行主序）。
///
/// `mask` 是**可选**的加性偏置（同形状；`None` = 无 mask）。返回值是新分配的对齐缓冲。
///
/// # 数值契约（见 `docs/ops.md` §2.6）
/// - 行内先减 max 再 `exp`；归一化写成 `e · (1/Σ)`（一次除法 + 乘法），不是逐元素除法；
/// - `exp` 是 `exp2` 的多项式近似（与 f64 参考的相对误差 ~1e-7 量级），**向量路径与标量
///   模拟逐位一致**（单测守着）；
/// - 下溢落到**精确 0**：`mask = −inf` 那种"完全屏蔽"的写法得到 0 权重，不是 1e-38；
/// - 不在契约内：输入或 mask 含 `+inf`（`inf − inf = NaN`，是定义域问题）。
///
/// # Errors
/// - `x.len() != rows × cols`（[`Error::Shape`]）；
/// - `mask` 给了但长度不符（[`Error::Shape`]）；
/// - `rows × cols` 溢出 `usize`（[`Error::Overflow`]）；
/// - `scale` 非有限（[`Error::NotFinite`]）。
pub fn softmax_rows(
    x: &[f32],
    mask: Option<&[f32]>,
    rows: usize,
    cols: usize,
    scale: f32,
) -> Result<AlignedVec<f32>> {
    if !scale.is_finite() {
        return Err(Error::NotFinite {
            op: "softmax_rows",
            what: "scale",
            value: scale as f64,
        });
    }
    let n = checked_mul("softmax_rows", "rows×cols", rows, cols)?;
    expect_len("softmax_rows", "x", x.len(), n)?;
    if let Some(m) = mask {
        expect_len("softmax_rows", "mask", m.len(), n)?;
    }
    let mut out = AlignedVec::<f32>::new(n);
    if n == 0 {
        return Ok(out);
    }
    crate::ops::softmax_rows::softmax_rows(
        x,
        mask.unwrap_or(&[]),
        scale,
        rows,
        cols,
        out.as_mut_slice(),
    );
    Ok(out)
}

/* ==================== NN：RMSNorm ==================== */

/// 行内 RMSNorm：`out[i][j] = x[i][j] / √(mean_j(x[i][j]²) + eps) · w[j]`。
///
/// `w` 是**按列**的可选权重（`None` = 无权重）。返回值是新分配的对齐缓冲。
///
/// # 数值契约（见 `docs/ops.md` §2.7）
/// - 平方和用**每个 k 步一条 `fma(x, x, acc)`**（一次舍入），再按固定次序两两归约；列尾补
///   `0.0`（`0² = 0`，精确无贡献）；
/// - 行级数学（`s/cols`、`+eps`、`sqrt`、`1/x`）在**标量域**算完再广播——省指令，而且
///   天然让"标量模拟"与向量路径逐位相同；
/// - 输出次序固定为 `(x · inv) · w`；
/// - 开方与倒数用**精确指令**，不用 `frsqrte` 估算（`docs/dev.md` §13.1 实测估算更慢）。
///
/// # Errors
/// - `eps` 非有限（[`Error::NotFinite`]）或 `≤ 0`（[`Error::NotPositive`]）；
/// - `x.len() != rows × cols`（[`Error::Shape`]）；
/// - `w` 给了但长度不等于 `cols`（[`Error::Shape`]）；
/// - `rows × cols` 溢出 `usize`（[`Error::Overflow`]）。
pub fn rms_norm(
    x: &[f32],
    w: Option<&[f32]>,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<AlignedVec<f32>> {
    if !eps.is_finite() {
        return Err(Error::NotFinite {
            op: "rms_norm",
            what: "eps",
            value: eps as f64,
        });
    }
    if eps <= 0.0 {
        return Err(Error::NotPositive {
            op: "rms_norm",
            what: "eps",
            value: eps as f64,
        });
    }
    let n = checked_mul("rms_norm", "rows×cols", rows, cols)?;
    expect_len("rms_norm", "x", x.len(), n)?;
    if let Some(w) = w {
        expect_len("rms_norm", "w", w.len(), cols)?;
    }
    let mut out = AlignedVec::<f32>::new(n);
    if n == 0 {
        return Ok(out);
    }
    crate::ops::rms_norm::rms_norm(x, w.unwrap_or(&[]), eps, rows, cols, out.as_mut_slice());
    Ok(out)
}

/* ==================== NN：逐元素激活 ==================== */

/// SiLU（swish）：`y[i] = x[i] / (1 + exp(−x[i]))`，逐元素，返回新分配的对齐缓冲。
///
/// # 数值契约（见 `docs/ops.md` §2.8）
/// - 公式写**直接除法**（`x / den`，单次舍入），不是 `x · (1/den)`——参考实现必须照抄
///   才算逐位一致；
/// - 门控分母 `1 + exp(−x)` 与 `exp` 用 `ops::nn_math` 那一份（自带输入夹取
///   `[−104, +88]`，所以 `±inf` 不会走进 magic 数技巧）；
/// - 逐元素、**无归约**，所以向量路径与标量尾逐位相同（测试里验）。这条与
///   [`softmax_rows`] 不同：那里有归约，位精确要靠固定次序的设计。
pub fn silu(x: &[f32]) -> AlignedVec<f32> {
    let mut out = AlignedVec::<f32>::new(x.len());
    if x.is_empty() {
        return out;
    }
    crate::ops::silu::silu_f32(x, out.as_mut_slice());
    out
}

/// GELU 的 sigmoid 近似（ggml `GELU_QUICK`）：`y[i] = x[i] / (1 + exp(−1.702·x[i]))`。
///
/// 契约同 [`silu`]：直接除法、共用 `nn_math` 的 `exp`、逐元素位精确。
/// erf 形式（PyTorch 默认那个）是另一个入口，见 `docs/ops.md` §2.8 的规划。
pub fn gelu_quick(x: &[f32]) -> AlignedVec<f32> {
    let mut out = AlignedVec::<f32>::new(x.len());
    if x.is_empty() {
        return out;
    }
    crate::ops::gelu_quick::gelu_quick_f32(x, out.as_mut_slice());
    out
}

/// GELU 的 erf 形式（PyTorch `gelu` 的默认定义）：
/// `y[i] = 0.5·x[i]·(1 + erf(x[i]/√2))`。
///
/// `erf` 没有硬件指令，用 **Abramowitz & Stegun 7.1.26**（本身绝对误差 ≤1.5e-7；折算到
/// 输出上最差 ≈2.1e-7，出现在 `|x| ≈ 3.08`）。所以这个入口是**近似**，不是"真 erf"——
/// 需要与 ggml 对齐的场景请用 [`gelu_quick`]。契约见 `docs/ops.md` §2.9。
///
/// 常数契约同 [`silu`]：`|x|` 用 `max(x, −x)`、最终写成 `0.5x + 0.5|x|·erf(|x|/√2)`
/// （避免 `copysign`）、`exp` 复用 `ops::nn_math`。
pub fn gelu_erf(x: &[f32]) -> AlignedVec<f32> {
    let mut out = AlignedVec::<f32>::new(x.len());
    if x.is_empty() {
        return out;
    }
    crate::ops::gelu_erf::gelu_erf_f32(x, out.as_mut_slice());
    out
}

/* ==================== NN：RoPE ==================== */

/// 旋转位置编码（RoPE）：`rows × cols` 行主序，每行**前 `n_dims` 列**参与旋转。
///
/// `cos`/`sin` 各 `rows × (n_dims/2)`、行主序（第 `r` 行第 `i` 列 = `θ_i` 的余弦/正弦），
/// 通常由 [`rope_tables`] 生成。表**与 mode 无关**：同一张表两种配对都能用。
///
/// # 数值契约（见 `docs/ops.md` §2.10）
/// - 每一对元素：`y0 = fma(x0, c, −(x1·s))`、`y1 = fma(x1, c, x0·s)`（**两次舍入**）；
/// - `[n_dims, cols)` 的元素**原样复制**（partial rope）；
/// - 逐元素、无归约；`NeoX` 走 LASX（8 对/向量），`GptJ` 走标量（原因见 `ops::rope` 模块文档）；
/// - 表生成本身**不在位精确承诺内**（`sin`/`cos` 来自平台 libm），承诺的是"给定同一张表，
///   向量与标量逐位相同"。
///
/// # Errors
/// - `x.len() != rows × cols`、`cos`/`sin` 长度不等于 `rows × n_dims/2`（[`Error::Shape`]）；
/// - `n_dims > cols`、`n_dims` 不是偶数（[`Error::BadValue`]）；
/// - `rows × cols` 溢出 `usize`（[`Error::Overflow`]）。
pub fn rope(
    x: &[f32],
    cos: &[f32],
    sin: &[f32],
    rows: usize,
    cols: usize,
    n_dims: usize,
    mode: RopeMode,
) -> Result<AlignedVec<f32>> {
    let n = checked_mul("rope", "rows×cols", rows, cols)?;
    expect_len("rope", "x", x.len(), n)?;
    if !n_dims.is_multiple_of(2) {
        return Err(Error::BadValue {
            op: "rope",
            what: "n_dims",
            requirement: "是偶数（元素两两成对）",
        });
    }
    if n_dims > cols {
        return Err(Error::Shape {
            op: "rope",
            what: "n_dims",
            expected: cols,
            got: n_dims,
        });
    }
    let table_len = checked_mul("rope", "rows×(n_dims/2)", rows, n_dims / 2)?;
    expect_len("rope", "cos", cos.len(), table_len)?;
    expect_len("rope", "sin", sin.len(), table_len)?;
    let mut out = AlignedVec::<f32>::new(n);
    if n == 0 {
        return Ok(out);
    }
    // SAFETY: 上面已校验 `x.len() == rows*cols`、表长 `== rows*(n_dims/2)`、`n_dims` 偶数且
    // `≤ cols` —— 正是 `ops::rope::rope_f32` 要求的那几条前提。
    unsafe {
        crate::ops::rope::rope_f32(x, cos, sin, rows, cols, n_dims, mode, out.as_mut_slice())
    };
    Ok(out)
}

/// 生成 RoPE 的 `cos`/`sin` 表：`positions.len() × (n_dims/2)`、行主序。
///
/// 第 `r` 行第 `i` 列 = `θ_i = positions[r] · freq_base^(−2i/n_dims)` 的余弦/正弦。
/// **标量、f64 算角度**（表只算一次，成本可忽略），且**不在位精确承诺内**：`sin`/`cos`
/// 由平台 libm 提供，跨实现不保证逐位一致。要与别的实现严格对齐，请自己造表并只依赖
/// [`rope`] 的旋转契约。
///
/// # Errors
/// - `n_dims` 不是正偶数（[`Error::BadValue`]；`n_dims = 0` 没有意义，不做旋转就别调这里）；
/// - `freq_base` 非有限（[`Error::NotFinite`]）或 `≤ 0`（[`Error::NotPositive`]）。
pub fn rope_tables(
    positions: &[f32],
    n_dims: usize,
    freq_base: f32,
) -> Result<(AlignedVec<f32>, AlignedVec<f32>)> {
    if n_dims == 0 || !n_dims.is_multiple_of(2) {
        return Err(Error::BadValue {
            op: "rope_tables",
            what: "n_dims",
            requirement: "是正偶数",
        });
    }
    if !freq_base.is_finite() {
        return Err(Error::NotFinite {
            op: "rope_tables",
            what: "freq_base",
            value: freq_base as f64,
        });
    }
    if freq_base <= 0.0 {
        return Err(Error::NotPositive {
            op: "rope_tables",
            what: "freq_base",
            value: freq_base as f64,
        });
    }
    let (c, s) = crate::ops::rope::rope_tables(positions, n_dims, freq_base);
    Ok((
        AlignedVec::fill_with(c.len(), |i| c[i]),
        AlignedVec::fill_with(s.len(), |i| s[i]),
    ))
}

/// 便捷入口：按 `positions` 现场建表并旋转（等价于 [`rope_tables`] + [`rope`]）。
///
/// 一次前向里想复用表就别用这个——它会每次重新建表（表是 `x` 的一半大小，建表成本不可忽略）。
///
/// # Errors
/// 同 [`rope`] 与 [`rope_tables`]。
pub fn rope_at(
    x: &[f32],
    positions: &[f32],
    rows: usize,
    cols: usize,
    n_dims: usize,
    freq_base: f32,
    mode: RopeMode,
) -> Result<AlignedVec<f32>> {
    expect_len("rope_at", "positions", positions.len(), rows)?;
    let (cos, sin) = rope_tables(positions, n_dims, freq_base)?;
    rope(x, &cos, &sin, rows, cols, n_dims, mode)
}

/* ==================== 矩阵乘 ==================== */

/// `C[m×n] = A[m×k] · B[k×n]`（行主序），返回新分配的对齐缓冲。
///
/// # Errors
/// - `a`/`b` 长度不等于 `m×k`/`k×n`（[`Error::Shape`]）；
/// - `m×k`、`k×n` 或 `m×n` 溢出 `usize`（[`Error::Overflow`]）。
pub fn matmul(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<AlignedVec<f32>> {
    expect_len("matmul", "a", a.len(), checked_mul("matmul", "m×k", m, k)?)?;
    expect_len("matmul", "b", b.len(), checked_mul("matmul", "k×n", k, n)?)?;
    let mut c = AlignedVec::<f32>::new(checked_mul("matmul", "m×n", m, n)?);
    crate::ops::matmul::matmul_f32(m, k, n, a, b, c.as_mut_slice());
    Ok(c)
}

/// 强制走"打包 B 面板"路径的 [`matmul`]（小 m、大 n 或大 k 时更快，见 dev.md §8.1）。
///
/// 与 [`matmul`] **逐位一致**（每个输出元素的累加次序相同）。`matmul` 会按形状自动
/// 选择；这个入口用于调用方想自己控制、或做 A/B 测量。
///
/// # Errors
/// 同 [`matmul`]。
pub fn matmul_f32_packed(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
) -> Result<AlignedVec<f32>> {
    expect_len(
        "matmul_f32_packed",
        "a",
        a.len(),
        checked_mul("matmul_f32_packed", "m×k", m, k)?,
    )?;
    expect_len(
        "matmul_f32_packed",
        "b",
        b.len(),
        checked_mul("matmul_f32_packed", "k×n", k, n)?,
    )?;
    let mut c = AlignedVec::<f32>::new(checked_mul("matmul_f32_packed", "m×n", m, n)?);
    crate::ops::matmul::matmul_f32_packed(m, k, n, a, b, c.as_mut_slice());
    Ok(c)
}

/// f64 版 [`matmul`]。
///
/// # Errors
/// 同 [`matmul`]。
pub fn matmul_f64(m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Result<AlignedVec<f64>> {
    expect_len(
        "matmul_f64",
        "a",
        a.len(),
        checked_mul("matmul_f64", "m×k", m, k)?,
    )?;
    expect_len(
        "matmul_f64",
        "b",
        b.len(),
        checked_mul("matmul_f64", "k×n", k, n)?,
    )?;
    let mut c = AlignedVec::<f64>::new(checked_mul("matmul_f64", "m×n", m, n)?);
    crate::ops::matmul_f64::matmul_f64(m, k, n, a, b, c.as_mut_slice());
    Ok(c)
}

/* ==================== 批量几何 ==================== */

/// `out[i] = √(xs[i]² + ys[i]² + zs[i]²)`（f64 SOA）。
///
/// # Errors
/// 三个数组长度不一致时返回 [`Error::Shape`]。
pub fn norm3_batch(xs: &[f64], ys: &[f64], zs: &[f64]) -> Result<AlignedVec<f64>> {
    let n = expect_same_len(
        "norm3_batch",
        &[("xs", xs.len()), ("ys", ys.len()), ("zs", zs.len())],
    )?;
    let mut out = AlignedVec::<f64>::new(n);
    crate::ops::norm3_batch::norm3_batch(xs, ys, zs, out.as_mut_slice());
    Ok(out)
}

/// 每个点到 `(px, py)` 的距离（f32）。
///
/// # Errors
/// `xs` 与 `ys` 长度不一致时返回 [`Error::Shape`]。
pub fn batch_distance2d(px: f32, py: f32, xs: &[f32], ys: &[f32]) -> Result<AlignedVec<f32>> {
    let n = expect_same_len("batch_distance2d", &[("xs", xs.len()), ("ys", ys.len())])?;
    let mut out = AlignedVec::<f32>::new(n);
    crate::ops::batch_distance2d::batch_distance2d(px, py, xs, ys, out.as_mut_slice());
    Ok(out)
}

/* ==================== 批量物理 ==================== */

/// `o = a + s·b`（三个分量各自成 SOA 数组）。
///
/// # Errors
/// 六个输入数组长度不一致时返回 [`Error::Shape`]。
#[allow(clippy::too_many_arguments)]
pub fn vec3_add_scaled_batch(
    ax: &[f64],
    ay: &[f64],
    az: &[f64],
    bx: &[f64],
    by: &[f64],
    bz: &[f64],
    s: f64,
) -> Result<[AlignedVec<f64>; 3]> {
    let n = expect_same_len(
        "vec3_add_scaled_batch",
        &[
            ("ax", ax.len()),
            ("ay", ay.len()),
            ("az", az.len()),
            ("bx", bx.len()),
            ("by", by.len()),
            ("bz", bz.len()),
        ],
    )?;
    let (mut ox, mut oy, mut oz) = (
        AlignedVec::<f64>::new(n),
        AlignedVec::<f64>::new(n),
        AlignedVec::<f64>::new(n),
    );
    crate::ops::vec3_add_scaled_batch::vec3_add_scaled_batch(
        ax,
        ay,
        az,
        bx,
        by,
        bz,
        s,
        ox.as_mut_slice(),
        oy.as_mut_slice(),
        oz.as_mut_slice(),
    );
    Ok([ox, oy, oz])
}

/// 中心引力 + J2 摄动加速度，返回 `[ax, ay, az]`。
///
/// # Errors
/// - 三个位置数组长度不一致（[`Error::Shape`]）；
/// - `mu <= 0` 或 `re <= 0`（[`Error::NotPositive`]）；
/// - `j2` 非有限（[`Error::NotFinite`]）。
pub fn j2_accel_batch(
    rx: &[f64],
    ry: &[f64],
    rz: &[f64],
    mu: f64,
    j2: f64,
    re: f64,
) -> Result<[AlignedVec<f64>; 3]> {
    // 校验顺序与 `lasx_j2_accel_batch_checked` 一致：先常数，后形状
    expect_positive("j2_accel_batch", "mu", mu)?;
    expect_positive("j2_accel_batch", "re", re)?;
    expect_finite("j2_accel_batch", "j2", j2)?;
    let n = expect_same_len(
        "j2_accel_batch",
        &[("rx", rx.len()), ("ry", ry.len()), ("rz", rz.len())],
    )?;
    let (mut ax, mut ay, mut az) = (
        AlignedVec::<f64>::new(n),
        AlignedVec::<f64>::new(n),
        AlignedVec::<f64>::new(n),
    );
    crate::ops::j2_accel_batch::j2_accel_batch(
        rx,
        ry,
        rz,
        mu,
        j2,
        re,
        ax.as_mut_slice(),
        ay.as_mut_slice(),
        az.as_mut_slice(),
    );
    Ok([ax, ay, az])
}

/// 批量弹道欧拉步（f32 SOA，**原地**推进）。
///
/// 常数不做校验（`dt`/`g` 由调用方负责），与 `lasx_ballistic_step` 一致。
///
/// # Errors
/// 七个数组（六个状态 + `k`）长度不一致时返回 [`Error::Shape`]。
#[allow(clippy::too_many_arguments)]
pub fn ballistic_step(
    x: &mut [f32],
    y: &mut [f32],
    z: &mut [f32],
    vx: &mut [f32],
    vy: &mut [f32],
    vz: &mut [f32],
    k: &[f32],
    dt: f32,
    g: f32,
) -> Result<()> {
    expect_same_len(
        "ballistic_step",
        &[
            ("x", x.len()),
            ("y", y.len()),
            ("z", z.len()),
            ("vx", vx.len()),
            ("vy", vy.len()),
            ("vz", vz.len()),
            ("k", k.len()),
        ],
    )?;
    crate::ops::ballistic_step::ballistic_step(x, y, z, vx, vy, vz, k, dt, g);
    Ok(())
}

/// 批量 RK4 J2 单步（f64 SOA，**原地**推进）。
///
/// # Errors
/// - 六个分量数组长度不一致（[`Error::Shape`]）；
/// - `mu <= 0` 或 `re <= 0`（[`Error::NotPositive`]）；
/// - `j2` 或 `dt` 非有限（[`Error::NotFinite`]）。
///
/// 多核版本见 [`crate::parallel::rk4_j2_step_batch`]。
#[allow(clippy::too_many_arguments)]
pub fn rk4_j2_step_batch(
    rx: &mut [f64],
    ry: &mut [f64],
    rz: &mut [f64],
    vx: &mut [f64],
    vy: &mut [f64],
    vz: &mut [f64],
    mu: f64,
    j2: f64,
    re: f64,
    dt: f64,
) -> Result<()> {
    expect_positive("rk4_j2_step_batch", "mu", mu)?;
    expect_positive("rk4_j2_step_batch", "re", re)?;
    expect_finite("rk4_j2_step_batch", "j2", j2)?;
    expect_finite("rk4_j2_step_batch", "dt", dt)?;
    expect_same_len(
        "rk4_j2_step_batch",
        &[
            ("rx", rx.len()),
            ("ry", ry.len()),
            ("rz", rz.len()),
            ("vx", vx.len()),
            ("vy", vy.len()),
            ("vz", vz.len()),
        ],
    )?;
    crate::ops::rk4_j2_step_batch::rk4_j2_step_batch(rx, ry, rz, vx, vy, vz, mu, j2, re, dt);
    Ok(())
}

/* ==================== 批量姿态与几何（f64 SOA） ==================== */

/// 批量三维叉积 `o = a × b`，返回 `[ox, oy, oz]`。
///
/// # Errors
/// 六个数组长度不一致时返回 [`Error::Shape`]。
pub fn cross3_batch(
    ax: &[f64],
    ay: &[f64],
    az: &[f64],
    bx: &[f64],
    by: &[f64],
    bz: &[f64],
) -> Result<[AlignedVec<f64>; 3]> {
    let n = expect_same_len(
        "cross3_batch",
        &[
            ("ax", ax.len()),
            ("ay", ay.len()),
            ("az", az.len()),
            ("bx", bx.len()),
            ("by", by.len()),
            ("bz", bz.len()),
        ],
    )?;
    let (mut ox, mut oy, mut oz) = alloc3(n);
    crate::ops::cross3_batch::cross3_batch(ax, ay, az, bx, by, bz, &mut ox, &mut oy, &mut oz);
    Ok([ox, oy, oz])
}

/// 批量三维单位化 `o = v/|v|`（零向量 → `(0,0,0)`），返回 `[ox, oy, oz]`。
///
/// # Errors
/// 三个数组长度不一致时返回 [`Error::Shape`]。
pub fn unitize3_batch(x: &[f64], y: &[f64], z: &[f64]) -> Result<[AlignedVec<f64>; 3]> {
    let n = expect_same_len(
        "unitize3_batch",
        &[("x", x.len()), ("y", y.len()), ("z", z.len())],
    )?;
    let (mut ox, mut oy, mut oz) = alloc3(n);
    crate::ops::unitize3_batch::unitize3_batch(x, y, z, &mut ox, &mut oy, &mut oz);
    Ok([ox, oy, oz])
}

/// 批量 `o = M·v`（`m` 为行主序 3×3），返回 `[ox, oy, oz]`。
///
/// # Errors
/// 9 个矩阵数组与 `x`/`y`/`z` 长度不一致时返回 [`Error::Shape`]。
pub fn mat3_mul_vec3_batch(
    m: [&[f64]; 9],
    x: &[f64],
    y: &[f64],
    z: &[f64],
) -> Result<[AlignedVec<f64>; 3]> {
    let names: [&'static str; 12] = [
        "m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "x", "y", "z",
    ];
    let mut lens = [("", 0usize); 12];
    for (k, name) in names.iter().enumerate() {
        lens[k] = (
            name,
            if k < 9 {
                m[k].len()
            } else {
                [x, y, z][k - 9].len()
            },
        );
    }
    let n = expect_same_len("mat3_mul_vec3_batch", &lens)?;
    let (mut ox, mut oy, mut oz) = alloc3(n);
    crate::ops::mat3_mul_vec3_batch::mat3_mul_vec3_batch(m, x, y, z, &mut ox, &mut oy, &mut oz);
    Ok([ox, oy, oz])
}

/// 批量单位化四元数（标量在前），返回 `[qw, qx, qy, qz]`。
///
/// `|q| < 1e-15` 的样本输出单位四元数 `(1,0,0,0)`。
///
/// # Errors
/// 四个数组长度不一致时返回 [`Error::Shape`]。
pub fn quat_normalize_batch(
    qw: &[f64],
    qx: &[f64],
    qy: &[f64],
    qz: &[f64],
) -> Result<[AlignedVec<f64>; 4]> {
    let n = expect_same_len(
        "quat_normalize_batch",
        &[
            ("qw", qw.len()),
            ("qx", qx.len()),
            ("qy", qy.len()),
            ("qz", qz.len()),
        ],
    )?;
    let mut q: [AlignedVec<f64>; 4] = std::array::from_fn(|_| AlignedVec::new(n));
    q[0].as_mut_slice().copy_from_slice(qw);
    q[1].as_mut_slice().copy_from_slice(qx);
    q[2].as_mut_slice().copy_from_slice(qy);
    q[3].as_mut_slice().copy_from_slice(qz);
    {
        let [qw, qx, qy, qz] = &mut q;
        crate::ops::quat_normalize_batch::quat_normalize_batch(qw, qx, qy, qz);
    }
    Ok(q)
}

/// 批量化四元数乘法 `a ⊗ b`（Hamilton 积，标量在前），返回 `[ow, ox, oy, oz]`。
///
/// # Errors
/// 八个数组长度不一致时返回 [`Error::Shape`]。
pub fn quat_mul_batch(a: [&[f64]; 4], b: [&[f64]; 4]) -> Result<[AlignedVec<f64>; 4]> {
    let names: [&'static str; 8] = ["aw", "ax", "ay", "az", "bw", "bx", "by", "bz"];
    let mut lens = [("", 0usize); 8];
    for (k, name) in names.iter().enumerate() {
        lens[k] = (name, if k < 4 { a[k].len() } else { b[k - 4].len() });
    }
    let n = expect_same_len("quat_mul_batch", &lens)?;
    let mut o: [AlignedVec<f64>; 4] = std::array::from_fn(|_| AlignedVec::new(n));
    {
        let [ow, ox, oy, oz] = &mut o;
        crate::ops::quat_mul_batch::quat_mul_batch(
            a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], ow, ox, oy, oz,
        );
    }
    Ok(o)
}

/// 批量用四元数旋转向量（先单位化，再 `o = R(q)·v`），返回 `[ox, oy, oz]`。
///
/// # Errors
/// 四元数与向量各数组长度不一致时返回 [`Error::Shape`]。
pub fn quat_rotate_batch(q: [&[f64]; 4], v: [&[f64]; 3]) -> Result<[AlignedVec<f64>; 3]> {
    let names: [&'static str; 7] = ["qw", "qx", "qy", "qz", "vx", "vy", "vz"];
    let mut lens = [("", 0usize); 7];
    for (k, name) in names.iter().enumerate() {
        lens[k] = (name, if k < 4 { q[k].len() } else { v[k - 4].len() });
    }
    let n = expect_same_len("quat_rotate_batch", &lens)?;
    let (mut ox, mut oy, mut oz) = alloc3(n);
    crate::ops::quat_rotate_batch::quat_rotate_batch(
        q[0], q[1], q[2], q[3], v[0], v[1], v[2], &mut ox, &mut oy, &mut oz,
    );
    Ok([ox, oy, oz])
}

/// 批量四元数 → 3×3 方向余弦阵（行主序），返回 `[m0..m8]`。
///
/// # Errors
/// 四个分量数组长度不一致时返回 [`Error::Shape`]。
pub fn quat_to_dcm_batch(q: [&[f64]; 4]) -> Result<[AlignedVec<f64>; 9]> {
    let n = expect_same_len(
        "quat_to_dcm_batch",
        &[
            ("qw", q[0].len()),
            ("qx", q[1].len()),
            ("qy", q[2].len()),
            ("qz", q[3].len()),
        ],
    )?;
    let mut m: [AlignedVec<f64>; 9] = std::array::from_fn(|_| AlignedVec::new(n));
    {
        let [m0, m1, m2, m3, m4, m5, m6, m7, m8] = &mut m;
        let refs: [&mut [f64]; 9] = [
            m0.as_mut_slice(),
            m1.as_mut_slice(),
            m2.as_mut_slice(),
            m3.as_mut_slice(),
            m4.as_mut_slice(),
            m5.as_mut_slice(),
            m6.as_mut_slice(),
            m7.as_mut_slice(),
            m8.as_mut_slice(),
        ];
        crate::ops::quat_to_dcm_batch::quat_to_dcm_batch(q[0], q[1], q[2], q[3], refs);
    }
    Ok(m)
}

/// 三个等长对齐缓冲。
fn alloc3(n: usize) -> (AlignedVec<f64>, AlignedVec<f64>, AlignedVec<f64>) {
    (AlignedVec::new(n), AlignedVec::new(n), AlignedVec::new(n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aligned::ALIGN;

    /// `api` 只是"切片 + 校验"的薄包装，数值必须与 C ABI 路径逐位一致。
    #[test]
    fn test_reduce_matches_c_abi() {
        let a: Vec<f32> = (0..1000).map(|i| (i % 37) as f32 - 18.0).collect();
        let b: Vec<f32> = (0..1000).map(|i| (i % 23) as f32 * 0.5).collect();
        let n = a.len() as i32;
        assert_eq!(sum(&a), crate::lasx_sum(a.as_ptr(), n));
        assert_eq!(
            dot(&a, &b).unwrap(),
            crate::lasx_dot(a.as_ptr(), b.as_ptr(), n)
        );

        let a64: Vec<f64> = a.iter().map(|&v| v as f64).collect();
        let b64: Vec<f64> = b.iter().map(|&v| v as f64).collect();
        assert_eq!(
            dot_f64(&a64, &b64).unwrap(),
            crate::lasx_dot_f64(a64.as_ptr(), b64.as_ptr(), n)
        );

        let ai: Vec<i8> = (0..1000).map(|i| (i % 255) as i8).collect();
        let bi: Vec<i8> = (0..1000).map(|i| (i % 91) as i8).collect();
        assert_eq!(
            dot_i8(&ai, &bi).unwrap(),
            crate::lasx_dot_i8(ai.as_ptr(), bi.as_ptr(), n)
        );
    }

    /// 空输入与单元素输入不该 panic。
    #[test]
    fn test_empty_and_single() {
        assert_eq!(sum(&[]), 0.0);
        assert_eq!(dot(&[], &[]).unwrap(), 0.0);
        assert_eq!(dot(&[2.0], &[3.0]).unwrap(), 6.0);
        assert_eq!(norm3_batch(&[], &[], &[]).unwrap().len(), 0);
        assert!(matmul(0, 0, 0, &[], &[]).unwrap().is_empty());
    }

    /// 长度不符必须是 `Err`，且错误里带上算子名、参数名、期望与实际。
    #[test]
    fn test_shape_errors() {
        let e = dot(&[1.0, 2.0], &[1.0]).unwrap_err();
        assert_eq!(
            e,
            Error::Shape {
                op: "dot",
                what: "b",
                expected: 2,
                got: 1
            }
        );
        assert_eq!(e.to_string(), "dot: 参数 b 的长度应为 2，实际 1");

        let mut y = [0.0f32; 2];
        assert!(axpy(1.0, &[1.0], &mut y).is_err());
        assert!(norm3_batch(&[1.0], &[1.0], &[]).is_err());
        assert!(batch_distance2d(0.0, 0.0, &[1.0], &[]).is_err());
        assert!(matches!(
            matmul(2, 2, 2, &[0.0; 3], &[0.0; 4]),
            Err(Error::Shape { what: "a", .. })
        ));
    }

    /// `dot_q4` 的 scale 数组按"≥ 组数"约定（见 manual §9.9）。
    #[test]
    fn test_q4_scale_length() {
        let qa = vec![0x21u8; 64]; // 2 组
        let qb = vec![0x11u8; 64];
        let sa = vec![1.0f32; 2];
        let sb = vec![1.0f32; 2];
        assert!(dot_q4(&qa, &sa, &qb, &sb).is_ok());
        // 少一组 scale：报错，且指出期望 2
        let err = dot_q4(&qa, &sa[..1], &qb, &sb).unwrap_err();
        assert_eq!(
            err,
            Error::Shape {
                op: "dot_q4",
                what: "sa",
                expected: 2,
                got: 1
            }
        );
        // 多给一组 scale 是合法的（文档约定是 ≥，不是 =）
        assert!(dot_q4(&qa, &[1.0f32; 8], &qb, &sb).is_ok());
        assert!(dot_q4(&qa, &sa, &qb[..32], &sb).is_err());
    }

    /// 物理常数非法必须是 `Err`，且与 `*_checked` 的规则一致。
    #[test]
    fn test_constant_errors() {
        let mut rx = vec![7.0e6f64; 4];
        let mut ry = vec![0.0f64; 4];
        let mut rz = vec![0.0f64; 4];
        let mut vx = vec![0.0f64; 4];
        let mut vy = vec![7.5e3f64; 4];
        let mut vz = vec![0.0f64; 4];
        let step =
            |rx: &mut [f64],
             ry: &mut [f64],
             rz: &mut [f64],
             vx: &mut [f64],
             vy: &mut [f64],
             vz: &mut [f64],
             mu: f64,
             j2: f64,
             re: f64,
             dt: f64| { rk4_j2_step_batch(rx, ry, rz, vx, vy, vz, mu, j2, re, dt) };
        // mu <= 0（与 lasx_*_checked 同为 NotPositive）
        assert!(matches!(
            step(&mut rx, &mut ry, &mut rz, &mut vx, &mut vy, &mut vz, 0.0, 1e-3, 6.4e6, 10.0),
            Err(Error::NotPositive { what: "mu", .. })
        ));
        // dt 非有限
        assert!(matches!(
            step(
                &mut rx,
                &mut ry,
                &mut rz,
                &mut vx,
                &mut vy,
                &mut vz,
                3.99e14,
                1e-3,
                6.4e6,
                f64::NAN
            ),
            Err(Error::NotFinite { what: "dt", .. })
        ));
        // j2 非有限
        assert!(matches!(
            step(
                &mut rx,
                &mut ry,
                &mut rz,
                &mut vx,
                &mut vy,
                &mut vz,
                3.99e14,
                f64::INFINITY,
                6.4e6,
                10.0
            ),
            Err(Error::NotFinite { what: "j2", .. })
        ));
        assert!(matches!(
            j2_accel_batch(&rx, &ry, &rz, 3.99e14, 1e-3, 0.0),
            Err(Error::NotPositive { what: "re", .. })
        ));
        // 检查顺序与 checked 版一致：常数先判，形状后判
        assert!(matches!(
            j2_accel_batch(&rx, &ry, &rz[..3], 0.0, 1e-3, 6.4e6),
            Err(Error::NotPositive { what: "mu", .. })
        ));
        assert!(matches!(
            j2_accel_batch(&rx, &ry, &rz[..3], 3.99e14, 1e-3, 6.4e6),
            Err(Error::Shape {
                op: "j2_accel_batch",
                ..
            })
        ));
    }

    /// 形状相乘溢出要报 `Overflow` 而不是 panic 或巨量分配。
    #[test]
    fn test_overflow() {
        let e = matmul(usize::MAX, 2, 1, &[], &[]).unwrap_err();
        assert_eq!(
            e,
            Error::Overflow {
                op: "matmul",
                what: "m×k"
            }
        );
        assert_eq!(e.to_string(), "matmul: m×k 溢出");
    }

    /// 需要输出的内核返回的缓冲必须落在对齐窗口上——这正是本层存在的理由之一。
    #[test]
    fn test_outputs_are_aligned() {
        let xs = vec![3.0f64; 1000];
        let ys = vec![4.0f64; 1000];
        let zs = vec![0.0f64; 1000];
        let out = norm3_batch(&xs, &ys, &zs).unwrap();
        assert_eq!(
            out.as_ptr() as usize % ALIGN,
            0,
            "norm3 输出未按 {ALIGN} 对齐"
        );
        assert!(out.iter().all(|&v| v == 5.0));

        let m = matmul(64, 64, 64, &vec![1.0f32; 64 * 64], &vec![1.0f32; 64 * 64]).unwrap();
        assert_eq!(m.as_ptr() as usize % ALIGN, 0, "matmul 输出未对齐");
        assert!(m.iter().all(|&v| v == 64.0));

        let d = batch_distance2d(0.0, 0.0, &[3.0f32; 100], &[4.0f32; 100]).unwrap();
        assert_eq!(d.as_ptr() as usize % ALIGN, 0);
        assert!(d.iter().all(|&v| v == 5.0));

        let [ox, oy, oz] = vec3_add_scaled_batch(&xs, &ys, &zs, &xs, &ys, &zs, 1.0).unwrap();
        for a in [&ox, &oy, &oz] {
            assert_eq!(a.as_ptr() as usize % ALIGN, 0);
        }
        assert_eq!((ox[0], oy[0], oz[0]), (6.0, 8.0, 0.0));

        let [ax, ay, az] = j2_accel_batch(&xs, &ys, &zs, 3.986e14, 1.0826e-3, 6.378e6).unwrap();
        assert!((ax.as_ptr() as usize).is_multiple_of(ALIGN) && ax[0] < 0.0 && az[0] == 0.0);
        assert_eq!(ay.len(), 1000);
    }

    /// 逐元素激活：与 C ABI 路径逐位一致、输出对齐、空输入不 panic。
    #[test]
    fn test_activations_match_c_abi() {
        let n = 135usize; // 不是 8 的倍数：跨过"向量主体 + 标量尾"的边界
        let x: Vec<f32> = (0..n).map(|i| (i as f32 - 67.0) * 0.31).collect();

        let y = silu(&x);
        assert_eq!(y.as_ptr() as usize % ALIGN, 0, "silu 输出未按 {ALIGN} 对齐");
        let mut want = vec![0f32; n];
        crate::lasx_silu(x.as_ptr(), want.as_mut_ptr(), n as i32);
        for i in 0..n {
            assert_eq!(y[i].to_bits(), want[i].to_bits(), "silu[{i}]");
        }

        let g = gelu_quick(&x);
        assert_eq!(g.as_ptr() as usize % ALIGN, 0, "gelu 输出未对齐");
        crate::lasx_gelu_quick(x.as_ptr(), want.as_mut_ptr(), n as i32);
        for i in 0..n {
            assert_eq!(g[i].to_bits(), want[i].to_bits(), "gelu_quick[{i}]");
        }

        let e = gelu_erf(&x);
        assert_eq!(e.as_ptr() as usize % ALIGN, 0, "gelu_erf 输出未对齐");
        crate::lasx_gelu_erf(x.as_ptr(), want.as_mut_ptr(), n as i32);
        for i in 0..n {
            assert_eq!(e[i].to_bits(), want[i].to_bits(), "gelu_erf[{i}]");
        }

        // 空输入：长度 0、不 panic
        assert!(silu(&[]).is_empty());
        assert!(gelu_quick(&[]).is_empty());
        assert!(gelu_erf(&[]).is_empty());
    }

    /// RoPE：与 C ABI 逐位一致、输出对齐、`rope_at` == `rope_tables` + `rope`；
    /// 错误路径（`n_dims` 奇数 / 超列 / 表长度不符 / `freq_base` 非法）都给 `Err`。
    #[test]
    fn test_rope_matches_c_abi_and_errors() {
        let (rows, cols, n_dims) = (5usize, 128usize, 128usize);
        let x: Vec<f32> = (0..rows * cols)
            .map(|k| (k as f32).mul_add(0.017, -1.0))
            .collect();
        let positions: Vec<f32> = (0..rows).map(|r| r as f32 * 1.5).collect();
        let (cos, sin) = rope_tables(&positions, n_dims, 10_000.0).unwrap();
        assert_eq!(cos.len(), rows * n_dims / 2);
        assert_eq!(cos.as_ptr() as usize % ALIGN, 0, "表未对齐");

        for mode in [RopeMode::NeoX, RopeMode::GptJ] {
            let y = rope(&x, &cos, &sin, rows, cols, n_dims, mode).unwrap();
            assert_eq!(y.as_ptr() as usize % ALIGN, 0, "输出未对齐");
            let mut want = vec![0f32; x.len()];
            crate::lasx_rope(
                x.as_ptr(),
                cos.as_ptr(),
                sin.as_ptr(),
                want.as_mut_ptr(),
                rows as i32,
                cols as i32,
                n_dims as i32,
                mode.as_i32(),
            );
            for k in 0..x.len() {
                assert_eq!(y[k].to_bits(), want[k].to_bits(), "{mode:?} k={k}");
            }
            // 便捷入口与"先建表再旋转"逐位一致
            let z = rope_at(&x, &positions, rows, cols, n_dims, 10_000.0, mode).unwrap();
            for k in 0..x.len() {
                assert_eq!(z[k].to_bits(), y[k].to_bits(), "{mode:?} rope_at k={k}");
            }
        }

        // 错误路径
        assert!(matches!(
            rope(&x, &cos, &sin, rows, cols, 7, RopeMode::NeoX),
            Err(Error::BadValue { what: "n_dims", .. })
        ));
        assert!(matches!(
            rope(&x, &cos, &sin, rows, cols, cols + 2, RopeMode::NeoX),
            Err(Error::Shape { what: "n_dims", .. })
        ));
        assert!(matches!(
            rope(&x, &cos[..3], &sin, rows, cols, n_dims, RopeMode::NeoX),
            Err(Error::Shape { what: "cos", .. })
        ));
        assert!(matches!(
            rope_tables(&positions, 0, 10_000.0),
            Err(Error::BadValue { what: "n_dims", .. })
        ));
        assert!(matches!(
            rope_tables(&positions, n_dims, 0.0),
            Err(Error::NotPositive {
                what: "freq_base",
                ..
            })
        ));
        assert!(matches!(
            rope_tables(&positions, n_dims, f32::NAN),
            Err(Error::NotFinite {
                what: "freq_base",
                ..
            })
        ));
        // 空输入
        assert!(rope(&[], &[], &[], 0, 8, 8, RopeMode::NeoX)
            .unwrap()
            .is_empty());
    }

    /// 原地内核：结果与 C ABI 路径逐位一致，且原地语义正确（axpy 不改 x）。
    #[test]
    fn test_in_place_kernels_match_c_abi() {
        let n = 777usize;
        let x = vec![1.0f32; n];
        let mut y = vec![2.0f32; n];
        axpy(-0.5, &x, &mut y).unwrap();
        assert!(y.iter().all(|&v| v == 1.5));
        assert!(x.iter().all(|&v| v == 1.0), "axpy 不该改 x");

        // 弹道步：6 个状态数组 + k，C ABI 与 api 各跑一遍（各自独立的缓冲）
        let (dt, g) = (0.01f32, -9.8f32);
        let mk = || {
            [
                vec![1.0f32; n],
                vec![2.0f32; n],
                vec![3.0f32; n],
                vec![4.0f32; n],
                vec![5.0f32; n],
                vec![6.0f32; n],
            ]
        };
        let k = vec![0.1f32; n];

        let mut want = mk();
        crate::lasx_ballistic_step(
            want[0].as_mut_ptr(),
            want[1].as_mut_ptr(),
            want[2].as_mut_ptr(),
            want[3].as_mut_ptr(),
            want[4].as_mut_ptr(),
            want[5].as_mut_ptr(),
            k.as_ptr(),
            n as i32,
            dt,
            g,
        );

        let mut got = mk();
        {
            // 一次解构出 6 个独立可变借用——索引写法过不了借用检查
            let [x, y, z, vx, vy, vz] = &mut got;
            ballistic_step(x, y, z, vx, vy, vz, &k, dt, g).unwrap();
        }
        for i in 0..6 {
            assert_eq!(got[i], want[i], "第 {i} 个状态数组与 C ABI 不一致");
        }
        {
            let [x, y, z, vx, vy, vz] = &mut got;
            assert!(matches!(
                ballistic_step(x, y, z, vx, vy, vz, &k[..3], dt, g),
                Err(Error::Shape {
                    op: "ballistic_step",
                    what: "k",
                    ..
                })
            ));
        }
    }

    /// 多步传播：`api` 与 `parallel` 必须逐位一致（切块不改结果）。
    #[test]
    fn test_rk4_matches_parallel_bit_for_bit() {
        let n = 20_000;
        let mk = || {
            (
                vec![7.0e6f64; n],
                vec![3.0e5f64; n],
                vec![1.0e5f64; n],
                vec![100.0f64; n],
                vec![7.5e3f64; n],
                vec![-50.0f64; n],
            )
        };
        let (mu, j2, re, dt) = (3.986_004_418e14, 1.082_626_68e-3, 6.378_137e6, 10.0);

        let (mut rx, mut ry, mut rz, mut vx, mut vy, mut vz) = mk();
        for _ in 0..5 {
            rk4_j2_step_batch(
                &mut rx, &mut ry, &mut rz, &mut vx, &mut vy, &mut vz, mu, j2, re, dt,
            )
            .unwrap();
        }

        let (mut px, mut py, mut pz, mut qx, mut qy, mut qz) = mk();
        let pool = crate::pool::WorkerPool::new(6);
        for _ in 0..5 {
            crate::parallel::rk4_j2_step_batch(
                &pool, mu, j2, re, dt, &mut px, &mut py, &mut pz, &mut qx, &mut qy, &mut qz,
            )
            .unwrap();
        }

        for i in 0..n {
            assert_eq!(rx[i].to_bits(), px[i].to_bits(), "rx 不一致 @ {i}");
            assert_eq!(vz[i].to_bits(), qz[i].to_bits(), "vz 不一致 @ {i}");
        }
    }
}
