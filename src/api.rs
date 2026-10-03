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
//! LASX 是 32 字节访存，缓冲区是否对齐直接影响性能（L1 驻留规模上实测 1.06–1.31×，
//! 大集下无关，见 `docs/dev.md` §7.6），
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

/// [`rms_norm`] 的**免分配**版本：结果写进调用方给的 `out`（长度必须等于 `rows × cols`）。
///
/// 为什么有它：`rms_norm` 每次调用都分配一个 `AlignedVec`。对"逐层保活"这种调用
/// （int8 模型每层都要"反量化 → 归一化 → 再量化"），分配会进到每层的关键路径上
/// （`docs/dev.md` §7.9 末表的保活口径里，`192×768` 的 RMSNorm 是 181.0 µs，含分配）。
/// 数值与 [`rms_norm`] **逐位一致**（同一个内核，只少一次分配；测试守着）。
///
/// # Errors
/// 与 [`rms_norm`] 同一套，外加 `out.len() != rows × cols`。
pub fn rms_norm_into(
    x: &[f32],
    w: Option<&[f32]>,
    rows: usize,
    cols: usize,
    eps: f32,
    out: &mut [f32],
) -> Result<()> {
    if !eps.is_finite() {
        return Err(Error::NotFinite {
            op: "rms_norm_into",
            what: "eps",
            value: eps as f64,
        });
    }
    if eps <= 0.0 {
        return Err(Error::NotPositive {
            op: "rms_norm_into",
            what: "eps",
            value: eps as f64,
        });
    }
    let n = checked_mul("rms_norm_into", "rows×cols", rows, cols)?;
    expect_len("rms_norm_into", "x", x.len(), n)?;
    expect_len("rms_norm_into", "out", out.len(), n)?;
    if let Some(w) = w {
        expect_len("rms_norm_into", "w", w.len(), cols)?;
    }
    if n == 0 {
        return Ok(());
    }
    crate::ops::rms_norm::rms_norm(x, w.unwrap_or(&[]), eps, rows, cols, out);
    Ok(())
}

/// [`softmax_rows`] 的**免分配**版本：结果写进调用方给的 `out`（长度必须等于 `rows × cols`）。
///
/// 与 [`softmax_rows`] **逐位一致**（同一个内核，只少一次分配）；校验口径相同，
/// 外加 `out.len() != rows × cols`。见 `docs/ops.md` §5.15 里"先做 `_into`"那条。
///
/// # Errors
/// 与 [`softmax_rows`] 同一套，外加 `out.len() != rows × cols`。
pub fn softmax_rows_into(
    x: &[f32],
    mask: Option<&[f32]>,
    rows: usize,
    cols: usize,
    scale: f32,
    out: &mut [f32],
) -> Result<()> {
    if !scale.is_finite() {
        return Err(Error::NotFinite {
            op: "softmax_rows_into",
            what: "scale",
            value: scale as f64,
        });
    }
    let n = checked_mul("softmax_rows_into", "rows×cols", rows, cols)?;
    expect_len("softmax_rows_into", "x", x.len(), n)?;
    expect_len("softmax_rows_into", "out", out.len(), n)?;
    if let Some(m) = mask {
        expect_len("softmax_rows_into", "mask", m.len(), n)?;
    }
    if n == 0 {
        return Ok(());
    }
    crate::ops::softmax_rows::softmax_rows(x, mask.unwrap_or(&[]), scale, rows, cols, out);
    Ok(())
}

/// 行内 **LayerNorm**：`out[i][j] = (x[i][j] − mean_i)/√(var_i + eps) · w[j] + b[j]`（行主序）。
///
/// 与 [`rms_norm`] 的差别只有**减均值**与**偏置**：契约（`docs/ops.md` §2.16）要求方差用
/// **两遍法**（先 mean、再 `Σ(x−mean)²`），不是 `E[x²] − mean²`；`w`/`b` 传 `None` 表示
/// 无权重/无偏置（**无偏置是跳过加法**，不是加 `0.0`——那会翻 `−0.0` 的符号）。
///
/// 存在理由：外部契约（ONNX 图 17 处 `LayerNormalization`，见 `docs/platform.md` §4）。
///
/// # Errors
/// - `x.len() != rows × cols` 或 `out.len() != rows × cols`（[`Error::Shape`]）；
/// - `w`/`b` 给了但长度不是 `cols`（[`Error::Shape`]）；
/// - `eps` 非有限（[`Error::NotFinite`]）或非正（[`Error::NotPositive`]）；
/// - `rows × cols` 溢出（[`Error::Overflow`]）。
pub fn layer_norm(
    x: &[f32],
    w: Option<&[f32]>,
    b: Option<&[f32]>,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<AlignedVec<f32>> {
    let n = checked_mul("layer_norm", "rows×cols", rows, cols)?;
    let mut out = AlignedVec::<f32>::new(n);
    layer_norm_into(x, w, b, rows, cols, eps, out.as_mut_slice())?;
    Ok(out)
}

/// [`layer_norm`] 的**免分配**版本（与分配版**逐位一致**，见 `docs/dev.md` §7.9 的 `_into` 实测）。
///
/// # Errors
/// 同 [`layer_norm`]。
pub fn layer_norm_into(
    x: &[f32],
    w: Option<&[f32]>,
    b: Option<&[f32]>,
    rows: usize,
    cols: usize,
    eps: f32,
    out: &mut [f32],
) -> Result<()> {
    if !eps.is_finite() {
        return Err(Error::NotFinite {
            op: "layer_norm",
            what: "eps",
            value: eps as f64,
        });
    }
    if eps <= 0.0 {
        return Err(Error::NotPositive {
            op: "layer_norm",
            what: "eps",
            value: eps as f64,
        });
    }
    let n = checked_mul("layer_norm", "rows×cols", rows, cols)?;
    expect_len("layer_norm", "x", x.len(), n)?;
    expect_len("layer_norm", "out", out.len(), n)?;
    if let Some(w) = w {
        expect_len("layer_norm", "w", w.len(), cols)?;
    }
    if let Some(b) = b {
        expect_len("layer_norm", "b", b.len(), cols)?;
    }
    if n == 0 {
        return Ok(());
    }
    crate::ops::layer_norm::layer_norm(x, w.unwrap_or(&[]), b.unwrap_or(&[]), eps, rows, cols, out);
    Ok(())
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

/* ==================== NN：f16 权重 ==================== */

/// f16 权重与 f32 向量的点积：`Σ f16_to_f32(a[i]) · b[i]`（`a` 是 **u16 位型**）。
///
/// # 数值契约（见 `docs/ops.md` §2.11）
/// - f16→f32 是**精确**转换（测试穷举全部 65536 个 f16 位型对硬件验证）；误差只来自
///   "权重本来是 f16"这件事本身（相对 ~4.9e-4）；
/// - **4 条独立累加链**、每 32 元素一轮（不足 32 补一轮 16 元素、再不足走标量尾），
///   元素下标 `j` 始终落在 lane `j % 8`；最后按固定次序 `(c0+c1)+(c2+c3)` 合并、再两两归约
///   ⇒ 向量与标量模拟逐位相同；
/// - `n = 0` ⇒ `0.0`；`NaN` 的 payload 不在逐位承诺内。
///
/// # Errors
/// `a.len() != b.len()`（[`Error::Shape`]）。
pub fn dot_f16(a: &[u16], b: &[f32]) -> Result<f32> {
    expect_len("dot_f16", "b", b.len(), a.len())?;
    // SAFETY: 上面刚校验等长。
    Ok(unsafe { crate::ops::dot_f16::dot_f16(a, b) })
}

/// f16 权重矩阵（`m × k` 行主序）× f32 向量：`y[r] = dot_f16(a[r*k..(r+1)*k], x)`。
///
/// 每行独立 ⇒ `y[r]` 与单独调 [`dot_f16`] **逐位相同**；`k = 0` 时 `y` 全 `0`
/// （空向量的点积是 0，不是"未定义"）。
///
/// # Errors
/// `a.len() != m × k`、`x.len() != k`（[`Error::Shape`]）；`m × k` 溢出（[`Error::Overflow`]）。
pub fn gemv_f16(m: usize, k: usize, a: &[u16], x: &[f32]) -> Result<AlignedVec<f32>> {
    expect_len(
        "gemv_f16",
        "a",
        a.len(),
        checked_mul("gemv_f16", "m×k", m, k)?,
    )?;
    expect_len("gemv_f16", "x", x.len(), k)?;
    let mut y = AlignedVec::<f32>::new(m);
    if m == 0 {
        return Ok(y);
    }
    if k == 0 {
        y.as_mut_slice().fill(0.0);
        return Ok(y);
    }
    // SAFETY: 长度自洽（m×k、k、m）。
    unsafe { crate::ops::dot_f16::gemv_f16(a, x, m, k, y.as_mut_slice()) };
    Ok(y)
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

/* ==================== N3：int8 推理（量化生产端 + GEMV） ==================== */

/// 最大绝对值（逐张量）：`max_i |x_i|`。
///
/// 契约（`docs/ops.md` §2.12）：与 `f32::max` 折叠同语义——**NaN 被忽略**，全 NaN 输入得
/// `0.0`；空输入得 `0.0`。`±inf` 不在契约内。
pub fn amax(x: &[f32]) -> Result<f32> {
    Ok(crate::ops::quant_i8::amax(x))
}

/// 逐行最大绝对值：`out[r] = max_j |x[r·cols + j]|`（`rows × cols` 行主序）。
///
/// per-token 激活量化与 per-channel 权重量化共用这一条（都是行主序按行归约）。
///
/// # Errors
/// - `x.len() != rows × cols`（[`Error::Shape`]）；
/// - `rows × cols` 溢出 `usize`（[`Error::Overflow`]）。
pub fn absmax_rows(x: &[f32], rows: usize, cols: usize) -> Result<AlignedVec<f32>> {
    let n = checked_mul("absmax_rows", "rows×cols", rows, cols)?;
    expect_len("absmax_rows", "x", x.len(), n)?;
    let mut out = AlignedVec::<f32>::new(rows);
    if rows == 0 || cols == 0 {
        return Ok(out);
    }
    crate::ops::quant_i8::absmax_rows(x, rows, cols, out.as_mut_slice());
    Ok(out)
}

/// 逐张量对称量化：`q = round_ties_even(x · (1/scale))`，返回 `(q, scale)`。
///
/// 契约（`docs/ops.md` §2.12）：`scale = max|x| / 127`；`q ∈ [−127, 127]`（**不使用 −128**）；
/// 取整是 **ties-to-even**；用**倒数乘**而不是除法（LASX 无向量除法）；全零（或全 NaN）输入
/// ⇒ `scale = 0` 且 `q` 全 0。
///
/// 量化误差 ≤ `scale/2`；`dequantize_i8` 的结果是 `scale` 的**精确整数倍**。
pub fn quantize_i8_per_tensor(x: &[f32]) -> Result<(AlignedVec<i8>, f32)> {
    let mut q = AlignedVec::<i8>::new(x.len());
    let scale = crate::ops::quant_i8::quantize_i8_per_tensor(x, q.as_mut_slice());
    Ok((q, scale))
}

/// 逐 **token** 激活量化（`[token, hidden]` 行主序）：每行一个 `scale`。
///
/// 与 [`quantize_i8_per_channel`] 是**同一个内核**、不同的语义名字——行主序下"按行归约"
/// 对激活就是 per-token，对权重就是 per-output-channel。分开命名是为了让调用点自证意图。
///
/// # Errors
/// - `x.len() != rows × cols`（[`Error::Shape`]）；
/// - `rows × cols` 溢出（[`Error::Overflow`]）。
pub fn quantize_i8_per_token(
    x: &[f32],
    rows: usize,
    cols: usize,
) -> Result<(AlignedVec<i8>, AlignedVec<f32>)> {
    quantize_i8_rows_impl("quantize_i8_per_token", x, rows, cols)
}

/// 逐**输出通道**权重量化（`[out_features, in_features]` 行主序）：每行一个 `scale`。
///
/// # Errors
/// 同 [`quantize_i8_per_token`]。
pub fn quantize_i8_per_channel(
    w: &[f32],
    rows: usize,
    cols: usize,
) -> Result<(AlignedVec<i8>, AlignedVec<f32>)> {
    quantize_i8_rows_impl("quantize_i8_per_channel", w, rows, cols)
}

/// [`quantize_i8_per_token`] / [`quantize_i8_per_channel`] 的公共实现（同一内核）。
fn quantize_i8_rows_impl(
    op: &'static str,
    x: &[f32],
    rows: usize,
    cols: usize,
) -> Result<(AlignedVec<i8>, AlignedVec<f32>)> {
    let n = checked_mul(op, "rows×cols", rows, cols)?;
    expect_len(op, "x", x.len(), n)?;
    let mut q = AlignedVec::<i8>::new(n);
    let mut scales = AlignedVec::<f32>::new(rows);
    if rows == 0 || cols == 0 {
        return Ok((q, scales));
    }
    crate::ops::quant_i8::quantize_i8_per_row(
        x,
        rows,
        cols,
        q.as_mut_slice(),
        scales.as_mut_slice(),
    );
    Ok((q, scales))
}

/// 逐张量反量化：`out[i] = (q[i] as f32) · scale`。
///
/// 与 [`quantize_i8_per_tensor`] 互为逆：结果逐位等于 `(q as f32) · scale`（一次舍入），
/// 因而是 `scale` 的精确整数倍。
pub fn dequantize_i8(q: &[i8], scale: f32) -> Result<AlignedVec<f32>> {
    let mut out = AlignedVec::<f32>::new(q.len());
    crate::ops::quant_i8::dequantize_i8(q, scale, out.as_mut_slice());
    Ok(out)
}

/// 逐行反量化：每行用自己的 `scales[r]`。
///
/// # Errors
/// - `q.len() != rows × cols` 或 `scales.len() != rows`（[`Error::Shape`]）；
/// - `rows × cols` 溢出（[`Error::Overflow`]）。
pub fn dequantize_i8_rows(
    q: &[i8],
    rows: usize,
    cols: usize,
    scales: &[f32],
) -> Result<AlignedVec<f32>> {
    let n = checked_mul("dequantize_i8_rows", "rows×cols", rows, cols)?;
    expect_len("dequantize_i8_rows", "q", q.len(), n)?;
    expect_len("dequantize_i8_rows", "scales", scales.len(), rows)?;
    let mut out = AlignedVec::<f32>::new(n);
    if rows == 0 || cols == 0 {
        return Ok(out);
    }
    crate::ops::quant_i8::dequantize_i8_per_row(q, rows, cols, scales, out.as_mut_slice());
    Ok(out)
}

/// int8 权重 × int8 激活的矩阵-向量乘（N3 的消费主力）：
/// `y[o] = (Σ_i W[o,i]·x[i]) · (scale_w[o] · scale_x)`。
///
/// 契约（`docs/ops.md` §2.13）：累加是**整数精确**的（i16 乘积 → i32 累加 → i64 落盘，
/// 与 `dot_i8` 同一内核）；出口是**一次乘法**——先把两个 scale 乘成一个（一次舍入）再乘
/// `acc`，写成 `((acc as f32)·sw)·sx` 是两次舍入、不保证逐位一致。
///
/// # Errors
/// - `w.len() != m × k`、`x.len() != k`、`scale_w.len() != m`（[`Error::Shape`]）；
/// - `m × k` 溢出（[`Error::Overflow`]）。
pub fn gemv_i8(
    w: &[i8],
    scale_w: &[f32],
    x: &[i8],
    scale_x: f32,
    m: usize,
    k: usize,
) -> Result<AlignedVec<f32>> {
    let n = checked_mul("gemv_i8", "m×k", m, k)?;
    expect_len("gemv_i8", "w", w.len(), n)?;
    expect_len("gemv_i8", "x", x.len(), k)?;
    expect_len("gemv_i8", "scale_w", scale_w.len(), m)?;
    let mut y = AlignedVec::<f32>::new(m);
    if m == 0 || k == 0 {
        return Ok(y);
    }
    crate::ops::gemv_i8::gemv_i8(w, scale_w, x, scale_x, m, k, y.as_mut_slice());
    Ok(y)
}

/// 批量 int8 矩阵乘（prefill 形态）：`y[t,o] = (Σ_i X[t,i]·W[o,i]) · (sw[o]·sx[t])`。
///
/// `x` 是 `m × k`（每 **token** 一个 scale，`scale_x` 长 `m`），`w` 是 `n × k`
/// （每 **输出通道** 一个 scale，`scale_w` 长 `n`），`y` 是 `m × n`。
///
/// 契约（`docs/ops.md` §2.14）：每个输出元素是**一次 `dot_i8`**（整数精确、顺序无关），
/// 出口与 `gemv_i8` 完全同口径——先把两个 scale 乘成一个（一次舍入）再乘 `acc as f32`。
/// `m = 1` 时与 [`gemv_i8`] **逐位一致**（测试守着）。
///
/// 与 [`gemv_i8`] 的分工：`gemv_i8` 每算一个 token 重读一遍权重（带宽受限）；
/// 本函数让权重在 L2 里被 `m` 个 token 复用（`m = 192` 时权重流量从 192×118 MB 降到 118 MB）。
///
/// # Errors
/// - `x.len() != m × k`、`w.len() != n × k`、`scale_x.len() != m`、`scale_w.len() != n`
///   （[`Error::Shape`]）；
/// - `m × k` 或 `n × k` 溢出（[`Error::Overflow`]）。
pub fn matmul_i8(
    x: &[i8],
    w: &[i8],
    scale_w: &[f32],
    scale_x: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<AlignedVec<f32>> {
    let n_x = checked_mul("matmul_i8", "m×k", m, k)?;
    let n_w = checked_mul("matmul_i8", "n×k", n, k)?;
    expect_len("matmul_i8", "x", x.len(), n_x)?;
    expect_len("matmul_i8", "w", w.len(), n_w)?;
    expect_len("matmul_i8", "scale_x", scale_x.len(), m)?;
    expect_len("matmul_i8", "scale_w", scale_w.len(), n)?;
    let mut y = AlignedVec::<f32>::new(checked_mul("matmul_i8", "m×n", m, n)?);
    if m == 0 || k == 0 || n == 0 {
        return Ok(y);
    }
    crate::ops::matmul_i8::matmul_i8(x, w, scale_w, scale_x, m, k, n, y.as_mut_slice());
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aligned::ALIGN;

    /* ==================== N3：int8 推理 ==================== */

    /// **int8 端到端口径的实测**（`docs/platform.md` §5.1：模型级验收标准还缺依据）。
    ///
    /// 口径：参考量用 **f64 累加**（既不是 f32 路径也不是 int8 路径）；
    /// 误差按 **∞-范数相对量** `max|Y_q − Y| / max|Y|` 报——这是"跨算子叠加后"的可复核口径。
    /// 同时报 f32 路径的同口径误差作对照（量化误差应当主导）。覆盖 per-token 激活 +
    /// per-channel 权重（`docs/dev.md` §21.3 的结论：这两个是前提而非优化项）。
    /// `_into` 版与分配版**逐位一致**（同一个内核、只少一次分配），错误路径打准。
    ///
    /// 动机：`docs/dev.md` §7.9 末表的保活口径里 RMSNorm 占 181.0 µs（含每次调用的分配），
    /// 而 `docs/ops.md` §5.15 指出的"要抠那 1.9% 先做 `_into`"就是这一条。
    #[test]
    fn test_nn_into_variants_match_allocating_bit_for_bit() {
        let (rows, cols) = (7usize, 33usize);
        let mut rng = crate::ops::testutil::Lcg(0x1111_2222);
        let x: Vec<f32> = (0..rows * cols).map(|_| 2.0 * rng.f64() as f32).collect();
        let w: Vec<f32> = (0..cols).map(|_| 0.5 + rng.f64() as f32).collect();
        let mask: Vec<f32> = (0..rows * cols).map(|_| 0.1 * rng.f64() as f32).collect();

        // rms_norm：带权重与不带权重两条
        for wopt in [Some(&w[..]), None] {
            let want = rms_norm(&x, wopt, rows, cols, 1e-5).unwrap();
            let mut got = vec![0f32; rows * cols];
            rms_norm_into(&x, wopt, rows, cols, 1e-5, &mut got).unwrap();
            for i in 0..rows * cols {
                assert_eq!(got[i].to_bits(), want[i].to_bits(), "rms_norm_into @ {i}");
            }
        }
        // softmax_rows：带 mask 与不带 mask 两条
        for mopt in [Some(&mask[..]), None] {
            let want = softmax_rows(&x, mopt, rows, cols, 1.0).unwrap();
            let mut got = vec![0f32; rows * cols];
            softmax_rows_into(&x, mopt, rows, cols, 1.0, &mut got).unwrap();
            for i in 0..rows * cols {
                assert_eq!(
                    got[i].to_bits(),
                    want[i].to_bits(),
                    "softmax_rows_into @ {i}"
                );
            }
        }
        // 错误路径：out 长度不符 / x 长度不符 / eps 非法 / scale 非有限
        let mut short = vec![0f32; rows * cols - 1];
        assert!(matches!(
            rms_norm_into(&x, None, rows, cols, 1e-5, &mut short),
            Err(Error::Shape { op, what, .. }) if op == "rms_norm_into" && what == "out"
        ));
        assert!(matches!(
            softmax_rows_into(&x, None, rows, cols, 1.0, &mut short),
            Err(Error::Shape { op, what, .. }) if op == "softmax_rows_into" && what == "out"
        ));
        let mut full = vec![0f32; rows * cols];
        assert!(matches!(
            rms_norm_into(&x[..x.len() - 1], None, rows, cols, 1e-5, &mut full),
            Err(Error::Shape { .. })
        ));
        assert!(matches!(
            rms_norm_into(&x, None, rows, cols, 0.0, &mut full),
            Err(Error::NotPositive { .. })
        ));
        assert!(matches!(
            softmax_rows_into(&x, None, rows, cols, f32::NAN, &mut full),
            Err(Error::NotFinite { .. })
        ));
        // 退化：n = 0
        assert!(rms_norm_into(&[], None, 0, 33, 1e-5, &mut []).is_ok());
        assert!(softmax_rows_into(&[], None, 0, 33, 1.0, &mut []).is_ok());
    }

    #[test]
    fn test_int8_pipeline_relative_error_is_bounded() {
        let mut worst_q = 0.0f64;
        let mut worst_f32 = 0.0f64;
        for &(m, k, n) in &[
            (192usize, 768usize, 768usize),
            (192, 768, 3072),
            (32, 1024, 1024),
        ] {
            for seed in 0..3u64 {
                let mut rng = crate::ops::testutil::Lcg(0x9e37_79b9 ^ seed);
                // 真实量级：激活 ±2、权重 ±0.1
                let x: Vec<f32> = (0..m * k).map(|_| 2.0 * rng.f64() as f32).collect();
                let w: Vec<f32> = (0..n * k).map(|_| 0.1 * rng.f64() as f32).collect();
                // f64 参考 + f32 对照（同一条 Σ X·W 的两种累加）
                let mut yref = vec![0f64; m * n];
                let mut yf32 = vec![0f32; m * n];
                for t in 0..m {
                    for o in 0..n {
                        let (mut acc, mut accf) = (0f64, 0f32);
                        for i in 0..k {
                            acc += x[t * k + i] as f64 * w[o * k + i] as f64;
                            accf = x[t * k + i].mul_add(w[o * k + i], accf);
                        }
                        yref[t * n + o] = acc;
                        yf32[t * n + o] = accf;
                    }
                }
                // int8 管线（出口已带 scale ⇒ 输出的就是反量化值）
                let (xq, sx) = quantize_i8_per_token(&x, m, k).unwrap();
                let (wq, sw) = quantize_i8_per_channel(&w, n, k).unwrap();
                let yq = matmul_i8(
                    xq.as_slice(),
                    wq.as_slice(),
                    sw.as_slice(),
                    sx.as_slice(),
                    m,
                    k,
                    n,
                )
                .unwrap();
                let scale = yref.iter().fold(0f64, |a, &b| a.max(b.abs())).max(1e-30);
                let rel = |v: &[f32]| {
                    v.iter()
                        .zip(&yref)
                        .fold(0f64, |a, (&q, &r)| a.max((q as f64 - r).abs()))
                        / scale
                };
                let (eq, ef) = (rel(yq.as_slice()), rel(&yf32));
                worst_q = worst_q.max(eq);
                worst_f32 = worst_f32.max(ef);
                println!("  m={m} k={k} n={n} seed={seed}: int8 {eq:.3e} / f32 {ef:.3e}");
            }
        }
        println!("  ⇒ 最坏：int8 {worst_q:.3e} / f32 {worst_f32:.3e}（∞-范数相对 f64 参考）");
        // **这就是 int8 端到端口径的验收界**（`docs/ops.md` §2.15）：实测最坏 6.05e-3，
        // 界取 1e-2（1.65× 余量）。f32 路径同口径 ~1.4e-6 作对照，两者差三个数量级。
        assert!(
            worst_q <= 1e-2,
            "int8 管线 ∞-相对误差 {worst_q:.3e} 超过标准 1e-2（见 docs/ops.md §2.15）"
        );
        assert!(worst_f32 <= 1e-5, "f32 对照路径异常：{worst_f32:.3e}");
    }

    /// 端到端：f32 → per-token 量化 → `gemv_i8`，与 f32 参考的相对误差在有界范围内。
    ///
    /// 界怎么定：权重与激活各引入 ≤ `scale/2` 的绝对误差，累加 `k` 项后最坏
    /// `≈ (k/2)·(sw·sx_max + sx·sw_max)`；随机数据的实测值远小于这个上界，这里按
    /// **实测留 3 倍余量**断言（定量界见 `docs/ops.md` §2.12，绝对最坏界不写成断言，
    /// 免得把"随机数据的实测"伪装成"理论保证"）。
    #[test]
    fn test_gemv_i8_pipeline_matches_f32_within_quantization_error() {
        let m = 16usize;
        let k = 64usize;
        let mut rnd = crate::ops::testutil::Lcg(0x1234_5678);
        let w_f32: Vec<f32> = (0..m * k).map(|_| rnd.f64() as f32).collect();
        let x_f32: Vec<f32> = (0..k).map(|_| rnd.f64() as f32).collect();
        // 权重按 per-channel（逐输出行）、激活按 per-tensor
        let (w_q, sw) = quantize_i8_per_channel(&w_f32, m, k).unwrap();
        let (x_q, sx) = quantize_i8_per_tensor(&x_f32).unwrap();
        let got = gemv_i8(w_q.as_slice(), sw.as_slice(), x_q.as_slice(), sx, m, k).unwrap();

        // f32 参考（原始数据）
        let want: Vec<f32> = (0..m)
            .map(|o| (0..k).map(|i| w_f32[o * k + i] * x_f32[i]).sum::<f32>())
            .collect();
        // 参考量级
        let norm = want.iter().fold(0.0f32, |a, b| a.max(b.abs())).max(1e-3);
        for o in 0..m {
            let rel = (got[o] - want[o]).abs() / norm;
            assert!(
                rel < 0.05,
                "o={o}: got={} want={} rel={rel}",
                got[o],
                want[o]
            );
        }
        // 与"量化数据的整数参考"必须逐位一致（这是硬口径，不靠误差界）
        for o in 0..m {
            let acc: i64 = (0..k).map(|i| w_q[o * k + i] as i64 * x_q[i] as i64).sum();
            let exact = (acc as i32 as f32) * (sw[o] * sx);
            assert_eq!(got[o].to_bits(), exact.to_bits(), "o={o}");
        }
    }

    /// 逐行量化 / 反量化的往返：误差 ≤ `scale/2`，且与逐张量版在同一行上一致。
    #[test]
    fn test_quantize_dequantize_roundtrip_error() {
        let rows = 7usize;
        let cols = 33usize;
        let mut rnd = crate::ops::testutil::Lcg(0xfeed_beef);
        let x: Vec<f32> = (0..rows * cols).map(|_| 5.0 * rnd.f64() as f32).collect();
        let (q, scales) = quantize_i8_per_token(&x, rows, cols).unwrap();
        let back = dequantize_i8_rows(q.as_slice(), rows, cols, scales.as_slice()).unwrap();
        for r in 0..rows {
            let s = scales[r];
            for j in 0..cols {
                let i = r * cols + j;
                let e = (back[i] - x[i]).abs();
                assert!(e <= 0.5 * s * (1.0 + 1e-6), "r={r} j={j} e={e} s={s}");
            }
        }
        // `absmax_rows` 与量化内部用的 amax 一致
        let am = absmax_rows(&x, rows, cols).unwrap();
        for r in 0..rows {
            let want = x[r * cols..(r + 1) * cols]
                .iter()
                .fold(0.0f32, |a, &b| a.max(b.abs()));
            assert_eq!(am[r].to_bits(), want.to_bits(), "row {r}");
            assert_eq!(scales[r].to_bits(), (want / 127.0).to_bits());
        }
    }

    /// 形状/长度/溢出错误路径：逐条打准（错误口径与其它 api 一致）。
    #[test]
    fn test_int8_api_error_paths() {
        // 长度不符
        match absmax_rows(&[0.0; 10], 3, 4) {
            Err(Error::Shape {
                op,
                what,
                expected,
                got,
            }) => {
                assert_eq!((op, what), ("absmax_rows", "x"));
                assert_eq!((expected, got), (12, 10));
            }
            other => panic!("应报长度错误：{other:?}"),
        }
        match quantize_i8_per_token(&[0.0; 5], 2, 4) {
            Err(Error::Shape { op, what, .. }) => {
                assert_eq!((op, what), ("quantize_i8_per_token", "x"))
            }
            other => panic!("应报长度错误：{other:?}"),
        }
        match dequantize_i8_rows(&[0i8; 8], 2, 4, &[1.0]) {
            Err(Error::Shape {
                op,
                what,
                expected,
                got,
            }) => {
                assert_eq!((op, what), ("dequantize_i8_rows", "scales"));
                assert_eq!((expected, got), (2, 1));
            }
            other => panic!("应报长度错误：{other:?}"),
        }
        // gemv_i8 的三处长度 + 一处溢出
        let w = [0i8; 12];
        let sw = [1.0f32; 3];
        let x = [0i8; 4];
        assert!(gemv_i8(&w[..11], &sw, &x, 1.0, 3, 4).is_err(), "w 长度");
        assert!(
            gemv_i8(&w, &sw[..2], &x, 1.0, 3, 4).is_err(),
            "scale_w 长度"
        );
        assert!(gemv_i8(&w, &sw, &x[..3], 1.0, 3, 4).is_err(), "x 长度");
        match gemv_i8(&[], &[], &[], 1.0, usize::MAX, 2) {
            Err(Error::Overflow { op, what }) => assert_eq!((op, what), ("gemv_i8", "m×k")),
            other => panic!("应报溢出：{other:?}"),
        }
        // 退化形状不炸
        assert!(absmax_rows(&[], 0, 4).unwrap().is_empty());
        assert_eq!(quantize_i8_per_token(&[], 0, 0).unwrap().0.len(), 0);
        assert_eq!(gemv_i8(&[], &[], &[], 1.0, 0, 0).unwrap().len(), 0);
        // 空输入的 amax 与全零输入
        assert_eq!(amax(&[]).unwrap(), 0.0);
        assert_eq!(amax(&[0.0, -0.0]).unwrap(), 0.0);
    }

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

    /// `dot_q4` 的 scale 数组按"≥ 组数"约定（组数 = `ceil(n_bytes/32)`；见 `docs/ops.md` §5.8）。
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

    /// f16 点积/GEMV：与 C ABI 逐位一致、`gemv` 逐行等于 `dot_f16`、输出对齐、错误路径。
    #[test]
    fn test_f16_matches_c_abi_and_errors() {
        let (m, k) = (6usize, 40usize);
        // 用精确可表示的小值（0x3c00 = 1.0 起，每 4 个 +1/4）
        let a: Vec<u16> = (0..m * k).map(|i| 0x3c00 + (i as u16 % 9)).collect();
        let x: Vec<f32> = (0..k).map(|i| (i as f32).mul_add(0.03, 0.7)).collect();

        let y = gemv_f16(m, k, &a, &x).unwrap();
        assert_eq!(y.as_ptr() as usize % ALIGN, 0, "gemv_f16 输出未对齐");
        assert_eq!(y.len(), m);
        let mut want = vec![0f32; m];
        crate::lasx_gemv_f16(
            a.as_ptr(),
            x.as_ptr(),
            want.as_mut_ptr(),
            m as i32,
            k as i32,
        );
        for r in 0..m {
            assert_eq!(y[r].to_bits(), want[r].to_bits(), "r={r}");
            // 逐行等于 dot_f16（逐位）
            let d = dot_f16(&a[r * k..(r + 1) * k], &x).unwrap();
            assert_eq!(y[r].to_bits(), d.to_bits(), "r={r} 与 dot_f16 不一致");
        }

        // 错误路径
        assert!(matches!(
            dot_f16(&a, &x[..k - 1]),
            Err(Error::Shape { what: "b", .. })
        ));
        assert!(matches!(
            gemv_f16(m, k, &a[..m * k - 1], &x),
            Err(Error::Shape { what: "a", .. })
        ));
        assert!(matches!(
            gemv_f16(m, k, &a, &x[..1]),
            Err(Error::Shape { what: "x", .. })
        ));
        // 退化：k = 0 ⇒ 全 0（空向量的点积是 0）；m = 0 ⇒ 空输出
        assert!(gemv_f16(m, 0, &[], &[]).unwrap().iter().all(|&v| v == 0.0));
        assert!(gemv_f16(0, k, &[], &x).unwrap().is_empty());
        assert_eq!(dot_f16(&[], &[]).unwrap(), 0.0);
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
