//! 形状进类型：**编译期排版 + 声明式调用**。
//!
//! 这一层是 [`crate::plan`] 与 [`crate::view`] 之上的"声明式"外壳：
//!
//! | 层 | 形状在哪 | 适合什么 |
//! |---|---|---|
//! | [`crate::api`] | 三个裸数字 `m, k, n` | 一次性调用、形状完全动态 |
//! | [`crate::view`] | 运行时字段（`MatRef`） | 布局要灵活（列主序、带跨距、转置视角） |
//! | [`crate::plan`] | 运行时字段（`MatmulPlan`） | `B` 固定、反复算 |
//! | **本模块** | **类型里（const 泛型）** | 形状编译期已知，想要"排版编译期定死 + 形状错误编译期报" |
//!
//! # 用法
//!
//! ```
//! use lasx_rs::shape::{Mat, MatBuf};
//!
//! const K: usize = 256;
//! const N: usize = 256;
//! const M: usize = 64;
//!
//! # let weights: Vec<f32> = (0..K * N).map(|i| (i % 13) as f32).collect();
//! # let batch: Vec<f32> = (0..M * K).map(|i| (i % 7) as f32).collect();
//! let w = Mat::<f32, { K }, { N }>::new(&weights)?;   // 权重 K×N
//! let x = Mat::<f32, { M }, { K }>::new(&batch)?;     // 输入 M×K
//!
//! // ① 公式写法：K 由类型对齐，输出形状自动推出 M×N
//! let y = x.mul(&w);
//! // ② 推理写法：把权重作用到一批输入上（同一件事）
//! let y2 = w.apply(&x);
//! assert_eq!(y.as_slice(), y2.as_slice());
//!
//! // ③ 声明"这个权重反复用"：打包一次，之后每次只算乘法
//! let wp = w.prepare();
//! let y3 = wp.apply(&x);
//! assert_eq!(y.as_slice(), y3.as_slice());
//! # Ok::<(), lasx_rs::api::Error>(())
//! ```
//!
//! **K 对不上是编译错误**，报错直接指在公式那一行：
//!
//! ```text
//! let x = Mat::<f32, 64, 512>::new(&batch)?;   // K 写成 512（应为 256）
//! let y = x.mul(&w);
//! error[E0308]: mismatched types
//!     |
//! 218 |     let y = x.mul(&w);
//!     |                --- ^^ expected `512`, found `256`
//! ```
//!
//! # DYN 档：行数运行时可变
//!
//! 真实业务里 `K/N`（模型结构）固定、**batch 行数每批不同**。`K/N` 是 const，
//! 就意味着"排版"（面板数、k 分块、列尾、打包字节数）照样在编译期算完；挪到运行期的
//! 只有行：
//!
//! ```
//! use lasx_rs::shape::{Mat, MatDyn, MatBufDyn};
//!
//! const K: usize = 256;
//! const N: usize = 256;
//! # let weights: Vec<f32> = (0..K * N).map(|i| (i % 13) as f32).collect();
//! let w = Mat::<f32, { K }, { N }>::new(&weights)?;
//! let wp = w.prepare();
//!
//! for rows in [64usize, 96] {                       // ← 每批行数不一样
//! #   let batch: Vec<f32> = (0..rows * K).map(|i| (i % 7) as f32).collect();
//!     let x = MatDyn::<f32, { K }>::new(&batch)?;   // 行数由长度推出（运行时）
//!     let mut y = MatBufDyn::<f32, { N }>::with_rows(rows);
//!     wp.apply_dyn_into(&x, &mut y);
//! #   let want = lasx_rs::api::matmul(rows, K, N, &batch, &weights)?;
//! #   assert_eq!(y.as_slice(), want.as_slice());
//! }
//! # Ok::<(), lasx_rs::api::Error>(())
//! ```
//!
//! 权重 `w` 仍是**全静态**类型 `Mat<K, N>`：DYN 只把"行"交给运行期，打包与排版不变。
//!
//! # 数值保证
//!
//! 本层不含任何新的数值路径：`Prepared` 里面就是 [`crate::plan::MatmulPlan`]（打包 +
//! k 分块，逐位一致，理由见 [`crate::plan`] 的模块文档）。所以本层与 [`crate::api::matmul`]
//! **逐位一致**，测试用 `to_bits()` 比对（含 DYN 档多种行数）。
//!
//! # 边界
//!
//! - `K == 0` 无法从长度推出行数 → 构造直接报错（退化情形用 [`crate::api::matmul`]）。
//! - 与 [`crate::ops::matmul`](crate) 同前提：**需要 LASX**。

use crate::aligned::AlignedVec;
use crate::api::Error;
use crate::plan::{MatmulPlan, PackedKernel};

/* ==================== 1. 排版方案（编译期算完） ==================== */

/// 打包/分块方案：`K/N` 是 const，所以这些全在 **const 求值**里定死。
///
/// 运行期只剩"常数驱动的循环"，没有阈值分支、没有形状判断。常量来源与算子层一致
/// （L2 预算 2 MiB、L1 预算 48 KiB、条带 32 列 f32；见 `docs/dev.md` §8）。
pub struct Layout<const K: usize, const N: usize>;

impl<const K: usize, const N: usize> Layout<K, N> {
    /// 一条带多少列（f32）。
    pub const STRIP: usize = 32;
    /// 完整条带数。
    pub const STRIPS: usize = N / 32;
    /// 一个面板放几条带（面板要留在 L2 里被整轮行扫描复用）。
    pub const PANEL_STRIPS: usize = {
        let per = K * 32 * 4;
        let v = 2 * 1024 * 1024 / per;
        if v < 1 {
            1
        } else if v > 256 {
            256
        } else {
            v
        }
    };
    /// 面板个数。
    pub const PANELS: usize = Self::STRIPS.div_ceil(Self::PANEL_STRIPS);
    /// 列尾（不足一条带的列，交给尾部内核）。
    pub const TAIL_COLS: usize = N % 32;
    /// `k` 要不要分块（条带 `K×32×4` 超过 L1 预算就分）。
    pub const K_CHUNKED: bool = K * 32 * 4 > 48 * 1024;
    /// 打包缓冲字节数。
    pub const PACK_BYTES: usize = K * 32 * Self::STRIPS * 4;
}

// 编译期自检：不合预期就**编译不过**（"排版在编译期"的凭据）
const _: () = assert!(Layout::<256, 256>::PANELS == 1);
const _: () = assert!(Layout::<256, 256>::TAIL_COLS == 0);
const _: () = assert!(!Layout::<256, 256>::K_CHUNKED);
const _: () = assert!(Layout::<1024, 4096>::K_CHUNKED);
const _: () = assert!(Layout::<1024, 4096>::PANELS > 1);

/* ==================== 2. 形状进类型 ==================== */

/// `R × C` 矩阵的只读视图（行主序、连续）。形状在类型里，运行期只存数据。
///
/// 零开销：`&[T]` + 两个 const 标签。
pub struct Mat<'a, T, const R: usize, const C: usize> {
    data: &'a [T],
}

/// `R × C` 矩阵的拥有者（输出用）。
pub struct MatBuf<T, const R: usize, const C: usize> {
    data: AlignedVec<T>,
}

/// **DYN 档**：行数运行时才知道，列数（`C`）在类型里。
pub struct MatDyn<'a, T, const C: usize> {
    data: &'a [T],
    rows: usize,
}

/// DYN 档的输出：列数在类型里，行数运行时。
pub struct MatBufDyn<T, const C: usize> {
    data: AlignedVec<T>,
    rows: usize,
}

impl<'a, T: Copy, const R: usize, const C: usize> Mat<'a, T, R, C> {
    /// 构造时**只查长度**；形状对不对是类型的事。
    ///
    /// # Errors
    /// `data.len() != R × C`（[`Error::Shape`]）。
    pub fn new(data: &'a [T]) -> Result<Self, Error> {
        if data.len() == R * C {
            Ok(Mat { data })
        } else {
            Err(Error::Shape {
                op: "Mat::new",
                what: "data",
                expected: R * C,
                got: data.len(),
            })
        }
    }

    /// 行数（编译期已知）。
    pub fn rows(&self) -> usize {
        R
    }

    /// 列数（编译期已知）。
    pub fn cols(&self) -> usize {
        C
    }

    /// 连续行主序切片。
    pub fn as_slice(&self) -> &'a [T] {
        self.data
    }
}

impl<T: Copy + Default, const R: usize, const C: usize> MatBuf<T, R, C> {
    /// 新分配（对齐），内容为 `T::default()`。
    pub fn new() -> Self {
        MatBuf {
            data: AlignedVec::new(R * C),
        }
    }

    /// 行数。
    pub fn rows(&self) -> usize {
        R
    }

    /// 列数。
    pub fn cols(&self) -> usize {
        C
    }

    /// 只读切片。
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    /// 可写切片。
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }
}

impl<T: Copy + Default, const R: usize, const C: usize> Default for MatBuf<T, R, C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T: Copy, const C: usize> MatDyn<'a, T, C> {
    /// 行数由长度推出（`rows = len / C`）。
    ///
    /// # Errors
    /// `C == 0`（[`Error::NotPositive`]），或长度不是 `C` 的整数倍（[`Error::Shape`]）。
    pub fn new(data: &'a [T]) -> Result<Self, Error> {
        if C == 0 {
            return Err(Error::NotPositive {
                op: "MatDyn::new",
                what: "C（列数）",
                value: 0.0,
            });
        }
        if !data.len().is_multiple_of(C) {
            return Err(Error::Shape {
                op: "MatDyn::new",
                what: "data（长度必须是 C 的整数倍）",
                expected: data.len() / C * C,
                got: data.len(),
            });
        }
        Ok(MatDyn {
            data,
            rows: data.len() / C,
        })
    }

    /// 运行时行数。
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// 列数（编译期已知）。
    pub fn cols(&self) -> usize {
        C
    }

    /// 连续行主序切片。
    pub fn as_slice(&self) -> &'a [T] {
        self.data
    }
}

impl<T: Copy + Default, const C: usize> MatBufDyn<T, C> {
    /// 按行数分配（列数 = `C`）。
    pub fn with_rows(rows: usize) -> Self {
        MatBufDyn {
            data: AlignedVec::new(rows * C),
            rows,
        }
    }

    /// 行数。
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// 列数。
    pub fn cols(&self) -> usize {
        C
    }

    /// 只读切片。
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    /// 可写切片。
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }
}

/* ==================== 3. `Auto` 判据（实测标定） ==================== */

/// `Auto` 决定并行前要求的最小工作量（**乘加次数** = `m × k × n`）。
///
/// 实测（`K=N=256`、12 线程、与单线程逐位对照，见 `docs/dev.md` §3.2）：
///
/// | 形状 | 工作量 | 池/单线程 |
/// |---|---|---|
/// | `64×64×64` | 262 k | **4.11×（池更慢）** |
/// | `1024×8×8` | 65 k | **3.19×（池更慢）** |
/// | `256×64×64` | 1048 k | 1.20×（更慢） |
/// | `32×256×256` | 2097 k | 1.01×（盈亏平衡） |
/// | `64×256×256` | 4194 k | 0.76×（更快） |
/// | `256×256×256` | 16777 k | 0.44× |
///
/// 所以门限定在 4 M，落在"盈亏平衡（2097 k）"与"明确更快（4194 k）"之间。
pub const AUTO_MIN_WORK: usize = 4_000_000;

/// `Auto` 判据：给形状与机器并行度，返回**应该用几个线程**（1 = 不派活）。
///
/// 两条都要满足才并行：
///
/// 1. **每线程至少 4 行**——微内核一次算 4 行、B 被复用 4 次，行数不够时并行只会
///    制造退化的尾块；
/// 2. **总工作量 ≥ [`AUTO_MIN_WORK`]**——派活本身要几十微秒（worker 唤醒），
///    小形状上它比干活还贵（实测 `64×64×64` 慢 4.11×）。
///
/// 另外一条与本节无关、但同样实测过的事实：`&self` 之后每次派活多一层同步，
/// 单价 44 ns（最坏情况）。在 1.95 µs 的调用上占 **3.2%**，在 ≥10 µs 的调用上
/// 已经落到 0.2% 以下（`examples/pool_shared_sync.rs`）。上面第二条门限同时把
/// 这个开销关在门外：低于 4 M 的工作量一律不派活、也就不付这层同步。
///
/// `machine` 传 0 视作 1。
pub fn auto_threads(rows: usize, k: usize, n: usize, machine: usize) -> usize {
    let machine = machine.max(1);
    let by_rows = rows / 4; // 每线程 4 行（微内核的 B 复用粒度）
    if by_rows < 2 {
        return 1;
    }
    let work = rows.saturating_mul(k).saturating_mul(n);
    if work < AUTO_MIN_WORK {
        return 1;
    }
    by_rows.min(machine).max(1)
}

/* ==================== 4. 声明式调用 ==================== */

/// "把权重打包好、以后反复用"的对象：形状在类型里，里面是 [`MatmulPlan`]。
///
/// 构造见 [`Mat::prepare`]；之后 [`Prepared::apply`] / [`Prepared::apply_dyn`] 只算乘法，
/// 零分配（`apply_into`）、零打包。
pub struct Prepared<T: PackedKernel, const K: usize, const N: usize> {
    plan: MatmulPlan<T>,
}

impl<'a, T: PackedKernel, const A: usize, const B: usize> Mat<'a, T, A, B> {
    /// `A·B` 写法：`self` 是 `M×K`，`rhs` 是 `K×N` → 输出 `M×N`。
    pub fn mul<const N: usize>(&self, rhs: &Mat<'_, T, B, N>) -> MatBuf<T, A, N> {
        rhs.apply(self)
    }

    /// 同上，写进已有输出（不分配）。
    pub fn mul_into<const N: usize>(&self, rhs: &Mat<'_, T, B, N>, out: &mut MatBuf<T, A, N>) {
        rhs.prepare().apply_into(self, out);
    }

    /// 推理写法：`self` 是权重 `K×N`，`x` 是 `M×K` → 输出 `M×N`。
    pub fn apply<const M: usize>(&self, x: &Mat<'_, T, M, A>) -> MatBuf<T, M, B> {
        self.prepare().apply(x)
    }

    /// DYN 档：`x` 的行数运行时。
    pub fn apply_dyn(&self, x: &MatDyn<'_, T, A>) -> MatBufDyn<T, B> {
        self.prepare().apply_dyn(x)
    }

    /// DYN 档写进已有输出（一次性，内部仍会打包一次；反复用请走 [`Mat::prepare`]）。
    pub fn apply_dyn_into(&self, x: &MatDyn<'_, T, A>, out: &mut MatBufDyn<T, B>) {
        self.prepare().apply_dyn_into(x, out);
    }

    /// 声明"这是反复用的权重"：**打包一次**，之后每次只算乘法。
    pub fn prepare(&self) -> Prepared<T, A, B> {
        Prepared {
            plan: MatmulPlan::from_row_major(self.data, A, B)
                .expect("形状由 Mat 的构造保证：长度 = A×B"),
        }
    }
}

impl<T: PackedKernel, const K: usize, const N: usize> Prepared<T, K, N> {
    /// 权重行数（编译期已知）。
    pub fn k(&self) -> usize {
        K
    }

    /// 权重列数（编译期已知）。
    pub fn n(&self) -> usize {
        N
    }

    /// 打包缓冲占多少字节（内存代价）。
    pub fn packed_bytes(&self) -> usize {
        self.plan.packed_bytes()
    }

    /// `C[M×N] = X[M×K] · self`，`M` 编译期已知；输出新分配。
    pub fn apply<const M: usize>(&self, x: &Mat<'_, T, M, K>) -> MatBuf<T, M, N> {
        let mut out = MatBuf::<T, M, N>::new();
        self.apply_into(x, &mut out);
        out
    }

    /// 同上，写进调用方的缓冲（不分配）。
    pub fn apply_into<const M: usize>(&self, x: &Mat<'_, T, M, K>, out: &mut MatBuf<T, M, N>) {
        // 形状已由 Mat/MatBuf 的构造保证，这里不可能失败
        self.plan
            .run_into(x.as_slice(), out.as_mut_slice())
            .expect("形状由 Mat/MatBuf 的构造保证");
    }

    /// DYN 档：行数运行时，`K/N` 仍 const；输出新分配。
    pub fn apply_dyn(&self, x: &MatDyn<'_, T, K>) -> MatBufDyn<T, N> {
        let mut out = MatBufDyn::<T, N>::with_rows(x.rows());
        self.apply_dyn_into(x, &mut out);
        out
    }

    /// DYN 档写进调用方的缓冲（不分配）。
    pub fn apply_dyn_into(&self, x: &MatDyn<'_, T, K>, out: &mut MatBufDyn<T, N>) {
        assert_eq!(
            out.rows(),
            x.rows(),
            "输出行数必须等于输入行数（DYN 档里这是运行时才知道的）"
        );
        self.plan
            .run_into(x.as_slice(), out.as_mut_slice())
            .expect("行数一致时形状必然成立");
    }
}

impl<T: PackedKernel, const K: usize, const N: usize> std::fmt::Debug for Prepared<T, K, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Prepared")
            .field("K", &K)
            .field("N", &N)
            .field("packed_bytes", &self.plan.packed_bytes())
            .finish()
    }
}

/// 四个形状类型只打印形状，不打印数据（也顺带让 `assert_eq!` 的失败信息可读）。
impl<T, const R: usize, const C: usize> std::fmt::Debug for Mat<'_, T, R, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Mat({R}×{C})")
    }
}

impl<T, const R: usize, const C: usize> std::fmt::Debug for MatBuf<T, R, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MatBuf({R}×{C})")
    }
}

impl<T, const C: usize> std::fmt::Debug for MatDyn<'_, T, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MatDyn({}×{C})", self.rows)
    }
}

impl<T, const C: usize> std::fmt::Debug for MatBufDyn<T, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MatBufDyn({}×{C})", self.rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(len: usize, seed: u64) -> Vec<f64> {
        let mut rng = crate::ops::testutil::Lcg(seed);
        (0..len).map(|_| rng.f64() * 2.0 - 1.0).collect()
    }

    fn f32s(len: usize, seed: u64) -> Vec<f32> {
        seq(len, seed).into_iter().map(|x| x as f32).collect()
    }

    fn bits_f32(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// 静态形状：与 `api::matmul` 逐位一致（含行尾 / 列尾 / 多面板）。
    ///
    /// const 泛型没法在运行期"按变量选形状"，所以每个形状显式列一行——这正是这一层的
    /// 使用方式：形状写死在类型里。
    #[test]
    fn test_shape_matches_api_bit_for_bit() {
        macro_rules! case {
            ($m:expr, $k:expr, $n:expr) => {{
                let (m, k, n) = ($m, $k, $n);
                let a = f32s(m * k, 0x51ed ^ (m * 7 + k) as u64);
                let b = f32s(k * n, 0xbeef ^ (k * 13 + n) as u64);
                let want = crate::api::matmul(m, k, n, &a, &b).unwrap();

                let bw = Mat::<f32, $k, $n>::new(&b).unwrap();
                let xw = Mat::<f32, $m, $k>::new(&a).unwrap();
                // 公式写法与声明式写法各来一遍，都要逐位一致
                let via_mul = xw.mul(&bw);
                let via_plan = bw.prepare().apply(&xw);
                assert_eq!(
                    bits_f32(via_mul.as_slice()),
                    bits_f32(&want),
                    "{}×{}×{} mul",
                    $m,
                    $k,
                    $n
                );
                assert_eq!(
                    bits_f32(via_plan.as_slice()),
                    bits_f32(&want),
                    "{}×{}×{} plan",
                    $m,
                    $k,
                    $n
                );
            }};
        }
        case!(1, 1, 1);
        case!(4, 4, 33); // 列尾
        case!(7, 64, 5); // n < 一条带（全走列尾）+ 行不足 4
        case!(8, 64, 37);
        case!(40, 64, 127); // 列尾 31 列
        case!(128, 128, 128);
        case!(513, 448, 130); // 行尾 + 列尾 + k 分块
    }

    /// DYN 档：同一个打包好的权重吃多种运行时行数，全部与 `api::matmul` 逐位一致。
    #[test]
    fn test_dyn_batches_match_api_bit_for_bit() {
        const K: usize = 256;
        const N: usize = 256;
        let weights = f32s(K * N, 0x1234);
        let w = Mat::<f32, { K }, { N }>::new(&weights).unwrap();
        let wp = w.prepare();
        assert_eq!(wp.packed_bytes(), K * N * 4);

        for rows in [1usize, 3, 16, 64, 96] {
            let batch = f32s(rows * K, 0x5a5a ^ rows as u64);
            let want = crate::api::matmul(rows, K, N, &batch, &weights).unwrap();

            let x = MatDyn::<f32, { K }>::new(&batch).unwrap();
            assert_eq!(x.rows(), rows);

            let y = wp.apply_dyn(&x);
            assert_eq!(bits_f32(y.as_slice()), bits_f32(&want), "DYN rows={rows}");

            let mut y2 = MatBufDyn::<f32, { N }>::with_rows(rows);
            wp.apply_dyn_into(&x, &mut y2);
            assert_eq!(
                bits_f32(y2.as_slice()),
                bits_f32(&want),
                "DYN into rows={rows}"
            );

            // 一次性路径（内部会重新打包）同样逐位一致
            assert_eq!(
                bits_f32(w.apply_dyn(&x).as_slice()),
                bits_f32(&want),
                "DYN 一次性 rows={rows}"
            );
        }
    }

    /// `K` 对不上是**编译期**错误——这条测试是"能编译"本身，见下方 trybuild 式的注释。
    /// （真正的反例在 `docs/dev.md` §19 里留了报错样本。）
    #[test]
    fn test_static_shapes_are_checked_by_types() {
        let a = f32s(6 * 4, 1);
        let b = f32s(4 * 5, 2);
        let wa = Mat::<f32, 4, 5>::new(&b).unwrap();
        let x = Mat::<f32, 6, 4>::new(&a).unwrap();
        // 6×4 · 4×5 = 6×5；行列都可以从类型读出来
        let y = x.mul(&wa);
        assert_eq!((y.rows(), y.cols()), (6, 5));
        assert_eq!(
            bits_f32(y.as_slice()),
            bits_f32(&crate::api::matmul(6, 4, 5, &a, &b).unwrap())
        );
    }

    #[test]
    fn test_errors_on_bad_lengths() {
        let data = vec![0f32; 10];
        assert!(Mat::<f32, 3, 4>::new(&data).is_err(), "10 ≠ 3×4");
        // DYN：长度必须是 C 的整数倍
        assert!(MatDyn::<f32, 4>::new(&data).is_err(), "10 不是 4 的倍数");
        assert!(MatDyn::<f32, 5>::new(&data).is_ok(), "10 = 2×5");
        // C = 0 无法推行数
        assert!(matches!(
            MatDyn::<f32, 0>::new(&[]),
            Err(Error::NotPositive { .. })
        ));
        // K = 0 的计划构造报错（行数无法从长度推出）
        let empty = Mat::<f32, 0, 8>::new(&[]).unwrap();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| empty.prepare()));
        assert!(r.is_err(), "K=0 的 prepare 应当报错");
    }

    #[test]
    fn test_layout_is_compile_time() {
        // 真正的"编译期"断言是文件上半部分那组 `const _: () = assert!(...)`；
        // 这里把常量取出来比一遍，免得读者以为它们依赖运行期参数。
        let (panels, bytes) = (Layout::<256, 256>::PANELS, Layout::<256, 256>::PACK_BYTES);
        assert_eq!(panels, 1);
        assert_eq!(bytes, 256 * 256 * 4);

        let chunked_big = Layout::<1024, 4096>::K_CHUNKED;
        let chunked_small = Layout::<256, 256>::K_CHUNKED;
        assert!(chunked_big);
        assert!(!chunked_small);

        let many = Layout::<2048, 4096>::PANELS;
        assert!(many > 1);
        let tail = Layout::<256, 100>::TAIL_COLS;
        assert_eq!(tail, 4);
    }

    /// `Auto` 判据必须与实测结论一致（数据见 `AUTO_MIN_WORK` 的文档）。
    #[test]
    fn test_auto_threads_matches_measurements() {
        // 行数不够（每线程 < 4 行）：不并行
        assert_eq!(auto_threads(16, 256, 256, 12), 1);
        assert_eq!(auto_threads(32, 256, 256, 12), 1);
        assert_eq!(auto_threads(7, 4096, 4096, 12), 1);
        // 行数够但工作量不够（实测池慢 4.11×/3.19×）：不并行
        assert_eq!(auto_threads(64, 64, 64, 12), 1);
        assert_eq!(auto_threads(1024, 8, 8, 12), 1);
        assert_eq!(auto_threads(256, 64, 64, 12), 1);
        // 两条都满足：并行，且不超过机器并行度
        assert_eq!(auto_threads(64, 256, 256, 12), 12);
        assert_eq!(auto_threads(256, 256, 256, 12), 12);
        assert_eq!(auto_threads(1024, 256, 256, 24), 24);
        // 机器只有 2 核时，行数够也只用 2 个
        assert_eq!(auto_threads(256, 256, 256, 2), 2);
        // machine = 0 视作 1
        assert_eq!(auto_threads(256, 256, 256, 0), 1);
    }
}
