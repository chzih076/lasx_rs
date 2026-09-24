//! 矩阵乘"计划"：**把 B 打包一次，之后反复算 `A·B`**。
//!
//! # 一句话用法
//!
//! ```
//! use lasx_rs::api;
//! use lasx_rs::plan::MatmulPlan;
//!
//! # let (m, k, n) = (40, 64, 96);
//! # let a: Vec<f32> = (0..m * k).map(|i| (i % 11) as f32 - 5.0).collect();
//! # let b: Vec<f32> = (0..k * n).map(|i| (i % 7) as f32 * 0.5).collect();
//! // ① 权重打包一次（构造时把 B 归拢成内核要的样子）
//! let plan = MatmulPlan::from_row_major(&b, k, n).unwrap();
//! // ② 之后每次调用都不再碰原始 B，也不再分配打包缓冲
//! let c1 = plan.run(&a).unwrap();
//!
//! // 结果与一次性调用的 api::matmul **逐位一致**
//! let c2 = api::matmul(m, k, n, &a, &b).unwrap();
//! assert_eq!(&c1[..], &c2[..]);
//! ```
//!
//! # 什么时候值得用
//!
//! 同一个 `B` 要乘很多个 `A` 时——推理里的权重矩阵、参数扫描、每帧都乘同一个变换矩阵。
//! `api::matmul` 每次都要**重新打包一遍 B**（B 要先按列条带重排，微内核才能顺序读），
//! 而打包的代价是 `O(k·n)`：小矩阵上它比乘法本身还贵。
//!
//! 代价也很直接：计划里存着一份打包好的 B（[`MatmulPlan::packed_bytes`] 报出字节数，
//! 通常是 `k×n×size_of::<T>()`），构造时要做一次 `O(k·n)` 的搬运。
//! 所以：
//!
//! | 场景 | 用哪个 |
//! |---|---|
//! | B 固定、要乘很多次 | **本模块**（`MatmulPlan`） |
//! | 只算一次，形状由调用方给 | [`crate::api::matmul`]（按形状自动选最快路径） |
//! | 要多核 | [`crate::parallel::matmul_f32`] / [`crate::parallel::matmul_f64`] |
//! | 只要一次、且想强制走打包路径做 A/B | [`crate::api::matmul_f32_packed`] |
//!
//! # 打包好的计划是只读的，可以多线程共用一份 B
//!
//! [`MatmulPlan::run_into`] 只借用 `&self`，所以把计划放进 `Arc`，多个线程就能共享
//! **同一份**打包好的 B、各自往自己的输出行段里写：
//!
//! ```
//! use lasx_rs::plan::MatmulPlan;
//! use std::sync::Arc;
//!
//! # let (m, k, n) = (8, 64, 96);
//! # let a: Vec<f32> = (0..m * k).map(|i| (i % 11) as f32 - 5.0).collect();
//! # let b: Vec<f32> = (0..k * n).map(|i| (i % 7) as f32 * 0.5).collect();
//! let plan = Arc::new(MatmulPlan::from_row_major(&b, k, n).unwrap());
//! let mut c = vec![0f32; m * n];
//! // 输出按行切开：每个线程自己一段，互不重叠
//! std::thread::scope(|s| {
//!     let (top, bottom) = c.split_at_mut((m / 2) * n);
//!     for (rows, part) in [(0..m / 2, top), (m / 2..m, bottom)] {
//!         let plan = Arc::clone(&plan);
//!         let a = &a[rows.start * k..rows.end * k];
//!         s.spawn(move || plan.run_into(a, part).unwrap());
//!     }
//! });
//! assert_eq!(&c[..], &plan.run(&a).unwrap()[..]);
//! ```
//!
//! # B 可以是任意布局
//!
//! [`MatmulPlan::new`] 接收一个 [`MatRef`]，所以**行主序、列主序（预先转置好的权重）、
//! 带行跨距、转置视角**都能直接喂进来，不需要调用方先复制成行主序：
//!
//! ```
//! use lasx_rs::view::MatRef;
//! use lasx_rs::plan::MatmulPlan;
//!
//! # let (m, k, n) = (12, 64, 40);
//! # let a: Vec<f32> = (0..m * k).map(|i| (i % 11) as f32 - 5.0).collect();
//! # let b_row: Vec<f32> = (0..k * n).map(|i| (i % 7) as f32 * 0.5).collect();
//! // 权重本来就是"转置存放"的（列主序），直接给它
//! let mut b_col = vec![0f32; k * n];
//! for p in 0..k {
//!     for j in 0..n {
//!         b_col[j * k + p] = b_row[p * n + j];
//!     }
//! }
//! let plan = MatmulPlan::new(&MatRef::col_major(&b_col, k, n).unwrap()).unwrap();
//! let c = plan.run(&a).unwrap();
//! let want = MatmulPlan::from_row_major(&b_row, k, n).unwrap().run(&a).unwrap();
//! assert_eq!(&c[..], &want[..]);
//! ```
//!
//! # 数值保证
//!
//! 计划走的是算子层的"打包 + k 分块"路径（`ops::matmul` 里的
//! `matmul_f32_packed_rows` / f64 对应物），它和流式路径、列块路径**逐位一致**：
//! 每个输出元素始终是沿 `k` 的单个累加器、累加次序不变，k 分块只是把同一个部分和
//! 分段落到 `C` 再读回（落盘/读回是精确的）。矩阵尾部的列（不足一条带的部分）交给
//! 同一个尾部内核处理，所以**没有"某条路径差 1 ulp"的例外**。
//!
//! # 边界
//!
//! - `k == 0` 时无法从 `a` 的长度推出 `m`，[`MatmulPlan::new`] 直接报错；
//!   这种退化情形用 [`crate::api::matmul`]。
//! - 与 `ops::matmul` 同前提：**需要 LASX**（无 LASX 的 CPU 上见手册 Caveats）。

use crate::aligned::AlignedVec;
use crate::api::Error;
use crate::view::MatRef;

/// 一种元素类型的打包矩阵乘内核（本 crate 只实现 `f32` 与 `f64`）。
///
/// 这一层存在的意义是把"f32 = 32 列一条带 / f64 = 16 列一条带"这类差异收在一处，
/// 于是 [`MatmulPlan`] 的构建与运行逻辑只写一遍。方法都是对算子层 `pub(crate)` 入口的
/// 直接转发，没有自己的数值逻辑——**结果与顺序路径逐位一致**，理由见模块文档"数值保证"。
pub trait PackedKernel: Copy + Default + Send + Sync {
    /// 一条带覆盖多少列（`f32` = 32，`f64` = 16）。
    const STRIP: usize;

    /// 一次打包放几条带（算子层按 L2 预算算出，面板要留在 L2 里被整轮行扫描复用）。
    fn strips_per_panel(k: usize) -> usize;

    /// 把 `B[p][jb..jb+nc]` 打进 `dst[(s·k + p)·STRIP + r]`。
    fn pack_panel(k: usize, n: usize, jb: usize, nc: usize, b: &MatRef<'_, Self>, dst: &mut [Self]);

    /// 用已打包的面板算所有行。
    // 参数表与算子层入口一一对应（k、n、列区间、面板、A、C、主体行数），不再包装一层
    #[allow(clippy::too_many_arguments)]
    fn packed_rows(
        k: usize,
        n: usize,
        jb: usize,
        nc: usize,
        packed: &[Self],
        a: &[Self],
        c: &mut [Self],
        m4: usize,
    );

    /// 一条行的列尾 `[0, cols)`：`b` 是**尾部列单独收拢成的**行主序 `k × cols` 矩阵。
    fn row_tail(a_row: &[Self], c_row: &mut [Self], b: &[Self], k: usize, cols: usize);
}

impl PackedKernel for f32 {
    const STRIP: usize = 32;

    fn strips_per_panel(k: usize) -> usize {
        crate::ops::matmul::pack_strips(k)
    }

    fn pack_panel(
        k: usize,
        n: usize,
        jb: usize,
        nc: usize,
        b: &MatRef<'_, Self>,
        dst: &mut [Self],
    ) {
        match b.as_row_major_contiguous() {
            // 连续行主序：直接用算子层自己的打包函数（逐位一致由构造保证，不是巧合）
            Some(flat) => crate::ops::matmul::pack_b(k, n, jb, nc, flat, dst),
            None => pack_panel_gathered(jb, nc, Self::STRIP, b, dst),
        }
    }

    fn packed_rows(
        k: usize,
        n: usize,
        jb: usize,
        nc: usize,
        packed: &[Self],
        a: &[Self],
        c: &mut [Self],
        m4: usize,
    ) {
        crate::ops::matmul::matmul_f32_packed_rows(k, n, jb, nc, packed, a, c, m4);
    }

    fn row_tail(a_row: &[Self], c_row: &mut [Self], b: &[Self], k: usize, cols: usize) {
        crate::ops::matmul::row_tail_range_f32(a_row, c_row, b, k, cols, 0, cols);
    }
}

impl PackedKernel for f64 {
    const STRIP: usize = 16;

    fn strips_per_panel(k: usize) -> usize {
        crate::ops::matmul_f64::pack_strips(k)
    }

    fn pack_panel(
        k: usize,
        n: usize,
        jb: usize,
        nc: usize,
        b: &MatRef<'_, Self>,
        dst: &mut [Self],
    ) {
        match b.as_row_major_contiguous() {
            Some(flat) => crate::ops::matmul_f64::pack_b_f64(k, n, jb, nc, flat, dst),
            None => pack_panel_gathered(jb, nc, Self::STRIP, b, dst),
        }
    }

    fn packed_rows(
        k: usize,
        n: usize,
        jb: usize,
        nc: usize,
        packed: &[Self],
        a: &[Self],
        c: &mut [Self],
        m4: usize,
    ) {
        crate::ops::matmul_f64::matmul_f64_packed_rows(k, n, jb, nc, packed, a, c, m4);
    }

    fn row_tail(a_row: &[Self], c_row: &mut [Self], b: &[Self], k: usize, cols: usize) {
        crate::ops::matmul_f64::row_tail_f64(a_row, c_row, b, k, cols, 0);
    }
}

/// `A[m×k] · B[k×n]` 里那个"打包好的 B"。
///
/// 构造见 [`MatmulPlan::new`] / [`MatmulPlan::from_row_major`]；之后用
/// [`MatmulPlan::run`] 或 [`MatmulPlan::run_into`] 反复算，行数 `m` 由 `a.len()/k` 推出。
pub struct MatmulPlan<T: PackedKernel> {
    /// `A` 的列数 = `B` 的行数。
    k: usize,
    /// `B` 的列数 = `C` 的列数。
    n: usize,
    /// `[0, n_strip)` 这些列由打包面板覆盖（`n_strip` 是 `STRIP` 的整数倍）。
    n_strip: usize,
    /// 打包好的列面板，按列顺序排列，末块可能更窄。
    panels: Vec<Panel<T>>,
    /// 尾部列 `[n_strip, n)`：不足一条带，单独收拢成行主序 `k × (n - n_strip)` 矩阵
    /// （这样就能复用算子层现成的尾部内核，不必再写一份按跨距取数的版本）。
    tail: Option<AlignedVec<T>>,
}

/// 一块打包好的列面板：`nc` 列，起始列是 `jb`。
struct Panel<T> {
    jb: usize,
    nc: usize,
    packed: AlignedVec<T>,
}

impl<T: PackedKernel> std::fmt::Debug for MatmulPlan<T> {
    /// 只打印"形状与代价"，不打印几百 KB 的打包数据。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MatmulPlan")
            .field("k", &self.k)
            .field("n", &self.n)
            .field("panels", &self.panels.len())
            .field("tail_cols", &(self.n - self.n_strip))
            .field("packed_bytes", &self.packed_bytes())
            .finish()
    }
}

impl<T: PackedKernel> MatmulPlan<T> {
    /// 按 [`MatRef`] 视图打包 `B`（`B` 的布局由视图决定，见模块文档）。
    ///
    /// 返回后**不再引用 `B`**：原矩阵可以释放，计划可以存起来或 `Arc` 起来共享。
    ///
    /// # Errors
    /// - `B.rows() == 0`（[`Error::NotPositive`]）——此时 `m` 无法从 `a` 的长度推出，
    ///   退化情形请用 [`crate::api::matmul`]；
    /// - 视图本身的形状错误在 [`MatRef`] 构造时就已经报过了。
    pub fn new(b: &MatRef<'_, T>) -> Result<Self, Error> {
        let k = b.rows();
        let n = b.cols();
        if k == 0 {
            return Err(Error::NotPositive {
                op: "MatmulPlan::new",
                what: "k（k=0 时无法从 a 的长度推出 m）",
                value: 0.0,
            });
        }
        let strip = T::STRIP;
        let n_strip = n / strip * strip;
        let strips_total = n_strip / strip;

        // 面板按算子层的 L2 预算切：太大就装不进 L2，整轮行扫描会反复回内存取。
        let per_panel = T::strips_per_panel(k).max(1);
        let mut panels = Vec::with_capacity(strips_total.div_ceil(per_panel));
        let mut s0 = 0;
        while s0 < strips_total {
            let s1 = (s0 + per_panel).min(strips_total);
            let nc = (s1 - s0) * strip;
            // `k × nc ≤ k × n`，而 `MatRef` 已经确认过 `k × n` 不溢出，故这里安全
            let mut packed = AlignedVec::<T>::new(k * nc);
            T::pack_panel(k, n, s0 * strip, nc, b, packed.as_mut_slice());
            panels.push(Panel {
                jb: s0 * strip,
                nc,
                packed,
            });
            s0 = s1;
        }

        let tail_cols = n - n_strip;
        let tail = if tail_cols == 0 {
            None
        } else {
            let mut t = AlignedVec::<T>::new(k * tail_cols);
            gather_columns(b, k, n_strip, tail_cols, t.as_mut_slice());
            Some(t)
        };

        Ok(MatmulPlan {
            k,
            n,
            n_strip,
            panels,
            tail,
        })
    }

    /// 行主序 `B`（`k × n`）的便捷构造，等价于
    /// `MatmulPlan::new(&MatRef::row_major(b, k, n)?)`。
    ///
    /// # Errors
    /// `b.len() != k × n` 或 `k == 0`。
    pub fn from_row_major(b: &[T], k: usize, n: usize) -> Result<Self, Error> {
        let view = MatRef::row_major(b, k, n)?;
        Self::new(&view)
    }

    /// `A` 的列数（= `B` 的行数），构造后固定。
    pub fn k(&self) -> usize {
        self.k
    }

    /// `B` 的列数（= `C` 的列数），构造后固定。
    pub fn n(&self) -> usize {
        self.n
    }

    /// 计划里那份打包好的 `B` 占多少字节（内存代价，便于和"每次重新打包"权衡）。
    pub fn packed_bytes(&self) -> usize {
        let elems: usize = self.panels.iter().map(|p| p.packed.len()).sum::<usize>()
            + self.tail.as_ref().map_or(0, |t| t.len());
        elems * std::mem::size_of::<T>()
    }

    /// 算 `C[m×n] = A[m×k] · B`，`m = a.len() / k`，输出是新分配的对齐缓冲。
    ///
    /// # Errors
    /// `a.len()` 不是 `k` 的整数倍（[`Error::Shape`]），或 `m×n` 溢出
    /// `usize`（[`Error::Overflow`]）。
    pub fn run(&self, a: &[T]) -> Result<AlignedVec<T>, Error> {
        let m = self.rows_of("MatmulPlan::run", a)?;
        let len = m.checked_mul(self.n).ok_or(Error::Overflow {
            op: "MatmulPlan::run",
            what: "m×n",
        })?;
        let mut c = AlignedVec::<T>::new(len);
        self.run_into(a, c.as_mut_slice())?;
        Ok(c)
    }

    /// 算 `C[m×n] = A[m×k] · B` 并写进调用方的缓冲，`m = a.len() / k`。
    ///
    /// 只借用 `&self`，所以可以把同一个计划 `Arc` 给多个线程，各自写**不相交的行段**
    /// （见模块文档的线程示例）。`a` 必须是行主序连续的 `m × k`；`c` 是行主序 `m × n`。
    ///
    /// # Errors
    /// - `a.len()` 不是 `k` 的整数倍；
    /// - `c.len() != m × n`；
    /// - `m×n` 溢出 `usize`。
    pub fn run_into(&self, a: &[T], c: &mut [T]) -> Result<(), Error> {
        let m = self.rows_of("MatmulPlan::run_into", a)?;
        let want = m.checked_mul(self.n).ok_or(Error::Overflow {
            op: "MatmulPlan::run_into",
            what: "m×n",
        })?;
        if c.len() != want {
            return Err(Error::Shape {
                op: "MatmulPlan::run_into",
                what: "c",
                expected: want,
                got: c.len(),
            });
        }
        if m == 0 || self.n == 0 {
            return Ok(());
        }
        // 校验完毕，交给"只算这些行"的内核（并行层用的是同一个入口）
        self.run_rows_into(a, c);
        Ok(())
    }

    /// 只算**给出的这些行**：`a`/`c` 已按行切好（长度分别是 `rows×k` / `rows×n`）。
    ///
    /// 面板与列尾都是计划里现成的，所以这里**零分配、零打包**——并行层用它把同一份打包好的
    /// `B` 分给多个 worker（每行仍是沿 `k` 单累加器升序，与 [`Self::run_into`] 逐位一致）。
    ///
    /// `c` 会被**完全覆盖**（含 `k == 0`：微内核的累加器从 0 起，行尾内核也会写）。
    ///
    /// # Panics
    /// `k == 0`，或 `c.len() != (a.len()/k) × n`。
    pub(crate) fn run_rows_into(&self, a: &[T], c: &mut [T]) {
        let rows = a.len() / self.k;
        assert_eq!(
            c.len(),
            rows * self.n,
            "块内 C 的长度应为 rows × n（{}×{}）",
            rows,
            self.n
        );
        if rows == 0 || self.n == 0 {
            return;
        }
        let m4 = rows / 4 * 4;
        for panel in &self.panels {
            T::packed_rows(self.k, self.n, panel.jb, panel.nc, &panel.packed, a, c, m4);
        }
        if let Some(tail) = &self.tail {
            let cols = self.n - self.n_strip;
            for i in 0..rows {
                let a_row = &a[i * self.k..(i + 1) * self.k];
                let c_row = &mut c[i * self.n + self.n_strip..(i + 1) * self.n];
                T::row_tail(a_row, c_row, tail.as_slice(), self.k, cols);
            }
        }
    }

    /// 从 `a` 的长度推 `m`：必须是 `k` 的整数倍。
    fn rows_of(&self, op: &'static str, a: &[T]) -> Result<usize, Error> {
        if !a.len().is_multiple_of(self.k) {
            return Err(Error::Shape {
                op,
                what: "a（长度必须是 k 的整数倍）",
                expected: a.len() / self.k * self.k,
                got: a.len(),
            });
        }
        Ok(a.len() / self.k)
    }
}

/// 通用打包：从**任意布局**的 `B` 里按 `STRIP` 列一条带取数。
///
/// 行主序时走连续拷贝，列主序/带跨距/转置视角时按元素取（一次 O(k·n) 的代价，
/// 换来回合调用时不再碰原矩阵）。两种写法写出的字节完全一样，所以结果与布局无关。
fn pack_panel_gathered<T: Copy>(
    jb: usize,
    nc: usize,
    strip: usize,
    b: &MatRef<'_, T>,
    dst: &mut [T],
) {
    let k = b.rows();
    let strips = nc / strip;
    for p in 0..k {
        let row = b.row(p);
        match row.as_slice() {
            Some(src) => {
                for s in 0..strips {
                    let d = (s * k + p) * strip;
                    let j = jb + s * strip;
                    dst[d..d + strip].copy_from_slice(&src[j..j + strip]);
                }
            }
            None => {
                for s in 0..strips {
                    let d = (s * k + p) * strip;
                    let j = jb + s * strip;
                    for r in 0..strip {
                        dst[d + r] = row.get(j + r);
                    }
                }
            }
        }
    }
}

/// 把 `B` 的第 `[j0, j0+cols)` 列收拢成行主序 `k × cols`（尾部列专用）。
fn gather_columns<T: Copy>(b: &MatRef<'_, T>, k: usize, j0: usize, cols: usize, dst: &mut [T]) {
    for p in 0..k {
        let row = b.row(p);
        let out = &mut dst[p * cols..(p + 1) * cols];
        match row.as_slice() {
            Some(src) => out.copy_from_slice(&src[j0..j0 + cols]),
            None => {
                for (j, o) in out.iter_mut().enumerate() {
                    *o = row.get(j0 + j);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::testutil::Lcg;

    /// 覆盖各种尾巴与面板组合：
    /// - `m` 不足 4 行（`m4 < m`）、`n` 不足一条带、条带余 8..31 列；
    /// - `(7, 64, 5)`：`n < STRIP`，一个完整条带都没有，全走列尾；
    /// - `(40, 64, 9000)` / `(8, 64, 5000)`：`n` 超过一个打包面板（f32 8192 列、f64 4096 列），
    ///   走多面板循环；
    /// - `(513, 448, 130)`：`k > K_CHUNK`，同时 `m` 有余行。
    const SHAPES: &[(usize, usize, usize)] = &[
        (1, 1, 1),
        (3, 5, 7),
        (4, 4, 32),
        (4, 4, 33),
        (5, 7, 40),
        (7, 64, 5),
        (8, 64, 37),
        (8, 64, 5000),
        (16, 33, 17),
        (33, 65, 31),
        (40, 64, 96),
        (37, 70, 100),
        (40, 64, 127),
        (40, 64, 9000),
        (64, 64, 64),
        (128, 128, 128),
        (513, 448, 130),
    ];

    fn seq(len: usize, seed: u64) -> Vec<f64> {
        let mut rng = Lcg(seed);
        (0..len).map(|_| rng.f64() * 2.0 - 1.0).collect()
    }

    fn bits_f32(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn bits_f64(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn f32_plan_matches_api_matmul_bit_for_bit() {
        for &(m, k, n) in SHAPES {
            let a: Vec<f32> = seq(m * k, 0x51ed ^ (m * 7 + k) as u64)
                .into_iter()
                .map(|x| x as f32)
                .collect();
            let b: Vec<f32> = seq(k * n, 0xbeef ^ (k * 13 + n) as u64)
                .into_iter()
                .map(|x| x as f32)
                .collect();
            let want = crate::api::matmul(m, k, n, &a, &b).unwrap();
            let plan = MatmulPlan::from_row_major(&b, k, n).unwrap();
            assert_eq!(
                plan.packed_bytes(),
                k * n * 4,
                "{m}×{k}×{n}: 打包缓冲应正好是 B 的一份拷贝"
            );

            let got = plan.run(&a).unwrap();
            assert_eq!(bits_f32(&got), bits_f32(&want), "{m}×{k}×{n}: run");
            assert_eq!(got.len(), m * n, "{m}×{k}×{n}: 输出长度");

            // run_into 与 run 必须给出同一批字节；反复调用也必须稳定
            let mut c = vec![f32::NAN; m * n];
            plan.run_into(&a, &mut c).unwrap();
            assert_eq!(bits_f32(&c), bits_f32(&want), "{m}×{k}×{n}: run_into");
            plan.run_into(&a, &mut c).unwrap();
            assert_eq!(bits_f32(&c), bits_f32(&want), "{m}×{k}×{n}: 第二次调用");
        }
    }

    #[test]
    fn f64_plan_matches_api_matmul_bit_for_bit() {
        for &(m, k, n) in SHAPES {
            let a: Vec<f64> = seq(m * k, 0x1234 ^ (m * 7 + k) as u64);
            let b: Vec<f64> = seq(k * n, 0x9876 ^ (k * 13 + n) as u64);
            let want = crate::api::matmul_f64(m, k, n, &a, &b).unwrap();
            let plan = MatmulPlan::from_row_major(&b, k, n).unwrap();
            let got = plan.run(&a).unwrap();
            assert_eq!(bits_f64(&got), bits_f64(&want), "{m}×{k}×{n}");
            assert_eq!(plan.packed_bytes(), k * n * 8, "{m}×{k}×{n}");
        }
    }

    #[test]
    fn f32_plan_accepts_column_major_and_transposed_b() {
        let (m, k, n) = (40, 64, 100);
        let a: Vec<f32> = seq(m * k, 7).into_iter().map(|x| x as f32).collect();
        let b_row: Vec<f32> = seq(k * n, 9).into_iter().map(|x| x as f32).collect();
        let want = bits_f32(&crate::api::matmul(m, k, n, &a, &b_row).unwrap());

        // 列主序：b_col[j*k + p] = b_row[p*n + j]
        let mut b_col = vec![0f32; k * n];
        for p in 0..k {
            for j in 0..n {
                b_col[j * k + p] = b_row[p * n + j];
            }
        }
        let by_col = MatmulPlan::new(&MatRef::col_major(&b_col, k, n).unwrap()).unwrap();
        assert_eq!(bits_f32(&by_col.run(&a).unwrap()), want, "列主序 B");

        // 同一个缓冲区换成"行主序 n×k 再转置"的视角，应当完全等价
        let by_tr = MatmulPlan::new(&MatRef::row_major(&b_col, n, k).unwrap().transpose()).unwrap();
        assert_eq!(bits_f32(&by_tr.run(&a).unwrap()), want, "转置视角 B");

        // 带行跨距的行主序 B（每行后面 5 个元素的 padding）
        let stride = n + 5;
        let mut padded = vec![0f32; k * stride];
        for p in 0..k {
            padded[p * stride..p * stride + n].copy_from_slice(&b_row[p * n..(p + 1) * n]);
        }
        let by_stride =
            MatmulPlan::new(&MatRef::row_major_strided(&padded, k, n, stride).unwrap()).unwrap();
        assert_eq!(bits_f32(&by_stride.run(&a).unwrap()), want, "带跨距 B");
    }

    #[test]
    fn plan_reports_shape_errors() {
        let b = vec![0f32; 64 * 96];
        // k = 0：m 无法从 a 推出（视图本身是合法的 0×96）
        let empty_b = MatRef::<f32>::row_major(&[], 0, 96).unwrap();
        match MatmulPlan::new(&empty_b) {
            Err(Error::NotPositive { op, what, .. }) => {
                assert_eq!(op, "MatmulPlan::new");
                assert!(what.contains('k'), "{what}");
            }
            other => panic!("期望 NotPositive，得到 {other:?}"),
        }
        // B 本身的长度不对，在 MatRef 那一层就报
        assert!(MatmulPlan::from_row_major(&b, 64, 95).is_err());

        let plan = MatmulPlan::from_row_major(&b, 64, 96).unwrap();
        assert_eq!((plan.k(), plan.n()), (64, 96));
        // a 不是 k 的整数倍
        let a = vec![0f32; 40 * 64 + 1];
        assert!(matches!(plan.run(&a), Err(Error::Shape { .. })));
        // c 长度不对
        let a = vec![0f32; 40 * 64];
        assert!(matches!(
            plan.run_into(&a, &mut vec![0f32; 40 * 96 + 1]),
            Err(Error::Shape { .. })
        ));
        // 正确长度就通过
        let mut c = vec![0f32; 40 * 96];
        plan.run_into(&a, &mut c).unwrap();
        assert!(c.iter().all(|x| *x == 0.0), "零 A 应得零 C");

        // m = 0（空 A）与 n = 0 都是合法退化情形
        let mut empty = Vec::new();
        plan.run_into(&[], &mut empty).unwrap();
        let plan_n0 = MatmulPlan::from_row_major(&[], 64, 0).unwrap();
        plan_n0.run_into(&a, &mut []).unwrap();
    }

    #[test]
    fn plan_is_shareable_across_threads() {
        let (m, k, n) = (64, 64, 96);
        let a: Vec<f32> = seq(m * k, 21).into_iter().map(|x| x as f32).collect();
        let b: Vec<f32> = seq(k * n, 22).into_iter().map(|x| x as f32).collect();
        let plan = std::sync::Arc::new(MatmulPlan::from_row_major(&b, k, n).unwrap());
        let mut c = vec![0f32; m * n];
        std::thread::scope(|s| {
            let (top, bottom) = c.split_at_mut((m / 2) * n);
            for (range, part) in [(0..m / 2, top), (m / 2..m, bottom)] {
                let plan = std::sync::Arc::clone(&plan);
                let a = &a[range.start * k..range.end * k];
                s.spawn(move || plan.run_into(a, part).unwrap());
            }
        });
        assert_eq!(
            bits_f32(&c),
            bits_f32(&crate::api::matmul(m, k, n, &a, &b).unwrap())
        );
    }
}
