# DSL 第 1 层的 Rust 侧类型化面：清点与落点建议

> **性质**：**只读清点 + 落点建议**，不是设计定稿。基线提交 **`33b5c81`**（清点时工作区干净）。
>
> **写域**：仅本文件。**未改动** `docs/dsl.md`（`dsl-plan` 写域）、`docs/dev.md`、`docs/ops.md`
> 与任何 `src/**`；全部结论来自源码阅读 + `cargo doc` 已有的文档注释。
>
> **"第 1 层"的口径（本文自定，供对齐）**：相对 `matmul!`/`CMat` 所处的**矩阵层**，第 1 层指
> **标量·逐元素层**与**索引空间层**，它要能表达 (a) **逐元素超越函数**（`exp`/`sin`/`cos`）与
> (b) **索引空间上的归约/积分**（`sum_range`/`integral`）。仓库里此前**没有**"第 N 层"的既有命名
> （检索 `第 1 层` / `Layer 1` 在 `docs/`、`src/` 均零命中），所以这一口径来自任务说明而非既有文档；
> 若 `docs/dsl.md` 的分层定义与本文不同，**以 `docs/dsl.md` 为准**，本文只提供"贴上去用什么"。

---

## 0. 结论速览

| # | 问题 | 一句话结论 |
|---|---|---|
| 1 | 形状进类型的现有载体 | `src/shape.rs` 的 `Layout`/`Mat`/`MatBuf`/`MatDyn`/`Prepared` 全部**靠 const 泛型**做编译期形状检查，构造时**只查长度**；`Policy::threads` 是**零尺寸类型上的纯判据**（不建池、不派活） |
| 2 | 类型化样板 `CMat` | `CMat<'a, R, C>` / `CMatBuf<R, C>` + `mul_into<const N>`（`src/ops/cmatmul.rs:1278`/`:1298`/`:1317`）；编译期保证由 **`compile_fail` doctest** 作证（`:1265`），而 `shape::Mat` **缺这一条**（只有 `text` 示例，`src/shape.rs:41-48`） |
| 3 | `api::*` 与 `_into` 约定 | **有统一约定且基本一致**（24 个 `_into`，`pub fn <name>_into(<inputs…>, out: &mut [T]) -> Result<()>`），但 **`matmul`/`matmul_f64` 没有 `_into`**（只有 `matmul_scaled_into`，`src/api.rs:1044`），另有一处返回 `Result<f32>` 的例外（`:1869`）；逐元素算子只有 `axpy`（就地）与三个激活函数，**`ops` 里的逐元素原语是 `pub(crate)`** |
| 4 | 第 1 层缺口的库侧落点 | **缺两个"类型化的域"**：① 逐元素**没有**类型化视图/按 dtype 分派的调用点（`shape::vec` 只有构造 + 点积）；② **没有任何索引空间类型**（检索 `pub struct Range`、`Slice`、`Index`、`Span`、`Domain` 全部零命中），`sum` 只吃裸切片。**可复用的现成件**是 `ops::nn_math::exp_seq/exp_vec`（位精确 f32 `exp`，`pub(crate)`）、`laplace::Weighting`/`trapezoid_weights`、`rewrite::{KernelPattern, plan}`、`fast_trig::{fast_exp, fast_cos_sin}` |
| 5 | 规范化建议 | 构造失败统一 `Result<Self, Error>`（把 `CMat::new -> Option` 收进来）；补 `matmul(_f64)_into`；`combine::*` 的"以 `out.len()` 驱动 + `assert!`"要么加 `Result` 要么降为 `pub(crate)`；**"决定代码生成/形状的进类型，决定数值路线的必须是显式值"**这一条口径要写死；`compile_fail` doctest 补齐 |

---

## 1. 形状进类型的现有载体（`src/shape.rs`）

### 1.1 四个载体

| 载体 | 位置 | 形状在哪 | 一句话 |
|---|---|---|---|
| `Layout<const K, const N>` | `src/shape.rs:182` | **全在 const 求值** | 排版方案（条带/面板/列尾/k 分块/打包字节）编译期算完，常量见 `:186`（`STRIP`）、`:202`（`PANELS`）、`:204`（`TAIL_COLS`）、`:206`（`K_CHUNKED`）、`:208`（`PACK_BYTES`） |
| `Mat<'a, T, const R, const C>` / `MatBuf<T, const R, const C>` | `src/shape.rs:223` / `:228` | **类型参数** | 零开销：`&[T]` + 两个 const 标签（`:220-222` 的文档原话） |
| `MatDyn<'a, T, const C>` / `MatBufDyn<T, const C>` | `src/shape.rs:233` / `:239` | **列进类型、行是运行期字段** | DYN 档：`K/N`（模型结构）固定、batch 行数每批不同 |
| `Prepared<P: Policy, T: PackedKernel, const K, const N>` | `src/shape.rs:507` | **类型参数** | 内部持一个 `MatmulPlan`；`apply*` 零分配、零重打包 |

`Layout` 的编译期自检用 `const _: () = assert!(…)`（`src/shape.rs:211-216`）——这是"排版在编译期"
的**凭据**，而不是注释承诺。

### 1.2 编译期形状检查靠类型，不靠运行期断言

- 构造时**只查长度**：`Mat::new` 的文档原话是"构造时**只查长度**；形状对不对是类型的事"
  （`src/shape.rs:245`），失败走 `Error::Shape { op, what, expected, got }`（`src/shape.rs:249-260`）。
- 形状匹配由 `mul`/`apply` 的 const 参数强制：`Mat::mul<const N>(&self, rhs: &Mat<'_, T, B, N>)`
  （`src/shape.rs:514`）——`K` 写错就是 `E0308`。
- **但"编不过"目前只是文档里的 `text` 块**（`src/shape.rs:39-48`，示例报错文本），**不是**
  `compile_fail` doctest；真正把这条钉住的是运行期测试 `test_static_shapes_are_checked_by_types`
  （`src/shape.rs:976`），它验的是"类型推导出的行列数正确 + 与 `api::matmul` 逐位一致"，**并没有
  验"形状不匹配确实编译失败"**（这只能由 `compile_fail` doctest 或 trybuild 证）。

### 1.3 `Policy::threads` 语义

- 定义：`pub trait Policy: Send + Sync { fn threads(rows, k, n) -> usize; }`（`src/shape.rs:443-446`），
  文档写明"**纯判据：不建池、不派活**"（`:444`）；`1` = 不派活。
- 策略是**零尺寸类型**（`:433` "策略是类型，不是运行期参数"）：
  `Auto`（`:467`，走 `auto_threads`，机器并行度**缓存**在 `OnceLock`，`:457-464`）、
  `Single`（`:478`，恒 1、**完全不碰池**）、`Exact<const N>`（`:487`，`N.max(1)`，**不设工作量门限**）。
- 判据本体：`auto_threads`（`:420-431`）两条门限——每线程 ≥ 4 行（`by_rows < 2` 即回 1）与
  总工作量 ≥ `AUTO_MIN_WORK = 4_000_000`（`:402`）。
- 并行实现方式是"**切几个行块**"而不是"池开几个线程"（`:441-442`），`Prepared::threads::<M>()`
  （`:570`）把判据暴露成纯函数供诊断。

---

## 2. 类型化 API 的样板：`CMat` / `CMatBuf`（`src/ops/cmatmul.rs`）

### 2.1 形状怎么进类型

| 件 | 位置 | 形态 |
|---|---|---|
| `CMat<'a, const R: usize, const C: usize>` | `:1278` | 两个**公有字段** `re: &'a [f32]` / `im: &'a [f32]`，分平面 |
| `CMat::new(re, im) -> Option<Self>` | `:1287` | 只查 `re.len() == im.len() == R*C` |
| `CMatBuf<const R, const C>` | `:1298` | `re: Vec<f32>` / `im: Vec<f32>`，`zeros()`（`:1307`） |
| `CMat::mul_into<const N>(&self, rhs: &CMat<'_, K, N>, out: &mut CMatBuf<R, N>, route: Route)` | `:1315-1336` | **形状全靠三个 const 参数对齐**：`self: R×K`、`rhs: K×N`、`out: R×N` |
| f64 孪生 `CMat64` / `CMatBuf64` / `mul_into` | `:1340` / `:1360` / `:1377` | 同语义 |
| `pub enum Route { Direct, Karatsuba }` | `:39` | **数值路线是运行期值参数**（不是类型） |

公开路径：`src/lib.rs:114` 的 `pub use ops::cmatmul;` ⇒ 用户看到 `lasx_rs::cmatmul::{CMat, CMatBuf, CMat64, CMatBuf64, Route}`。
（`mod ops;` 本身是私有的，见 `src/lib.rs:115`。）

### 2.2 编译期检查怎么作证：`compile_fail` doctest

- **有**：`src/ops/cmatmul.rs:1265-1273` 是一段 ` ```compile_fail ` doctest（`2×3` 乘 `4×5`，注释写
  "`3 ≠ 4` ⇒ 类型错误"），文档在 `:1262-1264` 说明理由——**"仓库里没有 trybuild/compile-fail 机制，
  这是标准做法且不引新依赖"**。
- **正面用法**也是 doctest：`:1249-1260`。
- **位精确契约**：`:1275-1276` 写明 `mul_into` 与直接调 `cmatmul_f32_into_route` **逐位一致**，
  由测试 `c_dsl_bitwise_same_as_direct`（`:1407`）与 `c_dsl_length_check`（`:1469`）守。
- **`mod ops` 可见性**：`ops` 私有 + `pub use ops::cmatmul` ⇒ 这是"**新层直接做成 `pub mod` +
  在 `lib.rs` 再导出**"的既有先例（`src/lib.rs:100-120` 的模块清单）。

### 2.3 与 `shape` 层的两处不一致（样板自身的裂缝，落地时要选一边）

1. **构造失败类型**：`CMat::new -> Option<Self>`（`:1287`）vs `Mat::new -> Result<Self, Error>`
   （`src/shape.rs:249`）、`VecRef::new -> Result`（`src/shape/vec.rs:52`）、`V3Ref::new -> Result`
   （`src/shape/batch/v3.rs:22`）。`Option` 丢掉了 `op/what/expected/got` 四项诊断。
2. **数值路线的表达**：`Route`（`:39`）与 `laplace::Weighting`（`src/laplace.rs:46`）、
   `laplace::FusedMode`（`:270`）、`ops::rope::RopeMode`（`src/ops/rope.rs:67`）都是**运行期枚举**，
   而 `Policy`（`src/shape.rs:443`）是**类型**。第 1 层必须明确选哪一边（见 §5 第 4 条）。

---

## 3. `api::*` 的现状与 `_into` 约定

### 3.1 matmul 家族

| 入口 | 位置 | 返回 |
|---|---|---|
| `matmul(m, k, n, a, b)` | `src/api.rs:967` | `Result<AlignedVec<f32>>`（**每次都分配**） |
| `matmul_f32_packed(…)` | `src/api.rs:982` | `Result<AlignedVec<f32>>` |
| `matmul_scaled_into(m, k, n, alpha, beta, a, b, c: &mut [f32])` | `src/api.rs:1044` | `Result<()>` ← **唯一的 matmul `_into`** |
| `matmul_f64(m, k, n, a, b)` | `src/api.rs:1083` | `Result<AlignedVec<f64>>` ← **没有 `_into`** |

`matmul_scaled_into` 的内部顺序是**范本**：`checked_mul` → `expect_len`（三个数组各一次）→
空形状早退 → 调内核 → `Ok(())`（`:1054-1076`）。

### 3.2 `_into` 约定：基本一致

统一形如 `pub fn <name>_into(<inputs…>, out: &mut [T]) -> Result<()>`，`src/api.rs` 里共 24 个：
`rms_norm_into`（`:389`）、`softmax_rows_into`（`:431`）、`layer_norm_into`（`:490`）、
`gather_rows_into`（`:552`）、`gather_rows_i8_into`（`:592`）、`silu_into`（`:706`）、
`gelu_quick_into`（`:719`）、`gelu_erf_into`（`:732`）、`rope_into`（`:753`）、
`norm3_batch_into`（`:1128`）、`batch_distance2d_into`（`:1156`）、`vec3_add_scaled_batch_into`（`:1230`）、
`cross3_batch_into`（`:1428`）、`unitize3_batch_into`（`:1490`）、`absmax_rows_into`（`:1853`）、
`quantize_i8_per_row_into`（`:1879`）、`quantize_i8_per_token_into`（`:1893`）、
`quantize_i8_per_channel_into`（`:1907`）、`dequantize_i8_into`（`:1941`）、
`dequantize_i8_rows_into`（`:1951`）、`gemv_i8_into`（`:1975`）、`matmul_i8_into`（`:2001`）。

**例外一处**：`quantize_i8_per_tensor_into(x, q: &mut [i8]) -> Result<f32>`（`:1869`）——把 scale 当
**返回值**而不是输出参数。

错误与长度：`Error` 枚举在 `src/api.rs:55`（`Shape{op,what,expected,got}` / `Overflow` / `NotFinite` /
`NotPositive`），`type Result<T>`（`:134`），校验件 `expect_len`（`:139`）与 `checked_mul`（`:188`）——
**新层的错误语义直接复用这四个变体与两个 helper 即可，不要新造。**

### 3.3 逐元素算子现状（"若有：axpy/add/scale"）

| 形态 | 位置 | 说明 |
|---|---|---|
| `ops::axpy(alpha, x, y)` | `src/ops/axpy.rs:15` | **`pub(crate)`**，就地 `y = alpha·x + y` |
| `api::axpy(alpha, x, y) -> Result<()>` | `src/api.rs:269` | 唯一公开的"就地逐元素"算子 |
| `api::silu` / `gelu_quick` / `gelu_erf` | `src/api.rs:658` / `:671` / `:689` | 返回 `AlignedVec<f32>`，**不返回 `Result`**（签名里就没有失败模式；口径写在 `docs/dev.md:5127`） |
| 上三者的 `_into` | `:706` / `:719` / `:732` | `Result<()>`，长度不符才失败 |
| `ops::nn_math::{exp_seq, exp_vec}` | `src/ops/nn_math.rs:55` / `:70` | **位精确 f32 `exp`**（标量串 + 向量版），`pub(crate)`；`silu`/`gelu` 共用 |
| `ops::sum(x) -> f32` | `src/ops/sum.rs:15` | `pub(crate)`；`api::sum`（`src/api.rs:217`）是它的 `pub` 包装，**不返回 `Result`**（空切片返回 0，全定义域） |
| `cmatmul::combine::{sub_f32, add_f32, sub3_f32, …}` | `src/ops/cmatmul.rs:663`（`pub mod combine`），函数在 `:687`/`:714`/`:741`/`:769`/`:796`/`:823` | **唯一 `pub` 的逐元素原语**，但**无类型**：长度取 `out.len()`，输入用 `assert!(a.len() >= n…)` **panic** |
| C ABI | `src/ffi/reduce.rs:47`（`lasx_axpy`） | 归约/axpy 的 FFI 面只有 4 个符号（`:12`/`:24`/`:35`/`:47`） |

**没有的东西**：公开的 `add`/`scale`/`mul` 逐元素算子、任何 `exp`/`sin`/`cos` 的公开入口
（`sin`/`cos` 在库里只出现在 `ops::rope` 的**表生成**里，且走平台 libm，`src/ops/rope.rs:197-219`；
`src/api.rs:801`/`:853` 明确"表生成本身不在位精确承诺内，`sin`/`cos` 来自平台 libm"）。

---

## 4. 第 1 层缺口的库侧落点（**最关键**）

### 4.1 (a) 类型化逐元素 `exp`/`sin`/`cos`：缺什么

**缺口 A1：逐元素没有类型化视图，也没有按 dtype 分派的调用点。**
`shape::vec` 只提供"构造 + 点积"：`VecRef<'a, T, const N>`（`src/shape/vec.rs:43`）、`VecBuf<T, N>`（`:74`）、
`DotPair`（`:149`）与落点 `dot_of`（`:163`）。**没有任何一元/二元 `map`**。

**缺口 A2：位精确 `exp` 现成但不可达。**
`ops::nn_math::exp_seq`（`src/ops/nn_math.rs:55`）与 `exp_vec`（`:70`）就是"位精确 f32 `exp`"，
但 `pub(crate)` 且**只有标量/向量两个标量级入口**，没有切片级驱动。f64 侧、`sin`/`cos` 侧**完全没有**
库内实现（`src/ops/` 检索 `sin`/`cos` 只命中 `rope` 的表参数）。

**缺口 A3：数值路线没有类型化/显式化的载体。**
`fast_trig` 提供两条可选路线（表驱动 `fast_cos_sin` `src/fast_trig.rs:73`、`fast_exp` `:108`；
递推 `recurrence_row` `:134`），契约是"**与 libm 不逐位一致 ⇒ 必须显式 opt-in**"（`src/fast_trig.rs:6-11`），
但**没有类型或参数表达"我要哪条路线"**——只有 `rewrite::KernelPattern::fast_trig_opt_in: bool`
（`src/rewrite.rs:65`）这个孤立的布尔。

**缺口 A4：递推路线目前是拉普拉斯形状的，不是通用逐元素形状。**
`laplace_batch_f32_recurrence<const M, const N, const NB>(sig, om, dt, f) -> (Vec<f32>, Vec<f32>, bool)`
（`src/laplace.rs:245`，f64 在 `:395`，融合在 `:289` + `FusedMode` `:270`）吃的是"每行一个 `(σ_j, ω_j)`"
的**核矩阵构造**语义，不是"给一个等距网格上的缓冲区算 `exp/cos/sin`"。

**缺口 A5：宏层的正式入口已被改判为"库侧类型化内建"。**
`#[fast_trig]` 现在是**无操作直通**（`macros/src/lib.rs:970-973`，原样返回 `item`），其文档
（`:962-969`）明确写："**快速超越函数的正式入口将改为库侧类型化内建（形状进类型、用户点名调用、
库内 `rewrite::plan()` + `recurrence_safe` 自动选 libm/递推）**；本属性降级为将来的糖"。⇒
**A4 的通用化就是这条改判要求的前置条件**；`rewrite::GENERATION_VIA_LIBRARY_ENTRY`
（`src/rewrite.rs:124`）也把生成侧指向库内入口而非宏。

### 4.2 (b) 索引空间 / 归约 `sum_range`/`integral`：缺什么

**缺口 B1：没有任何"索引空间"类型。**
检索 `pub struct Range|Slice|Index|Span|Domain` 在 `src/` **零命中**。等距网格现在是**散落的三个变量**：
`laplace::trapezoid_weights(n)` 只吃个数（`src/laplace.rs:30`）、
`rewrite::KernelPattern { base, step, n }`（`src/rewrite.rs:54-59`）、
`fast_trig::baseline_row(sig, om, dt, n, w)`（`src/fast_trig.rs:154`）里 `dt/n` 与 `n` 各自成对出现。
`ilaplace::DubnerAbate::omega(&self, k) -> f64`（`src/ilaplace.rs:135`）把频率索引空间表达成方法。

**缺口 B2：归约只吃裸切片、只有 f32、没有范围。**
`api::sum(x: &[f32]) -> f32`（`src/api.rs:217`）——**没有 f64 版、没有 `sum_range`、没有 typed 形式**；
`ops::sum` 是 `pub(crate)`（`src/ops/sum.rs:15`）。

**缺口 B3：积分权重是运行期枚举 + 每调用分配。**
`pub enum Weighting`（`src/laplace.rs:46`），其 `weights(self, n) -> (Vec<f64>, bool)`
（`:60`）**每次调用分配一个 `Vec<f64>`**，并把"是否回退梯形"塞进元组的第二项。

**缺口 B4：`matmul!` 层只表达"形状"，不表达"域"。**
`docs/dev.md` §9.5（`docs/dev.md:3139`）的结论："**能知道**：形状——类型级、编译期可查；
**不能知道**：取值范围"。第 1 层的索引空间要表达的正是"域 + 等距性"，这在现有类型里无处安放。

### 4.3 可以复用的现成件（不要重写）

| 现成件 | 位置 | 在第 1 层的角色 |
|---|---|---|
| `ops::nn_math::{exp_seq, exp_vec}` | `src/ops/nn_math.rs:55`/`:70` | 位精确 f32 `exp` 后端（提升可见性 + 加切片驱动即可） |
| `ops::sum` | `src/ops/sum.rs:15` | `sum`/`integral` 的归约后端 |
| `laplace::{Weighting, trapezoid_weights, recurrence_safe, RECURRENCE_MAX_PHASE*}` | `src/laplace.rs:46`/`:30`/`:204`/`:196`/`:201` | 求积权重 + **同源安全区守卫** |
| `laplace_batch_*_recurrence` | `src/laplace.rs:245`/`:395` | 递推 + 逐行回退 libm 的行为链路（`docs/dev.md` §9.5.2 `docs/dev.md:3199`） |
| `rewrite::{KernelPattern, plan, safe, safe_zone}` | `src/rewrite.rs:53`/`:84`/`:110`/`:115` | 识别 + 判定（纯函数，不生成代码）——**接新类型的天然接口** |
| `fast_trig::{fast_exp, fast_cos_sin}` | `src/fast_trig.rs:108`/`:73` | 非位精确快路线的后端 |
| `Policy` / `Exact<N>` 的"策略进类型"范式 | `src/shape.rs:443`/`:487` | 求积方案、精度路线的类型化模板 |
| `V2Ref::distance2d_into(&self, px, py, out: &mut VecBuf<f32, N>)` | `src/shape/batch/dist2d.rs:49` | **"类型化层不返回 `Result`"** 的先例（长度已在类型里） |

---

## 5. 规范化建议（供设计文档与后续整顿）

1. **构造失败统一为 `Result<Self, Error>`**：把 `CMat::new`/`CMat64::new`（`src/ops/cmatmul.rs:1287`/`:1349`）
   的 `Option` 改成 `Result`，与 `Mat::new`（`src/shape.rs:249`）、`VecRef::new`（`src/shape/vec.rs:52`）、
   `V3Ref::new`（`src/shape/batch/v3.rs:22`）、`F16Vec::new`（`src/shape/f16.rs:36`）一致。
   理由：`Option` 丢掉 `Error::Shape` 的 `op/what/expected/got`，而 `docs/dev.md` §19.5 已把
   "错误消息带期望/实际"列为接口层的收益。
2. **补齐 `_into` 的空洞**：`matmul`/`matmul_f64`（`src/api.rs:967`/`:1083`）按 `matmul_scaled_into`
   （`:1044`）的形状补 `matmul_into`/`matmul_f64_into`；`quantize_i8_per_tensor_into -> Result<f32>`
   （`:1869`）改成 `out_scale: &mut f32`，让"所有输出都在参数里"这条不留特例。
3. **逐元素原语的失败语义统一**：`cmatmul::combine::*`（`src/ops/cmatmul.rs:687` 起）目前
   "`out.len()` 驱动 + `assert!` panic"。要么加 `Result`（与 `api::*_into` 一致），要么显式降为
   `pub(crate)` 并在文档里说明"只给库内组合步用、长度由调用方保证"——**不能保持"pub 且 panic"**。
4. **写死一条分层口径**（第 1 层最需要）：**"决定代码生成/形状/调度的进类型；决定数值路线且需要
   显式 opt-in 的进显式值参数"**。于是：形状/长度/网格点数 ⇒ `const`（`Mat`/`VecBuf`/`Grid<const N>` 的做法）；
   求积方案（梯形/Simpson）⇒ 可以进类型（对齐 `Policy`，`src/shape.rs:443`）；
   精度路线（libm 位精确 / 表 / 递推）⇒ **必须是显式值参数**（对齐 `Route`，`src/ops/cmatmul.rs:39`），
   因为它改变数值、`docs/dev.md` §9.5.1（`docs/dev.md:3171`）已规定"**必须显式 opt-in、默认仍 libm/位精确**"。
5. **"谁守"逐条点名**（AGENTS.md 规则 5）：新层照抄 `src/shape/vec.rs:8-15` 与
   `src/shape/batch/mod.rs:24-31` 的表格式，至少覆盖四项——① 长度/形状 = **类型系统**；
   ② 输入输出不重叠 = **借用检查器**（`&VecRef` vs `&mut VecBuf`）；③ 数值一致 = **同一内核 + `to_bits()` 测试**；
   ④ 超范围回退 = **运行期守卫**（`laplace::recurrence_safe`，`src/laplace.rs:204`）。
6. **`compile_fail` doctest 补齐**：`CMat` 有（`src/ops/cmatmul.rs:1265`），`shape::Mat` 只有 `text` 示例
   （`src/shape.rs:41-48`）。给 `Mat::mul`、以及新层的 shape 不匹配路径各加一条，作为"编译期保证"
   的**凭据**（零依赖、rustdoc 标准做法——这条理由已经写在 `src/ops/cmatmul.rs:1263`）。
7. **命名与访问器统一**：现有 `<View>`/`<View>Buf` 规则很整齐（`VecRef`/`VecBuf`、`Mat`/`MatBuf`、
   `V3Ref`/`V3Buf`、`F16Vec`/`F16Mat`、`CMat`/`CMatBuf`）⇒ 第 1 层沿用（如 `Grid`/`GridBuf` 或
   `Span`/`SpanBuf`）；访问器统一 `len()/is_empty()/as_slice()/as_mut_slice()`（`src/shape/vec.rs:57-70`、`:86-104`）；
   构造统一 `new()`；就地写回统一 `*_into`。

---

## 6. 第 1 层落点建议清单

> 下表是**草案**（签名未定稿，供 `docs/dsl.md` 决策）。每条都标注与 `CMat`/`matmul!` 的一致方式。
> 落点文件按**现有模块边界**选，尽量不新增 crate、不新增第三方依赖（`Cargo.toml` 依赖只有内部
> `lasx_rs_macros`，见 `Cargo.toml:28-32`）。

| # | 缺口 | 落点（文件） | 签名草案 | 与 `CMat`/`matmul!` 的一致性 |
|---|---|---|---|---|
| L1 | A1 | `src/shape/elem.rs`（**新**，`lib.rs` 加 `pub mod shape::elem` 或并入 `shape::vec`） | `pub struct ElemBuf<const N: usize>(VecBuf<f32, N>);` + `impl VecRef<'_, f32, N> { pub fn exp_into(&self, out: &mut VecBuf<f32, N>); }` | 视图/缓冲成对命名；`new()` 构造**只查长度**（`src/shape.rs:245` 的同一口径）；**类型化层不返回 `Result`**（对齐 `V2Ref::distance2d_into`，`src/shape/batch/dist2d.rs:49`） |
| L2 | A2 | `src/api.rs`（先做 `pub` 包装，不动 `ops` 可见性） | `pub fn exp_into(x: &[f32], out: &mut [f32]) -> Result<()>`；`pub fn exp(x: &[f32]) -> AlignedVec<f32>` | 完全对齐 `silu`/`silu_into`（`src/api.rs:658`/`:706`）：**有失败模式才 `Result`**；后端复用 `ops::nn_math::exp_seq/exp_vec`（`src/ops/nn_math.rs:55`/`:70`） |
| L3 | A2 | `src/ops/transcend.rs`（**新**，或并入 `ops/nn_math.rs`） | `pub(crate) fn sin_cos_f32(x: &[f32], c: &mut [f32], s: &mut [f32])` —— 位精确基准走 libm，SIMD 留给后续 | 与 `ops::{silu,gelu_erf}` 同层同风格；**必须**在文档里点名"sin/cos 不在位精确承诺内"（`src/api.rs:801`/`:853` 已立此口径） |
| L4 | A3 | `src/shape/elem.rs`（与 L1 同文件） | `pub struct BitExact; pub struct Fast;` + `pub trait Precision { const NAME: &'static str; }`（或按 §5 第 4 条 做成显式**值** `pub enum Route { Libm, Table, Recurrence }`） | **二选一**：进类型 = 对齐 `Policy`（`src/shape.rs:443`）；进值 = 对齐 `Route`（`src/ops/cmatmul.rs:39`）。按 §5 第 4 条 的口径，**数值路线建议走值 + 必填** |
| L5 | A4 | `src/ops/recur.rs`（**新**）；`laplace_batch_*_recurrence` 改为它的消费者 | `pub(crate) fn recurrence_rows<const N: usize>(g: &Grid<N>, args: &[f64], op: TrigOp, out: &mut [f32]) -> bool`（返回 `all_safe`） | 与 `laplace::laplace_batch_f32_recurrence`（`src/laplace.rs:245`）同源——**守卫与逐行回退必须复用 `laplace::recurrence_safe`**（`src/laplace.rs:204`），不各写一份（`docs/dev.md` §9.5.1 的"同源"要求，`docs/dev.md:3191`） |
| L6 | A5 | 不改宏；把 `rewrite::KernelPattern`（`src/rewrite.rs:53`）的 `base/step/n` 换成 `&Grid<N>` | `pub fn plan(g: &Grid<N>, op: TrigOp, lin_in_index: bool, opt_in: bool) -> Rewrite` | `plan` 保持**纯函数、不生成代码**（`src/rewrite.rs:77-83`）；`#[fast_trig]` 继续做无操作占位（`macros/src/lib.rs:970`），直到库侧入口可用 |
| L7 | B1 | `src/shape/index.rs`（**新**，或 `src/shape/domain.rs`） | `pub struct Grid<const N: usize> { base: f64, step: f64 }` + `pub fn new(base: f64, step: f64) -> Result<Self, Error>`；`pub fn at(&self, i: usize) -> f64`；`pub fn is_uniform(&self) -> bool` | 与 `MatDyn` 同构（**"个数进类型、数值参数是运行期字段"**，`src/shape.rs:233`）；**注意硬约束：Rust 不允许 f64 作 const 泛型参数**，所以 `base/step` 只能是运行期字段、构造时校验一次 |
| L8 | B2 | `src/shape/vec.rs`（增补）或 `index.rs` | `impl<T: Copy + Default, const N: usize> VecBuf<T, N> { pub fn sum(&self) -> f64; pub fn sum_into(&self, out: &mut [f64]); }` | 复用 `ops::sum`（`src/ops/sum.rs:15`）；**注意 `api::sum` 只有 f32**（`src/api.rs:217`），f64 侧是新增；`sum` 无失败模式 ⇒ **不返回 `Result`**（对齐 `api::sum`） |
| L9 | B3 | `src/laplace.rs`（就地改造）或 `src/shape/quad.rs`（**新**） | `pub struct Trapezoid; pub struct Simpson;` + `pub trait Quadrature { fn weights<const N: usize>(&self, g: &Grid<N>, w: &mut [f64]); }` | 从 `pub enum Weighting`（`src/laplace.rs:46`）**升为类型**，对齐 `Policy`（`src/shape.rs:443`）；顺带**去掉每调用 `Vec` 分配**（`:60`）与 `(Vec<f64>, bool)` 元组返回 |
| L10 | B2+B3 | `src/shape/vec.rs` | `pub fn integral_into<Q: Quadrature, const N: usize>(f: &VecRef<'_, f64, N>, g: &Grid<N>, q: Q, out: &mut VecBuf<f64, N>)` | 与 `Prepared::apply_into`（`src/shape.rs:582`）同风格：`*_into` 写调用方缓冲、方案是类型参数、形状在类型里 |
| L11 | B4 | `docs/dev.md` §9.5（**不属本文写域**，列为需求） | — | `docs/dev.md` §9.5（`docs/dev.md:3139`）已判"暂不扩 DSL 语法、先把递推核 + opt-in 落成库内入口"⇒ L5/L6 **正是那句"下一轮"** |
| L12 | 基础设施 | `src/aligned.rs` 已够用；`Cargo.toml` **不加依赖** | — | 新层输出一律 `AlignedVec`/`VecBuf`（32 字节对齐），与 `api::*` 输出同一条约定（`src/shape/vec.rs:73`、`src/shape/vec.rs:15`） |

### 6.1 落地顺序建议（按依赖，不按价值）

1. **L7 `Grid<N>`**：它是 B 组一切的前提，且能立刻把 `rewrite::KernelPattern`（A4/L6）与
   `laplace` 的散落参数收敛进来——**纯新增类型，不动任何既有数值路径**。
2. **L2 `api::exp(_into)`**：后端现成（`ops::nn_math`），只差切片驱动 + 尾处理；做完之后
   `api::silu` 与它是同一形状的两条入口，第 1 层的"逐元素"就有位精确参照实现。
3. **L1 + L4 的类型化视图与精度路线**：需要先定 §5 第 4 条 那条口径（值 vs 类型），否则会像现在
   `Route`/`Weighting` 与 `Policy` 那样两套并存。
4. **L5/L6 递推通用化**：这是 `#[fast_trig]` 改判（`macros/src/lib.rs:962-969`）真正激活的前提；
   **必须**复用 `laplace::recurrence_safe` 与 `laplace_batch_*_recurrence` 的同源守卫。
5. **L9/L10 求积类型化**：把 `Weighting` 与 `trapezoid_weights` 收进类型层；顺带消掉每调用分配。

### 6.2 完成判据（建议写进 `docs/dsl.md`）

- 编译期保证：新层每处 shape/长度不匹配都要有 **`compile_fail` doctest**（对齐
  `src/ops/cmatmul.rs:1265` 的做法）。
- 数值保证：位精确路线与直接调 `api`/`ops` **逐位一致**（`to_bits()` 测试，对齐
  `c_dsl_bitwise_same_as_direct` `src/ops/cmatmul.rs:1407`）；非位精确路线与 libm 的**误差界实测**并进例外清单。
- 门禁：`bash scripts/doccheck.sh`（引用/表格列数/符号数）与 `cargo fmt --all --check` 全绿；
  新增公开符号数要与 `docs/ops.md` §4.6 的清点表同步。
