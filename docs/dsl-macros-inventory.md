# DSL 宏层清点（`macros/src/**`）——设计文档的输入

> **性质**：清点与结论，**不是决策记录**（决策进 `docs/dev.md`；本文件供 `docs/dsl.md` 引用与裁剪）。
> **基线**：`33b5c81`（`git rev-parse HEAD`）。**只读**清点；唯一写入是本文件。
> **证据约定**：正文用 `文件:行`；"展开物"来自 `-Zunpretty=expanded`，探针在 `/tmp/dsl_probe`（未入库，
> 复现命令见附录 A）。行号以基线为准。

---

## 0. 基线：这一层有什么

| 文件 | 行数 | 内容 |
|---|---|---|
| `macros/src/lib.rs` | 1050 | `matmul!`、`#[fast_trig]`、token 拼装/解析/诊断、二维生成 |
| `macros/src/vec.rs` | 640 | `dot!`、`gemv!`（一维形态），规则层与翻译层分家 |
| `macros/Cargo.toml` | 17 | `[dependencies]` **空**（无 `syn`/`quote`/`lasx_rs`） |
| `macros/tests/fast_trig_passthrough.rs` | 39 | `#[fast_trig]` 是"无操作直通"的行为证据 |

**入口清单**（`#[proc_macro]` / `#[proc_macro_attribute]`，全仓就这 4 个）：

| 入口 | 位置 | 形态 |
|---|---|---|
| `matmul!` | `macros/src/lib.rs:940-946` | 公式（`=` 可选、`alpha` 可选、右侧可为 plan） |
| `dot!` | `macros/src/lib.rs:951-954` → `macros/src/vec.rs:358` | 一维点积，标量输出 |
| `gemv!` | `macros/src/lib.rs:957-960` → `macros/src/vec.rs:397` | f16 权重 × f32 向量 |
| `#[fast_trig]` | `macros/src/lib.rs:970-973` | **属性宏，但原样返回 `item`**（占位） |

---

## 1. 现有能力

### 1.1 语法形态

文法（`macros/src/lib.rs:33-39`）：

```text
公式   := 操作数 '=' 操作数 '*' 操作数
        | 操作数 '*' 操作数
操作数 := ident '[' ident ',' ident ']'
        | ident                          ← 只在乘积右侧：视为 Prepared plan
标量   := ident | 字面量                ← 可选前缀，写在乘积最前面
```

三条键规则：

1. **下标首字符分档**（`macros/src/lib.rs:19-29`、`156-164`）：大写 = 编译期 `const`、小写或 `_` = 运行期值。
   这是选 proc-macro 而不是 `macro_rules!` 的**第一条理由**（`macro_rules!` 读不到 ident 文本）。
2. **角色由共享下标判**（`macros/src/lib.rs:41-50`、`598-606`）：`x[M,K]*w[K,N]` 与 `w[K,N]*x[M,K]` 都认，
   归一成 input/weight 后生成同一段代码。
3. **同一维必须写同一个下标**（`macros/src/lib.rs:686-713`），宏在展开期报双锚定错误。

一维形态（`macros/src/vec.rs:1-8`）：`dot!(x[K]*y[K])`、`gemv!(y[N] = w[N,K]*x[K])`、`gemv!(w[N,K]*x[K])`；
一维**只支持编译期长度**（`macros/src/vec.rs:11-12`、`132-147`）。

**明确不做**（`macros/src/lib.rs:52-54`）：`beta`（原地累加）、`+` 多操作数、无下标简写的静态/DYN 分档。
一维的 `+` 也有专门诊断（`macros/src/vec.rs:234-246`）。

### 1.2 展开产物（真实展开，非文档推断）

`matmul!` 全静态 —— `examples/formula_dsl.rs:21` 展开为：

```rust
{
    let _: &Mat<'_, _, { M }, { K }> = &x;
    let _: &Mat<'_, _, { K }, { N }> = &w;
    let _: &MatBuf<_, { M }, { N }> = &y;
    x.mul_into(&w, &mut y)
}
```

`matmul!` DYN（`examples/formula_dsl.rs:26`）：先 `let m = xd.rows();` + `assert_eq!(yd.rows(), m);`，
再三条类型标注，最后 `w.apply_dyn_into(&xd, &mut yd)`（展开见附录 A 的 `expanded_dsl.txt:35-53`）。

`matmul!` 的 `alpha` / plan 形态（探针 `/tmp/dsl_probe/src/main.rs`）：

```rust
matmul!(y1[M, N] = alpha * x[M, K] * w[K, N]);  →  x.mul_scaled_into(alpha, &w, &mut y1)
matmul!(y2[M, N] = alpha * x[M, K] * p);        →  p.apply_scaled_into(alpha, &x, &mut y2)
matmul!(y3[m, N] = xd[m, K] * p);               →  p.apply_dyn_into(&xd, &mut y3)   // 前面加 rows 断言
matmul!(x[M, K] * p);                           →  p.apply(&x)
matmul!(y[M, N] = x[M, K] * p);                 →  p.apply_into(&x, &mut y)
```

`dot!` / `gemv!`（`examples/formula_vec.rs` 展开）：

```rust
dot!(a[K] * b[K])            →  { lasx_rs::shape::vec::dot_of((&a, &b)) }
dot!(a[K] * wv[K])           →  同上（f16 反序也走同一个 impl）
gemv!(w[N, K] * a[K])        →  { let _: &lasx_rs::shape::F16Mat<'_, {N}, {K}> = &w;
                                  let _: &lasx_rs::shape::VecRef<'_, _, {K}> = &a;
                                  w.gemv(&a) }
gemv!(y2[N] = w[N, K] * a[K]) →  上面三条 + `let _: &mut lasx_rs::shape::VecBuf<_, {N}> = &mut y2;`
                                  + `w.gemv_into(&a, &mut y2)`
```

两条**生成代码的卫生差异**（现在是实打实的不一致）：

- `matmul!` 发射**裸类型名**（`Mat`/`MatBuf`/`MatDyn`/`MatBufDyn`，`macros/src/lib.rs:368-396`）⇒ 用户必须自己
  `use lasx_rs::shape::{…}`；
- `dot!`/`gemv!` 发射**绝对路径**（`macros/src/vec.rs:458-466`、`532-559`）⇒ 不需要任何 `use`。

### 1.3 形状如何在类型层表达

**就是 Rust const 泛型**（不是类型级整数库，没有 `typenum`）：

| 类型 | 形状进类型的方式 | 位置 |
|---|---|---|
| `Mat<'a, T, const R, const C>` / `MatBuf<T, const R, const C>` | R、C 全在类型里 | `src/shape.rs:223`、`228` |
| `MatDyn<'a, T, const C>` / `MatBufDyn<T, const C>` | 只把**列**放进类型，行是运行期字段 | `src/shape.rs:233`、`239` |
| `VecRef<'a, T, const N>` / `VecBuf<T, const N>` | 长度 `N` | `src/shape/vec.rs:43`、`74` |
| `F16Vec<'a, const N>` / `F16Mat<'a, const N, const K>` | 长度/形状（元素类型固定为 `u16` 位型） | `src/shape/f16.rs:27`、`69` |
| `V3Ref`/`V3Buf`/`RvMut`/`V2Ref` + `*Dyn` | SOA 批量样本维 | `src/shape/batch/v3.rs:11,54`、`rv.rs:13`、`dist2d.rs:7`、`dynamic.rs:23,30,200,305` |
| `CMat<'a, const R, const C>` / `CMat64` / `CMatBuf{64}` | 复数矩阵（**在 `ops` 层，不在 `shape` 层**） | `src/ops/cmatmul.rs:1278,1298,1340,1360`（重导出见 `src/lib.rs:114`） |

**派生量在 const 求值里算完**：`Layout<K,N>` 把条带/面板/列尾/k 分块/打包字节全做成关联 const
（`src/shape.rs:182-209`），并有编译期自检——**不满足就编译不过**：

```rust
const _: () = assert!(Layout::<256, 256>::PANELS == 1);        // src/shape.rs:212-216
const _: () = assert!(Layout::<1024, 4096>::K_CHUNKED);
```

dtype 面：`matmul!` 只覆盖 **f32/f64**（`PackedKernel` 的两个 impl：`src/plan.rs:142`、`182`）；
`alpha` 系列方法**只 f32**（`src/shape.rs:703`、`791`）；复数走独立的 `CMat` 手写路径，**DSL 不覆盖**。

### 1.4 编译期检查靠什么

四种机制，分工清楚：

| 机制 | 例子 | 位置 |
|---|---|---|
| **类型系统（主）** | `Mat<A,B>::mul_into(&Mat<B,N>, &mut MatBuf<A,N>)`：共享维 `B` 写错就是 E0308 | `src/shape.rs:512-521`；`Prepared::apply_into` `553-584` |
| **宏自己的名字层检查** | 同一维两侧不同名、同操作数两下标同名、下标个数不对 | `macros/src/lib.rs:686-713`、`466-487`；`macros/src/vec.rs:66-110` |
| **运行期断言（DYN 档）** | `assert_eq!(yd.rows(), m)`、`out.rows() == x.rows()` | 展开物；`src/shape.rs:605` |
| **`on_unimplemented`** | `dot!` 不支持的 dtype 对（如 f16·f16）给出可读提示 | `src/shape/vec.rs:144-148`，经 `dot_of` 的**元组**参数落地（`163-165`） |
| **`compile_fail` doctest** | 全仓**只有一处**（复数矩阵乘形状不匹配） | `src/ops/cmatmul.rs:1262-1265` |

**两条必须点名的"守不住"**：

1. **"大写下标必须是 const"宏守不了**：proc-macro **没有名字解析**，`M` 是不是 const 要等 rustc
   （`macros/src/lib.rs:124-126`，诊断表第 6 条：报 `cannot find value 'M' in this scope`）。
2. **负面诊断没有自动化验收**：仓库**故意不引 `trybuild`**（与"零第三方依赖"冲突），
   负面路径靠人工核对编译输出（`src/shape.rs:1126`、`src/ops/cmatmul.rs:1263`）。
   宏规则层能单测是因为被拆成了纯函数（`macros/src/lib.rs:975-1050`、`macros/src/vec.rs:571-640`）。

### 1.5 诊断能力（现有资产的真正价值）

`Diagnostic`（`macros/src/lib.rs:234-289`）与两条实现经验：

- **双锚定**：一条错误发两条 `compile_error!`，分别锚在两侧不一致的下标上（`251-260`、`274-284`）；
- **必须包成 block**：`matmul!` 会出现在表达式位置，裸着两条 `compile_error!` 解析不了（`266-269`）；
- **Span 必须铺满整棵小树**：只给字符串字面量设 Span 会指到整条宏调用（`305-310`）；
- **复用用户的 token**，只有样板用 `call_site()`，否则形状报错指不到公式里的位置（`131-138`）。

---

## 2. 宏基础设施：手写 token 流覆盖了什么

### 2.1 覆盖到的形态

| 类别 | 用到的 | 证据 |
|---|---|---|
| `TokenTree` | `Ident`、`Punct`、`Literal`、`Group` | 全仓（`grep TokenTree::`） |
| `Delimiter` | `Bracket`（形状）、`Parenthesis`（调用）、`Brace`（输出包裹）、`None`（类型标注） | `lib.rs:459`、`301-303`、`893`、`389` |
| `Spacing` | `Joint`（`::`、`'_`）、`Alone` | `vec.rs:470-475`、`562-568` |
| 输入的标点 | 只认 `*`、`=`、`,`（加一条 `+` 用于报错） | `lib.rs:441,509,472`；`vec.rs:215,238,408` |

两个踩过的坑（都写在代码里，值得设计文档继承）：

- `"(".parse::<TokenStream>()` **会失败**（不配对的定界符），括号必须用
  `Group::new(Delimiter::Parenthesis, …)` 构造（`lib.rs:136-138`）；
- `::` 的第一个 `:` 必须是 `Joint`，否则 rustc 读到两个独立冒号（`vec.rs:468-475`）。

### 2.2 没有 `syn`/`quote` ⇒ 表达边界（逐条带证据）

| 限制 | 证据 |
|---|---|
| **没有表达式树**：标量前缀是**单个 token** | `Formula.scalar: Option<TokenTree>`（`lib.rs:230`）；`parse_scalar` 只认 `Ident|Literal` 后跟 `*`（`lib.rs:577-588`）。探针：`(alpha + 1.0) * x[..] * w[..]` ⇒ 报「这里应该是操作数」 |
| **没有属性解析**：`#[fast_trig]` 的 `_attr` 完全不看 | `lib.rs:971`；文档明写"属性参数解析/表达式形状识别/描述符发射/接重写路径"**不做**（`lib.rs:969`）。探针：`#[fast_trig(step = 0.01, op = "exp")]` **能编过且参数被静默丢弃** |
| **没有语句/块遍历**：函数式宏只拿到自己那一段 token | 签名 `pub fn matmul(input: TokenStream)`（`lib.rs:941`）；属性宏拿到 `item` 但原样返回（`lib.rs:971-973`）⇒ **无法表达跨语句依赖** |
| **没有类型/泛型解析**：`annotate`/`type_ts` 只**发射**类型语法 | `lib.rs:364-396`；`vec.rs:532-559`。全仓没有一处解析 `Type` |
| **没有名字解析/类型信息** | `lib.rs:124-126`（诊断表第 6 条） |
| **没有路径解析**：只发射固定路径 | `path_of(["shape","vec","dot_of"])`（`vec.rs:458-466`） |
| **下标只能是裸标识符**：不是表达式 | `lib.rs:466-479`（`[Ident, ',', Ident]`）、`vec.rs:180-204`。探针：`x[M + 0, K]` ⇒ 报「下标必须是两个标识符」 |
| **参数列表只能是 `[i, j]`**：没有逗号分隔的表达式实参 | 同上；`expect_end` 只在结尾报"公式到乘积就结束了"（`vec.rs:234-246`）。`sum_range!(f, 0..N)` **无从落点** |
| **`macro_rules!` 不够用**（所以才上 proc-macro） | `lib.rs:19-22`：拿不到 ident 文本 ⇒ 读不到首字符 |

**可测性约束**（影响任何新原语的验收方式）：`proc_macro` 的 `TokenStream` 一旦离开真实宏调用就 panic，
所以**规则要拆成纯函数**才能单测（`macros/src/vec.rs:23-24`、`lib.rs:1045-1049`），
Span 层只能靠 example 编译核对。

---

## 3. 第 1 层缺口（最关键）

### 3.1 (a) 类型化逐元素 `exp!` / `sin!` / `cos!`

**缺口 A1：入口不存在。**
证据：全仓 `#[proc_macro]` 只有 `matmul`/`dot`/`gemv`（`lib.rs:940-960`）；`lasx_rs` 只重导出这三个
（`src/lib.rs:128`）。探针 `g_unary`：`lasx_rs::exp!(x)` ⇒ `error[E0433]: cannot find `exp` in `lasx_rs``。

**缺口 A2：现有文法容不下一元句式——它要求 `*` 两侧各一个带下标操作数。**
`*` 是必需的（`lib.rs:522-523`、`550`；`vec.rs:215`）。所以：
- 用 `matmul!` 写一元（1 个下标、无 `*`）⇒ 报「下标必须是两个标识符」（`lib.rs:474-478`）；
- 用 `dot!` 写单操作数 ⇒ 报「公式在这里就结束了，缺少 `* 操作数`」（`vec.rs:222-227`）。探针 `g_reduce` 的原始输出即这条；
- 用 `gemv!` ⇒ 左侧必须是 2 下标矩阵（`vec.rs:83-88`、`286-313`）。

**缺口 A3：库侧也没有可发射的逐元素面。**
`VecRef`/`VecBuf` 只有 `new/len/is_empty/as_slice`（`src/shape/vec.rs:47-112`）——**没有 map/unary**；
唯一按 dtype 分派的 trait 是给点积用的 `DotPair`（`src/shape/vec.rs:144-165`）。
⇒ 即使宏发 `x.exp()`，也没有这个方法；必须先补库侧类型化面。

**缺口 A4：参数/标量只能一个 token**（同 2.2 第一行）：`exp!(y[N] = x[N] * gain)` 这类"带参数的逐元素"
即使有了入口也写不出表达式。

**缺口 A5：范围信息不在类型里（这是"快路径"的根问题）。**
类型只有**形状**、**没有取值范围**（`docs/dev.md:3141-3143`）；所以"编译期挑表/挑递推"必须先有**显式范围声明**
（草案见 `docs/dev.md:3146-3162`，逐条给了"声明与生成代码一致"的三个条件）。
现在的安全区判据在库内：`rewrite::plan`（`src/rewrite.rs:84`）、`rewrite::safe`（`:110`）、
`rewrite::safe_zone`（`:115`，转发 `laplace::RECURRENCE_MAX_PHASE_STEP/_PHASE`，`src/laplace.rs:196,201`）。
`#[fast_trig]` 目前是**占位直通**（`lib.rs:970-973`），其 crate 文档明确"正式入口将改为库侧类型化内建，
本属性降级为将来的糖（可选）"（`lib.rs:962-969`）。

### 3.2 (b) 索引空间 / 归约 `sum_range!` / `integral!`

**缺口 B1：下标是"名字"，不是"索引表达式"。**
`dims: Option<(Ident, Ident)>`（`lib.rs:181`）在解析层被钉成 `[Ident, ',', Ident]`（`lib.rs:466-479`）。
探针 `g_index_expr`：`x[M + 0, K]` ⇒ 报「`x` 的下标必须是两个标识符」。
⇒ 没有 `0..N`、没有步长、没有仿射下标 `base + i*step` 的表达方式。

**缺口 B2：没有单操作数归约句式。**
`sum`/`sum_range!` 需要"一个操作数 + 一个索引空间"；现有两个入口都要两个操作数（见 A2 与探针 `g_reduce`）。
`docs/dev.md:4588` 已把这条记为"缺一个**单操作数归约**句式"（G1 只落地了二元的 `dot!`/`gemv!`）。

**缺口 B3：没有"逗号分隔的表达式实参"解析。**
`[i, j]` 之外，全仓没有解析 `(expr, expr, …)` 的地方（`lib.rs:466-479`；`vec.rs:209-231`）。
`sum_range!(x[K], 0..N)` 与 `integral!(f, a..b, step)` 都落不进任何现有分支。

**缺口 B4：索引空间没有类型级载体。**
可用的蓝图是 `Layout<K,N>`——**把索引空间的派生量做成关联 const、在 const 求值里算完 + 编译期自检**
（`src/shape.rs:182-216`）。但有一处硬限制：**const 泛型不能是 `f64`**，而索引步长天然是 f64；
现有建模是 `StepKind::Literal(f64)`（`src/rewrite.rs:42-49`）——**只能作为字面量 token / `const` 项传**，
不能进类型参数。若要把索引空间进类型，步长必须表示为**有理数对**（`STEP_NUM/STEP_DEN`）或整数化。
`docs/dev.md:4593` 的 G5（行内归约 + 广播 + 参数）是**另一个**原语，别和 B2 混。

**缺口 B5：跨语句依赖无法表达**（同 2.2 第三行）。
`integral!` 若要"识别模式 → 改写成递推"，判定输入（等距？线性？字面量步长？）来自**表达式形状**；
宏现在既看不到别的语句，也不能在展开期调库（见第 4 节），所以这条只能靠"生成的代码去调库内判据"。

### 3.3 三个缺口是同一件事的三层

| 层 | (a) 逐元素 | (b) 索引空间/归约 |
|---|---|---|
| **类型面（库）** | 无逐元素 trait（A3） | 无索引空间类型（B4） |
| **文法面（宏）** | 一元句式 + 参数表达式（A1/A2/A4） | 单操作数 + 索引表达式 + 表达式实参（B1/B2/B3） |
| **判据面（库）** | 范围声明 + libm/递推选择（A5） | 等距识别 + 安全区 + 回退（B5，`src/rewrite.rs`） |

---

## 4. 依赖环：`macros` **不**依赖 `lasx_rs`（确认）

**事实**：

- `macros/Cargo.toml:17` 是 `[dependencies]`，**下面为空**；全文件没有 `lasx_rs`，也没有 `syn`/`quote`，
  **也没有 `[dev-dependencies]`**（`macros/tests/fast_trig_passthrough.rs` 只用 `lasx_rs_macros::fast_trig` 自己）。
- 反方向存在：`src/lib.rs:128` `pub use lasx_rs_macros::{dot, gemv, matmul};`。

**⇒ 依赖环是硬的：`lasx_rs → lasx_rs_macros`，所以 `lasx_rs_macros → lasx_rs` 会成环。**

**对"宏能否复用库内 `rewrite::plan()`"的影响（结论）**：

1. 宏**不能在展开期**调用 `rewrite::plan()` / `recurrence_safe()` / `safe_zone()`——连名字都取不到
   （无依赖），更不能 `use` 库内类型（如 `KernelPattern`，`src/rewrite.rs:53`）。
2. 宏**能**做的是：把**描述符/调用**发射成生成代码（绝对路径 `lasx_rs::rewrite::plan(...)` 是可行写法，
   先例见 `macros/src/vec.rs:458-466` 的 `lasx_rs::shape::vec::dot_of`）。
3. 于是正确分层是：**判据（阈值、函数、常量）唯一留在库内**，生成代码去调用它。
   这条已经在本仓落地并被测试钉住：`rewrite::safe()` **转发** `laplace::recurrence_safe`
   （`src/rewrite.rs:110`）、`safe_zone()` **转发**库内常量（`:115`），测试
   `safe_zone_uses_library_constants` 断言"同一处定义、逐点一致"（`src/rewrite.rs:213-234`）；
   生成侧被明确指向库内入口 `laplace_batch_*_recurrence`（`src/rewrite.rs:119-125` 的
   `GENERATION_VIA_LIBRARY_ENTRY`；同一条见 `docs/dev.md:3205-3210`）。
4. **次级理由（推理，非实测）**：proc-macro 依赖要按**宿主**目标编译，而 `lasx_rs` 是
   loongarch64-only（`src/lib.rs` 平台要求）；即便环不存在，把整个 SIMD 库拖进宏 crate 的宿主构建
   也是坏方向。

**推论（设计文档可直接引用的一句）**：宏层的产出只能是**代码 + 描述符**，不能是**判定**；
凡"编译期分析"型的结论（范围、安全区、快慢路径）都必须以"库内函数 + 生成代码调用"的形式落地。

---

## 5. 建议的最小可行路径

> 每一档都标了**是否要新写 token 解析**与**工作量（小/中/大）**。工作量按"含测试与诊断"估。

### (a) 类型化逐元素 `exp!`/`sin!`/`cos!`

| 步 | 做什么 | 改动面 | 手写解析？ | 量 |
|---|---|---|---|---|
| a1 | 库侧加逐元素面：按 `DotPair`/`dot_of` 的**同一个形状**加 `UnaryOp` trait + `unary_of`（元组参数，unsupported dtype 走 `on_unimplemented`）；一等的两个 impl 是 `(&VecRef<f32,N>, Exp/Sin/Cos)`（f64 视需要） | `src/shape/vec.rs` 新增 ~80 行 + 测试 | 否（纯 Rust trait） | **小** |
| a2 | 宏侧加 1 个入口（如 `apply!`，或 `exp!`/`sin!`/`cos!` 三个薄壳共用一条展开）与**一元分支**：允许"1 个下标 + 无 `*` + 可选输出"。复用 `Diagnostic`/`annotate`/`require_const` | `macros/src/vec.rs` 或新模块；已有规则层可复用 | **是**（一元分支；不需要表达式解析） | **中** |
| a3 | （可选，独立一轮）opt-in 快路径：属性/声明 → 描述符 → 生成代码调库内入口 | 宏侧识别 + `#[fast_trig]` 真解析 | **是**（属性参数 + 表达式形状识别 = 要写表达式解析） | **大** |

**边界（必须写进设计文档）**：默认路径必须仍是 libm/位精确，快路径**只能显式 opt-in**
（`src/rewrite.rs:23-25`）；`alpha` 只 f32 的既有边界（`src/shape.rs:703`、`791`）不要再扩散到新原语。

### (b) 索引空间 / 归约

| 步 | 做什么 | 改动面 | 手写解析？ | 量 |
|---|---|---|---|---|
| b1 | 先只补**单操作数归约**（`sum`）：库侧 `shape/vec.rs` 加 `sum`/`SumPair`（形状同 `DotPair`），宏侧给 `dot!`/新 `sum!` 放行"单操作数"分支 | `src/shape/vec.rs` + `macros/src/vec.rs` | **是**（允许单操作数分支，仍不解析表达式） | **小–中** |
| b2 | 索引空间进类型：`IndexSpace<const START, const STEP_NUM, const STEP_DEN, const N>`（步长用有理数对绕开"const 泛型不能是 f64"），派生量照 `Layout` 的 const-eval + `const _: () = assert!` 写法 | `src/shape/`（新文件） | 否 | **中** |
| b3 | `sum_range!` / `integral!` 句式：`ident[space]`、范围与仿射下标、逗号分隔的表达式实参 | `macros/`（新解析分支） | **是**（要写一个小表达式解析器：range、`a + i*b`、字面量步长） | **中–大** |

**建议的落地顺序与理由**：**b1 → b2 → b3**（b1 就有独立价值且不需要表达式解析）；
**b3 应先有真实消费者再开工**——`docs/dev.md:4614-4625` 的纪律是"查过的否，不是没查过的待办"：
G2/G5/G6 在唯一外部消费项目里**全是 0 处调用**，G4/G7 因此直接否掉。
`integral!` 还与库内既有 `laplace_batch_*`（`src/laplace.rs:184-185` 起的 `matmul!` 组装）**语义重叠**，
新语法要么吃下它、要么明确切开，别两处都叫"积分"。

---

## 6. 建议的规范化清单（供设计文档与后续整顿）

### 6.1 命名

| # | 现状 | 问题 | 建议 |
|---|---|---|---|
| N1 | `Mat`/`MatBuf` + `MatDyn`/`MatBufDyn`（后缀 `Dyn`） | 与 `CMat`/`CMat64`（**后缀表精度**）冲突 | 精度一律进**类型参数**：`CMat<f64, R, C>`；`Dyn` 后缀保留 |
| N2 | `VecRef`/`VecBuf` vs `Mat`/`MatBuf` | 一维叫 `Vec*`、二维叫 `Mat*`，但都是"视图/拥有"两种 | 视图/拥有的后缀统一为 `Ref`/`Buf`（`Mat`→`MatRef` 会破坏冻结面，至少在文档里定名规则） |
| N3 | `F16Vec`/`F16Mat`（dtype 进名字） | 与"元素类型进参数"不一致 | **保留**（f16 在本库是 `u16` 位型，不是 Rust 类型）——但要在文档里写明这是**有意的例外** |
| N4 | batch：`V3Ref`/`V3Buf`/`RvMut`/`V2Ref` + `*Dyn` | `RvMut` 的 `Mut` 是第三套后缀 | 定名：`<shape><Ref|Buf|Mut><Dyn?>`，`RvMut` 视为 `RvRefMut` 的缩写并记录 |
| N5 | 宏名 `matmul!`/`dot!`/`gemv!` | 新原语命名未定 | 小写算子名 + `!`；`exp!`/`sin!`/`cos!`/`sum!` 同风格；**不要**用 `apply!` 作唯一入口（丢掉算子名） |
| N6 | 属性 `#[fast_trig]` | 与"快路径只是可选糖"的新定位不一致 | 若最终是"库侧内建 + opt-in"，建议改成描述**意图**而非**机制**的名字（如 `#[fast_math]`），并在文档里点明它不再承载判据 |

### 6.2 边界（每条都要标"谁守"，AGENTS.md 规则 5）

| # | 边界 | 谁守 | 位置 |
|---|---|---|---|
| B1 | 大写下标必须是 `const` | **rustc**（宏无名字解析） | `macros/src/lib.rs:124-126` |
| B2 | 同一维必须同名 | 宏 | `macros/src/lib.rs:686-713` |
| B3 | 一维只支持编译期长度 | 宏（诊断）+ 类型 | `macros/src/vec.rs:132-147` |
| B4 | plan 只能写乘积右侧 | 宏 | `macros/src/lib.rs:100-103`、`612-641` |
| B5 | `alpha` 只 f32、`beta` 无 DSL 写法 | 类型（方法不存在）/ 有意缺省 | `src/shape.rs:703,791`；`macros/src/lib.rs:52-54` |
| B6 | 不支持 dtype 对 | 类型系统 + `on_unimplemented` | `src/shape/vec.rs:144-148` |
| B7 | 负面诊断 | **只靠人工核对**（无 trybuild） | `src/shape.rs:1126` |

### 6.3 错误语义

1. **库侧**：`Error` 变体边界要统一——`Shape`（长度/形状不符）、`BadValue`（语义非法，如 `n_dims` 非偶数）、
   `NotPositive`/`NotFinite`（常数）三者不要互相借用；**已有一处有意的复用**：id 越界复用 `Shape`
   （理由与代价见 `docs/ops.md` §2.17），要把它写成"例外清单"而不是惯例。
2. **宏侧**：现在只有字符串消息，没有类别码。建议给每条诊断一个稳定类别（与 §19.8 的诊断表对齐），
   设计文档可以引用类别而不是抄文案。
3. **两套锚点规则要统一**：宏用"双锚定 + note"（`macros/src/lib.rs:251-260`），库用错误消息里的
   `op/what/expected/got`（`src/api.rs:57-66`）——设计文档里应写成"编译期错误锚公式、运行期错误带四元组"。
4. **"档不一致"要有专门措辞**：大/小写混用是最常见的误用，现在只在 `check` 的 note 里带一句
   （`macros/src/lib.rs:750-763`），值得升级成独立诊断类别。

### 6.4 生成代码的卫生（现在就不一致）

- **统一走绝对路径**：`dot!`/`gemv!` 已是（`macros/src/vec.rs:458-466`），`matmul!` 还是裸类型名
  （`macros/src/lib.rs:368-396`）。绝对路径不要求用户 `use`，报错也更贴生成点。
- **样板 ident 的 Span**：全部 `call_site()`（`macros/src/lib.rs:293-303`），用户 token 复用 —— 这条是硬经验
  （`131-138`），新原语必须继承。
- **表达式位置可用**：`matmul!` 能在表达式位置（`let y = matmul!(…)`），靠"包 block"实现
  （`macros/src/lib.rs:266-269`、`891-894`）——新原语同样要以"表达式"为最小保证。

### 6.5 验收方式

- **宏规则层**：纯函数单测（`macros/src/lib.rs:975-1050`、`macros/src/vec.rs:571-640`）——
  任何新原语都要按这个形状拆（规则层不碰 `Span`）。
- **展开物可审计**：`examples/formula_dsl.rs:1` 明写自己是 `-Zunpretty=expanded` 的观察对象。
  建议**每种形态一个 example**（静态/DYN/plan/alpha/一元/归约），让展开物进 code review。
- **负面路径**：用 `compile_fail` doctest（先例：`src/ops/cmatmul.rs:1262-1265`），
  **不要引 `trybuild`**（`src/shape.rs:1126` 的既有决定）。

---

## 附录 A：探针与复现（都在 `/tmp`，未入库）

```bash
# 1) 既有 example 的真实展开（仓库内置观察对象）
cargo rustc --release --example formula_dsl -- -Zunpretty=expanded > /tmp/expanded_dsl.txt
cargo rustc --release --example formula_vec -- -Zunpretty=expanded > /tmp/expanded_vec.txt
cargo rustc -p lasx_bench --release     -- -Zunpretty=expanded > /tmp/expanded_cli.txt   # plan 形态在第 4482-4484 行附近

# 2) 探针 crate（正例展开 + 4 个缺口的最小反例）
#    /tmp/dsl_probe/{Cargo.toml,src/main.rs,examples/g_*.rs}
cd /tmp/dsl_probe
cargo rustc --release -- -Zunpretty=expanded > /tmp/expanded_probe.txt   # alpha / plan / plan+DYN / 无输出
for ex in g_unary g_reduce g_index_expr g_scalar_expr; do
  cargo build --release --example "$ex" 2>&1 | grep -E '^error' | head -2
done
```

**四个缺口的实测输出**（基线 `33b5c81`）：

| 探针 | 写法 | rustc/macro 输出 |
|---|---|---|
| `g_unary` | `lasx_rs::exp!(x)` | `error[E0433]: cannot find `exp` in `lasx_rs`` |
| `g_reduce` | `dot!(x[K])` | `error: 公式在这里就结束了，缺少 `* 操作数`` |
| `g_index_expr` | `matmul!(y[M,N] = x[M + 0, K] * w[K,N])` | `error: `x` 的下标必须是两个标识符：`x[行下标, 列下标]`` |
| `g_scalar_expr` | `matmul!(y[M,N] = (alpha + 1.0) * x[M,K] * w[K,N])` | `error: 这里应该是操作数（形如 `x[M, K]`…）` |

**属性参数不解析的实测**：`#[lasx_rs_macros::fast_trig(step = 0.01, op = "exp")] fn f(…)` 在探针里
**编译通过**（`/tmp/dsl_probe/src/main.rs:11-14`）——参数被接受并静默丢弃，`lib.rs:971` 的
`_attr` 未使用即证据。
