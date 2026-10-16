# `lasx_rs` DSL 面「退化/删除」分析（目标：`lasx_rs` = **纯后端**）

> **口径（用户更正）**：迁移**不是换壳**。目标形态 = `simd_dsl`（引擎+宏+trait）→ **后端 `lasx_rs`**
> （trait 实现 + LASX 内核 + 原语 + C ABI）→ **算子库 `ops_dsl`**。
> ⇒ `lasx_rs` 里**任何 DSL 面都不该留**（宏 crate、`rewrite.rs`、DSL 规划、`shape::Prepared` 生成路径）。
> **本文只分析，不改代码。** 每条给 **文件:行 / 当前调用者 / 前置条件 / 验收判据**。

## 1. 删除清单（DSL 面）

| 项（文件:行） | 是什么 | 当前调用者 | 删除前置条件 |
|---|---|---|---|
| `macros/`（整 crate）：`matmul!`(`macros/src/lib.rs:970`)、`dot!`(:981)、`gemv!`(:987)、`#[fast_trig]`(:1000) | 旧 DSL 宏入口 | `src/{laplace,rewrite,shape}.rs`、`src/shape/{f16,vec}.rs`、`examples/{formula_dsl,formula_vec,laplace}.rs` | ① 所有调用点改走 `simd_dsl` 宏或**直接调后端方法**；② `ops_dsl` 补出缺口能力（§4）；③ `shape::Prepared` 生成路径已摘除（§2） |
| `src/rewrite.rs`（`TrigOp`:31、`StepKind`:42、`KernelPattern`:53、`Rewrite`:70、`plan`:84、`safe`:110、`safe_zone`:115） | 旧「核形态判据」层 | `src/laplace.rs`、`examples/formula_dsl.rs`、`src/lib.rs` | 判据**下沉引擎**（`simd_dsl::equivalence::guard_of` + `simd_dsl::safety::recurrence_safe`）；**安全区阈值逐值核对**（两套判据 ⇒ 数值路径不一致） |
| `src/plan.rs` 的 DSL 部分 | DSL 侧规划（与 `rewrite.rs` 配套） | `src/{laplace,rewrite}.rs` | 同 `rewrite.rs`；**只删 DSL 部分**，文件里若有非 DSL 规划工具则逐项判定 |
| `src/dsl_backend.rs`（`LasxBackend`，feature 门控） | **不是删除项**：它是"纯后端"形态的第一步（`impl simd_dsl::Backend`） | 自身 feature 门控 example | **保留，作为目标形态的落点**（见 §3） |

## 2. 降级清单（核心：`shape::Prepared` 与 `apply*`）

**先查"谁在调它"**（`src/shape.rs` 定义处之外的全部调用者）：

| 调用者（文件:行） | 性质 |
|---|---|
| `src/parallel.rs:1573`、`src/parallel.rs:1650` | **非宏调用者**（`p.apply_into(&x, &mut out)`：本仓自己的并行对照/基准代码） |
| `examples/laplace.rs:171` | **非宏调用者**（`prepared.apply_into(&mkc, &mut out)`） |
| `src/parallel.rs:1538`、`src/shape/batch/v3.rs:4` | 文档引用（非代码路径） |
| **旧宏生成的代码** | `matmul!` 的生成路径曾直接调 `apply*`（见 `docs/dsl-macros-inventory.md`） |

**结论**：存在**非宏调用者** ⇒ **不能随宏直接删**；建议 **`pub(crate)` 降级**（不再对外导出），
并**摘除"宏生成调用路径"**（宏删除后自然消失）。若将来 `parallel.rs`/example 也改用
`Backend::matmul_f32_packed` ⇒ 再降为**私有**或移入后端内部模块。

| 项（文件:行） | 建议 | 前置条件 |
|---|---|---|
| `Prepared<P,T,const K,const N>`（`src/shape.rs:507`） | 降 `pub(crate)` | 摘除宏生成路径；crate 内保留 `parallel.rs`/`examples/laplace.rs` 两个调用点 |
| `apply`(:524/:575)、`apply_into`(:582)、`apply_dyn`(:529/:597)、`apply_dyn_into`(:534/:604)、`apply_pooled`(:587)、`apply_dyn_pooled`(:614) | 随 `Prepared` 一起降级 | 同上 |
| `prepare`(:539)、`prepare_with`(:544) | 同上 | 同上 |

> **两条都是 `docs/dev.md §19.7` 冻结项改动** ⇒ **必须先在 `docs/dev.md` 的冻结小节记录**
> （触发条件 / 代价 / 影响面）再动代码；影响面 = 旧宏生成的全部公式代码（宏删除后归零）。

## 3. 保留清单（这些**属于后端**）

| 项 | 为什么属于后端 |
|---|---|
| `impl simd_dsl::Backend`（`src/dsl_backend.rs`） | 纯后端的**定义**：trait 实现是 `lasx_rs` 在新形态里的唯一身份 |
| LASX 内核（`src/arch/**`） | 手写 ISA 特化**只能在具体后端**（引擎/算子库一律架构中立） |
| `src/ops/**` 的**内核/原语**部分（37 个文件，含 `dot_i8`/`matmul_i8`/`dot_q4`/`gemv_i8`） | "该后端有能力做"的原语实现；**算子语义**搬 `ops_dsl`，**内核**留后端 |
| **C ABI 98 符号**（`src/ffi/*.rs`，`#[unsafe(no_mangle)] pub extern "C"`） | 外部消费者的**稳定契约** ⇒ 必须留；实现改为**内部直接实现、不经 DSL** |
| `src/parallel.rs`、`src/pool/` | 线程/分块是**后端内部**实现（引擎只表达 `CallOpts{threads}` 意图） |
| 位精确承诺（`docs/ops.md §2.5`） | 后端**硬承诺**：新路径要么与参考逐位一致，要么进例外清单 |

## 4. 搬运清单（`lasx_rs` 算子 → `ops_dsl`）

`src/ops/` 共 **37 个文件**（含 `testutil.rs`）。逐组给：**依赖什么 → `ops_dsl` 够不够 → 缺什么（待补项）→ C ABI 怎么留**。

| 算子（代表） | 现在依赖 | `ops_dsl` 够不够 | 缺什么（待补项） | C ABI |
|---|---|---|---|---|
| `silu.rs`、`gelu_erf.rs`、`gelu_quick.rs` | 逐元素 + `exp`/`erf` | **够**（`map_unary!`+`combine!`；`silu` 已 2 趟/1 暂存） | `erf` 若缺 ⇒ **`UnaryOp::Erf`** | 保留，改内部直调 |
| `dot.rs`、`dot_f16.rs`、`dot_f64.rs`、`dot_i8.rs`、`dot_q4.rs`、`gemv_i8.rs` | 一维归约 + dtype 通道 | **部分**（f32/f64 够；f16/i8/q4 无 dtype） | **f16/i8/q4 不迁**（`simd_dsl` 无这些 dtype） | 保留 |
| `matmul.rs`、`matmul_f64.rs`、`matmul_i8.rs` | 二维乘 + dtype | **部分**（f32 够；f64 有；i8 无） | i8 **不迁**；const 形状需 **`matmul_f32_pinned`（T1）** | 保留 |
| `softmax_rows.rs`、`rms_norm.rs`、`layer_norm.rs` | 行内归约 + 逐元素 | **部分**：有归约宏，**缺"按行的多趟组合"形态** | **行 batch 组合**待补 | 保留 |
| `rope.rs`、`gather.rs`、`axpy.rs` | 索引/就地/散射 | **缺**：新面**无就地算子宏**（就地是有意非目标） | 就地/散射**留在后端**（不迁） | 保留 |
| `cmatmul.rs` | 复数乘 | **够**：`simd_dsl::matmul_complex_f32`（Direct/Karatsuba） | 无 | 保留 |
| `laplace.rs`、`ilaplace.rs` | 核构造 + 乘 + 递推 | **部分**：正向已覆盖（`kernel!` + `ops_dsl::laplace`）；**逆向未做** | **`ops_dsl` 逆向算子**待补 | 保留 |
| 批量族（`batch_distance2d`/`ballistic_step`/`cross3_batch`/`rk4_j2_step_batch`/…） | 批处理 + 索引 | **缺**：新面按"逐元素/归约/矩阵乘"组织，**批量无对应形态** | **批量形态**待补（按真实消费者排） | 保留 |

**C ABI 处置（统一）**：98 个 `extern "C"` 入口**全部保留**，实现改为**后端内部直接实现**
（不经 DSL、不经 `Prepared` 生成路径）；**符号名与签名一个都不动**。

## 5. 顺序与前置条件（每步带验收判据）

1. **`ops_dsl` 补能力**（§4 待补项：行 batch 组合、`UnaryOp::Erf`、逆向算子、`_pinned` const 形状 = T1）
   —— 判据：新能力有**真实消费者** + `opcheck` 绿 + 有测试。
2. **`lasx_rs` 算子改实现**（语义搬 `ops_dsl`，内核留后端，C ABI 内部直调）
   —— 判据：**位精确不变**（逐位断言或已登记容差）+ **`nm -D` = 98** + CI 十四步 + 三端同步。
3. **删宏 / 删 `rewrite.rs` / 摘 `Prepared` 生成路径**（+ 在 `docs/dev.md §19.7` 记冻结项变更）
   —— 判据：`nm -D` 仍 98、位精确仍成立、CI 十四步、三端；宏 crate 删除后同步清理
   workspace 成员与 `[dependencies]`。

## 6. 风险表

| 项 | 触及位精确 | 触及 `§19.7` | 影响 C ABI | 处置 |
|---|---|---|---|---|
| `matmul!`/`dot!`/`gemv!` 删除（调用点改写） | **是**（累加次序 ⇒ 可能逐位→容差） | 否 | 否 | 逐例登记等价类 |
| `rewrite.rs` / `safety` 阈值搬迁 | **是**（决定数值路径） | 否 | 否 | 阈值逐值核对后再删 |
| `Prepared`/`apply*` 降 `pub(crate)` | 是（预打包复用路径） | **是** | 否 | **先记 §19.7 再改** |
| 98 个 C ABI 符号 | 是 | 否 | **是** | **不动**（实现改内部直调，签名/符号名不变） |
| `parallel`/`pool` | 否（分块不改每块次序） | 是（`Policy::threads`） | 否 | 保留为后端内部；如改先记 §19.7 |
| f16/i8/q4 dtype 算子 | 是 | 否 | 是 | **不迁**（留后端），C ABI 照旧 |

## 7. 统计

- **删除清单 4 条**（宏 crate / `rewrite.rs` / `plan.rs` 的 DSL 部分 / `dsl_backend.rs` 保留标注）；
- **降级清单 3 条**（`Prepared`、`apply*` 家族、`prepare*`）；
- **保留清单 6 条**；
- **搬运清单 8 组**（覆盖 `src/ops/` 37 个文件；其中 **5 组"部分/缺" ⇒ 待补项 5 项**）。
