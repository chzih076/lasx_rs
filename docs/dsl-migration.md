# 旧 DSL 面 → `simd_dsl` / `ops_dsl` 迁移映射（**决策可见**）

> **本文只做分析与登记，不改任何代码。** 旧面 = 本仓（`lasx_rs`）现有的宏/类型/方法；
> 新面 = `simd_dsl`（引擎 + `Backend` trait + 宏）与 `ops_dsl`（算子层）。
> **读法**：每行给 **旧项（文件:行）→ 用途/调用者 → 对应物（有/部分/无 + 证据名）→ 差距 →
> 处置（S1/S2/S3/不可迁）→ 风险**。
>
> 处置四档：**S1 = 可薄转发**（旧调用点改指向新宏/新方法，语义不变）；
> **S2 = 补能力后迁**（新面缺一条能力，先补再迁）；**S3 = 保留不动**（旧面自己就是最终形态）；
> **不可迁**（形态/承诺与新面冲突，明确不做）。
>
> 风险三类固定标注：**位精确**（是否触及逐位承诺）、**冻结**（是否触及 `docs/dev.md §19.7`）、**C ABI**。

## 1. 宏面（4 个宏 + 1 个属性宏，`macros/`）

| 旧项（文件:行） | 用途 / 调用者 | `simd_dsl`/`ops_dsl` 对应物 | 差距 | 处置 | 风险 |
|---|---|---|---|---|---|
| `matmul!`（`macros/src/lib.rs:970`） | 二维矩阵乘生成；调用者：`src/laplace.rs`、`examples/{formula_dsl,formula_vec}.rs` | **有**：`simd_dsl::matmul!`（形状名 + `;` 分界），`ops_dsl::matmul` | 旧面形状是**运行期 `usize`**（`Mat`/`MatBuf` 视图，编译期形状在 `const M/K/N` 上）⇒ 只能对**运行期路径**做等价转发；新面另有 `matmul_f32_pinned::<M,K,N>`（const 形状，**T1**）可承载旧面的 `const` 优势 | **S1**（运行期路径）+ **S2**（const 形状走 `_pinned`） | 位精确：**是**（须逐位断言或登记容差）；冻结：否；C ABI：否 |
| `dot!`（`macros/src/lib.rs:981`） | 一维点积；调用者：`src/shape/vec.rs`、`src/shape/f16.rs`、`src/laplace.rs` | **有**：`simd_dsl::dot!` / `dot_f32_pinned::<N>` | 旧面**累加次序**（一维形态的 K 方向）与新面 `dot!` 可能不同 ⇒ 逐位 vs 容差需**逐例登记** | **S1** + 容差登记 | 位精确：**是**（`dot` 换累加次序 ⇒ 容差类 + 给界）；冻结：否；C ABI：否 |
| `gemv!`（`macros/src/lib.rs:987`） | 矩阵-向量乘 `y[N] = w[N,K]*x[K]`；调用者：`src/shape/vec.rs:32`、`src/shape/f16.rs`、`examples/formula_vec.rs` | **部分**：`simd_dsl` 有 `matmul!`（二维）与 `dot!`（一维），**没有专门的 gemv 形态**；`ops_dsl::matmul` 也不含 | **缺**：(a) **一维输出形态**（`y[N]` 与 `x[K]` 的秩降），(b) `f16` 权重通道（`shape/f16.rs` 的 `F16Vec`/`F16Mat`，`gemv!` 目前**编译期报错**表示不支持 —— 见 `src/shape/f16.rs:13`） | **S2**（补"降秩形态"）／f16 面 **不可迁**（`simd_dsl` 无 f16 dtype，见 `docs/design.md` 的 dtype 演化规则） | 位精确：是；冻结：否；C ABI：否 |
| `#[fast_trig]`（`macros/src/lib.rs:1000`，`proc_macro_attribute`） | **无操作直通**（`macros/tests/fast_trig_passthrough.rs:39` 是行为证据）；`src/rewrite.rs:6` 注明"宏层集成留独立一轮" | **部分**：`simd_dsl::kernel!` + `Backend::build_kernel_f32` + `Route::Recurrence` 覆盖了它的**意图**（递推快档），但**没有同形的属性宏** | 形态不同：属性宏"标注一个函数 ⇒ 改其实现"，新面是"**调一次 `kernel!` 生成核**" ⇒ 语义等价但**不是薄转发** | **S2**（把"标注"改写成"调用"）或 **S3**（若旧消费者只有原型） | 位精确：否（`fast_trig` 是 opt-in 快档）；冻结：否；C ABI：否 |

## 2. 引擎 / 改写层

| 旧项（文件:行） | 用途 / 调用者 | 新面对应物 | 差距 | 处置 | 风险 |
|---|---|---|---|---|---|
| `rewrite::plan(&KernelPattern) -> Rewrite`（`src/rewrite.rs:84`）、`KernelPattern`(:53)、`Rewrite`(:70)、`TrigOp`(:31)、`StepKind`(:42) | 把核形态判成"可改写/不可改写"；调用者：`src/laplace.rs`、`examples/formula_dsl.rs`、`src/lib.rs` | **有（等价物）**：`simd_dsl::equivalence::guard_of` + `EnginePath` + `PrecisionClaim`；`kernel::build` 走 `Route` + `FallbackReport` | 旧面是**本仓自有的判据类型**；新面把判据搬到引擎（`guard_of`/`engine_claim`）⇒ 需**重写调用点**而非转发 | **S2** | 位精确：是（判据决定数值路径）；冻结：否；C ABI：否 |
| `rewrite::safe(om, dt, n)`(:110)、`safe_zone()`(:115) | 递推安全区判据 | **有**：`simd_dsl::safety::recurrence_safe`（唯一判据；阈值 `RECURRENCE_MAX_PHASE_STEP=1.5`、`RECURRENCE_MAX_PHASE=8192.0`） | 旧面阈值若与新面不同 ⇒ **必须核对数值**（安全区判据不能两套） | **S1**（转发到 `recurrence_safe`） | 位精确：是（回退口径）；冻结：否；C ABI：否 |

## 3. 冻结面：`shape::Prepared` 与 `apply*`（`docs/dev.md §19.7`）

| 旧项（文件:行） | 用途 / 谁在调 | 新面对应物 | 差距 | 处置 | 风险 |
|---|---|---|---|---|---|
| `Prepared<P, T, const K, const N>`（`src/shape.rs:507`） | **预打包 + 复用**的句柄 | **有（同构）**：`simd_dsl::packed::PackedB` + `Backend::pack_f32` / `matmul_f32_packed` | 新面 `PackedB` 是**不透明句柄**（后端私有布局）；旧面 `Prepared` 带 `Policy` 参数与 `const K/N` | **S3（保留为后端细节）** | 位精确：是；**冻结：是（§19.7）**；C ABI：否 |
| `apply<const M>`(:524/:575)、`apply_into`(:582)、`apply_dyn`(:529/:597)、`apply_dyn_into`(:534/:604)、`apply_pooled`(:587)、`apply_dyn_pooled`(:614) | 核应用的**全部调用点**由宏生成代码直接调用（见 `docs/dsl-macros-inventory.md`）⇒ **不能随手删** | **部**：`Backend::matmul_f32_packed` + `CallOpts{threads}` 覆盖"打包复用 + 线程" | 新面**没有** `const M` 的 `apply*` 家族（形状在类型里的那套）⇒ 需要 `_pinned` 家族补齐（**T1** 的延伸） | **S3 保留**（过渡期）+ **S2**（长期下沉到具体后端 crate） | 位精确：是；**冻结：是**；C ABI：否 |

**必须保留的理由（点名）**：`apply*` 是**宏生成的代码直接调用**的接口（`docs/dsl-macros-inventory.md`
的"生成代码调用了哪些类型名/方法"一节可核对）；删掉它等于让**已生成的公式 DSL 代码**失效 ——
这正是 `§19.7` 要求"先记再改"的原因。迁移期内它**只能作为旧后端的实现细节保留**，
新写的消费者一律走 `Backend::matmul_f32_packed`。

## 4. 运行期支撑

| 旧项 | 用途 / 调用者 | 新面对应物 | 差距 | 处置 | 风险 |
|---|---|---|---|---|---|
| `src/pool/`（行块线程池） | 大形状并行；调用者：`shape.rs` 的 `apply_pooled*` | **部分**：`simd_dsl::CallOpts{threads: Threads::{Auto,Single,Exact}}` 只**表达意图**，线程池在**后端**里 | 新面**不含线程池实现**（有意：引擎不碰线程）⇒ 池留在具体后端 crate | **S3** | 位精确：否（分块不改每块次序）；冻结：是（`Policy::threads` 在 §19.7）；C ABI：否 |
| `src/parallel.rs` | 归约/分块的并行原语 | 同上（后端内部） | 同上 | **S3** | 同左 |

## 5. 三问的结论

**Q1 `matmul!`/`dot!` 能否直接薄转发？**
**能，但必须选对路径**：旧形状名是**运行期 `usize`**（`Mat`/`MatBuf` 是运行期形状视图），
⇒ 应对到 **`simd_dsl` 的运行期路径**（`matmul_f32` / `dot_f32`），**不是** `_pinned`（const 形状）。
**位精确口径要逐例登记**：`matmul!` 的 `ikj`/分块次序若与旧面一致 ⇒ 可要求**逐位**；
`dot!` 换累加次序（多累加器）⇒ **容差类 + 给界**（`simd_dsl` 侧已有实测界可引用）。

**Q2 `gemv!` / `fast_trig!` / 属性宏各缺什么？**
- `gemv!`：缺**一维输出形态**（`y[N] = w[N,K]*x[K]`，秩降 + 广播语义）；f16 权重通道**不可迁**
  （新面无 f16 dtype；`simd_dsl` 的规则是"新 dtype 必须有默认实现"，见其 `docs/design.md`）。
- `#[fast_trig]`：缺**同形属性宏**；新面用 `kernel!` + `Route::Recurrence` **表达同一意图**，
  但形态是"调用"不是"标注"⇒ 消费者要改写法（`src/rewrite.rs:6` 已预告"宏层集成留独立一轮"）。

**Q3 `shape::Prepared`/`apply*` 与 `rewrite.rs` 迁移后归谁？**
- **`Prepared`/`apply*`：保留为旧后端的实现细节**（谁生成代码在调它 = 旧宏生成的公式代码），
  长期由"具体后端 crate"承接；新消费者用 `Backend::matmul_f32_packed` + `CallOptions`。
- **`rewrite.rs`：判据下沉到引擎**（新面 `equivalence::guard_of` + `safety::recurrence_safe`），
  本仓只留**薄的类型映射**（否则判据两套 ⇒ 数值路径不一致）。

## 6. S1 的最小验证方案（"同一份用户代码、转发前后结果一致"）

**载体**：`examples/formula_dsl.rs` + `examples/formula_vec.rs`（同一份用户代码，改一行宏名即可切换）。
**步骤**：
1. 同一输入数据（整数域 **+** 小数域各一组），旧面与新面**各跑一遍**，输出写到两块不同缓冲；
2. `matmul!`：断言 **`to_bits()` 逐位一致**（同一次序 ⇒ 逐位是应有结果；不一致则说明次序变了 ⇒ 必须登记）；
3. `dot!`：**先量掩码相对误差**（`|ref| > 1e-3·峰值`），若 `> 0` 则**登记为容差类 + 界**，
   而不是放宽断言 —— 登记位置：本仓 `docs/dsl.md` 的"等价类"小节（迁移期）+ 新面 `docs/bench.md` 的实测值；
4. **口径**：计时用同进程交替 A/B、≥5 对丢首对、中位 + 极差（新面 `docs/bench.md §5`）；
   正确性断言**放在计时之外**。

## 7. 条目统计

- **宏面 4 条**：S1 ×2（`matmul!`、`dot!`）＋S2 ×1（`gemv!`）＋S2/S3 ×1（`#[fast_trig]`）；
  f16 一维形态 **不可迁**。
- **引擎/改写 2 条**：S1 ×1（`safe`/`safe_zone`）、S2 ×1（`plan`/`KernelPattern`/`Rewrite`）。
- **冻结面 2 条**：S3 ×2（`Prepared`、`apply*` 家族）——**均触及 `§19.7`**。
- **运行期 2 条**：S3 ×2（`pool/`、`parallel.rs`）。
- 合计 **10 条**：**S1 ×3、S2 ×4（含 f16 不可迁的一半）、S3 ×4（含 1 处过渡）**。
