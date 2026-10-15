# lasx_rs 算子与用法手册

本手册面向 `lasx_rs` 的**调用方**，描述逐个内核的语义、数值契约与调用方式：C ABI、安全
Rust API、常驻工作池、对齐缓冲，以及以 ggml 为参照的 NN 算子现状与压缩路线。

**文档信息**

| 项 | 内容 |
|---|---|
| 适用代码版本 | `master`；导出符号的计数、完整清单与逐符号原型见 [`docs/c-abi.md`](c-abi.md)（C ABI 的唯一权威） |
| 库形态 | LoongArch64 LASX 256 位 / LSX 128 位批量数值内核；零依赖（仅 std + `#![feature(stdarch_loongarch)]`，需 nightly）；产物为 `cdylib + rlib` |
| 性能数据的位置 | **不在本文档**：全部性能数据的唯一权威位置是 `docs/dev.md` §7；编译期常量与阈值的 A/B 对照在 `docs/dev.md` 的对应设计小节 |
| 平台约束的位置 | **不在本文档**：硬件勘误（LA664 原子操作丢失更新）、内核分支生命周期、消费 ONNX 契约引出的缺口与承诺边界，都在 [`docs/platform.md`](platform.md)——**调用方通常无需读它**（勘误的命中点在库内部，库侧已自带 `_db` 与回归测试） |
| 目标读者 | 调用方（C / Rust / Dart），以及需要核对数值契约或边界行为的人 |
| 可复核项 | 符号总数可用 §4.6 的一行 `nm` 清点命令复核；跨文档引用一律写作 `docs/xxx.md §N`，可机械校验 |
| 写作约定 | 数字的唯一来源、测量口径与引用格式见 §16.4 |

**阅读路径**（每节的完整内容见其标题）

| 目标 | 路径 |
|---|---|
| 只调用库 | §1 分层与调用方式 → §4 符号总表 → 目标算子的 §5.x / §6.x / §7.x（§5.9–§5.13 为 NN 侧算子） |
| 关心数值行为 | §2.1（逐位一致的精确含义）→ §2.5（冻结约定）→ 对应算子的契约节（§2.6–§2.11） |
| 上多核，或需要形状/计划层 | §8 Rust 安全层（视图/计划/形状）、§9 池与并行（含调度策略） |
| 评估 NN 侧能力边界 | §12 现状与缺口对照（N1 已落地 / N2 按需 / N3 int8 推理，消费者明确） |
| 无损压缩与带宽路线 | §13 结论与路线 |

## 1. 分层与调用方式

| 层 | 给谁用 | 形态 | 校验 |
|---|---|---|---|
| `lasx_rs::ffi`（`extern "C"` 符号） | C / Dart 等 FFI 调用方 | 裸指针 + `int` 长度 | 原始符号零校验（误用即 UB）；`_checked` 带 `int *status`；另有 4 个运行时元信息符号。**C 侧权威说明见 [`docs/c-abi.md`](c-abi.md)** |
| `lasx_rs::api`（纯 Rust，不导出符号） | Rust 调用方 | `&[T]`/`&mut [T]` 进出，返回 `Result<_, api::Error>`，输出是 `AlignedVec` | 形状、溢出、物理常数 |
| `lasx_rs::view` + `lasx_rs::plan`（纯 Rust，不导出符号） | Rust，矩阵类调用 | `MatRef`/`MatMut` 视图；`MatmulPlan` 把 `B` 打包一次反复用 | 视图/计划在**构造时**校验一次 |
| `lasx_rs::pool` + `lasx_rs::parallel`（纯 Rust，不导出符号） | Rust，要多核 | 池 + 闭包 / 固定切分策略 | 派活取 `&self`（内部排队）、形状不符 panic |

四层走**同一段内核**：`api`/`view`/`plan`/`pool` 只是校验/调度薄包装，结果与直接调
`lasx_*` 逐位一致（`api`、`plan`、`parallel` 都有逐位对照测试）。怎么选：

- Rust 普通场景用 `lasx_rs::api`；形状不符返回 `Err`，不是 UB。
- 矩阵形状参数容易记错时用 `lasx_rs::view`；同一个 `B` 要乘很多个 `A` 时用
  `lasx_rs::plan::MatmulPlan`（见 §8.1、§8.2）。
- Rust 且已自校验形状、要绕开校验开销：直接用 crate 根的 `lasx_rs::lasx_dot(..)` 等，
  与 `lasx_rs::ffi::reduce::lasx_dot(..)` 等价（历史重导出）。
- C / Dart 用 `lasx_*`；需要把错误上抛给上层时用 `lasx_*_checked`。
- 多核：Rust 用 `pool::WorkerPool` + `parallel`，**池建一次、跨调用复用**。
  C/Dart 侧本库不提供线程池，自行切子区间后分别调 `lasx_*`（池不新增导出符号）。

## 2. 数值契约与逐位确定性

### 2.1 "逐位一致"的精确含义

分为两层，不可混同：

1. **同一次内核实现内部**：向量块与标量尾**同式、同结合序**，故任一分块的向量部分与
   标量尾部分逐位一致。这对逐样本（element-wise）批量内核成立。
2. **跨实现对照**：与测试中采用不同公式的独立标量参考对比，承诺是**相对误差 <1e-9**
   （不是逐位相等）。

例外与前提：

- `lasx_ballistic_step` 向量路径与标量 `euler_step` **非逐位一致**：`|v|²` 结合序不同
  （向量右结合 `vx² + (vy² + vz²)`，标量左结合 `(vx² + vy²) + vz²`）；物理量已对齐，
  有测试守护（见 6.4）。
- `lasx_axpy` 向量路径是单条 FMA（一次舍入），标量尾是 `mul`+`add`（两次舍入），至多差
  1 ulp，**无逐位守护**。
- `lasx_dot` 在 n < 24 时走纯标量 f32 累加，与 n ≥ 24 的 f64 分块累加路径公式不同，
  跨 n 边界不保证逐位连续。
- 三条路径（LASX / LSX / 标量尾）在设计上同式同结合序，故实际运行结果一致
  （可用 `lasx_force_lsx_thread` 验证）。

### 2.2 结合次序（硬约定）

| 量 | 结合序 |
|---|---|
| `norm3`/`unitize3`/`j2_accel`/`rk4_j2_step` 的 `\|r\|²` | `x² + (y² + z²)`（右结合） |
| `ballistic_step` 向量路径的 `\|v\|²` | `vx² + (vy² + vz²)`（右结合；标量参考为左结合） |
| 四元数范数 | `((w² + x²) + y²) + z²`（左结合） |
| 四元数乘法每个分量 | 左结合，如 `((aw·bw − ax·bx) − ay·by) − az·bz` |
| `mat3_mul_vec3` / `quat_rotate` 每行 | `((m0·x + m1·y) + m2·z)`（左结合） |
| 叉积分量 | `ay·bz − az·by` 等，先乘后减 |
| `j2_accel` 末项 | FMA：`fma(k·x, 5·zr2 − 1, vcen·x)` |
| `vec3_add_scaled` | 单次 FMA：`fma(s, b, a)` |

### 2.3 精度一览

| 内核类别 | 累加/近似方式 | 误差量级 |
|---|---|---|
| f32 点积/求和 | 每 256 元素把 f32 累加器落盘、升 f64；最终一次 `as f32` | 相对 f64 参考 <1e-5 |
| f32 矩阵乘 | 每个输出元素沿 k 的**单个 f32 累加器**（无 f64 落盘） | 相对 f64 参考 <1e-4 |
| f64 内核 | 全程 IEEE 双精度，FMA 单次舍入 | <1e-12（matmul_f64）/<1e-9（批量物理） |
| int8 点积 | 整数精确（i16 乘 → i32 累加 → i64 落盘），返回截断为 i32 | 精确（mod 2³²） |
| Q4 点积 | 组内整数精确；scale 为 f32×f32 再升 f64 | scale 精度约 2⁻²⁴ |
| `softmax_rows` | `exp` = `exp2` 的 6 次多项式 + magic 数取整（§2.6） | 行内权重与 f64 参考 <1e-6 |
| `rms_norm` | `fma(x,x,acc)` 平方和 + 8 lane 固定次序归约（§2.7） | 相对 f64 参考 <1e-5 |
| `silu`/`gelu_quick` | 复用同一串 `exp`，直接除法 `x/den`（§2.8） | 绝对 ≤7.6e-6（`\|x\| ≤ 20` 时 1.9e-6）、相对 1.6–1.7e-7 |
| `gelu_erf` | A&S 7.1.26 的 `erf` 近似（**不是真 erf**，§2.9） | 近似本身 2.1e-7 + f32 舍入；端到端 <1e-6（`\|x\| ≤ 16`） |
| `rope` | 每对两次舍入（`mul` + `fma`，§2.10） | 与 f64 参考绝对 <1e-6；保长度相对 <1e-6 |
| `dot_f16`/`gemv_f16` | f16→f32 **精确**转换 + 4 条链（§2.11） | 转换零误差；总误差来自 f16 量化 ≈4.9e-4 |

> 表里的"误差量级"都来自**实测**（各算子的单测与探针；精度包络的推导与全集读数见
> `docs/dev.md` §20.4、§20.5、§7）。`exp` 的端到端相对误差随 `|x|` 增长（`≈ |x|·4e-8 + 2e-7`），
> 上表取的是各算子实际用到的区间。

### 2.4 退化输入与边界约定

- 零向量单位化（`unitize3_batch`）：`|v| == 0` 输出 `(0,0,0)`，不是 `0·∞ = NaN`。
- 退化四元数（`quat_normalize_batch`，以及内部先单位化的 `quat_rotate`/`quat_to_dcm`）：
  `|q| < 1e-15` 输出单位四元数 `(1,0,0,0)`，不是 NaN；单位化用 `q·(1/|q|)`
  （只做一次除法再乘，见 `docs/dev.md` §13.6），
  不是"乘倒数"。
- `api::sum` 对空切片返回 `0.0`，对任何切片有定义，故不返回 `Result`。
- 长度为 0：`_checked` 允许空指针；`lasx_alloc(0)` 返回"对齐但不可解引用"的非空悬垂指针。
- 批量几何/姿态内核**允许输出与输入别名**（向量路径先取完该轮输入再写回），故
  `quat_to_dcm_batch` 的输出可直接喂给 `mat3_mul_vec3_batch`。


### 2.5 契约（2026-09-24 定稿）

本节是**冻结的约定**：改内核、加路径、上公式 DSL 均不可违反。每条均有测试覆盖。

**1）累加顺序。** 每个输出元素是**沿 `k` 的单个累加器、`k` 升序、每步一条 FMA**：

```text
c[i][j] = fma(a[i][k-1], b[k-1][j], fma(… fma(a[i][0], b[0][j], 0) …))
```

`alpha`/`beta`（若启用）只能在**累加结束之后**按固定次序施加。任何"改写结合次序"的优化
（把 `k` 拆成多个并行部分和再相加、用 `beta·y` 给累加器播种、跨 lane 水平归约）都**违反契约**。

**2）路径无关性。** LASX / LSX / 标量尾 / 打包面板 / k 分块 / 并行切块 / `MatmulPlan` /
`shape` 层，对同一输入给出**逐位相同**的结果（测试用 `to_bits()` 比对）。

成立的前提有两条，更换后端或编译选项时需重新检查：

- 所有中间结果**以与目标类型相同的精度存储**（IEEE 754 单/双精度）；
  **不存在扩展精度寄存器**（如 x87 的 80 位）导致的"寄存器里更精确"的差异。
  龙芯 LSX/LASX 是标准 IEEE 754，满足该前提。
- k 分块只是把**同一个部分和**分段写入 `C` 再读回：浮点数的存储/读取是精确的，
  故分块不改变任何一位。

**3）融合语义（已启用，2026-10-15）。** 支持 `C = alpha·A·B + beta·C`（原地）：

- **`beta` 默认 `0.0`**（覆盖输出，对齐 BLAS 的 `gemm`），**`alpha` 默认 `1.0`**；
- 施加次序固定为 `C = fma(beta, C_old, alpha·acc)`（两次舍入），**不是** BLAS 那种
  "用 `beta·C_old` 给累加器播种"（那会改变累加结构，违反第 1 条）。`acc` 与不带缩放的
  `lasx_matmul` 是**同一个**沿 `k` 的 f32 累加，所以融合与"先算完再整体乘"**逐位一致**
  （`to_bits()` 全量对照，6 形状 × 6 组 `α/β`）；
- **`beta == 0` 时 `C` 一个字节都不读**（CBLAS 语义：`C` 无需在入参里有效）——测试用 NaN
  探针守这条；
- **性能承诺只有一条**：`beta == 0` 时省掉对 `C` 的一整遍读写，**不改写 GEMM 内部结构**。
  实测（与"朴素两遍法"进程级交替）：宽薄档 −10.3% / −23.7%、小 `k` 累加档 −6.5%、
  方阵 ≈0；`beta != 0` 且 `k > 384`（打包路径要 k 分块、部分和会覆盖 `C_old`）**退回与
  二遍法同构**的实现，不假装更快。其他章节不得出现"融合接近 BLAS 语义""融合带来 2×"
  这类暗示；口径与边界详见 `docs/dev.md` §19.15；
- **入口**：Rust 侧 `api::matmul_scaled_into`、C ABI `lasx_matmul_scaled`（+ `_checked`，§4.7）、
  以及公式 DSL 的标量前缀（§5.5）。口径与边界详见 `docs/dev.md` §19.15；

**4）逐位一致的例外（仅此两处，均有测试）。** `lasx_ballistic_step` 的向量/标量分支
结合序不同（物理一致，绝对误差有测试覆盖）；`lasx_axpy` 允许 1 ulp 以内差异。


### 2.6 NN 侧（N1 批次）的数值契约：`lasx_softmax_rows`

`exp` 没有硬件指令，只能用多项式近似，因此该算子的契约需要明确"近似到什么程度、哪些
部分是逐位承诺、哪些不在契约内"。

**1）语义。** 行内 softmax（`rows × cols` 行主序）：

```text
out = exp(clamp_lo(scale·x + mask − 行内 max)) · (1/Σ)
```

- `mask` **可选**、同形状、**加性**（attention 的 `+ mask`）；C 侧用空指针表示无 mask；
- **先减行内 max 再 `exp`**（数值必需，否则溢出）；
- 归一化是 `e · (1/Σ)`（**一次除法 + 乘法**），不是逐元素除法——参考实现必须同样写
  `e * (1.0 / sum)` 才逐位一致。

**2）`exp` 的定义（该定义即契约本身）。** `exp(x) = exp2 的 6 次多项式(f) · 2ⁿ`：

```text
t = x·LOG2E;  n = (t + 12582912.0) − 12582912.0;  f = t − n     // |f| ≤ 0.5
p = Horner(f, (ln2)^k/k! , k ≤ 6)（全程 FMA）
kk = clamp(trunc(n) + 127, 0, 254);  exp(x) = p · bitcast_f32(kk << 23)
```

- **`exp` 端到端**的相对误差（含 `x·log2e` 那次 f32 乘法）：`|x| ≤ 1` 时 ≤ 2.1e-7、
  `|x| ≤ 8` 时 ≤ 6.2e-7、`|x| ≤ 20` 时 ≤ 1.1e-6、`|x| ≤ 87` 时 ≤ 3.9e-6
  （模型 ≈ `|x|·4e-8 + 2e-7`；本节正文里"多项式 ~1e-10"说的是**中间量** `2^f`，
  调用方拿到的是端到端结果，以这张包络为准）；
- **下溢恰好落到 0**：`x < −126.5/log2e = −87.6831` 时 `kk = 0` ⇒ `from_bits(0) = +0.0`
  （阈值由 `n = rint(x·log2e)` 的取整边界决定，**不是**数学上 `2^−127` 的 `−88.03`）。
  因此 `mask = −inf`（"完全屏蔽"）给出的是**精确的 0 权重**。输入在 `exp` 入口夹到
  `[−104, +88]`（既保证这一点，也避免 magic 数取整失效与浮点转整型越界）；
- **`exp` 自带夹取 ⇒ 它是全函数**：下溢侧精确 0，上溢侧 `exp(88) ≈ 1.65e38` 有限且可复现
  （不是 `inf`）。入口夹取已在算子内完成，调用方**不需要**再夹一次。
- 这套序列**标量与向量共用**，且行内求和用"8 个 per-lane 累加器 + 固定次序两两归约"，
  列尾用 `−1e30` 填充后走同一条 lane-wise 路径 ⇒ **向量路径与标量模拟逐位一致**
  （单测覆盖，`src/ops/softmax_rows.rs`）。设计推导见 `docs/dev.md` §20.1。

**3）不在契约内。** 输入或 `mask` 含 `+inf`：`inf − inf = NaN`，属定义域问题，算子不处理。

> 该契约由测试保障：逐位一致（标量模拟）、显式补齐 == 列尾路径、f64 参考精度、
> 退化输入（行内全相等 / `scale=0` / `−inf` mask / 极负值不出 NaN）。
> 后续 N1 算子（`rms_norm`、`silu`/`gelu`、`rope`、`dot_f16`）沿用同一套写法：
> **先在本节固定 op 序列与"谁守"的归属，再实现代码**。


### 2.7 NN 侧：`lasx_rms_norm`

与 §2.6 共用同一套"位精确骨架"（8 个 per-lane 累加器 + 固定次序两两归约 + 列尾填充），
只列**不同**的部分：

```text
s   = Σ_j x[j]²                    // 每步一条 fma(x, x, s)：一次舍入（不是 acc + x*x 的两次）
ms  = s / cols                     // 一次除法
inv = 1 / sqrt(ms + eps)           // 一次 sqrt + 一次除法
out = (x · inv) · w                // 次序固定：先 inv 再权重
```

- **列尾补 `0.0`**：`0² = 0`，对平方和**精确无贡献**（比 softmax 的 `−1e30` 填充加夹取更简洁，
  无需任何夹取）；累加仍是 `fma`，故"同一条 op 序列"这一性质在填充 lane 上也成立。
- **行级数学在标量域算完再广播**：每行一次 `s/cols`、一次 `sqrt`、一次 `1/x`。这既省指令，
  也**天然位精确**——不存在"向量 `sqrt` 与标量 `sqrt` 是否逐位相同"这一问题。
- **开方与倒数用精确指令**，不用 `frsqrte`/`frecipe` 估算 + 牛顿：`docs/dev.md` §13.1 已实测
  本机估算指令比精确除法**更慢**（6.8 周期 vs 4.0）；该结论直接复用，不再重测。
- `w` 可选（C 侧空指针），**长度为 `cols`**（按列的权重），每行复用。
- **不在契约内**：`eps = 0` 且整行全零 ⇒ `inv = +inf` ⇒ `out = 0·inf = NaN`。这是定义域
  问题（`eps` 就是为此存在），`api` 与 `_checked` 都要求 `eps` 有限且 `> 0`。

**由谁守**：逐位一致（标量模拟单测，覆盖 `cols` 1–17 的边界扫描与 1×1…6×257 多种形状）、
`fma` 契约（同上，模拟用 `mul_add`）、`cols` 边界、f64 参考精度 <1e-5（相对）、
`eps > 0`（`api`：`NotPositive`；`_checked`：`NonPositiveConstant`）。

**实测**：`12.01–18.38 GB/s`（单线程，2 条流），是朴素 Rust 行循环的 2.6–3.5×。与 softmax 的
差别是**遍数**（rms_norm 两遍 = 2 条流，softmax 三遍 = 3 条流），按每条流折算两者一致。
数字与锚点对比见 `docs/dev.md` §7.4。


### 2.8 NN 侧：逐元素激活 `lasx_silu` / `lasx_gelu_quick`

逐元素、**无归约**，因此位精确是构造性的：不存在"两种结合次序"，只要标量与向量执行同一串
op 即逐位相同。真正需要固定的是**公式的等价形式**。

**1）语义（两个算子只差一个系数 `c`）。**

```text
t   = c · x               // silu: c = 1.0（此时 1.0·x 精确）；gelu_quick: c = 1.702
den = 1 + exp(−t)         // exp 自带夹取 [−104, +88]，见 §2.6
y   = x / den             // 直接除法：单次舍入
```

- `lasx_silu` = SiLU/swish；`lasx_gelu_quick` = GELU 的 sigmoid 近似（`x·σ(1.702x)`），
  与 ggml 的 `GELU_QUICK` 同式。erf 形式（PyTorch 默认的 `0.5x(1+erf(x/√2))`）是另一个入口，
  规划见 `docs/dev.md` §20.4 第 3 条；
- **契约中的关键一条：写直接除法 `x / den`，而非 `x · (1/den)`**。两者数学等价、末位可能不同；
  参考实现必须照抄直接除法才算逐位一致。（softmax 的情形与之相反：其分母由所有列共享，一次
  倒数换 N 次除法是划算的；此处每个元素各自一次，没有共享机会。）

**2）由谁守**：逐位一致（`src/ops/silu.rs`、`src/ops/gelu_quick.rs` 的"标量模拟"单测，覆盖
`n` 1..=17 的边界宽度与 64/65/127/129 的多轮主体循环）；原地 == 异地（FFI 单测，逐位）；
`n`/空指针校验（`_checked`：`NegativeLength` / `NullPointer`，`n = 0` 合法）；精度（f64 参考）。

**3）不在契约内**：输入含 `NaN`（进 `NaN` 出）；`±inf`。`+inf` 恰好给出正确的极限
（`den = 1` ⇒ `y = +inf`），`−inf` 给出 `−inf`（**不是**数学极限 0）——因为 `exp` 的上界夹取
使分母停在有限值。列出该项是为了消除调用方的猜测空间：**不要把 `±inf` 传入激活算子**。

**4）实测精度**（与 f64 参考比，扫描 `[−150, 150]`，步长 5e-4）：

| 算子 | 最差绝对误差 | `\|x\| ≤ 20` 绝对 | 相对误差（`\|y\| > 1`，正侧） |
|---|---|---|---|
| `silu` | 7.6e-6 | 1.9e-6 | 1.7e-7 |
| `gelu_quick` | 7.6e-6 | 1.2e-6 | 1.6e-7 |

绝对误差那一列**主要是 f32 输入量化**（`x` 只有 24 位有效位：`|silu'(x)·Δx| ≤ |x|·6e-8`，
`|x| = 20` 时即已达 1.2e-6），不是算子本身的近似误差；正侧的相对误差被 `e^{−x}` 压掉，
故在 `|y| > 1` 的区间内优于 §2.6 的 `exp` 包络。

**5）实测带宽**：**4.55–9.20 GB/s**（单线程），朴素 Rust 循环（**同一个公式**）的 4.1–7.9×。
与 §2.7 的 rms_norm 同为 2 条流，吞吐却只有其一半到三分之一 ⇒ **这些算子是算力受限，
不是带宽受限**（同样的访存口径下 rms_norm 17.99 GB/s vs `silu` 7.13 GB/s，n ≈ 1M、
工作集 8 MB 仍在 L3 内）。瓶颈是 `exp`：每 8 个元素约 19 条向量指令外加**一次向量除法**，
而 rms_norm 每 8 个元素只需 2 条 `fma`。若要进一步提高吞吐，需改用更短的 `exp` 多项式
（以精度换速度）或重排多项式以缩短依赖链；该工作不在本节讨论范围内。
完整表见 `docs/dev.md` §7.4。

> 标量参照列存在一处易错点：`gelu_erf` 初版的参照实现写成了 f64 的"真 erf"（Taylor +
> 连分式），得到 400–500× ——该比值**不具备可比性**（它衡量的是"本库未提供 f64 精度 erf"
> 这一差异）。改用同一个 A&S f32 式子的朴素循环后为 7–10×，与另外两行口径一致。

> `gelu_erf` 是同一族的第三个（§2.9），带宽与这两个同档。


### 2.9 NN 侧：`lasx_gelu_erf`（erf 形式的 GELU）

**1）语义。** `y = 0.5·x·(1 + erf(x/√2))`——PyTorch `gelu` 的默认定义（对照表见 §2.8）。

**2）`erf` 的近似（契约的核心）。** `erf` 没有硬件指令，用 **Abramowitz & Stegun 7.1.26**：

```text
p = 0.3275911;  t = 1/(1 + p·|z|)                        // z = |x|/√2
erf(|z|) ≈ 1 − ((((a5 t + a4)t + a3)t + a2)t + a1)·t·exp(−z²)     // 全程 FMA
a = (0.254829592, −0.284496736, 1.421413741, −1.453152027, 1.061405429)
```

- 该式子**本身**的最大绝对误差是 A&S 给出的界 **1.5e-7**；用 glibc 的 `erf` 复测，
  折算到 gelu 输出上是 **2.1e-7**（出现在 `|x| ≈ 3.08`）；
- 因此该算子**并非"真 erf"**。若需与 ggml 对齐，使用 §2.8 的 `gelu_quick`；若需"真 erf"，
  则需自行实现（本库不承诺与 PyTorch 逐位相等）；
- `|x|` 用一条 `and 0x7FFFFFFF`（`arch::lasx::abs_f32x8`；本机 stdarch **没有** `xvfabs_s`，
  按位 `and`/`xor` 实际存在——早期版本写成"没有"是检索范围有误（只查了 `generated.rs`，
  未覆盖宏生成层），见 `docs/dev.md` §20.6 的更正）；
- **不带 `copysign` 的写法**：`y = 0.5·x + 0.5·|x|·erf(|x|/√2)`，与
  `0.5x(1 + sign(x)·erf(…))` 数学等价，但省掉"按符号选 lane"；
- `exp` 复用 §2.6 那一串（`ops::nn_math`）。

**3）由谁守**：逐位一致（`src/ops/gelu_erf.rs` 的标量模拟单测，`n` 1..=17 + 64/65/127/129；
**模拟实现必须显式写 `mul_add`**——初版遗漏该项，`x = −3` 处立即暴露 1 ulp 分歧）；
数值（与**独立实现的**高精度 f64 `erf`（Taylor + 连分式）对照，而非用同一个式子自比）；
边界（`x ≤ −5.6` ⇒ 精确 0、`x ≥ 6` ⇒ 位精确的恒等 `y = x`、正侧单调）；原地 == 异地（FFI 单测）。

**4）实测精度**（与高精度 f64 参考比）：

| `\|x\| ≤` | 1 | 4 | 16 | 40 |
|---|---|---|---|---|
| 最差绝对误差 | <6e-7 | <8e-7 | <1e-6 | <6e-6 |

误差由两部分构成：A&S 的 ~2.1e-7 + f32 舍入；`|x|` 较大时由**输入量化**主导（同 §2.8）。

**5）两个实测到的非理想行为（明确记录）**：

- **负侧尾部不单调**：真值从 `0⁻` 一侧趋近 0，而 `erf(|z|)` 在 `|x| ≈ 5.6` 处会**舍入到
  恰好 1.0** ⇒ `1 − 1 = 0` 给出**精确 0**；再向右（`x = −5.5`）又给出 ~`−1.6e-7` 的小负数。
  绝对幅度 ≤ A&S 的近似界，但"单调"这一直觉在 f32 下**不成立**，因此测试仅对 `x ≥ 0`
  断言单调。
- **`x = −inf` ⇒ NaN**（`0.5·inf + 0.5·(−inf)` 是未定式）。PyTorch 的 `gelu(−inf)` 同样是 NaN
  （同一个代数式），因此这不是本库特有的缺陷；契约只覆盖**有限输入**。`x = +inf` ⇒ `+inf` ✓。


### 2.10 NN 侧：`lasx_rope`（旋转位置编码）

**1）语义。** `rows × cols` 行主序，每行**前 `n_dims` 列**参与旋转，`[n_dims, cols)` 原样复制
（partial rope）。每一对 `(x0, x1)`：

```text
y0 = fma(x0, c, −(x1·s))        // 两次舍入：x1·s 一次，fma 一次
y1 = fma(x1, c,  (x0·s))
```

**2）两种配对 mode**（`mode` 的整数值是 C ABI 的一部分）：

| mode | 值 | `i ∈ [0, n_dims/2)` 配 | 对照实现 |
|---|---|---|---|
| `NeoX` | `0` | `(i, i + n_dims/2)` | HF `rotate_half` / Llama / Qwen |
| `GptJ` | `1` | `(2i, 2i+1)` | ggml `ROPE_TYPE_NORMAL` / 原始 GPT-J |

**3）`cos`/`sin` 表是输入**（`rows × (n_dims/2)`、行主序，第 `r` 行第 `i` 列 = `θ_i = positions[r]·freq_base^(−2i/n_dims)`）。
这样分层有两条理由：

- **角度不依赖数据**：一次前向里每层每位置算一次即足够（表是 `x` 的一半大小），将其放入
  "每个元素都算一遍"的内核，等于把一次性成本乘以 head 数——与 §2.6 的 `exp`（自变量依赖
  数据、必须逐元素计算）**情形相反**。llama.cpp 采用同样的划分（rope cache + 旋转内核）；
- **位精确承诺因此落在旋转上**：`sin`/`cos` 来自平台 libm，跨 libc 不保证逐位一致，本库
  **不承诺**表的位型，只承诺"**给定同一张表**，向量路径与标量路径逐位相同"。若需与其他
  实现严格对齐，应自行造表，仅依赖 [`lasx_rope`] 的旋转契约。
- 表**与 mode 无关**：同一张表两种配对都能用（`api::rope_tables` 不接 mode 参数）。

**4）`NeoX` 走 LASX、`GptJ` 走标量**（该结论由本机 ISA 实测得出）：

- `NeoX` 的两半各自连续 ⇒ 8 lanes = 8 个 `i`，2 load + 2 store，满 SIMD；
- `GptJ` 需要将 16 个连续元素拆成偶/奇两半。本机 `xvpickev.w`/`xvpickod.w` 是**128 位 lane
  内**操作且（实测排布）`pickev(a,b) = [b0,b2,a0,a2 | b4,b6,a4,a6]`——要恢复自然序，必须对
  `cos`/`sin` 施加同一个重排 π（跨 lane），单条 lane 内 shuffle 无法完成。因此 `GptJ`
  使用标量旋转（llama.cpp 的 rope 两种 mode 均为标量，该选择并不落后）。

**5）由谁守**：逐位一致（`ops::rope` 单测：两 mode × `n_dims/2` 1..=17 边界 + 多种形状、
`cos=1,sin=0` ⇒ **逐位恒等**、原地 == 异地在 FFI 层测）；数值（f64 参考的绝对误差 <1e-6、
旋转保长度相对误差 <1e-6）；形状（`api`：`n_dims` 偶数且 `≤ cols`、表长 = `rows×n_dims/2`；
`_checked`：同样规则 → `BadShape`/`NullPointer`）。
**只靠约定的一条**：C 侧调 `lasx_rope` 时表与 `x` 的长度自洽——原始符号零校验，
违反是**读写越界**（错结果甚至 UB），与 `docs/dev.md` §19.7 第 2 条同类。

**6）实测带宽**（`cargo run -p lasx_bench --release -- rope`，单线程；口径**3 条流**：
读 `x` + 写 `out` + 读表一半）：

| 形状 | `NeoX`（LASX） | `GptJ`（标量） | `NeoX`/标量 |
|---|---|---|---|
| 1×128 | 33 ns / 38.8 GB/s | 110 ns / 11.6 GB/s | 2.15× |
| 32×128 | 569 ns / 72.0 GB/s | 3.2 µs / 12.9 GB/s | 3.21× |
| 512×128 | 20.1 µs / 32.7 GB/s | 60.7 µs / 10.8 GB/s | 2.60× |
| 4096×128 | 157 µs / 33.4 GB/s | 485 µs / 10.8 GB/s | 2.65× |
| 8×4096（只转前 128 列） | 10.4 µs / 31.5 GB/s | 11.0 µs / 29.7 GB/s | 1.04× |

两条读数值得注意：**(a)** `GptJ` 比 `NeoX` 慢约 3×，这是"相邻配对的 SIMD 未实现"的代价
（若实现，潜在收益即该 3×）；**(b)** 部分旋转（`n_dims ≪ cols`）时两者都退化为 memcpy
主导（1.04×）——此时瓶颈不是旋转而是复制，与 softmax 的"遍数决定一切"属同一类结论。


### 2.11 NN 侧：`lasx_dot_f16` / `lasx_gemv_f16`（f16 权重）

**1）语义。** f16 权重以 **u16 位型**给出（本库零依赖，不引入 half crate）：

```text
dot_f16(a: &[u16], b: &[f32]) -> f32           // Σ f16_to_f32(a[i]) · b[i]
gemv_f16(a: &[u16], x: &[f32], m, k) -> [f32]  // y[r] = dot_f16(a[r*k..(r+1)*k], x)
```

**2）f16 → f32 是精确的**：f16 的每个值（含次正规、`±inf`）都能被 f32 精确表示，该步骤
**不引入误差**——误差仅来自"权重本身以 f16 存储"这一事实（相对 ~2^-11 ≈ 4.9e-4）。
原语 `xvfcvtl.s.h`/`xvfcvth.s.h` 是**128 位 lane 内**操作（各转换每个 lane 的低/高 4 个 f16），
因此一次 256 位载入还需再配两条 `xvpermi_q` 才能拼回自然序（详见 `arch::lasx` 的文档）。

**3）累加次序（契约本身）**：**4 条独立累加链**，每 32 个元素一轮（不足 32 走 16 元素轮、
再不足走标量尾；尾部**延续同一 lane** `j % 8`）：

```text
c0 = fma(cvt(a[i..i+8]),   b[i..i+8],   c0)     // 元素 i+k → lane k
c1 = fma(cvt(a[i+8..i+16]), b[i+8..i+16], c1)
c2 / c3 同前，偏移 16..32
最后：t = (c0 + c1) + (c2 + c3)（lane 内），再按固定次序两两归约 8 个 lane
```

**为什么采用 4 条链而非 1 条**（该选择由**实测**确定；初版为 1 条链）：
1 条链时每 16 个元素只有两条**互相依赖**的 FMA，将 ~4 周期的 FMA 延迟整段暴露——
实测 `1×2048` 为 565 ns；改为 4 条链后同一形状 **190 ns（3.0×）**，`dot n=4096` 从 1.2 µs
降至 359 ns。**"简单"的写法在此慢 3 倍，只有测量才能发现。**

**4）由谁守**：**转换的精确性可穷举验证**——f16 只有 65536 个位型，单测**遍历全部位型**，
将硬件 `xvfcvtl/fcvth` 与手写 `f16_to_f32` 逐位比较（`NaN` 的 payload 由硬件决定，只断言
`is_nan()`）；累加（向量 == 标量模拟，`n` 覆盖 0/1/7/8/15/16/17/31/32/33/63/64/1000）；
数值（与 f64 参考相对误差 <1e-6）；`gemv` 逐行 == `dot_f16`（逐位）；形状（`api`/`_checked`；
`k = 0 ⇒ y = 0`，空向量的点积是 0）。
**只靠约定的一条**：C 侧调用裸符号时 `a`/`x`/`y` 的长度自洽——违反是**读写越界**。

**5）实测**：`gemv` **3.75–48.2 GB/s**、`dot` **41.8–68.5 GB/s**（单线程；口径 = 权重字节
`m×k×2` + 输入输出）。与 §12.2 的参照比：llama.cpp 的 f16 GEMV 实测 **6.8 GB/s**，本实现
同量级形状（`1024×2048`）**10.5 GB/s ≈ 1.5×**；L1 驻留的小形状 48 GB/s 说明瓶颈不在算力。
**33.5 MB 那一行只有 3.75 GB/s 的原因已查清**（控制变量探针，见 `docs/dev.md` §20.7）：
**不是带宽**（纯 f16 流可达 15.8–16.4 GB/s），**也不是"`x` 被挤出 L1"**（多行微内核实测
无收益），而是 `f16→f32` 转换（+2.2 ms）与 `permi_q` 重排（+1.6 ms）占该 7.8 ms 的一半——
两者的数量由契约决定（`cvt` 每 8 元素一条是 ISA 上限；`permi_q` 是为了"元素下标 `j` 落在
lane `j%8`"的自然序，去掉即破坏位精确）。因此这是**结构代价，而非缺陷**；若要进一步提速，
必须修改本节的契约。完整表见 `docs/dev.md` §7.4。

### 2.12 N3 侧：int8 对称量化的生产端（`lasx_amax` / `lasx_absmax_rows` / `lasx_quantize_i8_*` / `lasx_dequantize_i8*`）

**1）语义。** 对称 int8，`qmax = 127`（`q ∈ [−127, 127]`，**不使用 −128**）：

```text
amax      = max_i |x_i|                        // 逐张量（lasx_amax）或逐行（lasx_absmax_rows）
scale     = amax / 127                         // amax == 0 ⇒ scale = 0
q_i       = round_ties_even(x_i · (1/scale))   // 取整是 ties-to-even
x̃_i       = (q_i as f32) · scale               // 反量化：一次乘法、一次舍入
```

**2）三条必须照抄的细节**（否则参考实现与内核不逐位一致）：

| # | 约定 | 为什么 |
|---|---|---|
| 1 | **用倒数乘**，不写 `x / scale` | LASX **没有向量除法指令**；与 `softmax_rows` 的 `e·(1/Σ)` 是同一条约定 |
| 2 | 取整是 **ties-to-even**（向量 `xvftintrne.w.s` / 标量 `round_ties_even()`） | 已实测两者一致：`0.5→0、1.5→2、2.5→2、−0.5→0、−1.5→−2、−2.5→−2` |
| 3 | **不做夹取**（`|q| ≤ 127` 由 scale 定义保证） | `scale = amax/127` 且 `|x| ≤ amax` ⇒ `|x·(1/scale)| ≤ 127`；省夹取既少 2 条向量指令，也让含 NaN 的输入在向量/标量两条路径上给出同一结果 |

**3）边界与退化。** `amax == 0`（全零或**全 NaN**）⇒ `scale = 0` 且 `q` 全 0（走显式分支，
不出现 `0 × inf = NaN`）。空输入 ⇒ `amax = 0`。`NaN` **被忽略**（归约与 `f32::max` 同语义），
`±inf` 不在契约内（`scale` 为 inf）。

**4）谁守什么。**

| 契约 | 由什么保证 |
|---|---|
| `|q| ≤ 127` | **scale 的定义**（`amax/127`）+ 取整；测试用"满量程数据 `max\|q\| ≥ 120`"与"边界样本打到 ±127"两条断言守 |
| 量化误差 ≤ `scale/2` | 契约本身（`round` 的定义）；测试对 999 个随机样本实测最坏值 |
| 反量化是 `scale` 的精确整数倍 | **一次乘法**；测试逐位比对 `(q as f32)·scale` |
| 长度一致 | Rust `api` 层：`Error::Shape`；**C 侧只靠约定**（违反是读写越界，与 §2.1 的例外清单同级） |
| 与 f64 参考的误差 | 本层是整数/量化运算，参考实现按上表照抄即逐位一致（不是"<1e-9"那种口径） |

**5）实测与实现形态。** `amax`/量化是"向量主体 + 标量尾"，两条路径同一串运算、逐位一致
（测试覆盖 `n = 0,1,2,3,7,8,9,15,16,17,31,33,64,127,128,1000` 与 NaN 混入）。
`dequantize` 目前是**逐元素标量**（i8→f32 的加宽链出来是 even/odd 交织 lane，写回连续下标
还需一次重排，未做向量化，也不声称它慢——没有实测就不写结论）。

### 2.13 N3 侧：`lasx_gemv_i8`（int8 权重 × int8 激活）

**1）语义。** `W` 是 `m × k` 行主序 int8 权重（**每输出通道一个 scale**），`x` 是 `k` 长 int8
激活（**每 token 一个 scale**，调用方在量化时得到）：

```text
y[o] = (Σ_i W[o,i]·x[i]) · (scale_w[o] · scale_x)
```

**2）数值契约。**

| 项 | 约定 |
|---|---|
| 累加 | **整数精确**：i16 乘积 → i32 累加 → 每 1024 字节落 i64（与 `lasx_dot_i8` **同一内核**，复用而非重写） |
| 出口 | **一次乘法**：先把两个 scale 乘成一个（一次舍入）再乘 `acc as f32`。写成 `((acc as f32)·sw)·sx` 是两次舍入，不保证逐位一致（有断言锁住） |
| 回绕 | `k ≤ 133 000` 时 `127²·k < 2³¹` 不回绕；更大 `k` 按 `dot_i8` 的 mod 2³² 语义（与 `lasx_dot_i8` 一致） |
| 偏置/残差 | **不在本内核内**：调用方在 f32 域自己加（残差连接、LayerNorm 都是 f32） |

**3）为什么复用 `dot_i8`。** GEMV 的每一行就是一次点积，而 `dot_i8` 已有"i16 乘积 → i32 累加、
每 1024 字节落盘 i64、行尾标量尾"的成熟实现与测试。一致性因此是**复用**来的，不是两处各写一遍
碰巧相同。

**4）谁守什么。** 累加精确性由整数运算保证（无舍入）；出口次序由契约 + 单测（"两次舍入会不同"
的样本）守；长度一致性由 Rust `api` 层守，C 侧只靠约定（越界同 §2.12）。

### 2.14 N3 侧：`lasx_matmul_i8`（批量 int8 矩阵乘）

**1）语义。** `X` 是 `m × k` 行主序 int8 激活（每 **token** 一个 scale，`scale_x` 长 `m`），
`W` 是 `n × k` 行主序 int8 权重（每 **输出通道** 一个 scale，`scale_w` 长 `n`）：

```text
Y[t,o] = (Σ_i X[t,i]·W[o,i]) · (scale_w[o] · scale_x[t])       // 即 Y = X·Wᵀ
```

**2）数值契约。** 每个输出元素是**一次 `dot_i8`**（与 §2.13 同一个整数内核 ⇒ 整数精确、
与顺序无关、`k > 133 000` 时按同一 mod 2³² 语义）；出口与 `gemv_i8` **完全同口径**
（两个 scale 先乘成一个再乘 `acc as f32`）。**`m = 1` 时与 `gemv_i8` 逐位一致**（单测守着）。

**3）实测结论（重要，与直觉相反）**：**批量没有收益，而且比 f32 GEMM 慢**——
`docs/dev.md` §21.8：对"逐 token 跑 `gemv_i8`"是 **1.00×**（9.03 vs 9.06 ms
@192×768×768），对 f32 `lasx_matmul`(n=192) 是 **0.4×（慢 2.5–2.7×）**。原因：本函数的复用
发生在 L2 而不是寄存器（每个输出元素整条读一遍 W 行），而 L2 带宽也只有 ~12 GB/s；
f32 一条 `xvfmadd.s` 是 16 FLOP/指令，本 ISA 的 int8 链只有 ~4–5 MAC/指令（没有融合 int8 点积）。

> **该用哪个**：**权重装不进缓存、必须从 DRAM 流**（单 token，或更大的模型）⇒ int8
> （每权重 1 B vs 4 B，省 4× 流量）；**批量且权重驻 L2/L3**（算力受限）⇒ **f32 `lasx_matmul`**。
> 对 `docs/dev.md` §21.2 那个消费者（192 token、每层权重 2.36 MB、L2 = 3 MB），**f32 更快**。
> 判据与配套结论（含批量形状的多线程派活建议）见 **§5.15**。

**4）谁守什么。** 与 §2.13 相同（整数内核 + 出口次序 + `api` 层长度校验；C 侧只靠约定）。

### 2.15 N3 侧：int8 路径的**端到端口径**（误差上界，不是位精确）

**为什么单列一节**：§2.12–§2.14 给的是**算子级**标准（同一算子的向量/标量路径**逐位一致**、
整数累加精确、量化误差 ≤ `scale/2`）。跨算子叠加之后不可能再谈位精确——量化本身有损——
所以端到端只能用**误差上界**。此前这条口径没写下来（`docs/platform.md` §5.1 记为待决策），
本节把它定死。

**口径（可复核，无歧义）**：

| 项 | 定义 |
|---|---|
| 被测管线 | **per-token 激活量化 → per-channel 权重量化 → `matmul_i8`**（出口已带两个 scale 的积，所以输出就是反量化值） |
| 参考量 | **f64 累加**的 `Σ_i X[t,i]·W[o,i]`（既不是 f32 路径也不是 int8 路径） |
| 误差 | **∞-范数相对量**：`max|Y_q − Y| / max|Y|`（不是逐元素相对误差——后者在 `Y≈0` 处会爆） |
| 上界 | **≤ 1e-2** |
| 对照 | 同口径的 **f32 路径 ≤ 1e-5**（实测 ~1.4e-6），用来证明"量化误差主导"而不是"两边都烂" |

**实测依据**（`api::tests::test_int8_pipeline_relative_error_is_bounded`，3 形状 × 3 种子）：

| 形状 | int8 ∞-相对 | f32 ∞-相对 |
|---|---|---|
| 192×768×768 | 5.216e-3 / 5.809e-3 / 5.256e-3 | ~1.0e-6 – 1.4e-6 |
| 192×768×3072 | 5.387e-3 / 5.608e-3 / 5.024e-3 | ~1.0e-6 – 1.2e-6 |
| 32×1024×1024 | 5.349e-3 / 6.047e-3 / 5.709e-3 | ~1.1e-6 – 1.4e-6 |

**最坏 6.047e-3**，界取 1e-2（1.65× 余量）。**读数在形状间稳定**（k 从 768 到 1024、
n 从 768 到 3072 都在 5–6e-3）——所以这条界对形状不敏感，可以直接当验收标准。

**适用范围（诚实边界）**：

- 这是**单层线性**的口径。**整模型级已实测（合成 12 层栈，2026-10-03）**：
  `api::tests::test_int8_model_level_error_over_twelve_layers`——每层 =
  `matmul_i8`（per-token + per-channel）→ f32 RMSNorm → `matmul_i8` → **残差相加**，共 12 层，
  参考量是**同结构、f64 精确算术、权重不量化**：

  | 栈 | ∞-相对误差（对 f64 参考） |
  |---|---|
  | **int8（12 层）** | **7.250e-4** |
  | 纯 f32 对照（12 层，不量化） | 1.725e-7 |

  **这条实测推翻了本节先前写的 `√L` 外推**（"12 层 ≈ 2.1e-2"）：实测比它**小 29 倍**，
  甚至比单层线性口径（6.05e-3）还小 8 倍。机制是**残差撑大输出量级（∞-范数的分母）、
  每层归一化重新标定激活**，而量化误差**不乘性累积**，故"误差随深度开方增长"在高斯随机
  权重上不成立。**测试守界 5e-3**（实测的 ~7 倍余量）。
- **该结论的边界**：上面是**合成栈**（随机权重、无注意力、无激活函数），**不是真模型**——
  它只给"深度累积量级"的证据，**不能当成具体模型的验收值**；∞-范数相对量还对输出量级
  敏感（残差越大分母越大），跨架构比较这个数**没有意义**；
- 归一化/激活/softmax 走 f32（§5.15 的组合路径），它们的误差不在本表内；
- 本节的界**由测试守**：改动量化或 `matmul_i8` 的出口口径让误差越过界，CI 会红（单层 1e-2、
  合成 12 层 5e-3 两条）。

### 2.16 NN 侧：`layer_norm`（行内 LayerNorm）

**存在理由**：外部契约要求（ONNX 图上 `LayerNormalization` 17 处，`docs/platform.md` §4）。
与 §2.7 的 `rms_norm` 是**同一套行归约骨架**，差别只有两处：**减均值**与**偏置 `b`**。

**1）语义。** `x`/`out` 是 `rows × cols` 行主序；`w`/`b` 是**按列**的权重与偏置：

```text
s    = Σ_j x[j]                    // 第 1 遍：8 lane 累加、固定次序两两归约、列尾补 0.0（无贡献）
mean = s / cols                    // 标量域，一次除法
ss   = Σ_j (x[j] − mean)²          // 第 2 遍（**补偏差 0.0**，不是补 x=0——那会凭空加上 mean²）
ms   = ss / cols                   // 标量域，一次除法
inv  = 1 / sqrt(ms + eps)          // 标量域，一次 sqrt + 一次除法
out  = (x[j] − mean) · inv · w[j] + b[j]   // 次序是契约：先减、再乘 inv、再乘 w、最后加 b
```

**2）三条必须照抄的细节**：

| # | 约定 | 为什么 |
|---|---|---|
| 1 | 方差用**两遍法**（先 mean，再 `Σ(x−mean)²`） | `E[x²] − mean²` 在 `mean ≫ std` 时灾难性相消 |
| 2 | `w`/`b` 传空 ⇒ **无权重等价乘 `1.0`**（精确）、**无偏置必须跳过加法** | 加 `0.0` 会把 `−0.0` 翻成 `+0.0`，破坏逐位一致（有测试守） |
| 3 | 行级除法/开方**只在标量域做一次**再 splat | 与标量模拟天然同式；顺带少指令（同 §2.7） |

**3）位精确。** 向量路径与标量模拟（同一个 8-lane 累加 + 固定次序归约 + 标量域行级数学）
**逐位一致**：`ops::layer_norm` 的测试覆盖 8 种形状（含列尾 1/7/17/33/129 与 `1×1`）×
4 种 `w`/`b` 有无组合；另有"平移不变性（LayerNorm 有、RMSNorm 没有）"与
"空偏置保 `−0.0`"两条行为断言。

**4）谁守什么。** 位精确由 `ops` 层测试守；长度/`eps` 由 `api` 层（`layer_norm_into`）
与 `ffi` 的 `_checked` 守；**"减均值不能退化成 RMSNorm"** 由本节那条平移不变性测试守
（这类"实现退化成近邻算子"的错误，位精确测试抓不到——它只比两条路径，不比语义）。


### 2.17 NN 侧：`gather_rows` / `gather_rows_i8`（按 id 取行，嵌入表）

**存在理由**：外部契约（ONNX 图上 `Gather` 34 处：词/段/位置嵌入），以及
`docs/platform.md` §4.2 对"怎么取"的决策。

**1）语义。** `ids` 是行号序列，`row_len` 是每行的元素数：

```text
out[i * row_len + j] = table[ids[i] * row_len + j]                              // gather_rows
out[i * row_len + j] = table[ids[i] * row_len + j] as f32 * scales[ids[i]]      // gather_rows_i8
```

**两条都没有累加** ⇒ 不存在结合次序问题：输出是 `table`/`ids`/`scales` 的确定逐元素函数
（int8 口径是"一次转换 + 一次乘法"）。所以它**不用 LASX 手写**——行拷贝（典型 384 个 f32 =
1.5 KB）用 `copy_from_slice`（即 `memcpy`，LLVM 会向量化）**更快也更容易验证**：
不需要列尾处理、不需要位精确论证。**这是有意的选择，不是漏写内核。**

**2）两条口径的分工**（**这条属于承诺边界，别混**）：`row_len` 与"表存 f32 还是 int8"由
**消费侧**决定；库侧只保证两种口径都正确。理由与尺寸账（fp32 表 384 MB vs int8 96 MB，
而每前向只取用 288 KB = 表的 0.08%）见 `docs/platform.md` §4.2。

**3）错误路径（都测了）**：

| 条件 | 错误 |
|---|---|
| `row_len == 0` | `NotPositive { what: "row_len" }` |
| `table.len()` 不是 `row_len` 的整数倍 | `Shape { what: "table" }`（`expected` = 向下取整到行） |
| 某个 `id >= 表行数` | `Shape { what: "ids" }`（`expected` = 表行数、`got` = 越界的那个 id） |
| `scales.len() != 表行数`（int8 口径） | `Shape { what: "scales" }` |
| `out.len() != ids.len() × row_len`（`_into`） | `Shape { what: "out" }` |

> **为什么 `id` 越界复用 `Shape` 而不新增错误变体**：`Error` 是 C ABI 的一部分
> （`_checked` 变体按变体映射错误码），**加一个变体就要动错误码表与所有绑定**；
> 而这里 `what = "ids"` + `expected`/`got` 已经把信息说全（消息里能看出是哪个 id 越界）。
> 这是**有意的取舍**，不是遗漏——真要新增变体时，属 `docs/dev.md` §19.7 的接口变更流程。

**4）谁守什么。** 边界（`row_len`/整除/id 范围/`scales` 长度/`out` 长度）由 `api` 与
`ffi` 的 `_checked` 守；**逐位一致**（`_into` vs 分配版、以及"与逐元素索引等价"）由
`ops::gather` 与 `api` 的测试守；"嵌入表存什么"由消费侧负责。



### 3.1 `SimdPath::detect()`

```rust
pub enum SimdPath { Lasx, Lsx }
impl SimdPath {
    pub fn detect() -> Self;   // hardware().lasx && !forced_lsx() ? Lasx : Lsx
    pub fn is_lasx(self) -> bool;
}
```

硬件能力用 `cpucfg` 读配置字 2：bit 7 = LASX、bit 6 = LSX。探测结果打包进**一个原子字节**
（写一次、之后只读），而非 `OnceLock`：快路径只剩一条 `Relaxed` 原子读，避免每次内核
调用 ~5 ns 的取值开销。**进程级只探测一次**，之后不可更新。支持降级的算子在入口
`match SimdPath::detect()` 分派到 `<name>_lasx` / `<name>_lsx`（少数内核的 `Lsx` 分支
退化为纯标量）。

**每调用开销**：`detect()` 的**边际**成本实测 **0.2–0.45 ns/次**（不是 1.5 ns——那个数里
绝大部分是测量循环自己的地板；口径修正与三次实现（含一次实测否决）见 `docs/dev.md` §7.14）。
它是"每次算子调用一次"的设计口径，**不要在循环里重复调用**；真要省，省的是调用侧那次
数据相关分支（`docs/dev.md` §7.13 的 `vec+detect` vs `vec`：每调用 2.8–4.2 ns），不是它内部的算术。

### 3.2 降级覆盖分类（46 个原始符号）

| 类别 | 个数 | 内核 | 无 LASX 时 |
|---|---|---|---|
| LASX + LSX 双路径 | 11 | `lasx_dot`、`lasx_norm3_batch`、`lasx_vec3_add_scaled_batch`、`lasx_j2_accel_batch`，以及全部 7 个姿态/几何算子 | LSX 128 位向量路径 |
| LASX + 标量降级 | 3 | `lasx_ballistic_step`、`lasx_batch_distance2d`、`lasx_rk4_j2_step_batch` | 全标量循环（无 LSX 向量路径） |
| LASX-only（无降级分支） | 25 | `lasx_matmul`、`lasx_matmul_scaled`（融合缩放，与前者同路径）、`lasx_matmul_f64`、`lasx_axpy`、`lasx_sum`、`lasx_dot_f64`、`lasx_dot_i8`、`lasx_dot_q4`、`lasx_layer_norm`（ONNX 批次），N1 全部 8 个（§2.6–§2.11），以及 **N3 全部 8 个**（§4.3） | 直接执行 LASX 指令 |
| 不涉及 SIMD（内存 + 运行时元信息） | 7 | `lasx_alloc`、`lasx_free`（内存工具）、`lasx_gather_rows`、`lasx_gather_rows_i8`（查表/行拷贝，`memcpy` 语义，见 §2.17）、`lasx_has_lasx`、`lasx_version`、`lasx_status_message` | 与路径无关（`lasx_has_lasx` 正是用来**问**这件事的） |

N1 批次**有意不写 LSX 降级**：目标平台是 3B6000（LA664，LASX 齐全），同核的 6000 机型
（3A6000 / 3C6000 / 3B6000）都支持 LASX；不为只有 LSX 的机型（如 2K3000 / 3B6000M，
LA364E 核）付这份复杂度（决策记录见 `docs/dev.md` §20.1）。

**N3 批次（int8 推理，8 个新符号，逐项见 §4.3 清单）同样 LASX-only**（`lasx_amax`、
`lasx_absmax_rows`、`lasx_quantize_i8_per_tensor`、`lasx_quantize_i8_per_row`、
`lasx_dequantize_i8`、`lasx_dequantize_i8_rows`、`lasx_gemv_i8`、`lasx_matmul_i8`；
契约见 §2.12/§2.13）：沿用 N1 的取舍，
无 LASX 的 CPU 上会执行 LASX 指令（见 §15 Caveats 与 README 的同一条警告）。
新符号一律**追加**，原始 15 个（§4.1）的签名与语义不变。

`_checked` 变体与其原始符号走同一条路径、同一个算子实现，覆盖范围相同。

### 3.3 LASX-only 的硬性约束

有一批 LASX-only 内核**没有 `has_lasx()` 检查，也没有降级分支**（完整清单与分类见 §3.2 的分类表）：

- 在**只有 LSX（128 位）、没有 LASX 的机型**（如 2K3000 / 3B6000M，LA364E 核）上调用会执行
  LASX 指令，产生**非法指令（SIGILL）**，不会自动降级。**C 侧调度前请先调 `lasx_has_lasx()`
  （§4.8）**——它返回 0 就说明当前生效路径只剩 LSX，此时只能调 LASX+LSX 的符号；不必自己去读
  `cpucfg`。LA464（3A5000）/ LA664（3A6000 等）支持 128/256 位两档向量指令，不属于此列。
- 源码模块文档对每个 LASX-only 内核都显式标注了这一点。
- `SimdPath::detect()` 与 `lasx_force_lsx_thread` 的降级只对有降级分支的内核有效；
  对 LASX-only 内核，`lasx_force_lsx_thread(true)` 不改变行为。

### 3.4 `lasx_force_lsx_thread` 与能力查询

```rust
pub fn lasx_force_lsx_thread(force: bool);           // 线程级，默认 false，仅 rlib 可见
let caps = lasx_rs::arch::hardware();                 // HwCaps { lasx, lsx }
let path = lasx_rs::arch::SimdPath::detect();         // Lasx | Lsx
```

C 侧**不必**读 `cpucfg`：`int lasx_has_lasx(void)`（§4.8）返回的正是
`SimdPath::detect().is_lasx()`——**同一个源**，所以它尊重强制降级（被降级时返回 0），
调用方问到的就是"算子此刻会走哪条路"。它是"调 `[LASX-only]` 之前"的安全前提，
不是可选优化；代价一次约 0.2–0.6 ns（`docs/dev.md` §7.14 的实测口径），按批次问一次即可。

用于在含 LASX 的真机上真实执行 LSX 分支（验证数值与性能），并非"无 LASX 真机"。测试后必须
显式置回 `false`。池化路径的 worker 是复用线程，与调用线程的 `FORCE_LSX` 无关
（`parallel::rk4_j2_step_batch` 在每块开头显式置 `false`）。


## 4. 导出符号总表（C ABI 的权威说明见 `docs/c-abi.md`）

> **这一节的内容已搬到 [`docs/c-abi.md`](c-abi.md)**——那里是 C ABI 的唯一权威：完整符号清单
> 与逐符号原型（含参数次序与 `[LASX-only]` 标注）、`_checked` 与状态码、内存与安全模型、
> 能力探测、链路与 ABI 稳定性策略。本节只保留下面这些**指向**，让既有的 `§4.x` 引用仍能落到
> 正确的话题上；**Rust 侧内容**（`api`/`view`/`plan`/`shape`/`pool`）不在搬迁范围内。

- 归约/稠密/量化/物理/内存 16 个裸符号 → `docs/c-abi.md` 第 3.1 节
- 批量姿态与几何 7 个 → 第 3.2 节
- NN 侧 N1 批次 8 个 → 第 3.3 节
- int8 推理 N3 批次 8 个 → 第 3.4 节
- ONNX 契约补充 3 个 → 第 3.5 节
- 运行时元信息与内存归还 4 个 → 第 3.6 节
- `_checked` 变体 44 个 → 第 3.7 节；状态码 0..6 → 第 3.8 节
- 矩阵乘**计划句柄** 7 个（4 裸 + 3 `_checked`，多核的正确姿势）→ 第 3.9 节与第 11 节
- 符号数清点、`nm -D` 自检、逐符号链接校验（`include/lasx_rs.h` + `scripts/check_header.c`）
  → 第 10 节

### 4.1 原始 15 个（历史契约，签名与语义不变）

**已搬移**：完整原型、路径标注与长度契约见 [`docs/c-abi.md`](c-abi.md) 第 3.1 节。
这 15 个符号的签名与语义**永不改动**（`lasx_alloc` 在其中，它没有 `_checked` 变体）。

### 4.2 批量姿态/几何 7 个

**已搬移**：完整原型与公共约定（四元数标量在前、体→惯、允许输出与输入别名）见
[`docs/c-abi.md`](c-abi.md) 第 3.2 节；逐算子语义与数值约定见本文 §7 的表。

### 4.3 N3：int8 推理 8 个（`docs/dev.md` §21）

**已搬移**：完整原型与路径见 [`docs/c-abi.md`](c-abi.md) 第 3.4 节。
契约（§2.12 量化、§2.13 GEMV）、用法（§5.14）、降级取舍（§3.2）。

### 4.4 `_checked` 变体 44 个

**已搬移**：规则（原型完全相同，只在末尾追加 `int *status`）、44 个名字与失败语义见
[`docs/c-abi.md`](c-abi.md) 第 3.7 节与第 5 节。

### 4.5 状态码（`LasxStatus`）

**已搬移**：0..6 的码表、`lasx_status_message()` 的 C 侧用法见 [`docs/c-abi.md`](c-abi.md)
第 3.8 节。Rust 侧另有 `LasxStatus::message()`（中文原因）、`is_ok()`、`from_i32()` 供绑定层
还原（见 §8）。

### 4.6 符号数清点

**已搬移**：分组计数、`nm -D` 清点命令、以及"哪些符号没有 `_checked` 变体"见
[`docs/c-abi.md`](c-abi.md) 第 3.0/3.6/3.7/10 节。本文 §3.2 给出 46 个原始符号的降级覆盖分类。

### 4.9 计划句柄：Rust 侧怎么用（2026-10-15 追加）

C 侧原型、生命周期与线程语义见 [`docs/c-abi.md`](c-abi.md) 第 3.9 节与第 11 节；**Rust 侧同一套
语义**在这几个入口上（C 只是它的投影）：

```rust
use lasx_rs::plan::MatmulPlan;
use lasx_rs::parallel;
use lasx_rs::pool::WorkerPool;

// ① 打包一次（权重矩阵反复用时最划算；返回后 b 可以释放）
let plan = MatmulPlan::from_row_major(&b, k, n)?;

// ②-A 调用方驱动（推荐）：只算一个行块，线程由你开 —— 同一个 plan 可被多线程并发借用
plan.run_block_into(&a[r0 * k..(r0 + rows) * k], &mut c[r0 * n..(r0 + rows) * n])?;

// ②-B 便利层：库内派发（线程数可调），并回填执行自检统计量
let pool = WorkerPool::new(12);          // 0/负 语义在 Rust 侧用 `WorkerPool::auto()`
let stats = parallel::matmul_f32_plan_with_stats(&pool, &plan, m, &a, &mut c)?;
println!("chunks={} ms_max={:.3}", stats.chunks, stats.ns_max as f64 / 1e6);
```

- `MatmulPlan::run_block_into` 与 C 的 `lasx_matmul_run` 是同一件事；
  `parallel::matmul_f32_plan{,_with_stats}` 对应 C 的 `lasx_matmul_run_threaded{,_checked}`。
- 位精确同 C 侧：这些入口与 `api::matmul` **逐位一致**（`src/plan.rs` 有逐位测试）。
- 线程数语义：C 侧 `<=0` 自动、`1` 串行、`N` 夹到物理核；Rust 侧用
  `WorkerPool::new(n)` / `WorkerPool::auto()` 表达同一件事（**不读环境变量**）。
- **全局池与"冻结"语义（2026-10-15 写清）**：Rust 侧 `pool::global()`（`shape`/DSL 路径默认用它）
  是**进程级 `OnceLock`，建一次、永不销毁** ⇒ **线程数在首次使用时就冻住**。**谁触发冻结**：
  第一次"够大"的 `shape` 调用（小形状走单线程档、**根本不初始化池**——有契约测试守着）。
  **显式出口**：Rust 侧在首次使用前调 `pool::init_global(n)`（**首次成功、之后返回 `false`**，
  有测试）；**C 侧不需要它**——`prepare`/`run_threaded` 的 `threads` 每次调用**显式给**，
  便利层的池按该线程数缓存在 plan 句柄里、随 `plan_destroy` 析构。
  **切分策略按形状自动选**（判据 `src/pool/sched.rs::pick_rows`，阈值表见 `docs/dev.md` §7.10 ⑩，
  **网格补扫、聚焦复验与最终口径见 `docs/dev.md` 的 §7.10 ⑬**）：大形状动态认领、其余静态/多块；
  **唯一按实测收窄的一档是 `threads = 16`（两窗口 15/15 格全静态）⇒ 它即使"每线程行数够多"也走静态**；
  `threads = 20` 曾一度被排除，**复核后不复现、已回退**（照常走判据）；`threads ∈ {8,10,12,14} ×
  每线程 20–32 行的"边界区"判据未定、维持保守（阈值 32）。**DSL/`shape` 路径与 C 侧装置共用
  这同一份判据**。
  **未动 `docs/dev.md` §19.7 冻结接口**：冻结的是"池的行块闭包签名 + start/切片对应"、`shape::Prepared::apply*`
  与 `Policy::threads`；本次只改**内部选哪个 `Pick`/句柄内部字段**，签名与 `Pick` 枚举都没变
  （理由入档在 `docs/dev.md` §7.10 ⑫）。

### 4.7 融合缩放（1 个，2026-10-15 追加）

**已搬移**：`lasx_matmul_scaled` 的完整原型（CBLAS 次序 `m,k,n,alpha,a,b,beta,c`）见
[`docs/c-abi.md`](c-abi.md) 第 3.1 节；数值口径见本文 §2.5 第 3 点与 §5.5 末段，决策与实测记录见
`docs/dev.md` §19.15。

### 4.8 运行时元信息与内存归还（4 个，2026-10-15 追加）

**已搬移**：`lasx_has_lasx` / `lasx_version` / `lasx_status_message` / `lasx_free` 的原型与语义见
[`docs/c-abi.md`](c-abi.md) 第 3.6 节；"先问能力再调"的顺序见第 6 节；内存配对与
"`free()` 不是 ABI 契约"见第 4 节。Rust 侧的对应面是 §3.1/§3.4/§10。


## 5. 归约、稠密与量化算子

### 5.1 `lasx_dot` — f32 点积（LASX+LSX）

`float lasx_dot(const float *a, const float *b, int n)`

- 语义（硬契约）：`Σ_{i<n} a[i]·b[i]`。
- n < 24 走**纯标量 f32 累加**：极小规模下向量建立 + f64 落盘 + 最终归约的固定开销摊不薄
  （实测 n=8 时向量路径反而更慢）。故跨 n=24 边界不保证逐位连续。
- n ≥ 24 由 `SimdPath::detect()` 分派。LASX：每 256 元素一块，块内 **4 条独立 f32 累加链**
  （每条 8 次 `lasx_xvfmadd_s`），块末把 4 个 f32×8 逐分量升 f64 加入 f64 累加器；
  随后 8 元素尾、标量尾。LSX：同样 256 元素一块、4 条链（每条 16 次 `lsx_vfmadd_s`），
  f64 标量累加。
- 精度：f64 分块落盘抑制 f32 累加误差，最终一次 `acc_d as f32` 舍入；对独立 f64 参考
  相对误差 <1e-5，LASX 与强制 LSX 两路各自校验。

### 5.2 `lasx_sum` — f32 归约（LASX-only）

`float lasx_sum(const float *x, int n)`

- 语义（硬契约）：`Σ_{i<n} x[i]`。与 `lasx_dot` 同构，把乘加换成 `lasx_xvfadd_s`：每 256
  元素一块、4 条链，块末升 f64 累加，8 元素尾，标量尾逐元素升 f64 累加。
- LASX-only：**无降级分支**。精度：相对独立 f64 参考 <1e-5。

### 5.3 `lasx_axpy` — `y += alpha·x`（LASX-only）

`void lasx_axpy(float alpha, const float *x, float *y, int n)`

- 语义（硬契约）：`y[i] += alpha·x[i]`，**原地**写回 `y`，不修改 `x`。
- 向量路径：`splat_f32(alpha)`，每 8 元素 `lasx_xvfmadd_s(vx, va, vy)`（单条 FMA）；
  标量尾 `y[j] += alpha*x[j]`。
- 精度：向量单舍入 vs 标量两次舍入，至多差 1 ulp，**无逐位守护**。LASX-only。

### 5.4 `lasx_dot_f64` — f64 点积（LASX-only）

`double lasx_dot_f64(const double *a, const double *b, int n)`

- 语义（硬契约）：`Σ a[i]·b[i]`（f64）。每向量 4 个 f64，主循环按 4 元素步进
  `lasx_xvfmadd_d`；结束时把 4 通道累加器落盘逐分量相加，标量尾直接累加。
- 精度：全程 IEEE 双精度。**无专门数值守护测试**。LASX-only。

### 5.5 `lasx_matmul` — f32 矩阵乘（LASX-only）

`void lasx_matmul(int m, int k, int n, const float *a, const float *b, float *c)`

- 语义（硬契约）：`C[m×n] = A[m×k]·B[k×n]`；A、B、C 均**行主序**。自 v1 起不变。
- 实现：i-k-j 列方向微内核，一次算 4 行 × 32 列；**不做 B 转置、不做跨 lane 水平归约**。按形状
  在打包面板 / 列块 / 流式三条路径间自动选择，三条路径**逐位一致**（输出元素按 k 升序累加）。
  打包缓冲**线程本地**复用（按需增长、跨调用保留），不是每次调用临时分配；其余两条路径不分配。
- 精度：每个输出元素沿 k 的**单个 f32 累加器**，无 f64 分块落盘，误差大于点积；对 f64
  参考相对误差 <1e-4。高精度场景改用 `lasx_matmul_f64`。
- `m == 0 || n == 0` 直接返回；`k == 0` 时 C 不会被写（`parallel::matmul_f32` 会显式
  `c.fill(0.0)`）。LASX-only。

**`alpha`/`beta` 融合（`C = alpha·A·B + beta·C`）**

```c
// C = alpha·A·B + beta·C（原地；行主序）。beta == 0 时不读 c。
void lasx_matmul_scaled(int m, int k, int n, float alpha,
                        const float *a, const float *b, float beta, float *c);
// 带错误通道：末尾多一个 int *status（校验口径同 lasx_matmul_checked）
void lasx_matmul_scaled_checked(int m, int k, int n, float alpha, const float *a,
                                const float *b, float beta, float *c, int *status);
```

```rust
// Rust 侧同一个算子
lasx_rs::api::matmul_scaled_into(m, k, n, alpha, beta, &a, &b, &mut c)?;

// 公式 DSL 的标量前缀 = alpha 配上 beta = 0（输出被完全覆盖）
matmul!(y[M, N] = alpha * x[M, K] * w[K, N]);   // → x.mul_scaled_into(alpha, &w, &mut y)
matmul!(y[M, N] = alpha * x[M, K] * p);         // plan 形态：p.apply_scaled_into(alpha, &x, &mut y)
```

- 数值口径（硬契约）：`alpha = 1, beta = 0` **逐位等于** [`lasx_matmul`]（同一条内核，收尾
  不多一次乘法）；其余情况 `C = fma(beta, C_old, alpha·acc)`，与"先算 `A·B` 再整体乘/加"
  **逐位一致**。`beta == 0` 时**不读 `C`**，可以传内容任意的缓冲。C 入口与 `api` 入口逐位一致
  （`ffi::matmul` 的测试守着）。
- 何时更快（实测，进程级交替 5 轮）：`beta == 0` 且**宽薄**（小 `k`、大 `m×n`）时省掉一整遍
  对 `C` 的读写——256×64×4096 **−10.3%**、100×64×19147 **−23.7%**；`beta != 0` 且 `k ≤ 384`
  小胜（−6.5%）；方阵/大 `k` 形状 ≈0，但**不会更慢**（`beta != 0` 且 `k > 384` 退回与朴素
  二遍法同构的实现）。完整数字、边界与"为什么上次否掉这次翻案"见 `docs/dev.md` §19.15。
- **只对 `f32` 提供**：`f64` 写 DSL 标量前缀是编译错误（没有 scaled 内核），要缩放请先
  `mul` 再整体乘——这是刻意的边界，不做静默两遍法。

### 5.6 `lasx_matmul_f64` — f64 矩阵乘（LASX-only）

`void lasx_matmul_f64(int m, int k, int n, const double *a, const double *b, double *c)`

- 语义与布局：与 `lasx_matmul` 完全同构（行主序、不转置），每向量 4 个 f64，一次算
  4 行 × 16 列；同样有打包 + k 分块与流式两条路径，小形状走流式，两路径逐位一致；
  线程本地打包缓冲同 5.5。
- 精度：全程 f64；对 f64 参考相对误差 <1e-12。LASX-only。

### 5.7 `lasx_dot_i8` — int8 量化点积（LASX-only）

`int lasx_dot_i8(const int8_t *a, const int8_t *b, int n)`

- 语义（硬契约）：`Σ a[i]·b[i]`，int8 输入、**i32 返回**。
- 整数精确：每向量 32 个 i8，带符号 8→16 位扩展乘得 16 个 i16 再拓宽为 i32 累加（上界
  128×127² ≈ 2.1 M，安全）；每 **1024 字节**落盘转 i64，最后 `as i32` 返回。
- 边界：超出 i32 范围时**回绕**（无饱和/报错），约 n > (2³¹−1)/127² ≈ 13.3 万（全 ±127 输入）
  时可能溢出。测试与 i64 精确参考**逐位相等**，覆盖块边界、i32 回绕、(−128)² 的 i16 边界；LASX-only。

### 5.8 `lasx_dot_q4` — Q4 量化点积（LASX-only）

`double lasx_dot_q4(const uint8_t *qa, const float *sa, const uint8_t *qb, const float *sb, int n_bytes)`

- 量化格式：每个 `u8` 含 **2 个 4-bit 无符号 nibble**；低 = `q & 0x0f`，高 = `(q>>4)&0x0f`。
- scale 布局：`sa`/`sb`（f32）每 **32 字节一组**一个值；组数 `n_groups = ceil(n_bytes/32)`。
  `sa`/`sb` 长度必须 ≥ `n_groups`；非 32 倍数时最后一组按 `j/32` 取 scale。
- 公式（源码注释）：`dot = Σ_g sa[g]·sb[g]·Σ_{组内 64 nibble}(qa·qb)`。组内整数点积精确，
  再乘该组 scale 积（f32 乘法）并升 f64 累加。
- 实现：每批 4 组（128 字节）独立求和；归约与 f64 累加**严格按组序**，与逐组循环逐位不变。
  nibble ∈ [0,15]：乘积 ≤225、组内每 lane ≤900、总和 ≤14400，整数部分无溢出。
- 精度：对精确参考相对误差 <1e-12；误差主要来自 f32 尺度表示（约 2⁻²⁴）。LASX-only。


### 5.9 `lasx_softmax_rows` — 行内 softmax（LASX-only）

| 层 | 签名 |
|---|---|
| C（裸） | `void lasx_softmax_rows(const float *x, const float *mask, float *out, int n_rows, int n_cols, float scale)` |
| C（`_checked`） | 同上 + 末尾 `int *status`；`mask` 允许 NULL，`x`/`out` 在 `n_rows×n_cols > 0` 时为空报 `NullPointer` |
| Rust | `api::softmax_rows(&[f32], Option<&[f32]>, rows, cols, scale) -> Result<AlignedVec<f32>>` |

- **语义**：`out = exp(scale·x + mask − 行内 max) · (1/Σ)`，`rows × cols` 行主序；
  `mask` 是可选**加性**偏置（attention 的 `+ mask`）；
- **数值契约**（`exp` 的 op 序列、下溢到精确 0、向量/标量逐位一致、`+inf` 不在契约内）
  见 §2.6——该算子的精度完全由那一节定义；
- **实测**：**5.22–7.00 GB/s**（单线程，2 条流口径），朴素 Rust 行循环的 4.5–5.6×，
  约为 llama.cpp `SOFT_MAX` 典型值（2.4–4 GB/s）的 1.3–2.9×。按真实访存（三遍：max /
  exp+sum / 归一）折算约 7.8–10.5 GB/s，已在单线程带宽上限附近。完整表见 `docs/dev.md` §7.4；
- **LASX-only**：无降级分支（新增符号一律 LASX-only，见 §3.2）。


### 5.10 `lasx_rms_norm` — 行内 RMSNorm（LASX-only）

| 层 | 签名 |
|---|---|
| C（裸） | `void lasx_rms_norm(const float *x, const float *w, float *out, int n_rows, int n_cols, float eps)` |
| C（`_checked`） | 同上 + 末尾 `int *status`；`w` 允许 NULL（无权重），`eps` 要求有限且 > 0 |
| Rust | `api::rms_norm(&[f32], Option<&[f32]>, rows, cols, eps) -> Result<AlignedVec<f32>>` |

- **语义**：`out = x / √(mean(x²)+eps) · w`，`rows × cols` 行主序，`w` 按列（长度 `cols`）；
- **数值契约**（`fma` 平方和、标量域行级数学、精确 sqrt、`eps > 0`）见 §2.7；
- **实测**：**12.01–18.38 GB/s**（单线程，2 条流口径），朴素 Rust 行循环的 2.6–3.5×；
  完整表见 `docs/dev.md` §7.4；
- 与 softmax 的对比见 §2.7 末尾（两遍 vs 三遍访存）。


### 5.11 `lasx_silu` / `lasx_gelu_quick` / `lasx_gelu_erf` — 逐元素激活（LASX-only）

> **Rust 侧有免分配版**：`api::silu_into` / `api::gelu_quick_into` / `api::gelu_erf_into`
> （写调用方缓冲，与分配版**逐位一致**）。**逐层激活每次都要调**，而实测分配占该调用的
> **10–15%**（192×3072、589 824 个元素；`docs/dev.md` §7.9 ③ 有 A/B 表）。
> C 侧本来就是调用方给 `out`（`lasx_silu` 等），没有这笔开销。

| 层 | 签名 |
|---|---|
| C（裸） | `void lasx_silu(const float *x, float *out, int n)`；`void lasx_gelu_quick(const float *x, float *out, int n)`；`void lasx_gelu_erf(const float *x, float *out, int n)` |
| C（`_checked`） | 同上 + 末尾 `int *status`；`n < 0` → `NegativeLength`，`n > 0` 时空指针 → `NullPointer` |
| Rust | `api::silu(&[f32]) -> AlignedVec<f32>`；`api::gelu_quick(&[f32])`；`api::gelu_erf(&[f32])` |

```rust
use lasx_rs::api;
let y = api::silu(&hidden);          // y[i] = x[i]/(1+exp(-x[i]))
let g = api::gelu_quick(&hidden);    // y[i] = x[i]/(1+exp(-1.702*x[i]))     —— ggml GELU_QUICK
let e = api::gelu_erf(&hidden);      // y[i] = 0.5*x[i]*(1+erf(x[i]/√2))    —— PyTorch gelu（近似的 erf）
```

```c
lasx_silu(x, out, n);                /* 允许 out == x（就地） */
lasx_gelu_quick_checked(x, out, n, &status);
lasx_gelu_erf_checked(x, out, n, &status);
```

- **形状只是 `n`**（逐元素，不需要 rows/cols），**允许就地**（`out == x`）：内核每轮先读完
  8 个再写回同样位置；
- **数值契约**：`silu`/`gelu_quick` 见 §2.8（直接除法 `x/den`、`exp` 用全库共用的那一串）；
  `gelu_erf` 见 §2.9（A&S 7.1.26 的 `erf` 近似，**不是真 erf**）；
- **实测**：**4.55–9.20 GB/s**（单线程，2 条流口径；`n` = 1024 / 64K / 1M / 8M），朴素 Rust
  循环（**同一个公式**）的 4.1–7.9×；`gelu_erf` 在小尺寸下比 `silu` 慢 ~1.7×，8M 时三者
  收敛（DRAM 受限）。完整表见 `docs/dev.md` §7.4；
- **算力受限，不是带宽受限**：同为 2 条流却只有 rms_norm 的一半到三分之一，瓶颈是 `exp`
  （每 8 元素 ~19 条向量指令 + 一次向量除法）——完整分析见 §2.8 第 5 条。


### 5.12 `lasx_rope` — 旋转位置编码（NeoX 走 LASX，GptJ 走标量）

> **Rust 侧有免分配版 `api::rope_into`**（与分配版逐位一致，两种 `RopeMode` 都测了）。
> **这是全库分配占比最高的算子：实测 39–41%**（192×768、n_dims = 128：83.9 → 50.6 µs，
> 内核只 ~51 µs 而输出 0.6 MiB 的分配约 33 µs），而 RoPE 在注意力里**每层都调**
> ⇒ 热路径上应该用 `_into`（A/B 表见 `docs/dev.md` §7.9 ③）。
>
> **更值钱的一条用法：`rope_tables` 必须 hoist 出层循环。** `api::rope_at` 每次调用都会
> **重算 cos/sin 表**（`rope_tables`：三角 + 两次逐元素拷贝 + 两次分配，再加输出分配），
> 实测 **515.7 / 541.0 / 526.8 µs vs 预建表 + `rope` 的 56.5 / 57.8 / 56.8 µs（差 89%）**
> ——**用现有 API 就能拿到这 9×，不需要任何新接口**：层循环外调一次
> `api::rope_tables(&positions, n_dims, freq_base)`，每层只调 `api::rope_into`。
> （口径：load 3.18–3.25 的安静窗口、3 轮；见 `docs/dev.md` §7.9 ⑤。）
>
> **`rope_tables` 本身也优化过（2026-10-04）**：它的 `powf` 原本在**内层循环**里（每个
> (位置, 维度) 对重算一次 ⇒ 192×64 = 12288 次），hoist 到循环外后只剩 64 次，
> 表重建 **~460 µs → ~232 µs（~2.0×）**，**输出逐位不变**（有测试 `test_rope_tables_bit_exact_vs_per_element_powf`
> 钉住；见 `docs/dev.md` §7.9 ⑥）。剩下的大头是 12288 次 f64 `sin_cos`。
>
> **这条不靠"并行建表"再压**：实测 12 个空线程的 `spawn+join`（启动下界）**271.4 / 273.3 /
> 265.1 µs 就已经超过串行建表的全部工作**（233.1 / 240.2 / 243.7 µs，192 位置 × n_dims=128）
> ⇒ 这个粒度撑不起"每调用起一组线程"。**正确的省法是把表 hoist 出层循环**（上一条，实测 9×）。
> 详细判据与边界见上文所引的该节 ⑥。

| 层 | 签名 |
|---|---|
| C（裸） | `void lasx_rope(const float *x, const float *cos, const float *sin, float *out, int n_rows, int n_cols, int n_dims, int mode)` |
| C（`_checked`） | 同上 + 末尾 `int *status`；形状/mode 非法 → `BadShape`，空指针 → `NullPointer` |
| Rust | `api::rope(&[f32], &[f32], &[f32], rows, cols, n_dims, RopeMode) -> Result<AlignedVec<f32>>`，另有 `api::rope_tables(positions, n_dims, freq_base)` 与便捷的 `api::rope_at(...)` |

```rust
use lasx_rs::api::{rope, rope_tables, RopeMode};
// 表建一次（每层/每个位置一组角度），旋转可以反复做
let (cos, sin) = rope_tables(&positions, 128, 10_000.0)?;
let y = rope(&q, &cos, &sin, seq_len, 128, 128, RopeMode::NeoX)?;
// 或者一步到位（每次都会重建表，别放进热循环）
let y = api::rope_at(&q, &positions, seq_len, 128, 128, 10_000.0, RopeMode::NeoX)?;
```

```c
lasx_rope(x, cos, sin, out, n_rows, n_cols, n_dims, /*mode=*/0);   /* 允许 out == x */
lasx_rope_checked(x, cos, sin, out, n_rows, n_cols, n_dims, 1, &status);
```

- `cos`/`sin` 各 `n_rows × (n_dims/2)`、行主序，**与 mode 无关**；
- `mode`：`0` = `NeoX`（`(i, i+d/2)`）、`1` = `GptJ`（`(2i, 2i+1)`）；
- **实测**：`NeoX` **28.6–71.9 GB/s**（1×128 时 37.7、32×128 时 71.9、4096×128 时 28.6），
  是朴素标量循环的 2.0–3.2×；`GptJ` **10.8–12.8 GB/s**（标量路径，见 §2.10 第 4 条）；
  部分旋转（`n_dims ≪ cols`，如 8×4096 只转 128 列）时两者都退化成 memcpy 主导（1.03×）。
  完整表见 `docs/dev.md` §7.4。


### 5.13 `lasx_dot_f16` / `lasx_gemv_f16` — f16 权重的点积与矩阵-向量

| 层 | 签名 |
|---|---|
| C（裸） | `float lasx_dot_f16(const uint16_t *a, const float *b, int n)`；`void lasx_gemv_f16(const uint16_t *a, const float *x, float *y, int m, int k)` |
| C（`_checked`） | 同上 + 末尾 `int *status`；负长度 → `NegativeLength`，空指针 → `NullPointer`，乘积溢出 → `SizeOverflow` |
| Rust | `api::dot_f16(&[u16], &[f32]) -> Result<f32>`；`api::gemv_f16(m, k, &[u16], &[f32]) -> Result<AlignedVec<f32>>` |

```rust
use lasx_rs::api;
let y = api::gemv_f16(4096, 2048, &w_f16, &x)?;   // w_f16 是 u16 位型（2 B/元素）
```

```c
float d = lasx_dot_f16(a_f16, b_f32, n);
lasx_gemv_f16_checked(a_f16, x, y, m, k, &status);
```

- **f16 以 `u16` 位型进出**（零依赖，不引入 half crate）；f16→f32 **精确**（单测穷举全部
  65536 个位型对硬件验证），所以误差只来自"权重本来是 f16"（相对 ~4.9e-4）；
- **累加次序**：4 条独立链、每 32 元素一轮、尾部延续 lane `j % 8`、固定次序两两归约
  （契约见 §2.11；改成 1 条链会慢 3×——实测）；
- `k = 0` ⇒ `y` 全 `0`（空向量的点积是 0）；`gemv` 每行与 `dot_f16` **逐位相同**；
- **实测**：`gemv` **3.75–48.2 GB/s**、`dot` **41.8–68.5 GB/s**（单线程；口径 = 权重字节
  + 输入输出）。与 §12.2 的参照（llama.cpp f16 GEMV **6.8 GB/s**）比，同量级形状
  （`1024×2048`）是 **10.5 GB/s ≈ 1.5×**；超出 L3 的 33.5 MB 形状只有 3.75 GB/s，瓶颈是
  `f16→f32` 转换与 `permi_q` 重排两项结构代价（机制见 §2.11 第 5 条）。完整表见 `docs/dev.md` §7.4。
- **并行**：`parallel::gemv_f16(&pool, m, k, &[u16], &[f32], &mut [f32])`（多核）；每行独立，
  按行块切分，与单线程**逐位一致**。实测（进程级交替 A/B）：`1024×2048` 单线程 14.0 GB/s →
  12 线程 **110 GB/s（7.9×）**、`4096×4096` 5.2 → 31 GB/s、`11008×4096` 4.3 → 22 GB/s；
  **线程数取物理核数**（本机 12），24 线程在中小形状反而更慢——依据见 §9 的"线程数怎么选"
  与 `docs/dev.md` §20.7。
- **声明式**：形状编译期已知时可直接写公式 `gemv!(w_f16[N, K] * x[K])` / `dot!(w_row[K] * x[K])`，
  与 `api::gemv_f16` / `api::dot_f16` 逐位一致（完整示例见 §8.3.1）。

### 5.14 N3：int8 推理（量化 → `gemv_i8`）

| 层 | 签名 |
|---|---|
| C | `float lasx_amax(const float *x, int n)`；`void lasx_absmax_rows(const float *x, int rows, int cols, float *out)`；`float lasx_quantize_i8_per_tensor(const float *x, int8_t *q, int n)`；`void lasx_quantize_i8_per_row(const float *x, int8_t *q, float *scales, int rows, int cols)`；`void lasx_dequantize_i8(const int8_t *q, float scale, float *out, int n)`；`void lasx_dequantize_i8_rows(const int8_t *q, const float *scales, float *out, int rows, int cols)`；`void lasx_gemv_i8(const int8_t *w, const float *scale_w, const int8_t *x, float scale_x, float *y, int m, int k)` |
| C（`_checked`） | 同上 + 末尾 `int *status`；负长度 → `NegativeLength`，空指针 → `NullPointer`，尺寸相乘溢出 → `SizeOverflow`（**i32 长度下 64 位不会触发**，只有 Rust `api` 层的 `usize` 参数可达） |
| Rust | `api::amax`、`api::absmax_rows`、`api::quantize_i8_per_tensor` → `(AlignedVec<i8>, f32)`、`api::quantize_i8_per_token` / `api::quantize_i8_per_channel` → `(AlignedVec<i8>, AlignedVec<f32>)`、`api::dequantize_i8` / `api::dequantize_i8_rows`、`api::gemv_i8` |

一个判别式模型（cross-encoder，无 KV cache、无自回归解码）的 int8 前向就是
"**激活按 token 量化 → 权重按输出通道量化 → GEMV → 残差/归一化留在 f32 域**"：

```rust
use lasx_rs::api;

// 1) 权重侧（一次性）：按输出通道量化，每行一个 scale
let (w_q, w_scale) = api::quantize_i8_per_channel(&w_f32, out_features, in_features)?;
// 2) 激活侧（每层每次）：按 token 量化——per-tensor 会让准确率掉 2.7 个点（§12.3 N3）
let (x_q, x_scale) = api::quantize_i8_per_token(&act, tokens, hidden)?;
// 3) 线性层：y[o] = (Σ w[o,i]·x[i]) · (w_scale[o]·x_scale[token])
let y = api::gemv_i8(
    w_q.as_slice(), w_scale.as_slice(), x_q.as_slice(), x_scale[0], out_features, hidden,
)?;
// 4) 出口反量化（把 int8 结果送回 f32 域做残差/归一化）
let y_f32 = api::dequantize_i8(&q_out, scale_out)?;
```

```c
float s = lasx_quantize_i8_per_tensor(x, q, n);          /* 返回 scale */
lasx_quantize_i8_per_row(x, q, scales, rows, cols);      /* per-token / per-channel */
lasx_gemv_i8(w, scale_w, x_q, scale_x, y, m, k);
```

- **契约**在 §2.12（量化/反量化）与 §2.13（GEMV）：用**倒数乘**、取整 **ties-to-even**、
  **不夹取**、出口**两个 scale 先乘成一个**；
- **精度**：量化误差 ≤ `scale/2`；`dequantize` 的结果是 `scale` 的精确整数倍；GEMV 的整数
  累加**精确**（不引入额外舍入），出口只多一次乘法；
- **未做**：整数/定点 softmax、int8 归一化——**已按实测否掉**（组合路径占该层 int8 GEMM
  的 ~2%，见 `docs/dev.md` §7.9 末表）：要保活就 `dequantize_i8_rows` → `api::rms_norm`/`
  softmax_rows` → `quantize_i8_per_token` 拼，别写定点版；
- **实测**（2026-10-02）：`gemv_i8` 在 118M 模型的真实层形状上是 **10.7–12.4 GB/s（权重字节）**
  ——贴着单流只读锚点 12.0 GB/s，**已是带宽饱和**；同形状 f32 `lasx_matmul`(n=1) 只有
  2.94–2.97 GB/s，比值 **14.6–16.8×**（其中 4× 来自 int8 的 1 B/权重，其余来自 f32 对照的
  GEMM 打包路径）。**per-token 量化不付性能代价**（per_row 与 per_tensor 在 192×768 上
  76.9 vs 79.5 µs、192×3072 上 271.8 vs 315.4 µs，逐行反而更快）。完整表见 `docs/dev.md` §7 的对应小节；
  **端到端 10–20 ms 的立项目标被实测推翻**（口径与账见 `docs/dev.md` §21.6）。

### 5.15 int8 还是 f32：一句话判据（给下游）

**先看权重装不装得进缓存，再选精度——不要按"int8 一定快"选。** 依据是四轮实测
（§2.12–§2.14 的契约、`docs/dev.md` §7.9 的横截面、§21.6–§21.8 的账）：

| 形态 | 选谁 | 为什么（实测） |
|---|---|---|
| **单 token / 权重 > 缓存**：必须从 DRAM 流权重 | **int8**（`gemv_i8`） | 每权重 1 B vs 4 B，两者都贴着 12 GB/s 锚点 ⇒ **省 4× 流量** |
| **批量（多 token 复用同一片权重）、权重驻 L2/L3** | **f32**（`lasx_matmul`；多核用 `parallel::matmul_f32`） | 算力受限：f32 一条 `xvfmadd.s` = **16 FLOP/指令**，LASX 的 int8 链只有 **~4–5 MAC/指令**（本 ISA **没有融合 int8 点积**）⇒ 实测 f32 快 **2.5×** |

两条配套结论：

- **归一化/softmax 有免分配版**：`api::rms_norm_into` / `api::softmax_rows_into` 把结果写进
  调用方缓冲，与分配版**逐位一致**（同一个内核，只少一次分配）。**这条不是"微优化"**：
  `192×768` 实测分配版 179.8–183.5 µs vs `_into` **75.2–75.7 µs（快 2.4×，分配占 58%）**，
  `192×3072` 是 703.8–717.0 vs 258.1–263.3 µs（占 63%）——逐层保活路径上一个前向省 ~11 ms
  （`docs/dev.md` §7.9 末表）。凡是"每层都要调一次"的算子，优先用 `_into`；
- **批量形状按 6–8 线程派活**：`192×768×3072` 这类形状 **8 线程最优（5.99×）**、12 线程 5.11×、
  24 线程 4.57×；立方形状（`512³`/`1024³`）在多核下仍是越大越好，`256³` 这类小 B 仍走回退路径。
  （该建议 2026-10-03 按 `need_shared_pack` 判据修正过一次；修史与对照测量见上文所引的同一张末表。）
- **单 token（`m = 1`）用 `parallel::gemv_i8_k`**：按 `k` 切块，12 线程实测 **5.4–6.9×**
  （`k ≥ 1 MiB` 才有收益；`k` 小到几 KiB 时并行是**负收益**，函数内已按"每块 ≥1024 且 ≥4 块"
  退化为串行）。注意它修的是**访存并行度**（单线程冷流只有 ~3 GB/s），不是"带宽不够"；
- **量化生产端本身不贵，且 per-token 免费**：per-row 与 per-tensor 同量级（逐行略快），
  所以"为了准确率选 per-token"（`docs/dev.md` §21.3）**不付性能代价**；
- **本页不含**注意力（QK^T/softmax）、归一化、embedding——那部分尚未测；端到端量级见
  `docs/dev.md` §21.6 的分档表。


### 5.16 `lasx_layer_norm` / `lasx_gather_rows` / `lasx_gather_rows_i8` — ONNX 契约补充（3 个）

来源：外部契约（ONNX 图上 `LayerNormalization` 17 处、`Gather` 34 处），`docs/platform.md` §4。

| 符号 | C 签名要点 | 语义 | 契约 |
|---|---|---|---|
| `lasx_layer_norm` | `void(const float* x, const float* w, const float* b, float* out, int rows, int cols, float eps)` | `out = (x − mean)/√(var+eps)·w + b`（`w`/`b` 可 NULL；方差**两遍法**） | §2.16 |
| `lasx_gather_rows` | `void(const float* table, const int* ids, float* out, int n_ids, int row_len, int n_table_rows)` | `out[i] = table[ids[i]]`（纯行拷贝，**不用 LASX**） | §2.17 |
| `lasx_gather_rows_i8` | `void(const int8_t* table, const float* scales, const int* ids, float* out, int n_ids, int row_len, int n_table_rows)` | 行拷贝 + 每行 scale 乘 | §2.17 |

**Rust 侧同时有免分配版**：`api::layer_norm_into`、`api::gather_rows_into`、
`api::gather_rows_i8_into`（以及对应的分配版）——**新算子一律给两个入口**，
理由见 `docs/dev.md` §7.9 的 `_into` 实测（分配占 `rms_norm` 调用的 58–63%）。

**"嵌入表存 f32 还是 int8"不是本库的决定**（`docs/platform.md` §4.2 的尺寸账）：库侧只保证
两种口径都正确、边界责任清楚（id 越界在 C 侧是 `BadShape`(3)，在 Rust 侧是
`Error::Shape { what: "ids" }`）。

---

## 6. 批量几何与物理算子

> **这一族的 `_into`（免分配）版本：决定"不做"**（2026-10-04）。两条理由：
> ① **已知消费者不再调用它们**——`loong-sci` 的 `engine.rs:96` 写着这一族的转发"**已经删掉了**"
> （2026-09），本库侧的消费侧审计也是 **0 调用点**；
> ② **签名重设成本高**：这族一次返回 **3–9 个缓冲**（`unitize3_batch` 3、`quat_to_dcm_batch` 9），
> 要做 `_into` 得先设计新签名（`&mut [&mut [T]; N]` 还是"平铺到一个大缓冲 + 偏移"）。
> 分配占比也不支持：`quat_to_dcm_batch` **对称 A/B 只有 3.4–3.8%**（n = 65536）、`norm3_batch`
> 17.3–17.7%（**旧设计、且两臂的对称复核被高负载窗口挡下**，见 `docs/dev.md` §7.9 ④）。
> **触发条件**：若将来出现"仿真循环里逐个调用这族"的消费者，先量它的占比、再按上面两条决定。
> 另外，这一族的调用方若改用"预分配 + FFI"（C ABI 本来就是调用方给缓冲），**根本没有这笔分配**。




### 6.1 `lasx_norm3_batch` — 批量 3 分量模长（LASX+LSX）

`void lasx_norm3_batch(const double *xs, const double *ys, const double *zs, double *out, int n)`

- 语义：`out[i] = √(xs[i]² + ys[i]² + zs[i]²)`，SOA 布局。
- 结合序固定 `x² + (y² + z²)`（右结合）；LASX 4 样本/向量、LSX 2 样本/向量、标量尾同式，
  分块内逐位一致。对独立标量参考（左结合写法）相对误差 <1e-9。

> **Rust 侧有免分配版 `api::norm3_batch_into(xs, ys, zs, out)`**：同一内核、与分配版**逐位一致**
> （分配版直接委托给它），只是输出缓冲由调用方给。热循环里反复调用时它省掉每次
> `AlignedVec::new` 的 **30.6 ns**（n=64 实测：155.6 → 124.9 ns/调用；`docs/dev.md` §7.14）。
> 一次性调用不必改。

### 6.2 `lasx_vec3_add_scaled_batch` — 批量缩放加（LASX+LSX）

`void lasx_vec3_add_scaled_batch(6×const double*, double s, 3×double*, int n)`

- 语义：`o[i] = a[i] + s·b[i]`，x/y/z 三分量独立（SOA）；覆盖加（`s=1`）与缩放（`a=0`）
  两个退化情形。
- 向量路径单条 FMA `lasx_xvfmadd_d(vs, vb, va)`；标量尾 `f64::mul_add(s, b, a)` 同式，
  分块内逐位一致。相对独立标量参考 <1e-9。

> **Rust 侧有免分配版 `api::vec3_add_scaled_batch_into(…, ox, oy, oz)`**：三个输出都由调用方给；
> n=64 实测省 **74.7 ns/调用**（159.9 → 85.2 ns）。

### 6.3 `lasx_batch_distance2d` — 批量 2D 距离（LASX+标量）

`void lasx_batch_distance2d(float px, float py, const float *xs, const float *ys, float *out, int n)`

- 语义：`out[i] = √(dx² + dy²)`，`dx = xs[i]-px`、`dy = ys[i]-py`。
- LASX 8 点/向量：`xvfsub_s` 求差、`xvfmul_s`+`xvfadd_s` 求平方和（`fadd(fmul,fmul)`）、
  `xvfsqrt_s` 开方；标量尾同式同结合序，分块内逐位一致。非 LASX 时**全标量**。

> **Rust 侧有免分配版 `api::batch_distance2d_into(px, py, xs, ys, out)`**：n=64 实测省
> **26.4 ns/调用**（94.8 → 68.3 ns）。

### 6.4 `lasx_ballistic_step` — 批量弹道欧拉步（LASX+标量）

`void lasx_ballistic_step(6×float*, const float *k, int n, float dt, float g)`

- 语义：SOA，n 发弹（每发 6 状态 + 阻力系数 k）同步推进一步**欧拉**，原地更新。
- 标量参考 `euler_step`（源码原样）：

  ```text
  v     = sqrt(vx² + vy² + vz²)
  drag  = k · v
  vx   -= drag · vx · dt
  vy   -= (drag · vy + g) · dt
  vz   -= drag · vz · dt
  x    += vx · dt        // 用更新后的新速度
  ```

- LASX 向量路径（8 发/向量）实际计算：

  ```text
  vsq  = vx² + (vy² + vz²)      // 右结合
  vmag = sqrt(vsq);  vdrag = k · vmag
  vnx  = vx − (vdrag·vx)·dt
  vny  = (vy − (vdrag·vy)·dt) − g·dt
  vnz  = vz − (vdrag·vz)·dt
  x'   = fma(vnx, dt, x)         // 位置用新速度
  ```

- 物理一致但**非逐位一致**（`|v|²` 结合序不同）。测试守护绝对误差（速度 <1e-3、
  位置 <1e-2；n=64、dt=0.005、g=9.81、k=1e-5）。早期版本的阻力符号、重力项、位置速度
  三处不一致**已修复**并对齐标量参考。非 LASX 时全标量。

### 6.5 `lasx_j2_accel_batch` — 批量 J2 引力加速度（LASX+LSX）

`void lasx_j2_accel_batch(3×const double* r, double mu, double j2, double re, 3×double*, int n)`

- 语义：中心点质量 + J2 摄动加速度，与 loong-sci `EarthModel::j2_acceleration` 同公式：

  ```text
  rm²  = x² + (y² + z²)          // 右结合
  rm = √rm²;  rm³ = rm·rm²;  rm⁵ = rm³·rm²
  j2k  = 1.5 · J2 · μ · Re²      // 取正值
  inv2 = 1/rm²;  invrm = inv2·rm;  inv3 = inv2·invrm;  inv5 = inv3·inv2
  zr2  = z²·inv2;  k = j2k·inv5;  vcen = −μ·inv3
  a_x  = fma(k·x, 5·zr2 − 1, vcen·x)
  a_y  = fma(k·y, 5·zr2 − 1, vcen·y)
  a_z  = fma(k·z, 5·zr2 − 3, vcen·z)
  ```

- `j2k` 取**正值**（注释"−k 即此值"），合并进末项 FMA 时符号为 `+k·x·(5·zr2−1)`，
  等价于减 J2 项。每样本完全独立，LASX 4 路 / LSX 2 路并行，无跨 lane 归约；向量与标量尾
  同式同 FMA，分块内逐位一致。对独立参考（不同公式实现）相对误差 <1e-9。

### 6.6 `lasx_rk4_j2_step_batch` — 批量 RK4 J2 步（LASX+标量）

`void lasx_rk4_j2_step_batch(6×double*, double mu, double j2, double re, double dt, int n)`

- 语义：经典 4 阶 Runge–Kutta 单步（中心项 + J2 力模型），批量推进 n 个轨道状态
  `[r(3), v(3)]`，**原地更新**（每块先读后写，块间不重叠）。
- 公式（FMA 形式，`h = dt`、`hh = dt/2`、`h6 = dt/6`）：

  ```text
  k1 = f(r0)
  r2 = fma(hh, v0, r0);  v2 = fma(hh, k1, v0);  k2 = f(r2)
  r3 = fma(hh, v2, r0);  v3 = fma(hh, k2, v0);  k3 = f(r3)
  r4 = fma(h,  v3, r0);  v4 = fma(h,  k3, v0);  k4 = f(r4)
  l  = fma(2, v2, v0) + fma(2, v3, v4)
  k  = fma(2, k2, k1) + fma(2, k3, k4)
  r' = fma(h6, l, r0);   v' = fma(h6, k, v0)
  ```

  `f(r)` 是 6.5 节 J2 加速度的 lane 版本（`vmu` 已预取负）。LASX 4 样本/向量，k1..k4 与
  组合算术全在寄存器内完成；非 LASX 走全标量 `rk4_j2_step_scalar`（同式同 FMA），分块内
  逐位一致。测试：50 步累积相对误差 <1e-9（dt=10）；另有 FMA vs 分离 mul+add 测试
  （单步 ≤1 ulp、50 步 <1e-9）。多核版本见 `parallel::rk4_j2_step_batch`。

### 6.7 `shape::batch` — 批量算子的形状层视图（N 进类型）

本节的算子都是"分量拆开的数组 + 一个 `n`"，长度只靠约定。形状层把这件事变成**构造时校验一次**：
`V3Ref`/`V3Buf`（x/y/z 三条）、`RvMut`（rx…vz 六条，原地）、`V2Ref`（x/y 两条，f32）。

```rust
use lasx_rs::shape::batch::{RvMut, V3Buf, V3Ref};
use lasx_rs::shape::{VecBuf, VecRef};

const N: usize = 1024;
let a = V3Ref::<f64, N>::new(&ax, &ay, &az)?;   // 三条不等长 ⇒ Err（点名是哪一条）
let b = V3Ref::<f64, N>::new(&bx, &by, &bz)?;
let mut mag = VecBuf::<f64, N>::new();
a.norm3_into(&mut mag);                          // = lasx_norm3_batch
let mut o = V3Buf::<f64, N>::new();
a.cross_into(&b, &mut o);                        // = lasx_cross3_batch
a.j2_accel_into(mu, j2, re, &mut o);             // = lasx_j2_accel_batch
a.add_scaled_into(&b, 0.5, &mut o);              // = lasx_vec3_add_scaled_batch
a.unitize3_into(&mut o);                         // = lasx_unitize3_batch

let mut rv = RvMut::<f64, N>::new(&mut rx, &mut ry, &mut rz, &mut vx, &mut vy, &mut vz)?;
rv.rk4_j2_step(mu, j2, re, dt);                  // = lasx_rk4_j2_step_batch（原地）
rv.position().at(0);                             // 取第 0 个样本的 (x, y, z)

let k = VecRef::<f32, N>::new(&k_arr)?;
let mut rv32 = RvMut::<f32, N>::new(&mut rx32, &mut ry32, &mut rz32,
                                    &mut vx32, &mut vy32, &mut vz32)?;
rv32.ballistic_step(&k, dt, g);                  // = lasx_ballistic_step（原地）
```

> **`api` 层的免分配孪生（本次补齐）**：上面演示的 `shape` 层方法**本来就是免分配的**
> （输出写进调用方给的 `VecBuf`/`V3Buf`），缺的一直是 `api` 层——它对五个小批量几何算子
> 只有分配型入口：`api::cross3_batch_into`（n=64 实测省 **88.6 ns/调用**：174.7 → 86.0）、
> `api::unitize3_batch_into`（省 **107.9 ns**：425.6 → 317.9），以及 §6.1/§6.2/§6.3 里那三个
> （`norm3_batch_into` 30.6 ns、`vec3_add_scaled_batch_into` 74.7 ns、`batch_distance2d_into`
> 26.4 ns）。三个输出的算子需要三份输出缓冲；与分配型入口**逐位一致**，由
> `src/api.rs` 的 `test_batch_geometry_into_bit_exact_and_contract` 守（覆盖 n = 0/1/7/8/9/
> 31/32/33/64/65）。口径与全部数字见 `docs/dev.md` §7.14 ③。

- **两组视图（按"长度在哪"分）**：`V3Ref`/`V3Buf`/`RvMut`/`V2Ref` 的长度在**类型里**（const `N`，
  适合形状编译期已知）；`V3Dyn`/`V3DynMut`/`RvDynMut`/`V2Dyn` 的长度是**运行期字段**
  （适合批量大小是运行时参数——下游库里的常态）。分工同 `shape::Mat` 与 `shape::MatDyn`。
  `*Dyn` 的构造以第一条分量为基准校验其余等长，输出数组在调用点再校验一次
  （`norm3_into`/`cross_into`/`distance2d_into` 等返回 `Result`）。
- **为什么要有这一层**：发布构建里长度不一致原本是 UB——内核用第一条数组的长度当 `n`
  （`let n = xs.len();`），而 FFI 转发层只有 `debug_assert_eq!`（发布构建编译掉）。
  三条/六条等长现在是 `new()` 的一次显式校验，**不等长就拿不到视图**，也就调不出内核。
- **数值**：这一层只做视图与转发，直接调 6.1–6.6 的内核，逐位一致（每个方法都有
  `to_bits()` 对照测试，覆盖 `N` 不是向量宽整数倍的尾部）。
- **别名**：输入是 `&V3Ref`（共享借用）、输出是 `&mut V3Buf`（独占），重叠由**借用检查器**拒绝。
- **不做的**：`quat_*` / `mat3_mul_vec3` / `cross3` 那一族（`docs/dev.md` §19.13 的 G4）
  在唯一的下游里没有任何调用点，所以不做"定点 3×3/4×4 展开后端"（`cross3`/`unitize3`
  是 3 分量视图的自然成员，顺手一起提供了）。
- **下游怎么用（推荐）**：自己的 SOA 缓冲类型上加一层访问器，把"三条字段 → 视图"收敛一次，
  调用点就只剩算子本身。例如 loong-sci 的 `Buf3`：

  ```rust
  // 自己的类型里（一次性）
  impl Buf3 {
      fn as_v3(&self) -> V3Dyn<'_, f64> { /* v3_of(&self.x, &self.y, &self.z) */ }
      fn as_v3_mut(&mut self) -> V3DynMut<'_, f64> { /* … */ }
  }
  // 调用点（每个算子一行）
  a.as_v3().add_scaled_into(&b.as_v3(), s, &mut o.as_v3_mut())?;
  ```

  这样不必给每个算子写宏（`docs/dev.md` §19.14 记了否掉 `batch!` 的实测理由）。


## 7. 批量姿态与几何算子（7 个）

公共约定：四元数**标量在前** `q = w + xi + yj + zk`，Hamilton 积 `i²=j²=k²=ijk=−1`；
旋转矩阵按 Wertz Ch.12.2 构造，方向是**体坐标 → 惯性坐标**（惯性转体请先取共轭）；
`|q| < 1e-15` 视为退化 ⇒ 输出单位四元数 `(1,0,0,0)`；分量求和结合次序固定，LASX / LSX /
标量尾三条路径**逐位一致**；全部内核**允许输出与输入别名**（向量路径先取完本次输入再
写回），因此 `quat_to_dcm_batch` 与 `mat3_mul_vec3_batch` 能串起来用。LASX 4 样本/向量、
LSX 2 样本/向量。

| 算子 | 语义与数值约定 |
|---|---|
| `lasx_cross3_batch` | 右手系 `o = a × b`，分量 `(ay·bz − az·by, az·bx − ax·bz, ax·by − ay·bx)`，与 loong-sci `orbit::cross3` 一致；逐位一致 |
| `lasx_unitize3_batch` | `o = v/\|v\|`；`\|v\|² = x²+(y²+z²)`，与 `norm3_batch` 同结合次序，故与"先 norm3 再逐分量除"逐位一致；`\|v\| == 0 → (0,0,0)` |
| `lasx_mat3_mul_vec3_batch` | `o = M·v`；`m0..m8` **行主序**（`M[r][c]=m[3r+c]`）；每行左结合 `((m0·x+m1·y)+m2·z)`；输出布局与 DCM 一致，可直接串联 |
| `lasx_quat_normalize_batch` | **原地**；范数左结合 `√(((w²+x²)+y²)+z²)`；`\|q\|<1e-15 → (1,0,0,0)`，否则逐分量**相除** `q/\|q\|` |
| `lasx_quat_mul_batch` | Hamilton 积：`.w=aw·bw−ax·bx−ay·by−az·bz`，`.x=aw·bx+ax·bw+ay·bz−az·by`，`.y=aw·by−ax·bz+ay·bw+az·bx`，`.z=aw·bz+ax·by−ay·bx+az·bw`；每行左结合，**不做内部归一化** |
| `lasx_quat_rotate_batch` | 输入**先单位化**（含退化规则），再 `o = R(q)·v`（体→惯）；`R` 与 `quat_to_dcm_batch` 共用同一份实现；每行左结合 `((r0·vx+r1·vy)+r2·vz)` |
| `lasx_quat_to_dcm_batch` | 输入**先按 7.4 规则单位化**（传未归一化四元数也安全）；输出行主序 `R = [1−2(y²+z²), 2(xy−wz), 2(xz+wy); 2(xy+wz), 1−2(x²+z²), 2(yz−wx); 2(xz−wy), 2(yz+wx), 1−2(x²+y²)]`，体→惯 |


## 8. Rust 安全层 `lasx_rs::api`

只走 rlib，不导出符号。切片进出、形状不符返回 `Err`、输出是 **64 字节对齐**的
`AlignedVec`；与 C ABI 走同一段内核，无可测开销。

| 函数 | 签名摘要 | 失败条件 |
|---|---|---|
| `api::sum` | `fn(&[f32]) -> f32` | 不会失败 |
| `api::dot` / `dot_f64` | `fn(&[T], &[T]) -> Result<T>` | 长度不一致 |
| `api::dot_i8` | `fn(&[i8], &[i8]) -> Result<i32>` | 长度不一致 |
| `api::dot_q4` | `fn(&[u8], &[f32], &[u8], &[f32]) -> Result<f64>` | 长度不一致；scale 长度 < `ceil(n/32)` |
| `api::axpy` | `fn(f32, &[f32], &mut [f32]) -> Result<()>` | 长度不一致 |
| `api::matmul` / `matmul_f64` | `fn(m, k, n, &[T], &[T]) -> Result<AlignedVec<T>>` | 形状不符、乘积溢出 |
| `api::matmul_f32_packed` | 同上，强制打包 B 面板路径 | 同上 |
| `api::norm3_batch` | `fn(3×&[f64]) -> Result<AlignedVec<f64>>` | 三数组不等长 |
| `api::batch_distance2d` | `fn(f32, f32, &[f32], &[f32]) -> Result<AlignedVec<f32>>` | 两数组不等长 |
| `api::vec3_add_scaled_batch` | `fn(6×&[f64], f64) -> Result<[AlignedVec<f64>; 3]>` | 六数组不等长 |
| `api::j2_accel_batch` | `fn(3×&[f64], mu, j2, re) -> Result<[AlignedVec<f64>; 3]>` | 不等长；`mu/re<=0`；`j2` 非有限 |
| `api::ballistic_step` | `fn(6×&mut [f32], &[f32], dt, g) -> Result<()>` | 七数组不等长（`dt`/`g` 不校验） |
| `api::rk4_j2_step_batch` | `fn(6×&mut [f64], mu, j2, re, dt) -> Result<()>` | 不等长；`mu/re<=0`；`j2`/`dt` 非有限 |
| `api::cross3_batch` | `fn(6×&[f64]) -> Result<[AlignedVec<f64>; 3]>` | 六数组不等长 |
| `api::unitize3_batch` | `fn(3×&[f64]) -> Result<[AlignedVec<f64>; 3]>` | 三数组不等长 |
| `api::mat3_mul_vec3_batch` | `fn([&[f64]; 9], 3×&[f64]) -> Result<[AlignedVec<f64>; 3]>` | 12 个数组不等长 |
| `api::quat_normalize_batch` | `fn(4×&[f64]) -> Result<[AlignedVec<f64>; 4]>` | 四数组不等长 |
| `api::quat_mul_batch` | `fn([&[f64]; 4], [&[f64]; 4]) -> Result<[AlignedVec<f64>; 4]>` | 八数组不等长 |
| `api::quat_rotate_batch` | `fn([&[f64]; 4], [&[f64]; 3]) -> Result<[AlignedVec<f64>; 3]>` | 七数组不等长 |
| `api::quat_to_dcm_batch` | `fn([&[f64]; 4]) -> Result<[AlignedVec<f64>; 9]>` | 四数组不等长 |

校验顺序与 `_checked` 一致（物理常数先于形状）；`dot_q4` 的 scale 按"≥ 组数"约定，多给合法。

`api::Error` 有四个变体，都带算子名与参数名，实现 `Display + std::error::Error`：

| 变体 | 触发条件 | 消息示例 |
|---|---|---|
| `Shape` | 长度/形状不符 | `matmul: 参数 b 的长度应为 k×n = 9，实际 4` |
| `Overflow` | `m×k` 等乘积溢出 `usize` | `matmul: m×k 溢出` |
| `NotFinite` | 常数是 NaN/±∞ | `rk4_j2_step_batch: 常数 dt 必须是有限数，得到 NaN` |
| `NotPositive` | 要求为正的常数 ≤ 0 | `j2_accel_batch: 常数 mu 必须为正，得到 0` |

```rust
use lasx_rs::api;

let a = [1.0f32, 2.0, 3.0, 4.0];
let b = [1.0f32; 4];

let d   = api::dot(&a, &b)?;                 // Result<f32, api::Error>
let s   = api::sum(&a);                      // 不会失败，直接 f32
let c   = api::matmul(2, 2, 2, &a, &b)?;     // Result<AlignedVec<f32>, _>
let out = api::norm3_batch(&xs, &ys, &zs)?;  // 输出已对齐

api::axpy(2.0, &x, &mut y)?;                 // 原地内核
api::rk4_j2_step_batch(&mut rx, &mut ry, &mut rz,
                       &mut vx, &mut vy, &mut vz, mu, j2, re, dt)?;

assert!(api::dot(&a, &b[..2]).is_err());     // 形状不符是 Err，不是 UB
```

危险的那条路仍然存在，只是要显式选它：`lasx_*` 参数是裸指针、函数本身不是 `unsafe fn`
（历史约定），**误用即 UB**——长度必须由调用方保证。

```rust
use lasx_rs::{lasx_dot, lasx_force_lsx_thread};

let d = lasx_dot(a.as_ptr(), b.as_ptr(), n as i32);
lasx_force_lsx_thread(true);          // 强制本线程走 LSX（仅测试/验证）
let d_lsx = lasx_dot(a.as_ptr(), b.as_ptr(), n as i32);
lasx_force_lsx_thread(false);         // 记得复位
```


### 8.1 矩阵视图 `view::MatRef` / `view::MatMut`

`api` 的矩阵接口把形状写成三个裸数字：`api::matmul(m, k, n, &a, &b)`。三个参数的含义以及
它们与切片长度的对应关系完全依赖调用方记忆——本仓库在此处发生过真实缺陷（把对齐后的列数
`n32` 当行跨距传进微内核、交换"既是循环上界又是 `B` 行跨距"的两个参数、基准中把 `(m,k,n)`
写反）。

`view` 把"形状 + 跨距"和数据放在一个对象里，**构造时校验一次**，之后按行列取数不出错：

| 构造/方法 | 说明 |
|---|---|
| `MatRef::row_major(&[T], rows, cols)` | 行主序视图（C 习惯：一行挨着一行） |
| `MatRef::col_major(&[T], rows, cols)` | 列主序视图（Fortran/BLAS 习惯，即"预先转置好的权重"） |
| `MatRef::row_major_strided(&[T], rows, cols, row_stride)` | 零拷贝切块：每行占 `row_stride` 个元素（可带 padding） |
| `rows()` / `cols()` / `is_empty()` / `row_stride()` / `col_stride()` | 形状与跨距读数 |
| `row(i)` → `RowRef` / `row_checked(i)` / `iter_rows()` | 取整行。`RowRef` 支持 `get(j)`、`as_slice()`（连续时）、`iter()`、`to_vec()` |
| `get(i, j)` | 单个元素（与布局无关） |
| `transpose()` | **零拷贝**转置视角；`m.transpose().row(j)` 就是第 `j` 列 |
| `as_row_major_contiguous()` → `Option<&[T]>` | 布局正好是连续行主序时给出切片（能不能直接下给向量内核的判据） |
| `to_row_major_vec()` | 不是那个布局时**明确复制**一份（代价写在名字里） |
| `MatMut::row_major(&mut [T], rows, cols)` / `row_mut(i)` / `fill(v)` | 可写视图（只提供行主序，内核输出都是行主序） |

失败发生在**构造**处，消息带期望值与实际值：

```rust
use lasx_rs::view::MatRef;

let data: Vec<f32> = (1..=12).map(|v| v as f32).collect();
let m = MatRef::row_major(&data, 4, 3)?;      // 4 行 3 列
assert_eq!(m.row(1).to_vec(), vec![4.0, 5.0, 6.0]);
assert_eq!(m.get(1, 2), 6.0);

// 同一块数据的列主序视角，零拷贝
let c = m.transpose();                         // 3 行 4 列
assert_eq!(c.row(0).to_vec(), vec![1.0, 4.0, 7.0, 10.0]);

// 长度不对：Error::Shape 说清期望与实际，而不是等算子里越界
let e = MatRef::<f32>::row_major(&data, 4, 4).unwrap_err();
assert!(e.to_string().contains("16") && e.to_string().contains("12"));
# Ok::<(), lasx_rs::api::Error>(())
```

> 为什么要多一个 `RowRef` 而不是直接返回 `&[T]`：**列主序矩阵的一行在内存里不连续**。
> 返回一个语义不成立的连续切片正是"行跨距用错"这类缺陷的温床，因此这里返回按下标取数的
> 小视图；布局连续时 `RowRef::as_slice()` 仍然给回真正的切片。


### 8.2 计划复用 `plan::MatmulPlan`

`api::matmul` 每次调用都要把 `B` 按列条带重新打包一遍（微内核才能顺序读 `B`，见 §5.5），
代价是 `O(k·n)`；小矩阵上它比乘法本身还贵。`MatmulPlan` 把这个打包**提前到构造时**，
之后每次调用只算乘法：

```rust
use lasx_rs::{api, plan::MatmulPlan, view::MatRef};

let (m, k, n) = (64, 256, 256);
let a = vec![0.5f32; m * k];
let b = vec![0.25f32; k * n];

let plan = MatmulPlan::from_row_major(&b, k, n)?;   // ① 打包一次
let c1 = plan.run(&a)?;                             // ② 反复算，m 由 a.len()/k 推出
let mut c2 = vec![0f32; m * n];
plan.run_into(&a, &mut c2)?;                        // ③ 写进自己的缓冲，零分配

// 与一次性 api::matmul 逐位一致
let want = api::matmul(m, k, n, &a, &b)?;
assert_eq!(&c1[..], &want[..]);
assert_eq!(&c2[..], &want[..]);
# Ok::<(), lasx_rs::api::Error>(())
```

| 接口 | 说明 |
|---|---|
| `MatmulPlan::new(&MatRef<T>)` | 从**任意布局**的 `B` 视图构造（行主序 / 列主序 / 带跨距 / 转置视角都行） |
| `MatmulPlan::from_row_major(&[T], k, n)` | 行主序 `B` 的便捷构造 |
| `run(&[T]) -> Result<AlignedVec<T>>` | 新分配对齐输出；`m = a.len()/k` |
| `run_into(&[T], &mut [T]) -> Result<()>` | 写进调用方缓冲。**只借用 `&self`**，可 `Arc` 给多线程各自写不相交的行段 |
| `k()` / `n()` / `packed_bytes()` | 形状与内存代价（打包缓冲字节数） |

边界与取舍：

- `k == 0` 时 `m` 无法从 `a.len()` 推出，构造函数直接返回 `Error::NotPositive`；退化情形用
  `api::matmul`。
- 计划里存着一份打包好的 `B`（`packed_bytes()`，通常就是 `k×n×size_of::<T>()`），构造要做
  一次 `O(k·n)` 搬运。**只算一次**就用 `api::matmul`（它按形状自选最快路径）。
- 计划固定走"打包 + k 分块"路径；`api::matmul` 在个别形状上会选流式/列块路径，两者仍逐位一致。
- 与 `matmul` 同前提：**需要 LASX**（LASX-only 清单见 §3.2）。
- 实测（本机单线程，`cargo run -p lasx_bench --release -- plan`）：把"每次重打包 + 每次新分配
  输出"两项都省掉后，`512³` f32 4.53 ms → 4.32 ms（1.05×）、`256³` f32 1.09×、
  `1000×64×4096` 1.04×、f64 `128³` 1.08×；详细的"① 一次性 API / ② C 复用 / ③ 免打包"
  三列对照与机制分析见 `docs/dev.md` §18。

> 计划是**多线程共享一份打包 `B`** 的正规做法：`run_into` 取 `&self`，把 `MatmulPlan` 放进
> `Arc` 即可，不需要 `parallel` 那一层，也不会有"每个线程各自打包一份 `B`"的病理
> （对比 `docs/dev.md` §13.7）。


### 8.3 形状进类型 `shape::Mat` / `shape::MatDyn`

前两节是"形状在运行期"。如果形状**编译期就知道**（模型结构固定、每层 `K/N` 写死在代码里），
`shape` 层把它搬进类型，于是：形状错误变成**编译错误**，排版（面板数、k 分块、列尾、
打包字节数）在 **const 求值**里定死。

| 类型 | 含义 |
|---|---|
| `Mat<'a, T, R, C>` | `R×C` 只读视图（零开销：`&[T]` + 两个 const） |
| `MatBuf<T, R, C>` | `R×C` 拥有者（输出），对齐缓冲 |
| `MatDyn<'a, T, C>` | **DYN 档**：行数运行时、列数（`K`）在类型里 |
| `MatBufDyn<T, C>` | DYN 档输出 |
| `Layout<K, N>` | 编译期排版方案（`PANELS` / `K_CHUNKED` / `TAIL_COLS` / `PACK_BYTES`…） |
| `Prepared<T, K, N>` | `B` 打包一次、反复用（内部就是 `MatmulPlan`） |

```rust
use lasx_rs::shape::{Mat, MatDyn, MatBufDyn};

const K: usize = 256;
const N: usize = 256;
let w = Mat::<f32, { K }, { N }>::new(&weights)?;   // 权重 K×N
let wp = w.prepare();                                // 打包一次

let x = Mat::<f32, 64, { K }>::new(&batch)?;
let y = x.mul(&w);            // ① 公式写法：K 由类型对齐 → MatBuf<64, K>
let y2 = wp.apply(&x);        // ② 声明式：复用打包好的权重

// ③ DYN 档：行数每批不同，K/N 仍是 const（排版照样编译期）
let xd = MatDyn::<f32, { K }>::new(&other_batch)?;
let mut yd = MatBufDyn::<f32, { N }>::with_rows(xd.rows());
wp.apply_dyn_into(&xd, &mut yd);
```

- **K 对不上是编译错误**，报错指在公式那一行（`expected 512, found 256`）；
- 构造函数只查长度，之后 `apply*` **不返回 `Result`**（形状已由类型担保）；
- `shape::auto_threads(rows, k, n, machine)` 是 `Auto` 判据的纯函数形式（多核接线在下一步）；
- 与 `api::matmul` **逐位一致**（走同一套打包 + k 分块内核，理由见 §2.5），
  含 DYN 档的多种行数。

**公式 DSL 里复用同一个 plan**：右操作数**不写方括号**就是 `Prepared`（它自带形状，方括号的
有无是天然区分信号）。这是消除"每次调用都重新打包"的正规写法：

```rust
use lasx_rs::matmul;
use lasx_rs::shape::{Mat, MatBuf};

const M: usize = 64;
const K: usize = 256;
const N: usize = 256;
let w = Mat::<f32, { K }, { N }>::new(&weights)?;
let x = Mat::<f32, { M }, { K }>::new(&batch)?;
let p = w.prepare();                      // 循环外一次：打包只发生在这里
let mut y = MatBuf::<f32, { M }, { N }>::new();
matmul!(y[M, N] = x[M, K] * p);           // 生成 `p.apply_into(&x, &mut y)`

// DYN 档（行数运行期）：`matmul!(yd[m, N] = xd[m, K] * p)` → `apply_dyn_into`
```

带方括号的写法 `matmul!(y[M, N] = x[M, K] * w[K, N])` 行为**完全不变**——它生成
`x.mul_into(&w, &mut y)`，而 `Mat::mul_into` 内部是 `rhs.prepare().apply_into(self, out)`
⇒ **每次调用都重新打包**。层循环里复用同一权重时两者的差是**纯打包开销**：256³ 实测
`142.2 µs → 99.3 µs`（4 轮丢弃首轮取中位数，**省 30%**），两臂与手写 hoist **逐位一致**。
plan 只能写在乘积**右侧**（左侧留给标量前缀的诊断）；**不做**隐式/指纹缓存——那会在缓冲区
复用时给出陈旧结果，与位精确承诺冲突。

> 三层的分工：**`view` 管布局**（列主序/跨距/转置）、**`plan` 管复用**（打包一次）、
> **`shape` 管形状**（进类型）。它们可以叠：`shape::Prepared` 内部就是 `plan::MatmulPlan`。

#### 8.3.1 一维（向量）与 f16：`dot!` / `gemv!`

矩阵之外，形状层还有两个子模块；`dot!` 与 `gemv!` 是 `matmul!` 的**一维形态**：

| 类型 | 含义 | DSL |
|---|---|---|
| `VecRef<'a, T, N>` | `N` 长只读向量视图 | `dot!` 的操作数 |
| `VecBuf<T, N>` | `N` 长拥有者（输出），32 字节对齐 | `gemv!(y[N] = …)` 的输出 |
| `F16Vec<'a, N>` | f16（`u16` 位型）权重向量 | `dot!(w[K] * x[K])` |
| `F16Mat<'a, N, K>` | f16 权重矩阵，**行主序 `[输出, 收缩]`** | `gemv!(y[N] = w[N, K] * x[K])` |
| `DotPair` / `dot_of` | 点积按 dtype 分派（f32·f32、f64·f64、f16·f32 双向） | `dot!` 的落点 |

```rust
use lasx_rs::shape::{F16Mat, VecBuf, VecRef};
use lasx_rs::{dot, gemv};

const K: usize = 4096;
const N: usize = 2;
let x = VecRef::<f32, K>::new(&act)?;                 // f32 激活
let y: VecBuf<f32, N> = gemv!(w_f16[N, K] * x[K]);    // 分配输出
gemv!(y[N] = w_f16[N, K] * x[K]);                     // 或写进已有缓冲
let s: f32 = dot!(act[K] * act[K]);                   // 点积（标量结果）
assert_eq!(y.as_slice(), lasx_rs::api::gemv_f16(N, K, &w_bits, &act)?.as_slice());
```

- **只支持编译期长度**（v1）：`VecRef`/`F16Vec`/`F16Mat` 的 `N`/`K` 是常量，小写下标会得到
  一条明确的诊断；运行期长度直接用 [`api::gemv_f16`](#513-lasx_dot_f16--lasx_gemv_f16--f16-权重的点积与矩阵-向量)
  或 §9 的 `parallel::gemv_f16`；
- **`gemv!` 只认一种读法**：权重在左、行主序 `[输出, 收缩]`（内核的布局）。写成
  `x[K] * w[N, K]` 会在**编译期**报错并给出正确写法；
- 数值与 `api`/`parallel` 对应函数**逐位一致**（同一后端；示例 `examples/formula_vec.rs`
  内联断言，含池化路径）；
- 生成代码用**绝对路径**（`lasx_rs::shape::…`），所以用户不必 `use` 类型或 trait——
  只有构造视图时才需要类型名。
- 覆盖缺口（`dot!`/`gemv!` 之外还有什么没有声明式写法）见 `docs/dev.md` §19.13。


## 9. 多核并行：`WorkerPool` 与 `parallel`

### 9.0 实战：用公式 DSL 做**单边数值拉普拉斯变换**（2026-10-15 加）

**做什么**：把 `F(s) = ∫₀^∞ f(t)·e^{-st}dt` 离散成**两条矩阵乘**，于是**一次 `matmul!` 算一批
信号**（`NB` 条并列）——这是公式 DSL 的卖点（形状进类型、位精确、自动并行）。

**数学与离散化**（`t_i = i·Δt`、`s_j = σ_j + iω_j`、`w_i` = 梯形权重）：

```text
Kc[j,i] = w_i·e^{-σ_j t_i}·cos(ω_j t_i)      Ks[j,i] = w_i·e^{-σ_j t_i}·sin(ω_j t_i)
Re F(s_j) = Σ_i Kc[j,i]·f_i                Im F(s_j) = -Σ_i Ks[j,i]·f_i
```

**跑起来**（两条：库模块 + 示例）：

```bash
cargo run --release --example laplace      # 解析对照 + 加密收敛表 + 同进程性能演示
# 或用库里的入口（形状是编译期常量，见 examples/laplace.rs 的用法）
cargo test --release --lib laplace         # 5 个测试：3 组解析对照 / 收敛 / 边界 / 位精确
```

```rust
use lasx_rs::laplace::laplace_batch_f32;
// 形状 M（评估点）× N（采样点）× NB（批量信号）都在类型里
let (re, im) = laplace_batch_f32::<M, N, NB>(&sigs, &oms, &dt, &f);   // f 是 N×NB 行主序
```

**误差量级（实测，`examples/laplace.rs` ②）**：固定区间 `T=6.4`、`f=e^{-0.5t}`、`f64`，
`N ∈ {64,128,256,512}`（`Δt` 减半）⇒ 最大相对误差 **1.7e-2 → 4.3e-3 → 1.1e-3 → 2.6e-4**，
相邻比 **~4.0×**（即 **梯形法 `O(Δt²)`**，与理论一致）；`f32` 在 `N=512` 时约 **1e-3**。
地板由**截断**决定（`T` 有限），固定 `Δt` 只加 `N` 不再改善——要更准就**同时**加 `T` 与减小 `Δt`。

**权重方案与误差量级（2026-10-15 加）**：`Weighting::{Trapezoid, Simpson}`，**默认 `Trapezoid`**
（向后兼容：改默认会改变已有调用的结果与精度特征）。核矩阵构造方式不变、**只换 `w_i`**，所以
DSL 用法故事完全不变（仍是两条矩阵乘、批量仍一次调用算 `N` 条）。

| 方案 | 权重 `w_i`（乘 `Δt` 前） | 阶数 | 适用条件 |
|---|---|---|---|
| `Trapezoid`（默认） | `½,1,…,1,½` | **`O(Δt²)`** | 任意 `N` |
| `Simpson` | `1,4,2,4,…,4,1`，再 `/3` | **`O(Δt⁴)`** | **`N ≥ 3` 且 `N` 为奇数**（偶数个区间） |

**回退规则（实测 + 测试覆盖）**：`N < 3` 或 `N` 为偶数（区间数为奇数、凑不出配对的 1/3 面板）⇒
**整段回退梯形**，并由 `laplace_batch_*_with` 的第二个返回值**如实标出**。
**为什么不"只丢尾巴"**：末段用低阶规则会引入 `O(Δt²)` 项、把整体阶数拖回二阶；整段回退**阶数诚实**。

**实测（`examples/laplace.rs` ④，`T=20`、`f=e^{-2t}`、解析 `1/(s+2)`、`f64`）**：

| N | 梯形误差 | 相邻比 | Simpson 误差 | 相邻比 |
|---|---|---|---|---|
| 65 | 2.771e-1 | — | 4.773e-2 | — |
| 129 | 7.219e-2 | **3.84×** | 3.881e-3 | **12.30×** |
| 257 | 1.824e-2 | **3.96×** | 2.614e-4 | **14.85×** |
| 513 | 4.573e-3 | **3.99×** | 1.665e-5 | **15.69×** |

⇒ 梯形 ≈4×/次（二阶）、**Simpson ≈16×/次（四阶）**；同一 `N` 下 Simpson 准 **5.8–274×**。
**⚠️ 阶数只在不被 `T` 的截断地板支配时才可见**：`T=6.4` 时 Simpson 很快撞到 ≈7e-5 的地板、
相邻比掉到 ≈1（而梯形仍 ≈4）——这条现象有测试断言（`convergence_order_trapezoid_vs_simpson`）。

**性能（同进程口径，`docs/dev.md` §6.4）**：`M=384, N=1024, NB=64` @12 线程
（每线程 32 行 ⇒ **命中动态认领阈值**）⇒ **DSL `matmul!` 0.1367 ms / 368 GFLOP/s
（0.868× 手动 plan）**；说明 **DSL 默认的自动策略（共享池 + 面板复用 + 按形状自动切分）已跟上**。

**边界与限制（必须知道）**：
- **不是任意精度**：截断 + 梯形 `O(Δt²)`，见上；要更高精度请减小 `Δt` 并加大 `T`；
- **`s` 靠近极点时不能用**：`f≡1` 的 `s=0` 真值发散，本函数给的是**截断和** `(N-1)Δt`；
  `ω=0` 时虚部**恒为 0**（`sin(0)=0`，有测试守着）；
- **不做逆变换**：只做正向（Bromwich/Mellin 反演不在范围内）；
- **`f` 是 `N×NB` 行主序**，`NB` 条信号**并列**（`NB=1` 也可以）。

内核本身是单线程的（每个 `lasx_*` 只处理一段连续内存）。多核靠可选的**常驻**线程池。

**切分策略按形状自动选**（2026-10-15 起；判据来自 `docs/dev.md` §7.10 ⑩ 的实测阈值表）：
**大形状（`per_thread ≥ 32` 且 `threads ≥ 12`）走动态小额认领**（实测 **+4.9%…+21.4%**），
其余（含 LLM 那种中等形状、宽薄、小形状）走**静态等分**（实测静态更快）。等待策略是
"短自旋 `SPIN_LIMIT = 1024` 圈 → `park`（`Condvar`）"——实测该默认值已最优
（`worker_loop` 自耗时 **<0.76%**），**不必调**。

| 接口 | 签名摘要 | 切分形状 |
|---|---|---|
| `WorkerPool::new` | `fn(threads: usize) -> Self` | 建池（≥1），**建一次、跨调用复用** |
| `WorkerPool::auto` | `fn() -> Self` | 按 `available_parallelism()` 建池 |
| `WorkerPool::threads` | `fn(&self) -> usize` | 池内线程数 |
| `pool::global` | `fn() -> &'static WorkerPool` | **进程级共享池**（首次使用时就地建），多线程共用同一个 |
| `pool::init_global` | `fn(threads) -> bool` | 首次使用前指定全局池线程数（已建好则返回 `false`） |
| `for_each_chunk_mut` | `fn(&self, data: &mut [T], f: F)` | 单数组按元素等分 |
| `for_each_chunks_mut` | `fn(&self, arrays: [&mut [T]; N], f: F)` | N 个**等长**数组同步等分 |
| `for_each_row_block_mut` | `fn(&self, rows, row_gran, arrays: [(&mut [T], usize); N], f: F)` | 按行块切分，行宽可不同；闭包首参为本块行数 |
| `for_each_row_block_mut_picked_work` | 同上 + `(work: usize, pick: Pick)` | **并行门槛用显式 `work`**（数组长度之和不代表工作量时用，如 `gemv_f16`） |

约束与语义：

- `MIN_PARALLEL_LEN = 4096`：一次调用涉及的元素总数低于该值时**自动原地串行**（单数组即
  长度；SOA 是各数组长度之和）。`MAX_ARRAYS = 16`：单次派活最多数组个数。
- **派活接口取 `&self`**：池内部用一把互斥把派活者串行化，所以多个线程可以同时拿同一个池
  派活（排队执行，结果各自正确），进程级共享池 [`pool::global`] 正是靠这条成立。代价是
  重入派活（在 `during` 回调里再派活）不再被借用检查器挡住，改由运行期 panic 拒绝
  （消息里带"重入"），而不是在锁上死等。
- worker 里闭包 panic 会被 `catch_unwind` 拦下并计数，主线程等全部 worker 收工后
  `resume_unwind` **原样续抛**；池状态干净、可继续复用（否则主线程会永久自旋）。
- **`during` 回调 panic 不会留下悬垂闭包**：发布之后挂析构守卫，展开路径也会先把 worker
  收干净（详见 `docs/dev.md` §3.2）。
- 每次调用**零分配**：作业槽是定长数组，只做指针算术。等待策略：先自旋 1024 次，仍无任务
  就 `Condvar` park（纯自旋在过订阅时会互相抢执行槽）。
- `for_each_chunks_mut` 各数组长度不一致时 panic；`for_each_row_block_mut` 在
  `row_gran == 0` 或第 k 个数组长度不等于 `rows × 行宽` 时 panic。`row_gran` 不要省：
  块大小会**向上取整到它的倍数**，让块边界落在微块边界上；传 1 表示没有行结构
  （实测 `row_gran=1` 并不更慢，见 `docs/dev.md` §14）。

`parallel` 策略层把常见负载的切分形状固定下来：

| 函数 | 说明 |
|---|---|
| `parallel::matmul_f32(&pool, m, k, n, a: &mut [f32], b: &[f32], c: &mut [f32]) -> Result<(), Error>` | 多核矩阵乘（行粒度 4），B 只读共享；`&pool` 可直接给 `pool::global()` |
| `parallel::matmul_f64(...)` | 同上，f64 |
| `parallel::gemv_f16(&pool, m, k, a: &[u16], x: &[f32], y: &mut [f32]) -> Result<(), Error>` | 多核 f16 权重 GEMV（行粒度 1）；只读权重由闭包按 `start` 切片，与单线程逐位一致 |
| `parallel::rk4_j2_step_batch(&pool, mu, j2, re, dt, 6×&mut [f64]) -> Result<(), Error>` | 多核批量 RK4 J2 单步（6 数组一次派活） |

**`parallel::*` 形状不符返回 `Err`**（与 [`api`](#8-rust-安全层-lasx_rsapi) 同一套
`Error::Shape`/`Error::Overflow` 与消息口径）；`matmul_*` 对 `m==0 || n==0` 直接返回 `Ok`
（不写 C），`k==0` 时 `c.fill(0.0)`。**池的低层 `for_each_*` 仍按契约 panic**——它是
"调用方自己保证形状"的原语，`parallel` 才是带校验的那一层。
数值与单线程**逐位一致**（只切行/下标区间），有逐位对照测试。
`parallel::rk4_j2_step_batch` 的 worker 是复用线程，`lasx_force_lsx_thread` 对它无效。

**线程数怎么选（实测 + 运行期探测）**：中小规模内核取**物理核数**，并且**给主线程留一个
空位**——别硬编码机型数字，用 [`arch::parallelism`]：

```rust
let p = lasx_rs::arch::parallelism();          // { logical, physical }，进程级缓存一次
let threads = p.physical.min(p.logical - 1).max(1);
let pool = lasx_rs::pool::WorkerPool::new(threads);
```

依据（本机 24 逻辑核 / 12 物理核，`docs/dev.md` §20.7）：

- **块数 < 线程数**的派活台阶**已修**：原协议等"全体 worker 签到"，没活的 worker 被调度到 CPU
  的时机成了关键路径（纯派活探针 `m=2, k=2048` 在 23/24 线程是 37/38 µs）。改成"逐槽代次 +
  只等有活的 worker"后回到 **1 µs**；`gemv 64×2048`@24 线程 0.043 → 0.007 ms（5.9×）。
- **worker 全都有活且占满 CPU** 时仍有调度代价：`gemv 1024×2048` 24 线程 0.078 vs 12 线程
  0.041 ms（25 个可运行线程挤 24 个 CPU，最后收尾的 worker 要等下一拍）。此时少给线程、
  留一个空位即可；矩阵乘同样受影响（128×256×256：22 线程 293 GF/s、24 线程 152 GF/s）。
- **权重 ≥32 MiB 是例外**：24 线程仍更快（4096×4096 0.73 vs 1.07 ms），那时多出来的逻辑核
  有活干、且几十微秒相对毫秒级调用可忽略。
- 完整表、cpuset 双向对照与"绑物理核不提速"的对照见 `docs/dev.md` §20.7。

```rust
use lasx_rs::aligned::AlignedVec;
use lasx_rs::pool::WorkerPool;

let pool = WorkerPool::new(12);            // 建一次，长期持有（也可以直接 `pool::global()`）

pool.for_each_chunk_mut(x.as_mut_slice(), |chunk| { /* ... */ });          // 单数组
pool.for_each_chunks_mut([rx, ry, rz], |[rx, ry, rz]| { /* ... */ });      // SOA 等长数组

let (bs, cs) = (b.as_slice(), c.as_mut_slice());                            // 行块
pool.for_each_row_block_mut(m, 4, [(&mut a, k), (cs, n)], |_start, rows, [ab, cb]| {
    lasx_rs::lasx_matmul(rows as i32, k as i32, n as i32,
                         ab.as_ptr(), bs.as_ptr(), cb.as_mut_ptr());
});

lasx_rs::parallel::matmul_f32(&pool, m, k, n,
                              a.as_mut_slice(), b.as_slice(), c.as_mut_slice())?;
for _ in 0..steps {
    lasx_rs::parallel::rk4_j2_step_batch(&pool, mu, j2, re, dt,
                                         rx, ry, rz, vx, vy, vz)?;
}
# Ok::<(), lasx_rs::api::Error>(())
```

> 池是要复用的：把 `WorkerPool::new` 放进热路径等于退化成"每次新建线程"。
> 拿不定主意就用 [`pool::global`]（一个进程一个池，多线程共用，见 §8.2 的 `Auto` 策略）。
> 完整可运行示例：`cargo run --release --example pool_axpy`、`--example matmul_pooled`
> （都自带与单线程的逐位对照）。


### 9.5 调度策略（`pool::sched`）：编译期选路 + 运行期穷尽枚举

切分方式独立成了一层，调用方可以显式指定：

| 策略 | 形状 | 什么时候用 |
|---|---|---|
| `Pick::Chunk` | 连续等分 | 每行代价相同的内核（`sum`/`dot`/`axpy`/SOA 批量） |
| `Pick::RowBlock` | 等分 + 对齐行粒度 | 矩阵乘（行粒度 4）；**默认** |
| `Pick::Blocked { block_rows }` | 固定块长、块数可多于线程数 | 轻度负载不均 |
| `Pick::Dynamic { block_rows }` | 原子计数器抢块 | 线程数超过物理核、每线程仍有 ≥32 行（实测 1024³/24 线程 +23%） |

- 策略是**零尺寸类型**（`sched::Chunk`/`RowBlock`/`Blocked<BLOCK>`），入口
  `pool.for_each_row_block_mut_with::<S, …>(…)` 对策略泛型化 ⇒ 编译期单态化、无运行时分支；
- 运行期选择用穷尽枚举 `Pick` + `pool.for_each_row_block_mut_picked(…)`：新增策略时
  所有分派点都会编译失败，逼作者逐处确认，而不是悄悄落进默认分支；
- `pool.for_each_row_block_mut_picked_deferred(…, f, during)`：在"已派活、尚未等待"时由主线程
  执行 `during`（矩阵乘用它做**双缓冲打包**：worker 算当前面板时主线程打包下一个，
  多面板形状实测 +24~29%，见 `docs/dev.md` §16）。`f`/`during`/数据在同一栈帧 ⇒ 不需要 unsafe；
- `parallel::matmul_f32` 默认用 `pool::pick_rows(m, threads, 4)` 自动选；
  `parallel::matmul_f32_with_pick(…)` 可显式指定（调优/对比用）；
- 切分本身是纯函数：`sched` 的测试对 `rows × threads × gran` 的组合断言"恰好覆盖
  `[0, rows)`、块不重叠、非末尾块对齐粒度"，池级测试断言"每种策略恰好访问每行一次"。

### 9.05 逆变换：Dubner–Abate + **跳变扣除**（2026-10-16 加）

**做什么**：由 `F(s)`（调用方实现 `ComplexFn`）求 `f(t)`，`t ≥ 0`（`cargo test --release --lib ilaplace`）。
**与正向的关系（别把故事说大）**：`F` 是**调用方代码**、必须逐点求值；**DSL 只负责"指数和"那部分**
（`cos/sin` 核 × `F` 值 → 两条实矩阵乘），且天然批量（`NB` 个 `F` 一起算）。**`F` 贵时主导成本在 `F`**。

**公式（跳变扣除 + D–A 级数）**：

```text
f(t) = f₀·u(t) + L⁻¹{ F(s) − f₀/s },      f₀ = f(0⁺) = lim_{s→∞} s·F(s)   ← **调用方给**
f̃(t) ≈ (e^{a t}/T)·[ ½F̃(a) + Σ_{k=1..N} Re{ F̃(a + i kπ/T)·e^{i kπ t/T} } ],  ω_k = kπ/T
```

**为什么必须扣除跳变**（本模块第一版就是栽在这里）：纯 D–A 的级数周期是 `2T`，而 `h(t)=f(t)e^{-at}`
在 `t=0/2T` 处**不连续**（`h(0)=f(0⁺)≠0`、`h(2T)≈0`）⇒ 系数按 `1/k` 衰减 ⇒ **误差 O(1/N)**。
实测（`F=1/s`、`T=12`、`a=1`）：**0.307(50) → 9.88e-2(150) → 4.98e-2(300) → 2.50e-2(600)**，
每翻倍只减半 ⇒ 要 `1e-3` 得 `N≈15000`。扣除 `f₀` 后 `h̃(0)=0` ⇒ **偶/奇延拓连续** ⇒ 收敛恢复。

**N 扫描并排（`F=1/(s+1)`、`t ∈ {0.5,1,2,3,4}`、`f₀=1` 对 `f₀=0`）**：

| N | **扣除 f₀=1** | 未扣除（f₀ 取错为 0） |
|---|---|---|
| 64 | 1.57e-2 | 1.14e-1 |
| 128 | 1.74e-3 | 1.20e-1 |
| 256 | 1.01e-3 | 2.98e-2 |
| 512 | **1.22e-4** | 2.96e-2 |

⇒ **扣除后 `N=512` 已到 1.22e-4（≤1e-3 达标 ✓）**；未扣除时**卡在 ~3e-2 的 1/N 地板**上不动。
**`f₀` 取错的后果就是这张表**：残留跳变 ⇒ 退回 1/N ⇒ 误差随 `N` 只按 `1/N` 降 —— 可用它自诊。

**`f₀` 的契约（谁保证）**：**首选调用方给**（`DubnerAbate::with_f0`；`f₀` 是初值定理的解析量）；
默认 `f₀ = 0`（不扣除，仅适用于 `f(0⁺)=0` 的 `F`）；`estimate_f0(f, s_probe)` 是**数值估计**，
**不可靠**，要用**必须显式**传进 `with_f0` —— **没有任何静默回退**。

**硬约束（`DubnerAbate::check` 会拦）**：① **周期 `2T`** ⇒ 只在 `0 < t < 2T` 内代表原函数
（建议 `T ≥ 3·t_max`）；② **`a·t_max ≲ 4…8`**（实测 `a=6 → 1e7`、`a=12 → 1e17`：`e^{at}` 放大绝对截断误差）。
**`t=0` 给跳变平均值**（`F=1/s` 时 ≈`f₀/2`）—— Fourier 反演的**已知行为，不是 bug**（有断言守着）。

**位精确（已证成）**：指数和的 DSL 路径与行切参考**逐位一致** —— `f32` 用 `api::matmul`、
`f64` 用同序手工（`api::matmul` 只有 `f32` 版）。
**交叉验证**：与**独立路线**（固定围道 `Re s = c` 的 Bromwich 梯形积分，无周期陷阱）结构一致（≤1e-1；
该参考自身只有 ~5e-2 精度，**精度不靠它保证**，靠解析对照 ≤1e-3）。

**限制清单**：不能保证任意精度 · `F` 的奇点须在 `Re s < a`（分支切割同理）· **逆问题病态** ·
`0 < t < 2T` · `t→0` 误差抬升（Gibbs） · **依赖 `f₀` 的正确性**。

### 9.06 复数矩阵乘（A 阶，2026-10-16 加）

**表示**：**分平面 `(Re, Im)` 两块实数组**（`ar/ai/br/bi` 行主序；`cr/ci` 复用调用方缓冲，
风格与库内 `_into` 一致）。**f32/f64 两条路径**：`ops::cmatmul::{cmatmul_f32_into, cmatmul_f64_into}`
（默认路线）与 `*_into_route(..., Route::{Direct, Karatsuba})`（显式选路线）。
**A–C 阶不加 C 符号**（`nm -D` 仍 94）。

| 路线 | 实矩阵乘条数 | 位精确契约（**谁守**） |
|---|---|---|
| `Route::Direct` | **4**：`Re=Ar·Br−Ai·Bi`、`Im=Ar·Bi+Ai·Br` | **每条乘积与对应实矩阵乘逐位一致** —— 测试 `direct_matches_four_real_matmul_bitwise`（与 4 条实内核的组合**逐位**比对）；和/差按 IEEE 逐元素 |
| `Route::Karatsuba`（**默认**） | **3**：`k1=(Ar+Ai)(Br+Bi)`、`k2=ArBr`、`k3=AiBi` | **只承诺容差**（`k1` 多一次加法、组合次序不同）⇒ **不承诺**与直算逐位；测试 `routes_agree_within_tolerance` + 本文档声明 |

**"完全 SIMD"的判据**：**每一次复数乘加都由 LASX 实矩阵乘内核完成、无标量回退** ——
**两条路线都满足**（4 条 / 3 条乘积都走 `ops::matmul::matmul_f32`/`matmul_f64`）；
Karatsuba 额外付 `Ar+Ai`/`Br+Bi` 两次**逐元素**加法（SIMD 化，但不是矩阵乘）。
**⚠️ 诚实边界（未逐指令核对）**：**乘法累加**确实全在 LASX 内核里；但最后把 4/3 条乘积
**组合**成 `Re`/`Im` 的那一步是**逐元素加减的标量 Rust 循环**（`cr[l]=ar_br[l]-ai_bi[l]` 之类，
LLVM 可能自动向量化，**但没有用 LASX intrinsic**）⇒ 若把"完全 SIMD"理解为**连这些加减也要 intrinsic**，
则**这一处尚未达标**（已记在此）。**佐证方式**：本轮**没有**做 `objdump`/`perf` 的逐指令核对
（**如实写"未逐指令核对"**），B 阶融合内核会把加减也一起收进向量寄存器，届时再补。

**同源性能（`diag_cmatmul_routes`，`192×768×3072`，同进程交替 6 对丢首对）**：

| 路线 | ms（中位） | 复数 GFLOP/s | 物理效率比¹ | 字面效率比² |
|---|---|---|---|---|
| `Direct`（4 条） | 63.13 | 57.4 | 0.953 | 0.238 |
| **`Karatsuba`（3 条）** | **53.35** | **67.9** | **1.128** | 0.376 |
| 单条实矩阵乘（基准） | 15.04 | 60.2 | — | — |

¹ 复数 GFLOP/s ÷ 单条实矩阵乘 GFLOP/s（`k` 条**串行**时的合计速率就是单条速率）。
² 复数 GFLOP/s ÷ (`k` × 单条实矩阵乘 GFLOP/s) —— 分母含义是"`k` 条**并行**时的合计速率"。
⇒ **选优：默认 `Karatsuba`**（同源快 **18%**；代价是放弃逐位、只承诺容差 ⇒ 要位精确请显式 `Route::Direct`）。

**⚠️ 成本口径（必须这么读，别误读成"复数比实数快"）**：与"**调用方自己调 `k` 条实矩阵乘**"比，
复数版**还是更贵**——
`Karatsuba` 53.35 ms vs **3 条 = 3×15.04 = 45.1 ms** ⇒ **贵 18%**；
`Direct` 63.13 ms vs **4 条 = 4×15.04 = 60.2 ms** ⇒ **贵 5%**。
⇒ **结论：复数 API 目前不是"比你自己调实矩阵乘更快"**；它的价值是**一次调用 + 正确的复数语义 +
可选逐位（`Route::Direct`）+ 后续融合内核（B 阶）的改进空间**。
**"效率比"只在同一路线内比才有意义**（"Karatsuba 比 Direct 快 18%"是真的；
"物理效率比 1.128"**不能**读成"复数比实数快"——它的分母是"`k` 条**串行**实矩阵乘"的合计速率）。
**宁可记"我们现在只是把它做对了，还没做快"。**
**线程数**：本模块是**串行核心**、**未做自动多线程**；要多线程请调用方**按分平面**切（各条实矩阵乘走
`parallel::*` + 计划/池复用）。小形状不退化由串行核心本身保证（无池开销）。
**限制**：Karatsuba 的 `Ar+Ai` 在极端量级下可能溢出（f32 尤甚）· 分平面需要 4 块输入/2 块输出 ·
**无**与实矩阵乘同级的位精确承诺（除直算路线的"各乘积逐位"）。

## 10. 对齐缓冲：`AlignedVec` 与 `lasx_alloc`

Rust 侧用 `AlignedVec`（纯 Rust，不导出符号）：

| 接口 | 说明 |
|---|---|
| `AlignedVec::new(len)` | 分配 `len` 个零值元素，首地址 64 字节对齐（`T: Copy + Default`） |
| `AlignedVec::fill_with(len, f)` | 分配并用 `f(i)` 填充 |
| `len` / `is_empty` / `as_slice` / `as_mut_slice` | 切片视图 |
| `as_ptr` / `as_mut_ptr` | 保证对齐的裸指针，可交给 C ABI |
| `Deref` / `DerefMut` | 可直接当 `&[T]` / `&mut [T]` 用（索引、`len()`、迭代、传参） |
| `Clone` | 克隆后**重新计算对齐偏移**并复制数据 |

**C 侧的 `lasx_alloc` / `lasx_free`**（32 字节对齐、未初始化；必须用 `lasx_free` 按同一个 `n`
归还；`free()` 不是 ABI 契约；`p == NULL` / `n <= 0` 是安全 no-op）**见
[`docs/c-abi.md`](c-abi.md) 第 4 节**——那里是这条契约的唯一权威。

对齐只影响性能、不影响正确性：LASX 的 `xvld/xvst` 是 32 字节访问，起始地址不落在 32 字节
边界时会跨缓存行。实测 glibc `malloc` 与 Dart FFI 的缓冲区只有约一半落在 32 字节边界上，
**不要假设缓冲区天然对齐**；未对齐时结果照样正确，且仍明显快于 LSX 路径。想稳定拿到对齐
内存：Rust 用 `AlignedVec`，C 用 `lasx_alloc`（配 `lasx_free`）或 `posix_memalign(&p, 32, n)`。
量化对比见 `cargo run -p lasx_bench --release -- align`，完整数据详见 `docs/dev.md`。


## 11. FFI 用法（C / Dart）

**已搬移**：C / C++ / Dart 的完整用法——四种链接方式与最小可运行程序、Dart 的
`lookupFunction` 例子、长度/生命周期/别名规则、裸 vs `_checked`、能力探测顺序——见
[`docs/c-abi.md`](c-abi.md)（第 2 节快速开始、第 4 节内存、第 5 节安全模型、第 6 节能力与降级）。
Rust 侧调用方式见 §8 与 §9。


## 12. NN 算子：当前状态与候选

> 这一节由原 `docs/nn-ops.md` 并入（该文件已删除），参照系是 llama.cpp / ggml。
> 这里给"当前状态 + 用法"，本机实测数字只作缺口依据引用，方法学与完整数据见 `docs/dev.md`。

### 12.1 ggml 算子全景（分类）

`ggml.h` 的 `enum ggml_op` 共约 102 个算子，主要类别：

| 类别 | 代表算子 |
|---|---|
| 一元/激活 | `UNARY`（SILU/GELU/GELU_QUICK/GELU_ERF/RELU/TANH/SIGMOID/EXP/NEG）、`GLU`/`GEGLU`、`LEAKY_RELU`、`CLAMP`、`SQR`/`SQRT`/`LOG`/`SIN`/`COS` |
| 逐元素二元 | `ADD`/`ADD1`/`ACC`、`SUB`、`MUL`、`DIV`、`SCALE`、`SET`、`CPY`/`CONT`、`DUP` |
| 归约 | `SUM`、`SUM_ROWS`、`CUMSUM`、`MEAN`、`ARGMAX`、`ARGSORT`、`TOP_K` |
| 矩阵 | `MUL_MAT`、`MUL_MAT_ID`、`OUT_PROD` |
| 形状/布局 | `RESHAPE`/`VIEW`/`PERMUTE`/`TRANSPOSE`/`CONCAT`/`REPEAT`/`PAD`/`ROLL`/`IM2COL`/`COL2IM` |
| 归一化 | `RMS_NORM`、`NORM`（LayerNorm）、`GROUP_NORM`、`L2_NORM` |
| 注意力 | `SOFT_MAX`、`ROPE`、`DIAG_MASK_INF`/`ZERO`、`FLASH_ATTN_EXT` |
| 卷积/池化 | `CONV_2D`/`CONV_3D`/`CONV_2D_DW`、`POOL_1D`/`POOL_2D`、`UPSCALE` |
| 索引/嵌入 | `GET_ROWS`、`SET_ROWS`、`TIMESTEP_EMBEDDING`、`ARANGE` |
| SSM/线性注意力 | `SSM_CONV`、`SSM_SCAN`、`GATED_LINEAR_ATTN`、`RWKV_WKV6`/`7` |
| 训练 | `CROSS_ENTROPY_LOSS`、`OPT_STEP_ADAMW`/`SGD`、各 `*_BACK` |

一次前向每层固定出现：`MUL_MAT`、`RMS_NORM`×2、`ROPE`×1~2、`SOFT_MAX`（或
`FLASH_ATTN_EXT`）、`SILU`/`GELU`、`ADD`×2~3、`MUL`×1~2；解码时还有 `GET_ROWS`。
量化格式热档：`q4_0`（4.5 bit/权重）、`q8_0`（8.5）、`q4_K`（4.5）、`q6_K`（6.5625）、
`q2_K`/`q3_K`/`q5_K`、`mxfp4`/`nvfp4`、`iq*`（1.5–4.5）。

### 12.2 lasx_rs 已落地 / 缺口

**已落地、可直接复用**：`lasx_matmul`/`lasx_matmul_f64`（f32/f64 稠密矩阵乘）、
`lasx_dot_q4`（Q4，每 32 字节一组 scale）、`lasx_dot_i8`（int8）、`lasx_dot`/`lasx_sum`/
`lasx_dot_f64`/`lasx_axpy`（归约与逐元素原语）、`lasx_norm3_batch` 及姿态/几何 7 个、
`pool`/`parallel` 多核设施、`api` 安全层。

**缺口（"现状"列是 llama.cpp 在本机的实测；"已落地"列是本库 2026-09-25 复测的读数，
完整表见 `docs/dev.md` §7）**：

| 缺口 | llama.cpp 现状 | 本库现状 | 结论 |
|---|---|---|---|
| f16 GEMV（decode） | **6.8 GB/s** | `lasx_gemv_f16` **3.75–48.2 GB/s**（同量级形状 `1024×2048` = 10.5 GB/s ≈ 1.5×） | 小形状驻留缓存时领先明显；超出 L3 后回落到 3.75 GB/s，瓶颈是 `f16→f32` 转换与 `permi_q` 重排这两项结构代价（已查清，见 §5.13） |
| f16/bf16 GEMM（prefill） | 8.4 / 16.6 GFLOP/s | f32 GEMM 60–68 GFLOP/s（`docs/dev.md` §8.5），**没有 f16 GEMM** | 仍是未做的缺口（N1 只做了 f16 的 GEMV/dot） |
| `SOFT_MAX` | 典型 2.4–4 GB/s | **5.22–7.00 GB/s**（朴素 Rust 的 4.5–5.6×） | 已落地，1.3–2.9× |
| `SILU`/`GELU`（激活） | 无 LoongArch 分支（`vec.cpp` 里 silu/gelu 有 AVX/NEON/SVE，唯独没有 LASX） | **4.55–9.20 GB/s**（朴素 Rust 的 4.1–7.9×） | 已落地三个入口；**算力受限**（同为 2 条流，rms_norm 到 18.0 GB/s，激活只有一半——`exp` + 向量除法是瓶颈） |
| `RMS_NORM` / `ROPE` | llama.cpp 侧 ≈1×（没有缺口） | rms_norm **12.0–18.4 GB/s**、rope(NeoX) **28.6–71.9 GB/s** | 本库把这两者也 LASX 化了（llama.cpp 的 rope 两种 mode 都是标量） |
| `q6_K` | 9.8 GB/s | 未做 | ~1.8×（N2 批次，需要 GGML 块格式互操作） |
| `mxfp4`/`nvfp4` | 5.5 / 4.0 GB/s | 未做 | ~3–4.5×（N2） |
| 已经没机会的 | `q4_0`/`q4_K`/`q2_K`/`q3_K`/`q8_0`/`iq*`、`f32`、`ADD` | — | ≈1×：llama.cpp 的 `arch/loongarch/quants.c` 已 LASX 化，再挤空间很小 |

三条结论（**前两条已由 2026-09-25 的复测提供实测支撑**）：

1. 量化点积**不是机会**（llama.cpp 已把 q4_0/q4_K/q5/q6_K（部分）/iq* LASX 化）；
2. 真正的短板是**非量化**那半边：f16/bf16 与 softmax 在 llama.cpp 里根本没有 LoongArch 分支；
   本库已经把 softmax/rms_norm/激活/rope/f16 GEMV 补齐并给出实测（上表）；
3. 剩下的缺口集中在 **f16/bf16 的 GEMM（prefill）** 与 **N2 的 GGML 量化格式互操作**——
   它们都需要新的、更大的工作（前者要有 f16 微内核，后者要按 GGML 块布局写）。

### 12.3 候选批次

**N1（推荐先做，纯 f32/f16，无格式锁定）**

| 算子 | 说明 | 状态 |
|---|---|---|
| `dot_f16` / `gemv_f16` | f16×f32 点积与矩阵-向量，寄存器内 `xvfcvtl_s_h`/`xvfcvth_s_h` 转换 | **已落地**（§2.11 契约、§5.13 用法）：转换穷举验证、4 条累加链，`gemv 1024×2048` 实测 10.5 GB/s（llama.cpp 同量级 6.8 GB/s） |
| `softmax_rows` | 行内 max→exp→sum→归一，支持 `scale` 与可选加性 mask | **已落地**（§2.6 契约、§5.9 用法） |
| `rms_norm`（+权重） | 每层两次，行归约，与 softmax 共用 | **已落地**（§2.7 契约、§5.10 用法） |
| `silu` / `gelu`（quick/erf） | FFN 激活，逐元素，与 softmax 共用 exp 近似 | **已落地**（§2.8/§2.9 契约、§5.11 用法）：`silu`、`gelu_quick`、`gelu_erf` 三个入口 |
| `rope`（NeoX / GPT-J 两种 mode） | f32，支持 `n_dims`/`freq_base`，可逐位对照 | **已落地**（§2.10 契约、§5.12 用法）：`NeoX` 走 LASX、`GptJ` 走标量（本机 shuffle 只在 128 位 lane 内，见 §2.10 第 4 条） |

**N2（GGML 格式互操作，按需）**：`quantize_rows_q8_0`（激活量化：Q8_0 块 + f16 scale）、
`dot_q4_0_q8_0`/`gemv_q4_0`（llama.cpp 的 Q4_0 块布局 18 B/32，与现有 `dot_q4` 布局不同）、
`dot_q6_K_q8_K`（对应 1.8× 那一档）。N2 的价值是"与 llama.cpp/GGUF 生态互通并可交叉
验证"；若只给自己的推理代码用，N1 就够，且每个算子都能用现有 `api`/`parallel` 包装。

**N2 的唯一具体消费者（loong-llm 的 8 处 `q8_matmul`）已不再作为下游目标**，所以这一批维持
"未做、按需"：没有调用点就没有验收对象，硬做等于自己给自己造需求。真要启动，触发条件是
"要在龙芯上跑 GGUF 权重并与 llama.cpp 交叉验证"（见 `docs/dev.md` §19.13 的 G7 记录）。

**N3（int8 推理，消费者明确）—— 2026-10-02 立项，工作区已落地**

消费者：一个 0.2B 量级的**判别式决策模型**（cross-encoder，非生成式，无 KV cache，
序列 ≤192 token，端侧 CPU）。形态、实测依据与批次划分来源见 `docs/dev.md` §21。
与 N1/N2 的关系：N1 是 f32/f16 且已落地；N2 是 **GGML 块格式互操作**（面向与 llama.cpp
交叉验证），N3 是**自用 int8 推理路径**，两者格式与消费者都不同，互不替代。

| 算子 | 说明 | 状态 |
|---|---|---|
| `lasx_amax` / `lasx_absmax_rows` | 最大绝对值（逐张量 / 逐行） | **已落地**（§2.12 契约、§5.14 用法） |
| `lasx_quantize_i8_per_tensor` | f32 → int8 + scale，对称（`qmax=127`） | **已落地**（§2.12） |
| `lasx_quantize_i8_per_row`（= per-token / per-channel） | 逐行各自求 scale：激活 `[token, hidden]` 是 per-token，权重 `[out, in]` 是 per-output-channel——**同一个内核**，`api` 层按意图分成 `quantize_i8_per_token` / `quantize_i8_per_channel` 两个名字 | **已落地**（§2.12；依据：per-tensor 使成对准确率 96.1% → 93.4%，per-token/per-channel 无损，`docs/dev.md` §21.3） |
| `lasx_dequantize_i8` / `lasx_dequantize_i8_rows` | int8 + scale → f32（逐张量 / 逐行） | **已落地**（§2.12） |
| `lasx_gemv_i8` | 权重 int8 × 激活 int8 → i32 精确累加；出口 `(acc as f32)·(sw[o]·sx)`——**两个 scale 先乘成一个**（权重 per-channel × 激活 per-token） | **已落地**（§2.13 契约、§5.14 用法） |
| 整数 / 定点 softmax | 多项式或查表近似 `exp` | **不做（实测）**：组合路径（反量化 → `softmax_rows` → 再量化）整层只占 int8 GEMM 的 **~2%**（`docs/dev.md` §7.9 末表）⇒ 省下的上限就是 2%，不值得引入定点 `exp` 的精度风险 |
| int8 `rms_norm` / 激活 | 归一化与激活的 int8 版本 | **不做（实测）**：`192×768` 的"反量化 + RMSNorm + 再量化"= 367.3 µs，占该层 GEMM 的 ~0.3%（同表）；**组合路径就是答案**（`dequantize_i8_rows` → `api::rms_norm` → `quantize_i8_per_token`），真要抠先做 `_into` 省分配 |
| `parallel::gemv_i8`（行切分） | 多核 GEMV | **不做**：按 `m` 行切分对"单 token"（`m = 1`）零并行度；`m` 大的档位 f32 GEMM 已赢（`docs/dev.md` §21.8） |
| `parallel::gemv_i8_k`（**k 切分**） | 多核 GEMV，`m = 1` 也能用 | **已落地**（`src/parallel.rs`）：把"块"当"行"交给现有派活接口，整数截断同态 ⇒ 与单线程**逐位一致**（测试守着）。**实测 12 线程 5.4–6.9×**（同表）。门槛：每块 ≥1024 元素且 ≥4 块才并行（更小是负收益：派活固定开销 > 内核） |
| `matmul_i8` | 批量 int8 矩阵乘（prefill：权重读一遍、token 复用） | **已落地**（§2.14 契约、§5.14 用法、`docs/dev.md` §7.9 的实测行），**但实测否掉了它的立项理由**（其 §21.8）：对"逐 token 跑 GEMV"是 **1.00×**、比 f32 `lasx_matmul`(n=192) **慢 2.5–2.7×** ⇒ 目标形态下应选 f32 GEMM；int8 只在"权重装不进缓存、必须流 DRAM"时占优（~4× 流量） |

**可复用**：`lasx_dot_i8` 的"i16 乘积 → i32 累加 → 每 1024 字节落 i64"结构（`gemv_i8` 直接复用
它逐行求点积，一致性由此而来）；`lasx_dot_q4` 的"组内整数点积 + 组 scale 乘"分层与 `xvhaddw`
水平加宽归约；`lasx_matmul` 的打包 + k 分块 + 列块骨架（prefill 式矩阵乘的起手式）；对齐分配。

**验收**（沿用 §2.5 与 `docs/dev.md` §6.4 的既有纪律）——已落地的部分逐条对上：

1. 位精确：向量路径与标量参考对同一输入**逐位一致**（`to_bits()` 比对）——`ops::quant_i8` 的
   7 个测试覆盖 `n` 取 `0,1,2,3,7,8,9,15,16,17,31,33,64,127,128,1000`、NaN 混入与边界样本；
   `gemv_i8` 与精确 i64 参考逐位一致（3 个测试）。**LSX 不适用**（N3 沿用 LASX-only 取舍，见 §3.2）；
2. 量化误差可解析界定：实测最坏值 ≤ `scale/2`（999 个随机样本）；
3. **量化实现的正确性单独断言**——满量程数据的 `max|q| ≥ 120`、边界样本打到 ±127、反量化逐位
   等于 `(q as f32)·scale`（`scale` 的整数倍）。这一条来自一次真实踩坑：漏除 `qmax` 会使权重
   退化为三值，而"指标看起来合理"不会暴露它（`docs/dev.md` §21.4 末尾）；
4. 性能结论按进程级交替 A/B、≥3 轮、丢弃首轮、取中位数：**已做**（3 轮轮间差 ≤1%，表见
   `docs/dev.md` §7）；其中**端到端 10–20 ms 的立项目标被实测推翻**（只算了权重流量、漏了序列里每个
   token 都要过一遍权重），目标已改成分档（`docs/dev.md` §21.6）；
5. C ABI 只追加不修改（**§4.1** 的原始 15 个 `lasx_*` 符号签名与语义永不改动）：N3 的 8 个符号
   + 8 个 `_checked` 变体（§4.3/§4.4 清单）都是**追加**，且测试断言 C 路径与 Rust `api` 路径逐位一致；
6. **给出 per-tensor / per-token / per-channel 三种模式的误差对照**——该选择直接决定 int8 能否使用。

**启动前应确认**（见 `docs/dev.md` §21.6）：`gemv_i8` 在真实形状上的倍数、per-token 量化的
内核开销、端到端延迟——三项目前都只有推算，无实测。


**不建议做**：q4_0/q4_K/q8_0/IQ* 的 vec_dot 重写（llama.cpp 已贴带宽/算力上限）；
`GET_ROWS`/`VIEW`/`PERMUTE` 这类纯搬运算子（带宽决定，SIMD 无事可做）。

复现与局限：

```bash
cd /home/lik/llama.cpp
grep GGML_LASX build/CMakeCache.txt
./build/bin/test-backend-ops perf -o MUL_MAT -b CPU
./build/bin/test-backend-ops perf -o SOFT_MAX,ROPE,ADD -b CPU
```

`test-backend-ops perf` 是**单线程**、合成形状；真实推理有 OpenMP 多线程与 KV cache 访存
模式，绝对数字会不同（相对结论更可靠）。`RMS_NORM`/`SILU`/`MUL`/`SCALE` 在这版里没有
perf 用例，只能按同族算子推断。


## 13. 无损压缩与带宽换算力：结论与路线

> 这一节由原 `docs/compression.md` 并入（该文件已删除），只保留**结论与路线**；
> 判定式与"什么时候压缩能赢"是可复用的结论，完整口径与库侧性能数据见 `docs/dev.md`。

### 13.1 结论

1. **通用无损压缩在本库数据上收益远比直觉小**：单次快照（内核一次调用吃的那 6 个 f64
   数组）最好约 **1.22×**（zlib-6 + 字节平面）/ 1.15×（lzma 原生）；轨迹（64 帧）也只有约
   **1.40×**。不是 2–4×。
2. **放进内核热路径不可行**：本机 zlib-6 解码只有 **0.37 GB/s**，而 RK4 单线程内核消费
   **3.3 GB/s**——差约 9 倍。lz4/zstd 也一样，且这三个库在 loongarch64 上没有 SIMD 路径，
   只会更慢。
3. **"牺牲算力换带宽"的方向本身是对的**，但能换来带宽的不是通用压缩，而是三件已实测验证
   过的事：**直接吃紧凑表示**（`lasx_dot_q4` 用 4-bit nibble，带宽是 f32 的 1/8）；
   **重算代替存储**（融合内核 `lasx_rk4_j2_step_batch` vs 用原语拼装，中间量从不落内存）；
   **分块提高复用**（matmul 分块，B 的流量下降，逐位不变）。

### 13.2 判定式：压缩什么时候能赢

**A. 链路上限场景**（网络、磁盘、跨进程传输）：`收益 = 压缩比 r`，解码只要快过链路速率
即可（一般轻松满足）。此时 1.2–1.4× 有意义：1 Gbps 链路上 1.4× 就是少传 29% 的时间。

**B. 内核内场景**（数据已在内存，要省内存带宽）：原始 `时间 = B/C`（内核消费速率 C），
压缩后 `时间 = B/D`（解码速率 D）；要赢需 `D > C`。**在内核里压缩比 r 几乎不出现，解码
速率必须超过内核自身的消费速率。** 本机关键数字：RK4 单线程 `C ≈ 3.3 GB/s`，12 线程聚合
约 25 GB/s（DRAM 上限），zlib-6 解码 `D ≈ 0.37 GB/s`——出局。B 场景要成立必须自己写 SIMD
解码器（定长位平面打包 + 移位），而且还要面对更致命的问题：**快照压缩比只有 1.2×**，
即便解码无限快也只是 1.2×。

### 13.3 路线

| 阶段 | 内容 | 判据 |
|---|---|---|
| **P0** | 判定式 + 实测 + 反例登记 | 已完成 |
| **P1** | **无损容器**（可选模块，零依赖自研）：面向**存储/传输**的轨迹、检查点、跨进程/跨语言边界；字节平面 + 定长位平面打包 | 比 ≥ 1.3×（目标 1.4×），解码 ≥ 链路速率；**明确不进内核热路径** |
| **P2.1** | matmul 改 8 行×16 列分块 | B 流量下降；**逐位一致**，无精度代价 |
| **P2.2** | RK4 中间量的重算/融合 | 把 6 数组工作集从"读+写 × N"压到更少遍 |
| **P2.3** | 把 `dot_q4`/`dot_i8` 的"直接吃紧凑表示"推广到更多算子 | **有精度代价，需授权**；明确每算子的精度-带宽曲线 |
| **P3** | 反例登记：**不做**"压缩→解压→算"的内核路径 | 除非自研 SIMD 解码 > 3.3 GB/s 且比 > 2×，否则不再评估 |

P2.1 是唯一"零风险、纯收益"的一项（逐位不变），建议作为下一步。

### 13.4 被否掉的做法

通用无损压缩进内核热路径（解码速率差 9 倍，且压缩比只有 1.2–1.4×）；帧间 XOR + 字节平面
（轨迹上反而没有帮助，相关性不够强）；把浮点当均匀字节估收益（随机 f32 也能压到约 0.816，
但只是"少传 18%"，不是数量级）；木桶模型只对"多线程 + DRAM 到顶"成立（1 线程时短板是
算力，受 `xvfdiv_d` 除法吞吐限制（每元素每步 4 次除法 + 4 次开方，见 `docs/dev.md` §13），
此时"牺牲算力换带宽"是反的）。

### 13.5 待定

P1 的无损容器放哪（`lasx_rs` 内的可选模块 / 独立 crate）；P2.3 的精度换带宽授权边界
（哪些算子允许有损、误差上限多少）；是否需要面向 **FFI 边界**的紧凑传输格式（Dart/脚本侧
的数组转换占了传播的绝大部分时间，那是 CPU 转换开销，紧凑表示能同时减小转换量与带宽）。


## 14. 构建、测试与常用命令

工具链：nightly Rust（含 `stdarch_loongarch`）、`edition = 2021`、零依赖、
`[lib] crate-type = ["cdylib", "rlib"]`；`.cargo/config.toml` 设 `rustflags = ["-C",
"target-feature=+lasx"]`（缺它也能编、也位精确，只是慢）。安装：
`rustup toolchain install nightly-loongarch64-unknown-linux-gnu`。

```bash
cargo build --release                 # 构建库（liblasx_rs.so + rlib）
cargo test --release                  # 单元测试 + 文档测试（需 LoongArch 真机）
cargo build --workspace --release     # 连基准 CLI（lasx_bench）一起构建

cargo run -p lasx_bench --release                 # 全部基准套件
cargo run -p lasx_bench --release -- matmul       # 只跑组名含子串的套件
cargo run -p lasx_bench --release -- attitude
cargo run -p lasx_bench --release -- scenario
cargo run -p lasx_bench --release -- align

# 单形状矩阵乘 A/B：<m> <k> <n> <stream|packed|cols|packed64|stream64>
cargo run --release --example matmul_ab -- 256 256 256 packed
cargo run --release --example matmul_ab -- 256 256 256 packed64

# 计划复用对照（① 一次性 API / ② C 复用 / ③ 免打包，见 docs/dev.md §18）
cargo run -p lasx_bench --release -- plan

cargo run --release --example pool_axpy        # 多核示例（自带逐位对照）
cargo run --release --example matmul_pooled

cargo build --release -p lasx_yll              # YouLiLong 原生扩展
cp target/release/liblasx.so yll/
```

基准套件组名（过滤子串）：`dot`、`sum`、`axpy`、`dot_f64`、`dot_i8`、`dot_q4`、`matmul`、
`plan`、`softmax`、`rms_norm`、`activation`、`rope`、`f16`、`int8_gemv`、`int8_quant`、
`attitude`、`large`、`norm3`、`vec3`、`distance2d`、`j2`、`ballistic`、`rk4`、`fma`、
`mt`、`align`、`dispatch`、`scenario`。方法学与全部读数详见 `docs/dev.md`。

**`lasx_yll`（YouLiLong 原生扩展）的覆盖面——已审计**：它暴露的是**精选子集**，
`yll/lib.ylh` + `yll/src/funcs.rs` 共 10 个脚本函数（`dot`、`sum`、`dot_i8`、`matmul`、
`norm3`、`distance2d`、`vec3_add_scaled`、`j2_accel`、`rk4_step`、`propagate`），
每个都走 `ffi::checked` + 参数级校验（`docs/dev.md` §19.12 那套"两层错误"的分工——
本层判数组/形状/常数，库的 `*_checked` 兜结构性错误）。
**N1 批次的 8 个与 N3 批次的 8 个都刻意没有进**——脚本语言的调用面由实际脚本需求驱动，
不是"C 有就必须有"（这也是 §12.3 里"没有调用点就没有验收对象"的同一条判断）。
要新增一个脚本函数：`funcs.rs` 写包装（参数校验 + `*_checked` + 返回值构造）、
`lib.ylh` 补声明、`cargo build --release -p lasx_yll` 后拷 `liblasx.so`。

```bash
# 导出符号自检：期望行数 = docs/c-abi.md 记的导出符号计数（本文不再复述该数）
nm -D --defined-only target/release/liblasx_rs.so \
  | awk '$2=="T" && $3 ~ /^lasx_/ {print $3}' | sort
```


## 15. Caveats 与限制

1. **降级覆盖不全**：历史 7 个 LASX-only 内核（`matmul`、`matmul_f64`、`axpy`、`sum`、`dot_f64`、
   `dot_i8`、`dot_q4`）在 LSX-only CPU 上会 SIGILL，不会自动降级；N1/N3 与 2026-10-15 追加的
   `matmul_scaled`、`layer_norm`/`gather_rows*` 同样按 §3.2 的分类（前者 LASX-only，后两者不涉及
   SIMD），完整分类见 §3.2。**C 侧的自我保护（先问 `lasx_has_lasx()`）见 [`docs/c-abi.md`](c-abi.md)
   第 6 节。**
2. **仅标量降级**：`ballistic_step`、`batch_distance2d`、`rk4_j2_step_batch` 的非 LASX
   分支是全标量循环，没有 LSX 128 位向量路径。
3. **两处不逐位一致的例外**：`ballistic_step` 向量/标量的 `|v|²` 结合序不同（物理一致，有绝对
   误差测试守护）；`axpy` 向量/标量尾至多差 1 ulp（FMA vs 分离乘加，无逐位守护）。
4. **`lasx_free` 的 `n` 必须与分配时一致**：契约与两种 no-op 见 [`docs/c-abi.md`](c-abi.md) 第 4 节
   （"layout 对得上"由构造成立；`n` 传错——非 0 且与分配时不同——会破坏堆，**只有约定**，违反是 UB）。
5. **原始符号长度参数无负数/零防护**：C 侧安全模型见 [`docs/c-abi.md`](c-abi.md) 第 5 节
   （裸 = 零校验、误用即 UB；需要错误通道就用 `_checked`）。
6. **f32 矩阵乘无 f64 落盘**：`lasx_matmul` 在 f32 累加器内直接累积，长 k 维误差大于点积；
   高精度用 `lasx_matmul_f64`。
7. **`dot_q4` 的 scale 长度约定**：`sa`/`sb` 长度必须 ≥ `ceil(n_bytes/32)`，按 32 字节组对齐；
   非 32 倍数时最后不足组按 `j/32` 索引（原型见 [`docs/c-abi.md`](c-abi.md) 第 3.1 节）。
8. **`dot_i8` 结果截断**：内部 i64 累加精确，返回 `as i32` 会回绕；业务上 n 应远小于约 13.3 万。
9. **`dot` 的小规模路径**：n < 24 用 f32 单精度累加，与 n ≥ 24 的 f64 分块公式不同，跨边界不
   保证逐位连续。
10. **检测缓存与钩子作用域**：LASX 检测依赖 `cpucfg` + CFG2.bit7，结果进程级缓存一次、之后不可
    更新（虚拟化/模拟器中 cpucfg 不可信则检测不可信）；`FORCE_LSX` 线程级、默认 `false`，测试
    必须自行复位；池化路径的 worker 不受调用线程钩子影响。
11. **对齐与内存布局**：向量加载/存储用偏移 0，LoongArch 允许非对齐访问，源码未显式声明对齐
    要求；SOA/数组布局假设连续无填充，跨语言调用方必须保证元素类型与步长严格匹配（f32=4 字节、
    f64=8 字节）。C 侧的对齐与别名规则见 [`docs/c-abi.md`](c-abi.md) 第 4 节。


## 16. 待确认、已修正与文档约定

本文由旧的四份文档（manual / nn-ops / compression / perf-report）合并重写，凡是旧描述与
**当前源码**不符的，一律以源码为准。§16.1–§16.3 是合并时发现的遗留问题，§16.4 是本文档的
写作约定。

### 16.1 已修复

| 问题 | 处理 |
|---|---|
| `lasx_quat_to_dcm_batch_checked` 的 9 个输出数组直接 `from_raw_parts_mut`，没走 `checked_slice_mut`，`NULL + n > 0` 会 UB | 已改为逐个 `checked_slice_mut` 校验，并补测试 `test_quat_to_dcm_null_output_is_reported` |
| 旧 `README.md` 写"逐位确定：向量化与标量结果一致"，口径过宽 | 已按本文 §2.1 的精确口径改写根 README（并列出 `ballistic_step` 不保证逐位一致、`axpy` 允许 1 ulp 两处例外） |

### 16.2 旧文档的过时描述（已随旧文档删除而消解，清单保留备查）

| 旧描述 | 当前源码（正文以此为准） |
|---|---|
| manual §2.2/§9.6：`lasx_matmul` 先转置 B | 不转置、列方向向量化（`src/ops/matmul.rs` 模块注释、`src/ffi/matmul.rs`） |
| manual §9.6：有 `n×k` 的 B 转置临时分配 | 无 B 转置；打包路径只有线程本地复用缓冲（按需增长、跨调用保留） |
| manual §2.1/§2.4：`dot`/`sum` 64 元素一块、8 次 FMA | 256 元素一块、4 条独立累加链；`dot` 另有 n < 24 的纯标量 f32 快路径 |
| manual §2.7/§3.1：`dot_i8` 每 32 字节落盘、逐 i16 转 i64 | i16 乘 → i32 累加器（上界 128×127²≈2.1 M），每 1024 字节合并落盘转 i64 |
| manual §2.6：`lasx_alloc` 用 `Vec::with_capacity` + `mem::forget` | `std::alloc::alloc(Layout::from_size_align(size, 32))`，保证 32 字节对齐 |
| manual §1.2/§9.10：`OnceLock` 缓存 `HwCaps` | `AtomicU8`（`src/arch/mod.rs`） |

这些差异都不影响数值语义（分块大小、缓冲实现、缓存机制都是内部结构），正文按源码写。

### 16.3 仍需外部信息

- ggml / llama.cpp 的算子统计（102 个算子、调用次数分布）来自仓库外的 `/home/lik/llama.cpp`，
  无法在仓库内复核；本文只把它们当"缺口依据"引用，不当作可复现的测量。
- 旧文档里的源码行号引用对应已删除的扁平版 `src/lib.rs`；本文一律用当前文件路径。

### 16.4 文档约定：数字来源、测量口径、引用格式

本节规定本文档与 `docs/dev.md` 的分工方式。约定是**状态**而不是过程：新增算子或复测之后，
把读数与契约按下面的表落位即可，不需要知道任何一次改动是在哪一轮做的。

**数字的唯一来源**

| 内容 | 唯一权威位置 |
|---|---|
| 契约（语义、结合次序、op 序列、精度包络、**谁守**、边界行为） | 本文 §2.5–§2.11（总精度一览在 §2.3） |
| 用法（签名、示例、约束） | 本文 §5.1–§5.13、§6.x、§7.x |
| **全部性能数据**（单线程内核表、大集占比、真实场景、多线程扩展） | `docs/dev.md` **§7** |
| 编译期常量与阈值的 A/B 对照 | `docs/dev.md` 的对应设计小节 |
| 设计推导与被否掉的方案 | `docs/dev.md` 的对应设计小节 |
| 与 llama.cpp 的缺口对照 | 本文 §12.2（只保留对照结论，本库读数一行） |

本文正文不放性能表，只给"一行结论 + 指针"。理由是**可审计性**：同一个数字存在两处，就必然
有一处先过期，而读者无法判断哪一处是新的。

**测量口径**（`docs/dev.md` §7 的每一张表都按此产生）

- 内核表**绑核**测（`taskset -c 4`），线程表**不绑核**测，两者分两次运行。绑核会把 12 个
  worker 挤到同一个物理核上，测到的是上下文切换而非池的扩展性。
- 每个测点：进程级交替 A/B、≥3 轮、丢弃首轮、取中位数（口径见 `docs/dev.md` §6.4）。
- 套件源码不写死读数：`micro.rs`、`scenario.rs` 注释中的数字必须由当次运行算出。
- 计数类事实只在一处维护：导出符号与降级覆盖见本文 §3.2 / §4，测试数量见 `docs/dev.md` §4。

**引用格式**

- 跨文档引用一律写成 `docs/xxx.md §N`（带文件名）；本文内部引用写成 `§N`，两者都可机械校验。
- 引用写到最小可解析粒度（如 `§2.10`、`docs/dev.md` §20.7）。`§` 编号是稳定标识，重构时保持
  编号不变，标题文字可以改。

## 17. 作为共享库 / C 项目使用（构建、安装、SONAME 与 ABI 规则）

### 17.1 构建产物

**已搬移**：一次构建的三种产物（`cdylib` / `staticlib` / `rlib`）与各自用途见
[`docs/c-abi.md`](c-abi.md) 第 2.1 节。

### 17.2 安装

**已搬移**：`scripts/install.sh --prefix=…` 的用法、装到 `<prefix>` 的六个文件、"失败即退出 +
装前校验（含 SONAME 核对）"见 [`docs/c-abi.md`](c-abi.md) 第 2.1 节。

### 17.3 链接与运行

**已搬移**：四种链接方式（动态 + rpath / 动态 + `LD_LIBRARY_PATH` / 静态 `.a` / pkg-config 驱动）、
最小程序与 `examples/c/` 的实测输出见 [`docs/c-abi.md`](c-abi.md) 第 2.2–2.4 节。

### 17.4 SONAME 与文件名布局

**已搬移**：SONAME `liblasx_rs.so.0` 的来源（`build.rs`）、构建产物名 vs 安装后的三个名字、
`readelf` / `nm` 自检见 [`docs/c-abi.md`](c-abi.md) 第 9 节与第 2.1 节。

### 17.5 ABI 兼容规则（什么时候可以动 major）

**已搬移**：只追加不改、major 变更条件、要同改的三处、0.x 的语义与"不同 major 并存"——
见 [`docs/c-abi.md`](c-abi.md) 第 9 节（那是 C ABI 承诺的唯一权威）。

### 17.6 `_checked` 变体的 `status` 语义

**已搬移**：调用约定、失败时的行为（只写状态、返回中性值、不触碰输出、`status` 可为 NULL）、
0..6 状态码表见 [`docs/c-abi.md`](c-abi.md) 第 3.7、3.8、5 节。

---

## 18. C 调用者心智模型

**已搬移**：本节整节搬到 [`docs/c-abi.md`](c-abi.md)——C 调用者的心智模型现在是那份文档的第
1–9 节（读者与范围、快速开始、符号参考、内存、安全模型、能力与降级、数值契约、线程模型、
ABI 稳定性策略）。下面保留小节标题与对应落点，供既有引用使用。

### 18.1 三层与它们的关系

→ [`docs/c-abi.md`](c-abi.md) 第 1 节（范围）与第 9 节（稳定性：C ABI 冻结/只追加；
Rust `api` 侧见本文 §8）。

### 18.2 裸 vs `_checked`：默认用 `_checked`

→ [`docs/c-abi.md`](c-abi.md) 第 5 节（对照表 + "该用哪个"的判据）。

### 18.3 LASX-only vs 可降级：先问再调

→ [`docs/c-abi.md`](c-abi.md) 第 6 节（正/误两段对照代码、LASX-only 清单、机型对照）。

### 18.4 位精确承诺与两处例外

→ [`docs/c-abi.md`](c-abi.md) 第 7 节（以本文 §2.5 的表述为准；另见本文 §2.1/§2.5）。

### 18.5 内存归属：`lasx_alloc` 与 `lasx_free` 必须配对

→ [`docs/c-abi.md`](c-abi.md) 第 4 节（配对、同一个 `n`、"`free()` 不是 ABI 契约"、两种 no-op）。

### 18.6 线程模型

→ [`docs/c-abi.md`](c-abi.md) 第 8 节（无逐调用全局状态；C ABI 不导出线程池；
`lasx_force_lsx_thread` 是 Rust API、线程局部、仅测试用）。

### 18.7 错误模型

→ [`docs/c-abi.md`](c-abi.md) 第 3.8 与第 5 节（状态码、`lasx_status_message`、失败不触碰输出）。

### 18.8 决策表："想做 X ⇒ 用 Y"

→ [`docs/c-abi.md`](c-abi.md) 第 1–9 节：决策表的每条在那里都有对应小节（要稳用 `_checked`、
热循环复用缓冲、先问能力、版本串与状态串的用法）。
