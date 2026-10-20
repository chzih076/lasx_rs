# lasx_rs 算子与用法手册

本手册面向 `lasx_rs` 的**调用方**：逐个算子的语义、数值契约（逐位承诺 / 容差与实测界 / 边界行为 /
错误面 / 能力位选择）与调用方式。C ABI 权威是 [`docs/c-abi.md`](c-abi.md)；性能数字唯一权威是
`docs/dev.md` 的性能章节（本文只留"一行结论 + 指针"）。

| 项 | 内容 |
|---|---|
| 适用代码版本 | `master`；导出符号清单/计数/逐符号原型见 [`docs/c-abi.md`](c-abi.md) |
| 库形态 | LoongArch64 LASX 256 位 / LSX 128 位批量数值内核；零依赖（std + `#![feature(stdarch_loongarch)]`，nightly）；产物 `cdylib + rlib` |
| 本文写什么 | 算子契约与用法；**不写**性能数据（`docs/dev.md`）、平台勘误（[`docs/platform.md`](platform.md)）、C ABI 细节（[`docs/c-abi.md`](c-abi.md)） |
| 目标读者 | C / Rust / Dart 调用方与数值契约核对者；写作约定见 §15.1 |

阅读路径：只调用库 → §1 + 总表 + §5/§6/§7；数值行为 → §2.1 → §2.5 → §2.6–§2.17；多核/形状/计划 → §8、
§9；NN 能力边界 / 压缩路线 → §12 / §13。

## 0. 算子 → 入口 / 精度类 总表

**逐位** = 向量路径与标量尾/模拟逐位一致（同式同结合序）；**容差** = 与独立参考有实测界（§2.3 与各契约
节）。C / Rust 入口走**同一段内核**。

| 算子族 | 算子（C 入口名去 `lasx_` 前缀） | Rust 入口 | 精度类 |
|---|---|---|---|
| 归约 | `dot`、`sum`、`dot_f64`、`axpy` | 同名 `api::*` | 容差（`axpy` ≤1 ulp） |
| 矩阵乘 | `matmul`、`matmul_f64`、`matmul_scaled`、计划句柄 7 个 | `api::matmul*`、`plan::MatmulPlan`、`shape::*` | **逐位** |
| int8 点积/量化 | `dot_i8`、`dot_q4`、`amax`、`absmax_rows`、`quantize_i8_*`、`dequantize_i8*`、`gemv_i8`、`matmul_i8` | 同名 `api::*` | 整数**精确**（mod 2³²）；量化 ≤ `scale/2` |
| f16 | `dot_f16`、`gemv_f16` | 同名 `api::*`、`parallel::gemv_f16` | **逐位**（转换精确、4 链固定次序） |
| norm/softmax | `softmax_rows`、`rms_norm`、`layer_norm` | 同名 `api::*`（含 `_into`） | **逐位** |
| 激活 | `silu`、`gelu_quick`、`gelu_erf` | 同名 `api::*`（含 `_into`） | **逐位**；对 f64 参考容差 |
| rope | `rope` | `api::{rope,rope_into,rope_tables,rope_at}` | **逐位**（同一张表） |
| 索引读 | `gather_rows`、`gather_rows_i8` | 同名 `api::*`（含 `_into`） | **逐位**（无算术） |
| 几何 | `norm3_batch`、`vec3_add_scaled_batch`、`batch_distance2d`、`cross3_batch`、`unitize3_batch`、`mat3_mul_vec3_batch` | 同名 `api::*`（含 `_into`） | **逐位**（分块内） |
| 物理 | `ballistic_step`、`j2_accel_batch`、`rk4_j2_step_batch` | 同名 `api::*` | `j2/rk4` **逐位**；`ballistic` 容差 |
| 姿态 | `quat_normalize_batch`、`quat_mul_batch`、`quat_rotate_batch`、`quat_to_dcm_batch` | `api::quat_*` | **逐位** |
| 复数 | `cmatmul_f32`、`cmatmul_f64`（+`_checked`） | `ops::cmatmul::*`、`cmatmul::{CMat,CMatBuf}` | `Direct` **逐位**；`Karatsuba` 容差（默认） |
| Laplace / 逆变换 / 导数 | 无 C 符号（Rust 模块） | `laplace::*`、`ilaplace` | 见 §9.05–§9.10 |
| 并行 / 内存 | —（Rust） | `pool::*`、`parallel::*`、`aligned::AlignedVec` | 与单线程**逐位**一致 |

## 1. 分层与调用方式

| 层 | 给谁用 | 形态 | 校验 |
|---|---|---|---|
| `lasx_rs::ffi`（`extern "C"`） | C / Dart | 裸指针 + `int` 长度 | 裸符号零校验（误用即 UB）；`_checked` 带 `int *status` |
| `lasx_rs::api` | Rust | `&[T]`/`&mut [T]`，返回 `Result<_, api::Error>`，输出 `AlignedVec` | 形状、溢出、物理常数 |
| `lasx_rs::view` + `lasx_rs::plan` | Rust，矩阵类 | `MatRef`/`MatMut` 视图；`MatmulPlan` 打包一次复用 | 构造时校验一次 |
| `lasx_rs::pool` + `lasx_rs::parallel` | Rust，多核 | 常驻池 + 闭包 / 固定切分策略 | 派活取 `&self`；形状不符 panic |

四层走**同一段内核**：`api`/`view`/`plan`/`pool` 只是校验/调度薄包装，结果与直接调 `lasx_*` **逐位一
致**（各有对照测试）。Rust 普通场景用 `api`；矩阵参数易记错用 `view`；同一 `B` 反复乘用 `MatmulPlan`；
已自校验形状可直接调 crate 根的 `lasx_rs::lasx_dot(..)`；C/Dart 用 `lasx_*`，要把错误上抛用 `_checked`；
多核自行切子区间（库不导出线程池）。

## 2. 数值契约与逐位确定性

### 2.1 "逐位一致"的精确含义

两层，不可混同：① **同一次内核实现内部**：向量块与标量尾**同式、同结合序** ⇒ 任一分块的向量部分与标
量尾逐位一致（逐元素批量内核成立）；② **跨实现对照**：与"不同公式的独立标量参考"比，承诺是**相对误差
<1e-9**（不是逐位相等）。**逐位承诺的例外（仅此两处，均有测试）**：① `lasx_ballistic_step` 向量路径与
标量 `euler_step` **非逐位一致**——`|v|²` 结合序不同（向量右结合、标量左结合），物理量已对齐、有绝对
误差测试；② `lasx_axpy` 向量路径是单条 FMA（一次舍入）、标量尾 `mul`+`add`（两次舍入），至多差 **1
ulp**、无逐位守护。补充：`lasx_dot` 在 `n < 24` 走纯标量 f32 累加、`n ≥ 24` 走 f64 分块累加 ⇒ 跨该边界
不保证逐位连续（两条路径设计上同式同结合序，可用 `lasx_force_lsx_thread` 验证）。

### 2.2 结合次序（硬约定）

| 量 | 结合序 |
|---|---|
| `norm3`/`unitize3`/`j2_accel`/`rk4_j2_step` 的 `\|r\|²` | `x² + (y² + z²)`（右结合） |
| `ballistic_step` 向量路径的 `\|v\|²` | `vx² + (vy² + vz²)`（右结合；标量参考左结合） |
| 四元数范数 / 乘法每分量 | `((w² + x²) + y²) + z²`（左结合）/ 每分量左结合 |
| `mat3_mul_vec3` / `quat_rotate` 每行 | `((m0·x + m1·y) + m2·z)`（左结合） |
| 叉积 / `j2_accel` 末项 / `vec3_add_scaled` | 先乘后减；FMA `fma(k·x, 5·zr2 − 1, vcen·x)`；`fma(s, b, a)` |

### 2.3 精度一览

| 内核类别 | 累加/近似方式 | 误差量级 |
|---|---|---|
| f32 点积/求和 | 每 256 元素落盘升 f64，最终一次 `as f32` | 相对 f64 参考 <1e-5 |
| f32 矩阵乘 | 每个输出元素沿 k 的**单个 f32 累加器**（无 f64 落盘） | 相对 f64 参考 <1e-4 |
| f64 内核 | 全程 IEEE 双精度，FMA 单次舍入 | <1e-12（matmul_f64）/<1e-9（批量物理） |
| int8 点积 / Q4 | i16 乘 → i32 → 每 1024 B 落 i64 / 组内整数精确 + 组 scale | 精确 mod 2³² / scale 精度约 2⁻²⁴ |
| `softmax_rows` / `rms_norm` | `exp` 多项式+magic 数 / `fma(x,x,acc)` 8 lane 归约 | 与 f64 参考 <1e-6 / <1e-5 |
| `silu`/`gelu_quick` / `gelu_erf` | 共用 `exp`、直接除法 / A&S 7.1.26（**不是真 erf**） | 绝对 ≤7.6e-6 / 端到端 <1e-6 |
| `rope` / `dot_f16`/`gemv_f16` | 每对两次舍入 / f16→f32 **精确**转换 + 4 链 | 绝对 <1e-6、保长度 <1e-6 / ≈4.9e-4（f16 量化） |

`exp` 端到端相对误差随 `|x|` 增长（≈`|x|·4e-8 + 2e-7`），上表取各算子实际用到的区间；误差都来自实测。

### 2.4 退化输入与边界约定

零向量单位化（`unitize3_batch`）：`|v| == 0` 输出 `(0,0,0)`（不是 `0·∞ = NaN`）。退化四元数：
`|q| < 1e-15` 输出单位四元数 `(1,0,0,0)`；单位化用 `q·(1/|q|)`（一次除法再乘）。`api::sum` 对空切片返
回 `0.0`，任何切片都有定义 ⇒ 不返回 `Result`。长度为 0：`_checked` 允许空指针；`lasx_alloc(0)` 返回
"对齐但不可解引用"的非空悬垂指针。批量几何/姿态内核**允许输出与输入别名**（向量路径先取完该轮输入再
写回）。

### 2.5 契约（冻结：改内核/加路径/上公式 DSL 均不可违反，每条有测试）

**1）累加顺序。** 每个输出元素是沿 `k` 的**单个累加器、k 升序、每步一条 FMA**：
`c[i][j] = fma(a[i][k-1], b[k-1][j], fma(… fma(a[i][0], b[0][j], 0) …))`。`alpha`/`beta`（若启用）只能
在累加**结束之后**按固定次序施加。任何"改写结合次序"的优化（拆并行部分和再相加、用 `beta·y` 给累加器
播种、跨 lane 水平归约）**违反契约**。**2）路径无关性。** LASX / LSX / 标量尾 / 打包面板 / k 分块 / 并
行切块 / `MatmulPlan` / `shape` 层对同一输入给出**逐位相同**结果（`to_bits()` 比对）。前提：中间结果以
目标精度存储（无扩展精度寄存器；龙芯 LSX/LASX 是标准 IEEE 754）、k 分块只是同一部分和分段写回 C（存
储/读取精确）。**3）融合语义（已启用）。** 支持 `C = alpha·A·B + beta·C`（原地）：`alpha` 默认 1.0、
`beta` 默认 0.0（覆盖输出，对齐 BLAS gemm）；施加次序固定 `C = fma(beta, C_old, alpha·acc)`（两次舍
入），**不是**用 `beta·C_old` 播种累加器；`acc` 与不带缩放的 `matmul` 是**同一个**沿 k 的 f32 累加 ⇒ 融
合与"算完再整体乘"**逐位一致**；**`beta == 0` 时 C 一个字节都不读**（CBLAS 语义；NaN 探针测试守）；
`beta != 0 且 k > 384`（打包路径 k 分块会覆盖 `C_old`）退回与二遍法同构的实现。性能承诺只有一条：
`beta == 0` 省一遍 C 读写（实测宽薄档 −10.3%/−23.7%、小 k −6.5%、方阵 ≈0），**不改写 GEMM 内部结
构**。入口：Rust `api::matmul_scaled_into`、C `lasx_matmul_scaled`（+`_checked`）、公式 DSL 标量前缀。
**只对 f32 提供**；f64 写 DSL 标量前缀是编译错误（要缩放请先 `mul` 再整体乘）。**4）逐位一致的例外**：
见 §2.1 的两处。

### 2.6 `lasx_softmax_rows`：行内 softmax 的数值契约

**1）语义。** `rows × cols` 行主序：`out = exp(clamp_lo(scale·x + mask − 行内 max)) · (1/Σ)`。`mask` 可
选、同形状、**加性**（attention 的 `+ mask`；C 侧空指针 = 无 mask）；**先减行内 max 再 `exp`**（否则溢
出）；归一化是 `e · (1/Σ)`（**一次除法 + 乘法**，不是逐元素除法）⇒ 参考实现必须同样写 `e * (1.0/sum)`
才逐位一致。**2）`exp` 的定义即契约。** `exp(x) = exp2 的 6 次多项式(f) · 2ⁿ`：
`t = x·LOG2E; n = (t + 12582912.0) − 12582912.0; f = t − n`（`|f| ≤ 0.5`）；
`p = Horner(f, (ln2)^k/k!, k ≤ 6)`（全程 FMA）；`kk = clamp(trunc(n) + 127, 0, 254); exp(x) = p ·
bitcast_f32(kk << 23)`。端到端相对误差（含 `x·LOG2E` 那次 f32 乘法）：`|x|≤1` ≤2.1e-7、`|x|≤8` ≤6.2e-7、
`|x|≤20` ≤1.1e-6、`|x|≤87` ≤3.9e-6（模型 ≈`|x|·4e-8 + 2e-7`）；**下溢恰好落 0**：
`x < −126.5/log2e ≈ −87.6831` ⇒ `kk=0` ⇒ `+0.0`（阈值由取整边界决定，不是数学上的 −88.03）⇒
`mask = −inf` 给出**精确 0 权重**；输入在 `exp` 入口夹到 `[−104,+88]`；`exp` 是**全函数**（下溢精确 0、
上溢 `exp(88)≈1.65e38` 有限可复现），调用方**不需要**再夹取。标量与向量共用该序列；行内求和用"8 个
per-lane 累加器 + 固定次序两两归约"，列尾 `−1e30` 填充后走同一 lane-wise 路径 ⇒ 向量与标量模拟**逐位一
致**。**3）不在契约内。** 输入或 `mask` 含 `+inf`：`inf − inf = NaN`（定义域问题，算子不处理）。**4）谁
守。** 逐位一致（标量模拟）、显式补齐 == 列尾路径、f64 参考精度、退化输入（行内全相等 / `scale=0` /
`−inf` mask / 极负值不出 NaN）。

### 2.7 `lasx_rms_norm`

与 §2.6 共用"位精确骨架"：`s = Σ_j x[j]²`（`fma(x,x,s)`，一次舍入）；`ms = s / cols; inv = 1 /
sqrt(ms + eps)`（各一次除法/开方，标量域）；`out = (x · inv) · w`（次序固定：先 inv 再权重）。**列尾补
`0.0`**（`0² = 0`，精确无贡献，无需夹取）；**行级数学在标量域算完再广播** ⇒ 天然位精确。**开方/倒数用
精确指令**，不用估算指令 + 牛顿迭代（本机实测估算更慢）；`w` 可选（C 侧空指针）、长度 `cols`、每行复
用。**不在契约内**：`eps = 0` 且整行全零 ⇒ `inv=+inf` ⇒ `0·inf=NaN`（定义域问题）；`api` 与 `_checked`
都要求 `eps` 有限且 `> 0`。**谁守**：逐位一致（标量模拟，覆盖 `cols` 1–17 与多种形状）、`fma` 契约、
f64 参考 <1e-5、`eps > 0`（`api`：`NotPositive`；`_checked`：`NonPositiveConstant`）。

### 2.8 `lasx_silu` / `lasx_gelu_quick`（逐元素激活）

逐元素、无归约 ⇒ 位精确是构造性的：只要标量与向量执行同一串 op 即逐位相同；需要固定的是公式的**等价
形式**：`t = c·x`（silu: c = 1.0；gelu_quick: c = 1.702，ggml GELU_QUICK 同式）；`den = 1 + exp(−t)`
（`exp` 自带夹取 `[−104,+88]`，§2.6）；`y = x / den`（**直接除法：单次舍入**）。**关键一条：写直接除法
`x / den`，不是 `x · (1/den)`**（数学等价、末位可能不同；参考实现照抄直接除法才逐位一致）。与 softmax 相
反：softmax 的分母整行共享，一次倒数换 N 次除法划算；此处每个元素各自一次、没有共享机会。**谁守**：逐
位一致（标量模拟单测，`n` 1..=17 + 64/65/127/129）、原地 == 异地逐位、`n`/空指针校验（`n=0` 合法）、
f64 参考精度。**不在契约内**：`NaN` 进 `NaN` 出；`+inf` 给出 `+inf`（正确极限）；`−inf` 给出 `−inf`
（**不是**数学极限 0，因 `exp` 上界夹取使分母停有限值）⇒ 不要传 `±inf`。实测精度（对 f64 参考，扫描
`[−150,150]` 步长 5e-4）：`silu` 最差绝对 7.6e-6（`|x|≤20` 1.9e-6、相对 1.7e-7）；`gelu_quick` 7.6e-6
（`|x|≤20` 1.2e-6、相对 1.6e-7）；绝对误差主因是 f32 输入量化，不是算子近似误差。

### 2.9 `lasx_gelu_erf`

语义 `y = 0.5·x·(1 + erf(x/√2))`（PyTorch `gelu` 默认定义）。**`erf` 近似（契约核心）**：**A&S
7.1.26**：`p = 0.3275911; t = 1/(1 + p·|z|)`（`z = x/√2`）；
`erf(|z|) ≈ 1 − ((((a5 t + a4)t + a3)t + a2)t + a1)·t·exp(−z²)`（全程 FMA）；
`a = (0.254829592, −0.284496736, 1.421413741, −1.453152027, 1.061405429)`。该式最大绝对误差 **1.5e-7**
（A&S 界）；用 glibc `erf` 复测折算到 gelu 输出 **2.1e-7**（`|x|≈3.08`）⇒ **并非"真 erf"**；要与 ggml
对齐用 §2.8 的 `gelu_quick`。`|x|` 用一条按位 `and 0x7FFFFFFF`（本机 stdarch 无 `xvfabs_s`）；写法
`y = 0.5·x + 0.5·|x|·erf(...)`（不带 `copysign`）；`exp` 复用 §2.6 那一串。**谁守**：逐位一致（标量模
拟，**必须显式写 `mul_add`**，遗漏会在 `x=−3` 暴露 1 ulp）、与独立高精度 f64 `erf` 对照、边界
（`x ≤ −5.6` 精确 0、`x ≥ 6` 位精确恒等 `y = x`、正侧单调）、原地 == 异地。**实测精度**（对高精度 f64
参考，最差绝对）：`|x|≤1` <6e-7、`≤4` <8e-7、`≤16` <1e-6、`≤40` <6e-6。**两个非理想行为（如实）**：①
负侧尾部不单调（`erf(|z|)` 在 `|x|≈5.6` 舍入到 1.0 ⇒ 精确 0，再向右出现 ~`−1.6e-7`；因此测试只对
`x ≥ 0` 断言单调）；② `x = −inf` ⇒ `NaN`（`0.5·inf + 0.5·(−inf)` 未定式，PyTorch 同样），契约只覆盖
有限输入；`x = +inf` ⇒ `+inf`。

### 2.10 `lasx_rope`（旋转位置编码）

`rows × cols` 行主序，每行**前 `n_dims` 列**参与旋转，`[n_dims, cols)` 原样复制。每对：
`y0 = fma(x0, c, −(x1·s))`、`y1 = fma(x1, c, (x0·s))`（两次舍入：`x1·s` 一次，fma 一次）。

| mode | 值 | `i ∈ [0, n_dims/2)` 配 | 对照实现 |
|---|---|---|---|
| `NeoX` | `0` | `(i, i + n_dims/2)` | HF `rotate_half` / Llama / Qwen |
| `GptJ` | `1` | `(2i, 2i+1)` | ggml `ROPE_TYPE_NORMAL` / 原始 GPT-J |

整数值是 C ABI 的一部分。**`cos`/`sin` 表是输入**（`rows × (n_dims/2)`、行主序；第 r 行第 i 列 = `θ_i =
positions[r]·freq_base^(−2i/n_dims)`）：角度不依赖数据 ⇒ 一次前向每层每位置算一次即够；**位精确承诺只落
在旋转上**（`sin/cos` 来自平台 libm，跨 libc 不保证逐位一致）——给定同一张表，向量与标量路径逐位相
同；要与其它实现严格对齐应自行造表；表**与 mode 无关**。**`NeoX` 走 LASX、`GptJ` 走标量**（本机 ISA
实测）：`NeoX` 两半各自连续 ⇒ 8 lanes = 8 个 i、满 SIMD；`GptJ` 需把 16 个连续元素拆偶/奇，本机
`xvpickev.w`/`xvpickod.w` 是 128 位 lane 内操作且需跨 lane 重排 π，单条 lane 内 shuffle 做不到 ⇒ 标量
（llama.cpp 两种 mode 也都是标量）。**谁守**：逐位一致（两 mode × `n_dims/2` 1..=17 + 多种形状、
`cos=1,sin=0` ⇒ 逐位恒等、原地 == 异地在 FFI 层测）；数值（f64 参考绝对 <1e-6、保长度相对 <1e-6）；形
状（`api`：`n_dims` 偶数且 `≤ cols`、表长 `rows×n_dims/2`；`_checked` 同规则 → `BadShape`/
`NullPointer`）。**只靠约定**：C 侧调裸符号时表与 `x` 长度自洽——违反是读写越界（错结果甚至 UB）。

### 2.11 `lasx_dot_f16` / `lasx_gemv_f16`（f16 权重）

`dot_f16(a: &[u16], b: &[f32]) -> f32`（`Σ f16_to_f32(a[i])·b[i]`）；`gemv_f16(a: &[u16], x: &[f32], m,
k) -> [f32]`（`y[r] = dot_f16(a[r*k..(r+1)*k], x)`）。f16 以 **u16 位型**给出（零依赖，不引入 half
crate）。**f16 → f32 精确**（f16 每个值含次正规/±inf 都能被 f32 精确表示）⇒ 该步不引入误差；误差仅来自
权重本身以 f16 存储（相对 ~2⁻¹¹ ≈ 4.9e-4）。**累加次序（契约本身）**：**4 条独立累加链**，每 32 元素一
轮（不足 32 走 16 元素轮、再不足标量尾；尾部延续同一 lane `j % 8`）：`c0 = fma(cvt(a[i..i+8]),
b[i..i+8], c0)`（元素 i+k → lane k；c1/c2/c3 偏移 8..32）；最后 `t = (c0 + c1) + (c2 + c3)`（lane 内），
再按固定次序两两归约 8 lane。**为什么 4 条链**：1 条链时每 16 元素只有两条互相依赖的 FMA、暴露 FMA 延
迟——实测 `1×2048` 565 ns，4 条链后 **190 ns（3.0×）**；`dot n=4096` 1.2 µs → 359 ns。**谁守**：**转换精
确性可穷举**（f16 仅 65536 位型，单测遍历全部并与手写转换逐位比，NaN payload 只断言 `is_nan()`）；累加
（向量 == 标量模拟，n 覆盖 0/1/7/8/15/16/17/31/32/33/63/64/1000）；数值（f64 参考 <1e-6）；`gemv` 逐行
== `dot_f16`（逐位）；形状（`k=0 ⇒ y=0`，空向量点积 0）。**只靠约定**：C 侧裸符号 `a`/`x`/`y` 长度自
洽——违反是读写越界。

### 2.12 int8 对称量化的生产端

`lasx_amax` / `lasx_absmax_rows` / `lasx_quantize_i8_*` / `lasx_dequantize_i8*`。对称 int8、
`qmax = 127`（`q ∈ [−127,127]`，**不使用 −128**）：`amax = max_i |x_i|`（逐张量或逐行）；`scale = amax /
127`（`amax == 0 ⇒ scale = 0`）；`q_i = round_ties_even(x_i · (1/scale))`；`x̃_i = (q_i as f32) · scale`
（一次乘法、一次舍入）。**三条必须照抄的细节**：① **用倒数乘**，不写 `x / scale`（LASX 无向量除法；与
softmax 的 `e·(1/Σ)` 同一约定）；② 取整 **ties-to-even**（向量 `xvftintrne.w.s` / 标量
`round_ties_even()`）；已实测两者一致：`0.5→0、1.5→2、2.5→2、−0.5→0、−1.5→−2、−2.5→−2`；③ **不做夹
取**（`|q|≤127` 由 scale 定义保证：`scale=amax/127` 且 `|x|≤amax` ⇒ `|x·(1/scale)|≤127`；省 2 条指令，
且含 NaN 输入时两路径同结果）。**边界与退化**：`amax == 0`（全零或全 NaN）⇒ `scale=0`、q 全 0（显式分
支，不出现 `0×inf=NaN`）；空输入 ⇒ `amax=0`；`NaN` 被忽略（与 `f32::max` 同语义）；`±inf` 不在契约内
（scale 为 inf）。**谁守**：`|q|≤127` 由 scale 定义 + 取整（测试满量程 `max|q|≥120` 与边界 ±127）；量化
误差 ≤ `scale/2`（契约 + 999 随机样本实测最坏值）；反量化是 scale 的精确整数倍（一次乘法、逐位比）；长
度一致（Rust `api`：`Error::Shape`；**C 侧只靠约定**，违反是读写越界）；与 f64 参考逐位一致。

### 2.13 `lasx_gemv_i8`（int8 权重 × int8 激活）

`W` 是 `m×k` 行主序 int8 权重（**每输出通道一个 scale**），`x` 是 k 长 int8 激活（**每 token 一个
scale**，调用方量化时得到）：`y[o] = (Σ_i W[o,i]·x[i]) · (scale_w[o] · scale_x)`。

| 项 | 约定 |
|---|---|
| 累加 | **整数精确**：i16 乘积 → i32 累加 → 每 1024 B 落 i64（与 `lasx_dot_i8` 同一内核） |
| 出口 | **一次乘法**：先把两个 scale 乘成一个（一次舍入）再乘 `acc as f32`。写成 `((acc as f32)·sw)·sx` 是两次舍入、不保证逐位（有断言） |
| 回绕 | `k ≤ 133 000` 时 `127²·k < 2³¹` 不回绕；更大 k 按 `dot_i8` 的 mod 2³² 语义 |
| 偏置/残差 | **不在本内核内**：调用方在 f32 域自己加 |

**谁守**：累加精确性由整数运算保证；出口次序由契约 + 单测；长度由 Rust `api` 守，C 侧只靠约定。

### 2.14 `lasx_matmul_i8`（批量 int8 矩阵乘）

`X` 是 `m×k` int8 激活（每 **token** 一个 scale，`scale_x` 长 m），`W` 是 `n×k` int8 权重（每 **输出通
道** 一个 scale，`scale_w` 长 n）：`Y[t,o] = (Σ_i X[t,i]·W[o,i]) · (scale_w[o] · scale_x[t])`（即
`Y = X·Wᵀ`）。**数值契约**：每个输出元素是**一次 `dot_i8`**（整数精确、与顺序无关、`k > 133000` 同 mod
2³² 语义）；出口与 `gemv_i8` 同口径（两个 scale 先乘成一个）；**`m = 1` 时与 `gemv_i8` 逐位一致**（单测
守）。**实测结论（与直觉相反）**：**批量没有收益、且比 f32 GEMM 慢**——对"逐 token 跑 `gemv_i8`"是
**1.00×**（9.03 vs 9.06 ms @192×768×768），对 f32 `lasx_matmul(n=192)` 是 **0.4×（慢 2.5–2.7×）**。原
因：复用发生在 L2 而非寄存器（每个输出元素整条读 W 行），L2 带宽 ~12 GB/s；f32 一条 FMA 16 FLOP/指
令，本 ISA int8 链只有 ~4–5 MAC/指令（无融合 int8 点积）。**该用哪个见 §5.15。**

### 2.15 int8 路径的端到端口径（误差上界，不是位精确）

算子级标准（§2.12–§2.14）是逐位/整数精确；跨算子叠加后量化本身有损 ⇒ 端到端只能用**误差上界**：被测
管线 = **per-token 激活量化 → per-channel 权重量化 → `matmul_i8`**（出口已带两 scale 积）；参考量 =
**f64 累加**的 `Σ_i X[t,i]·W[o,i]`（既不是 f32 也不是 int8 路径）；误差 = **∞-范数相对量**
`max|Y_q − Y| / max|Y|`（不是逐元素相对误差——`Y≈0` 处会爆）；上界 **≤ 1e-2**，同口径 f32 路径
**≤ 1e-5**（实测 ~1.4e-6）⇒ 证明量化误差主导。实测（3 形状 × 3 种子）：`192×768×768`
5.216e-3/5.809e-3/5.256e-3；`192×768×3072` 5.387e-3/5.608e-3/5.024e-3；`32×1024×1024`
5.349e-3/6.047e-3/5.709e-3 ⇒ **最坏 6.047e-3，界 1e-2（1.65× 余量）**；读数在形状间稳定（k 768–1024、
n 768–3072 都在 5–6e-3）。**整模型级已实测（合成 12 层栈）**：每层 = `matmul_i8` → f32 RMSNorm →
`matmul_i8` → 残差相加，参考为同结构 f64 精确算术、权重不量化 ⇒ **int8 12 层 ∞-相对误差 7.250e-4**
（纯 f32 对照 1.725e-7）。**该实测推翻了 `√L` 外推**（12 层 ≈ 2.1e-2 的旧估计）：实测比它小 29 倍、甚至
比单层线性口径还小 8 倍；机制是残差撑大输出量级（∞-范数分母）、每层归一化重新标定激活，且量化误差**
不乘性累积**。**测试守界 5e-3**（~7 倍余量）。边界：这是**合成栈**（随机权重、无注意力/激活），只给"深
度累积量级"的证据、**不是具体模型验收值**；∞-相对量对输出量级敏感，跨架构比较无意义；归一化/softmax
走 f32，其误差不在表内。**该表之外**：注意力（QKᵀ/softmax）、归一化、embedding 的端到端数字见
`docs/dev.md`。

### 2.16 `lasx_layer_norm`（行内 LayerNorm）

外部契约要求（ONNX 图上 `LayerNormalization` 17 处）。与 §2.7 同一套行归约骨架，差别只有**减均值**与
**偏置 b**：`s = Σ_j x[j]`（第 1 遍：8 lane 累加、固定次序归约、列尾补 0.0）；`mean = s / cols; ss = Σ_j
(x[j] − mean)²`（第 2 遍，**补偏差 0.0**，不是补 x=0——那会凭空加 mean²）；`ms = ss / cols; inv = 1 /
sqrt(ms + eps)`；`out = (x[j] − mean) · inv · w[j] + b[j]`（次序是契约：先减、再乘 inv、再乘 w、最后加
b）。**三条必须照抄的细节**：① 方差用**两遍法**（`E[x²]−mean²` 在 `mean ≫ std` 时灾难性相消）；②
`w`/`b` 传空 ⇒ 无权重等价乘 1.0（精确）、**无偏置必须跳过加法**（加 `0.0` 会把 `−0.0` 翻成 `+0.0`，有
测试守）；③ 行级除法/开方**只在标量域做一次**再广播。**位精确**：向量路径与标量模拟逐位一致（8 种形状
含列尾 1/7/17/33/129 与 1×1 × 4 种 w/b 组合）；另有"平移不变性（LayerNorm 有、RMSNorm 没有）"与"空偏置
保 `−0.0`"两条行为断言。**谁守**：位精确由 `ops` 测试守；长度/eps 由 `api` 与 `_checked` 守；"减均值不
能退化成 RMSNorm"由平移不变性测试守。

### 2.17 `lasx_gather_rows` / `lasx_gather_rows_i8`（按 id 取行）

外部契约（ONNX 图上 `Gather` 34 处）：`out[i*row_len + j] = table[ids[i]*row_len + j]`；int8 口径再乘
`scales[ids[i]]`。**两条都没有累加** ⇒ 无结合次序问题：输出是 `table`/`ids`/`scales` 的确定逐元素函数
（int8 是"一次转换 + 一次乘法"）。因此**不用 LASX 手写**：行拷贝（典型 384 f32 = 1.5 KB）用
`copy_from_slice`（memcpy，LLVM 会向量化）更快也更好验证——**有意的选择，不是漏写内核**。**口径分工**
（承诺边界）：`row_len` 与"表存 f32 还是 int8"由**消费侧**决定；库侧只保证两种口径都正确（fp32 表 384
MB vs int8 96 MB，每前向只用 288 KB = 表的 0.08%）。**错误路径（都测了）**：`row_len == 0` →
`NotPositive`；`table.len()` 非 `row_len` 整数倍 → `Shape{table}`（`expected` = 向下取整到行）；`id >= 表
行数` → `Shape{ids}`（`expected` = 表行数、`got` = 越界 id）；`scales.len() != 表行数` →
`Shape{scales}`；`out.len() != ids.len()×row_len`（`_into`）→ `Shape{out}`。> **为什么 id 越界复用
`Shape` 不新增变体**：`Error` 是 C ABI 的一部分（`_checked` 按变体映射错误码），加变体要动错误码表与所
有绑定；而 `what="ids"` + `expected/got` 已把信息说全。**谁守**：边界由 `api` 与 `_checked` 守；逐位一致
（`_into` vs 分配版、与逐元素索引等价）由 `ops::gather` 与 `api` 测试守。

## 3. 运行时路径与能力

### 3.1 `SimdPath::detect()`

`pub enum SimdPath { Lasx, Lsx }`；`detect()` 返回 `hardware().lasx && !forced_lsx() ? Lasx : Lsx`，
`is_lasx()` 判。硬件能力用 `cpucfg` 读配置字 2：bit 7 = LASX、bit 6 = LSX。探测结果打包进**一个原子字
节**（写一次、之后只读），不是 `OnceLock`：快路径只剩一条 `Relaxed` 原子读。**进程级只探测一次**、之后
不可更新。支持降级的算子在入口 `match SimdPath::detect()` 分派到 `<name>_lasx`/`<name>_lsx`（少数内核
Lsx 分支退化为纯标量）。每调用边际成本实测 **0.2–0.45 ns**（"每次算子调用一次"的设计口径；**不要在循环
里重复调用**）。

### 3.2 降级覆盖分类

| 类别 | 内核 | 无 LASX 时 |
|---|---|---|
| LASX + LSX 双路径 | `dot`、`norm3_batch`、`vec3_add_scaled_batch`、`j2_accel_batch` + 全部 7 个姿态/几何算子 | LSX 128 位向量路径 |
| LASX + 标量降级 | `ballistic_step`、`batch_distance2d`、`rk4_j2_step_batch` | 全标量循环（无 LSX 向量路径） |
| LASX-only（无降级） | `matmul`、`matmul_scaled`、`matmul_f64`、`axpy`、`sum`、`dot_f64`、`dot_i8`、`dot_q4`、`layer_norm`、N1 全部 8 个（§2.6–§2.11）、N3 全部 8 个 | 直接执行 LASX 指令 |
| 不涉及 SIMD | `alloc`/`free`、`gather_rows`/`gather_rows_i8`、`has_lasx`/`version`/`status_message` | 与路径无关 |

N1 批次**有意不写 LSX 降级**：目标平台 3B6000（LA664）LASX 齐全，同核机型都支持 LASX；不为只有 LSX 的
机型（2K3000/3B6000M，LA364E）付这份复杂度。N3 同样 LASX-only。新符号一律**追加**，原始 15 个符号签名
与语义不变；`_checked` 变体与原始符号同一条路径、同一覆盖。

### 3.3 LASX-only 的硬性约束

一批 LASX-only 内核**无 `has_lasx()` 检查、无降级分支**（清单见 §3.2）：在只有 LSX 的机型上调用会执行
LASX 指令 ⇒ **非法指令（SIGILL）**，不会自动降级。**C 侧调度前先调 `lasx_has_lasx()`**——返回 0 就说明
当前生效路径只剩 LSX；不必自己读 `cpucfg`。LA464（3A5000）/ LA664（3A6000 等）支持 128/256 位两档。
`SimdPath::detect()` 与 `lasx_force_lsx_thread` 的降级只对有降级分支的内核有效；对 LASX-only 内核，
`lasx_force_lsx_thread(true)` 不改变行为。`lasx_has_lasx()` 返回的正是 `SimdPath::detect().is_lasx()`
（同一个源，尊重强制降级），代价约 0.2–0.6 ns，按批次问一次即可。线程级开关
`lasx_force_lsx_thread(force: bool)`（默认 false，仅 rlib 可见；测试后必须复位）；`arch::hardware() ->
HwCaps { lasx, lsx }`。

## 4. 导出符号总表（C ABI 权威：`docs/c-abi.md`）

> 本节已搬到 [`docs/c-abi.md`](c-abi.md)：完整符号清单与逐符号原型（含参数次序与 `[LASX-only]` 标注）、
> `_checked` 与状态码、内存与安全模型、能力探测、链路与 ABI 稳定性策略。本节只保留指向；Rust 侧内容
> （`api`/`view`/`plan`/`shape`/`pool`）不在搬迁范围。

归约/稠密/量化/物理/内存 16 个裸符号 → 第 3.1 节；姿态与几何 7 个 → 第 3.2 节；N1 批次 8 个 → 第 3.3
节；int8 推理 N3 批次 8 个 → 第 3.4 节；ONNX 契约补充 3 个 → 第 3.5 节；元信息与内存归还 → 第 3.6 节；
`_checked` 变体 → 第 3.7 节；状态码 0..6 → 第 3.8 节；计划句柄 7 个 → 第 3.9 节与第 11 节；符号数清点、
`nm -D` 自检、逐符号链接校验 → 第 10 节。

### 4.1 原始 15 个（历史契约，签名与语义永不改动）

**已搬移**：完整原型、路径标注与长度契约见 [`docs/c-abi.md`](c-abi.md) 第 3.1 节。这 15 个符号的签名与
语义**永不改动**（`lasx_alloc` 在其中，它没有 `_checked` 变体）。

### 4.5 状态码（`LasxStatus`）/ 4.6 符号数清点 / 4.8 元信息与内存归还

**已搬移**：0..6 码表与 `lasx_status_message()` 用法、分组计数与 `nm -D` 清点命令、哪些符号没有
`_checked` 变体、`lasx_has_lasx`/`lasx_version`/`lasx_status_message`/`lasx_free` 的原型与语义——见
[`docs/c-abi.md`](c-abi.md) 第 3.0/3.6/3.7/3.8/10 节（"先问能力再调"见第 6 节；内存配对与"`free()` 不是
ABI 契约"见第 4 节）。Rust 侧另有 `LasxStatus::message()`（中文原因）、`is_ok()`、`from_i32()` 供绑定层
还原；本文 §3.2 给出降级覆盖分类。

### 4.9 计划句柄：Rust 侧怎么用

C 侧原型、生命周期与线程语义见 [`docs/c-abi.md`](c-abi.md) 第 3.9 与 11 节；**Rust 侧同一套语义**（C 只
是它的投影）：`MatmulPlan::from_row_major(&b, k, n)?` 打包一次；`plan.run_block_into(&a[r0*k..], &mut
c[r0*n..])?` 调用方驱动（与 C 的 `lasx_matmul_run` 同一件事）；或
`parallel::matmul_f32_plan_with_stats(&pool, &plan, m, &a, &mut c)?` 库内派发（对应 C 的
`lasx_matmul_run_threaded{,_checked}`）。位精确同 C：这些入口与 `api::matmul` **逐位一致**。线程数语义：
C 侧 `<=0` 自动、`1` 串行、N 夹到物理核；Rust 用 `WorkerPool::new(n)`/`auto()`（**不读环境变量**）。
**全局池与冻结语义**：Rust `pool::global()` 是**进程级 `OnceLock`、建一次永不销毁** ⇒ 线程数在首次使用
时冻住；第一次"够大"的 `shape` 调用触发（小形状走单线程档、**不初始化池**，有契约测试）。显式出口：首
次使用前 `pool::init_global(n)`（首次成功、之后返回 `false`）；C 侧不需要它（`threads` 每次显式给）。切
分策略按形状自动选（判据 `src/pool/sched.rs::pick_rows`，阈值与实测口径见 `docs/dev.md`）；DSL/`shape`
路径与 C 侧共用同一份判据。**未动冻结接口**：冻结的是池的行块闭包签名 + start/切片对应、
`shape::Prepared::apply*` 与 `Policy::threads`；本次只改内部选哪个 `Pick`/句柄内部字段，签名与 `Pick` 枚
举都没变。

## 5. 归约、稠密、norm 与量化算子

### 5.1 `lasx_dot` — f32 点积

`float lasx_dot(const float *a, const float *b, int n)`；语义（硬契约）`Σ_{i<n} a[i]·b[i]`。`n < 24` 走**
纯标量 f32 累加**（向量建立 + f64 落盘 + 归约的固定开销摊不薄，实测 n=8 向量更慢）⇒ 跨 `n=24` 不保证
逐位连续；`n ≥ 24` 由 `detect()` 分派。LASX：每 256 元素一块、块内 **4 条独立 f32 累加链**（各 8 次
FMA），块末 4 个 f32×8 逐分量升 f64 累加，随后 8 元素尾 + 标量尾；LSX 同构。精度：对独立 f64 参考相对
误差 **<1e-5**（LASX 与强制 LSX 两路各自校验）。

### 5.2 `lasx_sum` / 5.3 `lasx_axpy` / 5.4 `lasx_dot_f64`

`float lasx_sum(const float *x, int n)`：`Σ x[i]`；与 `dot` 同构（乘加换加法）——每 256 元素一块、4 条
链、块末升 f64 累加、8 元素尾、标量尾；LASX-only；相对独立 f64 参考 **<1e-5**。
`void lasx_axpy(float alpha, const float *x, float *y, int n)`：**硬契约** `y[i] += alpha·x[i]`，**原地**
写回 y、不修改 x；向量路径 splat α、每 8 元素一条 `xvfmadd_s`（单条 FMA），标量尾 `y[j] += alpha*x[j]`；
至多差 **1 ulp**、**无逐位守护**（§2.1 例外）。`double lasx_dot_f64(const double *a, const double *b, int
n)`：`Σ a[i]·b[i]`（f64）；每向量 4 个 f64、主循环 4 元素步进 `xvfmadd_d`；结束把 4 通道累加器落盘逐分量
相加，标量尾直接累加；全程 IEEE 双精度；**无专门数值守护测试**（如实标注）。

### 5.5 `lasx_matmul` — f32 矩阵乘（LASX-only）

`void lasx_matmul(int m, int k, int n, const float *a, const float *b, float *c)`。语义（硬契约）：
`C[m×n] = A[m×k]·B[k×n]`，A/B/C 行主序，自 v1 起不变。实现：i-k-j 列方向微内核（4 行 × 32 列）；**不转
置 B、不做跨 lane 水平归约**；按形状在打包面板/列块/流式三条路径间自动选，**三路逐位一致**（输出元素
按 k 升序累加）；打包缓冲**线程本地**复用（按需增长、跨调用保留），其余两路不分配。精度：每个输出元素
沿 k **单个 f32 累加器**（无 f64 落盘），对 f64 参考相对误差 **<1e-4**；高精度用 `lasx_matmul_f64`
（§5.6）；`m == 0 || n == 0` 直接返回；`k == 0` 时 C 不被写（`parallel::matmul_f32` 会显式
`c.fill(0.0)`）。

**融合缩放（`C = alpha·A·B + beta·C`，口径见 §2.5 第 3 点）**

```c
void lasx_matmul_scaled(int m, int k, int n, float alpha, const float *a,
                        const float *b, float beta, float *c);              // beta==0 时不读 c
void lasx_matmul_scaled_checked(/* 同上 + int *status */);
```

```rust
lasx_rs::api::matmul_scaled_into(m, k, n, alpha, beta, &a, &b, &mut c)?;
matmul!(y[M, N] = alpha * x[M, K] * w[K, N]);   // 公式 DSL 标量前缀（beta = 0）
```

`alpha=1,beta=0` **逐位等于** `lasx_matmul`（同一内核、收尾不多乘一次）；其余情况
`C = fma(beta, C_old, alpha·acc)`，与"先算 A·B 再整体乘/加"**逐位一致**；C 入口与 `api` 逐位一致。何时
更快（进程级交替 5 轮）：`beta==0` 且宽薄（小 k、大 m×n）省一遍 C 读写——256×64×4096 **−10.3%**、
100×64×19147 **−23.7%**；`beta!=0 且 k≤384` 小胜（−6.5%）；方阵/大 k ≈0 但不更慢。

### 5.6 `lasx_matmul_f64` — f64 矩阵乘（LASX-only）

`void lasx_matmul_f64(int m, int k, int n, const double *a, const double *b, double *c)`。语义与布局同
`lasx_matmul`（行主序、不转置）；每向量 4 个 f64，一次算 4 行 × 16 列；同样有打包 + k 分块与流式两条路
径，两路逐位一致；线程本地打包缓冲同 §5.5。精度：全程 f64；对 f64 参考相对误差 **<1e-12**。

### 5.7 `lasx_dot_i8` — int8 量化点积（LASX-only）

`int lasx_dot_i8(const int8_t *a, const int8_t *b, int n)`：`Σ a[i]·b[i]`，int8 输入、**i32 返回**。整数精
确：每向量 32 个 i8，带符号 8→16 位扩展乘得 16 个 i16 再拓宽为 i32 累加（上界 128×127² ≈ 2.1 M，安
全）；每 **1024 字节**落盘转 i64，最后 `as i32` 返回。边界：超出 i32 范围**回绕**（无饱和/报错），约
`n > (2³¹−1)/127² ≈ 13.3 万`（全 ±127）时可能溢出；与 i64 精确参考逐位相等。

### 5.8 `lasx_dot_q4` — Q4 量化点积（LASX-only）

`double lasx_dot_q4(const uint8_t *qa, const float *sa, const uint8_t *qb, const float *sb, int
n_bytes)`。量化格式：每个 `u8` 含 **2 个 4-bit 无符号 nibble**（低 `q & 0x0f`、高 `(q>>4)&0x0f`）。scale
布局：`sa`/`sb` 每 **32 字节一组**一个值，`n_groups = ceil(n_bytes/32)`；长度必须 ≥ `n_groups`；非 32 倍
数时最后一组按 `j/32` 取 scale。公式 `dot = Σ_g sa[g]·sb[g]·Σ_{组内 64 nibble}(qa·qb)`：组内整数点积精
确，再乘该组 scale 积（f32 乘法）并升 f64 累加；每批 4 组（128 B）独立求和；归约与 f64 累加严格按组序
（与逐组循环逐位不变）；nibble ∈[0,15]：乘积 ≤225、组内每 lane ≤900、总和 ≤14400，整数无溢出。精度：
对精确参考相对误差 **<1e-12**；误差主要来自 f32 scale 表示（约 2⁻²⁴）。

### 5.9 `lasx_softmax_rows`

| 层 | 签名 |
|---|---|
| C（裸） | `void lasx_softmax_rows(const float *x, const float *mask, float *out, int n_rows, int n_cols, float scale)` |
| C（`_checked`） | 同上 + `int *status`；`mask` 允许 NULL；`n_rows×n_cols > 0` 时 `x`/`out` 空报 `NullPointer` |
| Rust | `api::softmax_rows(&[f32], Option<&[f32]>, rows, cols, scale) -> Result<AlignedVec<f32>>`（`_into` 免分配） |

语义与数值契约见 §2.6；LASX-only；吞吐见 `docs/dev.md` 性能章节。

### 5.10 `lasx_rms_norm`

| 层 | 签名 |
|---|---|
| C（裸） | `void lasx_rms_norm(const float *x, const float *w, float *out, int n_rows, int n_cols, float eps)` |
| C（`_checked`） | 同上 + `int *status`；`w` 允许 NULL；`eps` 要求有限且 > 0 |
| Rust | `api::rms_norm(&[f32], Option<&[f32]>, rows, cols, eps) -> Result<AlignedVec<f32>>`（`_into` 免分配） |

语义 `out = x/√(mean(x²)+eps)·w`（`w` 按列、长度 cols）；数值契约见 §2.7；与 softmax 的差别是**遍数**
（rms_norm 两遍 = 2 条流，softmax 三遍 = 3 条流）。

### 5.11 `lasx_silu` / `lasx_gelu_quick` / `lasx_gelu_erf` — 逐元素激活

> **Rust 侧有免分配版**：`api::{silu_into, gelu_quick_into, gelu_erf_into}`（与分配版**逐位一致**）。**逐层
> 激活每次都要调**，实测分配占该调用 **10–15%**（192×3072、589 824 元素）；C 侧本来就是调用方给 `out`。

| 层 | 签名 |
|---|---|
| C（裸） | `void lasx_silu(const float *x, float *out, int n)`；`lasx_gelu_quick(...)`；`lasx_gelu_erf(...)` |
| C（`_checked`） | 同上 + `int *status`；`n < 0` → `NegativeLength`，`n > 0` 且空指针 → `NullPointer` |
| Rust | `api::{silu,gelu_quick,gelu_erf}(&[f32]) -> AlignedVec<f32>`（+`_into`） |

定义：`silu(x) = x/(1+exp(−x))`；`gelu_quick(x) = x/(1+exp(−1.702x))`（ggml GELU_QUICK 同式）；
`gelu_erf(x) = 0.5x(1+erf(x/√2))`（**近似的 erf**）。**形状只是 `n`**（逐元素）、**允许就地**（`out ==
x`：内核每轮先读完 8 个再写回同样位置）。数值契约：`silu`/`gelu_quick` 见 §2.8、`gelu_erf` 见 §2.9。

### 5.12 `lasx_rope` — 旋转位置编码

> **Rust 侧有免分配版 `api::rope_into`**（与分配版逐位一致，两种 `RopeMode` 都测）。**全库分配占比最高
> 的算子：实测 39–41%**（192×768、n_dims=128）⇒ 热路径用 `_into`。**更值钱的一条用法：`rope_tables` 必
> 须 hoist 出层循环**——`api::rope_at` 每次调用都重算 cos/sin 表（三角 + 两次拷贝 + 两次分配），实测
> **~5.2e2 µs vs 预建表 + `rope` 的 ~5.7e1 µs（差 89%，约 9×）**：层循环外调一次
> `rope_tables(&positions, n_dims, freq_base)`，每层只调 `rope_into`。

| 层 | 签名 |
|---|---|
| C（裸） | `void lasx_rope(const float *x, const float *cos, const float *sin, float *out, int n_rows, int n_cols, int n_dims, int mode)` |
| C（`_checked`） | 同上 + `int *status`；形状/mode 非法 → `BadShape`，空指针 → `NullPointer` |
| Rust | `api::rope(...) -> Result<AlignedVec<f32>>`（+`rope_into`）、`api::rope_tables(...)`、`api::rope_at(...)`（便捷，每次重建表） |

`cos`/`sin` 各 `n_rows × (n_dims/2)`、行主序、**与 mode 无关**；`mode`：`0`=`NeoX`、`1`=`GptJ`；**允许就
地**；数值契约与 mode 选择理由见 §2.10；LASX 只覆盖 `NeoX`（GptJ 标量）。

### 5.13 `lasx_dot_f16` / `lasx_gemv_f16`

| 层 | 签名 |
|---|---|
| C（裸） | `float lasx_dot_f16(const uint16_t *a, const float *b, int n)`；`void lasx_gemv_f16(const uint16_t *a, const float *x, float *y, int m, int k)` |
| C（`_checked`） | 同上 + `int *status`；负长度 `NegativeLength`、空指针 `NullPointer`、乘积溢出 `SizeOverflow` |
| Rust | `api::dot_f16(&[u16], &[f32]) -> Result<f32>`；`api::gemv_f16(m, k, &[u16], &[f32]) -> Result<AlignedVec<f32>>`（+`_into`） |

f16 以 `u16` 位型进出（零依赖）；f16→f32 **精确**（单测穷举全部 65536 位型对硬件验证），误差只来自权重
本身是 f16（相对 ~4.9e-4）；累加次序见 §2.11；`k = 0` ⇒ y 全 0；`gemv` 每行与 `dot_f16` **逐位相同**。
**并行**：`parallel::gemv_f16(&pool, m, k, &[u16], &[f32], &mut [f32])`，每行独立、按行块切分，与单线程
逐位一致；实测 `1024×2048` 单线程 → 12 线程 **7.9×**；线程数取物理核数（本机 12），24 线程在中小形状反
而更慢。**声明式**：形状编译期已知时可写 `gemv!(w_f16[N, K] * x[K])` / `dot!(w_row[K] * x[K])`，与
`api` 逐位一致。

### 5.14 int8 推理（量化 → `gemv_i8`）

| 层 | 签名（要点） |
|---|---|
| C | `lasx_amax`、`lasx_absmax_rows`、`lasx_quantize_i8_per_tensor`、`lasx_quantize_i8_per_row`、`lasx_dequantize_i8`、`lasx_dequantize_i8_rows`、`lasx_gemv_i8(const int8_t *w, const float *scale_w, const int8_t *x, float scale_x, float *y, int m, int k)` |
| C（`_checked`） | 同上 + `int *status`；负长度 `NegativeLength`、空指针 `NullPointer`、尺寸相乘溢出 `SizeOverflow` |
| Rust | `api::{amax,absmax_rows,quantize_i8_per_tensor,quantize_i8_per_token,quantize_i8_per_channel,dequantize_i8,dequantize_i8_rows,gemv_i8}`（`_into` 变体同族） |

判别式模型（cross-encoder，无 KV cache、无自回归解码）的 int8 前向 = "**激活按 token 量化 → 权重按输出通
道量化 → GEMV → 残差/归一化留在 f32 域**"：`quantize_i8_per_channel(&w_f32, out, in)?`（一次性）→
`quantize_i8_per_token(&act, tokens, hidden)?`（每层每次）→ `gemv_i8(w_q, w_scale, x_q, x_scale[0], out,
hidden)?` → `dequantize_i8(&q_out, scale_out)?`（回 f32 域）。契约见 §2.12/§2.13：倒数乘、ties-to-even、
不夹取、出口两 scale 先乘成一个；精度：量化误差 ≤`scale/2`、反量化是 scale 的精确整数倍、GEMV 整数累加
精确。**未做**：整数/定点 softmax、int8 归一化——**已按实测否掉**（组合路径占该层 int8 GEMM 的 ~2%）：
要保活就 `dequantize_i8_rows` → `api::rms_norm`/`softmax_rows` → `quantize_i8_per_token` 拼；**per-token
量化不付性能代价**（per_row 与 per_tensor 同量级、逐行略快）。

### 5.15 int8 还是 f32：一句话判据（给下游）

**先看权重装不装得进缓存，再选精度——不要按"int8 一定快"选。**

| 形态 | 选谁 | 为什么（实测） |
|---|---|---|
| **单 token / 权重 > 缓存**（必须从 DRAM 流权重） | **int8**（`gemv_i8`） | 每权重 1 B vs 4 B，两者都贴着单流只读锚点（~12 GB/s）⇒ **省 4× 流量** |
| **批量（多 token 复用同一片权重）、权重驻 L2/L3** | **f32**（`lasx_matmul`；多核 `parallel::matmul_f32`） | 算力受限：f32 一条 FMA = **16 FLOP/指令**，本 ISA int8 链只有 **~4–5 MAC/指令**（无融合 int8 点积）⇒ 实测 f32 快 **2.5×** |

配套结论：① **归一化/softmax 用 `_into`**（与分配版逐位一致）：`192×768` 实测分配版 ~1.8e2 µs vs
`_into` ~7.5e1 µs（**快 2.4×，分配占 58%**），`192×3072` 占 63% ⇒ 逐层保活路径一个前向省 ~11 ms；凡"每层
都要调一次"的算子优先用 `_into`；② **批量形状按 6–8 线程派活**：`192×768×3072` 这类形状 8 线程最优
（5.99×）、12 线程 5.11×、24 线程 4.57×；立方形状（512³/1024³）多核下越大越好，256³ 这类小 B 仍走回退
路径；③ **单 token（m=1）用 `parallel::gemv_i8_k`**：按 k 切块，12 线程实测 **5.4–6.9×**（`k ≥ 1 MiB`
才有收益；k 小到几 KiB 时并行是负收益，函数内按"每块 ≥1024 且 ≥4 块"退化为串行）——它修的是访存并行
度（单线程冷流只有 ~3 GB/s），不是"带宽不够"；④ **量化生产端本身不贵，per-token 免费**（per-row 与
per-tensor 同量级、逐行略快）⇒ 为准确率选 per-token 不付性能代价。

### 5.16 `lasx_layer_norm` / `lasx_gather_rows` / `lasx_gather_rows_i8`（ONNX 契约补充 3 个）

| 符号 | 语义 | 契约 |
|---|---|---|
| `lasx_layer_norm` | `out = (x − mean)/√(var+eps)·w + b`（`w`/`b` 可 NULL；方差**两遍法**） | §2.16 |
| `lasx_gather_rows` | `out[i] = table[ids[i]]`（纯行拷贝，**不用 LASX**） | §2.17 |
| `lasx_gather_rows_i8` | 行拷贝 + 每行 scale 乘 | §2.17 |

**Rust 侧同时有免分配版**：`api::{layer_norm_into,gather_rows_into,gather_rows_i8_into}`（以及对应分配
版）——**新算子一律给两个入口**。"嵌入表存 f32 还是 int8"不是本库的决定：库侧只保证两种口径都正确、边
界责任清楚（id 越界在 C 侧 `BadShape`(3)，Rust 侧 `Error::Shape`）。

## 6. 批量几何与物理算子

> **这一族的 `_into`（免分配）版本：决定"不做"**（消费者不再调用它们、签名重设成本高：一次返回 3–9 个
> 缓冲；分配占比也不支持——`quat_to_dcm_batch` 对称 A/B 只有 3.4–3.8%、`norm3_batch` 17.3–17.7%）。**触发
> 条件**：若将来出现"仿真循环里逐个调用这族"的消费者，先量占比再决定。调用方改用"预分配 + FFI"时**根本
> 没有这笔分配**。

### 6.1 `lasx_norm3_batch` — 批量 3 分量模长（LASX+LSX）

`void lasx_norm3_batch(const double *xs, const double *ys, const double *zs, double *out, int n)`；语义
`out[i] = √(xs[i]² + ys[i]² + zs[i]²)`，SOA 布局。结合序固定 `x² + (y² + z²)`（右结合）；LASX 4 样本/
向量、LSX 2 样本/向量、标量尾同式，分块内逐位一致；对独立标量参考（左结合写法）相对误差 **<1e-9**。
Rust `api::norm3_batch_into(xs, ys, zs, out)`：同一内核，与分配版逐位一致。

### 6.2 `lasx_vec3_add_scaled_batch` — 批量缩放加（LASX+LSX）

`void lasx_vec3_add_scaled_batch(6×const double*, double s, 3×double*, int n)`；语义
`o[i] = a[i] + s·b[i]`，三分量独立（SOA）；覆盖加（s=1）与缩放（a=0）两个退化情形。向量路径单条 FMA
`xvfmadd_d(vs, vb, va)`；标量尾 `f64::mul_add` 同式，分块内逐位一致；相对独立标量参考 **<1e-9**。Rust
`api::vec3_add_scaled_batch_into(…, ox, oy, oz)`。

### 6.3 `lasx_batch_distance2d` — 批量 2D 距离（LASX+标量）

`void lasx_batch_distance2d(float px, float py, const float *xs, const float *ys, float *out, int n)`；语义
`out[i] = √(dx² + dy²)`，`dx = xs[i]-px`、`dy = ys[i]-py`。LASX 8 点/向量：`xvfsub_s` 差、
`xvfmul_s`+`xvfadd_s` 平方和、`xvfsqrt_s` 开方；标量尾同式同结合序，分块内逐位一致；非 LASX 时全标量。
Rust `api::batch_distance2d_into(px, py, xs, ys, out)`。

### 6.4 `lasx_ballistic_step` — 批量弹道欧拉步（LASX+标量）

`void lasx_ballistic_step(6×float*, const float *k, int n, float dt, float g)`；语义：SOA，n 发弹（每发 6
状态 + 阻力系数 k）同步推进一步**欧拉**、原地更新。标量参考：`v = sqrt(vx² + vy² + vz²); drag = k·v;
vx -= drag·vx·dt; vy -= (drag·vy + g)·dt; vz -= drag·vz·dt; x += vx·dt`（用更新后的新速度）。LASX 向量路
径（8 发/向量）：`vsq = vx² + (vy² + vz²)`（右结合）；`vnx = vx − (vdrag·vx)·dt`；`vny = (vy −
(vdrag·vy)·dt) − g·dt`；`vnz = vz − (vdrag·vz)·dt`；`x' = fma(vnx, dt, x)`。**物理一致但非逐位一致**
（`|v|²` 结合序不同）。测试守护绝对误差（速度 <1e-3、位置 <1e-2；n=64、dt=0.005、g=9.81、k=1e-5）。非
LASX 时全标量。

### 6.5 `lasx_j2_accel_batch` — 批量 J2 引力加速度（LASX+LSX）

`void lasx_j2_accel_batch(3×const double* r, double mu, double j2, double re, 3×double*, int n)`；语义：中
心点质量 + J2 摄动加速度（与 loong-sci `EarthModel::j2_acceleration` 同公式）：

```text
rm² = x² + (y² + z²);  rm = √rm²;  rm³ = rm·rm²;  rm⁵ = rm³·rm²
j2k = 1.5·J2·μ·Re²（正值）；inv2 = 1/rm²; invrm = inv2·rm; inv3 = inv2·invrm; inv5 = inv3·inv2
zr2 = z²·inv2;  k = j2k·inv5;  vcen = −μ·inv3
a_x = fma(k·x, 5·zr2 − 1, vcen·x)；a_y 同；a_z = fma(k·z, 5·zr2 − 3, vcen·z)
```

`j2k` 取正值（等价于减 J2 项）；每样本完全独立（LASX 4 路 / LSX 2 路），无跨 lane 归约；向量与标量尾同
式同 FMA，分块内逐位一致；对独立参考相对误差 **<1e-9**。

### 6.6 `lasx_rk4_j2_step_batch` — 批量 RK4 J2 步（LASX+标量）

`void lasx_rk4_j2_step_batch(6×double*, double mu, double j2, double re, double dt, int n)`；语义：经典 4
阶 Runge–Kutta 单步（中心项 + J2 力模型），批量推进 n 个轨道状态 `[r(3), v(3)]`，**原地更新**（每块先读
后写、块间不重叠）。公式（`h = dt`、`hh = dt/2`、`h6 = dt/6`）：

```text
k1 = f(r0)
r2 = fma(hh, v0, r0);  v2 = fma(hh, k1, v0);  k2 = f(r2)
r3 = fma(hh, v2, r0);  v3 = fma(hh, k2, v0);  k3 = f(r3)
r4 = fma(h,  v3, r0);  v4 = fma(h,  k3, v0);  k4 = f(r4)
l  = fma(2, v2, v0) + fma(2, v3, v4)；k = fma(2, k2, k1) + fma(2, k3, k4)
r' = fma(h6, l, r0)；v' = fma(h6, k, v0)
```

`f(r)` 是 §6.5 的 lane 版本（`vmu` 已预取负）；LASX 4 样本/向量，k1..k4 与组合算术全在寄存器内；非 LASX
走全标量同式；分块内逐位一致。测试：50 步累积相对误差 <1e-9（dt=10）；另有 FMA vs 分离 mul+add 测试
（单步 ≤1 ulp、50 步 <1e-9）。多核见 `parallel::rk4_j2_step_batch`。

### 6.7 `shape::batch` — 批量算子的形状层视图（N 进类型）

上面算子都是"分量拆开的数组 + 一个 n"，长度只靠约定；形状层把它变成**构造时校验一次**：
`V3Ref`/`V3Buf`（x/y/z 三条）、`RvMut`（rx…vz 六条，原地）、`V2Ref`（x/y 两条，f32）。用法：
`V3Ref::<f64, N>::new(&ax, &ay, &az)?`（三条不等长 ⇒ `Err`，点名哪一条）；`a.norm3_into(&mut mag)`、
`a.cross_into(&b, &mut o)`、`a.j2_accel_into(mu, j2, re, &mut o)`、`a.add_scaled_into(&b, 0.5, &mut
o)`、`a.unitize3_into(&mut o)`——各自转发 §6.1–§6.5 内核。**两组视图**（按"长度在哪"分）：
`V3Ref`/`V3Buf`/`RvMut`/`V2Ref` 长度在**类型里**（const N）；`V3Dyn`/`V3DynMut`/`RvDynMut`/`V2Dyn` 长度在
**运行期字段**（`*Dyn` 构造以第一条分量为基准校验其余等长，输出数组在调用点再校验一次）。**为什么要有
这一层**：发布构建里长度不一致原本是 UB（内核用第一条数组长度当 n，FFI 转发层只有
`debug_assert_eq!`）；三条/六条等长现在是 `new()` 的一次显式校验，**不等长拿不到视图**。**数值**：只做
视图与转发，逐位一致（每个方法有 `to_bits()` 对照测试，覆盖 N 不是向量宽整数倍的尾部）；**别名**：输入
`&V3Ref`、输出 `&mut V3Buf`，重叠由借用检查器拒绝。**不做的**：`quat_*`/`mat3_mul_vec3`/`cross3` 那一族
的"定点 3×3/4×4 展开后端"——唯一下游没有调用点（`cross3`/`unitize3` 作为 3 分量视图的自然成员顺手提
供）。

## 7. 批量姿态与几何算子（7 个）

公共约定：四元数**标量在前** `q = w + xi + yj + zk`，Hamilton 积 `i²=j²=k²=ijk=−1`；旋转矩阵按 Wertz
Ch.12.2 构造，方向是**体坐标 → 惯性坐标**（惯性转体先取共轭）；`|q| < 1e-15` 视为退化 ⇒ 输出单位四元数
`(1,0,0,0)`；分量求和结合次序固定，LASX / LSX / 标量尾三路**逐位一致**；全部内核**允许输出与输入别
名**。LASX 4 样本/向量、LSX 2 样本/向量。

| 算子 | 语义与数值约定 |
|---|---|
| `lasx_cross3_batch` | 右手系 `o = a × b`：`(ay·bz − az·by, az·bx − ax·bz, ax·by − ay·bx)`；逐位一致 |
| `lasx_unitize3_batch` | `o = v/\|v\|`；`\|v\|² = x²+(y²+z²)`（与 norm3 同结合序 ⇒ 与"先 norm3 再逐分量除"逐位一致）；`\|v\| == 0 → (0,0,0)` |
| `lasx_mat3_mul_vec3_batch` | `o = M·v`；`m0..m8` **行主序**（`M[r][c]=m[3r+c]`）；每行左结合 `((m0·x+m1·y)+m2·z)` |
| `lasx_quat_normalize_batch` | **原地**；范数左结合 `√(((w²+x²)+y²)+z²)`；`\|q\|<1e-15 → (1,0,0,0)`，否则逐分量**相除** |
| `lasx_quat_mul_batch` | Hamilton 积（`.w=aw·bw−ax·bx−ay·by−az·bz` 等，每行左结合，**不做内部归一化**） |
| `lasx_quat_rotate_batch` | 输入**先单位化**（含退化规则），再 `o = R(q)·v`（体→惯）；`R` 与 `quat_to_dcm_batch` 共用实现 |
| `lasx_quat_to_dcm_batch` | 输入**先按退化规则单位化**；输出行主序 `R = [1−2(y²+z²), 2(xy−wz), 2(xz+wy); 2(xy+wz), 1−2(x²+z²), 2(yz−wx); 2(xz−wy), 2(yz+wx), 1−2(x²+y²)]`，体→惯 |

## 8. Rust 安全层 `lasx_rs::api`

只走 rlib、不导出符号。切片进出、形状不符返回 `Err`、输出是 **64 字节对齐**的 `AlignedVec`；与 C ABI 走
同一段内核，无可测开销。

| 函数 | 签名摘要 | 失败条件 |
|---|---|---|
| `api::sum` | `fn(&[f32]) -> f32` | 不会失败 |
| `api::dot` / `dot_f64` | `fn(&[T], &[T]) -> Result<T>` | 长度不一致 |
| `api::dot_i8` / `dot_q4` | `fn(&[i8], &[i8]) -> Result<i32>` / `fn(&[u8],&[f32],&[u8],&[f32]) -> Result<f64>` | 长度不一致；`dot_q4` 还要求 scale 长度 ≥ `ceil(n/32)` |
| `api::axpy` | `fn(f32, &[f32], &mut [f32]) -> Result<()>` | 长度不一致 |
| `api::matmul` / `matmul_f64` / `matmul_f32_packed` | `fn(m,k,n,&[T],&[T]) -> Result<AlignedVec<T>>` | 形状不符、乘积溢出 |
| 几何/物理/姿态族 | 见 §6、§7（+`_into`） | 数组不等长；`j2/rk4` 还校验 `mu/re>0`、`j2/dt` 有限 |

校验顺序与 `_checked` 一致（物理常数先于形状）；`dot_q4` 的 scale 按"≥ 组数"约定、多给合法。`api::Error`
四个变体（都带算子名与参数名，实现 `Display + std::error::Error`）：

| 变体 | 触发条件 | 消息示例 |
|---|---|---|
| `Shape` | 长度/形状不符 | `matmul: 参数 b 的长度应为 k×n = 9，实际 4` |
| `Overflow` | `m×k` 等乘积溢出 `usize` | `matmul: m×k 溢出` |
| `NotFinite` | 常数是 NaN/±∞ | `rk4_j2_step_batch: 常数 dt 必须是有限数，得到 NaN` |
| `NotPositive` | 要求为正的常数 ≤ 0 | `j2_accel_batch: 常数 mu 必须为正，得到 0` |

危险通路仍存在、只是要显式选它：`lasx_*` 参数是裸指针、函数本身不是 `unsafe fn`（历史约定），**误用即
UB**——长度必须由调用方保证。

### 8.1 矩阵视图 `view::MatRef` / `view::MatMut`

`api::matmul(m,k,n,&a,&b)` 的三个形状数字与切片长度的对应完全依赖记忆——本仓库在此处发生过真实缺陷。
`view` 把"形状 + 跨距"和数据放进一个对象，**构造时校验一次**：

| 构造/方法 | 说明 |
|---|---|
| `MatRef::row_major` / `col_major` / `row_major_strided` | 行主序 / 列主序 / 零拷贝切块（每行占 `row_stride` 元素） |
| `rows()`/`cols()`/`is_empty()`/`row_stride()`/`col_stride()` | 形状与跨距读数 |
| `row(i)` → `RowRef` / `row_checked` / `iter_rows` | 取整行；`RowRef` 支持 `get(j)`、`as_slice()`（连续时）、`iter()` |
| `get(i,j)` / `transpose()` | 单元素；**零拷贝**转置视角（`m.transpose().row(j)` 即第 j 列） |
| `as_row_major_contiguous()` / `to_row_major_vec()` | 连续时给切片 / 非该布局时**明确复制**（代价写在名字里） |
| `MatMut::row_major` / `row_mut(i)` / `fill(v)` | 可写视图（只提供行主序，内核输出都行主序） |

失败发生在**构造**处、消息带期望值与实际值（`Error::Shape`）。为什么多一个 `RowRef` 而不是直接返回
`&[T]`：**列主序矩阵的一行在内存里不连续**；返回语义不成立的连续切片正是"行跨距用错"的温床，布局连续
时 `RowRef::as_slice()` 仍给回真正的切片。

### 8.2 计划复用 `plan::MatmulPlan`

`api::matmul` 每次调用都要把 B 按列条带重新打包（微内核才能顺序读 B），代价 `O(k·n)`；小矩阵上比乘法本
身还贵。`MatmulPlan` 把打包**提前到构造时**：`MatmulPlan::from_row_major(&b, k, n)?` 打包一次 →
`plan.run(&a)?`（反复算，`m = a.len()/k`）或 `plan.run_into(&a, &mut c2)?`（写调用方缓冲、零分配）；与
`api::matmul` **逐位一致**（测试断言 `c1 == api::matmul(...)`）。接口：`MatmulPlan::new(&MatRef<T>)` 从
**任意布局**的 B 视图构造（行主序/列主序/带跨距/转置视角都行）；`from_row_major(&[T], k, n)` /
`run(&[T])` / `run_into(&[T], &mut [T])`；`k()`/`n()`/`packed_bytes()`。`k == 0` 时 m 无法从
`a.len()` 推出 ⇒ 构造返回 `Error::NotPositive`（退化情形用 `api::matmul`）；计划里存一份打包好的 B（构
造做一次 `O(k·n)` 搬运）；**只算一次**就用 `api::matmul`。计划固定走"打包 + k 分块"；`api::matmul` 个
别形状会选流式/列块路径，两者仍逐位一致；**需要 LASX**。`run_into` **只借用 `&self`**，可 `Arc` 给多线
程写不相交行段。实测（单线程）：`512³` 4.53→4.32 ms（1.05×）、`256³` 1.09×、`1000×64×4096` 1.04×、
f64 `128³` 1.08×；详细三列对照见 `docs/dev.md`。

### 8.3 形状进类型 `shape::Mat` / `shape::MatDyn`

形状**编译期已知**时，`shape` 层把它搬进类型：形状错误变**编译错误**，排版（面板数、k 分块、列尾、打包
字节数）在 `const` 求值里定死。类型：`Mat<'a, T, R, C>` / `MatBuf<T, R, C>`（`R×C` 只读视图 / 输出拥有
者）；`MatDyn<'a, T, C>` / `MatBufDyn<T, C>`（**DYN 档**：行数运行时、列数 K 在类型里）；
`Layout<K, N>` / `Prepared<T, K, N>`（编译期排版方案 `PANELS`/`K_CHUNKED`/`TAIL_COLS`/`PACK_BYTES`… / B
打包一次反复用，内部就是 `MatmulPlan`）。

```rust
use lasx_rs::shape::{Mat, MatDyn, MatBufDyn};
let w = Mat::<f32, 256, 256>::new(&weights)?;      // K/N 进类型
let wp = w.prepare();                              // 打包一次
let x = Mat::<f32, 64, 256>::new(&batch)?;
let y = x.mul(&w);                                 // ① 公式写法：K 由类型对齐
let y2 = wp.apply(&x);                             // ② 声明式：复用打包好的权重
let xd = MatDyn::<f32, 256>::new(&other_batch)?;   // ③ DYN：行数每批不同
let mut yd = MatBufDyn::<f32, 256>::with_rows(xd.rows());
wp.apply_dyn_into(&xd, &mut yd);
```

**K 对不上是编译错误**（报在公式那一行）；构造函数只查长度，之后 `apply*` **不返回 `Result`**（形状已由
类型担保）；与 `api::matmul` **逐位一致**（同一套打包 + k 分块内核），含 DYN 多种行数。**公式 DSL 里复用
同一个 plan**：右操作数**不写方括号**就是 `Prepared`（自带形状）：`let p = w.prepare();`（循环外一次）+
`matmul!(y[M, N] = x[M, K] * p);`（生成 `p.apply_into(&x, &mut y)`）。带方括号的
`matmul!(y[M,N] = x[M,K] * w[K,N])` 行为**完全不变**——它生成 `x.mul_into(&w, &mut y)`，而 `mul_into` 内
部是 `rhs.prepare().apply_into(...)` ⇒ **每次调用都重新打包**。层循环里复用同一权重时两者差是纯打包开
销：256³ 实测 `142.2 µs → 99.3 µs`（**省 30%**），两臂与手写 hoist 逐位一致。plan 只能写在乘积**右侧**
（左侧留给标量前缀的诊断）；**不做**隐式/指纹缓存（缓冲区复用时会给陈旧结果，与位精确承诺冲突）。三层
分工：`view` 管布局、`plan` 管复用、`shape` 管形状。

#### 8.3.1 一维（向量）与 f16：`dot!` / `gemv!`

类型：`VecRef<'a, T, N>` / `VecBuf<T, N>`（N 长只读视图 / N 长输出，32 B 对齐）；`F16Vec<'a, N>` /
`F16Mat<'a, N, K>`（f16 以 u16 位型；权重行主序 `[输出, 收缩]`）；`DotPair` / `dot_of`（点积按 dtype 分
派：f32·f32、f64·f64、f16·f32 双向）。用法：`let y: VecBuf<f32, N> = gemv!(w_f16[N, K] * x[K]);`（分配
输出）或 `gemv!(y[N] = w_f16[N, K] * x[K]);`（写已有缓冲）；`let s: f32 = dot!(act[K] * act[K]);`。**只
支持编译期长度**（v1）；运行期长度直接用 `api::gemv_f16` 或 `parallel::gemv_f16`。**`gemv!` 只认一种读
法**：权重在左、行主序 `[输出, 收缩]`；写成 `x[K] * w[N, K]` 在**编译期**报错并给出正确写法；数值与
`api`/`parallel` **逐位一致**；生成代码用绝对路径（用户不必 `use` 类型）。

## 9. 多核并行：`WorkerPool` 与 `parallel`

内核本身单线程（每个 `lasx_*` 只处理一段连续内存）；多核靠可选的**常驻**线程池。

### 9.0 实战：用公式 DSL 做单边数值拉普拉斯变换

`F(s) = ∫₀^∞ f(t)e^{-st}dt` 离散成**两条矩阵乘**（`t_i = i·Δt`、`s_j = σ_j + iω_j`、`w_i` 权重）：
`Kc[j,i] = w_i·e^{-σ_j t_i}·cos(ω_j t_i)`、`Ks[j,i] = w_i·e^{-σ_j t_i}·sin(ω_j t_i)`；
`Re F = Σ Kc·f`、`Im F = −Σ Ks·f`。入口：`laplace_batch_f32::<M, N, NB>(&sigs, &oms, &dt, &f)`（形状全在
类型里）。**误差量级**（`T=6.4`、`f=e^{-0.5t}`、f64）：`N ∈ {64,128,256,512}` ⇒ 最大相对误差 **1.7e-2 →
4.3e-3 → 1.1e-3 → 2.6e-4**，相邻比 ~4.0×（梯形法 `O(Δt²)`）；f32 在 N=512 约 1e-3。地板由**截断**（T 有
限）决定：固定 Δt 只加 N 不再改善，要更准须**同时**加 T 与减小 Δt。**权重方案**：`Weighting::{Trapezoid,
Simpson}`，**默认 Trapezoid**（改默认会改变已有调用的结果）。Simpson = `1,4,2,4,…,4,1` 再 /3、`O(Δt⁴)`，
要求 **N ≥ 3 且 N 为奇数**；`N<3` 或 N 偶数 ⇒ **整段回退梯形**并由第二个返回值如实标出（不"只丢尾巴"
——末段低阶会引入 `O(Δt²)` 项拖回二阶）。**实测收敛**（`T=20`、`f=e^{-2t}`、解析 `1/(s+2)`）：梯形相邻比
3.84/3.96/3.99×；Simpson 12.30/14.85/15.69×（同一 N 下 Simpson 准 5.8–274×）。**阶数只在不被 T 截断地
板支配时可见**（`T=6.4` 时 Simpson 撞 ~7e-5 地板、相邻比掉 ~1；有测试断言）。**边界与限制**：不是任意精
度（截断 + 梯形）；`s` 靠近极点时不能用（`f≡1` 的 `s=0` 真值发散，函数给截断和 `(N-1)Δt`；`ω=0` 时虚
部恒 0）；**不做逆变换**；`f` 是 `N×NB` 行主序、NB 条信号并列。

**切分策略按形状自动选**：大形状（`per_thread ≥ 32` 且 `threads ≥ 12`）走**动态小额认领**（实测
+4.9%…+21.4%），其余走**静态等分**。等待策略"短自旋 `SPIN_LIMIT = 1024` 圈 → `Condvar` park"实测已最优
（worker 自耗时 <0.76%），不必调。接口：`WorkerPool::new`/`auto`/`threads`（建池**一次跨调用复用**）；
`pool::global`/`pool::init_global`（进程级共享池 / 首次使用前指定线程数，已建好返回 `false`）；
`for_each_chunk_mut`/`for_each_chunks_mut`（单数组 / N 个**等长**数组等分）；
`for_each_row_block_mut`/`_picked_work`（按行块切分、行宽可不同；后者并行门槛用显式 `work`，如
`gemv_f16`）。**约束与语义**：`MIN_PARALLEL_LEN = 4096`（元素总数低于该值**自动原地串行**；SOA 是各数组长
度之和）、`MAX_ARRAYS = 16`；**派活接口取 `&self`**（池内部用互斥把派活者串行化 ⇒ 多线程可同时拿同一个
池派活，进程级共享池靠这条成立；代价：重入派活改由运行期 panic 拒绝）；worker 闭包 panic 被
`catch_unwind` 拦下并计数、主线程收工后 `resume_unwind` **原样续抛**，池状态干净可复用；每次调用**零分
配**（作业槽定长数组）；`for_each_chunks_mut` 数组长度不一致 panic；`for_each_row_block_mut` 在
`row_gran == 0` 或数组长度 ≠ `rows×行宽` 时 panic；`row_gran` 不要省（块大小向上取整到它的倍数；实测
`row_gran=1` 并不更慢）。

`parallel` 策略层固定常见负载：`matmul_f32`/`matmul_f64`（多核矩阵乘、行粒度 4，B 只读共享；`&pool` 可
直接给 `pool::global()`）、`gemv_f16`（行粒度 1，与单线程逐位一致）、`rk4_j2_step_batch`（6 数组一次派
活）。**`parallel::*` 形状不符返回 `Err`**（同一套 `Error::Shape`/`Overflow`）；`matmul_*` 对 `m==0||n==0`
直接返回 `Ok`（不写 C），`k==0` 时 `c.fill(0.0)`。**池的低层 `for_each_*` 仍按契约 panic**——它是"调用方
自己保证形状"的原语，`parallel` 才是带校验的那一层。数值与单线程**逐位一致**（只切行/下标区间）。
`parallel::rk4_j2_step_batch` 的 worker 是复用线程，`lasx_force_lsx_thread` 对它无效。**线程数怎么选（实
测 + 运行期探测）**：中小规模取**物理核数**并**给主线程留一个空位**：`let p =
lasx_rs::arch::parallelism();`（`{ logical, physical }`，进程级缓存一次）⇒ `let threads =
p.physical.min(p.logical - 1).max(1);`。依据（本机 24 逻辑核 / 12 物理核）：① 块数 < 线程数的派活台阶已
修（改"逐槽代次 + 只等有活的 worker"后 37/38 µs → 1 µs；gemv 64×2048@24 线程 0.043→0.007 ms）；② worker
全有活且占满 CPU 时仍有调度代价（gemv 1024×2048 24 线程 0.078 vs 12 线程 0.041 ms）⇒ 少给线程、留一个空
位；③ **权重 ≥32 MiB 是例外**：24 线程仍更快（4096×4096 0.73 vs 1.07 ms）。完整表与对照见
`docs/dev.md`。池要复用：把 `WorkerPool::new` 放进热路径 = 每次新建线程；拿不定主意用
`pool::global()`（示例 `--example pool_axpy` / `--example matmul_pooled`，自带逐位对照）。

### 9.05 逆变换：Dubner–Abate + 跳变扣除

**做什么**：由 `F(s)`（调用方实现 `ComplexFn`）求 `f(t)`、`t ≥ 0`。**与正向的关系**：F 是**调用方代
码**、必须逐点求值；DSL 只负责"指数和"那部分（cos/sin 核 × F 值 → 两条实矩阵乘），且天然批量。**F 贵
时主导成本在 F。**

```text
f(t) = f₀·u(t) + L⁻¹{ F(s) − f₀/s },      f₀ = f(0⁺) = lim_{s→∞} s·F(s)   ← **调用方给**
f̃(t) ≈ (e^{a t}/T)·[ ½F̃(a) + Σ_{k=1..N} Re{ F̃(a + i kπ/T)·e^{i kπ t/T} } ],  ω_k = kπ/T
```

**为什么必须扣除跳变**：纯 D–A 级数周期 2T，而 `h(t)=f(t)e^{-at}` 在 `t=0/2T` 处不连续 ⇒ 系数按 `1/k` 衰
减 ⇒ 误差 `O(1/N)`。实测（`F=1/s`、`T=12`、`a=1`）：0.307(50) → 9.88e-2(150) → 4.98e-2(300) →
2.50e-2(600)，每翻倍只减半 ⇒ 要 1e-3 得 N≈15000。扣除 f₀ 后 `h̃(0)=0` ⇒ 偶/奇延拓连续 ⇒ 收敛恢复。

| N | **扣除 f₀=1** | 未扣除（f₀ 取错为 0） |
|---|---|---|
| 64 | 1.57e-2 | 1.14e-1 |
| 128 | 1.74e-3 | 1.20e-1 |
| 256 | 1.01e-3 | 2.98e-2 |
| 512 | **1.22e-4** | 2.96e-2 |

（`F=1/(s+1)`、`t∈{0.5,1,2,3,4}`）⇒ 扣除后 N=512 已到 **1.22e-4（≤1e-3 达标）**；未扣除时卡在 ~3e-2 的
1/N 地板。**f₀ 取错的后果就是这张表**（残留跳变 ⇒ 退回 1/N）——可用它自诊。**`f₀` 的契约**：**首选调
用方给**（`DubnerAbate::with_f0`；f₀ 是初值定理的解析量）；默认 `f₀ = 0`（不扣除，仅适用于 `f(0⁺)=0`
的 F）；`estimate_f0(f, s_probe)` 是**数值估计、不可靠**，要用必须**显式**传进 `with_f0`——**没有任何静
默回退**。**硬约束（`check` 会拦）**：① 周期 2T ⇒ 只在 `0 < t < 2T` 内代表原函数（建议 `T ≥ 3·t_max`）；
② `a·t_max ≲ 4…8`（实测 `a=6 → 1e7`、`a=12 → 1e17`：`e^{at}` 放大绝对截断误差）。**`t=0` 给跳变平均
值**（F=1/s 时 ≈f₀/2）——Fourier 反演的已知行为，不是 bug（有断言）。**位精确（已证成）**：指数和的 DSL
路径与行切参考**逐位一致**（f32 用 `api::matmul`、f64 用同序手工）。**交叉验证**：与独立路线（固定围道
Re s = c 的 Bromwich 梯形积分，无周期陷阱）结构一致（≤1e-1；该参考自身只有 ~5e-2 精度，精度靠解析对照
≤1e-3 保证）。**限制**：不能保证任意精度；F 的奇点须在 `Re s < a`；逆问题病态；`0 < t < 2T`；`t→0` 误差
抬升（Gibbs）；**依赖 f₀ 的正确性**。

### 9.06 复数矩阵乘（A 阶）

**表示**：**分平面 `(Re, Im)` 两块实数组**（`ar/ai/br/bi` 行主序；`cr/ci` 复用调用方缓冲）。**f32/f64 两
条路径**：`ops::cmatmul::{cmatmul_f32_into, cmatmul_f64_into}`（默认）与
`*_into_route(..., Route::{Direct, Karatsuba})`（显式选路线）。**A–C 阶不加 C 符号**（`nm -D` 保持
94）；**D 阶已按 `docs/c-abi.md` 正式追加 4 个符号**（`lasx_cmatmul_{f32,f64}` + `_checked` 孪生）⇒
`nm -D` = 98。

| 路线 | 实矩阵乘条数 | 位精确契约（谁守） |
|---|---|---|
| `Route::Direct` | **4**：`Re=Ar·Br−Ai·Bi`、`Im=Ar·Bi+Ai·Br` | **每条乘积与对应实矩阵乘逐位一致**（`direct_matches_four_real_matmul_bitwise`）；和/差按 IEEE 逐元素 |
| `Route::Karatsuba`（**默认**） | **3**：`k1=(Ar+Ai)(Br+Bi)`、`k2=ArBr`、`k3=AiBi` | **只承诺容差**（k1 多一次加法、组合次序不同）⇒ 不承诺与直算逐位（`routes_agree_within_tolerance`） |
| `Route::Fused`（**B1：正确性原型已过、寄存器化未完成**） | 1 个交错微内核（Gauss 3 乘；**组合当前仍标量**） | 只承诺容差；已过全形状正确性与容差测试 |

**B1 现状（未进 `Route` 枚举、不是默认、不改任何既有行为）**：`cmatmul_interleaved_f32/f64`（交错
`[re,im]` 布局 + Gauss3）**已跑通全形状正确性**；**寄存器化尚未完成** ⇒ **没有性能结论**，**默认仍是
Karatsuba**；下一步用"lane 对交换 + 交错累加器"避免去交织 shuffle。**边界矩阵（三路线共用）**：`k=1`、
`n=1`、`m/k/n` 不整除、**非 4 倍 rows**、**非 32 倍 n**、大方阵、零维——`Direct`/`Karatsuba` 已覆盖；
`Fused` 落地时必须把同一套矩阵原样跑一遍。**"完全 SIMD"的判据（2026-10-16 定级修正）**：这是
`lasx_rs` 这个专门后端自己的实现选择与局部保证，**不再是本库的硬要求**——引擎/可移植侧接受 LLVM 自动向
量化也算 SIMD（红线 A）；手写 intrinsic **保留、不删**（A 是政策，不是删除令）。A 的实测依据：
`lasx/idio` = dot K=4096 **0.473×（手写 2.12×）**、matmul 64³ **0.253×（3.95×）**、matmul 192×768×3072
**0.115×（8.7×）**，核构造 `build_kernel_f32` **0.996×（≈持平）** ⇒ 手写收益只在"向量内核算子"且随形状
放大；组合步性能实测收益仅 **0.19% ⇒ 不是瓶颈**（负结论）。**不变的项**：位精确承诺、`nm -D` = 98、默认
路径行为、冻结接口、CI 全绿、三端同步。

**同源性能（历史记录，口径见 `docs/dev.md` 性能章节）**：`192×768×3072` 中位 ms：Direct 63.13、
Karatsuba 53.35、单条实矩阵乘 15.04 ⇒ **选优默认 Karatsuba**（同源快 18%，代价是放弃逐位、只承诺容差 ⇒
要位精确请显式 `Route::Direct`）。**成本口径（别误读成"复数比实数快"）**：复数版**仍比"调用方自己调 k
条实矩阵乘"更贵**——Karatsuba 53.35 vs 3×15.04=45.1（贵 18%）、Direct 63.13 vs 4×15.04=60.2（贵 5%）⇒
复数 API 的价值是**一次调用 + 正确复数语义 + 可选逐位 + 后续融合改进空间**；"效率比"只在同一路线内比才
有意义。**线程数**：串行核心、未自动多线程；要多线程请按分平面切。**限制**：Karatsuba 的 `Ar+Ai` 极端量
级可能溢出（f32 尤甚）；分平面需 4 块输入/2 块输出；除直算路线的"各乘积逐位"外无与实矩阵乘同级的位精确
承诺。

### 9.07 DSL 复数表面（C 阶）

**表面**：`lasx_rs::cmatmul::{CMat, CMatBuf, CMat64, CMatBuf64}`——**形状进类型**（分平面 `(Re,Im)`）：
`let a = CMat::<2, 3>::new(&ar, &ai).unwrap();`、`let b = CMat::<3, 5>::new(&br, &bi).unwrap();`、
`let mut c = CMatBuf::<2, 5>::zeros();`、`a.mul_into(&b, &mut c, Route::Karatsuba);`（R×K · K×N ⇒ R×N，类
型强制）。**为什么不做 `cmatmul!` 宏**：宏只是语法糖；**编译期形状检查**由类型给出已足够 ⇒ 本轮选类型化
API。**编译期形状证据**：`CMat::<2,3>` 乘 `CMat::<4,5>` **编不过**——用 rustdoc 的 **`compile_fail`
doctest**（不引 trybuild、已进 CI 文档测试步骤）。**位精确契约**：`mul_into` 对**同一条 `Route`** 的结果
与直接调 `cmatmul_*_into_route` **逐位一致**（`Direct`/`Karatsuba` 各一次、f32/f64）；长度给错返回
`None`（不 panic）。**第一个用例：拉普拉斯正向**——把两条实乘改成一次复数乘（`(Kc+iKs)·(f+0i)`）：数学
同值、累加次序不同 ⇒ **只承诺容差**（与旧路线 <1e-4，解析三组仍过）。**⚠️ 同源实测：复数路线更慢 ⇒ 默
认仍用旧路线**（64×1024×8）：旧两条实乘 2.8783 ms、复数 Direct 3.3329（1.16×）、Karatsuba 3.3598
（1.17×）——实信号时 f 虚部为 0，两条实乘就是这份工作的下界 ⇒ `laplace_batch_*` 默认保持两条实乘，复数
路线作为**可选**（`laplace_batch_f32_complex_route`）。**DSL 复数的价值是表达力与形状安全，不是速
度。** **逆变换性能（C 阶重做）**：`J×N×NB = 256×1024×512`（NB 放大后矩阵乘成为主导）：只指数和
19.40 ms / 55.34 复数 GFLOP/s；含 F 求值（512 个 `1/(s+θ)`、524k 次）32.83 ms ⇒ **F 求值占 ≈40.9%**
（F 贵时主导成本在 F）。线程数：串行核心（未自动多线程）。

### 9.08 运算微积分：导数定理

`L{f′}(s) = s·F(s) − f(0⁺)`；`L{f″}(s) = s²·F(s) − s·f(0⁺) − f′(0⁺)`；
`L{f⁽ⁿ⁾}(s) = sⁿ·F(s) − Σ_{k=0}^{n−1} s^{n−1−k}·f⁽ᵏ⁾(0⁺)`。**API**：
`derivative_spectrum_f64(n, s_re, s_im, fre, fim, init)` + 特化 `derivative1/2_spectrum_f64`。输入是与
`laplace_batch_*` 输出**同一批 s 节点上的 F**（分平面）+ 调用方给的初值；输出同批节点的
`G(s)=L{f⁽ⁿ⁾}`。逐节点复数乘/乘幂（2×2 实乘加），没有新造矩阵乘。**初值契约（谁守：调用方；库只做逐节
点算术）**：首选调用方给（解析式/已知量）；**不做任何数值估计、无静默回退**（与 §9.05 的 f₀ 同规）；
必须是**右极限**；`init.len() ≥ n`（断言）。**取错的后果（实测）**：残差 `= ε·s`——在 `s∈{2,8,32}` 给
`f(0⁺)` 注入 ε=0.01，残差 **0.02/0.08/0.32**（与 ε·s 逐位吻合、相对 1e-12）⇒ 大 |s| 处按 |s| 放大。**解
析对照**（3 组，实部+虚部，8 个 s 点）：`e^{−2t}`⇒`4/(s+2)`、`sin(3t)`⇒`−27/(s²+9)`、`t`⇒0，相对误差
**<1e-12**（t 的实部逐位 0）。**端到端往返**（正向 F → 导数定理 → 库内逆变换 → f″，`f=e^{−t}`，
`n=512, f0=1`）：t=0.5/1/2/3 ⇒ **8.6e-5 / 1.0e-4 / 4.7e-4 / 9.7e-4**。**限制（如实）**：① **阶跃/含 δ 的
情形不适用**（`f≡1` 时 `f″=δ′`，定理给 0——错；`derivative_theorem_step_counterexample` 钉住；要处理 δ
必须用广义函数，本模块不做）；② 要求 f 在 `[0,∞)` 足够次可微且指数阶；③ **浮点相消**：`sⁿ·F(s)` 在 n
大、|s| 大时灾难性相消——经验界 `n ≤ 3 且 |s| ≤ ~1e3` 可忽略，更大范围自行核算（不给"安全区"承诺）；
④ 只对初值正确性负责到**算术**层面（初值错 = 静默错结果，不是 UB）。

### 9.09 把 `s^k` 折进核 vs 算完 F 再逐节点乘

**(A) 后乘（默认，可选）**：`laplace_batch_*` 算 F ⇒ `derivative*_spectrum_*` 逐节点乘 `s^k` 再减初值
项；**(B) 折进核**（k≥1 可选）：`laplace_batch_{f32,f64}_weighted(..., k)`，核
`Kc[j,i] = w_i·dt·Re(s_j^k e^{−s_j t_i})`、`Ks[j,i] = w_i·dt·−Im(s_j^k e^{−s_j t_i})`（与 `kernels64`
同符号约定 `Im F = −Ks·f`），`s_j^k` 逐节点复幂。**分工**：核只管积分里的 `s^k·F`；**导数定理的初值项不
在积分里** ⇒ 仍由 `derivative*_spectrum_*` 逐节点减。两者合起来才是 `L{f⁽ⁿ⁾}`。契约：
`k = 0` 的加权入口与 `laplace_batch_*` **逐位一致**（`weighted_k0_bitwise_same_as_plain`，f32/f64，
`to_bits()`；实现上直接委派）；`k ≥ 1` 折进核 vs 后乘**不逐位一致**（舍入次序不同）⇒ 只承诺**舍入级差
异**（`folded_vs_postmultiply_difference_is_roundoff`：最大绝对差 8.88e-16、最大相对差 6.62e-16，并断言
"不逐位"）；三组解析（k=1/2）与解析 `s^kF` 一致（`weighted_k1_k2_three_analytic_cases`）；端到端往返与
解析 f″ 比（`weighted_round_trip_through_inverse`：8.6e-5/1.0e-4/4.7e-4/9.7e-4）。**性能（预判 + 待
测）**：核构造 `O(J·N)`、矩阵乘 `O(J·N·NB)` ⇒ 小 NB 时两者同量级（收益在此）、NB 大时矩阵乘主导（折进
核没用）。**折进核买到的是语义整齐，不是速度**；默认仍取后乘（A）。**限制**：折进核让**核元素量级随
|s|^k 放大** ⇒ 动态范围变大。实测（`|s|≈112`、dt=1/64、N=64）：核元素最大量级 k=0 为 7.81e-3、k=2 为
7.81e1（放大 1.0e4×）⇒ 经验界：**`|s|^k·dt` 与 f32 有效位数一起决定"折进核"能否用在 f32**；
`|s| ≳ 1e2 且 k ≥ 2` 时优先 f64（或改用后乘）。

### 9.10 超越函数 recon：核构造的三条路线（原型、opt-in）

**背景**：拉普拉斯核构造是 `O(J·N)` 的逐元素 `exp/cos/sin`；DSL 只有类型级形状、**没有取值范围信息** ⇒
想"编译期分析查表精度与范围"必须配**显式范围声明**，否则是空话。本轮**只做 recon 与测量原型，未写任何
DSL 语法**。三条路线：**(a) 基线**：`libm` 逐元素（现有 `laplace::kernels64`）——**位精确基准**，硬承诺不
变；**(b) 表 + 多项式**（`fast_trig::{fast_exp, fast_cos_sin}`）：`const fn` **编译期生成**的
`static COS_SIN_TAB`（`[0,π/2]` 129 点）+ 2 项 Taylor 修正；`exp` 走 `2^i·Taylor(f)` 位缩放；**(c) 等距递
推**（`fast_trig::recurrence_row`）：`e^{−σ i Δt}` 逐次乘、`cos/sin(ω i Δt)` 旋转递推。**精度（对 libm
基线，J=64×N=1024、核峰值 1.302e-3；掩码 = 只统计 `|ref| > 0.1·峰值`）**：

| 路线 | 最大**绝对**误差 | 掩码**相对**误差 | 掩码 f32 ULP 差 |
|---|---|---|---|
| (b) 表+多项式 | 2.37e-10 | **1.39e-6** | 16 |
| (c) 等距递推 | 1.36e-17 | **5.47e-14** | **0** |

**速度（历史记录，口径见 `docs/dev.md` 性能章节；同进程交替 6 对丢首对）**：核构造 ms（J=64×N=1024）：
(a) 1.8046（1.00×）、(b) 2.6240（**1.45× 更慢**）、(c) **0.3524（0.20×，快 5.1×）**；核构造占整条
62–77%，但"改后"是**推算**（三段独立测量、非同一次交替 A/B），**未落地为默认**。**契约（两条必须守）**：
① **快速超越函数必须显式 opt-in，默认仍 libm**；与 libm **不逐位一致**（写入例外清单，最大误差界：
**(b) ≤1.4e-6 相对**、**(c) ≤5.5e-14 相对（本网格）**）；② **共享表** `static COS_SIN_TAB`：`const fn`
编译期生成、零运行时初始化、每（类型,精度）一份、只读 ⇒ 多线程只读安全（无 `static mut`/
`UnsafeCell`）；大小 129 项 × 16 B = 2064 B（2 KiB），常驻 L1（本机 32 KiB/核）。**限制**：(c) 的误差随
N 与 ω·Δt 累积（旋转递推）；安全区与
回退判据见 §9.10.1；表要按显式范围声明选长度与归约区间。

#### 9.10.1 递推核已落成可选入口 + 稳定边界 + 真 A/B

**入口**（**opt-in，默认仍 libm**）：`laplace_batch_{f32,f64}_recurrence(sig, om, dt, f)` →
`(Re, Im, all_safe)`，与 `laplace_batch_*` 同接口（同 s 节点、同分平面、同批量）。**稳定边界扫描（误差 vs
ω·Δt × N；掩码同上）**：

| `ω·Δt` | N=64 | N=256 | N=1024 | N=4096 |
|---|---|---|---|---|
| 0.01 | 3.06e-15 | 7.59e-15 | 7.59e-15 | 7.59e-15 |
| 0.1 | 8.22e-15 | 2.57e-14 | 2.57e-14 | 2.57e-14 |
| 0.3 | 1.60e-14 | 1.88e-14 | 1.88e-14 | 1.88e-14 |
| 1.0 | 1.98e-14 | 4.41e-14 | 4.41e-14 | 4.41e-14 |
| π/2 | 3.64e-15 | 1.28e-14 | 1.28e-14 | 1.28e-14 |

⇒ **安全区比预想宽得多**（全网格 ≤4.4e-14），不是窄。**安全区判据（代码里显式）**：
`RECURRENCE_MAX_PHASE_STEP: f64 = 1.5`（`|ω|·Δt` 上限）、`RECURRENCE_MAX_PHASE: f64 = 8192.0`（
`|ω|·Δt·N` 上限）；`pub fn recurrence_safe(om, dt, n) -> bool { let step = om.abs()*dt; step <=
MAX_PHASE_STEP && step * n as f64 <= MAX_PHASE }`。**阈值处有回退路径**：`kernels64_recurrence` **逐行**
检查 `recurrence_safe`，任一行不安全就**该行**改用 `libm` 核（不是整批失败），并把 `all_safe=false` 返
回。测试 `recurrence_entry_falls_back_outside_safe_zone` 断言：安全区内递推 vs libm ≤1e-4；**超安全区
（ω·Δt=2.0）时结果与 libm 路径逐位一致**（回退可验证）。**真·同源交替 A/B（同进程、≥5 对、丢首对；
M=64,N=1024、全部行在安全区内）**：

| 情景 | libm 核 ms | 递推核 ms | 比值 | 实测 |
|---|---|---|---|---|
| 小 NB=2 | 2.1846 | **1.2353** | **0.57×** | −43.5% |
| 大 NB=64 | 1.5754 | **0.6455** | **0.41×** | −59.0% |

（iters 23/32、ms_max 2.285/1.687 ms；本轮是**实测交替**，取代上节推算表。）**例外清单措辞（新增第 3 条
例外）**：快速超越函数必须显式 opt-in（默认 libm）；`laplace_batch_*_recurrence` 结果与 libm 路径不逐位
一致；最大误差界：安全区内**掩码相对误差 ≤4.4e-14**（本表网格），随 ω·Δt/N 变化；超出安全区自动逐行回
退 libm（返回 `all_safe=false`）。建议：`ω·Δt ≤ 1.5` 且 `ω·Δt·N ≤ 8192` 时用递推，否则用 libm。

## 10. 对齐缓冲与 FFI

Rust 侧用 `AlignedVec`（纯 Rust，不导出符号）：`AlignedVec::new(len)` / `fill_with(len, f)`（分配 `len`
个零值元素、首地址 64 字节对齐 / 分配并用 `f(i)` 填充）；`len`/`is_empty`/`as_slice`/`as_mut_slice`/
`as_ptr`/`as_mut_ptr`（切片视图与保证对齐的裸指针，可交给 C ABI）；`Deref`/`DerefMut`/`Clone`（可直接当
`&[T]`/`&mut [T]` 用；克隆后**重新计算对齐偏移**并复制数据）。**C 侧 `lasx_alloc`/`lasx_free`**（32 字节
对齐、未初始化；必须用 `lasx_free` 按同一个 `n` 归还；`free()` 不是 ABI 契约；`p == NULL` / `n <= 0` 是
安全 no-op）**见 [`docs/c-abi.md`](c-abi.md) 第 4 节**——那是这条契约的唯一权威。对齐只影响性能、不影响
正确性：LASX 的 `xvld/xvst` 是 32 字节访问，起始地址不落在 32 字节边界时会跨缓存行；实测 glibc malloc 与
Dart FFI 缓冲区只有约一半落在 32 字节边界上，**不要假设缓冲区天然对齐**；未对齐时结果照样正确、且仍明
显快于 LSX 路径。想稳定拿到对齐内存：Rust 用 `AlignedVec`，C 用 `lasx_alloc`（配 `lasx_free`）或
`posix_memalign(&p, 32, n)`。**C/C++/Dart 的完整用法**（四种链接方式与最小可运行程序、Dart 的
`lookupFunction` 例子、长度/生命周期/别名规则、裸 vs `_checked`、能力探测顺序）见
[`docs/c-abi.md`](c-abi.md)（第 2 节快速开始、第 4 节内存、第 5 节安全模型、第 6 节能力与降级）；Rust 侧
调用方式见 §8 与 §9。

## 11. NN 算子：当前状态与候选

### 11.1 ggml 算子全景与缺口对照

`ggml.h` 的 `enum ggml_op` 共约 102 个算子（一元/激活、逐元素二元、归约、矩阵、形状/布局、归一化、注意
力、卷积/池化、索引/嵌入、SSM/线性注意力、训练）；一次前向每层固定出现 MUL_MAT、RMS_NORM×2、
ROPE×1~2、SOFT_MAX（或 FLASH_ATTN_EXT）、SILU/GELU、ADD×2~3、MUL×1~2，解码时还有 GET_ROWS；量化格式热
档 q4_0（4.5 bit/权重）、q8_0（8.5）、q4_K（4.5）、q6_K（6.5625）、q2_K/q3_K/q5_K、mxfp4/nvfp4、iq*
（1.5–4.5）。**已落地、可直接复用**：`lasx_matmul`/`lasx_matmul_f64`、`lasx_dot_q4`、`lasx_dot_i8`、
`lasx_dot`/`lasx_sum`/`lasx_dot_f64`/`lasx_axpy`、`lasx_norm3_batch` 及姿态/几何 7 个、`pool`/`parallel`
多核设施、`api` 安全层。**缺口**（"llama.cpp 现状"是 llama.cpp 在本机的实测；"本库现状"是本库复测读
数）：

| 缺口 | llama.cpp 现状 | 本库现状 | 结论 |
|---|---|---|---|
| f16 GEMV（decode） | 6.8 GB/s | `lasx_gemv_f16` 3.75–48.2 GB/s（同量级形状 1024×2048 = 10.5 GB/s ≈1.5×） | 小形状驻留缓存时领先明显；超出 L3 后回落到 3.75 GB/s，瓶颈是 f16→f32 转换与 permi_q 重排两项结构代价 |
| f16/bf16 GEMM（prefill） | 8.4 / 16.6 GFLOP/s | f32 GEMM 60–68 GFLOP/s，**没有 f16 GEMM** | 仍是未做缺口（N1 只做 f16 的 GEMV/dot） |
| `SOFT_MAX` | 典型 2.4–4 GB/s | 5.22–7.00 GB/s（朴素 Rust 的 4.5–5.6×） | 已落地，1.3–2.9× |
| `SILU`/`GELU`（激活） | 无 LoongArch 分支 | 4.55–9.20 GB/s（朴素 Rust 的 4.1–7.9×） | 已落地三个入口；**算力受限**（瓶颈是 exp + 向量除法） |
| `RMS_NORM`/`ROPE` | llama.cpp 侧 ≈1×（无缺口） | rms_norm 12.0–18.4 GB/s、rope(NeoX) 28.6–71.9 GB/s | 本库把这两者也 LASX 化了（llama.cpp 的 rope 两种 mode 都是标量） |
| `q6_K` / `mxfp4`/`nvfp4` | 9.8 / 5.5 / 4.0 GB/s | 未做 | ~1.8×（N2，需 GGML 块格式互操作）/ ~3–4.5×（N2） |
| 已经没机会的 | q4_0/q4_K/q2_K/q3_K/q8_0/iq*、f32、ADD | — | ≈1×：llama.cpp 的 `arch/loongarch/quants.c` 已 LASX 化 |

三条结论：① 量化点积**不是机会**（llama.cpp 已把多数格式 LASX 化）；② 真正的短板是**非量化**那半边
（f16/bf16 与 softmax 在 llama.cpp 里没有 LoongArch 分支；本库已补齐并给出实测）；③ 剩下的缺口集中在
**f16/bf16 GEMM（prefill）** 与 **N2 的 GGML 量化格式互操作**。

### 11.2 候选批次

**N1（纯 f32/f16，无格式锁定；已落地）**：`dot_f16`/`gemv_f16`（§2.11 契约、§5.13 用法：转换穷举验证、
4 条累加链）、`softmax_rows`（§2.6/§5.9）、`rms_norm`（+权重，§2.7/§5.10）、`silu`/`gelu`
（quick/erf）（§2.8/§2.9、§5.11）、`rope`（NeoX/Gpt-J 两 mode，§2.10/§5.12；NeoX 走 LASX、GptJ 标量）。
**N2（GGML 格式互操作，按需）**：`quantize_rows_q8_0`（Q8_0 块 + f16 scale）、`dot_q4_0_q8_0`/
`gemv_q4_0`（Q4_0 块布局 18 B/32，与现有 `dot_q4` 布局不同）、`dot_q6_K_q8_K`（对应 1.8× 那一档）。N2 的
价值是"与 llama.cpp/GGUF 生态互通并可交叉验证"；若只给自己的推理代码用，N1 就够。**N2 的唯一具体消费者
（loong-llm 的 8 处 `q8_matmul`）已不再作为下游目标** ⇒ 这批维持"未做、按需"：没有调用点就没有验收对
象。真要启动的触发条件："要在龙芯上跑 GGUF 权重并与 llama.cpp 交叉验证"。**N3（int8 推理，消费者明确；
已落地）**：消费者是一个 0.2B 量级的**判别式决策模型**（cross-encoder，非生成式、无 KV cache、序列
≤192 token、端侧 CPU）。与 N1/N2 的关系：N1 是 f32/f16 且已落地；N2 是 GGML 块格式互操作；N3 是**自用
int8 推理路径**，格式与消费者都不同、互不替代。

| 算子 | 说明 | 状态 |
|---|---|---|
| `lasx_amax` / `lasx_absmax_rows` | 最大绝对值（逐张量/逐行） | **已落地**（§2.12 契约、§5.14 用法） |
| `lasx_quantize_i8_per_tensor` | f32→int8 + scale，对称（qmax=127） | **已落地**（§2.12） |
| `lasx_quantize_i8_per_row`（= per-token/per-channel） | 逐行各自求 scale：激活 `[token,hidden]` 是 per-token、权重 `[out,in]` 是 per-output-channel（**同一内核**，`api` 按意图分两个名字） | **已落地**（依据：per-tensor 使成对准确率 96.1%→93.4%，per-token/per-channel 无损） |
| `lasx_dequantize_i8` / `lasx_dequantize_i8_rows` | int8 + scale → f32（逐张量/逐行） | **已落地**（§2.12） |
| `lasx_gemv_i8` | 权重 int8 × 激活 int8 → i32 精确累加；出口 `(acc as f32)·(sw[o]·sx)`——**两个 scale 先乘成一个** | **已落地**（§2.13 契约、§5.14 用法） |
| 整数/定点 softmax | 多项式或查表近似 exp | **不做（实测）**：组合路径整层只占 int8 GEMM 的 **~2%** ⇒ 省下的上限就是 2%，不值得引入精度风险 |
| int8 `rms_norm`/激活 | 归一化与激活的 int8 版本 | **不做（实测）**：`192×768` 的"反量化 + RMSNorm + 再量化" = 367.3 µs、占该层 GEMM 的 ~0.3% ⇒ **组合路径就是答案** |
| `parallel::gemv_i8`（行切分） | 多核 GEMV | **不做**：按 m 行切分对"单 token"（m=1）零并行度；m 大的档位 f32 GEMM 已赢 |
| `parallel::gemv_i8_k`（**k 切分**） | 多核 GEMV，m=1 也能用 | **已落地**：把"块"当"行"交给现有派活接口，整数截断同态 ⇒ 与单线程**逐位一致**；**12 线程 5.4–6.9×**；门槛每块 ≥1024 且 ≥4 块（更小是负收益） |
| `matmul_i8` | 批量 int8 矩阵乘（prefill） | **已落地但实测否掉立项理由**：对逐 token 跑 GEMV 是 **1.00×**、比 f32 `lasx_matmul(n=192)` **慢 2.5–2.7×** ⇒ 目标形态选 f32 GEMM；int8 只在"权重装不进缓存、必须流 DRAM"时占优（~4× 流量） |

**可复用**：`lasx_dot_i8` 的"i16 乘积 → i32 累加 → 每 1024 字节落 i64"结构（`gemv_i8` 直接复用）、
`lasx_dot_q4` 的"组内整数点积 + 组 scale 乘"分层与 `xvhaddw` 水平加宽归约、`lasx_matmul` 的打包 + k 分块
+ 列块骨架、对齐分配。**验收（沿用 §2.5 纪律）**：位精确（向量路径与标量参考逐位一致；`ops::quant_i8` 7
个测试覆盖 n 取 0..1000 与 NaN 混入；`gemv_i8` 与精确 i64 参考逐位一致；LSX 不适用）、量化误差可解析界
定（实测最坏 ≤ `scale/2`，999 随机样本）、量化实现正确性单独断言（满量程 `max|q|≥120`、边界打到 ±127、
反量化逐位等于 `(q as f32)·scale`）、性能按进程级交替 A/B ≥3 轮（端到端 10–20 ms 的立项目标已被实测推
翻：只算了权重流量、漏了序列里每个 token 都要过一遍权重 ⇒ 目标已改成分档）、C ABI 只追加不修改、给出
per-tensor/per-token/per-channel 三种模式的误差对照。**不建议做**：q4_0/q4_K/q8_0/IQ* 的 vec_dot 重写
（llama.cpp 已贴带宽/算力上限）；`GET_ROWS`/`VIEW`/`PERMUTE` 这类纯搬运算子（带宽决定，SIMD 无事可做）。
复现与局限：llama.cpp 的 `test-backend-ops perf` 是**单线程**、合成形状；真实推理有 OpenMP 多线程与 KV
cache 访存模式，绝对数字会不同（相对结论更可靠），且 `RMS_NORM`/`SILU`/`MUL`/`SCALE` 在该版没有 perf 用
例、只能按同族算子推断。

## 12. 无损压缩与带宽换算力：结论与路线

### 12.1 结论

① **通用无损压缩在本库数据上收益远比直觉小**：单次快照（内核一次调用吃的那 6 个 f64 数组）最好约
**1.22×**（zlib-6 + 字节平面）/1.15×（lzma 原生）；轨迹（64 帧）也只有约 **1.40×**，不是 2–4×。② **放进
内核热路径不可行**：本机 zlib-6 解码只有 **0.37 GB/s**，而 RK4 单线程内核消费 **3.3 GB/s**——差约 9 倍；
lz4/zstd 也一样，且这三个库在 loongarch64 上没有 SIMD 路径、只会更慢。③ **"牺牲算力换带宽"的方向本身是
对的**，但能换来带宽的不是通用压缩，而是三件已实测验证过的事：**直接吃紧凑表示**（`lasx_dot_q4` 用
4-bit nibble，带宽是 f32 的 1/8）、**重算代替存储**（融合内核 `lasx_rk4_j2_step_batch` vs 用原语拼装，
中间量从不落内存）、**分块提高复用**（matmul 分块，B 的流量下降、逐位不变）。

### 12.2 判定式：压缩什么时候能赢

**A. 链路上限场景**（网络/磁盘/跨进程传输）：收益 = 压缩比 r，解码只要快过链路速率即可（一般轻松满
足）。此时 1.2–1.4× 有意义：1 Gbps 链路上 1.4× 就是少传 29% 的时间。**B. 内核内场景**（数据已在内存，要
省内存带宽）：原始 `时间 = B/C`（内核消费速率 C），压缩后 `时间 = B/D`（解码速率 D）；要赢需 `D > C`。
**在内核里压缩比 r 几乎不出现，解码速率必须超过内核自身的消费速率。** 本机关键数字：RK4 单线程 C ≈
3.3 GB/s、12 线程聚合约 25 GB/s（DRAM 上限）、zlib-6 解码 D ≈ 0.37 GB/s——出局。B 场景要成立必须自己写
SIMD 解码器（定长位平面打包 + 移位），而且还要面对更致命的问题：**快照压缩比只有 1.2×**，即便解码无限
快也只是 1.2×。

### 12.3 路线

| 阶段 | 内容 | 判据 |
|---|---|---|
| **P0** | 判定式 + 实测 + 反例登记 | 已完成 |
| **P1** | **无损容器**（可选模块，零依赖自研）：面向存储/传输的轨迹、检查点、跨进程/跨语言边界；字节平面 + 定长位平面打包 | 比 ≥1.3×（目标 1.4×），解码 ≥链路速率；**明确不进内核热路径** |
| **P2.1** | matmul 改 8 行×16 列分块 | B 流量下降；**逐位一致**，无精度代价 |
| **P2.2** | RK4 中间量的重算/融合 | 把 6 数组工作集从"读+写 × N"压到更少遍 |
| **P2.3** | 把 `dot_q4`/`dot_i8` 的"直接吃紧凑表示"推广到更多算子 | **有精度代价，需授权**；明确每算子的精度-带宽曲线 |
| **P3** | 反例登记：**不做**"压缩→解压→算"的内核路径 | 除非自研 SIMD 解码 >3.3 GB/s 且比 >2×，否则不再评估 |

P2.1 是唯一"零风险、纯收益"的一项（逐位不变），建议作为下一步。**被否掉的做法**：通用无损压缩进内核热
路径（解码速率差 9 倍、压缩比只有 1.2–1.4×）；帧间 XOR + 字节平面（轨迹上反而没有帮助，相关性不够
强）；把浮点当均匀字节估收益（随机 f32 也能压到约 0.816，但只是"少传 18%"、不是数量级）；木桶模型只对
"多线程 + DRAM 到顶"成立（1 线程时短板是算力，受 `xvfdiv_d` 除法吞吐限制，此时"牺牲算力换带宽"是反
的）。**待定**：P1 的无损容器放哪（`lasx_rs` 内的可选模块 / 独立 crate）；P2.3 的精度换带宽授权边界（哪
些算子允许有损、误差上限多少）；是否需要面向 **FFI 边界**的紧凑传输格式（Dart/脚本侧数组转换占传播的
绝大部分时间，那是 CPU 转换开销，紧凑表示能同时减小转换量与带宽）。

## 13. 构建、测试与常用命令

工具链：nightly Rust（含 `stdarch_loongarch`）、`edition = 2021`、零依赖、
`[lib] crate-type = ["cdylib", "rlib"]`；`.cargo/config.toml` 设
`rustflags = ["-C", "target-feature=+lasx"]`（缺它也能编、也位精确，只是慢）。安装：
`rustup toolchain install nightly-loongarch64-unknown-linux-gnu`。

```bash
cargo build --release                 # 构建库（liblasx_rs.so + rlib）
cargo test --release                  # 单元测试 + 文档测试（需 LoongArch 真机）
cargo build --workspace --release     # 连基准 CLI（lasx_bench）一起构建
cargo run -p lasx_bench --release                 # 全部基准套件
cargo run -p lasx_bench --release -- matmul       # 只跑组名含子串的套件
cargo run -p lasx_bench --release -- plan         # 计划复用三列对照
cargo run --release --example pool_axpy           # 多核示例（自带逐位对照）
cargo build --release -p lasx_yll && cp target/release/liblasx.so yll/   # YouLiLong 扩展
```

基准套件组名（过滤子串）：`dot`、`sum`、`axpy`、`dot_f64`、`dot_i8`、`dot_q4`、`matmul`、`plan`、
`softmax`、`rms_norm`、`activation`、`rope`、`f16`、`int8_gemv`、`int8_quant`、`attitude`、`large`、
`norm3`、`vec3`、`distance2d`、`j2`、`ballistic`、`rk4`、`fma`、`mt`、`align`、`dispatch`、`scenario`。
方法学与全部读数详见 `docs/dev.md`。**`lasx_yll`（YouLiLong 原生扩展）的覆盖面——已审计**：它暴露的是
**精选子集**，`yll/lib.ylh` + `yll/src/funcs.rs` 共 10 个脚本函数（`dot`、`sum`、`dot_i8`、`matmul`、
`norm3`、`distance2d`、`vec3_add_scaled`、`j2_accel`、`rk4_step`、`propagate`），每个都走
`ffi::checked` + 参数级校验。**N1 批次的 8 个与 N3 批次的 8 个都刻意没有进**——脚本语言的调用面由实际脚
本需求驱动，不是"C 有就必须有"。要新增一个脚本函数：`funcs.rs` 写包装（参数校验 + `*_checked` + 返回值
构造）、`lib.ylh` 补声明、`cargo build --release -p lasx_yll` 后拷 `liblasx.so`。导出符号自检（期望行数
= [`docs/c-abi.md`](c-abi.md) 记的计数，本文不复述）：`nm -D --defined-only
target/release/liblasx_rs.so | awk '$2=="T" && $3 ~ /^lasx_/ {print $3}' | sort`。

## 14. Caveats 与限制

1. **降级覆盖不全**：历史 7 个 LASX-only 内核（`matmul`、`matmul_f64`、`axpy`、`sum`、`dot_f64`、
   `dot_i8`、`dot_q4`）在 LSX-only CPU 上会 SIGILL、不会自动降级；N1/N3 与后追加的 `matmul_scaled`、
   `layer_norm`/`gather_rows*` 同样按 §3.2 分类。C 侧自我保护（先问 `lasx_has_lasx()`）见
   [`docs/c-abi.md`](c-abi.md) 第 6 节。
3. **两处不逐位一致的例外**：`ballistic_step`（`|v|²` 结合序不同，物理一致，有绝对误差测试守护）、
   `axpy`（向量/标量尾至多差 1 ulp，FMA vs 分离乘加，无逐位守护）。
4. **`lasx_free` 的 `n` 必须与分配时一致**：契约与两种 no-op 见 [`docs/c-abi.md`](c-abi.md) 第 4 节（`n`
   传错——非 0 且与分配时不同——会破坏堆，**只有约定**，违反是 UB）。原始符号长度参数无负数/零防护：C
   侧安全模型见 [`docs/c-abi.md`](c-abi.md) 第 5 节（裸 = 零校验、误用即 UB；要错误通道用 `_checked`）。
5. **f32 矩阵乘无 f64 落盘**：`lasx_matmul` 在 f32 累加器内直接累积，长 k 维误差大于点积；高精度用
   `lasx_matmul_f64`。`dot` 的小规模路径：`n < 24` 用 f32 单精度累加，跨 `n=24` 不保证逐位连续。
6. **`dot_q4` 的 scale 长度约定**：`sa`/`sb` 长度必须 ≥ `ceil(n_bytes/32)`，按 32 字节组对齐；非 32 倍
   数时最后不足组按 `j/32` 索引。`dot_i8` 结果截断：内部 i64 累加精确，返回 `as i32` 会回绕；业务上 n
   应远小于约 13.3 万。
7. **检测缓存与钩子作用域**：LASX 检测依赖 `cpucfg` + CFG2.bit7，结果进程级缓存一次、之后不可更新（虚
   拟化/模拟器中 cpucfg 不可信则检测不可信）；`FORCE_LSX` 线程级、默认 false，测试必须自行复位；池化路
   径的 worker 不受调用线程钩子影响。
8. **对齐与内存布局**：向量加载/存储用偏移 0，LoongArch 允许非对齐访问，源码未显式声明对齐要求；SOA/
   数组布局假设连续无填充，跨语言调用方必须保证元素类型与步长严格匹配（f32=4 字节、f64=8 字节）；C 侧
   对齐与别名规则见 [`docs/c-abi.md`](c-abi.md) 第 4 节。

## 15. 待确认与文档约定

本文由旧四份文档合并重写，凡旧描述与**当前源码**不符一律以源码为准。已修复：
`lasx_quat_to_dcm_batch_checked` 的 9 个输出数组改为逐个 `checked_slice_mut` 校验（补测试
`test_quat_to_dcm_null_output_is_reported`）；根 `README.md` 的"逐位确定"口径按本文 §2.1 收窄。**仍需外部
信息**：ggml / llama.cpp 的算子统计（约 102 个算子、调用次数分布）来自仓库外 `/home/lik/llama.cpp`，无法
在仓库内复核，只当"缺口依据"引用。

### 15.1 文档约定：数字来源、测量口径、引用格式

| 内容 | 唯一权威位置 |
|---|---|
| 契约（语义、结合次序、op 序列、精度包络、**谁守**、边界行为） | 本文 §2.5–§2.11（总精度一览在 §2.3） |
| 用法（签名、示例、约束） | 本文 §5、§6、§7 |
| **全部性能数据**（单线程内核表、大集占比、真实场景、多线程扩展） | `docs/dev.md` 的性能章节 |
| 编译期常量与阈值的 A/B 对照、设计推导与被否掉的方案 | `docs/dev.md` 的对应设计小节 |
| 与 llama.cpp 的缺口对照 | 本文 §11.1（只保留对照结论） |

本文正文不放性能表，只给"一行结论 + 指针"（同一个数字存在两处，必有一处先过期，读者无法判断哪处是新的）。
**测量口径**（性能章节每张表都按此产生）：内核表**绑核**测（`taskset -c 4`）、线程表**不绑核**测（绑核会
把 12 worker 挤到同一物理核，测到的是上下文切换而非扩展性）；每个测点进程级交替 A/B、≥3 轮、丢弃首轮、
取中位数；套件源码不写死读数（数字由当次运行算出）。
**引用格式**：跨文档引用一律写 `docs/xxx.md §N`（带文件名）、本文内部写 `§N`；引用写到最小可解析粒度，
`§` 编号是稳定标识、重构时保持编号不变（标题文字可改）。

## 16. 作为共享库 / C 项目使用（构建、安装、SONAME 与 ABI 规则）

**已搬移**：构建产物（`cdylib`/`staticlib`/`rlib`）与安装脚本（`scripts/install.sh --prefix=…`，失败即退出
+ 装前校验含 SONAME 核对）、四种链接方式、`examples/c/` 实测输出、SONAME `liblasx_rs.so.0` 的来源与自
检、`_checked` 变体的调用约定与失败行为——见 [`docs/c-abi.md`](c-abi.md) 第 2.1–2.4、3.7、3.8、4、5、9 节。

### 16.5 ABI 兼容规则（什么时候可以动 major）

**已搬移**：只追加不改、major 变更条件、要同改的三处、0.x 的语义与"不同 major 并存"——见
[`docs/c-abi.md`](c-abi.md) 第 9 节（那是 C ABI 承诺的唯一权威；C 调用者心智模型见该文档第 1–9 节）。
