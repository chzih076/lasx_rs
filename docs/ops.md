# lasx_rs 算子与用法手册

> 本文是 `lasx_rs` 的「算子与用法」手册：逐个内核的语义、数值约定，以及 C ABI / Rust API /
> 多核池 / 对齐缓冲的用法，另含以 ggml 为参照的 NN 算子现状与无损压缩路线。
>
> 本文只讲"算子是什么、怎么用"。**架构细节、基准方法学与全部性能数据（含 A/B 实验记录）
> 归 `docs/dev.md`**；正文出现的个别读数都注明"详见 docs/dev.md"，不是本文的结论来源。
>
> 库形态：LoongArch64 LASX 256 位 / LSX 128 位批量数值内核，零依赖（仅 std +
> `#![feature(stdarch_loongarch)]`，需 nightly），`cdylib + rlib`。


## 1. 分层与调用方式

| 层 | 给谁用 | 形态 | 校验 |
|---|---|---|---|
| `lasx_rs::ffi`（45 个 `extern "C"` 符号） | C / Dart 等 FFI 调用方 | 裸指针 + `int` 长度 | 原始符号零校验（误用即 UB）；`_checked` 带 `int *status` |
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

分两层，不要混为一谈：

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
- 三条路径（LASX / LSX / 标量尾）在设计上都同式同结合序，因此实际上也一致
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

| 内核类别 | 累加方式 | 误差量级 |
|---|---|---|
| f32 点积/求和 | 每 256 元素把 f32 累加器落盘、升 f64；最终一次 `as f32` | 相对 f64 参考 <1e-5 |
| f32 矩阵乘 | 每个输出元素沿 k 的**单个 f32 累加器**（无 f64 落盘） | 相对 f64 参考 <1e-4 |
| f64 内核 | 全程 IEEE 双精度，FMA 单次舍入 | <1e-12（matmul_f64）/<1e-9（批量物理） |
| int8 点积 | 整数精确（i16 乘 → i32 累加 → i64 落盘），返回截断为 i32 | 精确（mod 2³²） |
| Q4 点积 | 组内整数精确；scale 为 f32×f32 再升 f64 | scale 精度约 2⁻²⁴ |

### 2.4 退化输入与边界约定

- 零向量单位化（`unitize3_batch`）：`|v| == 0` 输出 `(0,0,0)`，不是 `0·∞ = NaN`。
- 退化四元数（`quat_normalize_batch`，以及内部先单位化的 `quat_rotate`/`quat_to_dcm`）：
  `|q| < 1e-15` 输出单位四元数 `(1,0,0,0)`，不是 NaN；单位化用 `q·(1/|q|)`
  （只做一次除法再乘，见 `docs/dev.md` §13.6），
  不是"乘倒数"。
- `api::sum` 对空切片返回 `0.0`，对任何切片有定义，故不返回 `Result`。
- 长度为 0：`_checked` 允许空指针；`lasx_alloc(0)` 返回"对齐但不可解引用"的非空悬垂指针。
- 批量几何/姿态内核**允许输出与输入别名**（向量路径先取完本轮输入再写回），故
  `quat_to_dcm_batch` 的输出可直接喂给 `mat3_mul_vec3_batch`。


### 2.5 契约（2026-09-24 定稿）

这一节是**冻结的约定**：改内核、加路径、上公式 DSL 都不能违反。逐条都有测试守着。

**1）累加顺序。** 每个输出元素是**沿 `k` 的单个累加器、`k` 升序、每步一条 FMA**：

```text
c[i][j] = fma(a[i][k-1], b[k-1][j], fma(… fma(a[i][0], b[0][j], 0) …))
```

`alpha`/`beta`（若启用）只能在**累加结束之后**按固定次序施加。任何"改写结合次序"的优化
（把 `k` 拆成多个并行部分和再相加、用 `beta·y` 给累加器播种、跨 lane 水平归约）都**违反契约**。

**2）路径无关性。** LASX / LSX / 标量尾 / 打包面板 / k 分块 / 并行切块 / `MatmulPlan` /
`shape` 层，对同一输入给出**逐位相同**的结果（测试用 `to_bits()` 比对）。

成立的前提有两条，换后端或换编译选项时要重新检查：

- 所有中间结果**以与目标类型相同的精度存储**（IEEE 754 单/双精度）；
  **不存在扩展精度寄存器**（x87 80 位那种）导致的"寄存器里更精确"的差异。
  龙芯 LSX/LASX 是标准 IEEE 754，满足这条。
- k 分块只是把**同一个部分和**分段写入 `C` 再读回：浮点数的存储/读取是精确的，
  故分块不改变任何一位。

**3）融合语义（已定稿，尚未启用）。** 若将来支持 `y = alpha·A·B + beta·y`：

- **`beta` 默认 `0.0`**（覆盖输出，对齐 BLAS 的 `gemm`），**`alpha` 默认 `1.0`**；
- 只有显式写出 `+ beta * y[…]` 才启用累加语义，不会"因为没写 beta 就默认累加"；
- 施加次序固定为 `out = alpha·acc + beta·y_old`（两次舍入），**不是** BLAS 那种
  "用 `beta·y` 给累加器播种"（那会改变累加结构，违反第 1 条）；
- **性能承诺只有一句**：融合省掉对 `C` 的一整遍读写，**不改写 GEMM 内部结构**。
  别处不得出现"融合接近 BLAS 语义""融合带来 2×"这类暗示。

**4）逐位一致的例外（仅此两处，均有测试）。** `lasx_ballistic_step` 的向量/标量分支
结合序不同（物理一致，绝对误差有测试守着）；`lasx_axpy` 允许 1 ulp 以内差异。


### 2.6 NN 侧（N1 批次）的数值契约：`lasx_softmax_rows`

`exp` 没有硬件指令，只能用多项式近似，所以这个算子的契约要写清楚"近似到什么程度、哪些
东西是逐位承诺、哪些不在契约内"。

**1）语义。** 行内 softmax（`rows × cols` 行主序）：

```text
out = exp(clamp_lo(scale·x + mask − 行内 max)) · (1/Σ)
```

- `mask` **可选**、同形状、**加性**（attention 的 `+ mask`）；C 侧用空指针表示无 mask；
- **先减行内 max 再 `exp`**（数值必需，否则溢出）；
- 归一化是 `e · (1/Σ)`（**一次除法 + 乘法**），不是逐元素除法——参考实现必须同样写
  `e * (1.0 / sum)` 才逐位一致。

**2）`exp` 的定义（这就是"契约"本身）。** `exp(x) = exp2 的 6 次多项式(f) · 2ⁿ`：

```text
t = x·LOG2E;  n = (t + 12582912.0) − 12582912.0;  f = t − n     // |f| ≤ 0.5
p = Horner(f, (ln2)^k/k! , k ≤ 6)（全程 FMA）
kk = clamp(trunc(n) + 127, 0, 254);  exp(x) = p · bitcast_f32(kk << 23)
```

- 相对误差 ~1e-10（多项式），**不是** libm `expf` 的逐位替代；与本库 f64 参考的相对误差
  在 1e-7 量级（f32 本身的精度）；
- **下溢恰好落到 0**：`x < −88.03` 时 `kk = 0` ⇒ `from_bits(0) = +0.0`。因此
  `mask = −inf`（"完全屏蔽"）给出的是**精确的 0 权重**。输入在 `exp` 前夹到 `≥ −104`
  （既保证这点，也挡住 magic 数取整失效与浮点转整型越界）；
- 这套序列**标量与向量共用**，且行内求和用"8 个 per-lane 累加器 + 固定次序两两归约"，
  列尾用 `−1e30` 填充后走同一条 lane-wise 路径 ⇒ **向量路径与标量模拟逐位一致**
  （单测守着，`src/ops/softmax_rows.rs`）。设计推导见 `docs/dev.md` §20.1。

**3）不在契约内。** 输入或 `mask` 含 `+inf`：`inf − inf = NaN`，属定义域问题，算子不处理。

> 这条契约由测试守：逐位一致（标量模拟）、显式补齐 == 列尾路径、f64 参考精度、
> 退化输入（行内全相等 / `scale=0` / `−inf` mask / 极负值不出 NaN）。
> 后续 N1 算子（`rms_norm`、`silu`/`gelu`、`rope`、`dot_f16`）沿用同一套写法：
> **先在本节写死 op 序列与"谁守"，再写代码**。


## 3. 指令集路径与降级覆盖

### 3.1 `SimdPath::detect()`

```rust
pub enum SimdPath { Lasx, Lsx }
impl SimdPath {
    pub fn detect() -> Self;   // hardware().lasx && !forced_lsx() ? Lasx : Lsx
    pub fn is_lasx(self) -> bool;
}
```

硬件能力用 `cpucfg` 读配置字 2：bit 7 = LASX、bit 6 = LSX。探测结果打包进**一个原子字节**
（写一次、之后只读），不是 `OnceLock`：快路径只剩一条 `Relaxed` 原子读，避免每次内核
调用 ~5 ns 的取值开销。**进程级只探测一次**，之后不可更新。支持降级的算子在入口
`match SimdPath::detect()` 分派到 `<name>_lasx` / `<name>_lsx`（少数内核的 `Lsx` 分支
退化为纯标量）。

### 3.2 降级覆盖分类（22 个原始符号）

| 类别 | 内核 | 无 LASX 时 |
|---|---|---|
| LASX + LSX 双路径 | `lasx_dot`、`lasx_norm3_batch`、`lasx_vec3_add_scaled_batch`、`lasx_j2_accel_batch`，以及全部 7 个姿态/几何算子 | LSX 128 位向量路径 |
| LASX + 标量降级 | `lasx_ballistic_step`、`lasx_batch_distance2d`、`lasx_rk4_j2_step_batch` | 全标量循环（无 LSX 向量路径） |
| LASX-only（无降级分支） | `lasx_matmul`、`lasx_matmul_f64`、`lasx_axpy`、`lasx_sum`、`lasx_dot_f64`、`lasx_dot_i8`、`lasx_dot_q4` | 直接执行 LASX 指令 |
| 内存工具（不涉及 SIMD） | `lasx_alloc` | 与路径无关 |

`_checked` 变体与其原始符号走同一条路径、同一个算子实现，覆盖范围相同。

### 3.3 LASX-only 的硬性约束

7 个 LASX-only 内核**没有 `has_lasx()` 检查，也没有降级分支**：

- 在 LSX-only CPU（3A5000 / 3A6000 等）上调用会执行 LASX 指令，产生**非法指令
  （SIGILL）**，不会自动降级。使用方必须自行按 CPU 能力调度。
- 源码模块文档对每个 LASX-only 内核都显式标注了这一点。
- `SimdPath::detect()` 与 `lasx_force_lsx_thread` 的降级只对有降级分支的内核有效；
  对 LASX-only 内核，`lasx_force_lsx_thread(true)` 不改变行为。

### 3.4 `lasx_force_lsx_thread` 与能力查询

```rust
pub fn lasx_force_lsx_thread(force: bool);           // 线程级，默认 false，仅 rlib 可见
let caps = lasx_rs::arch::hardware();                 // HwCaps { lasx, lsx }
let path = lasx_rs::arch::SimdPath::detect();         // Lasx | Lsx
```

用于在含 LASX 的真机上真实执行 LSX 分支（验证数值与性能）；源码注释强调这是"诚实标注"
——并非无 LASX 真机。必须显式置回 `false`。池化路径的 worker 是复用线程，与调用线程的
`FORCE_LSX` 无关（`parallel::rk4_j2_step_batch` 在每块开头显式置 `false`）。


## 4. 导出符号总表（45 个）

权威清单来自 `nm -D --defined-only target/release/liblasx_rs.so`：**23 个未带 `_checked`
的 `lasx_*` + 22 个 `lasx_*_checked` = 45**（N1 批次新加 `lasx_softmax_rows` 与其 checked）。

### 4.1 原始 15 个（历史契约，签名与语义不变）

| 符号 | C 签名要点 | 语义 | 路径 |
|---|---|---|---|
| `lasx_dot` | `float(const float*, const float*, int)` | `Σ a[i]·b[i]` | LASX+LSX |
| `lasx_sum` | `float(const float*, int)` | `Σ x[i]` | LASX-only |
| `lasx_dot_f64` | `double(const double*, const double*, int)` | f64 点积 | LASX-only |
| `lasx_axpy` | `void(float alpha, const float*, float*, int)` | `y += alpha·x`（原地） | LASX-only |
| `lasx_matmul` | `void(int m, int k, int n, const float*, const float*, float*)` | `C=A·B`，行主序 | LASX-only |
| `lasx_matmul_f64` | `void(int m, int k, int n, const double*, const double*, double*)` | f64 矩阵乘 | LASX-only |
| `lasx_dot_i8` | `int(const int8_t*, const int8_t*, int)` | int8 点积，返回 i32 | LASX-only |
| `lasx_dot_q4` | `double(const uint8_t*, const float*, const uint8_t*, const float*, int)` | Q4 量化点积 | LASX-only |
| `lasx_alloc` | `float*(int n)` | 分配 32 字节对齐的未初始化 f32 缓冲 | — |
| `lasx_norm3_batch` | `void(3×const double*, double*, int)` | `out=√(x²+y²+z²)` | LASX+LSX |
| `lasx_vec3_add_scaled_batch` | `void(6×const double*, double s, 3×double*, int)` | `o=a+s·b`，三分量独立 | LASX+LSX |
| `lasx_batch_distance2d` | `void(float px, float py, const float*, const float*, float*, int)` | `d=√(dx²+dy²)` | LASX+标量 |
| `lasx_ballistic_step` | `void(6×float*, const float* k, int, float dt, float g)` | 批量弹道欧拉步（原地） | LASX+标量 |
| `lasx_j2_accel_batch` | `void(3×const double* r, double mu, double j2, double re, 3×double*, int)` | J2 引力加速度 | LASX+LSX |
| `lasx_rk4_j2_step_batch` | `void(6×double*, double mu, double j2, double re, double dt, int)` | RK4 J2 单步（原地） | LASX+标量 |

长度 `n`/`m,k,n`/`n_bytes` 均为 C `int`（Rust `i32`）；标量参数按值传；输入 `*const`、
输出 `*mut`。

### 4.2 批量姿态/几何 7 个（新增，历史 15 个一个没动）

| 符号 | C 签名要点 | 语义 | 路径 |
|---|---|---|---|
| `lasx_cross3_batch` | `void(6×const double*, 3×double*, int)` | `o = a × b` | LASX+LSX |
| `lasx_unitize3_batch` | `void(3×const double*, 3×double*, int)` | `o = v/\|v\|`；零向量 → `(0,0,0)` | LASX+LSX |
| `lasx_mat3_mul_vec3_batch` | `void(9×const double* m, 3×const double* v, 3×double*, int)` | `o = M·v`，`M` 行主序 | LASX+LSX |
| `lasx_quat_normalize_batch` | `void(4×double* q, int)` | 原地单位化；`\|q\|<1e-15` → `(1,0,0,0)` | LASX+LSX |
| `lasx_quat_mul_batch` | `void(8×const double*, 4×double*, int)` | Hamilton 积 `a⊗b` | LASX+LSX |
| `lasx_quat_rotate_batch` | `void(4×const double* q, 3×const double* v, 3×double*, int)` | 先单位化，再 `o=R(q)·v`（体→惯） | LASX+LSX |
| `lasx_quat_to_dcm_batch` | `void(4×const double* q, 9×double*, int)` | 四元数 → 3×3 DCM（行主序） | LASX+LSX |

### 4.3 `_checked` 变体 21 个

每个原始符号（`lasx_alloc` 除外）都有一个 `_checked` 变体：**签名完全一致，仅在末尾追加
一个 `int *status` 出参**：

```c
float  lasx_dot_checked(const float *a, const float *b, int n, int *status);
void   lasx_matmul_checked(int m, int k, int n, const float *a, const float *b,
                           float *c, int *status);
double lasx_dot_q4_checked(const uint8_t *qa, const float *sa, const uint8_t *qb,
                           const float *sb, int n_bytes, int *status);
```

21 个名字：`lasx_dot_checked`、`lasx_sum_checked`、`lasx_dot_f64_checked`、
`lasx_axpy_checked`、`lasx_matmul_checked`、`lasx_matmul_f64_checked`、
`lasx_dot_i8_checked`、`lasx_dot_q4_checked`、`lasx_norm3_batch_checked`、
`lasx_vec3_add_scaled_batch_checked`、`lasx_batch_distance2d_checked`、
`lasx_ballistic_step_checked`、`lasx_j2_accel_batch_checked`、
`lasx_rk4_j2_step_batch_checked`、`lasx_cross3_batch_checked`、
`lasx_unitize3_batch_checked`、`lasx_mat3_mul_vec3_batch_checked`、
`lasx_quat_normalize_batch_checked`、`lasx_quat_mul_batch_checked`、
`lasx_quat_rotate_batch_checked`、`lasx_quat_to_dcm_batch_checked`。

行为约定：先校验，失败时写入 `LasxStatus` 并返回**安全中性值**（数值型 `0.0`/`0`，`void`
型只写状态），**不触碰输出缓冲**；成功时写回 `Ok`（0）。`status` 可传 `NULL`（不关心原因，
但校验仍执行）。校验顺序：`j2_accel`/`rk4_j2_step` 先判长度与物理常数，再转切片；
`matmul*` 先判维度非负、再判乘积不溢出、最后转切片。C 层只能判**结构**（空指针、负长度、
溢出、物理常数）；"数组真实长度是否与声明形状一致"只有知道长度的上层判得了，`BadShape`
主要留给语言绑定。

### 4.4 状态码（`LasxStatus`）

| 码 | 名称 | 含义 |
|---|---|---|
| 0 | `Ok` | 成功 |
| 1 | `NullPointer` | 指针为空且长度非 0 |
| 2 | `NegativeLength` | 长度参数为负 |
| 3 | `BadShape` | 形状/长度自洽性不成立 |
| 4 | `SizeOverflow` | 尺寸相乘溢出 |
| 5 | `NonPositiveConstant` | `mu <= 0` 或 `re <= 0` |
| 6 | `NonFiniteConstant` | `j2`/`dt` 为 NaN 或无穷 |

Rust 侧另有 `LasxStatus::message()`（中文原因）、`is_ok()`、`from_i32()` 供绑定层还原。

### 4.5 符号数清点

| 阶段 | 原始符号 | `_checked` | 合计 |
|---|---|---|---|
| 历史 15 个 | 15 | 14（`lasx_alloc` 无 checked） | 29 |
| 加 7 个姿态/几何 | 22 | 21 | 43 |
| 加 1 个 NN（N1 首批） | 23 | 22 | **45** |

原始 15 个的名字与语义始终不变；新增的是姿态/几何 7 个、NN 1 个与其 checked 变体。


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
- 实现：i-k-j 形式**列方向**微内核（A 的行元素广播、B 沿列方向连续流式读取），一次算
  4 行 × 32 列（16 个 8 通道累加器）；**不做 B 转置、不做跨 lane 水平归约**。按形状在
  打包 B 面板 / 列块在外 / 流式三条路径间自动选择，三条路径**逐位一致**（每个输出元素的
  k 升序累加次序相同）。
- 打包路径使用**线程本地** `RefCell<Vec<f32>>` 复用打包缓冲（按需增长、跨调用保留），
  不是每次调用的 B 转置临时分配；流式/列块路径不额外分配。
- 精度：每个输出元素沿 k 的**单个 f32 累加器**，无 f64 分块落盘，误差大于点积；对 f64
  参考相对误差 <1e-4。高精度场景改用 `lasx_matmul_f64`。
- `m == 0 || n == 0` 直接返回；`k == 0` 时 C 不会被写（`parallel::matmul_f32` 会显式
  `c.fill(0.0)`）。LASX-only。

### 5.6 `lasx_matmul_f64` — f64 矩阵乘（LASX-only）

`void lasx_matmul_f64(int m, int k, int n, const double *a, const double *b, double *c)`

- 语义与布局：与 `lasx_matmul` 完全同构（行主序、不转置），每向量 4 个 f64，一次算
  4 行 × 16 列；同样有打包 + k 分块与流式两条路径，小形状走流式，两路径逐位一致；
  线程本地打包缓冲同 5.5。
- 精度：全程 f64；对 f64 参考相对误差 <1e-12。LASX-only。

### 5.7 `lasx_dot_i8` — int8 量化点积（LASX-only）

`int lasx_dot_i8(const int8_t *a, const int8_t *b, int n)`

- 语义（硬契约）：`Σ a[i]·b[i]`，int8 输入、**i32 返回**。
- 整数精确：每向量 32 个 i8，`lasx_xvmulwev_h_b`/`lasx_xvmulwod_h_b` 取偶/奇通道做带符号
  8→16 位扩展乘，`xvadd_h` 合并得 16 个 i16；再拓宽到 i32 累加（上界 128×127² ≈ 2.1 M，
  安全）；每 **1024 字节**把 i32 累加器落盘、转 i64 累加，最后 `s_acc as i32` 返回。
- 边界：总点积超出 i32 范围时**回绕**（无饱和/报错）。约 n > (2³¹−1)/127² ≈ 13.3 万
  （全 ±127 输入）时可能溢出。测试与 i64 精确参考**逐位相等**，覆盖块边界、i32 回绕、
  (−128)² 的 i16 边界。LASX-only。

### 5.8 `lasx_dot_q4` — Q4 量化点积（LASX-only）

`double lasx_dot_q4(const uint8_t *qa, const float *sa, const uint8_t *qb, const float *sb, int n_bytes)`

- 量化格式：每个 `u8` 含 **2 个 4-bit 无符号 nibble**；低 = `q & 0x0f`，高 = `(q>>4)&0x0f`。
- scale 布局：`sa`/`sb`（f32）每 **32 字节一组**一个值；组数 `n_groups = ceil(n_bytes/32)`。
  `sa`/`sb` 长度必须 ≥ `n_groups`；非 32 倍数时最后一组按 `j/32` 取 scale。
- 公式（源码注释）：`dot = Σ_g sa[g]·sb[g]·Σ_{组内 64 nibble}(qa·qb)`。组内整数点积精确，
  再乘该组 scale 积（f32 乘法）并升 f64 累加。
- 实现：每批 4 组（128 字节）先把 8 个载入发出，再算 4 个独立组内部分和；归约与 f64
  累加**严格按组序**，浮点求和次序与逐组循环完全一致（逐位不变）。nibble ∈ [0,15]，
  乘积 ≤225、组内每 lane ≤900、总和 ≤14400，整数部分无溢出。
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
  见 §2.6——那个算子的"精度"完全由那一节定义，读它比读这里的摘要有用；
- **实测**（单线程，`cargo run -p lasx_bench --release -- softmax`）：
  `128×128` 22.3 µs / 5.87 GB/s、`1024×1024` 1.21 ms / 6.96 GB/s、`32×4096` 144.6 µs /
  7.25 GB/s；是朴素 Rust 行循环的 **4.9–5.9×**，约为 llama.cpp `SOFT_MAX` 典型值
  （2.4–4 GB/s）的 **1.4–3×**。按真实访存（三遍：max / exp+sum / 归一）折算约 8.6–10.9 GB/s，
  已在单线程带宽上限附近（推导见 `docs/dev.md` §20.1）；
- **LASX-only**：无降级分支（未列在 §3.2 的降级表里，因为它是新增符号——新增即 LASX-only，
  这一点在 §12.3 的批次说明里也写着）。


## 6. 批量几何与物理算子

### 6.1 `lasx_norm3_batch` — 批量 3 分量模长（LASX+LSX）

`void lasx_norm3_batch(const double *xs, const double *ys, const double *zs, double *out, int n)`

- 语义：`out[i] = √(xs[i]² + ys[i]² + zs[i]²)`，SOA 布局。
- 结合序固定 `x² + (y² + z²)`（右结合）；LASX 4 样本/向量、LSX 2 样本/向量、标量尾同式，
  分块内逐位一致。对独立标量参考（左结合写法）相对误差 <1e-9。

### 6.2 `lasx_vec3_add_scaled_batch` — 批量缩放加（LASX+LSX）

`void lasx_vec3_add_scaled_batch(6×const double*, double s, 3×double*, int n)`

- 语义：`o[i] = a[i] + s·b[i]`，x/y/z 三分量独立（SOA）；覆盖加（`s=1`）与缩放（`a=0`）
  两个退化情形。
- 向量路径单条 FMA `lasx_xvfmadd_d(vs, vb, va)`；标量尾 `f64::mul_add(s, b, a)` 同式，
  分块内逐位一致。相对独立标量参考 <1e-9。

### 6.3 `lasx_batch_distance2d` — 批量 2D 距离（LASX+标量）

`void lasx_batch_distance2d(float px, float py, const float *xs, const float *ys, float *out, int n)`

- 语义：`out[i] = √(dx² + dy²)`，`dx = xs[i]-px`、`dy = ys[i]-py`。
- LASX 8 点/向量：`xvfsub_s` 求差、`xvfmul_s`+`xvfadd_s` 求平方和（`fadd(fmul,fmul)`）、
  `xvfsqrt_s` 开方；标量尾同式同结合序，分块内逐位一致。非 LASX 时**全标量**。

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


## 7. 批量姿态与几何算子（7 个）

公共约定：四元数**标量在前** `q = w + xi + yj + zk`，Hamilton 积 `i²=j²=k²=ijk=−1`；
旋转矩阵按 Wertz Ch.12.2 构造，方向是**体坐标 → 惯性坐标**（惯性转体请先取共轭）；
`|q| < 1e-15` 视为退化 ⇒ 输出单位四元数 `(1,0,0,0)`；分量求和结合次序固定，LASX / LSX /
标量尾三条路径**逐位一致**；全部内核**允许输出与输入别名**（向量路径先取完本轮输入再
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

`api` 的矩阵接口把形状写成三个裸数字：`api::matmul(m, k, n, &a, &b)`。谁是谁、跟切片长度
对不对得上全靠调用方记——本仓库在这上面真的踩过坑（对齐后的列数 `n32` 当行跨距传进微内核、
把"既是循环上界又是 `B` 行跨距"的参数交换、基准里 `(m,k,n)` 写反）。

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
> 与其返回一个骗人的切片（那正是"行跨距搞错"这类 bug 的温床），不如返回一个按下标取数的
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

> 三层的分工：**`view` 管布局**（列主序/跨距/转置）、**`plan` 管复用**（打包一次）、
> **`shape` 管形状**（进类型）。它们可以叠：`shape::Prepared` 内部就是 `plan::MatmulPlan`。


## 9. 多核并行：`WorkerPool` 与 `parallel`

内核本身是单线程的（每个 `lasx_*` 只处理一段连续内存）。多核靠可选的**常驻**线程池。

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
  `row_gran == 0` 或第 k 个数组长度不等于 `rows × 行宽` 时 panic。`row_gran` 不能省：
  块大小会**向上取整到它的倍数**，避免退化尾块把整块拖慢；传 1 表示没有行结构。

`parallel` 策略层把常见负载的切分形状固定下来：

| 函数 | 说明 |
|---|---|
| `parallel::matmul_f32(&pool, m, k, n, a: &mut [f32], b: &[f32], c: &mut [f32]) -> Result<(), Error>` | 多核矩阵乘（行粒度 4），B 只读共享；`&pool` 可直接给 `pool::global()` |
| `parallel::matmul_f64(...)` | 同上，f64 |
| `parallel::rk4_j2_step_batch(&pool, mu, j2, re, dt, 6×&mut [f64]) -> Result<(), Error>` | 多核批量 RK4 J2 单步（6 数组一次派活） |

**`parallel::*` 形状不符返回 `Err`**（与 [`api`](#8-rust-安全层-lasx_rsapi) 同一套
`Error::Shape`/`Error::Overflow` 与消息口径）；`matmul_*` 对 `m==0 || n==0` 直接返回 `Ok`
（不写 C），`k==0` 时 `c.fill(0.0)`。**池的低层 `for_each_*` 仍按契约 panic**——它是
"调用方自己保证形状"的原语，`parallel` 才是带校验的那一层。
数值与单线程**逐位一致**（只切行/下标区间），有逐位对照测试。
`parallel::rk4_j2_step_batch` 的 worker 是复用线程，`lasx_force_lsx_thread` 对它无效。

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

## 10. 对齐缓冲：`AlignedVec` 与 `lasx_alloc`

`ALIGN = 64`：既是 LASX 要求的 32 字节的倍数，也让首地址落在缓存行边界上。

| 接口 | 说明 |
|---|---|
| `AlignedVec::new(len)` | 分配 `len` 个零值元素，首地址 64 字节对齐（`T: Copy + Default`） |
| `AlignedVec::fill_with(len, f)` | 分配并用 `f(i)` 填充 |
| `len` / `is_empty` / `as_slice` / `as_mut_slice` | 切片视图 |
| `as_ptr` / `as_mut_ptr` | 保证对齐的裸指针，可交给 C ABI |
| `Deref` / `DerefMut` | 可直接当 `&[T]` / `&mut [T]` 用（索引、`len()`、迭代、传参） |
| `Clone` | 克隆后**重新计算对齐偏移**并复制数据 |

```c
float *lasx_alloc(int n);   // 保证 32 字节对齐、未初始化；库内不导出 lasx_free
```

- `lasx_alloc` 用 `Layout::from_size_align(n*4, 32)` 分配，**保证 32 字节对齐**（普通 `Vec`
  只保证 4 字节）。`n == 0` 返回"对齐但不可解引用"的非空悬垂指针；`n < 0` 因 layout 非法
  而 panic。归还时 layout 必须与分配时一致：

```rust
unsafe {
    let layout = std::alloc::Layout::from_size_align(n as usize * 4, 32).unwrap();
    std::alloc::dealloc(ptr as *mut u8, layout);
}
```

对齐只影响性能、不影响正确性：LASX 的 `xvld/xvst` 是 32 字节访问，起始地址不落在 32 字节
边界时会跨缓存行。实测 glibc `malloc` 与 Dart FFI 的缓冲区只有约一半落在 32 字节边界上，
**不要假设缓冲区天然对齐**；未对齐时结果照样正确，且仍明显快于 LSX 路径。想稳定拿到对齐
内存：C 用 `posix_memalign(&p, 32, n)` / `aligned_alloc(32, n)`，Rust 用 `AlignedVec`，或
本库的 `lasx_alloc`。量化对比见 `cargo run -p lasx_bench --release -- align`，完整数据
详见 `docs/dev.md`。


## 11. FFI 用法（C / Dart）

```c
#include <stdint.h>

float lasx_dot(const float *a, const float *b, int n);
void  lasx_axpy(float alpha, const float *x, float *y, int n);
void  lasx_matmul(int m, int k, int n, const float *a, const float *b, float *c);
void  lasx_norm3_batch(const double *xs, const double *ys, const double *zs,
                       double *out, int n);
void  lasx_rk4_j2_step_batch(double *rx, double *ry, double *rz,
                             double *vx, double *vy, double *vz,
                             double mu, double j2, double re, double dt, int n);

float a[1024], b[1024];
float d = lasx_dot(a, b, 1024);   /* n 是 int（i32） */

int st;
float d2 = lasx_dot_checked(a, b, 1024, &st);
if (st != LASX_OK) { /* LASX_OK == 0，其余状态码见 4.4 */ }
```

```dart
final lib = DynamicLibrary.open('liblasx_rs.so');

final lasxDot = lib.lookupFunction<
    Float Function(Pointer<Float> a, Pointer<Float> b, Int32 n),
    Float Function(Pointer<Float> a, Pointer<Float> b, int n)>('lasx_dot');

final lasxAlloc = lib.lookupFunction<          // 32 字节对齐的输出缓冲
    Pointer<Float> Function(Int32 n),
    Pointer<Float> Function(int n)>('lasx_alloc');
```

`lookupFunction` 的泛型参数（Native 侧 / Dart 侧签名）必须与 C 签名完全一致，否则 UB；
Dart 侧 `n` 为 `int`，Rust 侧为 `Int32`。生命周期：所有内核只在调用期间读/写传入缓冲区，
不持有任何指针（无回调、无异步），函数返回后即可安全释放输入/输出缓冲。长度必须与实际
容量一致，指针不得悬垂或越界，`n` 不得为负——原始符号不做这些校验，误用即 UB。


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

**缺口（数字为本机 llama.cpp 实测，详见 docs/dev.md）**：

| 缺口 | 现状 | 可达 | 倍数依据 |
|---|---|---|---|
| f16/bf16 GEMM（prefill） | 8.4 / 16.6 GFLOP/s | ~90（f32 实测） | ~11× / 5× |
| f16/bf16 GEMV（decode） | 6.8 / 9.1 GB/s | ~25（DRAM） | ~3.7× |
| `SOFT_MAX` | 典型 2.4–4 GB/s | ~~~25（DRAM）~~ | **已落地**：`lasx_softmax_rows` 实测 **5.7–7.3 GB/s**（§20/§12.4），朴素 Rust 循环的 4.9–5.9× |
| `q6_K` | 9.8 GB/s | ~18（q4_K 实测） | ~1.8× |
| `mxfp4`/`nvfp4` | 5.5 / 4.0 GB/s | ~18 | ~3–4.5× |
| 没有缺口 | `q4_0`/`q4_K`/`q2_K`/`q3_K`/`q8_0`/`iq*`、`f32`、`ROPE`、`ADD` | — | ≈1× |

三条结论：量化点积**不是机会**（llama.cpp 的 `arch/loongarch/quants.c` 已把 q4_0/q4_K/
q5/q6_K（部分）/iq* LASX 化，再挤空间很小）；真正的短板是**非量化**那半边——f16/bf16 与
softmax 根本没有 LoongArch 分支（`vec.cpp` 里 silu/softmax 有 AVX/NEON/SVE，唯独没有
LASX）；对本库而言这批算子价值更高，因为 `api`/`pool`/`parallel`/姿态批量算子可直接复用
作底座。

### 12.3 候选批次

**N1（推荐先做，纯 f32/f16，无格式锁定）**

| 算子 | 说明 | 状态 |
|---|---|---|
| `dot_f16` / `gemv_f16` | f16×f32 点积与矩阵-向量，寄存器内 `xvfcvtl_s_h`/`xvfcvth_s_h` 转换 | 待做 |
| `softmax_rows` | 行内 max→exp→sum→归一，支持 `scale` 与可选加性 mask | **已落地**（§2.6 契约、§5.9 用法） |
| `rms_norm`（+权重） | 每层两次，行归约，与 softmax 共用 | 待做 |
| `silu` / `gelu`（quick/erf） | FFN 激活，逐元素，与 softmax 共用 exp 近似 | 待做 |
| `rope`（NeoX / GPT-J 两种 mode） | f32，支持 `n_dims`/`freq_base`，可逐位对照 | 待做 |

**N2（GGML 格式互操作，按需）**：`quantize_rows_q8_0`（激活量化：Q8_0 块 + f16 scale）、
`dot_q4_0_q8_0`/`gemv_q4_0`（llama.cpp 的 Q4_0 块布局 18 B/32，与现有 `dot_q4` 布局不同）、
`dot_q6_K_q8_K`（对应 1.8× 那一档）。N2 的价值是"与 llama.cpp/GGUF 生态互通并可交叉
验证"；若只给自己的推理代码用，N1 就够，且每个算子都能用现有 `api`/`parallel` 包装。

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

工具链：nightly Rust（含 `stdarch_loongarch`），`edition = 2021`，零依赖，
`[lib] crate-type = ["cdylib", "rlib"]`；`.cargo/config.toml` 配置
`rustflags = ["-C", "target-feature=+lasx"]`。loongarch64 nightly 安装示例：
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

# 计划复用对照（① 一次性 API / ② C 复用 / ③ 免打包，见 dev.md §18）
cargo run -p lasx_bench --release -- plan

cargo run --release --example pool_axpy        # 多核示例（自带逐位对照）
cargo run --release --example matmul_pooled

cargo build --release -p lasx_yll              # YouLiLong 原生扩展
cp target/release/liblasx.so yll/
```

基准套件组名（过滤子串）：`dot`、`sum`、`axpy`、`dot_f64`、`dot_i8`、`dot_q4`、`matmul`、
`plan`、`attitude`、`large`、`norm3`、`vec3`、`distance2d`、`j2`、`ballistic`、`rk4`、`fma`、
`mt`、`align`、`dispatch`、`scenario`。方法学与全部读数详见 `docs/dev.md`。

```bash
# 导出符号自检：期望 45 行（23 原始 + 22 checked）
nm -D --defined-only target/release/liblasx_rs.so \
  | awk '$2=="T" && $3 ~ /^lasx_/ {print $3}' | sort
```


## 15. Caveats 与限制

1. **降级覆盖不全**：7 个 LASX-only 内核（`matmul`、`matmul_f64`、`axpy`、`sum`、
   `dot_f64`、`dot_i8`、`dot_q4`）在 LSX-only CPU 上会 SIGILL，不会自动降级。
2. **仅标量降级**：`ballistic_step`、`batch_distance2d`、`rk4_j2_step_batch` 的非 LASX
   分支是全标量循环，没有 LSX 128 位向量路径。
3. **`ballistic_step` 非逐位一致**：向量与标量 `|v|²` 结合序不同；物理一致，有绝对误差
   测试守护。
4. **`axpy` 向量/标量尾至多差 1 ulp**（FMA vs 分离乘加），无逐位守护。
5. **`lasx_alloc` 无对等释放导出**：无 `lasx_free`，调用方必须按同一 layout 归还，否则
   泄漏或 UB。
6. **原始符号长度参数无负数/零防护**：`n` 为 `i32` 后立即 `as usize`；负数变成巨大 usize，
   `from_raw_parts` 长度超界即 UB；`lasx_alloc` 对负 `n` 会 panic。必须保证 `n ≥ 0` 且与
   缓冲区实际容量一致。
7. **f32 矩阵乘无 f64 落盘**：`lasx_matmul` 在 f32 累加器内直接累积，长 k 维误差大于点积；
   高精度用 `lasx_matmul_f64`。
8. **`dot_q4` 的 scale 长度约定**：`sa`/`sb` 长度必须 ≥ `ceil(n_bytes/32)`，按 32 字节组
   对齐；非 32 倍数时最后不足组按 `j/32` 索引。
9. **`dot_i8` 结果截断**：内部 i64 累加精确，返回 `as i32` 会回绕；应保证业务上 n 远小于
   约 13.3 万的界。
10. **`dot` 的小规模路径**：n < 24 用 f32 单精度累加，与 n ≥ 24 的 f64 分块累加公式不同，
    跨边界不保证逐位连续。
11. **cpucfg 依赖与缓存**：LASX 检测依赖 `cpucfg` + CFG2.bit7；结果进程级缓存一次、之后
    不可更新。虚拟化/模拟器中若 cpucfg 不可信，检测结果也随之不可信。
12. **检测缓存与钩子作用域**：硬件能力进程级缓存；`FORCE_LSX` 线程级、默认 `false`，测试
    必须自行复位。池化路径的 worker 线程不受调用线程钩子影响。
13. **对齐与内存布局**：向量加载/存储用偏移 0，LoongArch 允许非对齐访问，源码未显式声明
    对齐要求；SOA/数组布局假设连续无填充，跨语言调用方必须保证元素类型与步长严格匹配
    （f32=4 字节、f64=8 字节）。


## 16. 待确认与已修正

本文由旧的四份文档（manual / nn-ops / compression / perf-report）合并重写，凡是旧描述与
**当前源码**不符的，一律以源码为准。合并时发现的问题分三类：

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
