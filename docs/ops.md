# lasx_rs 算子与用法手册

> 定位：**怎么用**（算子、契约、边界、判据）。设计与性能方法学见 `docs/dev.md`；C 签名见 `docs/c-abi.md`；
> 平台/工具链见 `docs/platform.md`。数值契约的权威来源是代码 + 测试，本文与它们同步维护。

## 0. 算子 → 入口 / 精度类 总表

| 族 | 算子（内核） | Rust 入口（`lasx_rs::api::*`） | C ABI（`lasx_*`） | 精度类 |
|---|---|---|---|---|
| 归约 | dot_f32 / sum / dot_f64 | `dot` / `sum` / `dot_f64` | `lasx_dot` / `lasx_sum` / `lasx_dot_f64` (+`_checked`) | 逐位 |
| 归约 | amax / absmax_rows | `amax` / `absmax_rows` | `lasx_amax` / `lasx_absmax_rows` (+`_checked`) | 逐位（边角语义固定） |
| 逐元素 | axpy / silu / gelu_* | `axpy` / `silu` / `gelu_quick` / `gelu_erf` | 同名 `lasx_*` (+`_checked`) | axpy/silu 逐位；gelu_quick **容差**（界 5e-2） |
| 矩阵乘 f32 | matmul / scaled / packed | `matmul` / `matmul_into` / `matmul_scaled_into` / `matmul_f32_packed` | `lasx_matmul` / `lasx_matmul_scaled` | 逐位（全路径） |
| 矩阵乘 f64 | matmul_f64 | `matmul_f64` / `matmul_f64_into` | `lasx_matmul_f64` | 逐位 |
| 计划句柄 | prepare / run / run_threaded / destroy | （C 侧主用）| `lasx_matmul_prepare` / `lasx_matmul_run` / `lasx_matmul_run_threaded` / `lasx_matmul_plan_destroy` | 逐位（计时前断言） |
| 复数矩阵乘 | cmatmul f32/f64 | （C 侧）| `lasx_cmatmul_f32` / `lasx_cmatmul_f64` | 逐位 |
| 量化 | per_tensor / per_row / per_token / per_channel | `quantize_i8_*` | `lasx_quantize_i8_*` | 逐位（ties-even + clamp） |
| 反量化 | dequantize / dequantize_rows | `dequantize_i8` / `dequantize_i8_rows` | `lasx_dequantize_i8*` | 逐位 |
| int8 GEMV/GEMM | gemv_i8 / matmul_i8 / dot_i8 / dot_q4 | `gemv_i8` / `matmul_i8` / `dot_i8` / `dot_q4` | 同名 `lasx_*` | dot_i8 逐位；gemv/matmul_i8 按量化契约 |
| NN | softmax_rows / rms_norm / layer_norm | `softmax_rows` / `rms_norm` / `layer_norm` (+`_into`) | 同名 `lasx_*` | 逐位（按登记公式） |
| NN | rope（半旋转） | `rope` / `rope_into` / `rope_tables` / `rope_at` | `lasx_rope` | 逐位 |
| f16 | dot_f16 / gemv_f16（f16 权重） | `dot_f16` / `gemv_f16` | 同名 | 容差类（求和次序） |
| 几何 f64 | norm3 / norm2 / vec3_add_scaled / cross3 / unitize3 / mat3_mul_vec3 | `*_batch` / `*_into` | `lasx_*_batch` | 逐位 |
| 几何 f64 | quat_normalize/mul/rotate/to_dcm | `quat_*` | `lasx_quat_*` | 逐位 |
| 物理批量 | batch_distance2d / ballistic_step / j2_accel_batch / rk4_j2_step_batch | 同名 | 同名 | 逐位 |
| 索引 | gather_rows / gather_rows_i8 | `gather_rows` / `gather_rows_i8` (+`_into`) | 同名 | 逐位 |
| 运行期 | 能力/版本/内存 | `has_lasx` / `version` / `alloc`/`free` | `lasx_has_lasx` / `lasx_version` / `lasx_alloc`/`lasx_free` | — |

`*_checked` = C ABI 的带长度校验变体（推荐）；无后缀为原始指针快路径（调用方保证长度）。**总共 98 个导出符号**。

## 1. 分层与调用方式

- **Rust**：`lasx_rs::api::*`（切片接口，长度自校验）；`*_into` = 写入调用方缓冲、**不分配**（性能路径/复用缓冲）。
- **C**：`docs/c-abi.md` 的 `lasx_*`（原始指针 + 长度；优先用 `*_checked`，返回 `LasxStatus`）。
- **线程模型**：库内不自建并行区（`WorkerPool` 由调用方通过 `parallel::matmul_f32` 等入口使用）；
  矩阵乘的 plan/run 由**调用方线程**驱动（打包一次、多次运行）。
- 错误统一 `Result<_, LasxError>`（Rust）/ `LasxStatus`（C）；**契约错误返回 Err，不 panic**。
- 别名：`*_into` 默认输入输出不重叠（调用方保证；重叠是未定义行为）。

## 2. 数值契约

### 2.1 "逐位一致"的精确含义

同一算子的所有实现路径（LASX 256 位 / LSX 128 位 / 标量尾 / 打包 / 列块 / 分块）对同一输入给出**逐位相同**的浮点结果，
测试以 `to_bits()` 全等断言。声明为**容差类**的算子必须给出实测界与参照物（§2.3）。

### 2.2 结合次序（硬约定）

- 归约类：默认**多累加器形状固定**（f32 8 路 / f64 4 路；行归约按行 8 路），参照物 = 顺序累加 ⇒ **容差类**；
  声明为逐位的归约（max/min/absmax、i8 整数）与累加顺序无关。
- 矩阵乘：k 方向**顺序累加**，分块/打包/行列并行**不改变累加次序** ⇒ 逐位；**不做 split-K**（会改次序）。
- 向量化尾块：标量尾与向量体同序 ⇒ 逐位。

### 2.3 精度一览

| 类别 | 代表算子 | 相对误差/界 | 说明 |
|---|---|---|---|
| 逐位 | matmul f32/f64、dot、axpy、norm/softmax/rms/layer、几何 f64、量化/反量化、gather、copy_strided、rope | 0（`to_bits` 全等） | 硬承诺 |
| 容差·多累加器 | dot_q4、dot_f16、行归约 `reduce_sum_rows` 等 | 实测界见测试（累加次序差异） | 与顺序累加参照比 |
| 容差·近似激活 | `gelu_quick` | max abs **2.063e-2**（界 5e-2） | tanh 近似 |
| 容差·f16 | dot_f16 / gemv_f16 | 按 f16 精度（≈1e-3 量级） | 权重 f16 转 f32 精确 |
| 端到端 int8 | quantize→gemv_i8→dequantize | 见 §5 的误差上界 | 不是位精确 |

### 2.4 退化输入与边界约定

- **空输入**（len=0）：合法，返回 0/空写；`rows=0` 或 `cols=0` 的批量算子合法。
- **长度/形状不符**：返回 `LengthMismatch`/`ShapeOverflow`（不 panic）；`*_checked` 变体返回对应 `LasxStatus`。
- **NaN/±0**：`max/min` 族顺序 fold（保留 NaN payload 与 −0/+0）；`amax` 全 NaN 行 → 0.0、−0 → +0（登记语义）。
- **非 2 的幂 / 行宽非 16 倍数**：走标量尾，结果仍逐位；不需要对齐。
- **±inf**：按 IEEE 传播（激活/归约的 inf 行为与标量参照一致）。

### 2.5 冻结契约（改内核/加路径/上 DSL 均不可违反，每条有测试）

1. 逐位语义与容差类登记不得静默改变；新路径必须与旧路径逐位或按登记容差。
2. 错误码与退化输入行为（§2.4）冻结；新增错误面须同步（DSL 仓）契约与测试。
3. C ABI 的 15 个原始符号（见 `docs/c-abi.md`）**签名与语义永不改动**。
4. 线程/池接口（`WorkerPool` 行块闭包、`Policy::threads`）冻结：宏与并行层按它生成代码。
5. `*_into` 的"不分配 + 不重叠"约定冻结。

## 3. 运行时路径与能力

### 3.1 `SimdPath::detect()`

启动时检测 CPU 能力：`Lasx`（256 位）/ `Lsx`（128 位）/ `Scalar`；选择对应三层实现。**同一二进制自适应**，
不需要按 CPU 编译两份。

### 3.2 降级覆盖分类

- **全路径逐位**：同一数学的三层实现（LASX/LSX/标量）——绝大多数算子。
- **LASX/LSX 有独立实现、标量兜底**：如 `norm3_batch`（LASX 4 f64/lane、LSX 2/lane、标量尾同式）。
- **LASX-only**：部分高价值内核只在 LASX 路径实现（§3.3）；LSX/标量机器走通用路径（仍逐位）。

### 3.3 LASX-only 的硬性约束

- 需要 `+lasx` 编译特性（仓库 `.cargo/config.toml` 已设）；**外部 crate 必须显式带**，否则静默回退标量（同一代码慢 26–46×）。
- 运行期缺 LASX 时：`has_lasx()==false`，`*_checked` 返回相应状态码；纯标量机器用通用实现（逐位）。
- `matmul`/`matmul_f64` 的快速路径要求尺寸达到内核对齐常量（小尺寸自动退标量尾/通用）。

## 4. 核心算子用法

> 每个算子的 C 签名/参数/错误码见 `docs/c-abi.md`；本节只给语义、约束与用法判据。

### 4.1 归约与 axpy

| 算子 | 语义 | 约束/注意 |
|---|---|---|
| `dot(x, y)` | Σxᵢyᵢ (f32) | 等长；逐位 |
| `dot_f64(x, y)` | Σxᵢyᵢ (f64) | 等长；逐位（4 路固定） |
| `sum(x)` | Σxᵢ | 累加形状同 dot |
| `axpy(alpha, x, y)` / `axpy_into` | y := alpha·x + y | 逐位；就地/非就地均有入口 |
| `amax(x)` / `absmax_rows(x, rows, cols, out)` | 最大绝对值 / 行 absmax | 全 NaN 行→0、−0→+0（登记） |

### 4.2 矩阵乘 f32/f64（含 plan）

- 布局：行主序 `A[m×k]`、`B[k×n]`、`C[m×n]`；`matmul_scaled` 支持 alpha 缩放。
- **打包一次、多次调用**：C 侧 `lasx_matmul_prepare`（打包 B/列映射）→ `lasx_matmul_run`（可多线程，
  调用方线程驱动）→ 用完 `lasx_matmul_plan_destroy`；逐位断言在计时前做（`*_checked`）。
- 线程数由调用方给（`Policy::threads`）；**不要在外层再切行块**（库内已并行，嵌套会慢）。
- 性能对照与结论见 `docs/dev.md` §4.4；**小形状开销**：C ABI 调用约比 Rust 切片多 6 ns（n=64）。

### 4.3 量化 / 反量化（int8）

- 生产端：`quantize_i8_per_tensor` / `per_row`（C 也提供 per_token/per_channel 语义入口）；**ties-even + clamp**，逐位。
- 反量化：`dequantize_i8`（整张一个 scale）/ `dequantize_i8_rows`（每行 scale）：`out = (q as f32)·scale[r]`，逐位。
- 端点值：`±127`、`−128`、`±0`、次正规、`scale=NaN/±inf` 均已覆盖（见测试）。

### 4.4 int8 推理：gemv / matmul / dot

- `gemv_i8(j,k,a,x,out)`：权重 i8 × 激活 i8 → i32（i64 累加 + 饱和）。**输出为 i32 累加值**，反量化/缩放由调用方做。
- `dot_i8(a,b)`：i8 点积 → i32（饱和）——逐位（整数结合律）。
- `dot_q4`：Q4 打包权重 × f32 激活（量化格式约定见量化生产端 scale/zero 语义）。
- `matmul_i8`：批量 int8 矩阵乘（**保留但非性能路径**：实测打不过 f32 GEMM，见（lasx 仓）docs/dev.md §8.1）。

### 4.5 NN 算子

| 算子 | 契约要点 |
|---|---|
| `softmax_rows(x, rows, cols, out)` | 行内 max → exp+行和 → 缩放；逐位（按登记公式） |
| `rms_norm` / `layer_norm` | 行内 RMS/LayerNorm（eps 参与）；逐位 |
| `silu` / `gelu_quick` / `gelu_erf` | 逐元素激活；silu 逐位、gelu_quick 容差（界 5e-2）、gelu_erf 精确版 |
| `rope` / `rope_into` / `rope_tables` | 半旋转（NeoX 风格）；表由 `rope_tables` 预计算、`rope_at` 取位置 |

### 4.6 f16 权重

- `dot_f16(f16_weights, f32_x)`、`gemv_f16`：f16 权重转 f32 **精确**（硬件 cvt），再与 f32 运算；
  适合 decode 期权重 f16、激活 f32。性能见 dev.md §4.3。

### 4.7 批量几何 / 物理算子（f64 SOA）

- 布局统一为 **SOA**（`xs, ys, zs` 分离切片；批量 = 每个分量数组等长）。
- `norm3_batch(xs,ys,zs,out)`、`norm2_batch_into`、`vec3_add_scaled_batch`、`cross3_batch`、`unitize3_batch`、
  `mat3_mul_vec3_batch`（9 个分量数组）、`quat_*` 四元数族：**全部逐位**（LASX 4 f64/lane + 标量尾同式）。
- 物理：`batch_distance2d`、`ballistic_step`、`j2_accel_batch`、`rk4_j2_step_batch`。
- 空批量合法（len=0 → 空写）；长度不一致返回 `LengthMismatch`。

### 4.8 复数矩阵乘

- `cmatmul_f32` / `cmatmul_f64`（C 侧）：实数/虚数分离的 ABI 布局，语义同复数 GEMM；逐位。
- （DSL 仓的引擎组合路径也用它；Rust 侧 `api` 暴露有限，按 C ABI 使用。）

### 4.9 索引：gather

- `gather_rows(src, rows, cols, ids, out)`（f32）/ `gather_rows_i8`：按 id 取整行。
- id 越界 = 契约错误（`Err`/状态码），**不是** UB；`*_into` 不分配。

## 5. 端到端口径与判据

### 5.1 int8 vs f32：一句话判据

**有量化依据（权重/激活可用 per-row scale 表示且精度容忍）→ int8 GEMV/GEMM；否则 f32**。
`matmul_i8` 非性能路径；decode 期优先 `gemv_i8`（C 侧 `lasx_gemv_i8_checked`）。

### 5.2 int8 路径误差上界（端到端口径）

- 量化误差：`|x − dequant(q)| ≤ 0.5·scale`（ties-even 舍入）；累积到 dot/GEMV 后按 i64 精确累加 + 缩放传播。
- 端到端（quantize → gemv_i8 → dequantize）误差以量化 scale 与累加饱和为界，**不是位精确**；
  逐位仅保证"同量化输入"下的整数部分。

## 6. C ABI 与状态码

- 98 个导出符号；**15 个原始符号**（`lasx_dot`/`sum`/`axpy`/`matmul`/`matmul_f64`/`matmul_scaled`/`dot_i8`/`dot_q4`/
  `amax`/`absmax_rows`/softmax/rms/silu 等）签名与语义**永不改动**；其余为扩展（多为 `*_checked` 长度校验变体）。
- 状态码 `LasxStatus`：`Ok` / `NullPointer` / `LengthMismatch` / `ShapeOverflow` / `NonFiniteInput` / `UnsupportedPath` /
  `InvalidPlan` / `PanicCaught`；`lasx_status_message(status)` 取可读消息。
- 内存：`lasx_alloc` / `lasx_free`（跨 FFI 分配/释放）；`matmul_plan_destroy` 归还计划句柄。
- C 签名、布局、示例见 `docs/c-abi.md`。

## 7. 能力/版本查询

- `has_lasx()` / `lasx_has_lasx()`：当前 CPU 是否 LASX；`version()` / `lasx_version()`：库版本。
- 路径自适配（§3.1）⇒ 调用方通常不需要分支；仅当"必须 LASX-only 内核"时查询并按 `UnsupportedPath` 处理。
