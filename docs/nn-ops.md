# 神经网络算子分析（以 llama.cpp / ggml 为参照）

> 目的：为 lasx_rs 选下一批算子找依据。参照物是本机 `/home/lik/llama.cpp`（已构建，
> `GGML_LASX=ON`），度量用 ggml 自带的 `test-backend-ops perf`——它按算子跑、单线程、
> 结果直接可比。所有数字都是**这台龙芯机器上开着 LASX 实测**的，不是估算。

## 1. ggml 的算子全景

`ggml.h` 的 `enum ggml_op` 共 **102 个算子**。按用途分类：

| 类别 | 算子 |
|---|---|
| 一元/激活 | `UNARY`（含 SILU/GELU/GELU_QUICK/GELU_ERF/RELU/TANH/SIGMOID/EXP/NEG…）、`GLU`/`GEGLU`/`GEGLU_ERF`/`GEGLU_QUICK`、`LEAKY_RELU`、`CLAMP`、`SQR`/`SQRT`/`LOG`/`SIN`/`COS` |
| 逐元素二元 | `ADD`/`ADD1`/`ADD_ID`/`ACC`、`SUB`、`MUL`、`DIV`、`SCALE`、`SET`、`CPY`/`CONT`、`DUP` |
| 归约 | `SUM`、`SUM_ROWS`、`CUMSUM`、`MEAN`、`ARGMAX`、`ARGSORT`、`TOP_K`、`COUNT_EQUAL` |
| 矩阵 | **`MUL_MAT`**、`MUL_MAT_ID`、`OUT_PROD` |
| 形状/布局 | `RESHAPE`/`VIEW`/`PERMUTE`/`TRANSPOSE`/`CONCAT`/`REPEAT`/`PAD`/`ROLL`/`IM2COL`/`COL2IM` |
| 归一化 | **`RMS_NORM`**（+BACK）、`NORM`（LayerNorm，+BACK）、`GROUP_NORM`、`L2_NORM` |
| 注意力 | **`SOFT_MAX`**（+BACK）、**`ROPE`**（+BACK）、`DIAG_MASK_INF`/`ZERO`、**`FLASH_ATTN_EXT`**（+BACK）、`WIN_PART`/`WIN_UNPART`、`GET_REL_POS`/`ADD_REL_POS` |
| 卷积/池化 | `CONV_2D`/`CONV_3D`/`CONV_2D_DW`/`CONV_TRANSPOSE_*`、`POOL_1D`/`POOL_2D`、`UPSCALE` |
| 索引/嵌入 | **`GET_ROWS`**（+BACK）、`SET_ROWS`、`TIMESTEP_EMBEDDING`、`ARANGE` |
| SSM/线性注意力 | `SSM_CONV`、`SSM_SCAN`、`GATED_LINEAR_ATTN`、`RWKV_WKV6`/`7`、`GATED_DELTA_NET`、`LIGHTNING_INDEXER` |
| 训练 | `CROSS_ENTROPY_LOSS`（+BACK）、`OPT_STEP_ADAMW`/`SGD`、各 `*_BACK` |

标 **粗体**的是 transformer 前向每层都会走到的。

## 2. 一次前向用到的算子（图构建统计）

对 `src/llama-graph.cpp` + `src/models/*.cpp` 里全部模型构建器统计 `ggml_*` 调用次数
（反映"覆盖面"，即有多少模型用到）：

| 算子 | 次数 | 算子 | 次数 |
|---|---|---|---|
| `ggml_add` | 503 | `ggml_sigmoid` | 61 |
| `ggml_get_rows` | 374 | `ggml_transpose` | 38 |
| **`ggml_mul_mat`** | **205** | `ggml_silu` | 23 |
| `ggml_mul` | 160 | `ggml_cpy` | 21 |
| `ggml_scale` | 123 | `ggml_sum_rows` | 19 |
| `ggml_cont` | 119 | **`ggml_rms_norm`** | 19 |
| `ggml_concat` | 105 | `ggml_tanh` | 18 |
| `ggml_permute` | 99 | `ggml_exp` | 13 |
| `ggml_view_*` | 287 | **`ggml_soft_max`** | 7 |

按**单个模型**的一次前向看，每层固定出现：`MUL_MAT`（QKV/O 投影 + FFN 三个矩阵、
MoE 更多）、`RMS_NORM` ×2、`ROPE` ×1~2、`SOFT_MAX` ×1（或 `FLASH_ATTN_EXT`）、
`SILU`/`GELU` ×1、`ADD` ×2~3、`MUL` ×1~2；解码时还有 `GET_ROWS`（词嵌入）。

## 3. 量化格式与块布局

llama.cpp 共 40+ 种类型，热的是这几档（`QK*` 为块内元素数）：

| 类型 | 块结构 | 比特/权重 | 用途 |
|---|---|---|---|
| `f32` | — | 32 | 精度基线 |
| `f16`/`bf16` | — | 16 | 常见权重与 KV cache |
| `q4_0` | `f16 d` + 16 B nibble / 32 | **4.5** | 最常用的 4-bit 量化 |
| `q8_0` | `f16 d` + 32×i8 / 32 | 8.5 | 激活量化（`Q8_0`）与高质量权重量化 |
| `q4_K` | `f16 d`+`f16 dmin`+12 B scales+128 B qs / 256 | 4.5 | K-quant 主力 |
| `q6_K` | 128 B ql+64 B qh+16 B scales+`f16 d` / 256 | 6.5625 | 输出层/高质量档 |
| `q2_K`/`q3_K`/`q5_K` | 见 `ggml-common.h` | 2.625 / 3.4375 / 5.5 | 低比特/高比特档 |
| `mxfp4`/`nvfp4` | 块内 E8M0/E4M3 缩放 + 4-bit | ~4.5 | 新格式（fp4 + 微缩放） |
| `iq1_s`…`iq4_xs` | 码本 + 索引 | 1.5–4.5 | 极限压缩档 |

## 4. 本机实测

构建确认：`build/CMakeCache.txt` 里 `GGML_LASX:BOOL=ON`，`libggml-cpu.so` 反汇编含
**3449 条 LASX 指令**——即下面的数字是**开了 LASX 之后**的结果。命令：

```bash
cd /home/lik/llama.cpp
./build/bin/test-backend-ops perf -o MUL_MAT -b CPU        # 矩阵乘各类型
./build/bin/test-backend-ops perf -o SOFT_MAX,ROPE,ADD -b CPU
```

### 4.1 解码 GEMV（m=4096, k=14336, n=1，单线程）

| type_a（权重） | 每次 | 相对 q4_0 | 有效带宽 | LASX 专用实现？ |
|---|---|---|---|---|
| `q4_0` | **1792 µs** | 1.00× | 18.4 GB/s | ✅ |
| `q4_K` | 1834 µs | 1.02× | 18.0 GB/s | ✅ |
| `iq2_s` | 1657 µs | 0.92× | ~18 GB/s | ✅ |
| `q2_K` | 646 µs | 0.36× | 29.8 GB/s（部分驻 L3） | ✅ |
| `q3_K` | 1319 µs | 0.74× | 19.1 GB/s | ✅ |
| `q8_0` | 3502 µs | 1.95× | 17.8 GB/s | ✅ |
| **`q6_K`** | **4925 µs** | **2.75×** | **9.8 GB/s** | ❌（通用 C） |
| `f32` | 9368 µs | 5.23× | **25.1 GB/s**（贴 DRAM 上限） | ✅ |
| **`f16`** | **17177 µs** | **9.58×** | **6.8 GB/s** | ❌ |
| **`bf16`** | **12907 µs** | **7.20×** | **9.1 GB/s** | ❌ |
| `mxfp4` | 5703 µs | 3.18× | ~5.5 GB/s | ❌ |
| `nvfp4` | 8285 µs | 4.62× | ~4.0 GB/s | ❌ |
| `q1_0` | 6349 µs | 3.54× | ~1.3 GB/s | ❌ |

### 4.2 prefill GEMM（m=4096, k=14336, n=512）

| type_a | 每次 | GFLOP/s |
|---|---|---|
| `f32` | 658.7 ms | **91.3**（接近 LASX f32 峰值 93.2） |
| **`f16`** | **7131 ms** | **8.4**（比 f32 慢 **10.9×**） |
| **`bf16`** | **3620 ms** | **16.6**（慢 5.5×） |
| `q4_0` | 281.9 ms | 213 |
| `q8_0` | 267.8 ms | 224 |

### 4.3 非矩阵算子

| 算子 | 形状 | 有效带宽 |
|---|---|---|
| `ADD` f32 | 24 MB | **94.4 GB/s**（缓存内） |
| `ADD` f32 | 48 kB | 9.9 GB/s |
| `ROPE` f32 | 16–147 MB 多种 mode | 4.7–72.7 GB/s（多数 ≥17） |
| **`SOFT_MAX` f32** | 0.6–655 MB 多种形状 | **0.78–6.17 GB/s**（典型 2.4–4） |

## 5. 缺口清单（按收益排序）

把"有效带宽 vs 本机可达"和"是否有 LASX 专用实现"对起来看：

| 缺口 | 现状 | 可达 | 倍数 | 依据 |
|---|---|---|---|---|
| **f16/bf16 GEMM（prefill）** | 8.4 / 16.6 GFLOP/s | ~90（f32 实测） | **~11× / 5×** | §4.2；`ggml_vec_dot_f16` 无 LASX 路径 |
| **f16/bf16 GEMV（decode）** | 6.8 / 9.1 GB/s | ~25（DRAM） | **~3.7×** | §4.1 |
| **`SOFT_MAX`** | 2.4–4 GB/s | ~25（DRAM） | **~7×** | §4.3；`vec.cpp` 有 AVX/NEON/SVE 分支，**无 LoongArch** |
| **`q6_K`** | 9.8 GB/s | ~18（q4_K 实测） | ~1.8× | §4.1 |
| **`mxfp4`/`nvfp4`** | 5.5 / 4.0 GB/s | ~18 | ~3–4.5× | 均走 `_generic` 标量 |
| `q1_0`/`q2_0` | 1.3 GB/s | ~18 | ~10×+ | 均走 `_generic` 标量 |
| **没有缺口** | `q4_0`/`q4_K`/`q2_K`/`q3_K`/`q8_0`/`iq*`、`f32`、`ROPE`、`ADD` | — | ≈1× | 已贴带宽/算力上限 |

三条结论：

1. **量化点积不是机会**：`arch/loongarch/quants.c`（94 KB）已经把 q4_0/q4_K/q5/q6_K(部分)/
   iq* 都 LASX 化了，q4_0 解码 GEMV 打到 18.4 GB/s（DRAM 的 74%），q4_K 的 GEMM 是
   实测 213 GFLOP/s。**在这上面再挤，空间很小**。
2. **真正的短板是"非量化"那半边**：f16/bf16（GEMV 3.7×、GEMM 11×）与 softmax（~7×）。
   它们不是"没优化到极致"，而是**根本没有 LoongArch 分支**（`vec.cpp` 里 silu/softmax
   手写了 AVX512/AVX2/SSE/NEON/SVE，唯独没有 LASX）。
3. 对**我们自己的库**（lasx_rs，服务于 Rust/Dart/YouLiLong，不一定要贴着 llama.cpp），
   同一批算子的价值更高：这些正是推理/训练的通用件，而且我们有 §2.16 那批批量算子、
   `pool`/`parallel` 多核设施、以及 `api` 安全层可以直接复用。

## 6. 对 lasx_rs 的建议批次

**批次 N1（推荐先做，纯 f32/f16，无格式锁定）**

| 算子 | 说明 | 预期依据 |
|---|---|---|
| `dot_f16` / `gemv_f16` | f16×f32 点积与矩阵-向量，寄存器内 `xvfcvtl_s_h`/`xvfcvth_s_h` 转换 | §5 的 3.7× |
| `softmax_rows` | 行内 max→exp→sum→归一，支持 `scale` 与可选加性 mask | §5 的 ~7× |
| `rms_norm`（+ 权重） | 每层两次，`sum_rows` 式归约 | 与 softmax 共用归约 |
| `silu` / `gelu`（quick/erf） | FFN 激活，逐元素 | 与 `softmax` 共用 exp 近似 |
| `rope`（NeoX / GPT-J 两种 mode） | f32，支持 `n_dims`/`freq_base` | 已有参考实现可逐位对照 |

**批次 N2（GGML 格式互操作，按需）**

| 算子 | 说明 |
|---|---|
| `quantize_rows_q8_0` | 激活量化（Q8_0 块 + f16 scale），N1 的 `dot_f16` 之外的量化入口 |
| `dot_q4_0_q8_0` / `gemv_q4_0` | **llama.cpp 的 Q4_0 块布局**（18 B/32），与现有 `dot_q4` 的布局不同 |
| `dot_q6_K_q8_K` | §5 里 1.8× 的那一档 |

批次 N2 的价值是"能与 llama.cpp/GGUF 生态互通并可交叉验证"；如果目标只是给自己的
推理代码用，N1 就够，且 N1 的每个算子都能用现有 `api` 层与 `parallel` 多核设施包装。

**不建议做的**：q4_0/q4_K/q8_0/IQ* 的 vec_dot 重写（llama.cpp 已经贴上限，见 §5 结论 1）；
`GET_ROWS`/`VIEW`/`PERMUTE` 这类纯搬运（带宽决定，SIMD 无事可做）。

## 7. 复现与局限

```bash
# 版本
cd /home/lik/llama.cpp && git log --oneline -1
# 确认 LASX 开着
grep GGML_LASX build/CMakeCache.txt
objdump -d build/bin/libggml-cpu.so | grep -cE '\bxvf(madd|ld|st)'
# 逐算子
./build/bin/test-backend-ops perf -o MUL_MAT -b CPU
./build/bin/test-backend-ops perf -o SOFT_MAX,ROPE,ADD -b CPU
```

局限：

- `test-backend-ops perf` 是**单线程**、合成形状；真实推理有 OpenMP 多线程与 KV cache
  访存模式，绝对数字会不同（相对结论更可靠）；
- 本机有后台负载（load ≈ 5.7），跨次比较会有 ±10% 抖动；
- §4.3 的 `GB/s` 是 ggml 自己统计的"每 run 搬运字节"，对 `SOFT_MAX` 这种含 exp 的算子
  只反映访存，不代表计算饱和；
- `RMS_NORM`/`SILU`/`MUL`/`SCALE` 在这版 `test-backend-ops` 里没有 perf 用例（只进
  correctness 集），所以它们**没有实测数字**，只能按同族算子推断（softmax 同为逐元素 +
  归约，且同样缺 LoongArch 分支）。
