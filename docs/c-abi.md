# lasx_rs 的 C ABI（参考）

> 本文是 C ABI 的权威说明：符号、调用约定、状态码、内存/句柄生命周期与示例。
> Rust 侧用法与数值契约见 `docs/ops.md`；设计/性能见 `docs/dev.md`。
> **98 个导出符号**；最初的 15 个 `lasx_*` 符号签名与语义**永不改动**（只追加）。

## 1. 快速上手

```c
#include <lasx_rs.h>   /* 见 §6 的安装/头文件约定 */

int status = 0;
float dot = lasx_dot_checked(a, b, (int32_t)n, &status);
if (status != LASX_OK) { fprintf(stderr, "%s\n", lasx_status_message(status)); return 1; }

/* 矩阵乘：打包一次、多次运行 */
LasxMatmulPlan *plan = NULL;
status = lasx_matmul_prepare_checked(m, k, n, b, &plan);
if (status != LASX_OK) return 1;
status = lasx_matmul_run_checked(plan, m, a, c, NULL);
lasx_matmul_plan_destroy(plan);
```

- **优先用 `*_checked` 变体**：带长度校验，错误经 `int *status` 返回，不 panic、不静默。
- 无 `_checked` 的是原始快路径：调用方保证指针/长度合法（非法输入是未定义行为）。
- 长度参数是 `int32_t`（`n`/`m`/`k`/`rows`/`cols` 等）；`i32` 范围外的尺寸返回 `LengthMismatch`/`ShapeOverflow`。

## 2. 通用约定

| 项 | 约定 |
|---|---|
| 指针 | `const T*` 只读、`T*` 只写/读写；`NULL` 在 `_checked` 下返回 `NullPointer` |
| 长度 | `int32_t`；必须与指针指向的缓冲一致（`_checked` 会校验非空与正长度） |
| 状态 | `_checked` 变体把 `LasxStatus` 写入调用方给的 `int *status`（可为 `NULL`） |
| 返回值 | 标量算子经返回值返回结果；数组算子写调用方缓冲；`void` 算子的错误只能经 `status` 看 |
| 错误不 panic | 契约错误返回状态码；被捕获的 panic 返回 `PanicCaught`（库已 `catch_unwind`） |
| 别名 | `*_into` 语义的写缓冲默认不与输入重叠（重叠是未定义行为，调用方保证） |
| 线程 | 库内不自建线程池；`lasx_matmul_run` 由**调用方线程**驱动；`lasx_matmul_run_threaded` 用库的常驻池（见 §5） |
| 生命周期 | `prepare` 得到的 `LasxMatmulPlan*` 在 `destroy` 前有效；`destroy` 后使用是 UB |

## 3. 状态码（`LasxStatus`）

| 值 | 含义 | 典型触发 |
|---|---|---|
| `LASX_OK` (0) | 成功 | — |
| `NullPointer` | 空指针 | 必填指针为 NULL |
| `LengthMismatch` | 长度不符 | 缓冲与声明尺寸不一致 |
| `ShapeOverflow` | 形状乘积溢出 | `rows*cols` 超出 `i32` |
| `NonFiniteInput` | 需要有限输入时遇到 NaN/±inf | 量化/某些激活入口 |
| `UnsupportedPath` | 该路径不可用 | 需要 LASX-only 内核但 CPU 无 LASX |
| `InvalidPlan` | 计划句柄无效 | `run` 前未 `prepare` / 已 destroy |
| `PanicCaught` | 内部 panic 被捕获 | 违反文档承诺的调用（应视为 bug） |

`const char *lasx_status_message(int status)` 返回**可行动**的可读消息（含参数名/期望/实际）。

### 3.6 库级符号白名单（恰好 5 个）

库级（非算子）符号固定为以下 5 个；新增库级符号必须**同时**改 `scripts/namecheck.sh` 与本表：

| 符号 | 用途 |
|---|---|
| `lasx_alloc` / `lasx_free` | 跨 FFI 的内存分配/释放 |
| `lasx_has_lasx` | 当前 CPU 是否支持 LASX |
| `lasx_version` | 库版本字符串 |
| `lasx_status_message` | 状态码→可读消息 |

保留给库级的词根：`alloc` / `free` / `version` / `has` / `message` / `init` / `fini` / `config`
——白名单之外的算子符号不得使用（例如 `lasx_dot_alloc` 违规）。

## 4. 数据布局

- **行主序**：`A[m×k]`、`B[k×n]`、`C[m×n]`（f32/f64），`stride` 不单独传（连续）。
- `matmul_scaled`：`C = alpha·A·B`（beta 视为 0）。
- f16 权重：`uint16_t*`（IEEE binary16 位型），激活 f32。
- int8：`int8_t*`；量化参数 `float scale` / `int32_t zero`（per-tensor 或 per-row/per-token/per-channel）。
- 几何/物理批量：**SOA**（分量数组分离；`xs, ys, zs` 等分别传指针，长度相同）。
- 复数矩阵乘：实部/虚部分离布局（见 `lasx_cmatmul_*` 的参数说明）。
- 计划句柄：`LasxMatmulPlan*`（不透明指针；内部持有打包后的 B 与列映射）。

## 5. 符号总表（按族；共 98 个）

| 族 | 原始（永不改签名） | `_checked` 变体 |
|---|---|---|
| 标量归约 | `lasx_dot`, `lasx_sum`, `lasx_dot_f64`, `lasx_axpy`, `lasx_amax`, `lasx_absmax_rows` | `lasx_dot_checked`, `lasx_sum_checked`, `lasx_dot_f64_checked`, `lasx_axpy_checked`, `lasx_amax_checked`, `lasx_absmax_rows_checked` |
| 矩阵乘 | `lasx_matmul`, `lasx_matmul_f64`, `lasx_matmul_scaled` | `lasx_matmul_checked`, `lasx_matmul_f64_checked`, `lasx_matmul_scaled_checked` |
| 计划句柄 | `lasx_matmul_prepare`, `lasx_matmul_run`, `lasx_matmul_run_threaded`, `lasx_matmul_plan_destroy` | `lasx_matmul_prepare_checked`, `lasx_matmul_run_checked`, `lasx_matmul_run_threaded_checked` |
| 复数矩阵乘 | `lasx_cmatmul_f32`, `lasx_cmatmul_f64` | `lasx_cmatmul_f32_checked`, `lasx_cmatmul_f64_checked` |
| 量化 | `lasx_quantize_i8_per_tensor`, `lasx_quantize_i8_per_row` | `lasx_quantize_i8_per_tensor_checked`, `lasx_quantize_i8_per_row_checked` |
| 反量化 | `lasx_dequantize_i8`, `lasx_dequantize_i8_rows` | `lasx_dequantize_i8_checked`, `lasx_dequantize_i8_rows_checked` |
| int8 推理 | `lasx_dot_i8`, `lasx_dot_q4`, `lasx_gemv_i8`, `lasx_matmul_i8` | `lasx_dot_i8_checked`, `lasx_dot_q4_checked`, `lasx_gemv_i8_checked`, `lasx_matmul_i8_checked` |
| NN | `lasx_softmax_rows`, `lasx_rms_norm`, `lasx_layer_norm`, `lasx_silu`, `lasx_gelu_quick`, `lasx_gelu_erf`, `lasx_rope` | `lasx_softmax_rows_checked`, `lasx_rms_norm_checked`, `lasx_layer_norm_checked`, `lasx_silu_checked`, `lasx_gelu_quick_checked`, `lasx_gelu_erf_checked`, `lasx_rope_checked` |
| f16 | `lasx_dot_f16`, `lasx_gemv_f16` | `lasx_dot_f16_checked`, `lasx_gemv_f16_checked` |
| 几何 f64 | `lasx_norm3_batch`, `lasx_vec3_add_scaled_batch`, `lasx_cross3_batch`, `lasx_unitize3_batch`, `lasx_mat3_mul_vec3_batch`, `lasx_quat_normalize_batch`, `lasx_quat_mul_batch`, `lasx_quat_rotate_batch`, `lasx_quat_to_dcm_batch` | 对应 `*_checked`（同族） |
| 物理批量 | `lasx_batch_distance2d`, `lasx_ballistic_step`, `lasx_j2_accel_batch`, `lasx_rk4_j2_step_batch` | 对应 `*_checked` |
| 索引 | `lasx_gather_rows`, `lasx_gather_rows_i8` | `lasx_gather_rows_checked`, `lasx_gather_rows_i8_checked` |
| 运行期 | `lasx_has_lasx`, `lasx_version`, `lasx_alloc`, `lasx_free`, `lasx_status_message` | — |

## 6. 计划句柄（打包一次、多次运行）

```c
LasxMatmulPlan *plan = NULL;
int st = lasx_matmul_prepare_checked(m, k, n, b, &plan);   /* 打包 B（+列映射），失败时 plan=NULL */
/* 之后可反复： */
for (int iter = 0; iter < iters; ++iter) {
    lasx_matmul_run_checked(plan, m, a, c, &st);            /* 调用方线程驱动；逐位 */
    /* 或：lasx_matmul_run_threaded_checked(plan, m, a, c, threads, &st); 用库的常驻池 */
}
lasx_matmul_plan_destroy(plan);   /* 必须调用；之后再用 plan 是 UB */
```

- `prepare` 之后 `plan` **只读**（可被多线程同时 `run`？——当前约定：一个 plan 同时只被一个线程 `run`，
  并发共享需调用方加锁）。
- 线程模型：`run` 不建线程（调用方驱动）；`run_threaded` 用库内常驻池（`threads` 参数给线程数）。
- 小形状固定开销：C ABI 比 Rust 切片 API 多约 6 ns/调用（n=64），高频小调用建议批量合并。

## 7. 错误处理与可行动消息

- 所有 `_checked` 入口在失败时**不改输出缓冲**（或写 0），调用方应先检查 `status` 再读结果。
- `lasx_status_message(status)` 给中文可读消息（含期望/实际）；建议日志里同时打印 `status` 数值。
- 违反文档承诺（如别名重叠）⇒ 未定义行为；库无法检测时不返回错误。

## 8. 版本与能力

```c
int has = lasx_has_lasx();          /* 1 = 当前 CPU 支持 LASX */
const char *ver = lasx_version();   /* 库版本字符串，静态生命周期 */
```

- 路径自适配：无 LASX 时通用实现仍逐位正确；只有"必须 LASX-only 内核"的调用会得到 `UnsupportedPath`。
- `lasx_alloc`/`lasx_free` 成对使用（跨 FFI 分配/释放；不要用调用方的 malloc/free 混用）。

## 9. 命名规范

### 9.1 四条规则（`scripts/namecheck.sh` 的权威出处）

1. **形态**：`^lasx_[a-z0-9]+(_[a-z0-9]+)*$`——无大写、无连续下划线、无首尾下划线。
2. **后缀**：尾 token 不得是"版本/口味"词（`v1`–`v9`、`ex`、`fast`、`slow`、`new`、`old`、`alt`、`lite`、`pro`、`plus`、
   `tmp`、`test`、`bak`、`experimental`、`legacy`）；`_checked` 是合法后缀，先剥掉再判。
3. **孪生**：每个 `*_checked` 必须有同名裸孪生（反之不要求）。
4. **库级**：库级白名单**恰好 5 个**（见 §3.6）且必须在产物中；白名单之外不得出现库级词根（见 §3.6 的保留词根表）。

## 10. 命名门禁常量（与 `scripts/namecheck.sh` 同步的守卫表）

| 常量 | 取值 |
|---|---|
| `SUFFIX_BLACKLIST` | `v1 v2 v3 v4 v5 v6 v7 v8 v9 ex fast slow new old alt lite pro plus tmp test bak experimental legacy` |
| `LIB_WHITELIST` | `lasx_alloc lasx_free lasx_has_lasx lasx_status_message lasx_version` |
| `LIB_TOKENS` | `alloc free version has message init fini config` |

> 改上述任一常量 ⇒ 必须同步本表与 §9.1（脚本头部有同样提示）。

## 11. 完整示例（dot + matmul）

```c
#include <stdio.h>
#include <stdint.h>
#include <lasx_rs.h>

int main(void) {
    const int m = 4, k = 4, n = 4;
    float a[16], b[16], c[16];
    for (int i = 0; i < 16; ++i) { a[i] = (float)i; b[i] = (float)(16 - i); }
    int st = 0;
    float d = lasx_dot_checked(a, b, 16, &st);
    if (st != 0) { printf("dot failed: %s\n", lasx_status_message(st)); return 1; }
    LasxMatmulPlan *plan = NULL;
    st = lasx_matmul_prepare_checked(m, k, n, b, &plan);
    if (st != 0) return 1;
    st = lasx_matmul_run_checked(plan, m, a, c, 0);
    lasx_matmul_plan_destroy(plan);
    printf("dot=%f c[0]=%f status=%d\n", d, c[0], st);
    return st != 0;
}
```

安装/头文件：`bash scripts/install.sh`（装 `liblasx_rs.so`、头文件与 pkg-config 文件）；
依赖方可 `pkg-config --cflags --libs lasx_rs` 获取编译/链接参数。
