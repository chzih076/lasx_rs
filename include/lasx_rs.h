/*
 * lasx_rs.h — LoongArch64 LASX(256 位)/LSX(128 位) 批量数值内核库的 C ABI。
 *
 * 生产用头文件：与库产物 `liblasx_rs.so`（SONAME `liblasx_rs.so.0`）/ `liblasx_rs.a`
 * 配套安装。**权威说明见 `docs/c-abi.md`**——C 侧的符号清单、逐符号原型、内存与安全模型、
 * 能力探测与降级、线程模型、ABI 稳定性策略都在那一份里，读它即可，不必再翻别处。
 * **本文件是手写并逐符号校验的**——权威符号清单是 `docs/c-abi.md` 第 3 节（94 个：
 * 50 个裸符号 + 44 个 `_checked` 变体），`scripts/check_header.c` 对本头文件里的
 * 每一个符号取地址（缺符号/改名/原型不匹配会在编译或链接期失败），并实测若干行为。
 *
 * 编译提示（C / C++ 均可 include）：
 *     cc -O2 -Iinclude app.c -L/path/to/lib -llasx_rs -lm
 * 静态链接：把 `-llasx_rs` 指向 `liblasx_rs.a`（loongarch64 目标）。
 *
 * ---------------------------------------------------------------------------
 * 契约摘要（细节以 docs/ops.md 为准）
 * ---------------------------------------------------------------------------
 * 数值契约与位精确（docs/ops.md §2）：
 *   - "位精确"指**同一算子**的 LASX / LSX / 标量尾 / 打包 / k 分块 / 并行切块等路径
 *     对同一输入给出**逐位相同**的结果（§2.1、§2.5）。跨实现对照是"相对误差 <1e-9"
 *     一档，不是逐位。
 *   - 例外仅两处（§2.1、§2.5 第 4 点）：`lasx_ballistic_step` 的向量/标量分支结合序
 *     不同（物理一致，有绝对误差测试）；`lasx_axpy` 允许 1 ulp 以内差异。
 *   - 累加次序是**硬契约**（§2.2、§2.5 第 1 点）：每个输出元素沿 k 升序、单条 FMA
 *     累加；写参考实现时必须照抄算子文档里的 op 序列。
 *   - `lasx_matmul_scaled` 的融合语义见 §2.5 第 3 点与 §5.5 末段：施加次序固定为
 *     `out = alpha·acc + beta·c_old`（两次舍入），`beta == 0` 时**不读 c**。
 *
 * `_checked` 变体（docs/ops.md §4.4、§4.5）：
 *   - 与对应裸符号**签名完全一致，仅在末尾多一个 `int *status` 出参**；
 *   - 先校验，失败时写入状态码并返回**安全中性值**（数值型 `0.0`/`0`，`void` 型只写
 *     状态），且**不触碰输出缓冲**；成功时写 `0`（`LASX_OK`）；
 *   - `status` 可以传 `NULL`（表示不关心原因，但校验仍执行）；
 *   - 校验只覆盖**结构**（空指针、负长度、乘积溢出、物理常数）；"缓冲真实长度是否
 *     等于声明的形状"只有知道长度的上层判得了（§4.4 末段），违反是读写越界。
 *
 * 线程安全（docs/ops.md §3.1、§9、§11）：
 *   - 所有内核**无逐调用全局可写状态**：能力探测结果进程级缓存一次、之后只读；
 *   - 不同线程用**不相交缓冲**并发调用是安全的；同一输出缓冲的并发写由调用方同步；
 *   - C ABI **不导出线程池**：多核需调用方自行切子区间（§9）；也不导出
 *     `lasx_force_lsx_thread`（仅 Rust rlib 可见，§3.4）；
 *   - 函数**不保留指针、不回调、不异步**：返回后输入/输出缓冲即可释放（§11）。
 *
 * 指令集要求（docs/ops.md §3.2、§3.3）：
 *   - 标了 `[LASX-only]` 的符号**没有降级分支**：在只实现 LSX(128 位) 的机型
 *     （如 2K3000 / 3B6000M，LA364E 核）上调用会执行 LASX 指令 → **非法指令(SIGILL)**；
 *   - **调任何 `[LASX-only]` 符号之前，先用 `lasx_has_lasx()` 判断**（见下面第 0 节）：
 *     返回 0 就说明当前生效路径只剩 LSX，此时只能调 `LASX+LSX` 的符号。它问的是
 *     **当前生效路径**（与算子走的路同源），所以不必也不该自己去读 `cpucfg`；
 *   - `la464`(3A5000) / `la664`(3A6000 等) 支持 128/256 位两档，不属于上面那类机型。
 *
 * 内存（docs/ops.md §10）：
 *   - `lasx_alloc` 返回 32 字节对齐的**未初始化** f32 缓冲；**必须用 `lasx_free(p, n)` 归还**
 *     （`n` 传当初分配时那个值）——两者共用同一 layout，"对齐"与"归还"是同一个契约的两半；
 *   - **不要**用 `free()` / `posix_memalign` 配对的释放函数代替 `lasx_free`：库走的是 Rust
 *     全局分配器的 alloc/dealloc 路径，本机恰好是 glibc malloc 只是实现细节，**不是 ABI 契约**；
 *   - `lasx_free(NULL, n)` 与 `lasx_free(p, n <= 0)`（含 `lasx_alloc(0)` 返回的对齐悬垂指针）
 *     都是**安全的 no-op**，不必在调用方特判；
 *   - 对齐只影响性能、不影响正确性：非 32 字节对齐的缓冲结果照样正确（§10）。
 *
 * 其它 ABI 约定（docs/ops.md §4.1 末段、§11）：
 *   - 长度/维度/形状参数一律是 C `int`（32 位，Rust `i32`）；标量按值传；
 *     输入 `const T *`、输出 `T *`；
 *   - 行主序（row-major）：矩阵/批量缓冲的内存布局按各函数注释；
 *   - 允许输出与输入**别名**（`out == in`）的算子会在注释里写明（逐元素与批量姿态/
 *     几何内核允许；矩阵乘亦允许 `c == a` 之类按 §5.5 的实现约定）。
 *
 * 本文件不引入任何第三方依赖，只需要 `<stdint.h>`。
 */

#ifndef LASX_RS_H
#define LASX_RS_H

/* ========================================================================
 * 架构防呆：本头文件只对应 **LoongArch64** 的 C ABI
 *
 * `lasx_rs` 是 LoongArch LASX/LSX 的**指令集绑定**实现（手写内在函数/内联汇编），
 * **不是可移植标量库**：在别的架构上编译/链接它没有意义，运行期更不可能可用。
 * 所以这里在**编译期**直接拦下——错架构的 `#include` 会立刻得到一条人话，
 * 而不是等到链接或运行时才发现。
 *
 * **逃生阀 `LASX_RS_ALLOW_FOREIGN_ARCH`**（给交叉编译 / 文档生成 / 头文件语法检查用）：
 * 定义它即可跳过这道编译期检查。请**只**在这些场景用它——它**只关掉编译期检查**，
 * 不会让运行时可用（库产物本身仍是 LoongArch 机器码，得在 LoongArch64 上运行）。
 *
 * 与之**不同**的另一件事：LSX-only 机型（**无 LASX**，如 2K3000/3B6000M）是
 * **能构建、能运行**的，只是不能调标了 `[LASX-only]` 的符号（会 SIGILL）。那是运行期问题，
 * 用 `lasx_has_lasx()` 防呆（见下面第 0 节）。**别把"错架构"和"无 LASX"混成一件事。**
 * ======================================================================== */

#if !defined(__loongarch__) && !defined(LASX_RS_ALLOW_FOREIGN_ARCH)
#error "lasx_rs 仅提供 LoongArch64 的 C ABI（LASX/LSX 指令集绑定）。错架构请勿链接；如确需交叉编译请显式定义 LASX_RS_ALLOW_FOREIGN_ARCH。"
#endif

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * 状态码（docs/ops.md §4.5；与 Rust 的 `LasxStatus` 逐值对应）
 * ======================================================================== */

typedef enum lasx_status {
    LASX_OK = 0,                    /* 成功 */
    LASX_NULL_POINTER = 1,          /* 指针为空且长度非 0 */
    LASX_NEGATIVE_LENGTH = 2,       /* 长度参数为负 */
    LASX_BAD_SHAPE = 3,             /* 形状/长度自洽性不成立 */
    LASX_SIZE_OVERFLOW = 4,         /* 尺寸相乘溢出 */
    LASX_NON_POSITIVE_CONSTANT = 5, /* mu <= 0 或 re <= 0 */
    LASX_NON_FINITE_CONSTANT = 6    /* j2/dt 为 NaN 或无穷 */
} lasx_status;

/* 便捷别名（与 §4.5 的短名一致；用宏是因为 C 侧常写 `st != LASX_OK`）。 */
#define LASX_STATUS_OK LASX_OK
#define LASX_STATUS_NULL_POINTER LASX_NULL_POINTER
#define LASX_STATUS_NEGATIVE_LENGTH LASX_NEGATIVE_LENGTH
#define LASX_STATUS_BAD_SHAPE LASX_BAD_SHAPE
#define LASX_STATUS_SIZE_OVERFLOW LASX_SIZE_OVERFLOW
#define LASX_STATUS_NON_POSITIVE_CONSTANT LASX_NON_POSITIVE_CONSTANT
#define LASX_STATUS_NON_FINITE_CONSTANT LASX_NON_FINITE_CONSTANT

/* ========================================================================
 * 0. 运行时元信息与内存归还（docs/ops.md §4.8、§3.1、§10）
 *
 * **先读这一节再写调用代码**——这 4 个符号就是为了让调用方不必靠"读 CPU 寄存器 +
 * 猜版本 + 硬编码错误码 + 赌释放函数"来用这个库：
 *   - lasx_has_lasx()      ：能不能调 [LASX-only]（LSX-only 机型上硬调 = SIGILL）
 *   - lasx_version()       ：链接到的是哪一版
 *   - lasx_status_message(): 把 `_checked` 的 status 翻成人能读的串（不必硬编码 0..6）
 *   - lasx_free()          ：与 lasx_alloc 配对的归还（free() 不是契约）
 * ======================================================================== */

/* 当前**生效路径**是否为 LASX：1 = 是，0 = 仅 LSX。
 *
 * **调用任何标了 [LASX-only] 的符号之前必须先问它**（lasx_dot / lasx_sum / lasx_matmul /
 * lasx_dot_i8 / 全部 int8 内核 …；标了 LASX+LSX 的符号两种机型都安全，不必问）。
 * 返回 0 时只能调 LASX+LSX 的符号，否则在 LSX-only 机型上执行 LASX 指令 → SIGILL。
 *
 * 它与算子入口**同源**（都走 SimdPath::detect()），所以回答的是"现在能不能安全调"，
 * 而不是"这块 CPU 有没有 LASX"：被 lasx_force_lsx_thread 强制降级时同样返回 0，
 * 与算子实际会走的路一致。
 *
 * 代价：一次进程级原子读 + 一次线程局部读（~0.2–0.6 ns），按批次问一次即可。 */
int lasx_has_lasx(void);

/* 库版本（`CARGO_PKG_VERSION`，如 "0.1.0"）。
 * **永不为 NULL**，指向静态 NUL 结尾字符串，生命周期 = 整个进程；**不要 free**、
 * 不要写它。可直接 printf("%s") / strcmp，不需要长度参数。 */
const char *lasx_version(void);

/* 状态码（0..6，见上面的 lasx_status 枚举）的人类可读说明。
 * **永不为 NULL**：任何未知整数（负数、>6）返回通用串"未知状态码"。
 * 返回静态 NUL 结尾字符串，**不要 free**。用法：
 *     int st;
 *     float v = lasx_dot_checked(a, b, n, &st);
 *     if (st != LASX_OK) fprintf(stderr, "lasx_dot: %s\n", lasx_status_message(st)); */
const char *lasx_status_message(int status);

/* 归还 lasx_alloc 分配的缓冲：`n` 必须是当初传给 lasx_alloc 的**同一个值**。
 * 两者共用同一 layout（"对齐"与"归还"是同一契约的两半，见 §10）。
 * **必须用它归还，不要用 free()**（本机 glibc 下偶然可用，但不是 ABI 契约）。
 * 两种 no-op（安全、不崩、不必调用方特判）：`p == NULL`；`n <= 0`——后者包括
 * `lasx_alloc(0)` 返回的那个"对齐但不可解引用"的悬垂指针（它从来不是堆块）。 */
void lasx_free(float *p, int n);

/* ========================================================================
 * 1. 归约、稠密线性代数、量化点积、物理（docs/ops.md §4.1、§4.7）
 * ======================================================================== */

/* 语义：sum_{i<n} a[i]*b[i]。n < 24 走纯标量 f32 累加；跨 n=24 边界不保证逐位连续。
 * 路径：LASX+LSX。长度契约：a、b 各至少 n 个 f32。 */
float lasx_dot(const float *a, const float *b, int n);

/* lasx_dot 的 _checked 变体：末尾多 `int *status`（下同，不再逐条重复说明）。 */
float lasx_dot_checked(const float *a, const float *b, int n, int *status);

/* 语义：sum_{i<n} x[i]。精度：相对独立 f64 参考 <1e-5。
 * 路径：[LASX-only]。长度契约：x 至少 n 个 f32。 */
float lasx_sum(const float *x, int n);

float lasx_sum_checked(const float *x, int n, int *status);

/* 语义：sum a[i]*b[i]（f64）。路径：[LASX-only]。
 * 长度契约：a、b 各至少 n 个 f64。 */
double lasx_dot_f64(const double *a, const double *b, int n);

double lasx_dot_f64_checked(const double *a, const double *b, int n, int *status);

/* 语义：y[i] += alpha*x[i]（原地写 y，不改 x）。路径：[LASX-only]。
 * 精度：向量单舍入 vs 标量两次舍入，至多差 1 ulp，无逐位守护（§2.1）。
 * 长度契约：x、y 各至少 n 个 f32。 */
void lasx_axpy(float alpha, const float *x, float *y, int n);

void lasx_axpy_checked(float alpha, const float *x, float *y, int n, int *status);

/* 语义：C[m*n] = A[m*k] * B[k*n]，A/B/C 均行主序。路径：[LASX-only]。
 * 精度：每个输出元素沿 k 的单个 f32 累加器（无 f64 落盘），相对 f64 参考 <1e-4。
 * 长度契约：a 至少 m*k、b 至少 k*n、c 至少 m*n 个 f32。
 * 边界：m == 0 || n == 0 直接返回；k == 0 时不写 C。 */
void lasx_matmul(int m, int k, int n, const float *a, const float *b, float *c);

void lasx_matmul_checked(
    int m, int k, int n, const float *a, const float *b, float *c, int *status);

/* 语义：C = alpha*(A*B) + beta*C（原地，行主序，CBLAS 参数次序）。路径：[LASX-only]。
 * 契约（§2.5 第 3 点、§5.5 末段）：
 *   - alpha == 1 && beta == 0 时与 lasx_matmul **逐位一致**；
 *   - beta == 0 时**不读 c**（c 无需在入参里有效）；施加次序固定为
 *     `out = alpha*acc + beta*c_old`（两次舍入），不是"用 beta*c 给累加器播种"；
 *   - beta != 0 且 k > 384 时退回与朴素二遍法同构的实现（不假装更快）。
 * 长度契约：a 至少 m*k、b 至少 k*n、c 至少 m*n 个 f32（beta != 0 时 c 还要可读）。 */
void lasx_matmul_scaled(
    int m, int k, int n, float alpha, const float *a, const float *b, float beta, float *c);

void lasx_matmul_scaled_checked(int m, int k, int n, float alpha, const float *a,
                                const float *b, float beta, float *c, int *status);

/* ------------------------------------------------------------------------
 * 1.1 矩阵乘**计划句柄**：把 B 打包一次，之后按块/按线程反复用
 *
 * 动机（docs/dev.md §7.10 ⑤ 有实测）：lasx_matmul 每次调用都要把**整个 B 重新打包**，
 * 多线程下"每个线程各调一次 lasx_matmul" ⇒ 打包量 = 线程数 × B，12 线程实测扩展比只有
 * 1.08×。计划把这笔开销提到调用之前；**线程怎么开、块怎么切由调用方决定**。
 *
 * 线程模型（两档，可以只用第一档；库**不**把"库内固定池"当唯一路径）：
 *   ① **调用方驱动（推荐）**：`lasx_matmul_run(plan, rows, a + r0*k, c + r0*n)`，用你自己的
 *      线程池 / OpenMP / std::thread / 协程 / 单线程串行都行。同一个 plan 可以被多个线程
 *      **并发**调用（各写自己的 c）——prepare 之后 plan 是**只读**的。
 *   ② **便利层**：`lasx_matmul_run_threaded(..., threads)`，库内派发；`threads <= 0` = 自动取
 *      物理核、`1` = 串行（不建线程）、`N` = N 个 worker（超物理核夹到物理核），动态取块。
 *      池**每次调用现建现拆** ⇒ **不用它就完全零进程级状态**（无全局池、不读环境变量、无 atexit）。
 *      热循环里想省这笔开销就用 ①。
 *
 * 生命周期 / 所有权（**契约**）：
 *   - `prepare` 会**复制并打包** `b`：返回后调用方可以释放 `b`；plan 归调用方所有；
 *   - plan 在 prepare 之后**只读**：可跨线程共享、可反复 run，直到 destroy；
 *   - `lasx_matmul_plan_destroy(NULL)` 是合法空操作；**重复 destroy、或 destroy 之后再用 = UB**；
 *   - `prepare_checked` 失败返回 `NULL` 且只写 `status`（`status` 可传 `NULL`）。
 *
 * 位精确：`run` / `run_threaded` 与单线程 `lasx_matmul` **逐位一致**（每个输出元素仍沿 k
 * 单累加器升序；切块位置、线程数、`rows` 是否为 4 的倍数都不改变累加次序）。
 *
 * 内存：约 `k*n*4` 字节（打包面板），`n` 不是 32 的倍数时另加一小段列尾副本。
 * ------------------------------------------------------------------------ */

/** 不透明计划句柄：只能当 token 传，**不要**解引用 / `sizeof` / 复制其内容。 */
typedef struct lasx_matmul_plan lasx_matmul_plan;

/* 打包 B（行主序 k×n）一次并返回计划句柄；失败返回 NULL（要原因用 _checked）。
 * 长度契约：b 至少 k*n 个 f32。 */
lasx_matmul_plan *lasx_matmul_prepare(int k, int n, const float *b);

/* lasx_matmul_prepare 的 _checked 变体。校验顺序：k/n 非负 → k×n 不溢出 → b 非空 →
 * 计划构造（k*n 与 b 的可用长度不符只有调用方判得了，见 docs/c-abi.md §11）。 */
lasx_matmul_plan *lasx_matmul_prepare_checked(int k, int n, const float *b, int *status);

/* 用计划算**一个行块**：a 是该块的 rows×k、c 是该块的 rows×n（都行主序、都指向块首）。
 * rows == 0 是合法空操作；rows 与 4 无关；同一 plan 可被多线程并发调用不同块。 */
void lasx_matmul_run(const lasx_matmul_plan *plan, int rows, const float *a, float *c);

/* lasx_matmul_run 的 _checked 变体：plan 非空 → rows 非负 → rows×k / rows×n 不溢出 →
 * a/c 非空且长度约定；**失败时不触碰 c**。 */
void lasx_matmul_run_checked(const lasx_matmul_plan *plan, int rows, const float *a, float *c,
                             int *status);

/* 便利层：库内派发，算整块 C[m*n] = A[m*k]*B。
 * threads <= 0 = 自动（物理核）、1 = 串行、N = N 个 worker（夹到物理核）。 */
void lasx_matmul_run_threaded(const lasx_matmul_plan *plan, int m, const float *a, float *c,
                              int threads);

/* _checked + **执行自检**出参（四个都可传 NULL）：
 *   threads_out 这次实际用的 worker 数（1 = 串行）——用它确认**没有静默退化成串行**；
 *   chunks_out  实际执行块数；ms_max_out / ms_min_out 最慢 / 最快一块的毫秒数。 */
void lasx_matmul_run_threaded_checked(const lasx_matmul_plan *plan, int m, const float *a, float *c,
                                      int threads, int *threads_out, int *chunks_out,
                                      double *ms_max_out, double *ms_min_out, int *status);

/* 释放计划（NULL 合法；重复释放 / 用后即用 = UB）。
 * 名字用 _destroy 而不是 _free：`free` 是 §9.1 保留给库级符号的词根（白名单已有 lasx_free）。 */
void lasx_matmul_plan_destroy(lasx_matmul_plan *plan);

/* 语义：C[m*n] = A[m*k] * B[k*n]（f64，行主序）。路径：[LASX-only]。
 * 精度：全程 f64，相对 f64 参考 <1e-12。
 * 长度契约：a 至少 m*k、b 至少 k*n、c 至少 m*n 个 f64。 */
void lasx_matmul_f64(int m, int k, int n, const double *a, const double *b, double *c);

void lasx_matmul_f64_checked(
    int m, int k, int n, const double *a, const double *b, double *c, int *status);

/* 语义：sum a[i]*b[i]，int8 输入、**i32 返回**（整数精确：i16 乘积 → i32 累加 →
 * 每 1024 字节落 i64）。路径：[LASX-only]。
 * 边界：总点积超出 i32 范围时**回绕**（无饱和/报错），约 n > 13.3 万（全 ±127）时可能发生。
 * 长度契约：a、b 各至少 n 个 int8_t。 */
int lasx_dot_i8(const int8_t *a, const int8_t *b, int n);

int lasx_dot_i8_checked(const int8_t *a, const int8_t *b, int n, int *status);

/* 语义：Q4 量化点积，返回 f64。路径：[LASX-only]。
 * 量化格式：每个 uint8_t 含 2 个 4-bit 无符号 nibble（低 = q & 0x0f，高 = (q>>4)&0x0f）；
 * sa/sb 每 32 字节一组一个 f32 scale，组数 = ceil(n_bytes/32)。
 * 长度契约：qa/qb 各至少 n_bytes 个 uint8_t；sa/sb 各至少 ceil(n_bytes/32) 个 f32。 */
double lasx_dot_q4(const uint8_t *qa, const float *sa, const uint8_t *qb, const float *sb,
                   int n_bytes);

double lasx_dot_q4_checked(const uint8_t *qa, const float *sa, const uint8_t *qb,
                           const float *sb, int n_bytes, int *status);

/* 分配 n 个 f32 的 **32 字节对齐、未初始化** 缓冲；n == 0 返回"对齐但不可解引用"的
 * 非空悬垂指针。**必须用 `lasx_free(p, n)` 按同一个 n 归还**（§10、上面第 0 节）：
 * 两者共用同一 layout；`free()` 在本机 glibc 下偶然可用，但**不是 ABI 契约**。
 * 库级元信息符号没有 `_checked` 变体——白名单与清单的唯一权威是 `docs/c-abi.md` §9.1；
 * 算子级哪个裸符号带 `_checked`，以 `docs/c-abi.md` 第 3 节的逐符号表为准。 */
float *lasx_alloc(int n);

/* 语义：out[i] = sqrt(xs[i]^2 + ys[i]^2 + zs[i]^2)（f64 SOA）。路径：LASX+LSX。
 * 结合序固定为 x^2 + (y^2 + z^2)（右结合）；相对独立标量参考 <1e-9。
 * 长度契约：xs/ys/zs/out 各至少 n 个 f64。 */
void lasx_norm3_batch(const double *xs, const double *ys, const double *zs, double *out, int n);

void lasx_norm3_batch_checked(const double *xs, const double *ys, const double *zs, double *out,
                              int n, int *status);

/* 语义：o[i] = a[i] + s*b[i]（x/y/z 三分量独立，f64 SOA）。路径：LASX+LSX。
 * 向量路径单条 FMA，标量尾同式；相对独立标量参考 <1e-9。
 * 长度契约：6 个输入与 3 个输出各至少 n 个 f64。 */
void lasx_vec3_add_scaled_batch(const double *ax, const double *ay, const double *az,
                                const double *bx, const double *by, const double *bz, double s,
                                double *ox, double *oy, double *oz, int n);

void lasx_vec3_add_scaled_batch_checked(const double *ax, const double *ay, const double *az,
                                        const double *bx, const double *by, const double *bz,
                                        double s, double *ox, double *oy, double *oz, int n,
                                        int *status);

/* 语义：out[i] = sqrt((xs[i]-px)^2 + (ys[i]-py)^2)。路径：LASX+标量（非 LASX 时全标量）。
 * 长度契约：xs/ys/out 各至少 n 个 f32。 */
void lasx_batch_distance2d(float px, float py, const float *xs, const float *ys, float *out,
                           int n);

void lasx_batch_distance2d_checked(float px, float py, const float *xs, const float *ys,
                                   float *out, int n, int *status);

/* 语义：SOA 批量弹道欧拉步，n 发弹（每发 6 状态 + 阻力系数 k）**原地**推进一步。
 * 路径：LASX+标量。注意：向量路径与标量 euler_step **非逐位一致**（|v|^2 结合序不同，
 * 物理一致，有绝对误差测试守护，§2.1/§6.4）。
 * 长度契约：x/y/z/vx/vy/vz 各至少 n 个可读写 f32；k 至少 n 个 f32。 */
void lasx_ballistic_step(float *x, float *y, float *z, float *vx, float *vy, float *vz,
                         const float *k, int n, float dt, float g);

void lasx_ballistic_step_checked(float *x, float *y, float *z, float *vx, float *vy, float *vz,
                                 const float *k, int n, float dt, float g, int *status);

/* 语义：中心引力 + J2 摄动加速度（f64 SOA，与 loong-sci EarthModel::j2_acceleration 同式）。
 * 路径：LASX+LSX；每样本独立、无跨 lane 归约，分块内逐位一致。
 * _checked 额外要求 mu > 0、re > 0、j2 有限（否则 5 / 6 号状态码）。
 * 长度契约：rx/ry/rz/ax/ay/az 各至少 n 个 f64。 */
void lasx_j2_accel_batch(const double *rx, const double *ry, const double *rz, double mu,
                         double j2, double re, double *ax, double *ay, double *az, int n);

void lasx_j2_accel_batch_checked(const double *rx, const double *ry, const double *rz, double mu,
                                 double j2, double re, double *ax, double *ay, double *az, int n,
                                 int *status);

/* 语义：经典 4 阶 Runge-Kutta 单步（中心项 + J2 力模型），批量推进 n 个 [r(3), v(3)]，
 * **原地更新**。路径：LASX+标量（非 LASX 走全标量同式实现，分块内逐位一致）。
 * _checked 额外要求 mu > 0、re > 0、j2/dt 有限。
 * 长度契约：rx/ry/rz/vx/vy/vz 各至少 n 个可读写 f64。 */
void lasx_rk4_j2_step_batch(double *rx, double *ry, double *rz, double *vx, double *vy,
                            double *vz, double mu, double j2, double re, double dt, int n);

void lasx_rk4_j2_step_batch_checked(double *rx, double *ry, double *rz, double *vx, double *vy,
                                    double *vz, double mu, double j2, double re, double dt, int n,
                                    int *status);

/* ========================================================================
 * 2. 批量姿态与几何（f64 SOA，docs/ops.md §4.2、§7）
 *
 * 公共约定（§7）：四元数**标量在前** q = w + xi + yj + zk；旋转方向是**体→惯**；
 * |q| < 1e-15 视为退化 ⇒ 输出单位四元数 (1,0,0,0)；LASX / LSX / 标量尾三路逐位一致；
 * **全部内核允许输出与输入别名**（向量路径先取完本轮输入再写回）。
 * ======================================================================== */

/* o = a x b（右手系）。长度契约：ax..bz、ox..oz 各至少 n 个 f64。 */
void lasx_cross3_batch(const double *ax, const double *ay, const double *az, const double *bx,
                       const double *by, const double *bz, double *ox, double *oy, double *oz,
                       int n);

void lasx_cross3_batch_checked(const double *ax, const double *ay, const double *az,
                               const double *bx, const double *by, const double *bz, double *ox,
                               double *oy, double *oz, int n, int *status);

/* o = v/|v|；|v| == 0 ⇒ (0,0,0)（不是 NaN）。长度契约：x/y/z、ox/oy/oz 各至少 n 个 f64。 */
void lasx_unitize3_batch(const double *x, const double *y, const double *z, double *ox,
                         double *oy, double *oz, int n);

void lasx_unitize3_batch_checked(const double *x, const double *y, const double *z, double *ox,
                                 double *oy, double *oz, int n, int *status);

/* o = M*v；m0..m8 为行主序 3x3（M[r][c] = m[3r+c]），每行左结合。
 * 长度契约：m0..m8、x/y/z、ox/oy/oz 各至少 n 个 f64。 */
void lasx_mat3_mul_vec3_batch(const double *m0, const double *m1, const double *m2,
                              const double *m3, const double *m4, const double *m5,
                              const double *m6, const double *m7, const double *m8,
                              const double *x, const double *y, const double *z, double *ox,
                              double *oy, double *oz, int n);

void lasx_mat3_mul_vec3_batch_checked(const double *m0, const double *m1, const double *m2,
                                      const double *m3, const double *m4, const double *m5,
                                      const double *m6, const double *m7, const double *m8,
                                      const double *x, const double *y, const double *z, double *ox,
                                      double *oy, double *oz, int n, int *status);

/* 四元数**原地**单位化；|q| < 1e-15 ⇒ (1,0,0,0)。长度契约：qw/qx/qy/qz 各至少 n 个可读写 f64。 */
void lasx_quat_normalize_batch(double *qw, double *qx, double *qy, double *qz, int n);

void lasx_quat_normalize_batch_checked(double *qw, double *qx, double *qy, double *qz, int n,
                                       int *status);

/* Hamilton 积 o = a (x) b（**不做内部归一化**）。标量在前。
 * 长度契约：aw..az、bw..bz、ow..oz 各至少 n 个 f64。 */
void lasx_quat_mul_batch(const double *aw, const double *ax, const double *ay, const double *az,
                         const double *bw, const double *bx, const double *by, const double *bz,
                         double *ow, double *ox, double *oy, double *oz, int n);

void lasx_quat_mul_batch_checked(const double *aw, const double *ax, const double *ay,
                                 const double *az, const double *bw, const double *bx,
                                 const double *by, const double *bz, double *ow, double *ox,
                                 double *oy, double *oz, int n, int *status);

/* o = R(q)*v（体→惯）；输入**先单位化**（含退化规则）。
 * 长度契约：qw..qz、vx/vy/vz、ox/oy/oz 各至少 n 个 f64。 */
void lasx_quat_rotate_batch(const double *qw, const double *qx, const double *qy, const double *qz,
                            const double *vx, const double *vy, const double *vz, double *ox,
                            double *oy, double *oz, int n);

void lasx_quat_rotate_batch_checked(const double *qw, const double *qx, const double *qy,
                                    const double *qz, const double *vx, const double *vy,
                                    const double *vz, double *ox, double *oy, double *oz, int n,
                                    int *status);

/* 四元数 → 3x3 DCM（行主序，体→惯）；输入**先单位化**。输出的 m0..m8 可直接按
 * lasx_mat3_mul_vec3_batch 的矩阵参数使用（两者布局一致，§7）。
 * 长度契约：qw..qz 各至少 n 个 f64；m0..m8 各至少 n 个 f64。 */
void lasx_quat_to_dcm_batch(const double *qw, const double *qx, const double *qy, const double *qz,
                            double *m0, double *m1, double *m2, double *m3, double *m4, double *m5,
                            double *m6, double *m7, double *m8, int n);

void lasx_quat_to_dcm_batch_checked(const double *qw, const double *qx, const double *qy,
                                    const double *qz, double *m0, double *m1, double *m2,
                                    double *m3, double *m4, double *m5, double *m6, double *m7,
                                    double *m8, int n, int *status);

/* ========================================================================
 * 3. NN 侧算子（N1 批次，docs/ops.md §4.6、§2.6–§2.11、§5.9–§5.13）
 * ======================================================================== */

/* 行内 softmax：out = exp(scale*x + mask - row_max) * (1/sum)，rows*cols 行主序。
 * mask **可选**（NULL = 无 mask）、同形状、**加性**（attention 的 `+ mask`）。
 * 数值契约见 §2.6：`exp` 的 op 序列、下溢到精确 0、向量/标量逐位一致、+inf 不在契约内。
 * 路径：[LASX-only]。长度契约：x、out 各至少 n_rows*n_cols 个 f32；mask 要么 NULL 要么同长。 */
void lasx_softmax_rows(const float *x, const float *mask, float *out, int n_rows, int n_cols,
                       float scale);

void lasx_softmax_rows_checked(const float *x, const float *mask, float *out, int n_rows,
                               int n_cols, float scale, int *status);

/* 行内 RMSNorm：out = x / sqrt(mean(x^2) + eps) * w，rows*cols 行主序，w 按列（长度 cols）。
 * w **可选**（NULL = 无权重）；_checked 要求 eps 有限且 > 0。
 * 数值契约见 §2.7。路径：[LASX-only]。
 * 长度契约：x、out 各至少 n_rows*n_cols 个 f32；w 要么 NULL 要么至少 n_cols 个 f32。 */
void lasx_rms_norm(const float *x, const float *w, float *out, int n_rows, int n_cols, float eps);

void lasx_rms_norm_checked(const float *x, const float *w, float *out, int n_rows, int n_cols,
                           float eps, int *status);

/* SiLU/swish：out[i] = x[i] / (1 + exp(-x[i]))。逐元素，**允许 out == x（就地）**。
 * 契约：**直接除法**（x/den，单次舍入），exp 与软门控共用 §2.6 那一串。
 * 路径：[LASX-only]。长度契约：x、out 各至少 n 个 f32。 */
void lasx_silu(const float *x, float *out, int n);

void lasx_silu_checked(const float *x, float *out, int n, int *status);

/* GELU 的 sigmoid 近似（ggml GELU_QUICK）：out[i] = x[i] / (1 + exp(-1.702*x[i]))。
 * 逐元素，允许就地；契约同 lasx_silu。路径：[LASX-only]。 */
void lasx_gelu_quick(const float *x, float *out, int n);

void lasx_gelu_quick_checked(const float *x, float *out, int n, int *status);

/* GELU 的 erf 形式（PyTorch gelu 默认）：out[i] = 0.5*x[i]*(1 + erf(x[i]/sqrt(2)))。
 * erf 用 A&S 7.1.26 **近似**（不是真 erf，§2.9）；逐元素，允许就地。
 * 路径：[LASX-only]。长度契约：x、out 各至少 n 个 f32。 */
void lasx_gelu_erf(const float *x, float *out, int n);

void lasx_gelu_erf_checked(const float *x, float *out, int n, int *status);

/* 旋转位置编码（partial RoPE）：每行前 n_dims 列旋转、其余原样复制。
 * mode：0 = NeoX（配对 (i, i + n_dims/2)）、1 = GptJ（配对 (2i, 2i+1)）；其它值就地返回。
 * cos/sin 各 n_rows*(n_dims/2)（行主序），**与 mode 无关**。允许 out == x（就地）。
 * 路径：NeoX 走 LASX，GptJ 走标量（§2.10 第 4 条）。
 * 长度契约（**只靠约定**，违反是读写越界）：x/out 各至少 n_rows*n_cols 个 f32；
 * cos/sin 各至少 n_rows*(n_dims/2) 个 f32；n_dims 偶数且 <= n_cols。 */
void lasx_rope(const float *x, const float *cos, const float *sin, float *out, int n_rows,
               int n_cols, int n_dims, int mode);

void lasx_rope_checked(const float *x, const float *cos, const float *sin, float *out, int n_rows,
                       int n_cols, int n_dims, int mode, int *status);

/* f16 权重点积：sum f16_to_f32(a[i]) * b[i]。f16 以 **uint16_t 位型**给出（零依赖）。
 * f16→f32 是精确转换；误差只来自"权重本来是 f16"（相对 ~4.9e-4，§2.11）。
 * 路径：[LASX-only]。长度契约：a 至少 n 个 uint16_t、b 至少 n 个 f32；n == 0 返回 0.0。 */
float lasx_dot_f16(const uint16_t *a, const float *b, int n);

float lasx_dot_f16_checked(const uint16_t *a, const float *b, int n, int *status);

/* f16 权重矩阵（m*k 行主序）× f32 向量：y[r] = lasx_dot_f16(a[r*k..], x)。
 * 每行独立、逐行与 lasx_dot_f16 **逐位相同**；k == 0 时 y 写 0。
 * 路径：[LASX-only]。长度契约：a 至少 m*k 个 uint16_t、x 至少 k 个 f32、y 至少 m 个 f32。 */
void lasx_gemv_f16(const uint16_t *a, const float *x, float *y, int m, int k);

void lasx_gemv_f16_checked(const uint16_t *a, const float *x, float *y, int m, int k,
                           int *status);

/* ========================================================================
 * 4. int8 推理（N3 批次，docs/ops.md §4.3、§2.12–§2.14、§5.14）
 *
 * 对称 int8 口径（§2.12）：scale = amax/127、q ∈ [-127,127]（**不用 -128**）、取整
 * ties-to-even、用**倒数乘**、**不夹取**；累加整数精确，出口一次乘法。
 * 路径：全部 [LASX-only]。
 * ======================================================================== */

/* 最大绝对值 max_i |x_i|（NaN 忽略；空输入或全 NaN 返回 0.0）。
 * 长度契约：x 至少 n 个 f32。 */
float lasx_amax(const float *x, int n);

float lasx_amax_checked(const float *x, int n, int *status);

/* 逐行最大绝对值：out[r] = max_j |x[r*cols + j]|（per-token/per-channel 的公共原语）。
 * 长度契约：x 至少 rows*cols 个 f32；out 至少 rows 个 f32。 */
void lasx_absmax_rows(const float *x, int rows, int cols, float *out);

void lasx_absmax_rows_checked(const float *x, int rows, int cols, float *out, int *status);

/* 逐张量对称量化，**返回 scale**（amax == 0 ⇒ scale = 0、q 全 0）。
 * 长度契约：x 至少 n 个 f32；q 至少 n 个 int8_t。 */
float lasx_quantize_i8_per_tensor(const float *x, int8_t *q, int n);

float lasx_quantize_i8_per_tensor_checked(const float *x, int8_t *q, int n, int *status);

/* 逐行对称量化，每行一个 scale（激活 [token, hidden] 即 per-token；权重 [out, in] 即
 * per-output-channel——同一个内核，Rust api 层按意图分成两个名字，§2.12）。
 * 长度契约：x、q 各至少 rows*cols；scales 至少 rows 个 f32。 */
void lasx_quantize_i8_per_row(const float *x, int8_t *q, float *scales, int rows, int cols);

void lasx_quantize_i8_per_row_checked(const float *x, int8_t *q, float *scales, int rows,
                                      int cols, int *status);

/* 逐张量反量化：out[i] = (q[i] as f32) * scale（一次乘法、一次舍入）。
 * 长度契约：q 至少 n 个 int8_t；out 至少 n 个 f32。 */
void lasx_dequantize_i8(const int8_t *q, float scale, float *out, int n);

void lasx_dequantize_i8_checked(const int8_t *q, float scale, float *out, int n, int *status);

/* 逐行反量化，每行用自己的 scales[r]。
 * 长度契约：q 至少 rows*cols 个 int8_t；scales 至少 rows 个 f32；out 至少 rows*cols 个 f32。 */
void lasx_dequantize_i8_rows(const int8_t *q, const float *scales, float *out, int rows, int cols);

void lasx_dequantize_i8_rows_checked(const int8_t *q, const float *scales, float *out, int rows,
                                     int cols, int *status);

/* int8 权重 x int8 激活的 GEMV：y[o] = (sum_i W[o,i]*x[i]) * (scale_w[o]*scale_x)。
 * 出口**两个 scale 先乘成一个**（一次舍入，§2.13）；整数累加精确；k > 133000 时按
 * mod 2^32 语义回绕。偏置/残差**不在本内核内**（调用方在 f32 域自己加）。
 * 长度契约：w 至少 m*k 个 int8_t；scale_w 至少 m 个 f32；x 至少 k 个 int8_t；y 至少 m 个 f32。 */
void lasx_gemv_i8(const int8_t *w, const float *scale_w, const int8_t *x, float scale_x, float *y,
                  int m, int k);

void lasx_gemv_i8_checked(const int8_t *w, const float *scale_w, const int8_t *x, float scale_x,
                          float *y, int m, int k, int *status);

/* 批量 int8 矩阵乘：Y[t,o] = (sum_i X[t,i]*W[o,i]) * (scale_w[o]*scale_x[t])（即 Y = X*W^T）。
 * 每个输出元素是一次 dot_i8；m == 1 时与 lasx_gemv_i8 逐位一致。
 * 实测注意（§2.14）：批量形态下它比 f32 lasx_matmul 慢 2.5–2.7×，只在"权重装不进缓存、
 * 必须从 DRAM 流"时占优——判据见 §5.15。
 * 长度契约：x 至少 m*k 个 int8_t；w 至少 n*k 个 int8_t；scale_w 至少 n 个 f32；
 * scale_x 至少 m 个 f32；y 至少 m*n 个 f32。 */
void lasx_matmul_i8(const int8_t *x, const int8_t *w, const float *scale_w, const float *scale_x,
                    float *y, int m, int k, int n);

void lasx_matmul_i8_checked(const int8_t *x, const int8_t *w, const float *scale_w,
                            const float *scale_x, float *y, int m, int k, int n, int *status);

/* ========================================================================
 * 5. ONNX 契约补充（docs/ops.md §4.6、§2.16/§2.17、§5.16）
 * ======================================================================== */

/* 行内 LayerNorm：out = (x - mean)/sqrt(var + eps) * w + b，rows*cols 行主序。
 * w/b **可选**（NULL = 无权重/无偏置）；方差用**两遍法**（避免灾难性相消）；
 * 无偏置时**跳过加法**（加 0.0 会把 -0.0 翻成 +0.0、破坏逐位一致）。
 * 路径：[LASX-only]。_checked 要求 eps 有限且 > 0。
 * 长度契约：x、out 各至少 n_rows*n_cols 个 f32；w/b 要么 NULL 要么至少 n_cols 个 f32。 */
void lasx_layer_norm(const float *x, const float *w, const float *b, float *out, int n_rows,
                     int n_cols, float eps);

void lasx_layer_norm_checked(const float *x, const float *w, const float *b, float *out,
                             int n_rows, int n_cols, float eps, int *status);

/* 按 id 取行（f32 嵌入表）：out[i] = table[ids[i]]，每行 row_len 个 f32。
 * **纯行拷贝、无算术**（无结合次序问题，逐位确定）；原始符号里 **id 越界即 UB**。
 * 路径：不涉及 SIMD（连 LASX 都没有，§2.17）。
 * 长度契约：table 至少 n_table_rows*row_len 个 f32；ids 至少 n_ids 个 int（每个在
 * [0, n_table_rows)）；out 至少 n_ids*row_len 个 f32。 */
void lasx_gather_rows(const float *table, const int *ids, float *out, int n_ids, int row_len,
                      int n_table_rows);

void lasx_gather_rows_checked(const float *table, const int *ids, float *out, int n_ids,
                              int row_len, int n_table_rows, int *status);

/* 按 id 取行并反量化（int8 表 + 每行一个 scale）：out[i][j] = table[ids[i]][j] * scales[ids[i]]。
 * 契约同 lasx_gather_rows（另加每行一次乘法）；"表存 f32 还是 int8"由消费侧决定（§2.17）。
 * 长度契约：table 至少 n_table_rows*row_len 个 int8_t；scales 至少 n_table_rows 个 f32；
 * ids 至少 n_ids 个 int；out 至少 n_ids*row_len 个 f32。 */
void lasx_gather_rows_i8(const int8_t *table, const float *scales, const int *ids, float *out,
                         int n_ids, int row_len, int n_table_rows);

void lasx_gather_rows_i8_checked(const int8_t *table, const float *scales, const int *ids,
                                 float *out, int n_ids, int row_len, int n_table_rows,
                                 int *status);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LASX_RS_H */
