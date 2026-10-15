# lasx_rs 的 C ABI（权威说明）

> **本文是 C ABI 的唯一权威**：C / C++ / 其它语言 FFI 调用方需要的全部内容——符号清单与
> 逐符号原型、内存归属、安全模型、能力探测与降级、线程模型、ABI 稳定性策略——都在这里，
> 且**只在这里维护**。`docs/ops.md` 里对应的 C 侧小节只留一句指针（Rust 侧内容仍在 ops）。
>
> **不在本文范围**：Rust 侧 API（`lasx_rs::api` / `view` / `plan` / `shape` / `pool` /
> `parallel`）见 [`docs/ops.md`](ops.md)；性能数据与方法学见 [`docs/dev.md`](dev.md)；
> 平台约束与硬件勘误见 [`docs/platform.md`](platform.md)。

**文档信息**

| 项 | 内容 |
|---|---|
| 适用代码版本 | `master`，**87 个导出符号**（46 个裸符号 + 41 个 `_checked` 变体） |
| 权威原型 | [`include/lasx_rs.h`](../include/lasx_rs.h)（手写、随库安装；机器校验见第 10 节） |
| 读者 | C / C++ 调用方，以及用 `dlopen`/`lookupFunction` 的其它语言绑定（Dart 例见 2.5） |
| 与源码同源 | 第 3 节的清单与参数次序逐条对照 `src/ffi/*.rs` 的 `pub extern "C" fn`；个数由 CI 门禁核对 |

---

## 1. 读者与范围

- **本文的服务对象**是"从 C 调这个库"的人：你拿到的产物是 `liblasx_rs.so`（SONAME
  `liblasx_rs.so.0`）/ `liblasx_rs.a` 与一个头文件；没有 Rust 工具链也能用，数值实现与
  Rust 侧**完全同一份**。
- **三件事由本文定死**（别去别处找权威，别处只有指针）：
  1. 有哪些符号、原型长什么样、哪些是 `[LASX-only]`（第 3 节）；
  2. 内存/线程/错误/能力探测的调用契约（第 4–6、8 节）；
  3. ABI 什么时候允许变（第 9 节），以及这些承诺由什么机器守着（第 10 节）。
- **数值结果**的权威在 `docs/ops.md`：把每个算子的 op 序列与精度包络当契约
  （`docs/ops.md` §2.2、§2.5–§2.17）。本文只复述"位精确 + 两处例外"这条摘要（第 7 节），不重复推导。
- **C 侧没有的东西**，先说清楚免得找：没有线程池（第 8 节）、没有 `*_into` 免分配孪生
  （那是 Rust `api` 的）、没有能力查询之外的硬件信息、没有 `lasx_force_lsx_thread`。

## 2. 快速开始

### 2.1 构建与安装

```bash
cargo build --release                     # 产出 target/release/liblasx_rs.{so,a,rlib}
scripts/install.sh --prefix=/tmp/stage    # 装进暂存目录（也可 --prefix /tmp/stage）
PKG_CONFIG_PATH=/tmp/stage/lib/pkgconfig pkg-config --cflags --libs lasx_rs
```

`install.sh`（`set -eu`、幂等）装这些东西，相对 `<prefix>`：

| 路径 | 内容 | 谁在用 |
|---|---|---|
| `lib/liblasx_rs.so.<version>` | 动态库本体（`<version>` 取自 `Cargo.toml`，如 `0.1.0`） | 打包脚本"钉到具体版本" |
| `lib/liblasx_rs.so.0` | 符号链接 → 本体 | **运行时** loader（按 SONAME 找这个名字） |
| `lib/liblasx_rs.so` | 符号链接 → 本体 | **链接期** `-llasx_rs` 找这个名字 |
| `lib/liblasx_rs.a` | 静态库 | 不想带 `.so` 的调用方 |
| `lib/pkgconfig/lasx_rs.pc` | pkg-config 元数据（由 `lasx_rs.pc.in` 生成） | `pkg-config --cflags --libs lasx_rs` |
| `include/lasx_rs.h` | C 头（**由仓库提供**，脚本只拷贝、缺了就报错） | `#include <lasx_rs.h>` |

失败即退出，且**校验都在写入之前**（不留半套安装）；`readelf` 可用时还会核对产物 SONAME，
不对就拒绝安装。

**错架构会被挡住**（本库只在 LoongArch64 上构建/安装，见 §2.6）：在非 LoongArch64 上
`cargo build` 得到的是一条人话，而**不是**几十条"目标不支持该指令"：

```text
error: failed to run custom build command for `lasx_rs v0.1.0`
  --- stderr
  lasx_rs 是 LoongArch LASX/LSX 指令集绑定实现（手写内联汇编/内在函数），不是可移植标量库。
  当前构建目标是 `x86_64`，不是 LoongArch64，已停止。请在本机 LoongArch64 上构建，或改用其它后端。
```

`install.sh` 另用 `readelf -h` 核对**产物架构**：把非 LoongArch 的 `.so` 装进 prefix 会被
**拒绝安装**（同样打印人话）。

### 2.2 四种链接方式

```bash
# ① 动态 + rpath（推荐）：运行期不必设 LD_LIBRARY_PATH
cc app.c -o app -I/tmp/stage/include -L/tmp/stage/lib -llasx_rs -Wl,-rpath,/tmp/stage/lib

# ② 动态 + 运行期指路（装到系统目录后也可以交给 ldconfig）
cc app.c -o app -I/tmp/stage/include -L/tmp/stage/lib -llasx_rs
LD_LIBRARY_PATH=/tmp/stage/lib ./app

# ③ 静态：直接给 .a；数学函数自己带 -lm
cc app.c -o app -I/tmp/stage/include /tmp/stage/lib/liblasx_rs.a -lm

# ④ pkg-config 驱动（CI/构建系统里最省事）
cc app.c -o app $(pkg-config --cflags --libs lasx_rs)
```

**仓库内直接试（不安装）**：`cargo build --release && make -C examples/c run`——Makefile 在
pkg-config 找不到 `lasx_rs` 时自动兜底链接 `target/release`，并补上 SONAME 需要的
`liblasx_rs.so.0` 符号链接（cargo 只产出 `liblasx_rs.so`，loader 却按 SONAME 找名字，这是
本地链接最容易踩的一脚）。

### 2.3 最小可运行程序

```c
#include <lasx_rs.h>
#include <stdio.h>

int main(void) {
    printf("lasx_rs %s\n", lasx_version());          /* ① 先知道是哪一版 */
    if (!lasx_has_lasx()) {                          /* ② 再问能力：调 [LASX-only] 的安全前提 */
        fprintf(stderr, "无 LASX 路径（仅 LSX 机型）：改用 LASX+LSX 的符号或自己的标量实现\n");
        return 2;
    }
    const int n = 8;
    float *a = lasx_alloc(n), *b = lasx_alloc(n);    /* 32 字节对齐、未初始化 */
    for (int i = 0; i < n; i++) { a[i] = (float)(i + 1); b[i] = 1.0f / (float)(i + 1); }

    int st = -1;
    float d = lasx_dot_checked(a, b, n, &st);        /* ③ 默认走 _checked：出错走 status */
    printf("lasx_dot_checked = %.6f  status=%d (%s)\n", (double)d, st, lasx_status_message(st));
    if (st != LASX_OK) { lasx_free(a, n); lasx_free(b, n); return 1; }

    lasx_free(a, n); lasx_free(b, n);                /* ④ 必须配对归还：free() 不是 ABI 契约 */
    return 0;
}
```

### 2.4 `examples/c/`（仓库里的可编译样例）

[`examples/c/quickstart.c`](../examples/c/quickstart.c) + [`examples/c/Makefile`](../examples/c/Makefile)
就是上面的程序，把"版本 → 能力 → 调用 → status → 归还"五步串起来：

```bash
make -C examples/c run
# [make] pkg-config 未找到 lasx_rs ⇒ 直接链接 .../target/release   （未安装时的兜底提示）
```

**实测输出（本机 Loongson-3B6000，2026-10-15）**：

```text
lasx_rs 版本：0.1.0
能力检查：lasx_has_lasx() = 1 ⇒ 可以调 [LASX-only] 符号
lasx_dot_checked(n=8) = 8.000000   status=0 (成功)
已用 lasx_free 归还两块缓冲（free() 不是 ABI 契约）
```

### 2.5 其它语言：以 Dart 为例

`lookupFunction` 的泛型参数（Native 侧 / 语言侧签名）必须与 C 签名**逐参数一致**，否则 UB：

```dart
final lib = DynamicLibrary.open('liblasx_rs.so');

final hasLasx = lib.lookupFunction<Int32 Function(), int Function()>('lasx_has_lasx');
final dot = lib.lookupFunction<
    Float Function(Pointer<Float> a, Pointer<Float> b, Int32 n),
    Float Function(Pointer<Float> a, Pointer<Float> b, int n)>('lasx_dot_checked');
final alloc = lib.lookupFunction<Pointer<Float> Function(Int32 n),
                                Pointer<Float> Function(int n)>('lasx_alloc');
final free_ = lib.lookupFunction<Void Function(Pointer<Float> p, Int32 n),
                                 void Function(Pointer<Float> p, int n)>('lasx_free');

if (hasLasx() == 0) { /* 只剩 LSX 路径：别调 [LASX-only] */ }
```

### 2.6 平台绑定与防呆：**错架构 ≠ 无 LASX**

这是两个完全不同的问题，**检测手段与失败时机都不一样**，不要混：

| | **错架构**（x86 / arm …） | **LSX-only 机型（无 LASX）**，如 2K3000 / 3B6000M |
|---|---|---|
| 构建 / 安装 | **被挡住**：编译期与安装期就失败，产物根本出不来 | **正常**：能构建、能安装、能运行 |
| 危险在哪 | 产物完全不可用（不是"慢一点"） | 只在调标了 `[LASX-only]` 的符号时：执行 LASX 指令 ⇒ **SIGILL** |
| 谁负责 | 本库的防线（下面五道） | **调用方**：运行时先问 `lasx_has_lasx()` 再调（§6） |
| 失败时机 | 编译期、安装期 | 运行期（调用那一行） |

一句话：**"错架构"是编译/安装期问题（防呆）；"无 LASX"是运行时问题（`lasx_has_lasx()` 先问再调）**。
`[LASX-only]` 的完整清单与降级分类见 `docs/ops.md` §3.2/§3.3。

**五道防线**（按触发顺序）：

| # | 位置 | 挡什么 / 怎么挡 |
|---|---|---|
| 1 | `build.rs` | 判**目标架构**（`CARGO_CFG_TARGET_ARCH`）：非 `loongarch64` ⇒ 打印人话并 `exit(1)`。因为判的是**目标**，`cargo check --target loongarch64` 的交叉检查照常通过 |
| 2 | `src/lib.rs`、`cli/src/main.rs` | `#[cfg(not(target_arch = "loongarch64"))] compile_error!(…)`：给**绕过构建脚本**的场景兜底（vendored 源码、直接调 rustc） |
| 3 | `include/lasx_rs.h` 顶部 | `#if !defined(__loongarch__) && !defined(LASX_RS_ALLOW_FOREIGN_ARCH)` ⇒ `#error`：C 侧**编译期**就红 |
| 4 | `scripts/install.sh` | `LC_ALL=C readelf -h` 核对 `Machine:` 是 LoongArch，不符 ⇒ **拒绝安装**；`readelf` 缺失 ⇒ 降级为告警 |
| 5 | `examples/c/Makefile` | `uname -m` 非 `loongarch64` ⇒ `$(error …)`；`FOREIGN=1` 显式覆盖 |

**为什么以构建脚本为主、`compile_error!` 只兜底**：`compile_error!` **不会中止**这次编译——
rustc 会把其余源码的错误一起报出来。实测 x86_64 原生构建时，它先冒出 **35 条**"目标不支持该
指令 / 找不到 intrinsic"，那句人话被埋在第 361 行；构建脚本先退出后，整个输出只剩 **1 条**。

**交叉编译 / 文档生成 / 头文件语法检查怎么办**（逃生阀）：

- C 头：显式定义 `LASX_RS_ALLOW_FOREIGN_ARCH` 即跳过那道 `#error`——**它只关掉编译期检查，
  不会让运行时可用**（产物仍是 LoongArch 机器码，得在 LoongArch64 上跑）。
- 示例：`make -C examples/c FOREIGN=1`（Makefile 会同时定义上面那个宏）。

---

## 3. 完整符号参考（87 = 46 裸 + 41 `_checked`）

### 3.0 怎么读这一节

- **原型**逐字对应 [`include/lasx_rs.h`](../include/lasx_rs.h)；`int` 一律是 C 32 位
  （Rust `i32`），`int8_t`/`uint8_t`/`uint16_t` 来自 `<stdint.h>`，输入 `const T *`、
  输出 `T *`，标量按值传。矩阵/批量缓冲**一律行主序**。
- **路径**列的含义（完整分类见 `docs/ops.md` §3.2）：
  - `LASX+LSX`：两种机型都安全（有 128 位 LSX 降级路径）；
  - `LASX+标量`：没有 LSX 向量路径，但非 LASX 时走全标量，**仍然安全**；
  - `[LASX-only]`：**没有降级分支**——在只实现 LSX 的机型上执行 LASX 指令 ⇒ **SIGILL**；
    调它之前必须先 `lasx_has_lasx()`（第 6 节）；
  - `不涉及 SIMD` / `与路径无关`：任何机型都能调。
- **`_checked`**：除 5 个裸符号（`lasx_alloc`、`lasx_free`、`lasx_has_lasx`、`lasx_version`、
  `lasx_status_message`）外，每个裸符号都有一个同名 + `_checked` 的变体：**原型只在末尾多
  一个 `int *status`，其余参数类型与次序完全相同**。完整 41 个名字见 3.7。
- 语义列只给"一句话"；**逐算子的 op 序列、精度包络与"谁守"**在 `docs/ops.md`
  （`docs/ops.md` §2.6–§2.17 是契约，§5.x/§6.x/§7.x 是用例）。写参考实现时以那里为准。

### 3.1 归约、稠密线性代数、量化点积、物理与内存（16 个裸符号）

| 符号 | C 原型 | 路径 | 语义 |
|---|---|---|---|
| `lasx_dot` | `float lasx_dot(const float *a, const float *b, int n)` | LASX+LSX | `Σ a[i]·b[i]`；`n < 24` 走纯标量 f32，跨 `n=24` 不保证逐位连续 |
| `lasx_sum` | `float lasx_sum(const float *x, int n)` | [LASX-only] | `Σ x[i]`；相对 f64 参考 <1e-5 |
| `lasx_dot_f64` | `double lasx_dot_f64(const double *a, const double *b, int n)` | [LASX-only] | f64 点积；全程 IEEE 双精度 |
| `lasx_axpy` | `void lasx_axpy(float alpha, const float *x, float *y, int n)` | [LASX-only] | `y[i] += alpha·x[i]`（原地写 y）；向量/标量尾至多差 1 ulp |
| `lasx_matmul` | `void lasx_matmul(int m, int k, int n, const float *a, const float *b, float *c)` | [LASX-only] | `C[m×n] = A[m×k]·B[k×n]`；`m==0\|\|n==0` 直接返回，`k==0` 不写 `C` |
| `lasx_matmul_scaled` | `void lasx_matmul_scaled(int m, int k, int n, float alpha, const float *a, const float *b, float beta, float *c)` | [LASX-only] | `C = alpha·(A·B) + beta·C`（原地，CBLAS 次序）；`beta == 0` **不读 `C`**；`alpha=1,beta=0` 与 `lasx_matmul` 逐位一致 |
| `lasx_matmul_f64` | `void lasx_matmul_f64(int m, int k, int n, const double *a, const double *b, double *c)` | [LASX-only] | f64 矩阵乘；相对 f64 参考 <1e-12 |
| `lasx_dot_i8` | `int lasx_dot_i8(const int8_t *a, const int8_t *b, int n)` | [LASX-only] | int8 点积，**i32 返回**（整数精确）；超 i32 回绕，约 `n > 13.3 万` 可能发生 |
| `lasx_dot_q4` | `double lasx_dot_q4(const uint8_t *qa, const float *sa, const uint8_t *qb, const float *sb, int n_bytes)` | [LASX-only] | Q4 点积：每字节 2 个 nibble，scale 每 32 字节一组（`ceil(n_bytes/32)` 个） |
| `lasx_alloc` | `float *lasx_alloc(int n)` | 与路径无关 | 32 字节对齐、**未初始化**；`n == 0` 给"对齐但不可解引用"的非空悬垂指针 |
| `lasx_norm3_batch` | `void lasx_norm3_batch(const double *xs, const double *ys, const double *zs, double *out, int n)` | LASX+LSX | `out[i] = √(xs²+ys²+zs²)`；结合序固定 `x²+(y²+z²)` |
| `lasx_vec3_add_scaled_batch` | `void lasx_vec3_add_scaled_batch(const double *ax, const double *ay, const double *az, const double *bx, const double *by, const double *bz, double s, double *ox, double *oy, double *oz, int n)` | LASX+LSX | `o = a + s·b`，x/y/z 独立 |
| `lasx_batch_distance2d` | `void lasx_batch_distance2d(float px, float py, const float *xs, const float *ys, float *out, int n)` | LASX+标量 | `out[i] = √((xs-px)²+(ys-py)²)` |
| `lasx_ballistic_step` | `void lasx_ballistic_step(float *x, float *y, float *z, float *vx, float *vy, float *vz, const float *k, int n, float dt, float g)` | LASX+标量 | 批量弹道欧拉步（**原地**）；向量/标量路径**非逐位一致**（第 7 节例外之一） |
| `lasx_j2_accel_batch` | `void lasx_j2_accel_batch(const double *rx, const double *ry, const double *rz, double mu, double j2, double re, double *ax, double *ay, double *az, int n)` | LASX+LSX | J2 引力加速度；每样本独立、无跨 lane 归约 |
| `lasx_rk4_j2_step_batch` | `void lasx_rk4_j2_step_batch(double *rx, double *ry, double *rz, double *vx, double *vy, double *vz, double mu, double j2, double re, double dt, int n)` | LASX+标量 | RK4 J2 单步（**原地**）；分块内与标量同式 |

### 3.2 批量姿态与几何（7 个裸符号，全部 LASX+LSX）

公共约定（`docs/ops.md` §7）：四元数**标量在前** `q = w + xi + yj + zk`；旋转为**体→惯**；
`|q| < 1e-15` 视为退化 ⇒ 单位四元数 `(1,0,0,0)`；**全部允许输出与输入别名**。

| 符号 | C 原型 | 语义 |
|---|---|---|
| `lasx_cross3_batch` | `void lasx_cross3_batch(const double *ax, const double *ay, const double *az, const double *bx, const double *by, const double *bz, double *ox, double *oy, double *oz, int n)` | `o = a × b` |
| `lasx_unitize3_batch` | `void lasx_unitize3_batch(const double *x, const double *y, const double *z, double *ox, double *oy, double *oz, int n)` | `o = v/\|v\|`；零向量 ⇒ `(0,0,0)`（不是 NaN） |
| `lasx_mat3_mul_vec3_batch` | `void lasx_mat3_mul_vec3_batch(const double *m0, …, const double *m8, const double *x, const double *y, const double *z, double *ox, double *oy, double *oz, int n)` | `o = M·v`；`m0..m8` 行主序 3×3 |
| `lasx_quat_normalize_batch` | `void lasx_quat_normalize_batch(double *qw, double *qx, double *qy, double *qz, int n)` | **原地**单位化；退化 ⇒ `(1,0,0,0)` |
| `lasx_quat_mul_batch` | `void lasx_quat_mul_batch(const double *aw, …, const double *bz, double *ow, …, double *oz, int n)` | Hamilton 积（**不做内部归一化**） |
| `lasx_quat_rotate_batch` | `void lasx_quat_rotate_batch(const double *qw, …, const double *qz, const double *vx, const double *vy, const double *vz, double *ox, double *oy, double *oz, int n)` | 先单位化，再 `o = R(q)·v` |
| `lasx_quat_to_dcm_batch` | `void lasx_quat_to_dcm_batch(const double *qw, …, const double *qz, double *m0, …, double *m8, int n)` | 四元数 → 3×3 DCM（行主序），可直接喂给 `lasx_mat3_mul_vec3_batch` |

> 表中 `…` 是**书写省略**，不是 C 语法：完整逐参数原型见头文件与 `src/ffi/attitude.rs`。

### 3.3 NN 侧 N1 批次（8 个裸符号，全部 [LASX-only]）

| 符号 | C 原型 | 语义 |
|---|---|---|
| `lasx_softmax_rows` | `void lasx_softmax_rows(const float *x, const float *mask, float *out, int n_rows, int n_cols, float scale)` | 行内 softmax；`mask` 可 NULL（加性 mask） |
| `lasx_rms_norm` | `void lasx_rms_norm(const float *x, const float *w, float *out, int n_rows, int n_cols, float eps)` | 行内 RMSNorm；`w` 可 NULL（按列权重）；要求 `eps > 0`（`_checked` 强制） |
| `lasx_silu` | `void lasx_silu(const float *x, float *out, int n)` | `x/(1+exp(-x))`；逐元素，允许 `out == x` |
| `lasx_gelu_quick` | `void lasx_gelu_quick(const float *x, float *out, int n)` | `x/(1+exp(-1.702x))`（ggml GELU_QUICK）；允许就地 |
| `lasx_gelu_erf` | `void lasx_gelu_erf(const float *x, float *out, int n)` | `0.5x(1+erf(x/√2))`；erf 是 A&S 7.1.26 **近似**，不是真 erf |
| `lasx_rope` | `void lasx_rope(const float *x, const float *cos, const float *sin, float *out, int n_rows, int n_cols, int n_dims, int mode)` | partial RoPE；`mode`：0=NeoX、1=GptJ；`cos/sin` 各 `n_rows×(n_dims/2)`；允许就地 |
| `lasx_dot_f16` | `float lasx_dot_f16(const uint16_t *a, const float *b, int n)` | f16（**u16 位型**）× f32 点积；f16→f32 精确 |
| `lasx_gemv_f16` | `void lasx_gemv_f16(const uint16_t *a, const float *x, float *y, int m, int k)` | f16 权重矩阵（`m×k` 行主序）× f32 向量；`k == 0` 时 `y` 写 0 |

### 3.4 int8 推理 N3 批次（8 个裸符号，全部 [LASX-only]）

对称 int8 口径：`scale = amax/127`、`q ∈ [-127,127]`（**不用 -128**）、取整 ties-to-even、
用倒数乘、不夹取；累加整数精确，出口一次乘法（`docs/ops.md` §2.12–§2.14）。

| 符号 | C 原型 | 语义 |
|---|---|---|
| `lasx_amax` | `float lasx_amax(const float *x, int n)` | `max\|x_i\|`（NaN 忽略；空输入 0.0） |
| `lasx_absmax_rows` | `void lasx_absmax_rows(const float *x, int rows, int cols, float *out)` | 逐行最大绝对值 |
| `lasx_quantize_i8_per_tensor` | `float lasx_quantize_i8_per_tensor(const float *x, int8_t *q, int n)` | 逐张量量化，**返回 scale** |
| `lasx_quantize_i8_per_row` | `void lasx_quantize_i8_per_row(const float *x, int8_t *q, float *scales, int rows, int cols)` | 逐行量化（per-token 激活 / per-channel 权重共用同一内核） |
| `lasx_dequantize_i8` | `void lasx_dequantize_i8(const int8_t *q, float scale, float *out, int n)` | `out = (q as f32)·scale` |
| `lasx_dequantize_i8_rows` | `void lasx_dequantize_i8_rows(const int8_t *q, const float *scales, float *out, int rows, int cols)` | 逐行反量化 |
| `lasx_gemv_i8` | `void lasx_gemv_i8(const int8_t *w, const float *scale_w, const int8_t *x, float scale_x, float *y, int m, int k)` | `y[o] = (Σ W[o,i]·x[i])·(sw[o]·sx)`；出口两个 scale 先乘成一个 |
| `lasx_matmul_i8` | `void lasx_matmul_i8(const int8_t *x, const int8_t *w, const float *scale_w, const float *scale_x, float *y, int m, int k, int n)` | `Y[t,o] = (Σ X[t,i]·W[o,i])·(sw[o]·sx[t])`；`m=1` 时与 `lasx_gemv_i8` 逐位一致 |

> **选型提醒**（实测，`docs/ops.md` §2.14/§5.15）：批量形态下 `lasx_matmul_i8` 比 f32
> `lasx_matmul` **慢 2.5–2.7×**；int8 只在"权重装不进缓存、必须从 DRAM 流"时占优。

### 3.5 ONNX 契约补充（3 个裸符号）

| 符号 | C 原型 | 路径 | 语义 |
|---|---|---|---|
| `lasx_layer_norm` | `void lasx_layer_norm(const float *x, const float *w, const float *b, float *out, int n_rows, int n_cols, float eps)` | [LASX-only] | 行内 LayerNorm（两遍法方差）；`w`/`b` 可 NULL；`eps > 0`（`_checked` 强制） |
| `lasx_gather_rows` | `void lasx_gather_rows(const float *table, const int *ids, float *out, int n_ids, int row_len, int n_table_rows)` | 不涉及 SIMD | `out[i] = table[ids[i]]`（纯行拷贝）；裸符号里 **id 越界即 UB** |
| `lasx_gather_rows_i8` | `void lasx_gather_rows_i8(const int8_t *table, const float *scales, const int *ids, float *out, int n_ids, int row_len, int n_table_rows)` | 不涉及 SIMD | 行拷贝 + 每行一次 scale 乘法 |

### 3.6 运行时元信息与内存归还（4 个裸符号，与路径无关）

| 符号 | C 原型 | 语义 |
|---|---|---|
| `lasx_has_lasx` | `int lasx_has_lasx(void)` | 当前**生效路径**是否 LASX（1/0）；调 `[LASX-only]` 前必须问（第 6 节） |
| `lasx_version` | `const char *lasx_version(void)` | 版本串（如 `"0.1.0"`）；**永不为 NULL**、静态数据、不要 free |
| `lasx_status_message` | `const char *lasx_status_message(int status)` | 状态码的静态说明串；未知码给通用串；**永不为 NULL**、不要 free |
| `lasx_free` | `void lasx_free(float *p, int n)` | 归还 `lasx_alloc` 的缓冲（`n` 必须是分配时那个值）；`p==NULL` 与 `n<=0` 是安全 no-op |

### 3.7 `_checked` 变体（41 个名字）

规则见 3.0：**同一个原型 + 末尾 `int *status`**。41 个名字（分组与 3.1–3.5 对应）：

- 归约/稠密/量化/物理/内存：`lasx_dot_checked`、`lasx_sum_checked`、`lasx_dot_f64_checked`、
  `lasx_axpy_checked`、`lasx_matmul_checked`、`lasx_matmul_scaled_checked`、
  `lasx_matmul_f64_checked`、`lasx_dot_i8_checked`、`lasx_dot_q4_checked`、
  `lasx_norm3_batch_checked`、`lasx_vec3_add_scaled_batch_checked`、
  `lasx_batch_distance2d_checked`、`lasx_ballistic_step_checked`、
  `lasx_j2_accel_batch_checked`、`lasx_rk4_j2_step_batch_checked`（15 个；`lasx_alloc` 没有）
- 批量姿态/几何：`lasx_cross3_batch_checked`、`lasx_unitize3_batch_checked`、
  `lasx_mat3_mul_vec3_batch_checked`、`lasx_quat_normalize_batch_checked`、
  `lasx_quat_mul_batch_checked`、`lasx_quat_rotate_batch_checked`、`lasx_quat_to_dcm_batch_checked`（7）
- NN 侧 N1：`lasx_softmax_rows_checked`、`lasx_rms_norm_checked`、`lasx_silu_checked`、
  `lasx_gelu_quick_checked`、`lasx_gelu_erf_checked`、`lasx_rope_checked`、
  `lasx_dot_f16_checked`、`lasx_gemv_f16_checked`（8）
- int8 N3：`lasx_amax_checked`、`lasx_absmax_rows_checked`、
  `lasx_quantize_i8_per_tensor_checked`、`lasx_quantize_i8_per_row_checked`、
  `lasx_dequantize_i8_checked`、`lasx_dequantize_i8_rows_checked`、`lasx_gemv_i8_checked`、
  `lasx_matmul_i8_checked`（8）
- ONNX 补充：`lasx_layer_norm_checked`、`lasx_gather_rows_checked`、
  `lasx_gather_rows_i8_checked`（3）

**5 个裸符号没有 `_checked` 变体**（46 − 41 = 5）：`lasx_alloc`、`lasx_free` 与三个元信息
符号——它们的入参里没有"可失败的结构"，`lasx_free` 的 `p==NULL`/`n<=0` 本来就是安全 no-op，
元信息符号约定永不返回 NULL。

### 3.8 状态码（`lasx_status_message` 的输入，也是 `status` 出参的取值域）

| 码 | 名称 | 含义 |
|---|---|---|
| 0 | `Ok` | 成功 |
| 1 | `NullPointer` | 指针为空且长度非 0 |
| 2 | `NegativeLength` | 长度参数为负 |
| 3 | `BadShape` | 形状/长度自洽性不成立（**留给知道真实长度的上层**，如语言绑定） |
| 4 | `SizeOverflow` | 尺寸相乘溢出 |
| 5 | `NonPositiveConstant` | `mu <= 0` 或 `re <= 0` |
| 6 | `NonFiniteConstant` | `j2`/`dt` 为 NaN 或无穷 |

打印请一律用 `lasx_status_message(st)`（0..6 与未知码都给静态串、**永不 NULL**），不要把
0..6 硬编码进调用方；Rust 侧同一个表的另一面见 `docs/ops.md` §4.5。

---

## 4. 类型、内存与生命周期

- **跨界类型只有 C 基础类型**：`float`、`double`、`int`、`int8_t`、`uint8_t`、`uint16_t`、
  `const char *`。**没有 Rust 类型跨界**（没有 `Vec`/`&[T]`/`Option`/`Result`/trait 对象），
  所以任何能声明 C 原型的语言都能调。
- **长度与形状**一律 C `int`（32 位，Rust `i32`）；指针 + 长度的语义是"**你声明有多长，
  库就按多长读写**"。库不持有你的指针（返回后即可释放/复用），不回调、不异步。
- **缓冲归调用方所有**：所有算子的 `out` 都由调用方提供（库只在 `lasx_alloc` 里分配内存）。
  可以反复传同一块缓冲来避免分配——这就是 C 侧的"复用"方式（Rust 的 `*_into` 孪生不导出给 C）。
- **`lasx_alloc` ↔ `lasx_free` 必须配对，且 `n` 必须一致**：
  - 两者共用同一 `f32_layout()`，"同一 layout"由构造成立；
  - `lasx_alloc(0)` 返回的是"对齐但不可解引用"的**非空悬垂指针**——它不是堆块，
    `lasx_free(p, n<=0)` 因此是 no-op（不会去 `dealloc` 它）；
  - **`free()` 不是 ABI 契约**：本机恰好是 glibc malloc 只是实现细节；换分配器/运行库之后
    用 `free()` 就是 UB。**`posix_memalign` 配对的释放函数同样不能代替 `lasx_free`**；
  - 传错的 `n`（非 0 且与分配时不同）破坏堆——这条**只有约定**，违反是 UB。
- **别名规则**：默认**假设输出与输入不重叠**（拿不准就当不能重叠）。显式允许就地/别名的
  有：逐元素激活与 RoPE（`out == x`）、批量姿态/几何全部 7 个（输入先取完再写回）、
  以及 `lasx_matmul_scaled` 的 `c`（`beta != 0` 时旧值参与计算）。其余算子请给独立输出。
- **对齐只影响性能**：32 字节对齐能让 `xvld/xvst` 不跨缓存行；未对齐结果照样正确。
  想要稳定对齐就用 `lasx_alloc`（配 `lasx_free`）或自己 `posix_memalign`。

## 5. 安全模型：裸 vs `_checked`

| | 裸（如 `lasx_dot`） | `_checked`（如 `lasx_dot_checked`） |
|---|---|---|
| 校验 | **零校验**：长度/空指针/形状违反契约即 **UB**（读写越界，不是"错结果"） | 只做**结构**校验：空指针、负长度、乘积溢出、物理常数合法性 |
| 失败通道 | 没有 | 末尾 `int *status` |
| 失败时的行为 | 未定义 | **只写 `status`**，返回安全中性值（数值型 `0.0`/`0`，`void` 型不写输出），**不触碰输出缓冲** ⇒ 可以在原地重试 |
| `status` 传 NULL | — | **合法**：表示"不关心原因"，但**校验仍然执行** |
| 开销 | 最低 | 一次分支（热路径上可测） |

**该用哪个**：默认用 `_checked`；只有在**热路径**且**长度已在调用点被证明**时才换裸符号
（例如外层校验一次、内层循环不再重复校验）。C 侧能校验的只有**结构**："数组真实长度是否
等于声明的形状"只有知道长度语义的上层（语言绑定、你自己的容器）判得了——那类错误在 C 侧
映射到 `BadShape`。

## 6. 能力与降级：先 `lasx_has_lasx()` 再调

`lasx_has_lasx()` 回答的是"**此刻能不能安全调 `[LASX-only]`**"：

> 这一节管的是**运行期**的能力问题（**无 LASX** 的 LSX-only 机型）；**错架构**（x86 等）是
> 另一件事，在编译/安装期就被挡住，见 §2.6——两者的检测手段与失败时机都不同。

```c
/* ✅ 正确：先问能力，再决定走哪条路 */
if (!lasx_has_lasx()) {
    /* 只剩 LSX 路径（机型只实现 128 位向量，或本进程被强制降级）：
       改用 LASX+LSX 的符号（如 lasx_norm3_batch），或落到你自己的标量实现 */
    return fallback_to_scalar();
}
int st;
float d = lasx_dot_checked(a, b, n, &st);   /* 从这里起，[LASX-only] 才是安全的 */

/* ❌ 错误：不问就调 —— LSX-only 机型上是 SIGILL（非法指令），不是"自动降级" */
float d = lasx_dot(a, b, n);
```

- **清单**（完整分类见 `docs/ops.md` §3.2）：**`[LASX-only]` = 25 个内核**——`lasx_sum`、
  `lasx_dot_f64`、`lasx_axpy`、`lasx_matmul`、`lasx_matmul_scaled`、`lasx_matmul_f64`、
  `lasx_dot_i8`、`lasx_dot_q4`、`lasx_layer_norm`，N1 批次 8 个（3.3），N3 批次 8 个（3.4）；
  **可降级 = 14 个**：`LASX+LSX` 11 个（`lasx_dot`、`lasx_norm3_batch`、
  `lasx_vec3_add_scaled_batch`、`lasx_j2_accel_batch` + 姿态/几何 7 个）、`LASX+标量` 3 个
  （`lasx_ballistic_step`、`lasx_batch_distance2d`、`lasx_rk4_j2_step_batch`）；
  **与路径无关**：`lasx_alloc`/`lasx_free`、`lasx_gather_rows`/`lasx_gather_rows_i8`、
  三个元信息符号。
- **机型对照**：`LA464`（3A5000）/ `LA664`（3A6000 / 3C6000 / 3B6000）**有 LASX**；
  `LA364E`（2K3000 / 3B6000M）**只有 LSX** ⇒ 在其上直调 `[LASX-only]` 会 SIGILL。
- **语义边界**：它返回的是"**当前生效路径**"（与算子入口同源），所以进程内被
  `lasx_force_lsx_thread` 强制降级时同样返回 0——这正是调用方要的"能不能调"，而不是"这块
  CPU 有没有 LASX"。想要"硬件有没有"用 Rust 侧 `arch::hardware()`（C ABI 不导出）。
- **代价**：一次进程级原子读 + 一次线程局部读（~0.2–0.6 ns）；按批次问一次即可，不必每次调用前问。

## 7. 数值契约（摘要，权威在 `docs/ops.md`）

- **位精确承诺**（`docs/ops.md` §2.1、§2.5）：**同一个算子**、同一输入，各路径（LASX / LSX /
  标量尾 / 打包面板 / k 分块 / 并行切块）给出**逐位相同**的结果（测试用 `to_bits()` 比对）。
  跨实现对照是"相对误差 <1e-9"一档，**不是**逐位相等。
- **例外仅两处**（均有测试与文档）：`lasx_ballistic_step` 的向量/标量分支结合序不同（物理
  一致，有绝对误差测试守护）；`lasx_axpy` 允许 1 ulp 以内差异。
- **累加次序是硬契约**：每个输出元素沿 `k` 升序、单条 FMA 累加；写参考实现时必须照抄
  `docs/ops.md` 里对应算子的 op 序列（§2.6–§2.17）。`lasx_matmul_scaled` 的融合施加次序固定为
  `out = alpha·acc + beta·c_old`（两次舍入），不是"用 `beta·c` 给累加器播种"。
- **跨语言要的判据**是"**同输入两次运行逐位一致**"，不是"与某个参考实现逐位一致"。
- **不是位精确的一档**：int8 端到端误差上界（`docs/ops.md` §2.15）、f16 权重的量化误差
  （权重本身是 f16，相对 ~4.9e-4）——这些是按误差界承诺，别当位精确用。

## 8. 线程模型

- **各入口线程安全**：没有逐调用的全局可写状态（能力探测结果进程级缓存一次，之后只读）。
  不同线程用**不相交缓冲**并发调用没问题；**同一输出缓冲的并发写**由调用方自己同步。
- **C ABI 不导出线程池**：没有 `WorkerPool`/`parallel` 的 C 入口。要多核就**调用方自己把
  区间切开**，在每个线程里调同一批符号（结果与单线程逐位一致，因为只切下标区间）。
- **`lasx_force_lsx_thread` 拿不到**：它是 Rust API、**线程局部**、仅供测试/实验（用来在
  含 LASX 的真机上真实执行 LSX 分支）。它只改当前线程的路径，且不影响池里复用的 worker。
- 内核不持有指针、不回调、不异步：函数返回后缓冲可立即释放。

## 9. ABI 稳定性策略

- **只追加，不改**：已有符号的**签名与语义永不改动**；新能力一律追加新符号（含 `_checked`
  变体与元信息符号）。最初的 15 个符号（见 `docs/ops.md` §4.1）尤其如此。
- **SONAME 固定为 `liblasx_rs.so.0`**（由 `build.rs` 的
  `cargo:rustc-cdylib-link-arg=-Wl,-soname,liblasx_rs.so.0` 写入）。构建产物名是
  `liblasx_rs.so`；**安装后**才有 `liblasx_rs.so.<version>` + `.so.0` + `.so` 三个名字
  （本体 / 运行时按 SONAME 找 / 链接期 `-llasx_rs` 找）。
- **什么时候可以动 major**：

| 变更 | 允许？ | SONAME |
|---|---|---|
| 追加新符号 | 允许 | 不变（仍 `.so.0`） |
| 修 bug 导致数值变化（仍在逐位承诺内） | 允许 | 不变 |
| 改已有符号的签名/语义，或删符号 | **破坏 ABI** | **major +1** |
| 改状态码取值/结构布局、改调用约定 | **破坏 ABI** | **major +1** |

- **major 变更要同改三处**：`build.rs` 的 SONAME、`scripts/install.sh` 的 `SONAME` 变量、
  本文第 9 节与 `docs/ops.md` §4 的规则表（并按 `AGENTS.md` 规则 3 先在 `docs/dev.md` 记一笔）。
- **0.x 的语义**：当前 `0.1.0`，major = 0 表示"**尚未承诺二进制兼容**"，但"只追加"这条实践
  已经在遵守。**不同 major 可以并存**（`liblasx_rs.so.0` 与未来的 `.so.1` 互不覆盖）。

### 9.1 命名规范（只管**未来新增**）

**为什么只管未来**：已发布的 87 个符号**只追加不改名**——改名即破坏 ABI（见上一节）。所以本节
只约束新增符号，既有情况见文末"历史例外"表。规则由
[`scripts/namecheck.sh`](../scripts/namecheck.sh) 机械校验（CI 单列一步，用法见脚本头部注释）。

| # | 规则 | 要点 |
|---|---|---|
| 1 | 形态 `lasx_<domain>_<op>[_<qualifier>]` | `lasx_quat_to_dcm_batch`（domain `quat` / op `to_dcm` / qualifier `batch`）、`lasx_dot_i8`（qualifier 是类型） |
| 2 | 全小写 + 下划线 | **无大写、无连续下划线、无首尾下划线**；数字允许（`lasx_j2_accel_batch`、`lasx_batch_distance2d`） |
| 3 | **尾部后缀只允许 `_checked`** | "版本/口味"后缀（`_v2`、`_ex`、`_fast`、`_new`…）不许出现在新符号里；要区分就换 `<op>` 名，或用**语义/类型**限定词（`_batch`、`_rows`、`_i8`、`_f16` 这类是合法的） |
| 4 | `_checked` 必须有同名裸孪生 | `lasx_dot` ↔ `lasx_dot_checked`；**反之不要求**：裸符号可以没有 `_checked` |
| 5 | 库级符号只有白名单这 5 个 | `lasx_version`、`lasx_has_lasx`、`lasx_status_message`、`lasx_alloc`、`lasx_free`（逐个语义见 §3.6）。**新增库级符号必须同时改三处**：本表、§3.6、`scripts/namecheck.sh` 的白名单 |
| 6 | 参数次序 | 维度（`m,k,n`）→ 标量（`alpha`,`beta`）→ 输入（`a`/`b`/`x`/`w`）→ 输出（`c`/`y`/`out`）→ **`status` 永远最后**（`_checked` 的 `status` 可传 `NULL`） |
| 7 | 词根沿用既有约定 | 长度/批量维度一律 `n`；输出单值 `out`、矩阵/向量 `c`/`y`、多分量 `ox,oy,oz`；类型后缀 `_i8`/`_q4`/`_f16`/`_f64`。**别新造同义词**（`len`/`dst`/`total`…） |

**正例 / 反例**（反例只用于说明规则，**不会**出现在产物里）：

```c
/* ✅ 正例：domain=quantize、op=i8_per_row；孪生 lasx_quantize_i8_per_row_checked */
void lasx_quantize_i8_per_row(const float *x, int rows, int cols, float *out, int8_t *q, float *scale);

/* ❌ 反例：一次踩三条 —— 大写（规则 2）、口味后缀 _fast（规则 3）、只有 _checked 没有裸孪生（规则 4） */
void lasx_Dot_fast_checked(const float *a, const float *b, int n, int *status);
```

**新增符号前自查 5 条**：① 名字里有没有非法后缀？② 有 `_checked` 就必须有同名裸孪生；
③ 是不是库级符号？是就进白名单并改上面说的三处；④ 参数次序照规则 6（`status` 最后）；
⑤ 是否沿用了既有词根（`n`/`out`/`c`/`_i8`…），别新造同义词。

**历史例外（审计于 2026-10-15）**：**零违规**——87 个符号在"形态 / 后缀 / 孪生 / 库级"四类审计上
全部通过（`bash scripts/namecheck.sh` ⇒ `OK（87 个符号，0 例外）`）。下表只记"不是违规、但需要
知道的读法与白名单"，**没有为了凑表而编的例外**：

| 项 | 情况 | 为什么不改 |
|---|---|---|
| 库级 5 个（规则 5） | 不符合算子形态 `lasx_<domain>_<op>` | 它们是**白名单本身**（设计如此），不是例外；改名会破坏已发布 ABI |
| `lasx_batch_distance2d` | `batch` 在**前**，而多数批量算子把它放尾部（`lasx_norm3_batch`） | **不算违规**：按规则 1 的读法是 domain=`batch`、op=`distance2d`；改名会破坏已发布 ABI |
| 参数名的历史差异（`out` / `c` / `ox,oy,oz` / `table,ids,row_len`） | 各算子按自身语义取名，未完全统一 | 参数名**不影响二进制兼容**，但头文件已发布：改名只带来"头/源码/文档"漂移风险，没有收益 |

## 10. 这份文档如何保持为真

文档里的承诺不是"写完就算"，下面这些机器会盯着；**任何一条红了都说明文档或代码漂了**：

| 守什么 | 谁在守 | 怎么守 |
|---|---|---|
| **逐符号原型一致**（87 个都在、参数次序对） | [`scripts/check_header.c`](../scripts/check_header.c) | 把**全部 87 个符号的地址**放进数组（缺符号/改名/原型漂移 ⇒ 编译或链接失败），再逐个实调 41 个 `_checked`（断言 `status` 位置与取值域），并含融合缩放/能力探测/`alloc`↔`free` 的行为断言 |
| **导出符号个数 = 本文写的数** | panel CI「ABI symbol count」 | 数**产物**的动态符号表 `nm -D target/release/liblasx_rs.so`（而不是数源码），与文档里的计数比对 |
| **本文与 ops.md 的计数/表格不再各写一份** | `scripts/doccheck.sh` | 符号计数写法必须与源码实际数量一致；表格列数一致；`§` 引用可解析；同一 `§X` 引用 ≤12 次 |
| **调用体验真的能跑** | panel CI「C example (compile + run)」 | `make -C examples/c FALLBACK=1 && ./examples/c/quickstart`（编译**并运行** 2.4 那个程序） |
| **新增符号的名字合规** | [`scripts/namecheck.sh`](../scripts/namecheck.sh)（CI 单列一步） | 从产物符号表校验四条：形态 `^lasx_[a-z0-9]+(_[a-z0-9]+)*$`、尾部后缀只允许 `_checked`、`_checked` 必须有裸孪生、库级只有白名单 5 个（规则见 §9.1） |
| **实现没变** | panel CI 的 build/test/clippy/rustdoc 四步 | 全绿是硬门槛 |

自检命令（本地随时可跑）：

```bash
gcc -O2 -Iinclude scripts/check_header.c target/release/liblasx_rs.a \
    -lm -lpthread -ldl -o /tmp/check_header && /tmp/check_header
nm -D --defined-only target/release/liblasx_rs.so | grep -c ' T lasx_'   # 应等于本文的计数
bash scripts/doccheck.sh                                                 # 引用/表格/计数
bash scripts/namecheck.sh                                                # 命名规范（§9.1）
make -C examples/c FALLBACK=1 && ./examples/c/quickstart                 # 调用体验
```

**维护规则**：新增/删除/改名任何导出符号时，**同一提交**里改四处——`include/lasx_rs.h`、
本文第 3 节的清单与计数、`docs/ops.md` 里指向本文的指针（若它带数字）、README 的文档索引；
`scripts/check_header.c` 的地址数组由 CI 的「C header vs ABI」步骤强制对齐。
