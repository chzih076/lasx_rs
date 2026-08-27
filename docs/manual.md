# lasx_rs 技术手册

> 龙芯（LoongArch）LASX 256 位 / LSX 128 位向量库 · 完整技术文档
>
> 本手册逐条对照 `src/lib.rs`（1601 行）撰写，忠实于源码，不增删任何行为描述。
> 源码版本：`50088c6` 之上（ballistic 向量路径修复 + 新增对照测试 + 零警告清理）。

---

## 目录

1. [架构与设计](#1-架构与设计)
2. [内核详解（15 个 FFI 内核）](#2-内核详解15-个-ffi-内核)
3. [量化内核（lasx_dot_q4 / lasx_dot_i8）](#3-量化内核lasx_dot_q4--lasx_dot_i8)
4. [批量物理内核（j2_accel_batch / rk4_j2_step_batch / ballistic_step）](#4-批量物理内核j2_accel_batch--rk4_j2_step_batch--ballistic_step)
5. [FFI 使用指南](#5-ffi-使用指南)
6. [性能基准方法](#6-性能基准方法)
7. [测试与验证](#7-测试与验证)
8. [构建与集成](#8-构建与集成)
9. [Caveats 与限制](#9-caveats-与限制)
10. [API 索引](#10-api-索引)

---

## 1. 架构与设计

### 1.1 库定位

lasx_rs 是面向龙芯平台的零依赖向量库，通过 LoongArch 扩展指令集提供批量数值内核：

- **LASX**（LoongArch SIMD eXtension，256 位）：`lasx_xv*` intrinsics，一次处理
  8×f32 / 4×f64 / 32×i8；
- **LSX**（LoongArch SIMD eXtension，128 位）：`lsx_v*` intrinsics，一次处理
  4×f32 / 2×f64，用于无 LASX 的龙芯 CPU（如 3A5000 / 3A6000）降级。

crate 形态为 `cdylib + rlib`：`cdylib` 供 C / Dart 等经 `extern "C"` FFI 调用，
`rlib` 供 Rust 直接链接。`#![feature(stdarch_loongarch)]` 依赖 **nightly** Rust
实验特性（`stdarch_loongarch` 提供了 LoongArch 内建函数集合）。

### 1.2 LASX / LSX 双路径机制

#### 硬件能力检测：`has_lasx()`

```rust
static HW_HAS_LASX: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
fn has_lasx() -> bool {
    let hw = *HW_HAS_LASX.get_or_init(|| unsafe {
        let mut cfg2: u32;
        std::arch::asm!("cpucfg {}, {}", out(reg) cfg2, in(reg) 2u32);
        (cfg2 & (1 << 7)) != 0
    });
    hw && !FORCE_LSX.with(|c| c.get())
}
```

- 通过 LoongArch `cpucfg` 指令读取配置字 2（`CFG2`），检测 **bit 7（0x80）**
  是否为 1，即硬件是否含 LASX；
- 结果用 `OnceLock<bool>` 缓存，**进程级只探测一次**（cpucfg 只读一次）；
- 每次调用 `has_lasx()` 还需与线程级 `FORCE_LSX` 标志求与——见下。

#### 线程级强制降级钩子：`lasx_force_lsx_thread`

```rust
thread_local! {
    static FORCE_LSX: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
pub fn lasx_force_lsx_thread(force: bool) {
    FORCE_LSX.with(|c| c.set(force));
}
```

- 钩子仅影响**调用线程**内的向量内核，是**线程级**而非进程级；
- 注释明确说明：旧机制 `LOONGSCI_FORCE_LSX`（进程级 `OnceLock` 缓存）在并发测试
  时会让其他依赖 LASX 黄金值的测试随机走 LSX 路径而失败，故改为线程级，避免环境
  变量污染并发测试；
- 用途：在含 LASX 的真机（如 3B6000）上"模拟"LSX-only CPU，让代码真实执行
  LSX intrinsic 分支以验证数值/性能。源码注释强调这是**诚实标注**——并非无 LASX
  真机，只是执行路径相同；
- 测试结束时需置回 `false`。

> **注意**：`lasx_force_lsx_thread` 是普通 `pub fn`（非 `#[unsafe(no_mangle)]`），
> 只对 **rlib** 侧 Rust 调用者可见，`cdylib` 不会导出该符号。

#### 降级覆盖范围（重要事实）

**并非所有内核都实现了 LASX+LSX 双路径**。按 `has_lasx()` 检查的分布，15 个内核
分三类：

| 类别 | 内核 | 无 LASX 时的路径 |
|---|---|---|
| **LASX + LSX 双路径** | `lasx_dot`、`lasx_norm3_batch`、`lasx_vec3_add_scaled_batch`、`lasx_j2_accel_batch` | LSX 128 位向量路径 |
| **LASX + 标量降级** | `lasx_ballistic_step`、`lasx_batch_distance2d`、`lasx_rk4_j2_step_batch` | 全标量循环（无 LSX 向量路径） |
| **LASX-only（无降级）** | `lasx_matmul`、`lasx_axpy`、`lasx_sum`、`lasx_dot_f64`、`lasx_matmul_f64`、`lasx_dot_i8`、`lasx_dot_q4` | 无 `has_lasx()` 检查，LSX-only CPU 上会执行 LASX 指令（非法指令风险） |
| 内存工具 | `lasx_alloc` | 不涉及 SIMD |

（详见 §9.1。）

#### 逐位确定性（内核内部）

对**逐样本（element-wise）批量内核**（`lasx_norm3_batch`、`lasx_vec3_add_scaled_batch`、
`lasx_j2_accel_batch`、`lasx_rk4_j2_step_batch`、`lasx_batch_distance2d`），源码通过
"向量块与标量尾循环**同式、同结合序**"保证任一分块的向量部分与标量尾部分逐位一致
（同一输入、同一执行流内确定性）。例如：

```rust
// lasx_norm3_batch 向量路径（LASX 4 路）：
let sq = lasx_xvfadd_d(lasx_xvfmul_d(vx, vx),
        lasx_xvfadd_d(lasx_xvfmul_d(vy, vy), lasx_xvfmul_d(vz, vz)));
// 标量尾循环（注释："与向量路径同结合（x² + (y²+z²)），保证任一分块逐位一致"）：
out[j] = (xs[j] * xs[j] + (ys[j] * ys[j] + zs[j] * zs[j])).sqrt();
```

LASX 路径的标量尾与 LSX 路径的标量尾也都用同式，因此 LASX/LSX/标量尾在
**分块边界内**逐位一致。这一性质是本库"逐位确定"声明的精确含义：**向量块与
标量尾之间**一致；而**跨实现**（如与测试中采用不同公式的独立标量参考对比）的
保证是**相对误差 <1e-9**（见 §7）。

> **例外**：`lasx_ballistic_step` 不在上述逐位一致之列——其向量块与标量路径
> 的 `|v|²` 结合序不同（向量右结合 vs 标量左结合），故非逐位一致；但三处物理
> 语义（阻力符号、重力项、位置用新速度）已对齐标量 `euler_step`，由
> `test_ballistic_step_batch_matches_scalar` 守护（绝对误差 <1e-3 / <1e-2，
> 见 §4.1、§9.3）。

### 1.3 SIMD 数据布局（SOA / 数组）

- **单数组流式**内核（`lasx_dot`、`lasx_matmul`、`lasx_axpy`、`lasx_sum`、
  `lasx_dot_f64`、`lasx_dot_i8`、`lasx_dot_q4`）：输入为连续 `f32`/`f64`/`i8`/`u8`
  数组，按连续内存每 8 / 4 / 32 元素切块加载；
- **SOA（Structure-of-Arrays）批量内核**（`lasx_norm3_batch`、
  `lasx_vec3_add_scaled_batch`、`lasx_j2_accel_batch`、`lasx_rk4_j2_step_batch`、
  `lasx_ballistic_step`、`lasx_batch_distance2d`）：
  每个分量一个连续数组（`xs/ys/zs`、`rx/ry/rz/vx/vy/vz`、`x/y/z/vx/vy/vz/k`、
  `xs/ys/out`），LASX 一次处理 **4 个样本的同一分量**（f64 轨道内核）或 **8 个**
  （f32 弹道/距离内核），LSX 一次处理 2 个。源码注释：供 loong-sci
  `space::propagate_orbits_batch`（多星同时传播）使用。
- **类型别名**：
  - LASX：`F32x8 = m256`，`F64x4 = m256d`；
  - LSX：`F32x4 = m128`，`F64x2 = m128d`。
- **加载/存储辅助**：`ld_f32/st_f32`、`ld_f64/st_f64`、`ld4_f32/st4_f32`、
  `ld2_f64/st2_f64` 全部用 `lasx_xvld/xvst` / `lsx_vld/vst` 偏移量 **0** 的普通
  内存加载；`zero_f32/zero_f64/zero4_f32` 用 `xvldi(0)`/`vldi(0)` 置零；
  `splat_f32/splat_f64/splat2_f64` 用 `xvreplgr2vr_w/d` / `vreplgr2vr_d` 经
  bit 重解释做广播。

### 1.4 FFI 设计

- **导出约定**：全部 15 个内核为 `pub extern "C"` + `#[unsafe(no_mangle)]`，
  符号名即函数名（`lasx_*`），C ABI 可直接 `dlsym` 绑定；
- **裸指针语义**：参数为 `*const` / `*mut` 裸指针，函数内部立即
  `std::slice::from_raw_parts` 转成切片再运算；源码顶部显式
  `#![allow(clippy::not_unsafe_ptr_arg_deref)]`，说明"extern C FFI 函数按约定
  解引用传入裸指针（调用方保证有效），非 unsafe fn 语义"——即这些函数对外签名
  是 safe 的，但**调用方有义务保证指针有效、长度正确**；
- **caller 责任**：长度（n / m/k/n / n_bytes）必须与实际缓冲区容量一致；指针
  不得为悬垂或越界；`n` 不得为负（见 §9）；输出缓冲区须由调用方分配好足够空间
  （`lasx_matmul*` 的 `c` 为 `m*n` 元素，批量内核的 `out/ax/ay/az` 为 `n` 元素）；
- **内存分配**：`lasx_alloc(n)` 是唯一在库内分配内存的导出，返回未初始化缓冲
  区的裸指针（`Vec::with_capacity` + `mem::forget`），**无对等释放导出**，调用方
  需自行用 `Vec::from_raw_parts` 归还（见 §5.4）；
- **零依赖**：`Cargo.toml` 无任何 `[dependencies]`，仅 std + nightly 实验特性。

---

## 2. 内核详解（15 个 FFI 内核）

> 各内核"精度保证"一栏描述其与自身标量尾/测试参考的数值关系；"性能"一栏仅列出
> README 中记录的实测数据，未记录者注明"README 无独立数据"。

### 2.1 `lasx_dot` — f32 点积（带 LSX 降级）

- **C 签名**：`float lasx_dot(const float *a, const float *b, int n)`
- **算法**：`Σ_{i<n} a[i]·b[i]`。
- **向量化策略**：
  - LASX 路径（`has_lasx()` 为真）：每向量 8 个 f32；主循环按 **64 元素一块**
    （8 次 `lasx_xvfmadd_s`），每块把 8 通道向量累加器落内存、逐分量转 `f64`
    加入 `acc_d`（**f64 累加**）并重置向量累加器；随后 8 元素向量尾循环；剩余
    标量尾循环。
  - LSX 路径：每向量 4 个 f32，**64 元素一块 = 16 次 `lsx_vfmadd_s`**，同样每块
    落盘到 f64 累加。
- **精度保证**：f32 向量内 FMA 累积，每 64 元素落盘到 f64 累加"抑制累加误差"，
  最终 `acc_d as f32` 一次舍入返回。相比纯 f32 累加显著降低舍入累积。
- **性能**：README 记录批量点积（n≥8）加速 **2.4–2.8×**（Loongson-3B6000）。
- **适用场景**：大规模 f32 内积、注意力相似度、向量相似性搜索。

### 2.2 `lasx_matmul` — f32 矩阵乘

- **C 签名**：`void lasx_matmul(int m, int k, int n, const float *a, const float *b, float *c)`
- **算法**：`C[m×n] = A[m×k] × B[k×n]`。A、B、C 均为**行主序**；源码先分配
  `b_t = vec![0f32; n*k]`，将 B 转置为 n×k 行主序（`b_t[j*k+p] = b[p*n+j]`），
  使每行内积沿连续内存步进。
- **向量化策略**：对每个输出元素 `C[i][j]` 用一个 8 通道 f32 向量累加器，沿
  k 维以 8 元素步进 `lasx_xvfmadd_s`，尾部标量求和。
- **精度保证**：**f32 向量内直接累加**（与 `lasx_dot` 不同，没有 f64 落盘），
  精度低于点积；尾循环累加至 `s`（f32）。无测试守护。
- **性能**：README 无独立数据。
- **适用场景**：f32 权重矩阵 × 激活矩阵（推理场景），对精度敏感的应用建议用
  `lasx_matmul_f64`。
- **备注**：B 转置产生 **O(n·k) 临时堆分配**（见 §9.6）。

### 2.3 `lasx_axpy` — y += a·x

- **C 签名**：`void lasx_axpy(float alpha, const float *x, float *y, int n)`
- **算法**：`y[i] += alpha · x[i]`。
- **向量化策略**：`splat_f32(alpha)` 广播；每 8 元素一次
  `lasx_xvfmadd_s(vx, va, vy)`（单指令 `alpha·x + y`），写回 `y`；标量尾循环
  `y[j] += alpha * x[j]`。
- **精度保证**：向量路径为 FMA（单舍入）；标量尾为分离 mul+add（两次舍入），
  两者至多差 1 ulp，且**无逐位一致守护**（见 §9.8）。
- **性能**：README 无独立数据。
- **适用场景**：BLAS 级 axpy、迭代求解器、梯度更新。

### 2.4 `lasx_sum` — f32 向量归约

- **C 签名**：`float lasx_sum(const float *x, int n)`
- **算法**：`Σ_{i<n} x[i]`。
- **向量化策略**：与 `lasx_dot` 同构——每 64 元素一块（8 次 `lasx_xvfadd_s`），
  每块落盘、逐分量转 f64 累加（`acc_d`），随后 8 元素向量尾、标量尾。
- **精度保证**：f64 累加抑制 f32 累加误差，最终一次 `acc_d as f32`。
- **性能**：README 无独立数据。
- **适用场景**：归约、softmax 分母、归一化因子。
- **⚠️ 无 LSX 降级**：`lasx_sum` 体内没有 `has_lasx()` 检查，**LASX-only**。

### 2.5 `lasx_dot_f64` — f64 点积

- **C 签名**：`double lasx_dot_f64(const double *a, const double *b, int n)`
- **算法**：`Σ a[i]·b[i]`（f64）。
- **向量化策略**：每向量 4 个 f64（`lasx_xvfmadd_d`），主循环按 4 元素步进；
  结束时向量累加器落盘、逐分量相加，标量尾直接累加。
- **精度保证**：全程 f64；向量 FMA 累加 + 落盘 f64 累加，误差为 IEEE 双精度
  舍入。无测试守护。
- **性能**：README 无独立数据。
- **适用场景**：f64 内积、轨道/物理标量积。
- **⚠️ 无 LSX 降级**：LASX-only。

### 2.6 `lasx_alloc` — 内存分配（FFI 侧）

- **C 签名**：`float *lasx_alloc(int n)`
- **算法**：`Vec::with_capacity(n)` → `as_mut_ptr()` → `mem::forget(v)`，返回
  裸指针。**内存未初始化**（capacity 分配但未写入）。
- **备注**：唯一在库内分配内存的导出；无 `lasx_free`，调用方必须自行归还
  （§5.4）。`n < 0` 时 `with_capacity` 会 panic（详见 §9.5）。
- **适用场景**：Dart / C 侧需要由本库分配输出缓冲时的配套工具。

### 2.7 `lasx_dot_i8` — int8 量化点积

- **C 签名**：`int lasx_dot_i8(const int8_t *a, const int8_t *b, int n)`
- **算法**：`Σ a[i]·b[i]`，int8 输入、i32 输出。
- **向量化策略**：每向量 **32 个 i8**（LASX 256 位 = 32 字节）；用
  `lasx_xvmulwev_h_b`（偶通道 i8 扩展乘→16 个 i16）与 `lasx_xvmulwod_h_b`
  （奇通道）分别做带符号乘，`lasx_xvadd_h` 合并，得 16 个 i16 部分积；每 32
  字节落盘，用 **i64 累加防溢出**；标量尾循环。
- **精度保证**：整数运算**精确**（无限精度意义下），前提是最终和可容纳于 i64
  累加器；返回时 `s_acc as i32` 直接截断——**n 足够大时 i32 结果可能回绕**
  （见 §9.12）。
- **性能**：README 无独立数据。
- **适用场景**：int8 量化模型内积、embedding 量化。
- **⚠️ 无 LSX 降级**：LASX-only。

### 2.8 `lasx_dot_q4` — Q4 量化点积

- **C 签名**：`double lasx_dot_q4(const uint8_t *qa, const float *sa, const uint8_t *qb, const float *sb, int n_bytes)`
- **算法**：Q4 量化点积（详见 §3）。`qa/qb` 为量化字节数组，`sa/sb` 为逐组
  scale（f32），`n_bytes` 为量化数据字节数。
- **向量化策略**：每 32 字节一组：`xvld` 载入，`xvandi_b`（掩低 4 位）与
  `xvsrli_b`（右移 4）拆出高低 nibble，`xvmulwev_h_b`/`xvmulwod_h_b` 分别乘偶/
  奇通道 i16 并 `xvadd_h` 合并，得到 16 个 i16 的部分积（低、高 nibble 两组分别
  求和再合并），落盘用 i64 累加出组内 64-nibble 点积；再乘该组 scale 乘积加入
  f64 总累加。
- **精度保证**：组内整数点积 i64 精确；scale 为 f32×f32 乘积（f32 精度）再转
  f64 乘以整数点积——scale 自身精度是量化误差的边界（见 §3.3）。
- **性能**：README 无独立数据。
- **适用场景**：Q4 量化模型（GGUF 风格）的快速内积。
- **⚠️ 无 LSX 降级**：LASX-only。

### 2.9 `lasx_matmul_f64` — f64 矩阵乘

- **C 签名**：`void lasx_matmul_f64(int m, int k, int n, const double *a, const double *b, double *c)`
- **算法 / 布局**：与 `lasx_matmul` 完全同构（行主序 + B 转置为 n×k），仅换为
  f64，每向量 **4 个 f64**（`lasx_xvfmadd_d`）。
- **精度保证**：全程 f64；向量累加器落盘后 f64 累加。无测试守护。
- **性能**：README 无独立数据。
- **适用场景**：f64 矩阵运算、物理/轨道动力学、高精度推理。
- **⚠️ 无 LSX 降级**：LASX-only。

### 2.10 `lasx_ballistic_step` — 批量弹道步（欧拉）

- **C 签名**：`void lasx_ballistic_step(float *x, float *y, float *z, float *vx, float *vy, float *vz, const float *k, int n, float dt, float g)`
- **算法**：SOA 布局，n 发弹（每发 6 状态 + 阻力系数 k）同步推进一步**欧拉**：
  `|v| = √(vx²+vy²+vz²)`；`drag = k·|v|`；`vx -= drag·vx·dt`、
  `vy -= (drag·vy + g)·dt`、`vz -= drag·vz·dt`；位置用**新速度**更新
  （`x += vx_new·dt`）。
- **向量化策略**：LASX **8 发并行**单循环；`|v|²` 用 `xvfmul_s`/`xvfadd_s`
  右结合组合，`xvfsqrt_s` 开方；阻力项用 `xvfsub_s` 做减、重力项
  `g·dt` 广播后单独相减、位置增量用 `xvfmadd_s`（`x + vx_new·dt`）。
- **降级**：无 LSX 向量路径——非 LASX 时**全标量** `euler_step`。
- **与标量参考一致性**：阻力符号、重力项、位置用新速度三处**均已修复对齐**
  标量 `euler_step`（见 §4.1）；物理上一致，仅因 `|v|²` 结合序不同
  （向量右结合 vs 标量左结合）**非逐位一致**，由测试
  `test_ballistic_step_batch_matches_scalar` 守护（速度 <1e-3、位置 <1e-2
  绝对误差）。
- **性能**：README 无独立数据（"全阶引力批量"3.84× 指 J2 相关内核）。
- **适用场景**：弹道轨迹批处理模拟。

### 2.11 `lasx_batch_distance2d` — 批量 2D 距离

- **C 签名**：`void lasx_batch_distance2d(float px, float py, const float *xs, const float *ys, float *out, int n)`
- **算法**：对每个点 `(xs[i], ys[i])` 求到 `(px, py)` 的欧氏距离
  `d = √(dx²+dy²)`，`dx = xs[i]-px`、`dy = ys[i]-py`。
- **向量化策略**：LASX **8 点并行**：`xvfsub_s` 求差、`xvfmul_s`+`xvfadd_s`
  求平方和（`fadd(fmul, fmul)` 结合序）、`xvfsqrt_s` 开方。
- **精度保证**：向量路径与标量尾**同式同结合序**（`dx*dx + dy*dy` 亦为
  `fadd(fmul, fmul)`），分块内逐位一致；`xvfsqrt_s` 与标量 `f32::sqrt` 同为严格
  舍入。无测试守护。
- **降级**：非 LASX 时全标量（无 LSX 向量路径）。
- **性能**：README 无独立数据。
- **适用场景**：空间查询、最近邻、碰撞检测的距离计算。

### 2.12 `lasx_norm3_batch` — 批量 3 分量模长（f64 SOA）

- **C 签名**：`void lasx_norm3_batch(const double *xs, const double *ys, const double *zs, double *out, int n)`
- **算法**：`out[i] = √(xs[i]²+ys[i]²+zs[i]²)`。
- **向量化策略**：LASX **4 样本/向量**（f64），`xvfmul_d`/`xvfadd_d` 右结合求
  `|r|²`、`xvfsqrt_d` 开方；LSX **2 样本/向量**（`lsx_v*`）；标量尾同式兜底。
- **精度保证**：向量路径与标量尾**同结合序**（`x² + (y²+z²)`），分块内逐位
  一致。测试 `test_norm3_batch_matches_scalar` 守护（与独立参考相对误差 <1e-9）。
- **性能**：README 无独立数据（与 J2 批量内核同属轨道传播链路）。
- **适用场景**：多星轨道模长计算、归一化。

### 2.13 `lasx_vec3_add_scaled_batch` — 批量"缩放加"

- **C 签名**：`void lasx_vec3_add_scaled_batch(const double *ax, const double *ay, const double *az, const double *bx, const double *by, const double *bz, double s, double *ox, double *oy, double *oz, int n)`
- **算法**：`o[i] = a[i] + s·b[i]`（x/y/z 三分量独立，SOA）。注释指出覆盖了
  **加（s=1）与缩放（a=0）** 两个退化情形。
- **向量化策略**：LASX 4 样本/向量，`lasx_xvfmadd_d(vs, vb, va)` **单条 FMA**
  完成；LSX 2 样本/向量；标量尾用 `f64::mul_add` 同式。
- **精度保证**：注释"单次 FMA，误差 ≤ 标量 mul+add 两舍入"；向量与标量尾同式
  → 分块内逐位一致。测试 `test_vec3_add_scaled_batch_matches_scalar` 守护
  （相对误差 <1e-9）。
- **性能**：README 无独立数据。
- **适用场景**：轨道状态缩放组合、`r + s·lsum` 形式的 RK 状态更新。

### 2.14 `lasx_j2_accel_batch` — 批量 J2 引力加速度

- **C 签名**：`void lasx_j2_accel_batch(const double *rx, const double *ry, const double *rz, double mu, double j2, double re, double *ax, double *ay, double *az, int n)`
- **算法**：中心点质量 + J2 摄动加速度（详见 §4.2），与 loong-sci
  `EarthModel::j2_acceleration` 同公式。
- **向量化策略**：逐样本（lane）独立运算 → **天然 4 路并行**（LASX f64）；
  LSX 2 路；标量尾同式兜底。
- **精度保证**：向量与标量尾逐位一致（同结合/除式/末项 FMA）；与测试独立参考
  （不同公式实现）相对误差 <1e-9（`test_j2_accel_batch_matches_scalar` 守护）。
- **性能**：README "全阶引力批量" **3.84×** 即属此类内核链路。
- **适用场景**：多星 J2 摄动加速度计算、轨道传播内层。

### 2.15 `lasx_rk4_j2_step_batch` — 批量 RK4 J2 步

- **C 签名**：`void lasx_rk4_j2_step_batch(double *rx, double *ry, double *rz, double *vx, double *vy, double *vz, double mu, double j2, double re, double dt, int n)`
- **算法**：经典 4 阶 Runge–Kutta 单步（中心项 + J2 力模型），批量推进 N 个
  轨道状态 `[r(3), v(3)]`（详见 §4.3）。
- **向量化策略**：LASX 一次处理 **4 个样本**；`k1..k4` 与组合算术全部在寄存器
  内完成——源码注释"少一次内存往返即省 30+ 次内核调用开销"；**原地更新**
  （每块先读后写，块间不重叠）；非 LASX 走全标量 `rk4_j2_step_scalar`。
- **精度保证**：向量路径与标量 `rk4_j2_step_scalar` **同式同 FMA 形式** →
  分块内逐位一致；测试多步累积相对误差 <1e-9（`test_rk4_j2_step_batch_matches_scalar`
  守护）。
- **性能**：README "24 线程批量传播" **~18×** 即此类内核链路（含多线程调度）。
- **适用场景**：多星轨道批量传播（loong-sci `propagate_orbits_batch`）。

---

## 3. 量化内核（lasx_dot_q4 / lasx_dot_i8）

### 3.1 `lasx_dot_i8`：8-bit 量化点积

- 输入：两个 `int8_t` 数组，长度 n；
- 计算：`Σ a[i]·b[i]`，精确整数运算；
- 向量化：LASX 一向量 32 个 i8。`lasx_xvmulwev_h_b` / `lasx_xvmulwod_h_b` 分别
  抽取**偶/奇通道**做带符号 8→16 位扩展乘法，各产出 16 个 i16；`xvadd_h` 合并；
  每 32 字节（一向量）落盘，逐 i16 转 i64 累加，"避免溢出"；
- 输出：`i32`。若 n 足够大使真实点积超出 i32 范围，`s_acc as i32` 会**回绕**
  （无饱和/报错），这是使用边界（§9.12）。

### 3.2 `lasx_dot_q4`：4-bit 量化点积（GGUF 风格）

- **量化格式**：每个 `u8` 字节含 **2 个 4-bit 无符号 nibble**：
  - 低 nibble = `qa[j] & 0x0f`（`xvandi_b` 掩码）；
  - 高 nibble = `(qa[j] >> 4) & 0x0f`（`xvsrli_b` 右移 4）；
- **scale 布局**：`sa` / `sb`（f32）每 **32 字节一组**一个值，组数
  `n_groups = ceil(n_bytes / 32)`（源码 `n.div_ceil(32)`）；
- **点积公式**（源码注释）：

  ```
  dot = Σ_g scale_a[g] · scale_b[g] · Σ_{64 nibble} (qa·qb)
  ```

  即：组内 64 个 nibble（32 字节 × 2）的整数点积先算出，再乘该组两个 scale 的
  乘积（f32），全部累加进 f64。

- **向量化**：`xvandi_b` 拆低 nibble、`xvsrli_b` 拆高 nibble，分别用
  `xvmulwev_h_b`/`xvmulwod_h_b`（偶/奇通道 i8×i8→i16）+ `xvadd_h` 合并，低、
  高两组再加，得 16 个 i16；落盘转 i64 累加出**组内点积**（精确整数）；再乘
  scale 积、加入 f64 累加。

### 3.3 精度特征

- 整数部分（nibble 点积、i8 点积）**精确**；
- 误差来源：**scale 为 f32 存储**，`sa[g]*sb[g]` 是 f32 乘法（约 2^-24 相对
  精度），再与整数点积相乘——量化误差主要由 scale 表示精度决定；最终累加在
  f64，组间累积不引入额外舍入；
- 本库两个量化内核均**无测试守护**（测试覆盖见 §7 均为 f64 批量物理内核）。

---

## 4. 批量物理内核（j2_accel_batch / rk4_j2_step_batch / ballistic_step）

### 4.1 `lasx_ballistic_step`：批量弹道步（欧拉）

**物理模型**（标量参考 `euler_step`，源码原样）：

```rust
let v  = (vx² + vy² + vz²).sqrt();
let drag = k * v;
vx -= drag * vx * dt;
vy -= (drag * vy + g) * dt;
vz -= drag * vz * dt;
x  += vx * dt;   // 注意：这里 vx 已是更新后的新速度
y  += vy * dt;
z  += vz * dt;
```

**LASX 向量路径**（8 发并行）实际计算的表达式（源码原样，修复后）：

```rust
vsq    = vx² + (vy² + vz²)          // xvfmul_s / xvfadd_s 右结合
vmag   = sqrt(vsq)
vdrag  = k * vmag
vdvx   = vdrag * vx                 // dvx = −drag·vx（此处先存正值，下面相减）
vdvy   = vdrag * vy
vdvz   = vdrag * vz
vnx    = vx − vdvx·dt               // xvfsub_s：vx -= drag·vx·dt
vny    = (vy − vdvy·dt) − g·dt      // 额外减 g·dt（重力项）
vnz    = vz − vdvz·dt
xnx    = x  + vnx_new·dt            // xvfmadd_s：位置用**新**速度
```

**修复对照**（此前版本的三处物理差异均已对齐标量 `euler_step`）：

| 项目 | 标量 `euler_step` | LASX 向量路径（修复后） |
|---|---|---|
| 阻力项符号 | `vx − drag·vx·dt`（减） | `vx − vdrag·vx·dt`（`xvfsub_s` 相减，一致） |
| 重力项 `g` | `vy − (drag·vy + g)·dt`（含 g） | `(vy − vdrag·vy·dt) − g·dt`（含 g，一致） |
| 位置更新速度 | **新** vx | **新** vnx（`xvfmadd_s(vnx, vdt, x)`，一致） |

**剩余差异（数值级，非物理）**：`|v|²` 的结合序不同——向量路径右结合
`vx² + (vy² + vz²)`，标量 `euler_step` 为左结合 `(vx² + vy²) + vz²`（Rust 从左到右
求值）。f32 下两者舍入路径不同，故向量块与标量路径**非逐位一致**，但物理等价，
测试 `test_ballistic_step_batch_matches_scalar` 断言速度绝对误差 <1e-3、位置
<1e-2（n=64，初始速度 vx=300+i·10、vy=40−i·2，dt=0.005，g=9.81，k=1e-5）。

> 结论：该内核目前两条路径物理一致，LASX CPU 上默认走向量路径的行为与标量
> `euler_step` 等价，无需再强制降级。此前的"不一致"状态已修复并有测试守护。

### 4.2 `lasx_j2_accel_batch`：J2 引力加速度

**物理公式**（源码注释 + 代码还原）：

```
rm²  = x² + (y² + z²)                // 右结合，与向量路径一致
rm   = √rm²
rm³  = rm · rm²
rm⁵  = rm³ · rm²
j2k  = 1.5 · J2 · μ · Re²            // 取正值（注释：−k 即此值）
zr2  = (z²) / rm²
k    = j2k / rm⁵
vcen = −μ / rm³
a_x  = k·x·(5·zr2 − 1) + vcen·x     // 末项用 FMA：fma(k·x, 5·zr2−1, vcen·x)
a_y  = k·y·(5·zr2 − 1) + vcen·y
a_z  = k·z·(5·zr2 − 3) + vcen·z
```

**批处理布局**：SOA——`rx/ry/rz` 各为 n 长连续 f64 数组，输出 `ax/ay/az` 同
布局；每样本（lane）计算完全独立，天然 4 路（LASX）/ 2 路（LSX）并行，无跨
lane 归约。

**数值特征**：
- `k` 在代码中取**正值**，合并进末项 FMA 时符号为 `+k·x·(5zr2−1)`；注释说明
  "k 正 = −k_j2"（公式中等价于减 J2 项）；
- 末项用 `lasx_xvfmadd_d` / `f64::mul_add`（单条 `fmadd.d`，LA664），向量路径
  与标量尾**同式同 FMA** → 分块内逐位一致；
- 测试参考 `scalar_j2`（不同实现：`zr2 = (z/rm)²`、分离 mul+add、左结合 rm²）
  与内核物理等价，50 步/单次断言相对误差 <1e-9。

### 4.3 `lasx_rk4_j2_step_batch`：RK4 J2 步

**RK4 步进公式**（FMA 形式，`h` = dt，`hh` = dt/2，`h6` = dt/6）：

```
k1 = f(r0)
r2 = fma(hh, v0, r0);  v2 = fma(hh, k1, v0)
k2 = f(r2)
r3 = fma(hh, v2, r0);  v3 = fma(hh, k2, v0)
k3 = f(r3)
r4 = fma(h,  v3, r0);  v4 = fma(h,  k3, v0)
k4 = f(r4)
l  = fma(2, v2, v0) + fma(2, v3, v4)      // = v0 + 2·v2 + 2·v3 + v4
k  = fma(2, k2, k1) + fma(2, k3, k4)      // = k1 + 2·k2 + 2·k3 + k4
r' = fma(h6, l, r0)
v' = fma(h6, k, v0)
```

其中 `f(r) = j2_accel_vec(...)`，即 §4.2 的 J2 加速度的 lane 版本（`vmu` 已
预取负），与 `lasx_j2_accel_batch` 同式。

**批处理布局**：SOA，6 个数组（r、v 各三分量）均 n 长；`n` 个状态互不依赖；
LASX 每向量 4 个样本，k1..k4 及组合算术**全在寄存器内**完成（源码注释：省 30+
次内核调用开销）；**原地更新**（每块先读后写，块间不重叠）。

**数值特征**：
- 全程 FMA（`lasx_xvfmadd_d` / 标量 `f64::mul_add`），标量
  `rk4_j2_step_scalar` 与向量路径逐式对应 → 分块内逐位一致；
- 测试 `test_rk4_j2_step_batch_matches_scalar`：批量向量路径 vs 逐星标量
  `rk4_j2_step_scalar`，**50 步**、dt=10、n∈{1,2,3,4,5,7,8,17}，相对误差
  <1e-9；
- 测试 `test_rk4_j2_step_scalar_fma_vs_plain_reference`：FMA 版（`mul_add`）vs
  分离 mul+add 的旧版参考 `step_plain`——单步 FMA 融合舍入差 ≤1 ulp，50 步
  累积相对差仍 <1e-9（物理等价）。

---

## 5. FFI 使用指南

### 5.1 C 调用

以 `lasx_dot` 为例（liblasx_rs.so / lasx_rs.dll / liblasx_rs.a 取决于平台）：

```c
#include <stdint.h>

/* 声明（与 Rust 侧签名一一对应） */
float lasx_dot(const float *a, const float *b, int n);
void  lasx_axpy(float alpha, const float *x, float *y, int n);
void  lasx_matmul(int m, int k, int n,
                  const float *a, const float *b, float *c);
void  lasx_ballistic_step(float *x, float *y, float *z,
                          float *vx, float *vy, float *vz,
                          const float *k, int n, float dt, float g);
void  lasx_j2_accel_batch(const double *rx, const double *ry, const double *rz,
                          double mu, double j2, double re,
                          double *ax, double *ay, double *az, int n);
void  lasx_rk4_j2_step_batch(double *rx, double *ry, double *rz,
                             double *vx, double *vy, double *vz,
                             double mu, double j2, double re,
                             double dt, int n);
/* ... 其余 15 个内核同理 */

float a[1024], b[1024];
float d = lasx_dot(a, b, 1024);   /* 注意：n 传入 int（i32） */
```

要点：
- `n` / `m,k,n` / `n_bytes` 均为 `int`（i32）；
- 标量参数 `dt`、`g`、`mu`、`j2`、`re`、`s`、`px`、`py`、`alpha` 按值传；
- 输出参数为 `*mut`（C 侧 `float*`/`double*`），需预先分配足够空间。

### 5.2 Dart 调用（DynamicLibrary）

```dart
final lib = DynamicLibrary.open('liblasx_rs.so');

final lasxDot = lib.lookupFunction<
    Float Function(Pointer<Float> a, Pointer<Float> b, Int32 n),
    Float Function(Pointer<Float> a, Pointer<Float> b, int n)>('lasx_dot');

// lasx_alloc：分配输出缓冲（需自行归还，见 5.4）
final lasxAlloc = lib.lookupFunction<
    Pointer<Float> Function(Int32 n),
    Pointer<Float> Function(int n)>('lasx_alloc');
```

注意：Dart 侧 `n` 为 `int`，Rust 侧为 `Int32`；`lookupFunction` 的泛型参数
（Native 侧签名 / Dart 侧签名）必须与 C 签名完全一致，否则 UB。

### 5.3 Rust 调用（rlib）

直接 `use lasx_rs::*` 即可（函数为 safe 但接受裸指针）：

```rust
use lasx_rs::{lasx_dot, lasx_force_lsx_thread};

let a: Vec<f32> = ...;
let b: Vec<f32> = ...;
let d = unsafe { lasx_dot(a.as_ptr(), b.as_ptr(), n) };
// 正确写法是 .as_ptr()——函数签名是裸指针，不是 &Vec/slice。

// 强制当前线程走 LSX 路径（验证/测试钩子，仅本线程生效）
lasx_force_lsx_thread(true);
let d_lsx = unsafe { lasx_dot(a.as_ptr(), b.as_ptr(), n) };
lasx_force_lsx_thread(false);
```

> **README 纠正**：主 README 用法示例写作 `lasx_rs::lasx_dot(&a, &b, n)`，
> 与真实签名（`*const f32`）不符，实际应传 `.as_ptr()`。

### 5.4 内存管理（lasx_alloc / 指针生命周期）

- `lasx_alloc(n)` 返回 `Vec::with_capacity(n)` 的裸指针，**未初始化**、库内无
  释放导出；
- 调用方释放（Rust 侧归还给 Vec）：

```rust
// 归还由 lasx_alloc 分配的缓冲区（假设已写入 n 个 f32）
unsafe {
    let v = Vec::from_raw_parts(ptr, n, n);
    drop(v); // 释放
}
```

- Dart 侧同样需要归还：持有原始指针与容量，用 `malloc` 的 release 语义（或经
  Dart 侧 `Vec.fromRawParts` 语义的辅助库）释放；**不释放则泄漏**；
- **生命周期约定**：所有内核只在调用期间读取/写入传入缓冲区，不持有任何指针
  （无回调、无异步），因此调用方在函数返回后即可安全释放输入/输出缓冲；
- `lasx_matmul` / `lasx_matmul_f64` 内部转置临时缓冲为函数内分配、函数内释放，
  与调用方无关。

---

## 6. 性能基准方法

### 6.1 已记录数据（来自 README，Loongson-3B6000，LA664）

| 内核 | 加速比 |
|---|---|
| 批量点积（n≥8） | 2.4–2.8× |
| 全阶引力批量 | 3.84× |
| 24 线程批量传播 | ~18× |

仓库内**没有**基准测试代码（无 `benches/`，无 criterion 依赖），上述为 README
记录的实测结果。

### 6.2 复现建议（基于现有钩子，无新增依赖）

本库提供的**线程级强制降级钩子**使"同机对比 LASX 与 LSX"成为可能——3B6000
含 LASX，直接用钩子让代码真实执行 LSX 分支：

1. **LASX vs 标量**：对 n 从 8 到 2^20 的批量输入计时 `lasx_dot` 等（排除
   首调缓存/cpucfg 探测），加速比 = 标量时间 / 向量时间；LASX 时间含 `has_lasx()`
   分支判断（首调一次 cpucfg，已缓存）。
2. **LASX vs LSX**：同一输入、同一线程分别以
   `lasx_force_lsx_thread(false)` 与 `lasx_force_lsx_thread(true)` 计时。
   注意 `FORCE_LSX` 是线程级，基准线程须自行置回 `false`。
3. **多线程批量传播**：对 `lasx_rk4_j2_step_batch` 按 24 线程分块调度（每线程
   独立 n，内核本身无全局状态，可并发调用），记录相对单线程的扩展比（README
   ~18×）。
4. **诚实性**：钩子只是让代码在 LASX CPU 上**执行 LSX 指令分支**，并非无 LASX
   真机；需要 LSX-only 真机数据时仍应在 3A5000/3A6000 上实测。

### 6.3 度量口径建议

- 用 `--release`（`cdylib`/`rlib` 优化构建）；
- 时间取多次中位数；大 n 预热；
- 注意 f32 点积路径的 64 元素分块落盘频率（每块一次内存往返）对吞吐的影响，
  可对比不同 n 观察块边界效应。

---

## 7. 测试与验证

### 7.1 测试清单（`src/lib.rs` 底部 `mod batch_tests`，共 6 个）

| # | 测试名 | 验证内容 | 断言 |
|---|---|---|---|
| 1 | `test_norm3_batch_matches_scalar` | `lasx_norm3_batch` vs 独立标量参考 `scalar_norm3` | 相对误差 <1e-9，n∈{0,1,2,3,4,5,7,8,16,33} |
| 2 | `test_vec3_add_scaled_batch_matches_scalar` | `lasx_vec3_add_scaled_batch` vs `scalar_add_scaled`（`x + s·y` 分离乘加） | <1e-9，n=37，s∈{0,0.5,2,−1.3,1} |
| 3 | `test_rk4_j2_step_batch_matches_scalar` | 批量 `lasx_rk4_j2_step_batch` vs 逐星标量 `rk4_j2_step_scalar`（同公式） | **50 步**累积 <1e-9，n∈{1,2,3,4,5,7,8,17}，dt=10 |
| 4 | `test_j2_accel_batch_matches_scalar` | `lasx_j2_accel_batch` vs `scalar_j2`（不同公式实现） | <1e-9，n∈{0,1,2,4,9,20} |
| 5 | `test_rk4_j2_step_scalar_fma_vs_plain_reference` | FMA 版 `rk4_j2_step_scalar`（`f64::mul_add`）vs 分离 mul+add 旧版参考 `step_plain` | 单步 ≤1 ulp；**50 步**累积 <1e-9 |
| 6 | `test_ballistic_step_batch_matches_scalar` | `lasx_ballistic_step` 向量路径 vs 标量 `euler_step` 全量参考（修复阻力符号+重力+位置用新速度后） | 速度绝对误差 <1e-3、位置 <1e-2，n=64，dt=0.005 |

> 注：第 6 项为**绝对误差**断言（非 `rel_err` 相对误差），因弹道量为 f32 且速度
> 可能过零，相对度量不稳定；该测试守护 ballistic 向量路径与标量参考的物理一致性
> （§4.1 的剩余结合序差异）。

### 7.2 测试装置

- **伪随机轨道状态生成** `states(n)`：LEO/GEO/椭圆/高轨混合（半长轴
  `6.8e6 ~ 6.8e6+(n-1)*1.7e6` m，偏心率 0.05~0.35，随机相位 `2.39996`/`1.131`），
  注释"避免巧合"；速度测试向量为确定性递增序列；
- **ballistic 测试输入**（第 6 项）：n=64，初始 `x/y/z=0`、`vx=300+i·10`、
  `vy=40−i·2`、`vz=0`（确定性递增），`k=1e-5` 全 1、`dt=0.005`、`g=9.81`；
- **相对误差度量** `rel_err(a,b) = |a−b| / max(1, |b|)`——对 ~1e7 量级轨道
  状态用相对度量，分母下限 1 防除零；
- **物理常数**：`mu = 3.986004418e14`，`j2 = 1.08262668e-3`，`re = 6.378137e6`。

### 7.3 逐位确定性保证的边界

- **内核内部**：有 `has_lasx()` 分支的批量内核，向量块与标量尾循环**同式同结合
  序**，任一分块的执行结果逐位一致（如 §1.2 所引 `lasx_norm3_batch` 注释）；
- **跨实现**：测试参考（如 `scalar_j2` 用 `(z/rm)²` 与左结合、`scalar_add_scaled`
  用分离 mul+add、`scalar_norm3` 用左结合）与内核采用不同表达式，断言为
  **相对误差 <1e-9** 而非逐位相等；
- **FMA vs plain**：单条 FMA 与分离乘加至多差 1 ulp，50 步 RK4 累积仍 <1e-9；
- 因此"逐位确定"应理解为**同实现内分块一致** + **跨实现物理等价 <1e-9**。

### 7.4 如何扩展测试

按现有模式（对照实现 + `rel_err`），可补充：

- `lasx_dot` / `lasx_sum` / `lasx_axpy` / `lasx_dot_f64` 的标量对照（f64 累加
  参考，或对 LSX 强制路径做黄金值对照）；
- `lasx_dot_i8` / `lasx_dot_q4` 的逐位整数参考（i64 精确累加对照）；
- 在 `lasx_force_lsx_thread(true)` 下重跑全部批量内核测试，验证 LSX 路径与
  LASX 路径结果一致（当前测试未显式覆盖 LSX 强制路径）；
- `lasx_matmul` / `lasx_matmul_f64` 与朴素三重循环参考的对照；
- 边界：n=0/1/2/3、块边界 64±1、n_bytes 非 32 倍数的 q4；
- 注意测试需在 **LoongArch 真机**上运行（GitHub 无 loongarch64 runner，见 §8）。

---

## 8. 构建与集成

### 8.1 工具链与特性

- **nightly Rust**（含 `stdarch_loongarch` 实验特性），库顶：
  `#![feature(stdarch_loongarch)]`；
- `Cargo.toml`：`edition = "2021"`，`license = "MIT"`，
  `[lib] crate-type = ["cdylib", "rlib"]`，零依赖；
- **`.cargo/config.toml`**（源码实际内容）：

  ```toml
  [build]
  rustflags = ["-C", "target-feature=+lasx"]
  ```

  > **与 README 的差异**：README 写作 `+lsx,+lasx`，但配置文件实际只有
  > `+lasx`。（LLVM LoongArch 子目标特性模型中 `lasx` 蕴含 `lsx`，因此该配置
  > 可正常编译；此为工具链语义，非源码自述。）

- 构建 / 测试：

  ```bash
  cargo +nightly build --release      # 产物：liblasx_rs.so (cdylib) + rlib
  cargo +nightly test --release       # 需 LoongArch 真机
  ```

- loongarch64 nightly 工具链安装示例（README 给出）：
  `rustup toolchain install nightly-loongarch64-unknown-linux-gnu`。

### 8.2 CI（.github/workflows）

**`ci.yml`**（GitHub Actions，两 job）：

1. `fmt`：ubuntu runner 装 nightly + rustfmt，`cargo fmt --all -- --check`；
2. `cross-check`：ubuntu runner 装 nightly + rust-src +
   loongarch64-unknown-linux-gnu target，`cargo check -Zbuild-std --target
   loongarch64-unknown-linux-gnu`——**用 -Zbuild-std 从源码构建 loongarch64
   std**（官方 nightly 分发不稳定/网络受限时可靠），cargo check 无需链接器，
   验证源码在 loongarch64 nightly 下可编译。文件末尾注释明确：**真机测试不在
   GitHub 执行**（官方 runner 无 loongarch64 二进制），由自建 Git Panel CI
   （http://192.168.1.29，龙芯真机，push 自动触发）覆盖。

**`git-panel-ci.yml`**（名为 "Git Panel CI"，4 步）：

1. `actions/checkout@v5`；
2. Format check（nightly rustfmt）；
3. `cargo +nightly build --release`；
4. `cargo +nightly test --release`；
5. `cargo +nightly clippy --all-targets 2>/dev/null || echo "clippy n/a"`。

> 注意两点（源码事实）：
> - 该文件 `runs-on: ubuntu-latest`，与 `ci.yml` 注释所述"龙芯真机"表述存在
>   张力——按其语义，这是部署到自建 Git Panel 的 runner 上执行的 workflow
>   （真机实际执行由 Git Panel 平台调度）；
> - clippy 步骤把 stderr 丢弃（`2>/dev/null`）且失败时回退 `echo "clippy n/a"`
>   （退出码 0）——注释写"zero warnings guard"，但实现上 clippy 失败**不会**
>   让步骤失败，实为"尽力而为"而非硬门禁。

### 8.3 集成清单

- Rust 依赖：加 `lasx_rs = { path = "..." }`（rlib）；
- 动态库：`cargo build --release` 后把 `liblasx_rs.so` 随应用分发（cdylib）；
- 真机验证：3B6000（LASX+LSX）与 3A5000/3A6000（LSX-only）各跑一遍
  `cargo test`；LSX-only 机上 **LASX-only 内核（§2.4/2.5/2.7/2.8/2.2/2.3/2.9）
  不可调用**。

---

## 9. Caveats 与限制

以下条目全部来自源码注释/结构推导，逐条对应 `src/lib.rs`。

### 9.1 降级覆盖不全（LASX-only 内核）

`lasx_matmul`、`lasx_axpy`、`lasx_sum`、`lasx_dot_f64`、`lasx_matmul_f64`、
`lasx_dot_i8`、`lasx_dot_q4` **没有 `has_lasx()` 检查**，直接执行 `lasx_xv*`
指令。在 LSX-only CPU（3A5000/3A6000）上调用会产生**非法指令**（SIGILL），
不会自动降级。使用方必须自行按 CPU 能力调度（只在 `lasx_force_lsx_thread`/
cpucfg 确认有 LASX 时调用，或改用 LSX 实现）。

### 9.2 仅标量降级的内核

`lasx_ballistic_step`、`lasx_batch_distance2d`、`lasx_rk4_j2_step_batch` 的非
LASX 分支是**全标量循环**，没有 LSX 128 位向量路径。

### 9.3 `lasx_ballistic_step` 向量/标量一致性（已修复）

早期版本向量路径阻力符号、重力项、位置更新所取速度均与标量 `euler_step`
不一致（见 §4.1 对照表），且无测试守护。**现已修复**：三处均对齐标量参考，
由 `test_ballistic_step_batch_matches_scalar` 守护（速度 <1e-3、位置 <1e-2
绝对误差）。残余差异仅 `|v|²` 结合序（向量右结合 vs 标量左结合），物理等价、
数值差异极小，非逐位一致。

### 9.4 `lasx_alloc` 无对等释放导出

库内分配、库外释放：`lasx_alloc` 返回未初始化裸指针，没有 `lasx_free`；释放
必须由调用方用 `Vec::from_raw_parts(ptr, n, n)` 归还（长度与容量都要对），
否则泄漏或 UB。

### 9.5 长度参数无负数/零防护

所有内核 `n`（或 `m/k/n`、`n_bytes`）为 `i32` 后立即 `as usize`。负数会变成
巨大 usize，`from_raw_parts` 长度超界 → **UB**；`lasx_alloc` 的
`Vec::with_capacity` 对负 `n` 会 **panic**。调用方必须保证 n ≥ 0 且与缓冲区
实际容量一致。

### 9.6 `lasx_matmul*` 的 O(n·k) 转置临时分配

两个矩阵乘内核都先分配 `vec![0f32/f64; n*k]` 存放 B 的转置（`b_t[j*k+p] =
b[p*n+j]`）。大矩阵时峰值内存 = m·k + k·n + n·k（临时）+ m·n（输出）。

### 9.7 矩阵乘无 f64 落盘（f32 版本精度）

`lasx_matmul` 在 f32 向量累加器内直接累积，无 `lasx_dot` 式的 f64 落盘，
长 k 维累加误差大于点积内核；高精度场景用 `lasx_matmul_f64`。

### 9.8 `lasx_axpy` 向量/标量尾至多差 1 ulp

向量路径 FMA 单舍入，标量尾 `y[j] += alpha * x[j]` 两舍入；无测试守护。对
需要逐位一致的长流水线应用，需自行注意。

### 9.9 `lasx_dot_q4` scale 数组长度约定

`sa`/`sb` 长度必须 ≥ `ceil(n_bytes/32)`（源码 `n.div_ceil(32)`），按 32 字节
组对齐；`n_bytes` 非 32 倍数时最后不足组按同样 `b/32` 索引取 scale（尾循环
同样按 `j/32`）。

### 9.10 cpucfg 依赖

LASX 检测依赖 `cpucfg` 指令 + `CFG2.bit7`。在虚拟化/模拟器中若 cpucfg 行为
不一致或返回不可信，`HW_HAS_LASX` 首次探测结果会被**进程级缓存**（OnceLock），
之后不可更新。

### 9.11 检测缓存与钩子的作用域

`HW_HAS_LASX` 进程级缓存一次；`FORCE_LSX` 线程级、默认 false，测试必须自己
置回。两者语义不同，混用（如跨线程切换钩子）不会互相干扰。

### 9.12 `lasx_dot_i8` 结果截断

内部用 i64 累加精确，但返回 `s_acc as i32`——总点积超出 i32 范围时**回绕**。
约当 n > (2^31-1)/127² ≈ 13.3 万（全 127 输入时）即可能溢出；实际应保证
业务上 n 远小于该界。

### 9.13 对齐与内存布局

所有向量加载/存储用 `xvld`/`xvst`/`vld`/`vst` 偏移 0，LoongArch 允许非对齐
访问，源码**未显式声明对齐要求**；但 SOA/数组布局假设连续、无填充，跨语言
调用方须保证数组元素类型与步长（f32=4 字节、f64=8 字节）严格匹配。

### 9.14 精度边界总结

- f32 点积/求和：每 64 元素落盘 f64 累加（抑制误差），最终一次 f64→f32 舍入；
- f64 内核：全程 IEEE 双精度，FMA 单舍入；
- 量化内核：整数部分精确，误差来自 f32 scale 表示（相对 ~2^-24）；
- 跨实现对照（不同公式）：相对误差 <1e-9；
- 逐样本批量内核的向量块与标量尾：同式同结合序，分块内逐位一致（除 §9.8
  指出的例外）；`lasx_ballistic_step` 因结合序不同非逐位一致，但有物理一致性
  测试守护（§9.3）。

---

## 10. API 索引

### 10.1 导出的 FFI 内核（`extern "C"` + `#[unsafe(no_mangle)]`，cdylib 导出）

| 符号 | C 签名 | 说明 |
|---|---|---|
| `lasx_dot` | `float lasx_dot(const float*, const float*, int)` | f32 点积；LASX+LSX 双路径；64 元素分块 f64 落盘 |
| `lasx_matmul` | `void lasx_matmul(int m, int k, int n, const float*, const float*, float*)` | f32 矩阵乘；B 转置；LASX-only |
| `lasx_axpy` | `void lasx_axpy(float, const float*, float*, int)` | `y += a·x`；FMA；LASX-only |
| `lasx_sum` | `float lasx_sum(const float*, int)` | f32 归约；64 元素分块 f64 落盘；LASX-only |
| `lasx_dot_f64` | `double lasx_dot_f64(const double*, const double*, int)` | f64 点积（4×f64）；LASX-only |
| `lasx_alloc` | `float* lasx_alloc(int)` | 分配未初始化 f32 缓冲（需外部释放） |
| `lasx_dot_i8` | `int lasx_dot_i8(const int8_t*, const int8_t*, int)` | int8 点积（32×i8，i64 防溢出）；LASX-only |
| `lasx_dot_q4` | `double lasx_dot_q4(const uint8_t*, const float*, const uint8_t*, const float*, int)` | Q4 量化点积（每 32 字节一组 scale）；LASX-only |
| `lasx_matmul_f64` | `void lasx_matmul_f64(int m, int k, int n, const double*, const double*, double*)` | f64 矩阵乘（4×f64）；B 转置；LASX-only |
| `lasx_ballistic_step` | `void lasx_ballistic_step(float*, float*, float*, float*, float*, float*, const float*, int, float, float)` | 批量弹道欧拉步（8 发并行）；LASX + 标量降级；与标量参考物理一致（测试守护，§4.1） |
| `lasx_batch_distance2d` | `void lasx_batch_distance2d(float, float, const float*, const float*, float*, int)` | 批量 2D 距离（8 点并行）；LASX + 标量降级 |
| `lasx_norm3_batch` | `void lasx_norm3_batch(const double*, const double*, const double*, double*, int)` | 批量 3 分量模长（f64 SOA）；LASX+LSX 双路径 |
| `lasx_vec3_add_scaled_batch` | `void lasx_vec3_add_scaled_batch(const double*, const double*, const double*, const double*, const double*, const double*, double, double*, double*, double*, int)` | 批量 `o = a + s·b`；LASX+LSX 双路径 |
| `lasx_j2_accel_batch` | `void lasx_j2_accel_batch(const double*, const double*, const double*, double, double, double, double*, double*, double*, int)` | 批量 J2 引力加速度；LASX+LSX 双路径 |
| `lasx_rk4_j2_step_batch` | `void lasx_rk4_j2_step_batch(double*, double*, double*, double*, double*, double*, double, double, double, double, int)` | 批量 RK4 J2 步（4 样本并行，寄存器内完成）；LASX + 标量降级 |

### 10.2 Rust 侧公开辅助

| 符号 | 签名 | 说明 |
|---|---|---|
| `lasx_force_lsx_thread` | `pub fn lasx_force_lsx_thread(force: bool)` | 线程级强制 LSX 降级钩子（仅 rlib 可见，非 FFI） |

### 10.3 内部辅助（私有，文档用途）

| 符号 | 说明 |
|---|---|
| `ld_f32` / `st_f32` / `ld_f64` / `st_f64` | LASX 256 位加载/存储（`xvld/xvst` 偏移 0） |
| `zero_f32` / `zero_f64` | LASX 置零（`xvldi(0)`） |
| `splat_f32` / `splat_f64` | LASX 广播（`xvreplgr2vr_w/d`） |
| `ld4_f32` / `st4_f32` / `zero4_f32` / `ld2_f64` / `st2_f64` | LSX 128 位加载/存储/置零 |
| `splat2_f64` | LSX f64 广播 |
| `has_lasx` | 硬件 LASX 检测（cpucfg CFG2.bit7）+ 线程级 FORCE_LSX 与操作 |
| `euler_step` | 弹道标量欧拉步（LSX/标量降级 + 向量路径尾循环） |
| `j2_accel_vec` | J2 加速度 lane 版本（RK4 向量路径用） |
| `rk4_j2_step_scalar` | RK4 J2 标量步（非 LASX 兜底 + 向量路径尾循环；FMA 版） |
| `mod batch_tests` | 6 个测试（见 §7） |

---

*本手册由源码逐行核对生成；如源码变更，请同步更新本文档。*
