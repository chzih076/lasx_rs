# lasx_rs

**lasx_rs 是为龙芯平台推出的向量库，支持调用龙芯架构里面的 LASX（256 位）与 LSX（128 位）向量指令集。**

批量数值内核（点积/缩放/范数/距离/引力加速度等），无 LASX CPU（3A5000/3A6000 等 LSX-only）
自动降级 LSX 路径，逐位确定。

## 特性

- **LASX 256 位**：`lasx_*` intrinsics 批量内核（axpy / dot / norm3 / 距离 / J2 加速度 / RK4 步）
- **批量姿态/几何（7 个）**：叉积、单位化、3×3·向量、四元数单位化/乘法/旋转/DCM——
  与 loong-sci `attitude`/`orbit` 的标量实现逐位一致，实测比标量快 **2.7–6.4×**
- **LSX 128 位降级**：LSX-only CPU 自动走 128 位路径（线程级强制降级钩子用于验证/测试）
- **Rust 原生 API**（`lasx_rs::api`）：`&[T]` 进出、形状不符返回 `Err`、输出是 32 字节
  对齐的 `AlignedVec`；与 C ABI 走同一段内核，**没有可测开销**
- **零依赖**：纯 std + `stdarch_loongarch`（nightly）
- **逐位确定**：向量化与标量结果一致（防回归测试守护）
- **可选常驻线程池 + 多核调用策略层**（`lasx_rs::pool` / `lasx_rs::parallel`，Rust API）：
  多核铺开批量内核、线程跨调用复用；小数据自动原地串行。矩阵乘 12 线程约 **5.4×**、
  16 线程约 **8.5×**。**不新增任何 C ABI 符号**（走 `rlib`）

## 仓库

| 平台 | 地址 |
|---|---|
| GitCode（主） | `git@gitcode.com:H076lik/lasx_rs.git` / <https://gitcode.com/H076lik/lasx_rs> |
| GitHub（镜像） | `git@github.com:chzih076/lasx_rs.git` / <https://github.com/chzih076/lasx_rs> |

## 代码结构

按"算法 / ABI / 工具"三层切分，算子一个文件，方便定位与增量添加：

```
lasx_rs/
├── Cargo.toml            # workspace 根 + lasx_rs 库（cdylib + rlib）
├── src/
│   ├── lib.rs            # crate 根：模块声明 + 历史 API 重导出
│   ├── arch/             # 架构抽象层
│   │   ├── mod.rs        #   SimdPath enum：能力探测 + 路径分派
│   │   ├── lasx.rs       #   LASX 256 位 load/store/splat 样板
│   │   └── lsx.rs        #   LSX 128 位样板
│   ├── api.rs            # Rust 原生 API：切片进出 + Result + 对齐输出（不导出符号）
│   ├── pool.rs           # 可选常驻线程池（WorkerPool）：纯 Rust API，不导出符号
│   ├── parallel.rs       # 把内核铺到池上的多核调用策略层（matmul / rk4 步）
│   ├── ops/              # 算子层：一个内核一个文件（含各自的 #[cfg(test)]）
│   │   ├── dot.rs  sum.rs  axpy.rs  dot_f64.rs  dot_i8.rs  dot_q4.rs
│   │   ├── matmul.rs  matmul_f64.rs
│   │   ├── norm3_batch.rs  vec3_add_scaled_batch.rs  batch_distance2d.rs
│   │   ├── ballistic_step.rs  j2_accel_batch.rs  rk4_j2_step_batch.rs
│   │   └── testutil.rs   #   测试共用夹具（仅 cfg(test)）
│   └── ffi/              # C ABI 导出层：只做裸指针 → 切片，不含计算
│       ├── reduce.rs  matmul.rs  quant.rs  batch.rs  physics.rs  memory.rs
├── examples/             # 调用方示例：pool_axpy.rs、matmul_pooled.rs
├── cli/                  # 独立 crate `lasx_bench`：性能基准 CLI
│   └── src/{main,group,timing,report,data,scalar_ref,suites/*}.rs
└── yll/                  # 独立 crate `lasx_yll`：YouLiLong 原生扩展（产出 liblasx.so）
    └── src/{lib,yll,convert,funcs}.rs + lib.ylh / youli.yaml / test_lasx.yli
```

分层约定：

- **算子只接收切片**（`&[T]` / `&mut [T]`），裸指针解引用与长度约定收敛在 `ffi` 一层；
- **两套 ABI**：22 个 `lasx_*` 零校验（误用即 UB）；另有 21 个 `lasx_*_checked`
  带 `int *status` 出参，校验失败写入错误码并返回安全中性值——需要把错误上抛给上层时用它；
- 支持降级的算子在入口用 `match SimdPath::detect()` 分派到 `<name>_lasx` / `<name>_lsx`；
- LASX-only 的算子在模块文档里显式标注；
- 历史 15 个 C ABI 符号名与语义保持不变（`lasx_rs::lasx_dot(..)` 继续可用）；
  姿态/几何批处理是**新增**的 7 个符号，不改动任何一个旧符号。

## 构建

**依赖 nightly Rust（含 `stdarch_loongarch` 实验特性）**：

```bash
# .cargo/config.toml 已配置：rustflags = ["-C", "target-feature=+lasx"]
cargo build --release          # 构建库（workspace 默认成员）
cargo test --release           # 单元测试 + 文档测试
cargo build --workspace --release   # 连基准 CLI 一起构建
```

> **注意**：本库依赖**实验版本 rustc**（nightly + `#![feature(stdarch_loongarch)]`），
> 需 loongarch64 nightly 工具链（如 `rustup toolchain install nightly-loongarch64-unknown-linux-gnu`）。

## 用法

**Rust（推荐）**：切片进、`Result` 出、需要输出的内核直接返回 32 字节对齐的缓冲。

```rust
use lasx_rs::api;

let a = [1.0f32, 2.0, 3.0, 4.0];
let b = [1.0f32; 4];

let d = api::dot(&a, &b)?;                       // Result<f32, api::Error>
let s = api::sum(&a);                            // 不会失败，直接返回 f32
let c = api::matmul(2, 2, 2, &a, &b)?;           // Result<AlignedVec<f32>, _>
let out = api::norm3_batch(&xs, &ys, &zs)?;      // 输出已对齐，无需自己 malloc

// 原地内核取 &mut
api::axpy(2.0, &x, &mut y)?;
api::rk4_j2_step_batch(&mut rx, &mut ry, &mut rz, &mut vx, &mut vy, &mut vz, mu, j2, re, dt)?;
```

形状不符是 `Err`（带算子名、参数名、期望值与实际值，实现了 `std::error::Error`），
**不是 UB**：

```rust
assert_eq!(
    api::dot(&a, &b[..2]).unwrap_err().to_string(),
    "dot: 参数 b 的长度应为 4，实际 2"
);
```

**C / Dart 等 FFI 调用方**用裸指针版本（零校验，误用即 UB，但也是最快的）：

```rust
// crate 根的历史重导出与显式路径等价
let out = lasx_rs::lasx_dot(a.as_ptr(), b.as_ptr(), n);
let out = lasx_rs::ffi::reduce::lasx_dot(a.as_ptr(), b.as_ptr(), n);
```

```dart
final lib = DynamicLibrary.open('liblasx_rs.so');
// 绑定 lasx_dot / lasx_axpy / lasx_norm3_batch 等
```

**要多核**时用库内常驻池（只走 `rlib`，不经过 C ABI）：

```rust
use lasx_rs::aligned::AlignedVec;
use lasx_rs::pool::WorkerPool;

let mut pool = WorkerPool::new(12);        // 建一次，跨调用/跨步复用
// 常见负载的切分形状已经固定好了
lasx_rs::parallel::matmul_f32(&mut pool, m, k, n, a.as_mut_slice(), b.as_slice(), c.as_mut_slice());
for _ in 0..steps {
    lasx_rs::parallel::rk4_j2_step_batch(&mut pool, mu, j2, re, dt, rx, ry, rz, vx, vy, vz);
}
```

完整可运行版本：`cargo run --release --example pool_axpy`、
`cargo run --release --example matmul_pooled`（都自带与单线程的逐位对照）。

## YouLiLong 原生扩展

`yll/` 是把本库接进 [YouLiLong](../../) 语言的原生扩展，构建后产出 `liblasx.so`：

```bash
cargo build --release -p lasx_yll
cp target/release/liblasx.so yll/
<YouLiLong>/target/release/youli_long yll/test_lasx.yli   # 端到端测试
```

```youlilong
use "./lasx"

a = [1.0, 2.0, 3.0, 4.0]
print(lasx.dot(a, [1.0, 1.0, 1.0, 1.0]))   // 10
print(lasx.norm3([3.0], [4.0], [0.0]))     // [5]

// 多星多步传播：多核 + 常驻池，数组只转换一次（比脚本里循环 call 单步快 110×）
final = lasx.propagate(rx, ry, rz, vx, vy, vz, mu, j2, re, dt, 200)

try {
    lasx.dot(a, [1.0])                     // 长度不一致
} catch (e) {
    print(e)   // 无效操作: a 与 b 长度不一致：4 vs 1
}
```

**所有调用错误都上抛**：参数不是数组、元素不是数值、长度/形状不自洽、int8 越界、
物理常数非法——一律返回 `yll_error(...)`，解释器抛成可 `try/catch` 的运行时错误，
不静默返回 0。详见 [yll/README.md](yll/README.md)。

## 性能基准

基准是独立的 CLI crate，零外部依赖：

```bash
cargo run -p lasx_bench --release              # 全部内核，约 2 分钟
cargo run -p lasx_bench --release -- matmul    # 只跑组名含 matmul 的套件
cargo run -p lasx_bench --release -- fma       # 纯寄存器 FMA 吞吐
cargo run -p lasx_bench --release -- scenario  # 真实调用场景（多步传播/高频小调用/融合 vs 拼接）
```

> **调用策略本身影响很大**（基准同时也是示例，`-- scenario` 给出实测）：
> 多步传播时**复用常驻线程池**比每步新建线程端到端快 **2.1–2.3×**；
> **用融合内核**（`lasx_rk4_j2_step_batch`）比用原语拼同一个 RK4 步快 **2.8×**；
> 小数组高频调用每次有 10–25 ns 的固定开销；**别把分配/克隆放进热路径**；
> 脚本侧循环调用批量内核时，**每次调用的数据搬运**（而非内核）才是瓶颈——
> YouLiLong 扩展里 `propagate` 比脚本循环快 **约 100×**（见 [yll/README](yll/README.md)）。

三种口径：**LASX**（原生 256 位）/ **强制 LSX**（线程级降级钩子）/ **标量**基线。

实测结论（Loongson-3B6000 / LA664，优化后）：

| 内核 | 规模 | LASX 计时 | 相对标量 |
|---|---|---|---|
| `lasx_dot` | n=4096 | 316 ns | 17.6× |
| `lasx_sum` | n=64 Ki | 6.1 µs | 14.8× |
| `lasx_dot_i8` | n=64 Ki | 7.9 µs | 1.35× |
| `lasx_matmul` f32 | 64³ | **7.6 µs** | 4.25× |
| `lasx_matmul` f32 | 128³ | **62.3 µs** | 3.57× |
| `lasx_matmul` f32 | 256³ | **545.5 µs** | 4.49× |
| `lasx_matmul` f32 | 512³ | **4.91 ms** | 3.96× |
| `lasx_matmul_f64` | 128³ | **134 µs** | 4.65× |

硬件 FMA 峰值实测 LASX 93.2 / LSX 46.5 GFLOP/s（2.00×）；f32 matmul 从 64³ 到 512³
**稳定在 54.6–68.9 GFLOP/s**（峰值的 59–74%），靠的是"列块在外 + 打包 B 面板"
（Goto/BLIS 的 packing，见 perf-report §19）——256³ 以上此前会掉到 ~30。

> **对齐建议**：LASX 是 32 字节访存，调用方缓冲区若 32 字节对齐可再快约 1.1–1.56×
> （L1 驻留规模上最明显）。**不要假设缓冲区天然对齐**——实测 glibc `malloc` 与
> Dart FFI 的缓冲区只有约一半落在 32 字节边界上；用 `posix_memalign(&p, 32, n)` /
> C11 `aligned_alloc(32, n)`，或本库的 `lasx_alloc`（已保证 32 字节对齐）。
> 代码片段见 [manual.md §5.5](docs/manual.md)，量化对比见
> `cargo run -p lasx_bench --release -- align`。
完整数据、根因分析与优化前后对比见 **[docs/perf-report.md](docs/perf-report.md)**。

## 文档

- **[docs/manual.md](docs/manual.md)**：完整技术手册（中文）——架构与设计、
  22 个 FFI 内核逐一详解、量化内核、批量物理内核、批量姿态/几何内核、FFI 使用指南（C/Dart/Rust）、
  性能基准方法、测试与验证、构建与集成、Caveats 与限制、API 索引；
- **[docs/perf-report.md](docs/perf-report.md)**：性能实测报告与优化记录；
- **[docs/nn-ops.md](docs/nn-ops.md)**：神经网络算子分析（llama.cpp/ggml 的 102 个算子、
  量化块布局、**本机 LASX 实测**缺口清单：f16 GEMM 11× / softmax 7× / f16 GEMV 3.7×）；
- **[docs/compression.md](docs/compression.md)**：无损压缩与「带宽换算力」的可行性
  测量（结论：快照 1.22×、轨迹 1.40×，且解码速率 0.37 GB/s 远低于内核消费
  3.3 GB/s，**不进内核热路径**）与后续路线；
- **[docs/README.md](docs/README.md)**：文档目录索引。

## 许可

MIT © 2026 lik（H076lik）。详情见 [LICENSE](LICENSE)。
