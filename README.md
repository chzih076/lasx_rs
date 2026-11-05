# lasx_rs

面向 **LoongArch64（龙芯 3B6000 / LA664，LASX 256 位）** 的批量数学内核库：**零第三方依赖**、
Rust 实现、导出稳定的 C ABI，并提供安全 Rust API、常驻工作池与形状/计划层。

## 能力概览

| 类别 | 内容 |
|---|---|
| 归约与稠密线性代数 | `dot`、`sum`、`axpy`、`matmul`（f32 / f64）、打包计划句柄 |
| 量化与 int8 | GGUF 风格 Q4、int8 量化/反量化生产端、`dot_i8`、`gemv_i8`、`matmul_i8` |
| NN 算子 | 行内 softmax / RMSNorm / LayerNorm、SiLU、GELU（sigmoid 近似 / erf 形式）、RoPE、f16 权重 GEMV |
| 批量几何与物理 | 3/2 分量模长、四元数族、J2 引力加速度、RK4 步、弹道、2D 距离、复数矩阵乘 |

- **位精确是硬承诺**：同一算子的 LASX / LSX / 标量尾 / 打包 / 分块 / 并行切块等全部路径，对同一输入给出
  **逐位相同**结果（测试用 `to_bits()` 比对）；例外仅登记过的容差类（见 `docs/ops.md`）。
- **98 个导出符号**（裸版本 + `_checked` 长度校验版本）；最初的 15 个 `lasx_*` 符号**签名与语义永不改动**。
- **矩阵乘**：打包 + k 分块 + 列块/流式三条逐位一致路径；f32 512³ 单线程约 **61.5 GFLOP/s**（该机微内核上限的 **88%**），
  64³ 到 **95%**；与 OpenBLAS 的对照见 `docs/dev.md` §4.4。
- **常驻工作池**：跨调用复用线程（不每次建/拆）；兼容调用方线程驱动。
- **接口层**：`view::MatRef/MatMut`（形状/跨距构造时校验）、`plan::MatmulPlan`（打包一次、反复使用、
  可 `Arc` 共享）、`shape`（形状进类型、编译期校验）、`lasx_rs_macros` 公式 DSL（`matmul!`/`dot!`/`gemv!`）。

## 平台要求（用之前先看）

| 要求 | 说明 |
|---|---|
| 只在 `loongarch64` 上编译 | 其它架构**编译期**就失败；依赖方请写 `[target.'cfg(target_arch = "loongarch64")'.dependencies]`（有意不给 x86 标量兜底） |
| 需要 nightly | 唯一 feature 门 `#![feature(stdarch_loongarch)]`；稳定后删两行即可上 stable |
| `+lasx` 由使用方加 | 仓库 `.cargo/config.toml` 设了 `-C target-feature=+lasx`，但**依赖方的 cargo 不读依赖自带的 config**；不带也能编、也逐位正确，想拿满性能请自行加（缺标志约慢 26–46×） |
| LA664 勘误 | 3A6000/3C6000/3B6000 的原子操作丢失更新会静默错结果；本库生产路径已全部改用带屏障的序 + 回归测试守着（`docs/platform.md`） |
| NN 侧算子按目标平台取舍 | 目标平台为支持 LASX 的 3A6000/3C6000/3B6000；**只有 LSX 的机型（如 3B6000M / LA364E）执行 LASX 指令会 SIGILL** |

## 快速开始

```bash
# 需要 nightly + LoongArch 真机
cargo build --release                     # liblasx_rs.so + rlib
cargo test --release                      # 单元测试 + 宏测试 + 文档测试
cargo clippy --all-targets -- -D warnings # 零警告是硬门槛
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p lasx_rs

# 基准
cargo run -p lasx_bench --release

# C 项目：安装后的编译/链接参数
pkg-config --cflags --libs lasx_rs
```

```rust
use lasx_rs::api;                     // 安全切片接口
let mut y = vec![0.0f32; 4 * 4];
api::matmul_into(4, 4, 4, &a, &b, &mut y)?;   // 行主序 A[m×k]·B[k×n]
let s = api::dot(&x, &w)?;            // 点积

// 公式 DSL：形状进类型，编译期对齐
use lasx_rs::matmul;
matmul!(y[M, N] = x[M, K] * w[K, N]);
```

## 文档索引

| 文档 | 内容 |
|---|---|
| `docs/ops.md` | **算子与用法手册**：入口、数值契约、边界、int8/f32 判据 |
| `docs/dev.md` | **设计/决策**：架构与分层、内核、性能方法学与数字、精度承诺、接口冻结 |
| `docs/c-abi.md` | **C ABI 参考**：符号表、签名约定、状态码、内存与计划句柄、示例 |
| `docs/platform.md` | **平台约束**：LA664 勘误、平台支持状态、ONNX 契约缺口与承诺边界 |

## 与 DSL 仓（`simd_dsl`）的关系

- 本仓**已纯化：零 DSL 依赖**（`dsl-backend` feature、`simd_dsl`/`ops_dsl`/`attention_block` 依赖与 DSL 示例均已移出）。
- **依赖方向**：DSL 仓的适配器 `backends/lasx` 在 `Cargo.toml` 里钉本仓 git rev；升级本仓 = DSL 仓 bump rev 并重跑其门禁/A-B。
- 引擎选路、融合 plan、能力位口径属于 DSL 仓契约：见（DSL 仓）`docs/engine_integration.md`、`docs/backend_guide.md`。
- 本仓只提供内核 + 诚实的性能声明；「引擎怎么用这些内核」不在本仓范围。
