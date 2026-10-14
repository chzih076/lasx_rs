# lasx_rs

面向 **LoongArch64（龙芯 3B6000 / LA664，LASX 256 位）** 的批量数学内核库：零第三方依赖、
Rust 实现、导出稳定的 C ABI，并提供安全 Rust API、常驻工作池与形状/计划层。

覆盖四类负载：

| 类别 | 内容 |
|---|---|
| 归约与稠密线性代数 | `dot`、`sum`、`axpy`、`matmul`（f32 / f64） |
| 量化点积 | int8、GGUF 风格 Q4 |
| int8 推理（N3 批次） | 量化/反量化生产端（`amax`、per-tensor/per-row `quantize_i8`、`dequantize_i8`）+ `gemv_i8`/`matmul_i8`。**实测见下**：批量形态下 f32 GEMM 反而更快，判据在 [docs/ops.md](docs/ops.md) §5.15 |
| 批量物理与姿态/几何 | J2 引力加速度、RK4 步、弹道、3 分量模长、四元数旋转/DCM 等 |
| NN 侧算子（N1 批次） | 行内 softmax、RMSNorm、SiLU、GELU（sigmoid 近似 / erf 形式）、RoPE、f16 权重 GEMV |

## 特点

- **零第三方依赖**：仅 `core::arch::loongarch64` 的 LASX/LSX intrinsic 与 std。
- **81 个导出符号**（41 个裸版本 + 40 个带 `int *status` 的 `_checked` 版本）。
  最初的 15 个 `lasx_*` 符号签名与语义**永不改动**，新增能力一律追加；
  另有 `*_checked` 变体提供结构化错误码，两者走同一条内核实现。
- **位精确**：同一算子的 LASX / LSX / 标量尾 / 打包 / k 分块 / 并行切块等全部路径，对同一输入
  给出**逐位相同**的结果（测试用 `to_bits()` 比对）。例外仅两处且均有测试与文档：
  `lasx_ballistic_step` 的向量/标量分支结合次序不同，`lasx_axpy` 允许 1 ulp 以内差异。
- **矩阵乘**：打包 + k 分块 + 列块/流式三条逐位一致的路径，f32 512³ 单线程 60.9 GFLOP/s，
  为该机器微内核上限（含 A 标量广播时 2.00 FMA/周期 = 70.3 GFLOP/s）的 **87%**；
  64³（不打包）到 **97%**。同日交叉对照 OpenBLAS 0.3.34 单线程见 `docs/dev.md` §8.5。
- **常驻工作池**：跨调用复用线程，n = 2^18 单步 12 线程加速 **8.15×**；多星多步传播场景
  端到端比"每步新建线程"快 **2.55×**、比单线程快 **8.09×**。
- **接口层**：`view::MatRef`/`MatMut` 把形状与跨距变成构造时校验一次的对象；
  `plan::MatmulPlan` 把 `B` 打包一次反复使用（只读、可 `Arc` 共享）；
  `shape` 层把形状放进类型并在编译期校验；`lasx_rs_macros` 提供公式 DSL
  （`matmul!` 矩阵乘、`dot!` 点积、`gemv!` 矩阵-向量）。
- **NN 侧算子按目标平台取舍**：N1 批次**不提供 LSX 降级**（目标平台 6000 系列均支持 LASX），
  在无 LASX 的 CPU 上执行 LASX 指令会产生 SIGILL，使用时需自行按 CPU 能力调度。

## 平台要求（用之前先看这一段）

| 要求 | 说明 |
|---|---|
| **只在 `loongarch64` 上编译** | 其它架构**编译期**就过不去（`src/` 里 0 处 `target_arch` 门控）。依赖方请写成 `[target.'cfg(target_arch = "loongarch64")'.dependencies]`，否则你的 x86 CI 会直接红（Cargo 为所有目标解析依赖、不按平台跳过）。这是**有意**的：不给 x86 加标量兜底，避免把"用错平台"从编译期错误退化成运行期错结果 |
| **需要 nightly** | 唯一 feature 门是 `#![feature(stdarch_loongarch)]`（跟踪 [issue #117427](https://github.com/rust-lang/rust/issues/117427)）：stable 上 `std::arch::loongarch64` 仍是 unstable（实测 stable 1.98.1 与 nightly 1.100.0 都仍要求该门）。**等它稳定后**删掉那两行 `#![feature(...)]` 就能在 stable 上编译 |
| **`+lasx` 只影响性能** | 仓库 `.cargo/config.toml` 里有 `-C target-feature=+lasx`，但**依赖方的 cargo 不会读依赖自带的 config**。实测**不带它也能编、也位精确**；想拿满性能请自行加 |
| **LA664 勘误** | 3A6000/3C6000/3B6000 的原子操作丢失更新会**静默错结果**；本库生产路径已全部用带屏障的序，并有"同输入跑两次逐位一致"的回归测试守着（见 [docs/platform.md](docs/platform.md) §1） |

## 快速开始

```bash
# 需要 nightly（#![feature(stdarch_loongarch)]）与 LoongArch 真机
cargo build --release                    # 产出 liblasx_rs.so + rlib
cargo test --workspace --release         # 218 库单测 + 5 宏单测 + 17 文档测试（另 2 个标 ignore）
cargo clippy --workspace --all-targets -- -D warnings   # 零警告是硬门槛
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p lasx_rs

# 基准：不带过滤跑全部套件，带子串只跑匹配套件
cargo run -p lasx_bench --release
cargo run -p lasx_bench --release -- softmax

# 单形状 A/B（进程级交替测量的入口）：<m> <k> <n> <stream|packed|cols|packed64|stream64|pool> [线程数] [策略]
cargo run --release --example matmul_ab -- 512 512 512 packed
```

`.cargo/config.toml` 设置 `rustflags = ["-C", "target-feature=+lasx"]`，因此默认产物假定 CPU
支持 LASX。基准的两条纪律：**单线程内核表绑核测、带线程的表不绑核测**，两者分两次运行
（理由与影响见 `docs/dev.md` §7 开头）。

## 实测性能

> **对外可比的两条结论**（口径、原始数、边界见 `docs/dev.md` §7.10；同机 Loongson-3B6000、
> 同一负载窗口、形状逐行对齐、各 3 轮）：
> **① 单线程 GEMM 不输 OpenBLAS 0.3.34**——64³ **68.6 vs 52.4 GFLOP/s**、128³ **67.2 vs 57.7**
> （小形状领先 15–30%），256³–512³ 打平（±5% 内）；f64 同型。
> **② 多线程：中位数打平，但我们方差大得多**——LLM 形状 `192×768×3072` 12 线程：我们中位数
> **2.64 ms**（样本 2.42–3.79，极差 **1.57×**），OpenBLAS 中位数 ~**2.63 ms**（样本 2.599–2.656，
> 极差 **2%**）⇒ **原先"我们慢约 10%"已撤回**（那是单次采样；两侧分布大面积重叠，我们最好一次
> 2.42 ms 比 OpenBLAS 最好还快）。**真正的差距是方差**（我们的池对负载/状态更敏感）。
> 口径与全部数据见 `docs/dev.md` §7.10 ② 与 §7.11 ⑤（**≥5 轮中位数 + 极差**，勿引用单次值）。
>
> **③ 公式 DSL 的用法提示（实测省 31–34%）**：`matmul!` 生成的 `mul_into` **每次调用都会打包 B**
> （`rhs.prepare().apply_into(...)`）——这是设计语义。**在层循环里反复用同一个权重时，请自己
> hoist 准备阶段**：循环外 `let plan = w.prepare();`，循环里 `plan.apply_into(&x, &mut y);`。
> 实测（256³）：`matmul!` 每次 **143.7 µs** → hoist 后 **96.5–99.1 µs**，与直接池化（97.6–99.8 µs）
> 持平。这类"**准备一次、循环里只用**"的模式在 RoPE 上同样成立（`rope_tables` 一次 + 每层
> `rope_into`，实测省 **89%**，见 `docs/ops.md` §5.12）。
> **别误读标量列**：`+lasx` 下 LLVM 会把标量参考**自动向量化**，所以 matmul 表里的 3.3–4.7× 是
> "手写 intrinsic vs 编译器自动向量化"，**不是"vs 朴素循环"**。

以下为 **2026-09-25** 在同一台 Loongson-3B6000 上的一次连续采集（负载约 2–3；单线程结果
绑核，线程相关结果不绑核）。完整表格、方法学与逐项口径见
**[docs/dev.md §7](docs/dev.md)** —— 该节是全部性能数据的唯一权威位置；编译期常量/阈值的
对照（k 分块、列块宽度、调度策略等）记录在各自的设计小节。

| 项 | 读数 |
|---|---|
| FMA 吞吐上限（16 条独立链，纯寄存器） | **140.2 GFLOP/s**（3.98 条/周期）；LSX 69.3（2.00×） |
| 微内核上限（含 A 标量广播） | **2.00 FMA/周期**（f32 70.3、f64 35.2 GFLOP/s） |
| `lasx_matmul` f32 64³ / 256³ / 512³ | 68.6 / 63.3 / **60.9** GFLOP/s（同日见 `docs/dev.md` §8.5） |
| `lasx_matmul_f64` 128³ / 512³ | 31.3 / 31.4 GFLOP/s（同日见 `docs/dev.md` §8.5） |
| `lasx_matmul`（`parallel`，12 线程） | 128³ / 512³ 加速 5.06× / 5.34×（323.8 / 302.1 GFLOP/s） |
| 单流只读 / 双流只读 / 12 线程只读（32 MiB） | 12.0 / 10.4 / 22.0 GB/s |
| `lasx_softmax_rows` / `lasx_rms_norm` | 5.22–7.00 / 12.01–18.38 GB/s |
| `lasx_silu` / `lasx_gelu_quick` / `lasx_gelu_erf` | 4.74–9.20 / 4.55–8.61 / 4.09–5.51 GB/s |
| `lasx_rope`（NeoX，LASX 路径） | 28.6–71.9 GB/s |
| `lasx_dot_f16` / `lasx_gemv_f16` | 41.8–68.5 / 3.75–48.2 GB/s |
| 常驻池 vs 每步新建线程（跨 200 步复用） | 37.36 ms vs 95.37 ms（**2.55×**，单线程 302.28 ms） |

与同机 OpenBLAS 0.3.34（单线程）的对照、以及各形状的根因分析见 `docs/dev.md` §8.5。

> 性能数据的解读规则：§7 的表是一次连续采集的横截面，**只能同表内比较**；跨运行（换一次
> 采集、换一天）必须重测。实测到的窗口间差异可达 40%，因此"改前跑一遍、改后跑一遍"不能
> 用来判断快慢——那要用进程级交替（`docs/dev.md` §6.4）。

## 算子一览

| 分类 | 符号 |
|---|---|
| 归约/稠密 | `lasx_dot`（f32，带 LSX 降级）、`lasx_sum`、`lasx_axpy`、`lasx_dot_f64`、`lasx_matmul`、`lasx_matmul_f64` |
| 量化 | `lasx_dot_i8`（int8）、`lasx_dot_q4`（每 32 字节一组 scale） |
| int8 推理 N3（8 个） | `lasx_amax`、`lasx_absmax_rows`、`lasx_quantize_i8_per_tensor`、`lasx_quantize_i8_per_row`、`lasx_dequantize_i8`、`lasx_dequantize_i8_rows`、`lasx_gemv_i8`、`lasx_matmul_i8` |
| 批量几何 | `lasx_batch_distance2d`、`lasx_norm3_batch`、`lasx_vec3_add_scaled_batch` |
| 批量物理 | `lasx_j2_accel_batch`、`lasx_rk4_j2_step_batch`、`lasx_ballistic_step` |
| 批量姿态（7 个） | `lasx_cross3_batch`、`lasx_unitize3_batch`、`lasx_mat3_mul_vec3_batch`、`lasx_quat_normalize_batch`、`lasx_quat_mul_batch`、`lasx_quat_rotate_batch`、`lasx_quat_to_dcm_batch` |
| NN 侧 N1（8 个） | `lasx_softmax_rows`、`lasx_rms_norm`、`lasx_silu`、`lasx_gelu_quick`、`lasx_gelu_erf`、`lasx_rope`、`lasx_dot_f16`、`lasx_gemv_f16` |
| 内存 | `lasx_alloc`（32 字节对齐；库内不导出对应的释放函数） |

每个算子的 C 签名、语义、数值契约与"由谁守"见 `docs/ops.md`（§4 符号总表、§2.x 数值契约、
§5.x 用法）。安全 Rust API 见 `lasx_rs::api`（32 个 `pub fn`），另有 `lasx_rs::view` /
`lasx_rs::plan` / `lasx_rs::shape` / `lasx_rs::pool` / `lasx_rs::parallel`。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/ops.md](docs/ops.md) | **算子与用法**：81 个符号总表、数值契约与逐位确定性、降级覆盖、C/Rust/Dart 调用、池与并行、NN 算子现状与缺口对照、int8/f32 判据（§5.15） |
| [docs/dev.md](docs/dev.md) | **架构与性能**：分层与不可破坏的约定、测试与 CI、性能方法学、**全部实测数据（§7）**、矩阵乘与并行深挖、被否掉的方案清单、复现步骤与已知缺口 |
| [docs/platform.md](docs/platform.md) | **平台约束与待补实现**：LA664 原子操作丢失更新（勘误、命中点、措施）与"库侧为什么必须自带 `_db`"、内核分支生命周期、消费 ONNX 契约引出的缺口（LayerNorm/Gather）、承诺边界待划定的项 |

两份文档的分工是硬约定：**契约与用法在 `docs/ops.md`，性能数据在 `docs/dev.md` §7，
同一份数字只存一处**。文档内所有跨文档引用都写成 `docs/xxx.md §N` 形式，可机械校验。

## 仓库

| 远端 | 地址 |
|---|---|
| GitCode（主） | `git@gitcode.com:H076lik/lasx_rs.git` / <https://gitcode.com/H076lik/lasx_rs> |
| GitHub（镜像） | `git@github.com:chzih076/lasx_rs.git` / <https://github.com/chzih076/lasx_rs> |
| 局域网 panel | `http://192.168.1.64/api/git/lasx_rs.git`（CI 走同一个 panel） |

CI 四步：`Format check (rustfmt)` → `Build release (nightly, +lasx)` → `Test release` →
`Clippy (zero warnings guard)`。**CI 不跑基准**：基准受后台负载影响，不适合作为门禁。

## 许可

MIT © 2026 lik（H076lik）。详情见 [LICENSE](LICENSE)。
