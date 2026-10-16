# lasx_rs

面向 **LoongArch64（龙芯 3B6000 / LA664，LASX 256 位）** 的批量数学内核库：零第三方依赖、
Rust 实现、导出稳定的 C ABI，并提供安全 Rust API、常驻工作池与形状/计划层。

覆盖四类负载：

| 类别 | 内容 |
|---|---|
| 归约与稠密线性代数 | `dot`、`sum`、`axpy`、`matmul`（f32 / f64） |
| 量化点积 | int8、GGUF 风格 Q4 |
| int8 推理（N3 批次） | 量化/反量化生产端（`amax`、per-tensor/per-row `quantize_i8`、`dequantize_i8`）+ `gemv_i8`/`matmul_i8`。**实测见下**：批量形态下 f32 GEMM 反而更快，判据见 [docs/ops.md](docs/ops.md) |
| 批量物理与姿态/几何 | J2 引力加速度、RK4 步、弹道、3 分量模长、四元数旋转/DCM 等 |
| NN 侧算子（N1 批次） | 行内 softmax、RMSNorm、SiLU、GELU（sigmoid 近似 / erf 形式）、RoPE、f16 权重 GEMV |

## 特点

- **零第三方依赖**：仅 `core::arch::loongarch64` 的 LASX/LSX intrinsic 与 std。
- **98 个导出符号**（52 个裸版本 + 46 个带 `int *status` 的 `_checked` 版本）。
  最初的 15 个 `lasx_*` 符号签名与语义**永不改动**，新增能力一律追加；
  另有 `*_checked` 变体提供结构化错误码，两者走同一条内核实现。
- **位精确**：同一算子的 LASX / LSX / 标量尾 / 打包 / k 分块 / 并行切块等全部路径，对同一输入
  给出**逐位相同**的结果（测试用 `to_bits()` 比对）。例外仅两处且均有测试与文档：
  `lasx_ballistic_step` 的向量/标量分支结合次序不同，`lasx_axpy` 允许 1 ulp 以内差异。
- **矩阵乘**：打包 + k 分块 + 列块/流式三条逐位一致的路径，f32 512³ 单线程 61.5 GFLOP/s，
  为该机器微内核上限（含 A 标量广播时 2.00 FMA/周期 = 70.3 GFLOP/s）的 **88%**；
  64³ 到 **95%**。同日与 OpenBLAS 0.3.34 单线程的交叉对照见 `docs/dev.md`。
- **常驻工作池**：跨调用复用线程，n = 2^18 单步 12 线程加速 **8.62×**；多星多步传播场景
  端到端比"每步新建线程"快 **2.24×**、比单线程快 **8.08×**。
- **接口层**：`view::MatRef`/`MatMut` 把形状与跨距变成构造时校验一次的对象；
  `plan::MatmulPlan` 把 `B` 打包一次反复使用（只读、可 `Arc` 共享）；
  `shape` 层把形状放进类型并在编译期校验；`lasx_rs_macros` 提供公式 DSL
  （`matmul!` 矩阵乘、`dot!` 点积、`gemv!` 矩阵-向量）。
- **NN 侧算子按目标平台取舍**：N1 批次**不提供 LSX 降级**（目标平台是 **LA664 核的 3A6000 / 3C6000 /
  3B6000，均支持 LASX**；**注意 `3B6000M` 虽同为 6000 命名，但基于 `LA364E`，只有 LSX，不在目标之列**），
  在这类只有 LSX 的机型上执行 LASX 指令会产生 SIGILL，使用时需自行按 CPU 能力调度。

## 平台要求（用之前先看这一段）

| 要求 | 说明 |
|---|---|
| **只在 `loongarch64` 上编译** | 其它架构**编译期**就过不去（`src/` 里 0 处 `target_arch` 门控）。依赖方请写成 `[target.'cfg(target_arch = "loongarch64")'.dependencies]`，否则你的 x86 CI 会直接红（Cargo 为所有目标解析依赖、不按平台跳过）。这是**有意**的：不给 x86 加标量兜底，避免把"用错平台"从编译期错误退化成运行期错结果 |
| **需要 nightly** | 唯一 feature 门是 `#![feature(stdarch_loongarch)]`（跟踪 [issue #117427](https://github.com/rust-lang/rust/issues/117427)）：stable 上 `std::arch::loongarch64` 仍是 unstable（实测 stable 1.98.1 与 nightly 1.100.0 都仍要求该门）。**等它稳定后**删掉那两行 `#![feature(...)]` 就能在 stable 上编译 |
| **`+lasx` 只影响性能** | 仓库 `.cargo/config.toml` 里有 `-C target-feature=+lasx`，但**依赖方的 cargo 不会读依赖自带的 config**。实测**不带它也能编、也位精确**；想拿满性能请自行加 |
| **LA664 勘误** | 3A6000/3C6000/3B6000 的原子操作丢失更新会**静默错结果**；本库生产路径已全部用带屏障的序，并有"同输入跑两次逐位一致"的回归测试守着（见 [docs/platform.md](docs/platform.md)） |

## 快速开始

```bash
# 需要 nightly（#![feature(stdarch_loongarch)]）与 LoongArch 真机
cargo build --release                    # 产出 liblasx_rs.so + rlib
cargo test --workspace --release         # 305 个单元测试 + 8 个宏测试 + 19 个文档测试（另 2 个标 ignore）
cargo clippy --workspace --all-targets -- -D warnings   # 零警告是硬门槛
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p lasx_rs

# C 项目：安装后的编译/链接参数（见 docs/c-abi.md 的「快速开始」）
pkg-config --cflags --libs lasx_rs

# 基准：不带过滤跑全部套件，带子串只跑匹配套件
cargo run -p lasx_bench --release
cargo run -p lasx_bench --release -- softmax

# 单形状 A/B（进程级交替测量的入口）：<m> <k> <n> <stream|packed|cols|packed64|stream64|pool> [线程数] [策略]
cargo run --release --example matmul_ab -- 512 512 512 packed
```

```c
/* C 侧最短路径；完整心智模型（三层 / 裸 vs _checked / 线程 / 错误模型）见 docs/c-abi.md */
if (lasx_has_lasx()) { /* 调 [LASX-only] 符号前先问：0 = 仅 LSX，直调会 SIGILL */ }
float d = lasx_dot_checked(a, b, n, &st);            /* 默认 _checked：失败只写 status、不触碰输出 */
if (st != LASX_OK) fprintf(stderr, "%s\n", lasx_status_message(st));   /* 别硬编码 0..6 */
lasx_free(a, n);                                     /* 与 lasx_alloc 配对；free() 不是 ABI 契约 */
```

`.cargo/config.toml` 设置 `rustflags = ["-C", "target-feature=+lasx"]`，因此默认产物假定 CPU
支持 LASX。基准的两条纪律：**单线程内核表绑核测、带线程的表不绑核测**，两者分两次运行
（理由与影响见 `docs/dev.md`）。

## 实测性能

> **对外可比的两条结论**（权威表、每格 5 轮中位数与毫秒极差、逐轮 `loadavg`、A/B/C 分级、
> **先验一致性**与"旧值 vs 新值处置"都在 `docs/dev.md` §7.10；两边装置已入库
> （`scripts/bench_openblas.c` + `scripts/bench_openblas.sh`）⇒ 可复现）：
> **① 单线程 f32：小形状领先，大形状略落后**——64³ **67.8 vs 53.8 GFLOP/s（+27%）**、
> 128³ **65.8 vs 57.6（+14%）**、256³ 打平（62.4 vs 61.7）、**384³ 60.6 vs 64.0（−5.2%）**、
> **512³ 60.9 vs 64.9（−6.2%）**；宽薄 `256×64×4096` **61.7 vs 41.0（+50%）**。
> **f64 整体落后 4–15%**（64³ 打平、256³/384³/512³ 各 −7.5% / −4.4% / −6.3%，
> **最弱的结构点是 f64 128³：23.7 vs 28.1 G，−15%**）。
> **② 多线程：位精确与扩展性在现有 C ABI 下不可兼得**——`192×768×3072` f32 @12 线程，
> **C ABI 不导出线程池**，只能由调用方切分：
> **要位精确 ⇒ 只能按行切（`rows`）**——**10/10 形状逐位一致**（含 k=384/385/386 两侧与
> m=7/13/100/1000 这类不整除行块），代价是**几乎不扩展**（14.46 ms，**1.08×**）；
> **`cols`（预转置）更快（3.84 ms、4.09×），但它不是位精确等价物**——**6/10 形状与单线程
> 结果不逐位一致**（它把 `Cᵀ=Bᵀ·Aᵀ` 拆成小尺寸调用，尺寸一变库内路径就变）⇒
> **凡要求位精确的场景不要用 `cols`**。
> 根因：`rows` 下每个线程各自重打包整个 `B`（**113 MB/调用**），而散射只占 **9%**——
> 缺的是"**打包一次 + 共享面板**"，这**用现有 C ABI 表达不出来**：C 侧有效档 ≈ **236 G**
> （Rust 池 **310 G**、OpenBLAS 同形状 **2.46 ms / 368 G**）。**真正的 C 侧多核解要由库侧
> 提供"打包一次、可复用"的计划/池入口——正在实现中**，落地后它才是推荐路径。
>
> **跟谁比（必读）**：对手是 **OpenBLAS 0.3.34、`DYNAMIC_ARCH`、core `la464`（LA64 优化内核，
> **非**通用 C 兜底）、`NO_AFFINITY`**；装置每次对照的**第一行**都会打印它的完整身份（`OBIDENT`）。
> **正式口径 = 同进程调用级交替**（2026-10-15 起；理由：SIMD 库的性能就是"同一进程、同一份数据上
> 调它它干多久"，跨进程会把进程启动/建池/调度/他人干扰掺进来）。**新口径下 12 线程已盖章
> ≥300 GFLOP/s**：`plan`（调用方常驻线程 + 动态认领 + 短自旋）**中位 310.99 G（2.9132 ms，极差
> 2.8763–3.0358，A 级窗口）**，对手 OpenBLAS 同轮 **357.22 G**（差 13%）。**旧跨进程数（如 274 G）
> 一律是参考/旧账**，别拿它下结论。
> **调用模型也必须一起看**：上面"打平/只差几个百分点"的数是**"调用方自己驱动（plan + 调用方线程）"
> 这个模型**下的。**库自管线程的便利路径（`lasx_matmul_run_threaded`）现在 ≈1.02–1.05× 手动最优**
> （同进程交替：池缓存进句柄后 **2.7038 ms / 335 G**，每次现建池是 3.0986 ms ⇒ 缓存 **+14.6%**）
> ⇒ **快路径与便利层差距已经很小**。它的**新代价**（务必知道）：**同一 plan 上并发 `run_threaded`
> 会共用同一个池 ⇒ 派活串行**（要真并行请用不同 plan，或 `parallel::*` + 自管池）；**`threads`
> 变化会重建池**。
> **旧数一律不可引用**：早先那批"便利层慢 N 倍 / 本库慢约 10%"都是**跨进程/坏窗口**读数
> （具体旧值见 **git 历史**：`git log -p docs/dev.md`）——按 `docs/dev.md` §6.4 的规矩
> "**'差 N 倍'必须先同进程交替复核**"，它们**只作旧账、不得引用**；本文件与 `docs/dev.md` 只保留
> **现行口径与结论**。
> **旧口径（Rust 池路径）的"打平"读数也属旧账、不得引用**（旧值见 **git 历史**）；
> 它与现行的 C ABI 切法**不可并排比较**，**现行正式口径见上一条（同进程调用级交替）与
> `docs/dev.md` §7.10 ⑮（盖章句）**。口径与全部现行数据见 `docs/dev.md`（**≥5 轮中位数 + 极差**，
> 不引用单次值；旧数字不可引用这一条规则见 `docs/dev.md` 的 §6.4）。
>
> **公式 DSL 的复用提示**：`matmul!` 的 `mul_into` 每次调用都会重新打包 `B`；层循环里复用同一
> 权重时应把准备阶段 hoist 出循环（256³ 实测 **143.7 µs → 96.5–99.1 µs**，与直接池化持平）。
> **DSL 已支持在公式里直接复用 plan**——右操作数**不写方括号**即为 `Prepared`：
> `let p = w.prepare(); … matmul!(y[M, N] = x[M, K] * p);` 生成 `p.apply_into(&x, &mut y)`，> 与手写 hoist 逐位一致（`to_bits()` 比对）；带方括号的旧写法 `w[K, N]` 行为完全不变。
> **实战例子（2026-10-15 加）**：**用 DSL 做单边数值拉普拉斯变换**——`Re F = Kc·f`、`Im F = -Ks·f`
> 两条矩阵乘，**一次调用算一批信号**：`cargo run --release --example laplace`（解析对照误差
> 1e-6…1e-3、加密收敛 ~4×/Δt 减半、DSL **368 GFLOP/s**）。用法与限制见 [docs/ops.md](docs/ops.md) §9.0。
> 完整用法与 RoPE 上的同类收益见 [docs/ops.md](docs/ops.md)。
> **标量列口径**：`+lasx` 下 LLVM 会把标量参考**自动向量化**，故 matmul 表中的 3.3–4.7× 表示
> "手写 intrinsic vs 编译器自动向量化"，并非"vs 朴素循环"。

以下数字为 **2026-10-15** 在同一台 Loongson-3B6000 上按**方差测量法**重测的结果：每组 **6 轮**
（丢弃首轮、取后 5 轮中位数）、进程级；单线程表绑核、带线程的表不绑核；分级按
`docs/dev.md` §6.4（本机桌面常驻负载 ≈2.5–3，绝大多数轮为 B 级正式数）。**每个数字的极差、
逐轮 loadavg 与完整口径见 [docs/dev.md](docs/dev.md) §7**；编译期常量/阈值的对照（k 分块、
列块宽度、调度策略等）记录在其中的设计小节。**测量工具**（复现这条链条）：
[`scripts/bench_rounds.sh`](scripts/bench_rounds.sh)（按组分批、轮间 settle、逐轮记 loadavg 与
A/B/C 分级）与 [`scripts/benchvar.rs`](scripts/benchvar.rs)（按行对齐各轮、输出 `中位数（极差）`，
`render` 模式直接输出可粘贴的表格行）。

| 项 | 读数 |
|---|---|
| FMA 吞吐上限（16 条独立链，纯寄存器） | **139.4 GFLOP/s**（3.96 条/周期，极差 130.9–140.1）；LSX 69.6（2.00×） |
| 微内核上限（含 A 标量广播） | **2.00 FMA/周期**（f32 70.3、f64 35.2 GFLOP/s；来自 `docs/dev.md` §6.3 的 `kernel_probe`，本轮未重测） |
| `lasx_matmul` f32 64³ / 256³ / 512³ | 66.9 / 62.7 / **61.5** GFLOP/s |
| `lasx_matmul_f64` 128³ / 512³ | 30.6 / 30.9 GFLOP/s |
| `lasx_matmul`（`parallel`，12 线程） | 128³ / 512³ 加速 5.89× / 5.51×（393.4 / 331.5 GFLOP/s） |
| 单流只读 / 双流只读 / 12 线程只读（32 MiB） | 10.7 / 9.6 / 18.5 GB/s |
| `lasx_softmax_rows` / `lasx_rms_norm` | 4.95–6.72 / 12.16–18.29 GB/s |
| `lasx_silu` / `lasx_gelu_quick` / `lasx_gelu_erf` | 3.88–9.18 / 4.54–8.61 / 3.59–4.93 GB/s |
| `lasx_rope`（NeoX，LASX 路径） | 33.3–71.7 GB/s |
| `lasx_dot_f16` / `lasx_gemv_f16` | 38.1–68.3 / 4.2–64.7 GB/s |
| 常驻池 vs 每步新建线程（跨 200 步复用） | 37.47 ms vs 83.81 ms（**2.24×**，单线程 302.69 ms） |

与同机 OpenBLAS 0.3.34（单线程）的对照、以及各形状的根因分析见 `docs/dev.md`。

> 性能数据的解读规则：上表是一次连续采集的横截面，**只能同表内比较**；跨运行（换一次采集、
> 换一天）必须重测。实测到的窗口间差异可达 40%，因此"改前跑一遍、改后跑一遍"不能用来判断
> 快慢——那要用进程级交替测量（见 `docs/dev.md`）。

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

每个算子的 C 签名与 C 侧调用契约（内存/安全模型/能力探测）见 [docs/c-abi.md](docs/c-abi.md)；
语义、数值契约与"由谁守"见 `docs/ops.md`。安全 Rust API 见 `lasx_rs::api`
（32 个 `pub fn`），另有 `lasx_rs::view` / `lasx_rs::plan` / `lasx_rs::shape` / `lasx_rs::pool` /
`lasx_rs::parallel`。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/c-abi.md](docs/c-abi.md) | **C ABI（唯一权威）**：完整符号清单与逐符号原型、内存与生命周期、安全模型（裸 vs `_checked`）、能力探测与降级、线程模型、ABI 稳定性策略、快速开始（四种链接方式） |
| [docs/ops.md](docs/ops.md) | **算子与用法（Rust 侧）**：数值契约与逐位确定性、降级覆盖分类、池与并行、矩阵视图/计划/形状层、NN 算子现状与缺口对照、int8/f32 判据；C ABI 见 `docs/c-abi.md` |
| [docs/dev.md](docs/dev.md) | **架构与性能**：分层与不可破坏的约定、测试与 CI、性能方法学、**全部实测数据**、矩阵乘与并行深挖、被否掉的方案清单、复现步骤与已知缺口 |
| [docs/platform.md](docs/platform.md) | **平台约束与待补实现**：LA664 原子操作丢失更新（勘误、命中点、措施）与"库侧为什么必须自带 `_db`"、内核分支生命周期、消费 ONNX 契约引出的缺口（LayerNorm/Gather）、承诺边界待划定的项 |

文档分工是硬约定：**C ABI 在 `docs/c-abi.md`（符号清单与 C 侧契约只在那里维护）**，
Rust 侧契约与用法在 `docs/ops.md`，性能数据在 `docs/dev.md`，**同一份数字只存一处**。
文档内所有跨文档引用都写成"文档名 + 节号"形式，可机械校验。

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
