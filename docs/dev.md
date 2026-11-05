# lasx_rs 开发文档：架构、契约与性能

> 定位：**设计/决策**文档（怎么用见 `docs/ops.md`，C ABI 见 `docs/c-abi.md`，平台见 `docs/platform.md`）。
> **性能数字的唯一权威位置是 §4**，其它章节只给结论与指针、不复制表。
> 历史记录（逐批/逐 rev 的流水账）已从本文移除：需要时查 `git log`；本文只保留当前真值与最近的决策摘要。

## 1. 总览与文档口径

### 1.1 本文怎么读

- 只想用库：看 `README.md` → `docs/ops.md`。
- 要改内核/调优：先读 §3（约定）、§4（口径），再动手。
- 要做 C/FFI 集成：`docs/c-abi.md`；平台/工具链：`docs/platform.md`。
- 与 DSL 仓（`simd_dsl`）的边界：§2.4。

### 1.2 文档纪律（写在门禁里，不是口号）

1. **数字只能有一个权威出处**：性能数字只在 §4；其它文档引用时给结论 + 指针。
2. **跨文档引用必须点名归属**：写 `（lasx 仓）docs/dev.md §x` / `（DSL 仓）对应文档 §x`；裸 `§x` 只指本文件。
3. `scripts/doccheck.sh` 强制：`§` 引用可解析、表格列数一致、文档写的导出符号数与源码一致。CI 的 fmt job 跑同一条。
4. 改代码前先看 §3.1 的「不可破坏的约定」；动接口先看 §6 的冻结清单。

### 1.3 性能口径速查

- 正式数字 = **同进程、调用级交替 A/B**（≥6 对、丢首对、中位数 + 极差），**跨进程/跨窗口的旧数字一律不可引用**。
- 每份报告必须带 `iters` / `ms_max` / `loadavg` + 分级（§4.1）。
- **单线程内核**用 `taskset` 绑核减噪；**带线程的表不绑核**。
- 判据顺序：**位精确 > 速度**；"理论上应该更快"不算验收（§5）。
- 复现命令见 §4.6。

## 2. 架构与分层

### 2.1 仓库结构

| 路径 | 职责 |
|---|---|
| `src/arch/` | LASX/LSX/标量三层指令封装与 CPU 特性检测 |
| `src/ops/` | 内核实现（matmul/dot/reduce/norm/激活/几何/量化/FFT/stride/gather…） |
| `src/parallel.rs` | 并行矩阵乘与行块调度（调用方线程模型） |
| `src/pool/` | `WorkerPool` 线程池（持久 worker、futex 唤醒、按槽派活） |
| `src/plan.rs` | 打包/计划句柄（`MatmulPlan` 类 API 的内部实现） |
| `src/ffi/` | C ABI（98 个导出符号，见 `docs/c-abi.md`） |
| `src/api.rs` | Rust 侧薄封装（`*_into` 入口） |
| `macros/` | `lasx_rs_macros`：公式 DSL 过程宏（`matmul!`/`dot!`/`gemv!`，形状进类型、编译期对齐；`publish=false`，由 `lasx_rs` 重导出） |
| `examples/` | 可运行示例（`laplace`、`matmul_ab`、`matmul_wide`、`kernel_probe`…） |
| `scripts/` | 门禁与基准（doccheck/namecheck/bench_*/install…） |
| `lasx_bench/` | 基准二进制（全量横截面、形状 A/B、微内核探针） |

### 2.2 命名规范

- 内核：`<族>_<dtype>[_<模式>]`（如 `matmul_f32`、`dot_f64`、`norm3_batch`、`dequantize_i8_rows`）；
  `_into` = 写调用方缓冲、不分配；`_rows`/`_cols` = 归约/映射的轴。
- 内部 helper：`*_lasx` / `*_lsx` / `*_scalar` 三分（同一数学、不同宽度）。
- 已裁决的 5 条例外（不按规范的名字）与理由见 `git log` 中 2026-09 的命名规范提交。

### 2.3 构建、工具链与平台防呆

- **必须 nightly**（依赖 `stdarch_loongarch`/LASX intrinsic）；稳定工具链不可用是**已知且接受**的边界。
- **`+lasx` 必须显式开启**：仓库 `.cargo/config.toml` 已设 `rustflags = ["-C","target-feature=+lasx"]`。
  **外部探针/新 crate 不会自动继承**——缺标志会静默回退标量（同一份代码慢 26–46×），任何 LASX 测量必须确认标志。
- 平台：LoongArch64（LA464/LA664）；LSX 128 位与 LASX 256 位两条路径 + 标量尾，**三者逐位一致**（§5）。
- 构建：`cargo build --release`；测试：`cargo test --release`；门禁：`bash scripts/doccheck.sh`、`bash scripts/namecheck.sh`。
- 导出符号数写进文档时必须 = `src/ffi/` 的 `pub extern "C" fn lasx_*` 实际数量（当前 **98 个导出符号**）。

### 2.4 与 DSL 仓（`simd_dsl`）的关系

- **本仓已纯化：零 DSL 依赖**。`dsl-backend` feature、`simd_dsl`/`ops_dsl`/`attention_block` 依赖与
  DSL 示例/conformance 均已移出本仓；`cargo tree -e normal` 只剩本仓 + `lasx_rs_macros`，CI 有 Purity guard。
- **依赖方向**：DSL 仓的适配器 crate `backends/lasx` 在 `Cargo.toml` 里**钉本仓 git rev**；升级本仓 =
  DSL 仓 bump 适配器 rev + 重跑适配器门禁与 A/B。
- 「引擎怎么选路/能力位/融合 plan」属于 DSL 仓的契约：见（DSL 仓）`docs/engine_integration.md`、
  （DSL 仓）`docs/backend_guide.md`。本仓只提供内核 + 诚实的性能声明。
- **能力位口径**：`Capability` 是**路由/性能声明**（true ⇒ 调用方不做外层并行、走本仓单次调用），
  不等于"有没有实现"；本仓刻意低估若干能力位以免路由变慢（详见 §7.2.2）。

## 3. 内核层

### 3.1 不可破坏的约定（改代码前必读）

1. **位精确是硬承诺**：同一算子所有路径（LASX/LSX/标量尾/打包/列块/分块）对同一输入给出**逐位相同**结果，
   测试用 `to_bits()`。例外只允许是登记过的容差类（§5.2）。
2. **不许静默换路径**：更快但改累加次序的路径，要么声明容差类（给实测界 + 参照物），要么不做。
3. **NaN/±0 语义固定**：`max/min` 族顺序 fold 保留 NaN payload 与 −0/+0；归约/映射的 NaN 行为按登记表（§5.2）。
4. **别名**：`*_into` 默认"输入输出不重叠"（调用方保证，不是 `Err`）；有重叠支持的地方单独注明。
5. **阈值/常量**（`PAR_MIN_WORK`、行块阈值等）属实现细节，可调；**等价类与错误码不可随手改**。

### 3.2 并行层（`WorkerPool`）

- 持久 worker（24 逻辑核机器，按 `available_parallelism` 建池），futex 唤醒；**逐槽代次**派活：
  块数 < 线程数时只等"有活的 worker"（修复过"全体签到"的派活台阶：`gemv 64×2048`@24 0.043→0.007 ms）。
- 线程数策略（调用方模型）：`threads ≥ 20 && per_thread ≥ 32` 两条粗规则，块长取每线程 1/4。
- 经验：24 逻辑核上"全占满"可能比 12 慢一倍（调度代价）；**调用方少给线程**（物理核数、留一个空位）。
- 每线程池的固定开销 ≈486 µs（24 次建/回收）；因此**池只建一次、跨调用复用**。

### 3.3 矩阵乘深挖（当前结论）

- 微内核主循环 **31 条/k 步**：4×`xvld`(B) + 4×`xvldrepl.w`(A) + 16×`xvfmadd.s` + 6 addi + 1 bnez（FMA 密度 51.6%）。
- 实测上限 **2.0 FMA/周期**（`examples/kernel_probe.rs`）；**"A 广播墙"假设已证伪**：换广播指令仍 2.00 FMA/周期。
- 1180³/1024³ 的扫描与共享面板、列瓦片、k 分块等尝试见 §3.6 与 §4.4；**位精确 > 速度**贯穿所有决策。
- 打包在并行层是**串行**的（已知缺口 §8.1）：`m < 512` 与宽形状吃不到共享面板收益。

### 3.4 大集、形状专项与内核上限

- **大集锚点**（4–32 MiB 纯搬运）：单线程流式 **6.5–14.7 GB/s**，12 线程 **15.6–24.3 GB/s**；读写混合比纯读低 30–40%。
- 各内核/锚点：`dot` 89–95%、`sum` 81–94%、`norm3_batch` f64 66–68%、`vec3_add_scaled` 84–96%、
  `rk4_j2_step` 单线程 32–40%/12 线程 72–86%；**先看"离锚点多远"再决定优化方向**。
- NN 内核横向：`rms_norm` 2 遍 6.1–9.1 GB/s；`softmax_rows` 3 遍 1.7–2.2；`silu/gelu_quick` 2 遍 1.9–4.6（**算力受限**）；
  `gelu_erf` 1.8–2.7；`rope` 13.3–28.7；`dot_f16` 38.1–68.3（小形状驻缓存）。
- 真实调用形态：池扩展性 1→2→4 线程 2.03×/4.04×，8 线程后停 8.1–9.1×（带宽）；**融合内核比拼原语快 3.5×**（中间量不出寄存器）；
  C ABI 固定开销 n=64 19 ns vs Rust 13 ns（小数组高频调用可见）。

### 3.5 NN 与 int8 批次（要点）

- N1：`rms_norm`/`softmax_rows`/`silu`/`gelu_*`/`rope` 已落地；`gelu_erf` 精确版与 `gelu_quick` 近似版并存（后者容差类，§5.2）。
- N3 int8：量化生产端（per-row scale、ties-even + clamp）、`dot_i8`（i64 累加 + 饱和）、`gemv_i8`、`matmul_i8`（保留但非性能路径）。
- f16：只做 GEMV/dot（`dot_f16` 38–68 GB/s）；f16 GEMM 未做。

### 3.6 定向优化与被否掉的尝试（内核侧）

| 尝试 | 结论 | 关键原因/数字 |
|---|---|---|
| 红黑分块 5 点 stencil | **否** | 4096² 0.82–0.86×；三行滚动缓冲 1.38–1.75×（逐位）才采纳 |
| 融合原语（P4-e）标量版 | **否**（后被推翻） | 标量融合比向量化两步慢 1.793×；向量化后 0.363×（快 2.75×，逐位） |
| gather/scatter 8-lane 大改 | **否** | 上限 1.037×（memcpy 上限） |
| 估算倒数 + 牛顿迭代 | **否** | 本机实测不划算 |
| `std::thread` 并行建 rope 表 | **否** | 唤醒/同步成本吃掉收益 |
| 核↔矩阵乘融合（现场生成） | **不采用** | 维护/优化成本过高；融合原语走契约层是另一条路 |
| "完全 SIMD"外包给 LLVM | **否（红线取 A）** | LLVM 迭代器版 1.48×/组合 1.09×；朴素 6.24×/3.20× |

## 4. 性能与测量方法学（权威）

### 4.1 口径总纲（正式口径 = 同进程调用级交替 A/B）

1. **同进程、调用级交替**：各臂都链进同一进程、同一份数据/缓冲；每对交替；计时前**双检查**（各臂互比 `tol` 内 **且** 与串行参考逐位一致）。
2. **≥6 对、丢首对**，各臂**中位数 + 极差**；每臂报 `iters` 与 `ms_max`。
3. **每对 `loadavg` + 分级**：A（1/5 分钟同时 ≤3）/ B（可作正式数）/ C（只作趋势）。本机常驻 2.5–3 + 测量本身 +1 ⇒ **A 级基本不可达**。
4. **跨进程/跨窗口旧数字不可引用**：同档位跨窗口可差 2–3×；"差 N 倍"必须同进程交替复核（单次取样可造 4.4× 假象）。
5. **绑核规则**：`taskset` 只对单线程内核减噪；**任何带线程的表不绑核**（12 worker 挤一核可测出"负扩展"）。
6. **跨库公平四条**：同数据同预热；布局映射明确；计时前先验一致性；线程对齐且切法标注。
7. **前台纪律**：门禁/测量不接管道（判据只看退出码）；测前查 `loadavg`、nice 降权、不混跑两个线程库。
8. **性能结论必须实测签字**：模型只给上界（§5）。

### 4.2 机器与上限（先知道天花板在哪）

- 机器：Loongson-3B6000（LoongArch64/LA664，2.2 GHz，24 逻辑核）；`rustc 1.100.0-nightly`；桌面常驻 2.5–3，跨轮同一读数可差 ±20%。
- GEMM 微内核上限 **2.0 FMA/周期**；纯 FMA 探针须 **≥16 条独立累加链**（8 条量到的是延迟上限）且 **5×10⁶ 轮**（访存型 200 轮），否则计时分辨率不足。
- 外部探针必须自带 `+lasx`（仓库 `.cargo/config.toml` 不作用于外部 crate）。

### 4.3 大集锚点与内核占比

| 规模 | 单流只读 | 双流只读 | 双流字节读 | 6 流 3读3写 | 单线程拷贝 | 12 线程只读 | 12 线程拷贝 |
|---|---|---|---|---|---|---|---|
| 4 Mi | 14.7 | 10.5 | 39.9 | 6.7 | 7.8 | 24.3 | 16.7 |
| 16 Mi | 11.4 | 9.5 | 12.9 | 6.6 | 7.1 | 18.1 | 17.1 |
| 8 Mi (f32) | 11.4 | 10.3 | 21.0 | 6.5 | 7.3 | 19.7 | 15.6 |
| 32 Mi (f32) | 10.7 | 9.6 | 9.3 | 6.8 | 7.2 | 18.5 | 16.4 |

要点：单线程 6.5–14.7 GB/s、12 线程 15.6–24.3；读写混合比纯读低 30–40%；6 流压在 6.5–6.8 且与规模无关 ⇒ **大集下并行度比指令宽度重要**。

各内核站到锚点的比例：`dot` 89–95%、`sum` 81–94%、`norm3_batch`(f64) 66–68%、`vec3_add_scaled` 84–96%、`j2_accel_batch` 58–74%、
`axpy` 108–125%（读写混合锚点偏低）、`rk4_j2_step`(f64) 32–40%（单线程）/72–86%（12 线程）。

### 4.4 矩阵乘 vs OpenBLAS 的现行结论

> 对手身份行：**OpenBLAS 0.3.34、DYNAMIC_ARCH、core `la464`（LA64 优化内核，非通用 C 兜底）、NO_AFFINITY、MAX_THREADS=32**；
> 调用模型 = **调用方自己驱动（plan + 调用方线程）**。这些条件必须一起写，否则数字不可比。

- **单线程**（`taskset -c 4`）：小形状领先（64³ +27%、128³ +14%），256³ 打平，384³/512³ 慢 5–6%。
- **多线程 `192×768×3072` f32 @12**：`rows`（按行切）62.6 G 且几乎不扩展；`cols` 236.2 G 但 6/10 形状不逐位；
  `plan`（计划句柄）正式中位 **239.8 G**；`plan-tight` **245.8 G**（逐位断言）；结构上界 **346–355 G**（瓶颈是 L3 带宽/停顿，不是 DRAM）；
  `omp` 档 199.2 G；`threaded` 便利层 129.9 G。**"≥300 G 正式中位数"在本机窗口给不出** ⇒ 结论写"结构性达成、窗口受限"，不拿单次 306.6 G 当结论。
- **宽薄 `256×64×4096` @12**：plan 0.6355 ms（211.2 G）vs OB 0.6122 ms（219.2 G）——落后 3.6%。
- **1t 派发伪影已查清**：显式关自动并行后 1024³ 0.980×、1536³ 1.002×（B 级）⇒ **1t 差距为测量口径问题，已抹平**。
- **24 线程大矩阵**：1024³ 97.5→393.3 GFLOP/s（3.9× OB）、16 线程 421 G（§7.2.1 记录）；剩余缺口是并行打包（§8.1）。
- 窗口纪律：高负载窗口同一档位三次可 74.7/77.9/108.6 G ⇒ 必须带 `iters`/`ms_max`/`loadavg`。

### 4.5 方法学教训（多次踩过）

① 跨台架/跨窗口差 1.4× 不能直接推结构原因；② "差 N 倍"必须同进程复核；③ 测量前确认机制真的被触发；
④ 中位数相除会伪造比率（必须同一次运行内算）；⑤ 负载窗口要当场判废（C 级只作趋势）；
⑥ **缺 `+lasx` 的测量一律作废**（会静默回退标量，慢 26–46×）。

### 4.6 复现命令

```bash
# 全量基准（约 1 分钟；输出即 §4 的横截面表）
cargo run -p lasx_bench --release
# 单形状进程级交替 A/B：<m> <k> <n> <packed|cols|stream|packed64|stream64|pool> [线程数]
cargo run -p lasx_bench --release -- matmul_ab 512 512 512 pool 12
# 微内核探针：<变体> [k] [轮数]
cargo run -p lasx_bench --release -- kernel_probe fma16
# 池线程数扫描
cargo run -p lasx_bench --release -- pool_scaling
# OpenBLAS 对照（同二进制内交替、自动打印对手身份行）
bash scripts/bench_openblas.sh
```

## 5. 精度承诺与容差

### 5.1 承诺与红线

- **位精确是硬承诺**：同一算子所有实现路径对同一输入给出逐位相同 `f32`/`f64` 结果（`to_bits()` 比对）。
- **不许静默换路径**：更快但换累加次序的路径，要么明确声明**容差类** + 实测界 + 参照物，要么不做。
- **换近似 = 改数值**：`xvfrecpe`+牛顿、`xvrsqrt`、FMA 收缩、多累加器归约都属容差类，须授权并登记。
- 测试侧：容差类必须有"与参照物"的实测界；逐位类用 `to_bits` 全等断言。

### 5.2 逐位类 / 容差类清单（当前契约）

- **逐位类**：matmul f32/f64（全路径）、dot f32/f64/i8、sum/max/min/absmax 行/列归约、norm/softmax/rms/layer（同公式）、
  几何 f64 五核（`norm3`/`vec3_add_scaled`/`unitize3`/`cross3`/`mat3_mul_vec3`）、`norm2_batch`、`dequantize_i8_rows`、
  量化生产端（ties-even + clamp）、`copy_strided`、`gather/scatter`、FFT（与标量参照同 `to_bits`；见边界）。
- **容差类**（登记值见对应测试）：`gelu_quick`（max abs 2.063e-2，界 5e-2）、`silu` 近似、`exp`/`sqrt` 的 LASX 入口（与标量同为 IEEE 时逐位）、
  多累加器归约（`reduce_sum_rows` 等按行分量）、`dot_q4`/`dot_f16` 的求和次序。
- 边角语义固定：`amax` 全 NaN 行 → 0.0、−0 → +0；`max/min` 族保持顺序 fold（NaN payload/±0 逐位）。

### 5.3 结构性依据与边界

- 位精确靠"同一操作序列"：LASX/LSX/标量三层用相同数学、相同累加次序（或明确多累加器形状再登记容差）；
  非 IEEE 指令（近似倒数/rsqrt）不进逐位路径。
- 精度边界：f32 相对误差 ~1e-7/元素；累加类误差随 n 增长（容差类登记实测界）。

## 6. 接口冻结与契约

### 6.1 冻结清单（要改先在本文加一节说明理由）

1. **池的行块闭包**（`WorkerPool` 派活接口）：宏与并行层按它生成调用，改签名会连带改宏与测试。
2. **`shape::Prepared` 的 `apply*`**：打包/分块路径的公共入口。
3. **`Policy::threads`**：调用方线程意图的唯一表达。
> 冻结不是形式，是**成本声明**：每改一次都要连带改宏、测试与（DSL 仓）适配器；先记录再改。

### 6.2 错误通道、panic 与别名契约

- 错误统一 `Result<_, LasxError>`（C ABI 用错误码 + `*_checked` 变体）；**契约错误返回 `Err`，不 panic**；
  panic 只用于"调用方违反文档承诺"（如别名重叠）且必须在文档点名。长度/形状不符走错误码，错误消息需可行动。
- 别名：`*_into` 默认输入输出不重叠；gather/scatter 的索引与数据重叠不支持（未定义行为，调用方保证）。

### 6.3 计划句柄与调用模型

- 打包一次、多次调用：`MatmulPlan`（C ABI 的 plan 句柄）把 B 打包/列主序映射存下来跨调用复用；
  多线程时由**调用方线程**驱动（库内不开并行区），实际数字见 §4.4。
- C 侧调用体验与示例见 `docs/c-abi.md`。

## 7. 现状沉淀

### 7.1 最近里程碑

| 日期 | 主题 | 结论 |
|---|---|---|
| 2026-11-05 | 缺口收窄沉淀 | 覆盖矩阵/能力位口径/Norm2BatchF64+DequantizeI8Rows 两个新核/GEMM 研究结论（§7.2） |
| 2026-11-05 | 适配器 16/16 plan 全真 | DSL 仓 `backends/lasx` 适配器把两个新核接上（逐位） |
| 2026-11-05 | P1 修复（bench 缺 +lasx） | 自包含 bench 显式设 RUSTFLAGS；修复前 LASX 档慢 26–46× 为测量假象 |
| 2026-10-28 | 依赖链翻转 | 本仓纯化（零 DSL 依赖）；DSL 仓适配器钉本仓 rev |
| 2026-10-22 | 组合桥清零 | 18 个入口升级引擎 `Algorithm`；独立复核 342/342 digest 0 差异（DSL 仓工作） |

完整历史（逐批/逐 rev 表、历史读数）见 `git log`——**旧读数不要再引用**（§4.1）。

### 7.2 缺口收窄沉淀：覆盖矩阵 / 能力位口径 / 新算子 / GEMM 研究

> 审计原始记录：`/home/lik/tmp_keep/results/lasx-gap-audit-178.md` 等（本地研究文件，不入库）。
> **所有 LASX 测量必须带 `+lasx`**（§4.5）；loadavg/grade 按 §4.1 标注。

#### 7.2.1 覆盖矩阵摘要

- DSL 仓 `Backend` **62 个方法**：适配器 **45 个具名 override**，其余走 trait 默认。
- **16 个 `PatternKind` 全真**：f32 5（Axpy/SoftmaxRows/RmsNormRows/LayerNormRows/RopeApply）+ f64 几何 6
  （Vec3AddScaled/Norm3/**Norm2**/Unitize3/Cross3/Mat3MulVec3）+ i8 4（Quantize/**Dequantize**/DotI8/GemvI8）+ GeluQuick。
- **16 个功能性默认方法**（无 LASX 核或接入更慢，保持 portable）：`map_unary_f64`、`reduce_sum_f64`、
  `reduce_weighted_sum_f32/f64`、`reduce_cols_map_f32`、`map_ternary_f32`、`map_nary_f32/f64`、`map_unary_masked_f32`、
  `build_banded_kernel_f32`、`stencil_rows_f32`、`exec_fused_plan_portable` + 4 个 display/helper。
- **为何不接**：stencil/build_banded/nary/masked/cols_map 无对应 LASX 核；`reduce_weighted_sum → dot` 薄 override
  实测比默认**慢 2.3–15.6×**（默认已被 LLVM 自动向量化）⇒ 不做；f64 unary 需新增核、收益未证。

#### 7.2.2 能力位语义口径（**刻意低估，勿"修正诚实性"**）

`supports(cap)` 是**路由/性能声明**（true ⇒ 调用方单次调用、不做外层并行），**不是"有没有实现"**。
本仓因此刻意低估；适配器 conformance 有断言钉住这些返回值，防止后人误改。

| 变体 | 实际实现 | 上报 | 理由（关键实测） |
|---|---|---|---|
| MatmulF64 | `matmul_f64` 有核（单线程） | **false** | true ⇒ 跳过行并行；整块/引擎 512³ 慢 3.28×、1024×256×1024 慢 2.47× |
| Elementwise | binary/affine/scalar/rows/cols 专用；unary 仅 Exp/Sqrt | false | true ⇒ 丢行并行，且多数 unary 仍走标量 |
| Reduction | 12 个 reduce override 专用 | false | 同上：定义应 true、路由应 false |
| Transcendental | 仅 Exp/Sqrt 有核 | false | 核不全 |
| ComplexRoutes | binary ✓、`map_ternary_f32` 走默认 | false | true 需补 ternary（无通用 3 输入指令） |
| F64Elementwise | affine/binary f64 ✓、`map_unary_f64` 标量 | true | **族级近似**；引擎 0 消费 ⇒ 无行为影响（已在适配器注释标注） |
| Threaded | matmul 内部自动并行 | true | 语义 = "内核自动并行"；调用方据此不嵌套线程 |
| FusedGeometry/Quant | 语义已转融合 plan 探测 | 不报（保留枚举） | 避免双口径 |

#### 7.2.3 新算子（均为**逐位**，已接 plan）

- **`Norm2BatchF64`**（`src/ops/norm2_batch.rs`、`api::norm2_batch_into`）：`out[i]=√(x²+y²)`，f64 SOA；
  LASX 4 f64/lane + 2 路展开，LSX 2/lane，标量尾同式；硬件 sqrt 与标量同 IEEE 正确舍入 ⇒ `to_bits` 全等。
  A/B（plan vs portable，6 对丢首对中位）：**1M 3.96× / 8M 3.64×**。测试含 n=0/1/3/4/8/33/1024 与错误面。
- **`DequantizeI8Rows`**（`src/ops/dequantize_i8_rows.rs`、`api::dequantize_i8_rows{,_into}`）：
  `out[r,c]=(q as f32)·scale[r]`；i8→i32 两级拓宽 + lane 重排（无 i8→f32 直转指令）+ `ffint` + 行标量乘。逐位
  （含 ±127/−128/±0/次正规/NaN/±inf scale）。A/B：**1M 2.11×**、**8M 1.09–1.52×（中位 ~1.29，带宽受限、抖动大）**。
  **风险（如实）**：8M 未稳定越过 1.2× 门槛，若窗口复测不支持可**一行 revert**（`supports_plan` 回 false 即回退 portable）；
  LSX-only CPU 不接（128-bit lane 更窄、重排占比更高，走同式标量）。

#### 7.2.4 GEMM 研究：源码级已到头（1t f32 vs OpenBLAS 0.3.34 la464）

基线（`taskset -c 4`、进程级交替、6 轮丢首取上中位，B 级窗口）：**512³ 0.929× / 1024³ 0.961× / 1536³ 0.956×**（OB/LASX）。

perf 归因：512³ 每次调用 LASX 多 ~8.3% cycles、+4.9% 指令、IPC 低 3.2%、L1D miss 1.8×、分支指令 5.5×；
FMA/cycle **1.67 vs 1.80**（远未到 2.0 墙）⇒ 剩余 ~6% 主因是 **k 内层循环取指/分支/指针开销 + 部分 L1 miss**，不是 FMA 端口。
已否决假设（均逐位不变）：`pack_b` 分组 gmax（pack 仅占 1.78% ⇒ 上限 <0.1%）、`pipe_b` 软件流水（−7.5%/−8.5%）、
prefetch +1.7%（唯一正项、远低门槛）、k2 展开（instructions −9.9%、branches −46%，但 IPC −12.8%、ms +2.4% ⇒ 更慢）。
**结论**：源码级优化到头；跨过 0.98× 只剩 asm 微内核（历史已否决）或接受 512³≈0.93–0.95、1024³≈1.0× 的现状。

#### 7.2.5 不做清单（写理由，防重复投入）

- `DequantizeI8Rows` 8M 风险档：保留但标注（§7.2.3；可一行 revert）。
- spectral / stencil2d / 几何 f32 手写 LASX 核：**不做**（无对应核/带宽级收益/已由通用 map+reduce+f64 plan 覆盖）。
- `reduce_weighted_sum_f32/f64 → dot`：**不做**（比默认慢 2.3–15.6×）。
- `map_ternary_f32`（为 ComplexRoutes 补）：**不做**（无通用 3 输入指令；cmatmul 热核已是 `matmul_f32`）。

### 7.3 跨仓 rev 管理（摘要）

DSL 仓适配器 `backends/lasx/Cargo.toml` 钉本仓 git rev。升级流程：本仓提交 → DSL 仓 bump rev →
DSL 仓重跑适配器门禁（fmt/build/test/clippy/algo 冒烟/conformance）+ 三方式 A/B。**本仓不再钉任何 DSL rev**
（纯化后零 DSL 依赖）。完整 rev 历史见 `git log` 的 `Cargo.toml` 变更。

## 8. 已知边界与欠账

### 8.1 缺口与待办（按"要不要做"排）

| 缺口 | 现状 | 下一步 / 触发条件 |
|---|---|---|
| 并行面板打包 | 打包在并行层串行 ⇒ `m < 512` 与宽形状吃不到共享面板收益 | 池加"按固定块长切分"接口；去掉 `SHARED_PACK_MIN_M` 后预计 +15–20% |
| f64 大矩阵 | CLI 只到 512³；1024³ 只有历史读数 | 扩规模并核对是否仍落后 OB ~1.1× |
| f64 内核离天花板 11% | 单线程 512³ 89%；多核已由共享打包修复 | 剩余是访存结构（A 重读 + C 按 k 块读写） |
| 宽形状 `100×448×19147` | 55.0 GFLOP/s，与 OB 54.8 持平 | 已排除 A 打包/C 分块/共享打包；剩预转置 B 或列块流水 |
| `dot_q4` 大集 | 3.6–7.2 GB/s（38–44%） | 已到 2.0 FMA/周期墙；除非授权改浮点结合次序 |
| `rk4`/`j2` 再快 | 每元素 4 次除法（sqrt 4 次）≈算术地板 | 只能改公式/放宽精度，或并行化 4 个 RK4 阶段 |
| 多核大矩阵 vs OB | 1024³ 24 线程 393.3 G（3.9×）、16 线程 421 | 见"并行打包" |
| 动态调度细调 | 两条粗规则，块长=每线程 1/4 | 再榨 5–10% 需可信负载模型（当前噪声 ±10% 量不出） |
| unsafe 审查 | Miri + cargo-careful 已接入、lint 已启用 | Rudra/Kani 未接入（需 `rustc-dev` + 特定 nightly） |
| int8 推理路径 | 量化生产端/per-row scale/`gemv_i8` 已落地；`matmul_i8` 保留但非性能路径 | 整数 softmax/归一化**不做**（实测不值得） |

### 8.2 NN 与 int8 未完成项

- 激活是**算力受限**：提速要缩短 `exp` 多项式或重排依赖链（不是加带宽）。
- f16 GEMM（prefill）未做：N1 只有 f16 GEMV/dot；外部参照 8.4/16.6 GFLOP/s vs 本库 f32 GEMM 60–67。
- N2（GGML 量化格式互操作，按需）：`quantize_rows_q8_0`、`dot_q4_0_q8_0`/`gemv_q4_0`、`dot_q6_K_q8_K`。

### 8.3 平台与外部约束

- **LA664 勘误与内核生命周期**由 `docs/platform.md` 记录（不在本文重复）。
- 工具链依赖 nightly（`stdarch_loongarch`）；稳定工具链不可用是**已知且接受**的边界。
- 本仓零 DSL 依赖；依赖方向与 rev 管理见 §7.3 与 §2.4。

### 8.4 测量环境限制（结论的适用范围）

- 常驻负载 2.5–3 + 测量本身 +1 ⇒ A 级基本不可达，正式结论绝大多数是 **B 级**；C 级只作趋势。
- 跨窗口不可比（同档位可差 2–3×）；报告必须带 `iters`/`ms_max`/`loadavg`。
- perf 软件事件（context-switches/cpu-migrations）在此机器不计到这些线程 ⇒ 不能用它支撑"同步没问题"；
  硬件事件（`llc-*`）可用；比率必须同一次运行内算。
- OpenBLAS 对照只在"对手身份行"打印后有效；调用模型必须一起写。
