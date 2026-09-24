# lasx_rs

面向 **LoongArch64（龙芯 3B6000 / LA664，LASX 256 位）** 的数学内核库：零依赖、Rust 实现、
导出稳定的 C ABI，另有安全 Rust API 与常驻工作池。

覆盖三类负载：**归约与稠密线性代数**（`dot`/`sum`/`axpy`/`matmul` 的 f32 与 f64）、
**量化点积**（int8、GGUF 风格 Q4）、**批量物理与姿态/几何**（J2 引力、RK4 步、弹道、
3 分量模长、四元数旋转/DCM 等）。

## 特点

- **零第三方依赖**，只用 `core::arch::loongarch64` 的 LASX/LSX intrinsic。
- **47 个导出符号**：24 个裸版本 + 23 个带 `int *status` 的 `_checked` 版本。
  最初的 15 个 `lasx_*` 符号签名与语义**永不改动**，新能力一律追加。
- **位精确**：同一算子的 LASX / LSX / 标量尾 / 打包 / 分块等所有路径，对同一输入给出
  **逐位相同**的结果（测试用 `to_bits()` 比对）。**边界**：`lasx_ballistic_step` 的向量与标量
  实现在不同循环结构下不保证逐位一致，`lasx_axpy` 允许 1 ulp 以内差异；跨实现数值差 < 1e-9。
- **打包 + k 分块的矩阵乘**：f32 512³ 约 60 GFLOP/s（单线程，实测峰值 140 GFLOP/s 的 43%）。
- **常驻工作池**：多星多步传播场景端到端比"每步新建线程"快 1.77×（跨 200 步复用）。
- **矩阵视图与打包计划**：`view::MatRef` 把形状/跨距变成构造时校验一次的对象；
  `plan::MatmulPlan` 把 `B` 打包一次反复用（`B` 固定时每次调用省掉一次 `O(k·n)` 搬运与一次
  输出分配，且与 `api::matmul` 逐位一致），计划只读，可 `Arc` 给多线程共享同一份打包 `B`。

## 快速开始

```bash
# 需要 nightly（#![feature(stdarch_loongarch)]）与 LoongArch 真机
cargo build --release          # 产出 liblasx_rs.so + rlib
cargo test --release           # 97 单测 + 11 文档测试（另 2 个 doc 示例标 ignore）
cargo clippy --workspace --release --all-targets   # 零警告是硬门槛

# 基准：不带参数跑全部套件，带子串只跑匹配套件
cargo run -p lasx_bench --release -- large
cargo run -p lasx_bench --release -- dot_q4

# 单形状矩阵乘 A/B：<m> <k> <n> <packed|cols|stream|packed64|stream64>
cargo run --release --example matmul_ab -- 512 512 512 packed
```

`.cargo/config.toml` 里 `rustflags = ["-C", "target-feature=+lasx"]`，所以默认产物假定 CPU 有 LASX；
`.cargo/config.toml` 的示例从简：见文件内注释。

## 实测性能（2026-09-22，Loongson-3B6000，单线程）

| 项 | 读数 |
|---|---|
| FMA 吞吐上限（16 条独立链） | **140.3 GFLOP/s**（3.96 条/周期）；LSX 70.1（2.00×） |
| 含 A 广播的内核墙（微内核探针） | **2.00 FMA/周期**（f32 70.3、f64 35.1 GFLOP/s） |
| `lasx_matmul` f32 64³/256³/512³ | 68.5 / 61.8 / 59.7 GFLOP/s |
| `lasx_matmul_f64` 128³/512³ | 31.3 / 27.6 GFLOP/s |
| `lasx_matmul`（12 线程，`parallel`） | 128³ **563** GFLOP/s（加速 8.5×） |
| 单流只读 / 双流只读 / 12 线程只读（32 MiB） | 7.5–8.9 / 7.0–8.9 / 17.4 GB/s |
| 常驻池 vs 每步新建线程（n=2^18，8 线程） | 894.7 µs vs 1.08 ms（7.80× vs 7.20×） |

与同机 OpenBLAS 0.3.34（单线程）对照：f32 128³ 快 1.34×、256³ 快 1.04×、512³ 落后 1.06×、
1024³ 落后 1.19×；f64 512³ 落后 1.12×。完整表格、方法学与根因分析见 `docs/dev.md`。

## 算子一览

| 分类 | 符号 |
|---|---|
| 归约/稠密 | `lasx_dot`（f32，带 LSX）、`lasx_sum`、`lasx_axpy`、`lasx_dot_f64`、`lasx_matmul`、`lasx_matmul_f64` |
| 量化 | `lasx_dot_i8`（int8）、`lasx_dot_q4`（每 32 字节一组 scale） |
| 批量几何 | `lasx_batch_distance2d`、`lasx_norm3_batch`、`lasx_vec3_add_scaled_batch` |
| 批量物理 | `lasx_j2_accel_batch`、`lasx_rk4_j2_step_batch`、`lasx_ballistic_step` |
| 批量姿态（7 个） | `lasx_cross3_batch`、`lasx_unitize3_batch`、`lasx_mat3_mul_vec3_batch`、`lasx_quat_normalize_batch`、`lasx_quat_mul_batch`、`lasx_quat_rotate_batch`、`lasx_quat_to_dcm_batch` |
| 内存 | `lasx_alloc`（32 字节对齐） |

每个算子的签名、语义与数值约定见 `docs/ops.md`；另有安全 Rust API（`lasx_rs::api`，22 个函数）、
矩阵视图与打包计划（`lasx_rs::view` / `lasx_rs::plan`）、多核接口（`lasx_rs::pool` / `lasx_rs::parallel`）。

## 文档

- **[docs/ops.md](docs/ops.md)** —— 算子与用法：47 个符号总表、数值契约、降级覆盖、
  Rust/C/Dart 调用、池与并行、NN 算子现状、无损压缩路线。
- **[docs/dev.md](docs/dev.md)** —— 架构与性能：分层与约定、测试与 CI、性能方法学、
  全量实测数据、矩阵乘深挖、**被否掉的尝试清单**、复现步骤与已知缺口。

## 仓库

| 远端 | 地址 |
|---|---|
| GitCode（主） | `git@gitcode.com:H076lik/lasx_rs.git` / <https://gitcode.com/H076lik/lasx_rs> |
| GitHub（镜像） | `git@github.com:chzih076/lasx_rs.git` / <https://github.com/chzih076/lasx_rs> |
| 局域网 panel | `http://192.168.1.64/api/git/lasx_rs.git`（CI 走同一个 panel） |

CI 四步：`Format check` → `Build release` → `Test release` → `Clippy (zero warnings guard)`。
**CI 从不跑基准**（基准受后台负载影响，不适合当门禁）。

## 许可

MIT © 2026 lik（H076lik）。详情见 [LICENSE](LICENSE)。
