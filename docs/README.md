# lasx_rs 文档

本目录收录 lasx_rs（LoongArch LASX/LSX 向量库）的技术文档。

## 文档结构

```
docs/
├── README.md   # 本文档：文档目录索引
└── manual.md   # 完整技术手册（主文档）
```

| 文档 | 内容 |
|---|---|
| [manual.md](manual.md) | **完整技术手册**（10 章）：架构与设计（LASX/LSX 双路径机制、SOA 布局、FFI 设计）、15 个 FFI 内核逐一详解（C 签名/算法/向量化/精度/适用场景）、量化内核（q4/i8）、批量物理内核（J2/RK4/弹道）、FFI 使用指南（C/Dart/Rust/内存管理）、性能基准方法、测试与验证（5 个测试）、构建与集成（nightly/CI）、Caveats 与限制、API 索引 |

## 快速入口

- 想了解**整体架构与双路径降级机制**：见 manual.md 第 1 章
- 想查**某个内核**的签名与算法：见 manual.md 第 2 章（15 个内核各一小节）
- 想在 **C / Dart / Rust** 中调用：见 manual.md 第 5 章
- 想了解**精度/逐位确定**的边界：见 manual.md 第 7 章与第 9 章
- 想**复现性能基准**：见 manual.md 第 6 章

## 与根 README 的关系

- 根 [README.md](../README.md)：项目简介、构建、用法与基准速览；
- 本目录 manual.md：逐行对照 `src/lib.rs` 的深度技术参考。

> **注意**：manual.md 以 `src/lib.rs` 实际代码为准，个别处与根 README 表述
> 不同（如 `.cargo/config.toml` 实际仅 `+lasx`、`lasx_ballistic_step` 向量路径
> 与标量参考不一致等），详见 manual.md 第 8、9 章。
