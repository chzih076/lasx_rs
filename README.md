# lasx_rs

**lasx_rs 是为龙芯平台推出的向量库，支持调用龙芯架构里面的 LASX（256 位）与 LSX（128 位）向量指令集。**

批量数值内核（点积/缩放/范数/距离/引力加速度等），无 LASX CPU（3A5000/3A6000 等 LSX-only）
自动降级 LSX 路径，逐位确定。

## 特性

- **LASX 256 位**：`lasx_*` intrinsics 批量内核（axpy / dot / norm3 / 距离 / J2 加速度 / RK4 步）
- **LSX 128 位降级**：LSX-only CPU 自动走 128 位路径（线程级强制降级钩子用于验证/测试）
- **零依赖**：纯 std + `stdarch_loongarch`（nightly）
- **逐位确定**：向量化与标量结果一致（防回归测试守护）

## 构建

**依赖 nightly Rust（含 `stdarch_loongarch` 实验特性）**：

```bash
# .cargo/config.toml 已配置：rustflags = ["-C", "target-feature=+lsx,+lasx"]
cargo build --release
cargo test --release
```

> **注意**：本库依赖**实验版本 rustc**（nightly + `#![feature(stdarch_loongarch)]`），
> 需 loongarch64 nightly 工具链（如 `rustup toolchain install nightly-loongarch64-unknown-linux-gnu`）。

## 用法

```rust
// 作为依赖：crate-type 含 rlib + cdylib
let out = lasx_rs::lasx_dot(&a, &b, n);
// 或 FFI：cdylib 导出 C ABI（extern "C" lasx_* 符号）
```

FFI 调用示例（Dart）：

```dart
final lib = DynamicLibrary.open('liblasx_rs.so');
// 绑定 lasx_dot / lasx_axpy / lasx_norm3_batch 等
```

## 性能基准（Loongson-3B6000，LA664）

| 内核 | 加速比 |
|---|---|
| 批量点积（n≥8） | 2.4-2.8× |
| 全阶引力批量 | 3.84× |
| 24 线程批量传播 | ~18× |

## 许可

MIT © 2026 lik（H076lik）。详情见 [LICENSE](LICENSE)。
