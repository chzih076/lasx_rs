# lasx_rs

Batch numerical kernel library for **LoongArch64 (Loongson-3B6000 / LA664, LASX 256-bit)**:
zero third-party dependencies, pure Rust, a stable C ABI, plus a safe Rust API, a persistent worker
pool and shape/plan layers.

> The **Chinese README (`README.md`) is authoritative**; this file is a short English guide.

## Capabilities

| Area | Content |
|---|---|
| Reduction & dense linear algebra | `dot`, `sum`, `axpy`, `matmul` (f32 / f64), packed plan handle |
| Quantization & int8 | GGUF-style Q4, int8 quantize/dequantize producers, `dot_i8`, `gemv_i8`, `matmul_i8` |
| NN operators | row softmax / RMSNorm / LayerNorm, SiLU, GELU (sigmoid approx / erf), RoPE, f16-weight GEMV |
| Batch geometry & physics | 2-/3-component norms, quaternions, J2 acceleration, RK4 step, ballistic, 2D distance, complex matmul |

- **Bit-exactness is a hard promise**: every implementation path (LASX / LSX / scalar tail / packed /
  blocked / parallel tiles) yields bit-identical results for the same input (checked with `to_bits()`);
  the only exceptions are registered tolerance classes (see `docs/ops.md`).
- **98 exported symbols** (raw + `_checked` length-checked variants); the original 15 `lasx_*` symbols
  never change signature or semantics.
- **Persistent worker pool** (threads reused across calls) and caller-driven threading for matmul plans.

## Platform requirements

| Requirement | Notes |
|---|---|
| Builds only on `loongarch64` | Other architectures fail at compile time; use `[target.'cfg(target_arch = "loongarch64")'.dependencies]` |
| Nightly toolchain | single feature gate `#![feature(stdarch_loongarch)]` |
| Add `+lasx` yourself | the repo's `.cargo/config.toml` is not read by dependents; without it builds stay correct but ~26–46× slower |
| LA664 erratum | atomic lost-update on 3A6000/3C6000/3B6000 silently corrupts results; production paths use barrier-ordered operations plus a regression test (`docs/platform.md`) |
| LASX-only CPUs | LSX-only parts (e.g. 3B6000M / LA364E) will SIGILL on LASX instructions |

## Quick start

```bash
cargo build --release
cargo test --release
cargo clippy --all-targets -- -D warnings
cargo run -p lasx_bench --release
```

## Docs

| Document | Content |
|---|---|
| `docs/ops.md` | operator & usage manual (contracts, boundaries, int8-vs-f32 criteria) |
| `docs/dev.md` | design/decisions: architecture, kernels, performance methodology & numbers, precision promises |
| `docs/c-abi.md` | C ABI reference: symbols, calling conventions, status codes, memory & plan handles |
| `docs/platform.md` | platform constraints: LA664 erratum, support status, ONNX contract gaps |

## Relation to the DSL repo (`simd_dsl`)

- This repo is **purified: zero DSL dependencies**.
- The DSL repo's adapter `backends/lasx` pins this repo's git rev; upgrading = bump rev there and rerun its gates/A-B.
- Engine routing, fused plans and capability semantics belong to the DSL repo: see (DSL repo) `docs/engine_integration.md`.
