# lasx_rs

A batch numerical kernel library for **LoongArch64 (Loongson-3B6000 / LA664, LASX 256-bit)**:
zero third-party dependencies, implemented in Rust, exporting a stable C ABI on top of a safe
Rust API, a resident worker pool, and shape/plan layers.

Four workload classes are covered:

| Class | Kernels |
|---|---|
| Reductions and dense linear algebra | `dot`, `sum`, `axpy`, `matmul` (f32 / f64) |
| Quantized dot products | int8, GGUF-style Q4 |
| Batch physics and attitude/geometry | J2 gravity acceleration, RK4 step, ballistic step, 3-component norm, quaternion rotation/DCM, … |
| NN operators (N1 batch) | row-wise softmax, RMSNorm, SiLU, GELU (sigmoid approximation and erf form), RoPE, f16-weight GEMV |

## Features

- **Zero third-party dependencies** — only `core::arch::loongarch64` LASX/LSX intrinsics and std.
- **59 exported symbols** (30 plain + 29 `_checked` variants taking a trailing `int *status`).
  The signatures and semantics of the original 15 `lasx_*` symbols are **frozen**; new capability
  is only ever appended. Both symbol families are backed by the same kernel implementations.
- **Bitwise determinism** — for a given input, every path of an operator (LASX / LSX / scalar tail
  / packed / k-chunked / parallel blocks) produces results that are **identical bit for bit**
  (verified with `to_bits()` comparisons). There are exactly two documented exceptions, each with
  tests: `lasx_ballistic_step` uses different association orders in its vector and scalar branches,
  and `lasx_axpy` permits a difference of at most 1 ulp.
- **Matrix multiplication** — three bitwise-identical paths (packed, column-block, streaming) with
  packing and k-chunking; single-threaded f32 512³ reaches 60.1 GFLOP/s, which is **85%** of this
  machine's microkernel ceiling (2.00 FMA/cycle with A-broadcast, i.e. 70.3 GFLOP/s).
- **Resident worker pool** — amortizes dispatch across calls: 8.15× on a 12-thread n = 2^18 step,
  and in a multi-satellite multi-step propagation it is **2.55×** faster than creating threads per
  step and **8.09×** faster than single-threaded.
- **Interface layers** — `view::MatRef`/`MatMut` turn shape and stride into objects validated once
  at construction; `plan::MatmulPlan` packs `B` once and reuses it (read-only, `Arc`-shareable);
  the `shape` layer encodes shapes in the type system with compile-time checks;
  `lasx_rs_macros` provides a formula DSL.
- **NN operators are LASX-only by design** — the N1 batch deliberately omits an LSX fallback,
  because every target platform in the 6000 series supports LASX. On a CPU without LASX these
  kernels execute LASX instructions and raise SIGILL; callers must dispatch by CPU capability.

## Getting started

```bash
# Requires nightly (#![feature(stdarch_loongarch)]) and a LoongArch machine
cargo build --release                    # produces liblasx_rs.so + rlib
cargo test --workspace --release         # 155 lib tests + 3 macro tests + 15 doc tests (2 ignored)
cargo clippy --workspace --all-targets -- -D warnings   # zero warnings is a hard gate
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p lasx_rs

# Benchmarks: no filter runs every suite, a substring selects matching suites
cargo run -p lasx_bench --release
cargo run -p lasx_bench --release -- softmax

# Single-shape A/B (the entry point for process-level alternating measurement):
#   <m> <k> <n> <stream|packed|cols|packed64|stream64|pool> [threads] [strategy]
cargo run --release --example matmul_ab -- 512 512 512 packed
```

`.cargo/config.toml` sets `rustflags = ["-C", "target-feature=+lasx"]`, so the default artifacts
assume LASX support. Two measurement rules apply to the benchmarks: **single-threaded kernel tables
are measured pinned to one core, thread-related tables are measured unpinned**, in two separate
runs (rationale and impact: `docs/dev.md` §7).

## Measured performance

The figures below come from a single continuous collection on one Loongson-3B6000 on
**2026-09-25** (background load ≈2–3; single-threaded results pinned to one core, thread-related
results unpinned). The complete tables, the methodology, and the exact traffic conventions are in
**[docs/dev.md §7](docs/dev.md)**, which is the single authoritative location for performance data;
comparisons of compile-time constants and thresholds (k-chunking, column-block width, scheduling)
live in the respective design sections.

| Item | Result |
|---|---|
| FMA throughput ceiling (16 independent chains, registers only) | **140.2 GFLOP/s** (3.98/cycle); LSX 69.3 (2.00×) |
| Microkernel ceiling (with A-broadcast) | **2.00 FMA/cycle** (f32 70.3, f64 35.2 GFLOP/s) |
| `lasx_matmul` f32 64³ / 256³ / 512³ | 68.8 / 62.2 / **60.1** GFLOP/s |
| `lasx_matmul_f64` 128³ / 512³ | 31.0 / 29.0 GFLOP/s |
| `lasx_matmul` via `parallel`, 12 threads | 5.06× / 5.34× at 128³ / 512³ (323.8 / 302.1 GFLOP/s) |
| Single-stream / two-stream / 12-thread read (32 MiB) | 12.0 / 10.4 / 22.0 GB/s |
| `lasx_softmax_rows` / `lasx_rms_norm` | 5.22–7.00 / 12.01–18.38 GB/s |
| `lasx_silu` / `lasx_gelu_quick` / `lasx_gelu_erf` | 4.74–9.20 / 4.55–8.61 / 4.09–5.51 GB/s |
| `lasx_rope` (NeoX, LASX path) | 28.6–71.9 GB/s |
| `lasx_dot_f16` / `lasx_gemv_f16` | 41.8–68.5 / 3.75–48.2 GB/s |
| Resident pool vs threads-per-step (reused across 200 steps) | 37.36 ms vs 95.37 ms (**2.55×**; single-threaded 302.28 ms) |

Comparisons against OpenBLAS 0.3.34 (single-threaded) on the same machine, together with the
per-shape root-cause analysis, are in `docs/dev.md` §8.5.

> How to read these numbers: the tables in §7 are a cross-section of one continuous collection and
> are **only comparable within that run**. Cross-run comparisons require re-measurement; observed
> window-to-window variation reaches 40%, so "run before, run after" cannot establish whether a
> change is faster — use process-level alternation (`docs/dev.md` §6.4).

## Operator overview

| Class | Symbols |
|---|---|
| Reduction / dense | `lasx_dot` (f32, with LSX fallback), `lasx_sum`, `lasx_axpy`, `lasx_dot_f64`, `lasx_matmul`, `lasx_matmul_f64` |
| Quantized | `lasx_dot_i8` (int8), `lasx_dot_q4` (one scale per 32 bytes) |
| Batch geometry | `lasx_batch_distance2d`, `lasx_norm3_batch`, `lasx_vec3_add_scaled_batch` |
| Batch physics | `lasx_j2_accel_batch`, `lasx_rk4_j2_step_batch`, `lasx_ballistic_step` |
| Batch attitude (7) | `lasx_cross3_batch`, `lasx_unitize3_batch`, `lasx_mat3_mul_vec3_batch`, `lasx_quat_normalize_batch`, `lasx_quat_mul_batch`, `lasx_quat_rotate_batch`, `lasx_quat_to_dcm_batch` |
| NN, N1 batch (8) | `lasx_softmax_rows`, `lasx_rms_norm`, `lasx_silu`, `lasx_gelu_quick`, `lasx_gelu_erf`, `lasx_rope`, `lasx_dot_f16`, `lasx_gemv_f16` |
| Memory | `lasx_alloc` (32-byte aligned; no matching deallocator is exported) |

C signatures, semantics, numerical contracts, and the "who enforces this" annotation for every
operator are in `docs/ops.md` (§4 symbol table, §2.x contracts, §5.x usage). The safe Rust API is
`lasx_rs::api` (32 `pub fn`), alongside `lasx_rs::view`, `lasx_rs::plan`, `lasx_rs::shape`,
`lasx_rs::pool`, and `lasx_rs::parallel`.

## Documentation

| Document | Contents |
|---|---|
| [docs/ops.md](docs/ops.md) | **Operators and usage**: the 59-symbol table, numerical contracts and bitwise determinism, degradation coverage, C/Rust/Dart invocation, pool and parallelism, NN operator status and gap comparison |
| [docs/dev.md](docs/dev.md) | **Architecture and performance**: layers and invariants, tests and CI, performance methodology, **all measured data (§7)**, matmul and parallel deep dives, rejected-approach list, reproduction steps, known gaps |

The division of labour between the two documents is a hard rule: **contracts and usage live in
`ops.md`, performance data lives in `dev.md §7`, and any given number appears exactly once**.
Cross-document references are always written as `docs/xxx.md §N`, which makes them mechanically
checkable.

## Repository

| Remote | URL |
|---|---|
| GitCode (primary) | `git@gitcode.com:H076lik/lasx_rs.git` / <https://gitcode.com/H076lik/lasx_rs> |
| GitHub (mirror) | `git@github.com:chzih076/lasx_rs.git` / <https://github.com/chzih076/lasx_rs> |
| LAN panel | `http://192.168.1.64/api/git/lasx_rs.git` (CI runs on the same panel) |

CI has four steps: `Format check (rustfmt)` → `Build release (nightly, +lasx)` → `Test release` →
`Clippy (zero warnings guard)`. **CI never runs benchmarks**: they are sensitive to background load
and are therefore not suitable as a gate.

## License

MIT © 2026 lik (H076lik). See [LICENSE](LICENSE).
