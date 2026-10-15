# lasx_rs

A batch numerical kernel library for **LoongArch64 (Loongson-3B6000 / LA664, LASX 256-bit)**:
zero third-party dependencies, implemented in Rust, exporting a stable C ABI on top of a safe
Rust API, a resident worker pool, and shape/plan layers.

Four workload classes are covered:

| Class | Kernels |
|---|---|
| Reductions and dense linear algebra | `dot`, `sum`, `axpy`, `matmul` (f32 / f64) |
| Quantized dot products | int8, GGUF-style Q4 |
| int8 inference (N3 batch) | quantization/dequantization producers (`amax`, per-tensor/per-row `quantize_i8`, `dequantize_i8`) plus `gemv_i8`/`matmul_i8`. **Measured**: for batched shapes f32 GEMM is actually faster — decision rule in [docs/ops.md](docs/ops.md) |
| Batch physics and attitude/geometry | J2 gravity acceleration, RK4 step, ballistic step, 3-component norm, quaternion rotation/DCM, … |
| NN operators (N1 batch) | row-wise softmax, RMSNorm, SiLU, GELU (sigmoid approximation and erf form), RoPE, f16-weight GEMV |

## Features

- **Zero third-party dependencies** — only `core::arch::loongarch64` LASX/LSX intrinsics and std.
- **94 exported symbols** (50 plain + 44 `_checked` variants taking a trailing `int *status`).
  The signatures and semantics of the original 15 `lasx_*` symbols are **frozen**; new capability
  is only ever appended. Both symbol families are backed by the same kernel implementations.
- **Bitwise determinism** — for a given input, every path of an operator (LASX / LSX / scalar tail
  / packed / k-chunked / parallel blocks) produces results that are **identical bit for bit**
  (verified with `to_bits()` comparisons). There are exactly two documented exceptions, each with
  tests: `lasx_ballistic_step` uses different association orders in its vector and scalar branches,
  and `lasx_axpy` permits a difference of at most 1 ulp.
- **Matrix multiplication** — three bitwise-identical paths (packed, column-block, streaming) with
  packing and k-chunking; single-threaded f32 512³ reaches 61.5 GFLOP/s, which is **88%** of this
  machine's microkernel ceiling (2.00 FMA/cycle with A-broadcast, i.e. 70.3 GFLOP/s), and 95% at
  64³. Same-day cross-section against OpenBLAS 0.3.34: `docs/dev.md`.
- **Resident worker pool** — amortizes dispatch across calls: 8.62× on a 12-thread n = 2^18 step,
  and in a multi-satellite multi-step propagation it is **2.24×** faster than creating threads per
  step and **8.08×** faster than single-threaded.
- **Interface layers** — `view::MatRef`/`MatMut` turn shape and stride into objects validated once
  at construction; `plan::MatmulPlan` packs `B` once and reuses it (read-only, `Arc`-shareable);
  the `shape` layer encodes shapes in the type system with compile-time checks;
  `lasx_rs_macros` provides a formula DSL (`matmul!` for matrix products, `dot!` for dot
  products, `gemv!` for matrix-vector products).
- **NN operators are LASX-only by design** — the N1 batch deliberately omits an LSX fallback,
  because the target platforms (**LA664** cores: 3A6000 / 3C6000 / 3B6000) all support LASX.
  Note that **3B6000M** carries a 6000-series name but is built on the **LA364E** core with LSX
  only, and is therefore *not* a target. On such LSX-only CPUs these kernels execute LASX
  instructions and raise SIGILL; callers must dispatch by CPU capability.

## Getting started

```bash
# Requires nightly (#![feature(stdarch_loongarch)]) and a LoongArch machine
cargo build --release                    # produces liblasx_rs.so + rlib
cargo test --workspace --release         # 267 unit tests + 6 macro tests + 18 doc tests (2 ignored)
cargo clippy --workspace --all-targets -- -D warnings   # zero warnings is a hard gate
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p lasx_rs

# C projects: compile/link flags after install (see "Getting started" in docs/c-abi.md)
pkg-config --cflags --libs lasx_rs

# Benchmarks: no filter runs every suite, a substring selects matching suites
cargo run -p lasx_bench --release
cargo run -p lasx_bench --release -- softmax

# Single-shape A/B (the entry point for process-level alternating measurement):
#   <m> <k> <n> <stream|packed|cols|packed64|stream64|pool> [threads] [strategy]
cargo run --release --example matmul_ab -- 512 512 512 packed
```

```c
/* Shortest path from C; full mental model (layers / raw vs _checked / threads / errors): docs/c-abi.md */
if (lasx_has_lasx()) { /* ask before calling [LASX-only]: 0 = LSX only, a direct call is SIGILL */ }
float d = lasx_dot_checked(a, b, n, &st);            /* _checked by default: on failure status only, output untouched */
if (st != LASX_OK) fprintf(stderr, "%s\n", lasx_status_message(st));   /* never hardcode 0..6 */
lasx_free(a, n);                                     /* pair with lasx_alloc; free() is not an ABI contract */
```

`.cargo/config.toml` sets `rustflags = ["-C", "target-feature=+lasx"]`, so the default artifacts
assume LASX support. Two measurement rules apply to the benchmarks: **single-threaded kernel tables
are measured pinned to one core, thread-related tables are measured unpinned**, in two separate
runs (rationale and impact: `docs/dev.md`).

## Measured performance

The figures below were re-measured on one Loongson-3B6000 on **2026-10-15** using the
**variance method**: **6 rounds per group** (first round discarded, median of the remaining 5),
process-level; single-threaded tables pinned, thread-related tables unpinned. Grading follows
`docs/dev.md` §6.4 (the desktop here idles at load ≈2.5–3, so nearly all rounds are grade B and
usable as official numbers). **Per-number ranges, per-round loadavg and the full conventions are in
[docs/dev.md](docs/dev.md) §7**, the single authoritative location for performance data;
comparisons of compile-time constants and thresholds (k-chunking, column-block width, scheduling)
live in the respective design sections.

| Item | Result |
|---|---|
| FMA throughput ceiling (16 independent chains, registers only) | **139.4 GFLOP/s** (3.96/cycle, range 130.9–140.1); LSX 69.6 (2.00×) |
| Microkernel ceiling (with A-broadcast) | **2.00 FMA/cycle** (f32 70.3, f64 35.2 GFLOP/s; from `docs/dev.md` §6.3's `kernel_probe`, not re-measured) |
| `lasx_matmul` f32 64³ / 256³ / 512³ | 66.9 / 62.7 / **61.5** GFLOP/s |
| `lasx_matmul_f64` 128³ / 512³ | 30.6 / 30.9 GFLOP/s |
| `lasx_matmul` via `parallel`, 12 threads | 5.89× / 5.51× at 128³ / 512³ (393.4 / 331.5 GFLOP/s) |
| Single-stream / two-stream / 12-thread read (32 MiB) | 10.7 / 9.6 / 18.5 GB/s |
| `lasx_softmax_rows` / `lasx_rms_norm` | 4.95–6.72 / 12.16–18.29 GB/s |
| `lasx_silu` / `lasx_gelu_quick` / `lasx_gelu_erf` | 3.88–9.18 / 4.54–8.61 / 3.59–4.93 GB/s |
| `lasx_rope` (NeoX, LASX path) | 33.3–71.7 GB/s |
| `lasx_dot_f16` / `lasx_gemv_f16` | 38.1–68.3 / 4.2–64.7 GB/s |
| Resident pool vs threads-per-step (reused across 200 steps) | 37.47 ms vs 83.81 ms (**2.24×**; single-threaded 302.69 ms) |

> **Two cross-library conclusions** (authoritative table, per-cell 5-round medians and ms ranges,
> per-round `loadavg`, A/B/C grading, the **prior-consistency check** and the old-vs-new disposition
> are all in `docs/dev.md` §7.10; both harnesses are in the repo
> (`scripts/bench_openblas.c` + `scripts/bench_openblas.sh`) ⇒ reproducible):
> **① Single-threaded f32: ahead at small shapes, slightly behind at large ones** — 64³
> **67.8 vs 53.8 GFLOP/s (+27%)**, 128³ **65.8 vs 57.6 (+14%)**, 256³ a tie (62.4 vs 61.7),
> **384³ 60.6 vs 64.0 (−5.2%)**, **512³ 60.9 vs 64.9 (−6.2%)**; wide/thin `256×64×4096`
> **61.7 vs 41.0 (+50%)**. **f64 is 4–15% behind overall** (64³ a tie; 256³/384³/512³ at
> −7.5% / −4.4% / −6.3%; **the weakest structural point is f64 128³: 23.7 vs 28.1 G, −15%**).
> **② Multi-threaded: with today's C ABI you cannot have both bit-exactness and scaling** —
> `192×768×3072` f32 @12 threads; the **C ABI does not export the thread pool**, so the caller must
> split: **for bit-exactness you can only split by row (`rows`)** — **10/10 shapes bit-for-bit
> identical** (including k=384/385/386 on both sides and non-divisible row blocks like m=7/13/100/1000)
> — at the cost of **almost no scaling** (14.46 ms, **1.08×**). **`cols` (pre-transposed) is faster
> (3.84 ms, 4.09×) but is not a bit-exact equivalent** — **6/10 shapes differ from the
> single-threaded result** (it decomposes `Cᵀ=Bᵀ·Aᵀ` into small calls, so a size change changes the
> in-library path) ⇒ **do not use `cols` where bit-exactness is required**.
> Root cause: with `rows` every thread re-packs the whole `B` (**113 MB per call**) while the scatter
> is only **9%** of the gap — what is missing is "**pack once + a shared panel**", which **today's
> C ABI cannot express**: the effective C-side tier is ≈ **236 G** (Rust pool **310 G**, OpenBLAS
> **2.46 ms / 368 G**). **The real C-side multi-core fix — a library-provided "pack once, reuse"
> plan/pool entry — is being implemented**, and will be the recommended path once it lands.
> **The older figure (Rust pool path) was ≈ a tie**: median **2.64 ms** (samples 2.42–3.79, range
> **1.57×**) vs OpenBLAS **2.63 ms** (2.599–2.656, range **2%**) — the earlier "we are ~10% slower"
> was withdrawn on that line, but **it is the Rust-pool convention and cannot be compared
> side-by-side with this round's C-ABI splits** (both lines are kept; disposition in `docs/dev.md` §7.10 ④).

> **Who we compare against (must read)**: the opponent is **OpenBLAS 0.3.34, `DYNAMIC_ARCH`,
> core `la464` (a LoongArch-optimised kernel, *not* the generic C fallback), `NO_AFFINITY`**;
> every harness run prints its full identity (`OBIDENT`) as the first line.
> **The official convention is now same-process call-level alternation** (2026-10-15; rationale: a
> SIMD library's performance *is* "how long a call takes on the same data inside one process";
> cross-process runs fold in process start-up, per-process pool creation, scheduling and other
> tenants). **Under it, 12 threads is now sealed at ≥300 GFLOP/s**: the `plan` path (caller-resident
> threads + dynamic claiming + short spin) measures **310.99 G median (2.9132 ms, range
> 2.8763–3.0358, grade-A window)**, vs OpenBLAS **357.22 G** in the same alternating run (13% apart).
> **Older cross-process figures (e.g. 274 G) are reference/legacy only** — do not draw conclusions
> from them.
> **The call model matters just as much**: the "tie / a few percent behind" figures above are for
> **"caller-driven (plan + the caller's own threads)"**. The **library-managed convenience path
> (`lasx_matmul_run_threaded`) is now ≈1.02–1.05× the manual optimum** (same-process alternating:
> **2.7038 ms / 335 G** with the pool cached inside the handle; creating the pool per call costs
> 3.0986 ms, so caching bought **+14.6%**). Its **new costs, please note**: **concurrent
> `run_threaded` on the *same* plan shares one pool ⇒ dispatch serialises** (use separate plans, or
> `parallel::*` with your own pool, for real parallelism), and **changing `threads` rebuilds the pool**.
> **All older figures are not citable**: the earlier "convenience path N× slower / we are ~10% slower"
> numbers were **cross-process / bad-window** readings (old values live in **git history**:
> `git log -p docs/dev.md`). Per the rule in `docs/dev.md` §6.4 ("**any 'N×' claim must be
> re-checked same-process**") they are **old ledger only**; this file and `docs/dev.md` keep
> **current conventions and conclusions only**.

> **Formula-DSL reuse note**: `matmul!`'s `mul_into` re-packs `B` on every call, so a layer loop
> that reuses the same weight should hoist the preparation out of the loop (256³ measured
> **143.7 µs → 96.5–99.1 µs**, on par with the direct pooled call). **The DSL can now reuse a plan
> directly** — an operand written *without* brackets is a `Prepared`: with
> `let p = w.prepare();` outside the loop, `matmul!(y[M, N] = x[M, K] * p);` generates
> `p.apply_into(&x, &mut y)`, bit-for-bit identical to the hand-written hoist (`to_bits()`
> comparison). The bracketed form `w[K, N]` is unchanged. Full usage: [docs/ops.md](docs/ops.md).

> How to read these numbers: the tables above are a cross-section of one continuous collection and
> are **only comparable within that run**. Cross-run comparisons require re-measurement; observed
> window-to-window variation reaches 40%, so "run before, run after" cannot establish whether a
> change is faster — use process-level alternation (`docs/dev.md`).

## Operator overview

| Class | Symbols |
|---|---|
| Reduction / dense | `lasx_dot` (f32, with LSX fallback), `lasx_sum`, `lasx_axpy`, `lasx_dot_f64`, `lasx_matmul`, `lasx_matmul_f64` |
| Quantized | `lasx_dot_i8` (int8), `lasx_dot_q4` (one scale per 32 bytes) |
| int8 inference, N3 (8) | `lasx_amax`, `lasx_absmax_rows`, `lasx_quantize_i8_per_tensor`, `lasx_quantize_i8_per_row`, `lasx_dequantize_i8`, `lasx_dequantize_i8_rows`, `lasx_gemv_i8`, `lasx_matmul_i8` |
| Batch geometry | `lasx_batch_distance2d`, `lasx_norm3_batch`, `lasx_vec3_add_scaled_batch` |
| Batch physics | `lasx_j2_accel_batch`, `lasx_rk4_j2_step_batch`, `lasx_ballistic_step` |
| Batch attitude (7) | `lasx_cross3_batch`, `lasx_unitize3_batch`, `lasx_mat3_mul_vec3_batch`, `lasx_quat_normalize_batch`, `lasx_quat_mul_batch`, `lasx_quat_rotate_batch`, `lasx_quat_to_dcm_batch` |
| NN, N1 batch (8) | `lasx_softmax_rows`, `lasx_rms_norm`, `lasx_silu`, `lasx_gelu_quick`, `lasx_gelu_erf`, `lasx_rope`, `lasx_dot_f16`, `lasx_gemv_f16` |
| Memory | `lasx_alloc` (32-byte aligned; no matching deallocator is exported) |

C signatures and the C-side calling contract (memory, safety model, capability probe) are in
[docs/c-abi.md](docs/c-abi.md); per-operator semantics, numerical contracts, and the
"who enforces this" annotation are in `docs/ops.md`. The safe Rust API is `lasx_rs::api`
(32 `pub fn`), alongside `lasx_rs::view`, `lasx_rs::plan`, `lasx_rs::shape`, `lasx_rs::pool`,
and `lasx_rs::parallel`.

## Documentation

| Document | Contents |
|---|---|
| [docs/c-abi.md](docs/c-abi.md) | **The C ABI (single authority)**: the full symbol list with prototypes, memory and lifetime, safety model (raw vs `_checked`), capability probe and degradation, threading model, ABI stability policy, and a getting-started section (four ways to link) |
| [docs/ops.md](docs/ops.md) | **Operators and usage (Rust side)**: numerical contracts and bitwise determinism, degradation coverage, pool and parallelism, matrix view/plan/shape layers, NN operator status and gap comparison, int8-vs-f32 decision rule; for the C ABI see `docs/c-abi.md` |
| [docs/dev.md](docs/dev.md) | **Architecture and performance**: layers and invariants, tests and CI, performance methodology, **all measured data**, matmul and parallel deep dives, rejected-approach list, reproduction steps, known gaps |
| [docs/platform.md](docs/platform.md) | **Platform constraints and open implementation**: the LA664 atomic lost-update erratum (hit points and mitigations) and why the library must emit `_db` itself, kernel-branch lifecycle, gaps implied by the ONNX contract (`LayerNormalization`/`Gather`), and promise boundaries still to be drawn |

The division of labour is a hard rule: **the C ABI lives in `docs/c-abi.md`** (the symbol list and
C-side contract are maintained only there), Rust contracts and usage live in `docs/ops.md`, and
performance data lives in `docs/dev.md` — **any given number appears exactly once**.
Cross-document references are always written as a document name plus a section number,
which makes them mechanically checkable.

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
