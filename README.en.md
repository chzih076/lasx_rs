# lasx_rs

**lasx_rs is a vector library for the Loongson platform, providing access to
the LASX (256-bit) and LSX (128-bit) vector instruction sets of the
LoongArch architecture.**

Batch numeric kernels (dot product / axpy / norm / distance / gravity
acceleration, etc.). On CPUs without LASX (e.g. 3A5000 / 3A6000, LSX-only),
the library automatically falls back to the LSX path, bitwise deterministic.

## Features

- **LASX 256-bit**: `lasx_*` intrinsic batch kernels (axpy / dot / norm3 /
  distance / J2 acceleration / RK4 step)
- **LSX 128-bit fallback**: automatic 128-bit path on LSX-only CPUs
  (thread-level forced-fallback hook for verification / testing)
- **Zero dependencies**: pure std + `stdarch_loongarch` (nightly)
- **Bitwise deterministic**: vectorized results match scalar reference
  (guarded by regression tests)

## Repository

| Platform | URL |
|---|---|
| GitCode (primary) | `git@gitcode.com:H076lik/lasx_rs.git` / <https://gitcode.com/H076lik/lasx_rs> |
| GitHub (mirror) | `git@github.com:chzih076/lasx_rs.git` / <https://github.com/chzih076/lasx_rs> |

## Building

**Requires nightly Rust (with the experimental `stdarch_loongarch` feature)**:

```bash
# .cargo/config.toml already sets: rustflags = ["-C", "target-feature=+lsx,+lasx"]
cargo build --release
cargo test --release
```

> **Note**: this library depends on an **experimental rustc** (nightly +
> `#![feature(stdarch_loongarch)]`). Install a loongarch64 nightly toolchain,
> e.g. `rustup toolchain install nightly-loongarch64-unknown-linux-gnu`.

## Usage

```rust
// As a dependency (crate-type includes rlib + cdylib)
let out = lasx_rs::lasx_dot(&a, &b, n);
// Or via FFI: the cdylib exports C ABI symbols (extern "C" lasx_*)
```

Dart FFI example:

```dart
final lib = DynamicLibrary.open('liblasx_rs.so');
// bind lasx_dot / lasx_axpy / lasx_norm3_batch etc.
```

## Benchmarks (Loongson-3B6000, LA664)

| Kernel | Speedup |
|---|---|
| Batch dot product (n≥8) | 2.4-2.8× |
| Full-order gravity batch | 3.84× |
| 24-thread batch propagation | ~18× |

## License

MIT © 2026 lik (H076lik). See [LICENSE](LICENSE).
