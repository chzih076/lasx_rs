//! 唯一的构建脚本职责：给 `cdylib` 固定 **SONAME**（`liblasx_rs.so.0`）。
//!
//! # 为什么用 `build.rs` 而不是 `.cargo/config.toml`
//!
//! `cargo:rustc-cdylib-link-arg` **只作用于 cdylib 产物**：rlib / staticlib / 测试 / 基准都
//! 不受影响。而 `.cargo/config.toml` 里的 `rustflags` 会作用到**所有目标**（含依赖与测试），
//! 把链接器参数洒得到处都是。这里也**没有引入任何构建依赖**（只往 stdout 打 `cargo:`
//! 指令）——零第三方依赖是本库的卖点之一，打包环节不能破它。
//!
//! # SONAME 的取值与 ABI 承诺
//!
//! `liblasx_rs.so.0`：major = **0**，与 `Cargo.toml` 的版本策略一致（0.x 阶段不承诺二进制
//! 兼容）。本库对 C ABI 的**实际承诺**写在 `docs/ops.md` §4.1/§4.7：**符号只追加、不改
//! 签名与语义**（现有 83 个符号），所以 major 只在"不得不破坏 ABI"时才 +1——那时要同时改
//! 这里、`scripts/install.sh` 的符号链接与 `docs/ops.md` 的规则表。
//!
//! 版本化的**文件名**（`liblasx_rs.so.0.1.0`）不在这里做：`cargo` 的产物名固定是
//! `liblasx_rs.so`，安装脚本负责复制成版本化文件名并建 `.so.0` / `.so` 两个符号链接
//! （见 `scripts/install.sh`）。这样"构建产物"与"发行布局"各管一段，职责不重叠。
fn main() {
    println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,liblasx_rs.so.0");
    // 本文件自己变了要重跑（否则改了 SONAME 不会重新链接）。
    println!("cargo:rerun-if-changed=build.rs");
}
