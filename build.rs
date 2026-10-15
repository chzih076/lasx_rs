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
    // ---------------------------------------------------------------------
    // 架构防呆（第一道，也是**报错最少**的一道）：错架构在编译**之前**就停下。
    //
    // 为什么放在这里：`src/lib.rs` 里也有一条 `compile_error!`（同一口径），但
    // `compile_error!` **不会中止**这次编译——rustc 会继续把其余源码的错误一起报出来。
    // 实测在 x86_64 上不加这道检查时是 **35 条**"目标不支持该指令/找不到 intrinsic"，
    // 用户真正需要的那句话被淹在里面。构建脚本先退出，就只剩下面这一段人话。
    //
    // 判定用的是 `CARGO_CFG_TARGET_ARCH`（**目标**架构，不是宿主）：交叉编译到
    // loongarch64 时它是 `loongarch64`，照常通过（GitHub 的 x86 runner 就是这么跑
    // `cargo check --target loongarch64` 的）。
    // ---------------------------------------------------------------------
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_arch != "loongarch64" {
        eprintln!(
            "\n\
             lasx_rs 是 LoongArch LASX/LSX 指令集绑定实现（手写内联汇编/内在函数），\n\
             不是可移植标量库。\n\
             \n\
             当前构建目标是 `{target_arch}`，不是 LoongArch64，已停止。\n\
             请在本机 LoongArch64 上构建，或改用其它后端。\n\
             \n\
             依赖方请写成：\n\
             \x20   [target.'cfg(target_arch = \"loongarch64\")'.dependencies]\n\
             \x20   lasx_rs = \"0.1\"\n\
             否则 x86/arm 上的 CI 会停在这里（这是有意的：把\"用错平台\"从运行期静默错结果\n\
             收敛成编译期硬错误）。\n\
             \n\
             注意区分另一件事：LSX-only 机型（**无 LASX**，如 2K3000/3B6000M）是**能构建、\n\
             能运行**的，只是不能调 [LASX-only] 符号（会 SIGILL）——那用 C 侧\n\
             `lasx_has_lasx()` / Rust 侧 `SimdPath::detect()` 防呆，与本检查无关。\n"
        );
        std::process::exit(1);
    }

    println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,liblasx_rs.so.0");
    // 本文件自己变了要重跑（否则改了 SONAME 不会重新链接）。
    println!("cargo:rerun-if-changed=build.rs");
}
