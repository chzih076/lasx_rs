//! FFI 侧的**运行时元信息**：能力查询与库版本。
//!
//! 这两个符号的存在本身就是对"心智模型"的修补：
//!
//! - 没有能力查询时，头文件只能写"调用方须自行读 `cpucfg` CFG2.bit7"——把安全责任
//!   推给调用方去读 CPU 寄存器，而在 LSX-only 机型上误调 `[LASX-only]` 符号是**SIGILL**；
//! - 没有版本查询时，调用方无法在运行时确认自己链接的是哪一版（`_checked` 的错误码
//!   含义也是跟着版本走的）。
//!
//! 两者都**永不返回 NULL**，返回的字符串生命周期 = 整个进程（静态数据，**不要 free**）。
//!
//! ```c
//! #include <lasx_rs.h>
//! #include <stdio.h>
//!
//! if (!lasx_has_lasx()) {                 /* ← 调 [LASX-only] 之前必须问 */
//!     fprintf(stderr, "本机无 LASX（%s）\n", lasx_version());
//!     return 1;
//! }
//! ```

use std::ffi::c_char;

/// 当前**生效路径**是否为 LASX：`1` = 是，`0` = 仅 LSX。
///
/// C 签名：`int lasx_has_lasx(void)`
///
/// # 语义（与算子完全一致）
///
/// 直接返回 `SimdPath::detect().is_lasx()` 的结果——也就是**算子入口此刻会走的路径**。
/// 所以它尊重 `lasx_force_lsx_thread`（该函数仅 Rust rlib 可见，见 `docs/ops.md` §3.4）：
/// 被强制降级时本函数返回 `0`，与算子的实际行为一致。调用方问的是"**现在能不能安全
/// 调 `[LASX-only]` 符号**"，答案必须与算子走的路一致，否则这个查询就没有意义。
///
/// # 什么时候必须调
///
/// **调用任何标了 `[LASX-only]` 的符号之前**（如 `lasx_dot`、`lasx_sum`、`lasx_matmul`、
/// `lasx_dot_i8`）。标了 `LASX+LSX` 的符号在两种机型上都安全，不需要问。
///
/// 代价：一次进程级原子读 + 一次线程局部读（实测边际 0.2–0.6 ns，见 `docs/dev.md` §7.14），
/// 所以放在"每个批次调一次"的位置即可，不必每个元素都问。
#[unsafe(no_mangle)]
pub extern "C" fn lasx_has_lasx() -> i32 {
    i32::from(crate::arch::SimdPath::detect().is_lasx())
}

/// 库版本（`CARGO_PKG_VERSION`），指向**静态 NUL 结尾字符串**。
///
/// C 签名：`const char *lasx_version(void)`
///
/// **永不为 NULL**，生命周期 = 进程（静态数据）；**不要 `free`**，也不要写它。
/// 返回值可以直接 `printf("%s")` / `strcmp`，**不需要**额外的长度参数。
#[unsafe(no_mangle)]
pub extern "C" fn lasx_version() -> *const c_char {
    // `concat!` 生成带结尾 NUL 的 `&'static str`；`as_ptr` 就是 C 串首地址。
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::{SimdPath, hardware, lasx_force_lsx_thread};
    use std::ffi::CStr;

    /// ② `lasx_has_lasx()` 必须与 `SimdPath::detect().is_lasx()` **逐一对应**，
    /// 并在强制降级下返回 0——这是给调用方的心智模型证据（"问了就不会 SIGILL"）。
    #[test]
    fn test_has_lasx_matches_detect_and_stays_nonzero_after_restore() {
        // ① 与算子同源：常态下两者必须一致
        assert_eq!(
            lasx_has_lasx(),
            i32::from(SimdPath::detect().is_lasx()),
            "lasx_has_lasx() 与算子路径不一致"
        );
        if hardware().lasx {
            assert_eq!(lasx_has_lasx(), 1, "含 LASX 的机型应报 1");
        }

        // ② 强制降级：必须报 0（否则调用方会去调 [LASX-only] → SIGILL）
        lasx_force_lsx_thread(true);
        assert_eq!(
            lasx_has_lasx(),
            0,
            "lasx_force_lsx_thread(true) 之后必须报 0"
        );
        assert_eq!(lasx_has_lasx(), i32::from(SimdPath::detect().is_lasx()));

        // ③ 复原（本测试线程的 TLS 必须置回，免得影响复用的测试线程）
        lasx_force_lsx_thread(false);
        assert_eq!(lasx_has_lasx(), i32::from(SimdPath::detect().is_lasx()));
    }

    /// ④ `lasx_version()` 非 NULL、NUL 结尾、且与 `CARGO_PKG_VERSION` 一致。
    #[test]
    fn test_version_is_static_nul_terminated_and_matches_cargo() {
        let p = lasx_version();
        assert!(!p.is_null(), "lasx_version() 不得返回 NULL");
        // SAFETY: 契约是"指向静态 NUL 结尾字符串"；非 NULL 已断言。
        let s = unsafe { CStr::from_ptr(p) };
        assert_eq!(s.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
        // 同一指针 = 静态数据（不是临时缓冲），也说明"不要 free"是对的
        assert_eq!(p, lasx_version());
    }
}
