//! 架构抽象层：指令集能力探测 + 向量路径分派。
//!
//! 内核不再各自写 `if has_lasx() { .. } else { .. }`，而是统一调用
//! [`SimdPath::detect`] 解析一次当前线程该走的路径，再用 `match` 分派到对应实现。
//! `cpucfg` 结果进程级缓存；`lasx_force_lsx_thread` 的强制降级是**线程级**的。
//!
//! 各内核的降级覆盖面并不一致（例如 `lasx_sum` / `lasx_matmul*` 是 LASX-only，
//! `lasx_ballistic_step` 的降级是纯标量而非 LSX），具体见对应 `ops` 模块的文档。

pub mod lasx;
pub mod lsx;

use std::cell::Cell;

/// 当前线程应使用的向量路径。
///
/// 只有两个变体，因为本库只提供 256 位（LASX）与 128 位（LSX）两套实现；
/// 个别内核在 `Lsx` 分支上退化为纯标量，这在其模块文档中单独标注。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SimdPath {
    /// LASX 256 位（如 3B6000 / LA664）。
    Lasx,
    /// LSX 128 位（如 3A5000 / 3A6000 等 LSX-only 机器）。
    Lsx,
}

impl SimdPath {
    /// 解析当前线程应走的路径。
    ///
    /// 硬件能力由 `cpucfg` 探测并进程级缓存；`lasx_force_lsx_thread(true)`
    /// 会让本线程强制走 LSX（仅用于测试/验证，不改变硬件探测结果）。
    #[inline]
    pub fn detect() -> Self {
        if hardware().lasx && !forced_lsx() {
            SimdPath::Lasx
        } else {
            SimdPath::Lsx
        }
    }

    /// 是否为 256 位路径。内核里最常见的判断，写成一目了然的谓词。
    #[inline]
    pub fn is_lasx(self) -> bool {
        matches!(self, SimdPath::Lasx)
    }
}

/// 本机硬件能力（`cpucfg` 探测结果，只读一次并进程级缓存）。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct HwCaps {
    /// CPUCFG word 2 bit 7：LASX（256 位）。
    pub lasx: bool,
    /// CPUCFG word 2 bit 6：LSX（128 位）。
    pub lsx: bool,
}

static HW_CAPS: std::sync::OnceLock<HwCaps> = std::sync::OnceLock::new();

/// 查询本机硬件能力。
#[inline]
pub fn hardware() -> HwCaps {
    *HW_CAPS.get_or_init(|| {
        let mut cfg2: u32;
        // SAFETY: cpucfg 是只读的 CPU 能力查询指令，读取 word 2 无副作用。
        unsafe {
            std::arch::asm!("cpucfg {}, {}", out(reg) cfg2, in(reg) 2u32);
        }
        HwCaps {
            lasx: (cfg2 & (1 << 7)) != 0,
            lsx: (cfg2 & (1 << 6)) != 0,
        }
    })
}

thread_local! {
    /// 线程级强制 LSX 降级（验证/测试钩子）。
    ///
    /// 线程级而非进程级：旧机制 `LOONGSCI_FORCE_LSX` 用进程级 `OnceLock` 缓存，
    /// 并发测试时会让其他依赖 LASX 黄金值的测试随机走 LSX 路径而失败。
    static FORCE_LSX: Cell<bool> = const { Cell::new(false) };
}

#[inline]
fn forced_lsx() -> bool {
    FORCE_LSX.with(Cell::get)
}

/// 强制当前线程的向量内核走 LSX 降级路径。
///
/// 用于在含 LASX 的机器（如本项目的 3B6000）上**真实执行** LSX 分支来验证数值与性能，
/// 等价于模拟 LSX-only CPU——如 3A5000/3A6000。诚实标注：这不是无-LASX 真机。
///
/// 仅本线程生效；测试结束时应置回 `false`。
pub fn lasx_force_lsx_thread(force: bool) {
    FORCE_LSX.with(|c| c.set(force));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_is_lasx_without_force() {
        lasx_force_lsx_thread(false);
        // 本项目的 CI 只在含 LASX 的真机上跑测试；x86_64 交叉 check 不执行本测试。
        assert!(SimdPath::detect().is_lasx());
    }

    #[test]
    fn test_force_lsx_is_thread_local_and_reversible() {
        lasx_force_lsx_thread(true);
        assert_eq!(SimdPath::detect(), SimdPath::Lsx);
        lasx_force_lsx_thread(false);
        assert_eq!(SimdPath::detect(), SimdPath::Lasx);
    }

    #[test]
    fn test_hw_caps_reports_lsx_and_lasx() {
        let caps = hardware();
        assert!(caps.lsx, "所有 LoongArch 3A/3B 均含 LSX");
        assert!(caps.lasx, "本机为 3B6000，含 LASX");
    }
}
