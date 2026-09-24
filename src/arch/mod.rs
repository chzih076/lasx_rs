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

impl HwCaps {
    /// 打包成原子字节（bit0=LASX、bit1=LSX、bit7=已探测）。
    #[inline]
    fn to_bits(self) -> u8 {
        (if self.lasx { BIT_LASX } else { 0 }) | (if self.lsx { BIT_LSX } else { 0 }) | BIT_DONE
    }

    #[inline]
    fn from_bits(bits: u8) -> Self {
        HwCaps {
            lasx: bits & BIT_LASX != 0,
            lsx: bits & BIT_LSX != 0,
        }
    }
}

const BIT_LASX: u8 = 0x01;
const BIT_LSX: u8 = 0x02;
/// 探测完成的哨兵位。有了它，"尚未探测"（0）与"探测结果是两者皆无"（仅 bit7）可区分。
const BIT_DONE: u8 = 0x80;

/// 能力探测结果打包进**一个原子字节**，写一次、之后只读。
///
/// 为什么不用 `OnceLock`：实测 `OnceLock::get_or_init` 每次取值约 **5 ns**，
/// 而每次内核调用都要过这里——对 `lasx_dot` n=24（约 14 ns）这种小调用，
/// 它一家就占了约 1/3。这里快路径只剩一条 `Relaxed` 原子读（约 0.5 ns）。
///
/// 发布用一次 `compare_exchange` 而非自旋：同一台机器上探测结果确定，
/// 并发首调时谁先发布都一样，失败方直接采用已发布的值即可。
static HW_BITS: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// 查询本机硬件能力。
///
/// 快路径（已探测）是一条 `Relaxed` 原子读；未探测时走 `probe_hw_caps`。
#[inline]
pub fn hardware() -> HwCaps {
    use std::sync::atomic::Ordering;
    let bits = HW_BITS.load(Ordering::Relaxed);
    if bits & BIT_DONE != 0 {
        return HwCaps::from_bits(bits);
    }
    probe_hw_caps()
}

/// 首次调用时才走这里（`#[cold]`，不污染热路径的代码布局）。
#[cold]
fn probe_hw_caps() -> HwCaps {
    use std::sync::atomic::Ordering;
    let mut cfg2: u32;
    // SAFETY: cpucfg 是只读的 CPU 能力查询指令，读取 word 2 无副作用。
    unsafe {
        std::arch::asm!("cpucfg {}, {}", out(reg) cfg2, in(reg) 2u32);
    }
    let caps = HwCaps {
        lasx: (cfg2 & (1 << 7)) != 0,
        lsx: (cfg2 & (1 << 6)) != 0,
    };
    let want = caps.to_bits();
    match HW_BITS.compare_exchange(0, want, Ordering::Release, Ordering::Relaxed) {
        Ok(_) => caps,
        // 别的线程先发布了：结果与本次一致，直接用它的
        Err(published) => HwCaps::from_bits(published),
    }
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
