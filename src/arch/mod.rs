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

/// 本机并行度（进程视角）：逻辑 CPU 与物理核。
///
/// 用途是**给调用方一个可移植的线程数依据**，不做任何机型假设：本库的目标是全部支持
/// LSX/LASX 的龙芯 CPU（3A5000/3A6000/3B6000…），所以线程数一律运行时探测。
///
/// | 字段 | 来源 | 说明 |
/// |---|---|---|
/// | [`logical`](Self::logical) | `std::thread::available_parallelism()` | **cpuset/cgroup 感知**（进程真正能用的逻辑 CPU 数） |
/// | [`physical`](Self::physical) | `/sys/devices/system/cpu/*/topology/` | 数唯一的 `(package, core)` 对（SMT 兄弟算一个核）；读不到 sysfs 时**退化为 `logical`** |
///
/// 两条实测过的坑：
///
/// - `available_parallelism()` 在本机是**系统调用，68.9 µs/次**（`docs/dev.md` §19.10），
///   所以这里用 `OnceLock` 缓存——**不要在热路径调用**（池只在构造时读一次）。
/// - sysfs 列出的是**整机**的 CPU，不认识 cpuset：所以 `physical` 与 `logical` 取小，
///   在受限容器里表现为"物理核数被逻辑预算截断"。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Parallelism {
    /// 进程可用的逻辑 CPU 数（cpuset/cgroup 感知）。
    pub logical: usize,
    /// 物理核数（SMT 兄弟算一个核）；探测不到时等于 `logical`。
    pub physical: usize,
}

/// 查询本机并行度（进程级缓存一次）。
pub fn parallelism() -> Parallelism {
    use std::sync::OnceLock;
    static CACHE: OnceLock<Parallelism> = OnceLock::new();
    *CACHE.get_or_init(|| {
        let logical = std::thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(1)
            .max(1);
        let physical = physical_cores_from_sysfs().map_or(logical, |p| p.clamp(1, logical));
        Parallelism { logical, physical }
    })
}

/// 从 sysfs 数物理核：唯一的 `(physical_package_id, core_id)` 组合数。
///
/// 读不到（非 Linux、容器里没挂 sysfs、格式变化）返回 `None` —— 调用方退化为 `logical`。
/// 只在首次调用时走一遍目录，之后由 [`parallelism`] 的缓存挡住。
#[cold]
fn physical_cores_from_sysfs() -> Option<usize> {
    let mut pairs = std::collections::BTreeSet::new();
    for entry in std::fs::read_dir("/sys/devices/system/cpu").ok()?.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        // 只认 `cpu<数字>`：跳过 `cpufreq`/`cpuidle` 之类的兄弟条目
        let Some(rest) = name.strip_prefix("cpu") else {
            continue;
        };
        if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        let base = entry.path().join("topology");
        let read = |f: &str| std::fs::read_to_string(base.join(f)).ok();
        let (Some(pkg), Some(core)) = (read("physical_package_id"), read("core_id")) else {
            continue;
        };
        let (Ok(pkg), Ok(core)) = (pkg.trim().parse::<u32>(), core.trim().parse::<u32>()) else {
            continue;
        };
        pairs.insert((pkg, core));
    }
    (!pairs.is_empty()).then_some(pairs.len())
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

    /// 并行度探测的可移植性契约：**只断言关系，不断言具体数字**（目标是全部 LSX/LASX 龙芯）。
    #[test]
    fn test_parallelism_is_sane_and_cached() {
        let p = parallelism();
        let again = parallelism();
        assert_eq!(p, again, "同一进程内必须稳定（OnceLock 缓存）");
        assert!(p.logical >= 1);
        assert!(p.physical >= 1);
        assert!(
            p.physical <= p.logical,
            "物理核数不能超过进程可用的逻辑 CPU 数（physical={} logical={}）",
            p.physical,
            p.logical
        );
        // 与 std 的探测口径一致（本机 available_parallelism 就是 logical）
        let std_logical = std::thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(1);
        assert_eq!(p.logical, std_logical);
    }
}
