//! 真实调用场景：基准同时也是"调用方怎么写"的示例。
//!
//! 前面几组套件测的是"一个内核调一次、缓冲预先备好"的理想情形。但调用方实际是
//! **多步、成串、复用缓冲、跨调用复用线程**的，而且**调用策略本身**会显著影响
//! 端到端表现（若把分配/克隆放进热路径，测出来的就不是内核了）。
//!
//! 这一组把三种真实形态量出来：
//!
//! 1. [`propagate`]：多星 × 多步轨道传播（对应 loong-sci `propagate_orbits_batch`）；
//! 2. [`hotloop`]：小数组高频调用——暴露"每次调用的固定开销"占比；
//! 3. [`fused_vs_composed`]：用融合内核 vs 用原语拼出同一个 RK4 步。

use crate::data::{states, velocities, AlignedBuf, Soa6, J2, MU, RE};
use crate::suites::micro::{rk4_parallel_scope, SpinPool};
use crate::timing::{fmt_t, timeit};
use lasx_rs::*;
use std::hint::black_box;

/// 场景 1：多星 × 多步轨道传播。
///
/// 关键对比是**线程池跨步复用**：单步基准里建池成本要靠重复取样摊掉，
/// 而真实传播一跑几百步，池只建一次——这才是常驻池的用法。
pub fn propagate() {
    let n = 1 << 16;
    let steps = 200usize;
    let threads = 12usize;
    let (rx, ry, rz) = states(n);
    let (vx, vy, vz) = velocities(n);
    let base = Soa6::new(rx, ry, rz, vx, vy, vz);

    let mut b1 = base.clone();
    let t_single = timeit(|| {
        for _ in 0..steps {
            b1.step(false);
        }
        let _ = black_box(b1.rx[0]);
    });

    let mut b2 = base.clone();
    let t_scope = timeit(|| {
        for _ in 0..steps {
            rk4_parallel_scope(&mut b2, threads);
        }
        let _ = black_box(b2.rx[0]);
    });

    // 池只建一次，跨 steps 步复用
    let pool = SpinPool::new(threads);
    let mut b3 = base.clone();
    let t_pool = timeit(|| {
        for _ in 0..steps {
            pool.run(&mut b3);
        }
        let _ = black_box(b3.rx[0]);
    });

    let work = n as f64 * steps as f64;
    println!();
    println!("## 真实场景 1：多星 × 多步轨道传播（n = {n} 星 × {steps} 步，dt = 10 s）");
    println!();
    println!("| 推进方式 | {steps} 步总时间 | 每步 | 吞吐（样本·步/秒） | 相对单线程 |");
    println!("|---|---|---|---|---|");
    for (name, t) in [
        ("单线程", t_single),
        ("12 线程 · 每步新建线程", t_scope),
        ("12 线程 · 常驻自旋池（跨步复用）", t_pool),
    ] {
        let per_step = t.as_secs_f64() / steps as f64;
        println!(
            "| {name} | {} | {} | {:.1} M/s | {:.2}× |",
            fmt_t(t),
            fmt_t(std::time::Duration::from_secs_f64(per_step)),
            work / t.as_secs_f64() / 1e6,
            t_single.as_secs_f64() / t.as_secs_f64()
        );
    }
    println!();
    println!(
        "> 池建一次、跨 {steps} 步复用后，派活开销被 {steps} 步摊薄；\
         每步新建线程则每步都要付约 0.5 ms。"
    );
}

/// 场景 2：小数组高频调用——每次调用的固定开销（FFI 边界 + 能力探测 + 切片构造）
/// 在小规模上占比最大，这是调用方最容易吃亏的地方。
pub fn hotloop() {
    const CALLS: usize = 200_000;
    println!();
    println!("## 真实场景 2：小数组高频调用（每种规模调用 {CALLS} 次，取每次耗时）");
    println!();
    println!("| 内核 | n | 每次调用 | 吞吐 | 说明 |");
    println!("|---|---|---|---|---|");
    for &(name, n) in &[
        ("lasx_dot f32", 8usize),
        ("lasx_dot f32", 16),
        ("lasx_dot f32", 64),
        ("lasx_dot f32", 256),
    ] {
        let mut rng = crate::data::Lcg::new(0x4004 ^ n as u64);
        let a = AlignedBuf::fill_with(n, |_| rng.f32());
        let b = AlignedBuf::fill_with(n, |_| rng.f32());
        let t = timeit(|| {
            for _ in 0..64 {
                let _ = black_box(lasx_dot(a.as_ptr(), b.as_ptr(), n as i32));
            }
        }) / 64;
        let note = if n < 24 {
            "走标量快路径（无能力探测）"
        } else {
            "走 LASX 路径（每次都要过一次分派）"
        };
        println!(
            "| {name} | {n} | {} | {:.2} G elem/s | {note} |",
            fmt_t(t),
            n as f64 / t.as_secs_f64() / 1e9
        );
    }
    println!();
    println!(
        "> 对照 `dispatch` 套件：`SimdPath::detect()` 约 0.9 ns/次。\
         规模越小，这部分固定开销占比越高。"
    );
}

/// 场景 3：融合内核 vs 原语拼接。
///
/// 本库只导出了"加/缩放/加速度/点积"这些原语。调用方若想自己拼一个 RK4 步，
/// 至少要 18 次 FFI 调用，每次都要把 6~9 个 f64 数组过一遍内存；而
/// `lasx_rk4_j2_step_batch` 把这些全放在寄存器里完成。这一组把代价量出来。
pub fn fused_vs_composed() {
    let n = 1 << 16;
    let steps = 20usize;
    let h = 10.0f64;
    let (rx, ry, rz) = states(n);
    let (vx, vy, vz) = velocities(n);
    let base = Soa6::new(rx, ry, rz, vx, vy, vz);

    // 融合：一步一次 FFI
    let mut fused = base.clone();
    let t_fused = timeit(|| {
        for _ in 0..steps {
            fused.step(false);
        }
        let _ = black_box(fused.rx[0]);
    });

    // 拼接：用公开原语手写同一套 RK4 公式
    let mut comp = Composed::new(&base);
    let t_comp = timeit(|| {
        for _ in 0..steps {
            comp.step(h);
        }
        let _ = black_box(comp.probe());
    });

    let work = n as f64 * steps as f64;
    println!();
    println!("## 真实场景 3：同一个 RK4 步，「融合内核」vs「拼原语」（n = {n}，{steps} 步）");
    println!();
    println!("| 调用策略 | FFI 调用/步 | {steps} 步总时间 | 每步 | 吞吐（样本·步/秒） |");
    println!("|---|---|---|---|---|");
    println!(
        "| `lasx_rk4_j2_step_batch`（融合） | **1** | {} | {} | {:.1} M/s |",
        fmt_t(t_fused),
        fmt_t(std::time::Duration::from_secs_f64(
            t_fused.as_secs_f64() / steps as f64
        )),
        work / t_fused.as_secs_f64() / 1e6
    );
    println!(
        "| 拼 18 次原语（j2_accel + vec3_add_scaled） | 18 | {} | {} | {:.1} M/s |",
        fmt_t(t_comp),
        fmt_t(std::time::Duration::from_secs_f64(
            t_comp.as_secs_f64() / steps as f64
        )),
        work / t_comp.as_secs_f64() / 1e6
    );
    println!();
    println!(
        "> 拼接版慢 {:.1}×：除 18 次调用开销外，每次调用都要把 6~9 个 f64 数组\
         写回内存再读出来，而这些中间量在融合版里从未离开寄存器。",
        t_comp.as_secs_f64() / t_fused.as_secs_f64()
    );
}

/// 用公开原语拼出来的 RK4 单步（场景 3 的对照实现）。
///
/// 槽位在构造时一次性分配、跨步复用——**中间量绝不能每步现分配**，否则测到的
/// 是分配器而不是调用策略（这正是 rk4 标量基线踩过的坑）。
///
/// 组合顺序与融合内核不完全一致（原语只有 `a + s·b`，凑 `v + 2v2 + 2v3 + v4`
/// 要拆三步），故这是**性能对照**而非逐位对照。
struct Composed {
    /// 每槽 3 个三分量缓冲：0=r 1=v 2=k1 3=r2 4=v2 5=k2 6=r3 7=v3 8=k3
    /// 9=r4 10=v4 11=k4 12=t1 13=t2 14=lsum 15=u1 16=u2 17=ksum
    bufs: Vec<AlignedBuf<f64>>,
    ns: usize,
}

const SLOT_R: usize = 0;
const SLOT_V: usize = 1;
const SLOT_K1: usize = 2;
const SLOT_R2: usize = 3;
const SLOT_V2: usize = 4;
const SLOT_K2: usize = 5;
const SLOT_R3: usize = 6;
const SLOT_V3: usize = 7;
const SLOT_K3: usize = 8;
const SLOT_R4: usize = 9;
const SLOT_V4: usize = 10;
const SLOT_K4: usize = 11;
const SLOT_T1: usize = 12;
const SLOT_T2: usize = 13;
const SLOT_LSUM: usize = 14;
const SLOT_U1: usize = 15;
const SLOT_U2: usize = 16;
const SLOT_KSUM: usize = 17;
const SLOTS: usize = 18;

impl Composed {
    fn new(base: &Soa6) -> Self {
        let n = base.len();
        let mut bufs: Vec<AlignedBuf<f64>> = (0..SLOTS * 3).map(|_| AlignedBuf::new(n)).collect();
        // 状态槽写入初值
        bufs[SLOT_R * 3].as_mut_slice().copy_from_slice(&base.rx);
        bufs[SLOT_R * 3 + 1]
            .as_mut_slice()
            .copy_from_slice(&base.ry);
        bufs[SLOT_R * 3 + 2]
            .as_mut_slice()
            .copy_from_slice(&base.rz);
        bufs[SLOT_V * 3].as_mut_slice().copy_from_slice(&base.vx);
        bufs[SLOT_V * 3 + 1]
            .as_mut_slice()
            .copy_from_slice(&base.vy);
        bufs[SLOT_V * 3 + 2]
            .as_mut_slice()
            .copy_from_slice(&base.vz);
        Composed { bufs, ns: n }
    }

    /// 某槽位三分量的裸指针（槽位构造后不再增删，指针稳定）。
    fn p(&mut self, slot: usize) -> [*mut f64; 3] {
        let i = slot * 3;
        [
            self.bufs[i].as_mut_ptr(),
            self.bufs[i + 1].as_mut_ptr(),
            self.bufs[i + 2].as_mut_ptr(),
        ]
    }

    /// `dst = a + s·b`（一次公开原语调用，支持原地）
    fn add_scaled(&mut self, dst: usize, a: usize, b: usize, s: f64) {
        let d = self.p(dst);
        let x = self.p(a);
        let y = self.p(b);
        let n = self.ns as i32;
        lasx_vec3_add_scaled_batch(x[0], x[1], x[2], y[0], y[1], y[2], s, d[0], d[1], d[2], n);
    }

    /// `out = a(r)`
    fn j2(&mut self, r: usize, out: usize) {
        let rr = self.p(r);
        let o = self.p(out);
        let n = self.ns as i32;
        lasx_j2_accel_batch(rr[0], rr[1], rr[2], MU, J2, RE, o[0], o[1], o[2], n);
    }

    /// 一步 RK4：4 次加速度 + 14 次缩放加 = **18 次 FFI 调用**
    fn step(&mut self, h: f64) {
        self.j2(SLOT_R, SLOT_K1); // k1 = a(r)
        self.add_scaled(SLOT_R2, SLOT_R, SLOT_V, h / 2.0); // r2 = r + h/2·v
        self.add_scaled(SLOT_V2, SLOT_V, SLOT_K1, h / 2.0); // v2 = v + h/2·k1
        self.j2(SLOT_R2, SLOT_K2); // k2 = a(r2)
        self.add_scaled(SLOT_R3, SLOT_R, SLOT_V2, h / 2.0); // r3 = r + h/2·v2
        self.add_scaled(SLOT_V3, SLOT_V, SLOT_K2, h / 2.0); // v3 = v + h/2·k2
        self.j2(SLOT_R3, SLOT_K3); // k3 = a(r3)
        self.add_scaled(SLOT_R4, SLOT_R, SLOT_V3, h); // r4 = r + h·v3
        self.add_scaled(SLOT_V4, SLOT_V, SLOT_K3, h); // v4 = v + h·k3
        self.j2(SLOT_R4, SLOT_K4); // k4 = a(r4)

        // lsum = v + 2v2 + 2v3 + v4 → (v + 2v2) + 2(v3 + 0.5v4)
        self.add_scaled(SLOT_T1, SLOT_V, SLOT_V2, 2.0);
        self.add_scaled(SLOT_T2, SLOT_V3, SLOT_V4, 0.5);
        self.add_scaled(SLOT_LSUM, SLOT_T1, SLOT_T2, 2.0);

        // ksum = k1 + 2k2 + 2k3 + k4 → (k1 + 2k2) + 2(k3 + 0.5k4)
        self.add_scaled(SLOT_U1, SLOT_K1, SLOT_K2, 2.0);
        self.add_scaled(SLOT_U2, SLOT_K3, SLOT_K4, 0.5);
        self.add_scaled(SLOT_KSUM, SLOT_U1, SLOT_U2, 2.0);

        // 原地更新状态
        self.add_scaled(SLOT_R, SLOT_R, SLOT_LSUM, h / 6.0);
        self.add_scaled(SLOT_V, SLOT_V, SLOT_KSUM, h / 6.0);
    }

    /// 便于 bench 里 black_box 取一个值。
    fn probe(&self) -> f64 {
        self.bufs[SLOT_R * 3][0]
    }
}

/// 依次跑三个真实场景（各自打印自己的表）。
pub fn run_all() {
    propagate();
    hotloop();
    fused_vs_composed();
}
