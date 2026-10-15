//! 指令级与调用级微基准：LSX/LASX 的**延迟/吞吐**、`SimdPath::detect()` 的**每调用成本**、
//! 以及极小 n 的 `dot` **同口径四路对照**。
//!
//! 为什么需要它（2026-10-14）：优化 `dot` 小 n 时只能拿"每调用时间"当证据，而那个口径把
//! **分派与脚手架**也算在内，导致反复得出相互矛盾的结论（详见 `src/ops/dot.rs` 的注释）。
//! 本套件把三个量分开测：
//! * **延迟** = 一条依赖链的长度 N 的耗时 / N（8 条独立链则量吞吐）；
//! * **周期换算**用本机实测主频（`/proc/cpuinfo`：Loongson-3B6000 **2.2 GHz**）；
//! * **同口径**：`dot` 的几种实现各在同一个紧循环里调用 `ITERS` 次 ⇒ 调用开销相同，
//!   差异即内核差异。
use crate::report::Row;
use lasx_rs::arch::SimdPath;
use lasx_rs::{lasx_cross3_batch, lasx_norm3_batch};
use std::arch::loongarch64::*;
use std::hint::black_box;
use std::mem::transmute;
use std::time::Instant;

/// 本机实测主频（MHz）。`/proc/cpuinfo` 报 `CPU MHz: 2200.00`，且无 cpufreq sysfs ⇒ 固定主频。
const MHZ: f64 = 2200.0;

const ITERS_LAT: usize = 20_000_000;
const ITERS_CALL: usize = 10_000_000;

#[inline(always)]
unsafe fn ld8(p: *const f32) -> m256 {
    unsafe { transmute(lasx_xvld(p as *const i8, 0)) }
}
#[inline(always)]
unsafe fn st8(p: *mut f32, v: m256) {
    unsafe { lasx_xvst(transmute::<m256, m256i>(v), p as *mut i8, 0) }
}
#[inline(always)]
unsafe fn ld4(p: *const f32) -> m128 {
    unsafe { transmute(lsx_vld(p as *const i8, 0)) }
}
#[inline(always)]
unsafe fn st4(p: *mut f32, v: m128) {
    unsafe { lsx_vst(transmute::<m128, m128i>(v), p as *mut i8, 0) }
}

/// 一条依赖链：延迟 = 总时间 / n。
#[inline(never)]
fn lasx_fma_lat(n: usize) -> f32 {
    let buf = [1.0f32; 8];
    let a = unsafe { ld8(buf.as_ptr()) };
    let mut c = a;
    for _ in 0..n {
        c = unsafe { lasx_xvfmadd_s(a, a, c) };
    }
    let mut t = [0f32; 8];
    unsafe { st8(t.as_mut_ptr(), c) };
    t[0]
}

/// 8 条独立链：吞吐 = 总时间 / (n·8)。
#[inline(never)]
fn lasx_fma_tp(n: usize) -> f32 {
    let buf = [1.0f32; 8];
    let a = unsafe { ld8(buf.as_ptr()) };
    let (mut c0, mut c1, mut c2, mut c3) = (a, a, a, a);
    let (mut c4, mut c5, mut c6, mut c7) = (a, a, a, a);
    for _ in 0..n {
        c0 = unsafe { lasx_xvfmadd_s(a, a, c0) };
        c1 = unsafe { lasx_xvfmadd_s(a, a, c1) };
        c2 = unsafe { lasx_xvfmadd_s(a, a, c2) };
        c3 = unsafe { lasx_xvfmadd_s(a, a, c3) };
        c4 = unsafe { lasx_xvfmadd_s(a, a, c4) };
        c5 = unsafe { lasx_xvfmadd_s(a, a, c5) };
        c6 = unsafe { lasx_xvfmadd_s(a, a, c6) };
        c7 = unsafe { lasx_xvfmadd_s(a, a, c7) };
    }
    let s = unsafe {
        let u = lasx_xvfadd_s(lasx_xvfadd_s(c0, c1), lasx_xvfadd_s(c2, c3));
        let v = lasx_xvfadd_s(lasx_xvfadd_s(c4, c5), lasx_xvfadd_s(c6, c7));
        lasx_xvfadd_s(u, v)
    };
    let mut t = [0f32; 8];
    unsafe { st8(t.as_mut_ptr(), s) };
    t[0]
}

#[inline(never)]
fn lsx_fma_lat(n: usize) -> f32 {
    let buf = [1.0f32; 4];
    let a = unsafe { ld4(buf.as_ptr()) };
    let mut c = a;
    for _ in 0..n {
        c = unsafe { lsx_vfmadd_s(a, a, c) };
    }
    let mut t = [0f32; 4];
    unsafe { st4(t.as_mut_ptr(), c) };
    t[0]
}

#[inline(never)]
fn lsx_fma_tp(n: usize) -> f32 {
    let buf = [1.0f32; 4];
    let a = unsafe { ld4(buf.as_ptr()) };
    let (mut c0, mut c1, mut c2, mut c3) = (a, a, a, a);
    let (mut c4, mut c5, mut c6, mut c7) = (a, a, a, a);
    for _ in 0..n {
        c0 = unsafe { lsx_vfmadd_s(a, a, c0) };
        c1 = unsafe { lsx_vfmadd_s(a, a, c1) };
        c2 = unsafe { lsx_vfmadd_s(a, a, c2) };
        c3 = unsafe { lsx_vfmadd_s(a, a, c3) };
        c4 = unsafe { lsx_vfmadd_s(a, a, c4) };
        c5 = unsafe { lsx_vfmadd_s(a, a, c5) };
        c6 = unsafe { lsx_vfmadd_s(a, a, c6) };
        c7 = unsafe { lsx_vfmadd_s(a, a, c7) };
    }
    let s = unsafe {
        let u = lsx_vfadd_s(lsx_vfadd_s(c0, c1), lsx_vfadd_s(c2, c3));
        let v = lsx_vfadd_s(lsx_vfadd_s(c4, c5), lsx_vfadd_s(c6, c7));
        lsx_vfadd_s(u, v)
    };
    let mut t = [0f32; 4];
    unsafe { st4(t.as_mut_ptr(), s) };
    t[0]
}

// ---- 极小 n 的 dot：四种实现，同口径对照（各 ITERS_CALL 次）----

/// 现状：单链 + 4 路展开（`src/ops/dot.rs` 的小 n 回退）。
#[inline(never)]
fn dot_ours(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut acc = 0f32;
    let mut i = 0;
    while i + 4 <= n {
        acc += a[i] * b[i];
        acc += a[i + 1] * b[i + 1];
        acc += a[i + 2] * b[i + 2];
        acc += a[i + 3] * b[i + 3];
        i += 4;
    }
    while i < n {
        acc += a[i] * b[i];
        i += 1;
    }
    acc
}

/// bench 的"标量"列写法：普通索引循环（`+lasx` 下 LLVM 会向量化）。
#[inline(never)]
fn dot_ref(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// 候选：`f32x8` 乘法 + 定序归约（n ≤ 8 一条向量；其后每 8 个一条）。
#[inline(never)]
fn dot_vec(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut t = [0f32; 8];
    let mut s = 0f32;
    let mut i = 0;
    while i + 8 <= n {
        unsafe {
            let va = ld8(a.as_ptr().add(i));
            let vb = ld8(b.as_ptr().add(i));
            st8(t.as_mut_ptr(), lasx_xvfmul_s(va, vb));
        }
        let mut k = 0;
        while k < 8 {
            s += t[k];
            k += 1;
        }
        i += 8;
    }
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

/// 候选 + 每调用一次分派（复现"crate 版没兑现探针收益"的原因）。
#[inline(never)]
fn dot_vec_dispatch(a: &[f32], b: &[f32]) -> f32 {
    if matches!(SimdPath::detect(), SimdPath::Lasx) {
        black_box(dot_vec(a, b))
    } else {
        dot_ref(a, b)
    }
}

fn call_ns(name: &str, n: usize, f: impl Fn(&[f32], &[f32]) -> f32) -> f64 {
    let a: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.0625).collect();
    let b: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.03125).collect();
    let mut acc = 0f32;
    for _ in 0..10_000 {
        acc += black_box(f(&a, &b));
    }
    let t = Instant::now();
    for _ in 0..ITERS_CALL {
        acc += black_box(f(&a, &b));
    }
    let ns = t.elapsed().as_secs_f64() * 1e9 / ITERS_CALL as f64;
    println!("| `dot` 同口径 | n={n:2} {name:12} | **{ns:6.2} ns/调用** | — | acc={acc:.1} |");
    ns
}

/// 空循环的"测量地板"：`iters` 次只做一件 `black_box` 累加，用来判断 ns/调用 里有多少
/// 是循环与外层记账本身。
fn loop_floor_ns(iters: usize) -> f64 {
    let mut acc = 0u32;
    let t = Instant::now();
    for _ in 0..iters {
        acc = acc.wrapping_add(black_box(1));
    }
    let ns = t.elapsed().as_secs_f64() * 1e9 / iters as f64;
    println!("LAT arm=loop_floor ns={ns:.3} iters={iters} acc={acc}");
    ns
}

/// **8 路展开的吞吐探针**：`WITH_DETECT = false` 是地板（每次异或一个常量），
/// `true` 时每次异或 `detect()` 的结果。8 条独立的 `xor` 链 + 8 次调用，循环本身的开销
/// 摊到 1/8，于是量的是 `detect()` 的**吞吐成本（µop 数）**，而不是外层循环的依赖链
/// ——朴素写法（`acc += black_box(detect())`）的依赖链约 1.37 ns/迭代，会把 1 ns 级的
/// 函数完全罩住（实测地板 1.367、detect 1.371–1.425，分辨不出来）。
#[inline(never)]
fn unrolled_probe<const WITH_DETECT: bool>(iters: usize) -> u64 {
    let mut acc = [0u64; 8];
    let rounds = iters / 8;
    for _ in 0..rounds {
        for (k, slot) in acc.iter_mut().enumerate() {
            let v = if WITH_DETECT {
                SimdPath::detect() as u64
            } else {
                k as u64
            };
            *slot ^= black_box(v);
        }
    }
    acc.iter().fold(0u64, |x, y| x ^ *y)
}

/// 极小形状算子的每调用时间。`n` 越小，固定开销（FFI + 切片构造 + `detect()`）占比越大。
fn tiny_batch_ns(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    out: &mut [f64],
    oz: &mut [f64],
    iters: usize,
) -> f64 {
    let n = out.len();
    let mut ox = vec![0f64; n];
    let mut oy = vec![0f64; n];
    let mut sink = 0f64;
    // norm3_batch：out = √(x²+y²+z²)
    for _ in 0..10_000 {
        lasx_norm3_batch(
            x.as_ptr(),
            y.as_ptr(),
            z.as_ptr(),
            out.as_mut_ptr(),
            n as i32,
        );
    }
    let t = Instant::now();
    for _ in 0..iters {
        lasx_norm3_batch(
            x.as_ptr(),
            y.as_ptr(),
            z.as_ptr(),
            out.as_mut_ptr(),
            n as i32,
        );
        sink += black_box(out[0]);
    }
    let norm3 = t.elapsed().as_secs_f64() * 1e9 / iters as f64;

    // cross3_batch：o = a × b（六个输入 + 三个输出）
    let t = Instant::now();
    for _ in 0..iters {
        lasx_cross3_batch(
            x.as_ptr(),
            y.as_ptr(),
            z.as_ptr(),
            y.as_ptr(),
            z.as_ptr(),
            x.as_ptr(),
            ox.as_mut_ptr(),
            oy.as_mut_ptr(),
            oz.as_mut_ptr(),
            n as i32,
        );
        sink += black_box(ox[0]);
    }
    let cross3 = t.elapsed().as_secs_f64() * 1e9 / iters as f64;
    println!("LAT arm=tiny_n{n} ns_norm3={norm3:.2} ns_cross3={cross3:.2} sink={sink:.1}");
    norm3
}

/// 8 路展开探针的四种"被异或的值从哪来"。**循环形状完全一致**（8 条独立 `xor` 链 +
/// 每次异或一次 `black_box`），只有这个值的来源不同 ⇒ 差值才是那一项的真实每调用成本。
const MODE_CONST: u8 = 0;
const MODE_HOISTED: u8 = 1;
const MODE_INLINE_DETECT: u8 = 2;
const MODE_INLINE_HARDWARE: u8 = 3;

/// 8 路展开 + 可插拔"每调用一次的动作"（`MODE_*`）。
///
/// 为什么不用"两个不同形状的循环"来比：§7.14 的教训——`vec+detect` 与 `vec` 各自
/// 循环的地板不同，差值会把地板差也算进去。这里四个 `MODE` 的循环体逐条对应，
/// 展开度、依赖链、`black_box` 位置都一样。
#[inline(never)]
fn probe8<const MODE: u8>(calls: usize) -> f64 {
    let hoisted = if MODE == MODE_HOISTED {
        SimdPath::detect() as u64
    } else {
        0
    };
    let mut acc = [0u64; 8];
    let rounds = calls / 8;
    let t = Instant::now();
    for _ in 0..rounds {
        for (k, slot) in acc.iter_mut().enumerate() {
            let v = match MODE {
                MODE_CONST => k as u64,
                MODE_HOISTED => hoisted,
                MODE_INLINE_DETECT => SimdPath::detect() as u64,
                MODE_INLINE_HARDWARE => lasx_rs::arch::hardware().lasx as u64,
                _ => 0,
            };
            *slot ^= black_box(v);
        }
    }
    let ns = t.elapsed().as_secs_f64() * 1e9 / calls as f64;
    black_box(acc.iter().fold(0u64, |x, y| x ^ y));
    ns
}

/// 候选 + 每次调用一次能力检查，但**检查值的来源由参数决定**。
///
/// 受控对照的关键：同一个函数体、同一个循环形状，只让"`flag` 从哪来"变化 ⇒
/// `dot_flagged(检测值)` 与 `dot_flagged(循环外算好的值)` 之差就是 §7.13 里
/// `vec+detect` − `vec` 那 2.8–4.2 ns 中**真正属于能力检查**的部分。
#[inline(never)]
fn dot_flagged(flag: bool, a: &[f32], b: &[f32]) -> f32 {
    if flag { dot_vec(a, b) } else { dot_ref(a, b) }
}

/// 通用计时：先预热再计时 `iters` 次，返回 ns/次。
fn timed_ns<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    for _ in 0..(iters / 10).max(1000) {
        f();
    }
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed().as_secs_f64() * 1e9 / iters as f64
}

/// 每调用固定开销的**受控拆解**（`lasx_bench lat callcost`）。
///
/// 方法论（§7.14 的教训）：要断言"差值 = 某项的真实每调用成本"，必须
/// ① **同一个二进制、同一个循环形状**（同一展开度、同一依赖链），只让被测量的那一项变化；
/// ② 差值要显著大于地板噪声。
///
/// 三个条目各给**地板 / 含该项 / 边际**三行，`ROUNDS` 轮进程内交替、每轮记 `loadavg`，
/// 最后按"丢首轮 + 中位数 + 极差"汇总（原始逐轮值也都打印，便于独立复核）。
///
/// - **A 能力检查**（`SimdPath::detect()`，14 个算子入口各一次）：8 路展开探针，
///   `const`（地板）/ `hoisted`（循环外算一次）/ `inline`（循环内每次算）/
///   `hardware`（只读原子字节，用来把 detect 拆成"原子读 + TLS 读"）。
/// - **B `_checked` vs 裸入口**（同一个算子、n=64）：校验 + 状态写回的净成本。
/// - **C 分配 vs 复用**：`AlignedVec::new` 本身、`api` 分配型入口 vs 复用入口、
///   `shape` 层 `Mat::mul` vs `Mat::mul_into`。
pub fn callcost() {
    use lasx_rs::aligned::AlignedVec;
    use lasx_rs::shape::{Mat, MatBuf};

    const ROUNDS: usize = 5;
    const P_CALLS: usize = 8_000_000; // 8 路展开探针的"调用次数"
    const C_ITERS: usize = 1_000_000;
    const N64: usize = 64;

    let x: Vec<f64> = (0..N64).map(|i| 1.0 + i as f64 * 0.25).collect();
    let y: Vec<f64> = (0..N64).map(|i| 2.0 + i as f64 * 0.5).collect();
    let z: Vec<f64> = (0..N64).map(|i| 3.0 + i as f64 * 0.75).collect();
    let mut out = vec![0f64; N64];
    let mut sink = 0f64;

    // C 的 shape 层对照（32³ matmul，同一份重量/输入）
    let w_data: Vec<f32> = (0..32 * 32).map(|i| (i % 17) as f32 * 0.125).collect();
    let x_data: Vec<f32> = (0..32 * 32).map(|i| (i % 11) as f32 * 0.0625).collect();
    let w = Mat::<f32, 32, 32>::new(&w_data).unwrap();
    let xm = Mat::<f32, 32, 32>::new(&x_data).unwrap();
    let mut outm = MatBuf::<f32, 32, 32>::new();

    let mut raw: Vec<(String, Vec<f64>)> = Vec::new();
    let push = |name: &str, v: f64, store: &mut Vec<(String, Vec<f64>)>| match store
        .iter_mut()
        .find(|(k, _)| k == name)
    {
        Some((_, vs)) => vs.push(v),
        None => store.push((name.to_string(), vec![v])),
    };

    for r in 1..=ROUNDS {
        let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
        let load = load.split_whitespace().next().unwrap_or("?").to_string();

        // ---- A：能力检查 ----
        let a_floor = probe8::<MODE_CONST>(P_CALLS);
        let a_hoisted = probe8::<MODE_HOISTED>(P_CALLS);
        let a_inline = probe8::<MODE_INLINE_DETECT>(P_CALLS);
        let a_hw = probe8::<MODE_INLINE_HARDWARE>(P_CALLS);
        for (k, v) in [
            ("A_floor", a_floor),
            ("A_hoisted", a_hoisted),
            ("A_inline", a_inline),
            ("A_hardware", a_hw),
        ] {
            println!("CALL r={r} load={load} item=A variant={k} ns={v:.3} calls={P_CALLS}");
            push(k, v, &mut raw);
        }
        let a_marg = a_inline - a_hoisted;
        println!("CALL r={r} load={load} item=A variant=A_marginal ns={a_marg:.3} calls={P_CALLS}");
        push("A_marginal", a_marg, &mut raw);

        // ---- A2：§7.13 的 `vec+detect` − `vec` = 2.8–4.2 ns 到底有多少是"能力检查" ----
        // 同一个 `dot_flagged` 函数体，只换 flag 的来源：循环外算好的值 vs 循环内每次 detect()。
        for dn in [8usize, 24] {
            let da: Vec<f32> = (0..dn).map(|i| 1.0 + i as f32 * 0.0625).collect();
            let db: Vec<f32> = (0..dn).map(|i| 0.5 + i as f32 * 0.03125).collect();
            let d_base = timed_ns(C_ITERS, || {
                sink += black_box(dot_vec(&da, &db)) as f64;
            });
            let d_hoist = timed_ns(C_ITERS, || {
                sink += black_box(dot_flagged(black_box(true), &da, &db)) as f64;
            });
            let d_inline = timed_ns(C_ITERS, || {
                sink += black_box(dot_flagged(
                    black_box(matches!(SimdPath::detect(), SimdPath::Lasx)),
                    &da,
                    &db,
                )) as f64;
            });
            let d_dispatch = timed_ns(C_ITERS, || {
                sink += black_box(dot_vec_dispatch(&da, &db)) as f64;
            });
            for (k, v) in [
                (format!("A2_n{dn}_vec"), d_base),
                (format!("A2_n{dn}_flagged_hoisted"), d_hoist),
                (format!("A2_n{dn}_flagged_detect"), d_inline),
                (format!("A2_n{dn}_dispatch_asis"), d_dispatch),
                (format!("A2_n{dn}_marg_detect"), d_inline - d_hoist),
                (format!("A2_n{dn}_marg_structure"), d_dispatch - d_inline),
            ] {
                println!("CALL r={r} load={load} item=A2 variant={k} ns={v:.2} iters={C_ITERS}");
                push(&k, v, &mut raw);
            }
        }

        // ---- B：`_checked` vs 裸入口（同一算子 n=64）----
        let b_raw = timed_ns(C_ITERS, || {
            lasx_norm3_batch(
                x.as_ptr(),
                y.as_ptr(),
                z.as_ptr(),
                out.as_mut_ptr(),
                N64 as i32,
            );
            sink += black_box(out[0]);
        });
        let b_checked = timed_ns(C_ITERS, || {
            let mut st = 0i32;
            lasx_rs::ffi::checked::lasx_norm3_batch_checked(
                x.as_ptr(),
                y.as_ptr(),
                z.as_ptr(),
                out.as_mut_ptr(),
                N64 as i32,
                &mut st,
            );
            sink += black_box(out[0]) + black_box(st) as f64;
        });
        println!("CALL r={r} load={load} item=B variant=B_floor_raw ns={b_raw:.2} iters={C_ITERS}");
        println!(
            "CALL r={r} load={load} item=B variant=B_checked ns={b_checked:.2} iters={C_ITERS}"
        );
        println!(
            "CALL r={r} load={load} item=B variant=B_marginal ns={:.2} iters={C_ITERS}",
            b_checked - b_raw
        );
        push("B_floor_raw", b_raw, &mut raw);
        push("B_checked", b_checked, &mut raw);
        push("B_marginal", b_checked - b_raw, &mut raw);

        // ---- C：分配 vs 复用 ----
        let c_new = timed_ns(C_ITERS, || {
            let v = AlignedVec::<f64>::new(N64);
            sink += black_box(v[0]);
        });
        let c_api = timed_ns(C_ITERS, || {
            let v = lasx_rs::api::norm3_batch(&x, &y, &z).unwrap();
            sink += black_box(v[0]);
        });
        let c_mul = timed_ns(C_ITERS, || {
            let m = xm.mul(&w);
            sink += black_box(m.as_slice()[0]) as f64;
        });
        let c_mul_into = timed_ns(C_ITERS, || {
            xm.mul_into(&w, &mut outm);
            sink += black_box(outm.as_slice()[0]) as f64;
        });
        for (k, v) in [
            ("C_new64", c_new),
            ("C_api_norm3", c_api),
            ("C_shape_mul", c_mul),
            ("C_shape_mul_into", c_mul_into),
        ] {
            println!("CALL r={r} load={load} item=C variant={k} ns={v:.2} iters={C_ITERS}");
            push(k, v, &mut raw);
        }
        // 边际：分配只出现在 `new` / `api`（分配型）/ `mul`（分配型）那一侧
        for (k, v) in [
            ("C_marg_api_vs_raw", c_api - b_raw),
            ("C_marg_mul_vs_into", c_mul - c_mul_into),
        ] {
            println!("CALL r={r} load={load} item=C variant={k} ns={v:.2} iters={C_ITERS}");
            push(k, v, &mut raw);
        }

        // ---- C2：热循环里反复调用，**分配型 vs `_into` 孪生**（本次落地的那一刀）----
        // 同一内核、同一循环形状，只有"输出缓冲从哪来"不同 ⇒ 差值就是省下的那笔分配。
        let mut ox = vec![0f64; N64];
        let mut oy = vec![0f64; N64];
        let mut oz = vec![0f64; N64];
        let xf: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let yf: Vec<f32> = y.iter().map(|&v| v as f32).collect();
        let mut of = vec![0f32; N64];
        let mut c2: Vec<(&str, f64, f64)> = Vec::new();
        // ① norm3_batch（1 个输出）
        let a1 = timed_ns(C_ITERS, || {
            let v = lasx_rs::api::norm3_batch(&x, &y, &z).unwrap();
            sink += black_box(v[0]);
        });
        let i1 = timed_ns(C_ITERS, || {
            lasx_rs::api::norm3_batch_into(&x, &y, &z, &mut ox).unwrap();
            sink += black_box(ox[0]);
        });
        c2.push(("norm3_batch", a1, i1));
        // ② cross3_batch（3 个输出）
        let a2 = timed_ns(C_ITERS, || {
            let v = lasx_rs::api::cross3_batch(&x, &y, &z, &y, &z, &x).unwrap();
            sink += black_box(v[0][0]);
        });
        let i2 = timed_ns(C_ITERS, || {
            lasx_rs::api::cross3_batch_into(&x, &y, &z, &y, &z, &x, &mut ox, &mut oy, &mut oz)
                .unwrap();
            sink += black_box(ox[0]);
        });
        c2.push(("cross3_batch", a2, i2));
        // ③ unitize3_batch（3 个输出）
        let a3 = timed_ns(C_ITERS, || {
            let v = lasx_rs::api::unitize3_batch(&x, &y, &z).unwrap();
            sink += black_box(v[0][0]);
        });
        let i3 = timed_ns(C_ITERS, || {
            lasx_rs::api::unitize3_batch_into(&x, &y, &z, &mut ox, &mut oy, &mut oz).unwrap();
            sink += black_box(ox[0]);
        });
        c2.push(("unitize3_batch", a3, i3));
        // ④ vec3_add_scaled_batch（3 个输出）
        let a4 = timed_ns(C_ITERS, || {
            let v = lasx_rs::api::vec3_add_scaled_batch(&x, &y, &z, &y, &z, &x, 0.75).unwrap();
            sink += black_box(v[0][0]);
        });
        let i4 = timed_ns(C_ITERS, || {
            lasx_rs::api::vec3_add_scaled_batch_into(
                &x, &y, &z, &y, &z, &x, 0.75, &mut ox, &mut oy, &mut oz,
            )
            .unwrap();
            sink += black_box(ox[0]);
        });
        c2.push(("vec3_add_scaled_batch", a4, i4));
        // ⑤ batch_distance2d（1 个输出，f32）
        let a5 = timed_ns(C_ITERS, || {
            let v = lasx_rs::api::batch_distance2d(0.25, -1.5, &xf, &yf).unwrap();
            sink += black_box(v[0]) as f64;
        });
        let i5 = timed_ns(C_ITERS, || {
            lasx_rs::api::batch_distance2d_into(0.25, -1.5, &xf, &yf, &mut of).unwrap();
            sink += black_box(of[0]) as f64;
        });
        c2.push(("batch_distance2d", a5, i5));

        for (name, a, i) in c2 {
            let save = a - i;
            println!(
                "CALL r={r} load={load} item=C2 variant={name}_alloc ns={a:.2} iters={C_ITERS}"
            );
            println!(
                "CALL r={r} load={load} item=C2 variant={name}_into ns={i:.2} iters={C_ITERS}"
            );
            println!(
                "CALL r={r} load={load} item=C2 variant={name}_save ns={save:.2} iters={C_ITERS}"
            );
            push(&format!("C2_{name}_alloc"), a, &mut raw);
            push(&format!("C2_{name}_into"), i, &mut raw);
            push(&format!("C2_{name}_save"), save, &mut raw);
        }
    }

    // ---- 汇总：丢首轮（r=1）后取中位数 + 极差 ----
    for (name, vs) in &raw {
        let mut v = vs.clone();
        if v.len() > 1 {
            v.remove(0); // 丢首轮
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = if v.len() % 2 == 1 {
            v[v.len() / 2]
        } else {
            (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0
        };
        println!(
            "CALLSUM item={} median={med:.3} min={:.3} max={:.3} rounds={}",
            name,
            v.first().copied().unwrap_or(f64::NAN),
            v.last().copied().unwrap_or(f64::NAN),
            v.len()
        );
    }
    println!("CALLSUM sink={sink:.1}");
}

/// 定向测量 `detect()` 的**每调用固定开销**与"极小形状算子"的每调用时间。
///
/// 机器可读行（`LAT arm=… ns=… …` + `loadavg=`），供**进程级交替**的 before/after A/B 用。
///
/// 顺序很重要：`lasx_force_lsx_thread` 一旦被调用过（新实现会把 `FORCE_USED` 置位），
/// `detect()` 就多走一条线程局部读——真实用户的进程里没人调它，所以
///
/// 1. 先测**极小形状算子**、`hardware()`、以及两种 `detect()` 循环（force 从未用过的常态，
///    也就是被优化的那条路）；
/// 2. 最后才调一次 `lasx_force_lsx_thread(false)`，再测一次 `detect()`（force 分支的代价）。
///
/// 两个状态都在同一个进程、同一个测量窗口里，比值不受窗口漂移影响。
///
/// 另外：本函数用的迭代次数是固定的，所以 `perf stat -e instructions` 的**整进程指令数**
/// 在 before/after 之间可直接相减——两个 `detect` 大循环共 2e7 次调用，差值除以次数就是
/// "每次 `detect()` 省下几条指令"（`detect_fresh` 那 1e7 次走的是被优化的路径）。
pub fn detect_overhead() {
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    let load = load.split_whitespace().next().unwrap_or("?").to_string();

    const ITERS_DETECT: usize = 10_000_000;
    const ITERS_TINY: usize = 1_000_000;
    const ITERS_PROBE: usize = 8_000_000;

    loop_floor_ns(ITERS_DETECT);

    // ---- 极小形状算子（每个入口各调用一次 `detect()`）----
    for tiny_n in [8usize, 64] {
        let x: Vec<f64> = (0..tiny_n).map(|i| 1.0 + i as f64 * 0.25).collect();
        let y: Vec<f64> = (0..tiny_n).map(|i| 2.0 + i as f64 * 0.5).collect();
        let z: Vec<f64> = (0..tiny_n).map(|i| 3.0 + i as f64 * 0.75).collect();
        let mut out = vec![0f64; tiny_n];
        let mut oz = vec![0f64; tiny_n];
        tiny_batch_ns(&x, &y, &z, &mut out, &mut oz, ITERS_TINY);
    }

    // ---- `hardware()`（只读一个原子字节）----
    {
        let mut acc = 0u32;
        let t = Instant::now();
        for _ in 0..ITERS_DETECT {
            acc = acc.wrapping_add(black_box(lasx_rs::arch::hardware().lasx) as u32);
        }
        let ns = t.elapsed().as_secs_f64() * 1e9 / ITERS_DETECT as f64;
        println!("LAT arm=hardware ns={ns:.3} iters={ITERS_DETECT} loadavg={load} acc={acc}");
    }

    // ---- 8 路展开探针：地板与"含 detect"的差 = detect 的吞吐边际成本 ----
    let floor_ns = {
        let t = Instant::now();
        let v = unrolled_probe::<false>(ITERS_PROBE);
        let ns = t.elapsed().as_secs_f64() * 1e9 / ITERS_PROBE as f64;
        println!("LAT arm=unrolled_floor ns={ns:.3} iters={ITERS_PROBE} loadavg={load} acc={v}");
        ns
    };
    let det_thr = {
        let t = Instant::now();
        let v = unrolled_probe::<true>(ITERS_PROBE);
        let ns = t.elapsed().as_secs_f64() * 1e9 / ITERS_PROBE as f64;
        println!("LAT arm=unrolled_detect ns={ns:.3} iters={ITERS_PROBE} loadavg={load} acc={v}");
        ns
    };
    println!(
        "LAT arm=unrolled_marginal ns={:.3} loadavg={load}",
        det_thr - floor_ns
    );

    // ---- 朴素循环（与历史 1.52 ns 口径一致，便于对照）----
    {
        let mut acc = 0u32;
        let t = Instant::now();
        for _ in 0..ITERS_DETECT {
            acc = acc.wrapping_add(black_box(SimdPath::detect()) as u32);
        }
        let ns = t.elapsed().as_secs_f64() * 1e9 / ITERS_DETECT as f64;
        println!("LAT arm=detect_fresh ns={ns:.3} iters={ITERS_DETECT} loadavg={load} acc={acc}");
    }

    // ---- 用过一次 force API 之后的 `detect()`（只有测试/基准会走到这个状态）----
    lasx_rs::lasx_force_lsx_thread(false);
    {
        let mut acc = 0u32;
        let t = Instant::now();
        for _ in 0..ITERS_DETECT {
            acc = acc.wrapping_add(black_box(SimdPath::detect()) as u32);
        }
        let ns = t.elapsed().as_secs_f64() * 1e9 / ITERS_DETECT as f64;
        println!(
            "LAT arm=detect_after_force ns={ns:.3} iters={ITERS_DETECT} loadavg={load} acc={acc}"
        );
    }
}

pub fn lat(_rows: &mut Vec<Row>) {
    // 定向臂（进程级交替 A/B 用）：`lasx_bench lat detect` 只跑"每调用固定开销"这一组，
    // 不夹带指令级长跑，便于把 before/after 两个二进制交替起来测。
    if let Some(arm) = std::env::args().nth(2) {
        match arm.as_str() {
            "detect" => {
                detect_overhead();
                return;
            }
            // 每调用固定开销的受控拆解（同一二进制、同一循环形状，逐项"地板/含该项/边际"）。
            "callcost" => {
                callcost();
                return;
            }
            // 只跑**一个**循环：给 `perf stat -e instructions` 用——整进程指令数 ÷ 1e7
            // 就是"每次 detect() 几条指令"，before/after 相减即"省下几条"（纯计数、无噪声）。
            "detect_only" => {
                let mut acc = 0u32;
                let t = Instant::now();
                for _ in 0..ITERS_CALL {
                    acc = acc.wrapping_add(black_box(SimdPath::detect()) as u32);
                }
                let ns = t.elapsed().as_secs_f64() * 1e9 / ITERS_CALL as f64;
                println!("LAT arm=detect_only ns={ns:.3} iters={ITERS_CALL} acc={acc}");
                return;
            }
            _ => {}
        }
    }
    println!("| 指令级（本机 {MHZ} MHz；周期 = ns × 主频） | 延迟 | 吞吐 | 备注 |");
    println!("|---|---|---|---|");
    let t = Instant::now();
    let _ = black_box(lasx_fma_lat(ITERS_LAT));
    let lasx_lat_ns = t.elapsed().as_secs_f64() * 1e9 / ITERS_LAT as f64;
    let t = Instant::now();
    let _ = black_box(lasx_fma_tp(ITERS_LAT));
    let lasx_tp_ns = t.elapsed().as_secs_f64() * 1e9 / (ITERS_LAT as f64 * 8.0);
    let t = Instant::now();
    let _ = black_box(lsx_fma_lat(ITERS_LAT));
    let lsx_lat_ns = t.elapsed().as_secs_f64() * 1e9 / ITERS_LAT as f64;
    let t = Instant::now();
    let _ = black_box(lsx_fma_tp(ITERS_LAT));
    let lsx_tp_ns = t.elapsed().as_secs_f64() * 1e9 / (ITERS_LAT as f64 * 8.0);
    let c = |ns: f64| ns * MHZ / 1000.0;
    println!(
        "| **LASX** `xvfmadd_s` f32x8 | **{lasx_lat_ns:.2} ns = {:.1} 周期** | {lasx_tp_ns:.3} ns/op = {:.2} 条/周期 | 8 链量吞吐 |",
        c(lasx_lat_ns),
        1.0 / c(lasx_tp_ns)
    );
    println!(
        "| **LSX** `vfmadd_s` f32x4 | **{lsx_lat_ns:.2} ns = {:.1} 周期** | {lsx_tp_ns:.3} ns/op = {:.2} 条/周期 | 8 链量吞吐 |",
        c(lsx_lat_ns),
        1.0 / c(lsx_tp_ns)
    );

    // `detect()` 的每调用成本（含 TLS 读 `forced_lsx()` + 原子读 `hardware()`）。
    let mut acc = 0u32;
    for _ in 0..100_000 {
        acc = acc.wrapping_add(SimdPath::detect() as u32);
    }
    let t = Instant::now();
    for _ in 0..ITERS_CALL {
        acc = acc.wrapping_add(black_box(SimdPath::detect()) as u32);
    }
    let det = t.elapsed().as_secs_f64() * 1e9 / ITERS_CALL as f64;
    let t = Instant::now();
    for _ in 0..ITERS_CALL {
        acc = acc.wrapping_add(black_box(SimdPath::detect().is_lasx()) as u32);
    }
    let det2 = t.elapsed().as_secs_f64() * 1e9 / ITERS_CALL as f64;
    println!(
        "| `SimdPath::detect()` | **{det:.2} ns/次** | `is_lasx()` {det2:.2} ns/次 | 含 TLS 读 + 原子读（acc={acc}） |"
    );
    println!();

    for n in [8usize, 16, 24] {
        call_ns("ours(现状)", n, dot_ours);
        call_ns("标量参考", n, dot_ref);
        call_ns("vec(候选)", n, dot_vec);
        call_ns("vec+detect", n, dot_vec_dispatch);
        println!();
    }
}
