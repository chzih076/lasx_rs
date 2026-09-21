//! 对 YouLiLong 脚本暴露的函数。
//!
//! 每个函数都是"脚本参数 → 校验 → 调 [`lasx_rs::ffi::checked`] → 构造返回值"，
//! 任何一步失败都返回 `yll_error(...)`，解释器会把它抛成脚本可 `try/catch` 的运行时错误。
//!
//! 校验分两层：本层能判的（是不是数组、元素是不是数值、数组长度与声明形状是否自洽、
//! 物理常数是否合法）先判，给出**具体到参数名与下标**的消息；再由库的 `*_checked`
//! 兜住结构性问题（空指针、负长度、溢出）。

use std::ffi::c_int;
use std::sync::{Mutex, OnceLock};

use lasx_rs::ffi::checked::{
    lasx_batch_distance2d_checked, lasx_dot_checked, lasx_dot_i8_checked,
    lasx_j2_accel_batch_checked, lasx_matmul_checked, lasx_norm3_batch_checked,
    lasx_rk4_j2_step_batch_checked, lasx_sum_checked, lasx_vec3_add_scaled_batch_checked,
};
use lasx_rs::ffi::status::LasxStatus;

use crate::convert::{
    push_f64_array, push_f64_arrays, read_f32, read_f64_soa, read_f64_soa_at, read_i8,
};
use crate::yll::{arg_f64, arg_int, error, yll_float, yll_int, YllContextC, YllValueWrapper};

/// 进程内唯一的常驻线程池：建一次，跨调用/跨步复用。
///
/// 多步传播的收益几乎全在"池复用"上（每次新建线程要付约 0.5 ms/步，见
/// `docs/perf-report.md` §13.3）。解释器不提供 GIL、可能并发调用本扩展，而池的派活
/// 是**独占**的（`for_each_*` 取 `&mut self`），故用 `Mutex` 串行化两个脚本线程的调用。
fn pool() -> std::sync::MutexGuard<'static, lasx_rs::pool::WorkerPool> {
    static POOL: OnceLock<Mutex<lasx_rs::pool::WorkerPool>> = OnceLock::new();
    // 中毒说明上一次调用在持锁时 panic（例如形状断言被绕过）；池在 panic 后状态是干净的
    // （worker 已全部收工），故取回内部值继续用，而不是让整个扩展从此不可用。
    POOL.get_or_init(|| Mutex::new(lasx_rs::pool::WorkerPool::auto()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// 统一收口：把 `Result` 里的错误消息变成解释器会抛出的错误值。
///
/// 所有对外函数都只写 `Ok(...)`/`Err(String)`，不直接构造错误值——错误上抛只有这一条路径。
fn guard(f: impl FnOnce() -> Result<*mut YllValueWrapper, String>) -> *mut YllValueWrapper {
    match f() {
        Ok(v) => v,
        Err(msg) => error(&msg),
    }
}

/// 库返回的状态码 → 脚本侧错误消息。
fn status_err(what: &str, code: i32) -> String {
    match LasxStatus::from_i32(code) {
        Some(LasxStatus::Ok) | None => format!("{what}: 未知状态码 {code}"),
        Some(s) => format!("{what}: {}", s.message()),
    }
}

/// 断言库调用成功；失败即返回错误消息。
macro_rules! ok_or {
    ($what:expr, $st:expr) => {
        if $st != LasxStatus::Ok as i32 {
            return Err(status_err($what, $st));
        }
    };
}

/* ==================== 归约 ==================== */

/// `dot(a, b) -> float`：f32 点积。
#[unsafe(no_mangle)]
pub extern "C" fn fn_dot(
    _ctx: *mut YllContextC,
    argc: c_int,
    argv: *mut *mut YllValueWrapper,
) -> *mut YllValueWrapper {
    guard(|| unsafe {
        if argc < 2 {
            return Err("dot(a, b) 需要 2 个数组参数".into());
        }
        let a = read_f32(argv, 0, "a")?;
        let b = read_f32(argv, 1, "b")?;
        if a.len() != b.len() {
            return Err(format!("a 与 b 长度不一致：{} vs {}", a.len(), b.len()));
        }
        let mut st = 0i32;
        let d = lasx_dot_checked(a.as_ptr(), b.as_ptr(), a.len() as i32, &mut st);
        ok_or!("dot", st);
        Ok(yll_float(d as f64))
    })
}

/// `sum(x) -> float`：f32 归约。
#[unsafe(no_mangle)]
pub extern "C" fn fn_sum(
    _ctx: *mut YllContextC,
    argc: c_int,
    argv: *mut *mut YllValueWrapper,
) -> *mut YllValueWrapper {
    guard(|| unsafe {
        if argc < 1 {
            return Err("sum(x) 需要 1 个数组参数".into());
        }
        let x = read_f32(argv, 0, "x")?;
        let mut st = 0i32;
        let s = lasx_sum_checked(x.as_ptr(), x.len() as i32, &mut st);
        ok_or!("sum", st);
        Ok(yll_float(s as f64))
    })
}

/// `dot_i8(a, b) -> int`：int8 量化点积。
#[unsafe(no_mangle)]
pub extern "C" fn fn_dot_i8(
    _ctx: *mut YllContextC,
    argc: c_int,
    argv: *mut *mut YllValueWrapper,
) -> *mut YllValueWrapper {
    guard(|| unsafe {
        if argc < 2 {
            return Err("dot_i8(a, b) 需要 2 个数组参数".into());
        }
        let a = read_i8(argv, 0, "a")?;
        let b = read_i8(argv, 1, "b")?;
        if a.len() != b.len() {
            return Err(format!("a 与 b 长度不一致：{} vs {}", a.len(), b.len()));
        }
        let mut st = 0i32;
        let d = lasx_dot_i8_checked(a.as_ptr(), b.as_ptr(), a.len() as i32, &mut st);
        ok_or!("dot_i8", st);
        Ok(yll_int(d as i64))
    })
}

/* ==================== 矩阵乘 ==================== */

/// `matmul(a, b, m, k, n) -> array`：`C[m×n] = A[m×k]·B[k×n]`，A/B 为行主序扁平数组。
#[unsafe(no_mangle)]
pub extern "C" fn fn_matmul(
    _ctx: *mut YllContextC,
    argc: c_int,
    argv: *mut *mut YllValueWrapper,
) -> *mut YllValueWrapper {
    guard(|| unsafe {
        if argc < 5 {
            return Err("matmul(a, b, m, k, n) 需要 5 个参数".into());
        }
        let a = read_f32(argv, 0, "a")?;
        let b = read_f32(argv, 1, "b")?;
        let m = arg_int(argv, 2, "m")?;
        let k = arg_int(argv, 3, "k")?;
        let n = arg_int(argv, 4, "n")?;
        if m < 0 || k < 0 || n < 0 {
            return Err(format!("m/k/n 必须非负，得到 ({m}, {k}, {n})"));
        }
        let (m, k, n) = (m as usize, k as usize, n as usize);
        // 形状自洽性——只有这一层知道数组的真实长度，故 BadShape 在这里判
        if a.len() != m * k {
            return Err(format!(
                "a 的形状不对：声明 {m}×{k}={} 个元素，实际 {}",
                m * k,
                a.len()
            ));
        }
        if b.len() != k * n {
            return Err(format!(
                "b 的形状不对：声明 {k}×{n}={} 个元素，实际 {}",
                k * n,
                b.len()
            ));
        }
        let mut c = lasx_rs::aligned::AlignedVec::<f32>::new(m * n);
        let mut st = 0i32;
        lasx_matmul_checked(
            m as i32,
            k as i32,
            n as i32,
            a.as_ptr(),
            b.as_ptr(),
            c.as_mut_ptr(),
            &mut st,
        );
        ok_or!("matmul", st);
        let out: Vec<f64> = c.iter().map(|&v| v as f64).collect();
        Ok(push_f64_array(&out))
    })
}

/* ==================== 批量几何 ==================== */

/// `norm3(xs, ys, zs) -> array`：`out[i] = √(x²+y²+z²)`（f64）。
#[unsafe(no_mangle)]
pub extern "C" fn fn_norm3(
    _ctx: *mut YllContextC,
    argc: c_int,
    argv: *mut *mut YllValueWrapper,
) -> *mut YllValueWrapper {
    guard(|| unsafe {
        if argc < 3 {
            return Err("norm3(xs, ys, zs) 需要 3 个数组参数".into());
        }
        let [xs, ys, zs] = read_f64_soa(argv, ["xs", "ys", "zs"])?;
        let n = xs.len();
        let mut out = lasx_rs::aligned::AlignedVec::<f64>::new(n);
        let mut st = 0i32;
        lasx_norm3_batch_checked(
            xs.as_ptr(),
            ys.as_ptr(),
            zs.as_ptr(),
            out.as_mut_ptr(),
            n as i32,
            &mut st,
        );
        ok_or!("norm3", st);
        Ok(push_f64_array(&out))
    })
}

/// `distance2d(px, py, xs, ys) -> array`：每个点到 `(px, py)` 的距离（f32）。
#[unsafe(no_mangle)]
pub extern "C" fn fn_distance2d(
    _ctx: *mut YllContextC,
    argc: c_int,
    argv: *mut *mut YllValueWrapper,
) -> *mut YllValueWrapper {
    guard(|| unsafe {
        if argc < 4 {
            return Err("distance2d(px, py, xs, ys) 需要 4 个参数".into());
        }
        let px = arg_f64(argv, 0, "px")?;
        let py = arg_f64(argv, 1, "py")?;
        let xs = read_f32(argv, 2, "xs")?;
        let ys = read_f32(argv, 3, "ys")?;
        if xs.len() != ys.len() {
            return Err(format!("xs 与 ys 长度不一致：{} vs {}", xs.len(), ys.len()));
        }
        let n = xs.len();
        let mut out = lasx_rs::aligned::AlignedVec::<f32>::new(n);
        let mut st = 0i32;
        lasx_batch_distance2d_checked(
            px as f32,
            py as f32,
            xs.as_ptr(),
            ys.as_ptr(),
            out.as_mut_ptr(),
            n as i32,
            &mut st,
        );
        ok_or!("distance2d", st);
        let out64: Vec<f64> = out.iter().map(|&v| v as f64).collect();
        Ok(push_f64_array(&out64))
    })
}

/* ==================== 物理 ==================== */

/// `vec3_add_scaled(ax, ay, az, bx, by, bz, s) -> [ox, oy, oz]`：`o = a + s·b`。
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn fn_vec3_add_scaled(
    _ctx: *mut YllContextC,
    argc: c_int,
    argv: *mut *mut YllValueWrapper,
) -> *mut YllValueWrapper {
    guard(|| unsafe {
        if argc < 7 {
            return Err("vec3_add_scaled(ax, ay, az, bx, by, bz, s) 需要 7 个参数".into());
        }
        let s = arg_f64(argv, 6, "s")?;
        let [ax, ay, az] = read_f64_soa_at(argv, ["ax", "ay", "az"], 0)?;
        let [bx, by, bz] = read_f64_soa_at(argv, ["bx", "by", "bz"], 3)?;
        let n = ax.len();
        if bx.len() != n {
            return Err(format!("a 与 b 长度不一致：{n} vs {}", bx.len()));
        }
        let (mut ox, mut oy, mut oz) = (
            lasx_rs::aligned::AlignedVec::<f64>::new(n),
            lasx_rs::aligned::AlignedVec::<f64>::new(n),
            lasx_rs::aligned::AlignedVec::<f64>::new(n),
        );
        let mut st = 0i32;
        lasx_vec3_add_scaled_batch_checked(
            ax.as_ptr(),
            ay.as_ptr(),
            az.as_ptr(),
            bx.as_ptr(),
            by.as_ptr(),
            bz.as_ptr(),
            s,
            ox.as_mut_ptr(),
            oy.as_mut_ptr(),
            oz.as_mut_ptr(),
            n as i32,
            &mut st,
        );
        ok_or!("vec3_add_scaled", st);
        Ok(push_f64_arrays(&[&ox, &oy, &oz]))
    })
}

/// `j2_accel(rx, ry, rz, mu, j2, re) -> [ax, ay, az]`：中心引力 + J2 摄动。
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn fn_j2_accel(
    _ctx: *mut YllContextC,
    argc: c_int,
    argv: *mut *mut YllValueWrapper,
) -> *mut YllValueWrapper {
    guard(|| unsafe {
        if argc < 6 {
            return Err("j2_accel(rx, ry, rz, mu, j2, re) 需要 6 个参数".into());
        }
        let [rx, ry, rz] = read_f64_soa(argv, ["rx", "ry", "rz"])?;
        let mu = arg_f64(argv, 3, "mu")?;
        let j2 = arg_f64(argv, 4, "j2")?;
        let re = arg_f64(argv, 5, "re")?;
        let n = rx.len();
        let mut ax = lasx_rs::aligned::AlignedVec::<f64>::new(n);
        let mut ay = lasx_rs::aligned::AlignedVec::<f64>::new(n);
        let mut az = lasx_rs::aligned::AlignedVec::<f64>::new(n);
        let mut st = 0i32;
        lasx_j2_accel_batch_checked(
            rx.as_ptr(),
            ry.as_ptr(),
            rz.as_ptr(),
            mu,
            j2,
            re,
            ax.as_mut_ptr(),
            ay.as_mut_ptr(),
            az.as_mut_ptr(),
            n as i32,
            &mut st,
        );
        ok_or!("j2_accel", st);
        Ok(push_f64_arrays(&[&ax, &ay, &az]))
    })
}

/// `rk4_step(rx, ry, rz, vx, vy, vz, mu, j2, re, dt) -> [rx, ry, rz, vx, vy, vz]`：
/// 批量 RK4 J2 单步（原地推进）。
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn fn_rk4_step(
    _ctx: *mut YllContextC,
    argc: c_int,
    argv: *mut *mut YllValueWrapper,
) -> *mut YllValueWrapper {
    guard(|| unsafe {
        if argc < 10 {
            return Err("rk4_step(rx, ry, rz, vx, vy, vz, mu, j2, re, dt) 需要 10 个参数".into());
        }
        let mut p = read_f64_soa(argv, ["rx", "ry", "rz"])?;
        let mut v = read_f64_soa_at(argv, ["vx", "vy", "vz"], 3)?;
        let mu = arg_f64(argv, 6, "mu")?;
        let j2 = arg_f64(argv, 7, "j2")?;
        let re = arg_f64(argv, 8, "re")?;
        let dt = arg_f64(argv, 9, "dt")?;
        if p[0].len() != v[0].len() {
            return Err(format!(
                "位置与速度长度不一致：{} vs {}",
                p[0].len(),
                v[0].len()
            ));
        }
        let n = p[0].len();
        let mut st = 0i32;
        lasx_rk4_j2_step_batch_checked(
            p[0].as_mut_ptr(),
            p[1].as_mut_ptr(),
            p[2].as_mut_ptr(),
            v[0].as_mut_ptr(),
            v[1].as_mut_ptr(),
            v[2].as_mut_ptr(),
            mu,
            j2,
            re,
            dt,
            n as i32,
            &mut st,
        );
        ok_or!("rk4_step", st);
        Ok(push_f64_arrays(&[&p[0], &p[1], &p[2], &v[0], &v[1], &v[2]]))
    })
}

/// `propagate(rx, ry, rz, vx, vy, vz, mu, j2, re, dt, steps) -> [rx, ry, rz, vx, vy, vz]`：
/// 多星 × 多步 RK4 J2 传播（多核）。
///
/// 与 [`fn_rk4_step`] 的区别只在**调用策略**：这里是常驻线程池跑 `steps` 步、池跨步复用，
/// 而 `rk4_step` 是单线程单步。同样的内核与线程数，端到端快约 2.1–2.3×（perf-report §13.3）。
///
/// 多步传播是**原地推进**的，脚本侧拿到的是终态；要逐步观察请用循环里的 `rk4_step`。
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn fn_propagate(
    _ctx: *mut YllContextC,
    argc: c_int,
    argv: *mut *mut YllValueWrapper,
) -> *mut YllValueWrapper {
    guard(|| unsafe {
        if argc < 11 {
            return Err(
                "propagate(rx, ry, rz, vx, vy, vz, mu, j2, re, dt, steps) 需要 11 个参数".into(),
            );
        }
        let p = read_f64_soa(argv, ["rx", "ry", "rz"])?;
        let v = read_f64_soa_at(argv, ["vx", "vy", "vz"], 3)?;
        let mu = arg_f64(argv, 6, "mu")?;
        let j2 = arg_f64(argv, 7, "j2")?;
        let re = arg_f64(argv, 8, "re")?;
        let dt = arg_f64(argv, 9, "dt")?;
        let steps = arg_int(argv, 10, "steps")?;
        if p[0].len() != v[0].len() {
            return Err(format!(
                "位置与速度长度不一致：{} vs {}",
                p[0].len(),
                v[0].len()
            ));
        }
        if steps < 0 {
            return Err(format!("steps 必须非负，得到 {steps}"));
        }
        for (name, val) in [("mu", mu), ("j2", j2), ("re", re), ("dt", dt)] {
            if !val.is_finite() {
                return Err(format!("{name} 必须是有限数，得到 {val}"));
            }
        }
        if mu <= 0.0 || re <= 0.0 {
            return Err(format!("mu 与 re 必须为正：mu={mu}, re={re}"));
        }
        // 0 步或空数组：直接原样返回，不进池（也避免建池）
        if steps == 0 || p[0].is_empty() {
            return Ok(push_f64_arrays(&[&p[0], &p[1], &p[2], &v[0], &v[1], &v[2]]));
        }

        // 解构出 6 个独立绑定：`p[0].as_mut_slice()` 这种索引写法过不了借用检查
        let [mut px, mut py, mut pz] = p;
        let [mut qx, mut qy, mut qz] = v;
        let mut pool = pool();
        for _ in 0..steps as usize {
            lasx_rs::parallel::rk4_j2_step_batch(
                &mut pool,
                mu,
                j2,
                re,
                dt,
                px.as_mut_slice(),
                py.as_mut_slice(),
                pz.as_mut_slice(),
                qx.as_mut_slice(),
                qy.as_mut_slice(),
                qz.as_mut_slice(),
            );
        }
        Ok(push_f64_arrays(&[&px, &py, &pz, &qx, &qy, &qz]))
    })
}
