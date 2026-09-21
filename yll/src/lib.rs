//! `lasx_rs` 的 YouLiLong 原生扩展。
//!
//! 脚本侧用法：
//!
//! ```text
//! use "./lasx"
//!
//! a = [1.0, 2.0, 3.0, 4.0]
//! b = [1.0, 1.0, 1.0, 1.0]
//! print(lasx.dot(a, b))          // 10
//! print(lasx.sum(a))             // 10
//!
//! try {
//!     lasx.dot(a, [1.0])          // 长度不一致
//! } catch (e) {
//!     print("捕获到: " + e)        // 错误由 yll_error 上抛
//! }
//! ```
//!
//! 约定（见 YouLiLong `docs/高级特性/C-C++原生扩展开发指南.md`）：
//!
//! - 入口符号 `yll_init_lasx`，用 `yll_module_new` + `yll_module_add_func` 注册；
//! - 对外函数签名固定为 `YllValueWrapper* f(YllContextC*, int, YllValueWrapper**)`；
//! - **所有校验失败都返回 `yll_error(...)`**，解释器会抛成可 `try/catch` 的运行时错误
//!   ——不静默返回 0、不吞掉错误；
//! - 解释器不提供 GIL，函数可能被多线程并发调用；除 `propagate` 外都是纯函数
//!   （无共享可变状态）。`propagate` 持有一个进程内常驻线程池，用 `Mutex` 串行化
//!   并发调用——池的派活本身是独占的。

#![allow(non_camel_case_types)]

mod convert;
mod funcs;
mod yll;

use yll::{
    str_value, with_cstr, yll_int, yll_module_add_const, yll_module_add_func, yll_module_new,
    YllCFunc, YllModuleDefC,
};

/// 注册一个对外函数（名字与文档都是临时 C 串，调用期间存活）。
///
/// # Safety
/// `module` 必须是 `yll_module_new` 的返回值。
unsafe fn add(
    module: *mut YllModuleDefC,
    name: &str,
    func: YllCFunc,
    min_args: i32,
    max_args: i32,
    doc: &str,
) {
    with_cstr(name, |n| {
        with_cstr(doc, |d| {
            yll_module_add_func(module, n, func, min_args, max_args, d)
        })
    });
}

/// 注册一个常量。
///
/// # Safety
/// `module` 必须是 `yll_module_new` 的返回值。
unsafe fn add_const(module: *mut YllModuleDefC, name: &str, value: *mut yll::YllValueWrapper) {
    with_cstr(name, |n| yll_module_add_const(module, n, value));
}

/// 模块初始化入口：脚本 `use "./lasx"` 时由解释器调用。
#[unsafe(no_mangle)]
pub extern "C" fn yll_init_lasx() -> *mut YllModuleDefC {
    unsafe {
        let module = with_cstr("lasx", |n| {
            with_cstr(env!("CARGO_PKG_VERSION"), |v| yll_module_new(n, v))
        });

        // ---- 归约 ----
        add(
            module,
            "dot",
            funcs::fn_dot,
            2,
            2,
            "dot(a, b) -> float：f32 点积 Σ a[i]·b[i]",
        );
        add(
            module,
            "sum",
            funcs::fn_sum,
            1,
            1,
            "sum(x) -> float：f32 归约 Σ x[i]",
        );
        add(
            module,
            "dot_i8",
            funcs::fn_dot_i8,
            2,
            2,
            "dot_i8(a, b) -> int：int8 量化点积（元素需在 -128..=127）",
        );

        // ---- 矩阵乘 ----
        add(
            module,
            "matmul",
            funcs::fn_matmul,
            5,
            5,
            "matmul(a, b, m, k, n) -> array：C[m×n]=A[m×k]·B[k×n]，A/B 为行主序扁平数组",
        );

        // ---- 批量几何 ----
        add(
            module,
            "norm3",
            funcs::fn_norm3,
            3,
            3,
            "norm3(xs, ys, zs) -> array：out[i]=√(x²+y²+z²)（f64）",
        );
        add(
            module,
            "distance2d",
            funcs::fn_distance2d,
            4,
            4,
            "distance2d(px, py, xs, ys) -> array：各点到 (px,py) 的距离（f32）",
        );

        // ---- 批量物理 ----
        add(
            module,
            "vec3_add_scaled",
            funcs::fn_vec3_add_scaled,
            7,
            7,
            "vec3_add_scaled(ax, ay, az, bx, by, bz, s) -> [ox, oy, oz]：o = a + s·b",
        );
        add(
            module,
            "j2_accel",
            funcs::fn_j2_accel,
            6,
            6,
            "j2_accel(rx, ry, rz, mu, j2, re) -> [ax, ay, az]：中心引力 + J2 摄动加速度",
        );
        add(
            module,
            "propagate",
            funcs::fn_propagate,
            11,
            11,
            "propagate(rx, ry, rz, vx, vy, vz, mu, j2, re, dt, steps) -> [rx, ry, rz, vx, vy, vz]：多星多步 RK4 J2 传播（多核，常驻线程池跨步复用）",
        );
        add(
            module,
            "rk4_step",
            funcs::fn_rk4_step,
            10,
            10,
            "rk4_step(rx, ry, rz, vx, vy, vz, mu, j2, re, dt) -> [rx, ry, rz, vx, vy, vz]：批量 RK4 J2 单步",
        );

        // ---- 常量 ----
        add_const(module, "VERSION", str_value(env!("CARGO_PKG_VERSION")));
        add_const(module, "ALIGN", yll_int(lasx_rs::aligned::ALIGN as i64));

        module
    }
}
