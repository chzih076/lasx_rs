//! YouLiLong C-API 绑定（`yll.h`）。
//!
//! 只声明扩展实际用到的符号（`yll.h` 里还有 `yll_value_ref/unref`、`yll_obj_*`、
//! `yll_mutex_*` 等，本扩展用不到就不绑定）。约定见 YouLiLong 文档
//! `docs/高级特性/C-C++原生扩展开发指南.md` §5 与
//! `docs/API参考/C-API头文件参考.md`。
//!
//! **错误约定**：C 函数返回 `yll_error("...")` 构造的错误值时，解释器会自动抛出
//! 运行时错误，脚本可用 `try/catch` 捕获。本扩展的所有对外函数因此在任何校验失败
//! 时都返回错误值，而不是静默返回 0 或半个结果。

#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_int, c_void, CString};

/// 不透明的值包装（`yll.h` 中 `struct YllValueWrapper`）。
pub type YllValueWrapper = c_void;
/// 解释器上下文（本扩展不使用，按约定接收并忽略）。
pub type YllContextC = c_void;
/// 模块定义（`YllModuleDefC`）。
pub type YllModuleDefC = c_void;

/// 对外 C 函数的统一签名。
pub type YllCFunc =
    extern "C" fn(*mut YllContextC, c_int, *mut *mut YllValueWrapper) -> *mut YllValueWrapper;

extern "C" {
    // ---- 构造 Value ----
    pub fn yll_int(i: i64) -> *mut YllValueWrapper;
    pub fn yll_float(f: f64) -> *mut YllValueWrapper;
    pub fn yll_str(s: *const c_char) -> *mut YllValueWrapper;
    pub fn yll_arr_new() -> *mut YllValueWrapper;
    pub fn yll_arr_push(arr: *mut YllValueWrapper, value: *mut YllValueWrapper);

    // ---- 读取 Value ----
    pub fn yll_is_int(v: *mut YllValueWrapper) -> bool;
    pub fn yll_is_float(v: *mut YllValueWrapper) -> bool;
    pub fn yll_is_arr(v: *mut YllValueWrapper) -> bool;
    pub fn yll_as_int(v: *mut YllValueWrapper) -> i64;
    pub fn yll_as_float(v: *mut YllValueWrapper) -> f64;
    pub fn yll_arr_len(v: *mut YllValueWrapper) -> i64;
    pub fn yll_arr_get(v: *mut YllValueWrapper, index: i64) -> *mut YllValueWrapper;

    // ---- 错误 ----
    pub fn yll_error(message: *const c_char) -> *mut YllValueWrapper;

    // ---- 模块定义 ----
    pub fn yll_module_new(name: *const c_char, version: *const c_char) -> *mut YllModuleDefC;
    pub fn yll_module_add_func(
        module: *mut YllModuleDefC,
        name: *const c_char,
        func: YllCFunc,
        min_args: c_int,
        max_args: c_int,
        doc: *const c_char,
    );
    pub fn yll_module_add_const(
        module: *mut YllModuleDefC,
        name: *const c_char,
        value: *mut YllValueWrapper,
    );
}

/// 借出一个临时 C 字符串，**仅在本次调用期间有效**。
///
/// 参考实现（`native-ext/gpio2k3000`）用 `CString::into_raw()`，每次调用泄漏一次分配。
/// 这里改成"调用期间存活、调用后释放"：`yll_*` 构造器会把字符串内容拷进自己的 Value，
/// 因此回调返回后指针即失效是安全的。
///
/// 字符串含 NUL 时退化为空串（`CString::new` 失败），不 panic。
pub fn with_cstr<R>(s: &str, f: impl FnOnce(*const c_char) -> R) -> R {
    let cs = CString::new(s).unwrap_or_default();
    f(cs.as_ptr())
}

/// 构造错误值（解释器会把它抛成可 `try/catch` 的运行时错误）。
pub fn error(msg: &str) -> *mut YllValueWrapper {
    with_cstr(msg, |p| unsafe { yll_error(p) })
}

/// 构造字符串值。
pub fn str_value(s: &str) -> *mut YllValueWrapper {
    with_cstr(s, |p| unsafe { yll_str(p) })
}

/* ==================== 参数读取辅助 ==================== */

/// 第 `idx` 个参数（调用方保证 `idx < argc`）。
///
/// # Safety
/// `argv` 必须至少有 `idx + 1` 个有效元素。
pub unsafe fn arg(argv: *mut *mut YllValueWrapper, idx: usize) -> *mut YllValueWrapper {
    *argv.add(idx)
}

/// 读取一个数值参数（整数或浮点都接受，统一取 `f64`）。
///
/// # Safety
/// 见 [`arg`]。
pub unsafe fn arg_f64(
    argv: *mut *mut YllValueWrapper,
    idx: usize,
    name: &str,
) -> Result<f64, String> {
    let v = arg(argv, idx);
    if yll_is_float(v) || yll_is_int(v) {
        Ok(yll_as_float(v))
    } else {
        Err(format!("参数 `{name}` 必须是数值"))
    }
}

/// 读取一个整数参数。
///
/// # Safety
/// 见 [`arg`]。
pub unsafe fn arg_int(
    argv: *mut *mut YllValueWrapper,
    idx: usize,
    name: &str,
) -> Result<i64, String> {
    let v = arg(argv, idx);
    if yll_is_int(v) {
        Ok(yll_as_int(v))
    } else if yll_is_float(v) {
        // 允许 4.0 这种写法，但不能有小数部分
        let f = yll_as_float(v);
        if f.fract() == 0.0 && f.is_finite() {
            Ok(f as i64)
        } else {
            Err(format!("参数 `{name}` 必须是整数"))
        }
    } else {
        Err(format!("参数 `{name}` 必须是整数"))
    }
}

/// 读取数组参数的长度，并确认它确实是数组。
///
/// # Safety
/// 见 [`arg`]。
pub unsafe fn arg_arr_len(
    argv: *mut *mut YllValueWrapper,
    idx: usize,
    name: &str,
) -> Result<usize, String> {
    let v = arg(argv, idx);
    if !yll_is_arr(v) {
        return Err(format!("参数 `{name}` 必须是数组"));
    }
    let len = yll_arr_len(v);
    if len < 0 {
        return Err(format!("参数 `{name}` 的长度非法（{len}）"));
    }
    Ok(len as usize)
}
