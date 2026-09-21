//! YouLiLong 值与库缓冲之间的转换。
//!
//! 输入统一转成 [`AlignedVec`]——LASX 是 32 字节访存，而脚本侧的数组缓冲区对齐与否
//! 是抽签（实测 glibc `malloc` / Dart FFI 只有约一半落在 32 字节边界上），转换时
//! 顺手对齐，内核才能跑满。输出同样先写进对齐缓冲，再逐个 push 回数组。

use lasx_rs::aligned::AlignedVec;

use crate::yll::{
    arg, arg_arr_len, yll_arr_get, yll_arr_new, yll_arr_push, yll_as_float, yll_as_int, yll_float,
    yll_is_float, yll_is_int, YllValueWrapper,
};

/// 单个数组允许的最大元素数（防止脚本传入荒谬形状导致巨量分配）。
pub const MAX_LEN: usize = 1 << 28;

/// 读一个数组参数，逐元素经 `convert` 转成 `T`，写进对齐缓冲。
///
/// 三个具体读取器（`f64`/`f32`/`i8`）的差别只在元素转换与报错文案，故泛型化到这里。
///
/// # Safety
/// `argv` 必须有至少 `idx + 1` 个有效元素（解释器按 `argc` 保证）。
unsafe fn read_array<T: Copy + Default>(
    argv: *mut *mut YllValueWrapper,
    idx: usize,
    name: &str,
    convert: impl Fn(f64, usize, &str) -> Result<T, String>,
) -> Result<AlignedVec<T>, String> {
    let len = arg_arr_len(argv, idx, name)?;
    if len > MAX_LEN {
        return Err(format!("参数 `{name}` 过大（{len} > {MAX_LEN}）"));
    }
    let arr = arg(argv, idx);
    let mut out = AlignedVec::<T>::new(len);
    for (i, slot) in out.as_mut_slice().iter_mut().enumerate() {
        let v = yll_arr_get(arr, i as i64);
        let raw = if yll_is_float(v) {
            yll_as_float(v)
        } else if yll_is_int(v) {
            yll_as_int(v) as f64
        } else {
            return Err(format!("参数 `{name}` 的第 {i} 个元素不是数值"));
        };
        *slot = convert(raw, i, name)?;
    }
    Ok(out)
}

/// 读成对齐的 `f64` 缓冲（`f64` 内核用）。
///
/// # Safety
/// 见 [`read_array`]。
pub unsafe fn read_f64(
    argv: *mut *mut YllValueWrapper,
    idx: usize,
    name: &str,
) -> Result<AlignedVec<f64>, String> {
    read_array(argv, idx, name, |f, _, _| Ok(f))
}

/// 读成对齐的 `f32` 缓冲（`f32` 内核用；按 IEEE 就近舍入收窄）。
///
/// # Safety
/// 见 [`read_array`]。
pub unsafe fn read_f32(
    argv: *mut *mut YllValueWrapper,
    idx: usize,
    name: &str,
) -> Result<AlignedVec<f32>, String> {
    read_array(argv, idx, name, |f, _, _| Ok(f as f32))
}

/// 读成对齐的 `i8` 缓冲（量化内核用；要求是 −128..=127 上的整数）。
///
/// # Safety
/// 见 [`read_array`]。
pub unsafe fn read_i8(
    argv: *mut *mut YllValueWrapper,
    idx: usize,
    name: &str,
) -> Result<AlignedVec<i8>, String> {
    read_array(argv, idx, name, |f, i, name| {
        if f.fract() != 0.0 || !(-128.0..=127.0).contains(&f) {
            return Err(format!("参数 `{name}` 的第 {i} 个元素不是 int8（{f}）"));
        }
        Ok(f as i8)
    })
}

/// 读一组等长数组（SOA 内核的多个分量），任一长度不一致即报错。
///
/// # Safety
/// 见 [`read_array`]。
pub unsafe fn read_f64_soa<const N: usize>(
    argv: *mut *mut YllValueWrapper,
    names: [&str; N],
) -> Result<[AlignedVec<f64>; N], String> {
    read_f64_soa_at(argv, names, 0)
}

/// 同 [`read_f64_soa`]，但从第 `offset` 个参数开始读。
///
/// # Safety
/// 见 [`read_array`]。
pub unsafe fn read_f64_soa_at<const N: usize>(
    argv: *mut *mut YllValueWrapper,
    names: [&str; N],
    offset: usize,
) -> Result<[AlignedVec<f64>; N], String> {
    let mut out: [AlignedVec<f64>; N] = std::array::from_fn(|_| AlignedVec::new(0));
    for (k, name) in names.iter().enumerate() {
        out[k] = read_f64(argv, offset + k, name)?;
    }
    let len = out[0].len();
    if let Some(bad) = (0..N).find(|&k| out[k].len() != len) {
        return Err(format!(
            "数组长度不一致：`{}` 有 {} 个元素，`{}` 有 {} 个",
            names[0],
            len,
            names[bad],
            out[bad].len()
        ));
    }
    Ok(out)
}

/// 把一组 `f64` 值变成 YouLiLong 数组。
///
/// # Safety
/// 必须在解释器线程内调用（`yll_*` 构造器非线程安全）。
pub unsafe fn push_f64_array(values: &[f64]) -> *mut YllValueWrapper {
    let arr = yll_arr_new();
    for &v in values {
        yll_arr_push(arr, yll_float(v));
    }
    arr
}

/// 把多组 `f64` 变成"数组的数组"（多分量内核的返回值）。
///
/// # Safety
/// 见 [`push_f64_array`]。
pub unsafe fn push_f64_arrays(arrays: &[&[f64]]) -> *mut YllValueWrapper {
    let outer = yll_arr_new();
    for a in arrays {
        yll_arr_push(outer, push_f64_array(a));
    }
    outer
}
