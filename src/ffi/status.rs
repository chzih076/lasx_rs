//! FFI 调用的返回状态：`lasx_*_checked` 变体的错误通道。
//!
//! 原有 15 个 `lasx_*` 符号保持"调用方保证指针/长度合法"的 C 约定不变（误用即 UB）。
//! 需要**把错误上抛给调用方**时改用对应的 `lasx_*_checked`：它多一个
//! `int *status` 出参，函数体先做结构与物理常数校验，失败时把 [`LasxStatus`]
//! 写进去并返回一个安全的中性值（数值型返回 `0.0`/`0`，`void` 型只写状态）。
//!
//! `status` 传 `NULL` 表示"不关心状态"，此时校验仍会执行（避免 UB），只是丢失原因。
//!
//! ```c
//! int st;
//! float d = lasx_dot_checked(a, b, n, &st);
//! if (st != LASX_OK) { /* 处理错误 */ }
//! ```

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（策略见 `docs/dev.md` §17）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX/LSX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]

/// `lasx_*_checked` 的状态码。`0` 为成功，其余为失败原因。
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LasxStatus {
    /// 成功。
    Ok = 0,
    /// 指针为空且长度非 0。
    NullPointer = 1,
    /// 长度参数为负。
    NegativeLength = 2,
    /// 矩阵形状自相矛盾（如 `a.len() != m*k`）。
    BadShape = 3,
    /// 尺寸相乘溢出 `usize`。
    SizeOverflow = 4,
    /// 物理常数非正（`mu <= 0`、`re <= 0`）。
    NonPositiveConstant = 5,
    /// 物理常数为 NaN/无穷（`j2`、`dt`）。
    NonFiniteConstant = 6,
}

impl LasxStatus {
    /// 人类可读的原因（供上层拼错误消息）。
    pub fn message(self) -> &'static str {
        match self {
            LasxStatus::Ok => "成功",
            LasxStatus::NullPointer => "指针为空",
            LasxStatus::NegativeLength => "长度为负",
            LasxStatus::BadShape => "矩阵形状不匹配",
            LasxStatus::SizeOverflow => "尺寸相乘溢出",
            LasxStatus::NonPositiveConstant => "常数必须为正",
            LasxStatus::NonFiniteConstant => "常数必须是有限值",
        }
    }

    /// 是否成功。
    pub fn is_ok(self) -> bool {
        matches!(self, LasxStatus::Ok)
    }

    /// 由 C 侧写回的整数还原状态码；未知值返回 `None`。
    ///
    /// 供上层（如语言绑定）把 `int *status` 翻回枚举再映射成本语言的错误。
    pub fn from_i32(v: i32) -> Option<LasxStatus> {
        Some(match v {
            0 => LasxStatus::Ok,
            1 => LasxStatus::NullPointer,
            2 => LasxStatus::NegativeLength,
            3 => LasxStatus::BadShape,
            4 => LasxStatus::SizeOverflow,
            5 => LasxStatus::NonPositiveConstant,
            6 => LasxStatus::NonFiniteConstant,
            _ => return None,
        })
    }

    /// 写入出参（`NULL` 表示调用方不关心）。
    #[inline]
    pub(crate) fn write(self, out: *mut i32) {
        if !out.is_null() {
            // SAFETY: FFI 约定——非空时调用方保证 out 可写一个 i32。
            unsafe { *out = self as i32 };
        }
    }
}

/// `i32` 长度 → `usize`，负数报错。
#[inline]
pub(crate) fn checked_len(n: i32) -> Result<usize, LasxStatus> {
    if n < 0 {
        Err(LasxStatus::NegativeLength)
    } else {
        Ok(n as usize)
    }
}

/// 由裸指针 + 长度构造只读切片；`n > 0` 时指针不得为空。
///
/// # Safety
/// `p` 必须指向至少 `n` 个可读的 `T`。
#[inline]
pub(crate) unsafe fn checked_slice<'a, T>(p: *const T, n: usize) -> Result<&'a [T], LasxStatus> {
    if p.is_null() {
        return if n == 0 {
            Ok(&[])
        } else {
            Err(LasxStatus::NullPointer)
        };
    }
    Ok(std::slice::from_raw_parts(p, n))
}

/// 由裸指针 + 长度构造可写切片；`n > 0` 时指针不得为空。
///
/// # Safety
/// `p` 必须指向至少 `n` 个可写的 `T`，且不与其他在用引用重叠。
#[inline]
pub(crate) unsafe fn checked_slice_mut<'a, T>(
    p: *mut T,
    n: usize,
) -> Result<&'a mut [T], LasxStatus> {
    if p.is_null() {
        return if n == 0 {
            Ok(&mut [])
        } else {
            Err(LasxStatus::NullPointer)
        };
    }
    Ok(std::slice::from_raw_parts_mut(p, n))
}

/// 形状相乘，溢出报错。
#[inline]
pub(crate) fn checked_mul(a: usize, b: usize) -> Result<usize, LasxStatus> {
    a.checked_mul(b).ok_or(LasxStatus::SizeOverflow)
}

/// 物理常数必须为正且有限（`mu`、`re`）。
#[inline]
pub(crate) fn checked_positive(x: f64) -> Result<(), LasxStatus> {
    if !x.is_finite() {
        Err(LasxStatus::NonFiniteConstant)
    } else if x <= 0.0 {
        Err(LasxStatus::NonPositiveConstant)
    } else {
        Ok(())
    }
}

/// 常数必须是有限值（`j2`、`dt` 允许为负/零，但不能是 NaN/无穷）。
#[inline]
pub(crate) fn checked_finite(x: f64) -> Result<(), LasxStatus> {
    if x.is_finite() {
        Ok(())
    } else {
        Err(LasxStatus::NonFiniteConstant)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_codes_and_messages() {
        assert_eq!(LasxStatus::Ok as i32, 0);
        assert!(LasxStatus::Ok.is_ok());
        for s in [
            LasxStatus::NullPointer,
            LasxStatus::NegativeLength,
            LasxStatus::BadShape,
            LasxStatus::SizeOverflow,
            LasxStatus::NonPositiveConstant,
            LasxStatus::NonFiniteConstant,
        ] {
            assert!(!s.is_ok());
            assert!(!s.message().is_empty());
        }
    }

    #[test]
    fn test_from_i32_roundtrip() {
        for s in [
            LasxStatus::Ok,
            LasxStatus::NullPointer,
            LasxStatus::NegativeLength,
            LasxStatus::BadShape,
            LasxStatus::SizeOverflow,
            LasxStatus::NonPositiveConstant,
            LasxStatus::NonFiniteConstant,
        ] {
            assert_eq!(LasxStatus::from_i32(s as i32), Some(s));
        }
        assert_eq!(LasxStatus::from_i32(99), None);
        assert_eq!(LasxStatus::from_i32(-1), None);
    }

    #[test]
    fn test_status_write_handles_null() {
        LasxStatus::Ok.write(std::ptr::null_mut()); // 不应崩溃
        let mut out = -1i32;
        LasxStatus::BadShape.write(&mut out);
        assert_eq!(out, LasxStatus::BadShape as i32);
    }

    #[test]
    fn test_checked_len() {
        assert_eq!(checked_len(0), Ok(0));
        assert_eq!(checked_len(7), Ok(7));
        assert_eq!(checked_len(-1), Err(LasxStatus::NegativeLength));
        assert_eq!(checked_len(i32::MIN), Err(LasxStatus::NegativeLength));
    }

    #[test]
    fn test_checked_slice_null_and_zero() {
        let empty: [f32; 0] = [];
        assert_eq!(unsafe { checked_slice(empty.as_ptr(), 0) }.unwrap(), &[]);
        // 长度为 0 时空指针是合法的
        assert!(unsafe { checked_slice::<f32>(std::ptr::null(), 0) }.is_ok());
        // 长度非 0 时空指针必须报错
        assert_eq!(
            unsafe { checked_slice::<f32>(std::ptr::null(), 3) },
            Err(LasxStatus::NullPointer)
        );
        // 长度非 0 时空指针必须报错（可写）
        assert_eq!(
            unsafe { checked_slice_mut::<f32>(std::ptr::null_mut(), 3) },
            Err(LasxStatus::NullPointer)
        );
    }

    #[test]
    fn test_checked_mul_overflow() {
        assert_eq!(checked_mul(3, 4), Ok(12));
        assert_eq!(checked_mul(usize::MAX, 2), Err(LasxStatus::SizeOverflow));
    }

    #[test]
    fn test_checked_constants() {
        assert!(checked_positive(1.0).is_ok());
        assert_eq!(checked_positive(0.0), Err(LasxStatus::NonPositiveConstant));
        assert_eq!(checked_positive(-1.0), Err(LasxStatus::NonPositiveConstant));
        assert_eq!(
            checked_positive(f64::NAN),
            Err(LasxStatus::NonFiniteConstant)
        );
        assert!(checked_finite(0.0).is_ok());
        assert!(checked_finite(-3.5).is_ok());
        assert_eq!(
            checked_finite(f64::INFINITY),
            Err(LasxStatus::NonFiniteConstant)
        );
    }
}
