//! FFI 侧内存管理的 C ABI 导出。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（策略见 `docs/dev.md` §17）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX/LSX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]

/// LASX 256 位访存推荐的对齐（字节）。
///
/// 32 字节的 `xvld/xvst` 只有在 32 字节边界上才不会跨 64 字节缓存行；
/// 实测 f32 内核在 L1 驻留规模上对齐可再快 **1.06×–1.31×**（n ≤ 8192），
/// 而大集（n = 65536）反而 0.92×——带宽主导。完整表在 `docs/dev.md` §7.6，
/// 原始数据由 `cli` 的 `align` 套件现算。
pub const LASX_ALIGN: usize = 32;

/// 分配 n 个 `f32` 的缓冲区并交出所有权（内存**未初始化**）。
///
/// C 签名：`float *lasx_alloc(int n)`
///
/// 返回的指针保证 [`LASX_ALIGN`] 字节对齐（`Vec` 只保证 4 字节，会让 LASX 的
/// 32 字节访存每次都跨缓存行）。其余行为与旧版一致：内存未初始化、无 `lasx_free`、
/// 调用方负责归还（见手册 Caveats）。
#[unsafe(no_mangle)]
pub extern "C" fn lasx_alloc(n: i32) -> *mut f32 {
    let size = (n as usize).saturating_mul(std::mem::size_of::<f32>());
    if size == 0 {
        // 空分配：返回"对齐但不可解引用"的悬垂指针（与 Vec::with_capacity(0) 一致），
        // 而非 null——保持与旧行为相同的"永远非空"契约。
        return LASX_ALIGN as *mut f32;
    }
    let layout =
        std::alloc::Layout::from_size_align(size, LASX_ALIGN).expect("lasx_alloc: 布局非法");
    // SAFETY: layout 的 size 非 0；分配失败时走 handle_alloc_error（与 Vec 的 OOM 行为一致）。
    let p = unsafe { std::alloc::alloc(layout) };
    if p.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    p as *mut f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lasx_alloc_is_aligned() {
        for &n in &[1i32, 8, 37, 1024, 4096] {
            let p = lasx_alloc(n);
            assert!(!p.is_null(), "n={n} 不应返回 null");
            assert_eq!(
                p as usize % LASX_ALIGN,
                0,
                "n={n} 分配的指针未按 {LASX_ALIGN} 字节对齐"
            );
            // 可写性：写入首尾元素（n=1 时首尾同一元素）
            // SAFETY: 刚分配了 n 个 f32 的独立缓冲区。
            unsafe {
                p.write(1.0);
                if n > 1 {
                    p.add(n as usize - 1).write(2.0);
                    assert_eq!(p.add(n as usize - 1).read(), 2.0);
                }
                assert_eq!(p.read(), 1.0);
                // 归还：与分配时的 Layout 匹配
                let layout =
                    std::alloc::Layout::from_size_align((n as usize) * 4, LASX_ALIGN).unwrap();
                std::alloc::dealloc(p as *mut u8, layout);
            }
        }
    }

    #[test]
    fn test_lasx_alloc_zero_len_is_dangling_but_aligned() {
        let p = lasx_alloc(0);
        assert!(!p.is_null());
        assert_eq!(p as usize % LASX_ALIGN, 0);
    }
}
