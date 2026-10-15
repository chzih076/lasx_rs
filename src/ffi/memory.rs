//! FFI 侧内存管理的 C ABI 导出。

// 本文件豁免 `clippy::undocumented_unsafe_blocks`（分层策略：内核模块在函数级统一写 SAFETY，不逐块重复）：
// 这里的 unsafe 都是"在刚校验过长度的切片上调用 LASX/LSX intrinsic"，同一组前提在
// **函数级 SAFETY 段**里统一说明；逐块重复注释只会把真正的不变量淹没。
#![allow(clippy::undocumented_unsafe_blocks)]

/// LASX 256 位访存推荐的对齐（字节）。
///
/// 32 字节的 `xvld/xvst` 只有在 32 字节边界上才不会跨 64 字节缓存行；
/// 实测 f32 内核在 L1 驻留规模上对齐可再快 **1.06×–1.31×**（n ≤ 8192），
/// 而大集（n = 65536）反而 0.92×——带宽主导。原始数据由 `cli` 的 `align` 套件现算。
pub const LASX_ALIGN: usize = 32;

/// `lasx_alloc` / `lasx_free` **共用的布局**：`n` 个 `f32`、[`LASX_ALIGN`] 字节对齐。
///
/// 抽成一个函数是"同一 layout"这条契约的**结构性保证**——分配与归还各写一份
/// `Layout::from_size_align` 是这类 API 最典型的静默错（改一处忘另一处 = 堆破坏）。
/// `n <= 0` 返回 `None`（没有可分配/可归还的布局）。
#[inline]
fn f32_layout(n: i32) -> Option<std::alloc::Layout> {
    if n <= 0 {
        return None;
    }
    let size = (n as usize).saturating_mul(std::mem::size_of::<f32>());
    // size 非 0（n > 0），align = LASX_ALIGN 是 2 的幂 ⇒ 这里不会失败。
    std::alloc::Layout::from_size_align(size, LASX_ALIGN).ok()
}

/// 分配 n 个 `f32` 的缓冲区并交出所有权（内存**未初始化**）。
///
/// C 签名：`float *lasx_alloc(int n)`
///
/// 返回的指针保证 [`LASX_ALIGN`] 字节对齐（`Vec` 只保证 4 字节，会让 LASX 的
/// 32 字节访存每次都跨缓存行）。内存未初始化；**必须用 [`lasx_free`] 按同一 `n` 归还**
/// （两个函数共用 `f32_layout`，所以 layout 不会漂）。`n == 0` 返回对齐的悬垂指针。
#[unsafe(no_mangle)]
pub extern "C" fn lasx_alloc(n: i32) -> *mut f32 {
    let Some(layout) = f32_layout(n) else {
        // 空分配（n <= 0）：返回"对齐但不可解引用"的悬垂指针（与 Vec::with_capacity(0)
        // 一致），而非 null——保持"永远非空"的历史契约。
        return LASX_ALIGN as *mut f32;
    };
    // SAFETY: layout 的 size 非 0；分配失败时走 handle_alloc_error（与 Vec 的 OOM 行为一致）。
    let p = unsafe { std::alloc::alloc(layout) };
    if p.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    p as *mut f32
}

/// 归还 [`lasx_alloc`] 分配的缓冲区。
///
/// C 签名：`void lasx_free(float *p, int n)`
///
/// `n` 必须是当初传给 `lasx_alloc` 的同一个值——两者共用 `f32_layout`，
/// 所以"同一 layout"由构造保证，而不是靠调用方猜。
///
/// # 两种 no-op（必须是 no-op，不能崩）
///
/// - `p == NULL`：空指针。
/// - `n <= 0`：包括 `lasx_alloc(0)` 返回的那个"**对齐但不可解引用**的悬垂指针"
///   （它本来就不是堆块，绝不能交给 `dealloc`），以及误传的负长度。
///
/// 与 `lasx_alloc` 对称：`lasx_alloc(0)` 不分配，`lasx_free(p, 0)` 就不归还。
///
/// **不要**用 `free()` 或 `posix_memalign` 配对的释放函数代替：库用的是 Rust 全局分配器
/// 的 `alloc/dealloc` 路径（本机是 glibc malloc，但这是实现细节、不是 ABI 契约）。
#[unsafe(no_mangle)]
pub extern "C" fn lasx_free(p: *mut f32, n: i32) {
    if p.is_null() {
        return;
    }
    let Some(layout) = f32_layout(n) else {
        return; // n <= 0：见上，含 n == 0 的悬垂指针
    };
    // SAFETY: 契约是"`p` 来自同一 `n` 的 `lasx_alloc`"；layout 与分配时逐字段相同
    // （同一个 `f32_layout`）。非空已在上方排除。
    unsafe { std::alloc::dealloc(p as *mut u8, layout) };
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
                // 归还：与分配配对（同一 layout 由 `f32_layout` 保证）
                lasx_free(p, n);
            }
        }
    }

    /// `lasx_alloc` → 写 → `lasx_free` 配对；两种 no-op 路径不得崩。
    #[test]
    fn test_lasx_free_pairs_with_alloc_and_noop_paths() {
        // ① 配对：写满整个缓冲再归还（内容读回校验，确保归还前缓冲确实归我们）
        for &n in &[1i32, 8, 37, 1024, 4096] {
            let p = lasx_alloc(n);
            assert!(!p.is_null() && (p as usize).is_multiple_of(LASX_ALIGN));
            // SAFETY: 刚分配了 n 个 f32 的独立缓冲区。
            unsafe {
                for i in 0..n as usize {
                    p.add(i).write(i as f32);
                }
                for i in [0usize, n as usize - 1] {
                    assert_eq!(p.add(i).read(), i as f32);
                }
            }
            lasx_free(p, n);
        }

        // ② no-op：NULL（任何 n）
        lasx_free(std::ptr::null_mut(), 0);
        lasx_free(std::ptr::null_mut(), 16);
        lasx_free(std::ptr::null_mut(), -4);

        // ③ no-op：`lasx_alloc(0)` 的对齐悬垂指针 + n = 0 / 负长度
        // （它从来不是堆块，交给 dealloc 就是 UB——所以这里必须只走 no-op 分支）
        let dangling = lasx_alloc(0);
        assert!(!dangling.is_null());
        assert_eq!(
            dangling as usize % LASX_ALIGN,
            0,
            "n=0 也要求对齐的悬垂指针"
        );
        lasx_free(dangling, 0);
        lasx_free(dangling, -1);

        // ④ 归还后可以再次分配同尺寸（分配器未被破坏）
        let p = lasx_alloc(64);
        lasx_free(p, 64);
    }
}
