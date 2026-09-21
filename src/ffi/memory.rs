//! FFI 侧内存管理的 C ABI 导出。

/// 分配 n 个 `f32` 的缓冲区并交出所有权（内存**未初始化**）。
///
/// C 签名：`float *lasx_alloc(int n)`
///
/// 返回的指针由调用方负责释放（本库不导出 `lasx_free`，见手册 Caveats）。
#[unsafe(no_mangle)]
pub extern "C" fn lasx_alloc(n: i32) -> *mut f32 {
    let mut v: Vec<f32> = Vec::with_capacity(n as usize);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}
