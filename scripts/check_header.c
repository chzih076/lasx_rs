/*
 * check_header.c — `include/lasx_rs.h` 与 `liblasx_rs.so` 的**逐符号 ABI 校验**。
 *
 * 五件事：
 *   ① 把**全部 94 个符号**（50 裸 + 44 `_checked`，docs/ops.md §4）的地址放进一个数组
 *      ⇒ 任何一个缺失/改名/未声明都会在**编译期或链接期**失败（不是"运行时才发现"）；
 *   ② 逐符号**调用**所有 44 个 `_checked`（零长度/NULL 入参 ⇒ 校验路径必然执行），
 *      断言状态码被写进末尾的 `int *status` 且落在 `[0, 6]` ⇒ 同时校验了
 *      "`status` 是最后一个参数"这条 ABI 约定；
 *   ③ 实测融合缩放的三条行为（§5.5 末段、§2.5 第 3 点）：
 *      alpha=1,beta=0 与 `lasx_matmul` **逐位一致**；beta=0 时**不读 c**（NaN 探针）；
 *      `_checked` 空指针 ⇒ 状态非 0 且**不触碰输出**；
 *   ④ 实测**计划句柄**（docs/c-abi.md 的"计划与句柄"一节）的三条行为：调用方按行切块、
 *      便利层 `threads=1`、便利层 `threads=0`（自动）**都与单线程 `lasx_matmul` 逐位一致**，
 *      外加句柄生命周期（`destroy(NULL)` 空操作）与 `_checked` 的边界状态码；
 *   ⑤ 实测运行时元信息与内存归还（§4.8）：`lasx_version` / `lasx_status_message` 非 NULL、
 *      `lasx_has_lasx()` 与状态码范围、`lasx_alloc`→写→`lasx_free` 配对 + 两种 no-op。
 *
 * 编译（在仓库根目录；`target/release/liblasx_rs.so` 必须已构建）：
 *     gcc -O2 -Iinclude scripts/check_header.c -Ltarget/release -llasx_rs -lm \
 *         -Wl,-rpath,$PWD/target/release -o /tmp/check_header
 * 运行前注意 **SONAME**：产物的 SONAME 是 `liblasx_rs.so.0`，而 `cargo build` 只产出
 * `liblasx_rs.so`（不建版本化文件/符号链接），所以本地跑之前先补一个：
 *     ln -sf liblasx_rs.so target/release/liblasx_rs.so.0
 * 运行：
 *     /tmp/check_header
 * 退出码：0 = 全部通过；1 = 有断言失败。
 *
 * 建议加 `-Wall -Wextra`。本文件刻意用了"函数指针 → `void *`"的转换（POSIX 定向支持，
 * `dlsym` 的返回值就是这种往返）来做到"取地址即引用"；`-Wpedantic` 下会对此告警。
 *
 * 注：`(void *)&func` 这种函数指针↔对象指针转换不是 ISO C 的严格子集，但在 POSIX 上
 * 是**定向支持**的（`dlsym` 的返回值正是这种往返）。这里只是为了"取地址即引用"，
 * 不会真的通过 void* 调用。
 */

#include "lasx_rs.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* ① 全部 94 个导出符号的地址（缺失/改名 ⇒ 编译或链接失败）             */
/* ------------------------------------------------------------------ */

#define LASX_SYMBOL_COUNT 94

static void *const g_syms[] = {
    /* --- §4.1 归约 / 稠密 / 量化 / 物理 --- */
    (void *)lasx_dot,
    (void *)lasx_dot_checked,
    (void *)lasx_sum,
    (void *)lasx_sum_checked,
    (void *)lasx_dot_f64,
    (void *)lasx_dot_f64_checked,
    (void *)lasx_axpy,
    (void *)lasx_axpy_checked,
    (void *)lasx_matmul,
    (void *)lasx_matmul_checked,
    (void *)lasx_matmul_scaled,
    (void *)lasx_matmul_scaled_checked,
    (void *)lasx_matmul_f64,
    (void *)lasx_matmul_f64_checked,
    (void *)lasx_dot_i8,
    (void *)lasx_dot_i8_checked,
    (void *)lasx_dot_q4,
    (void *)lasx_dot_q4_checked,
    (void *)lasx_alloc,
    /* --- §4.8 运行时元信息与内存归还（2026-10-15 新增的 4 个） --- */
    (void *)lasx_free,
    (void *)lasx_has_lasx,
    (void *)lasx_version,
    (void *)lasx_status_message,
    (void *)lasx_norm3_batch,
    (void *)lasx_norm3_batch_checked,
    (void *)lasx_vec3_add_scaled_batch,
    (void *)lasx_vec3_add_scaled_batch_checked,
    (void *)lasx_batch_distance2d,
    (void *)lasx_batch_distance2d_checked,
    (void *)lasx_ballistic_step,
    (void *)lasx_ballistic_step_checked,
    (void *)lasx_j2_accel_batch,
    (void *)lasx_j2_accel_batch_checked,
    (void *)lasx_rk4_j2_step_batch,
    (void *)lasx_rk4_j2_step_batch_checked,
    /* --- §4.2 批量姿态 / 几何 --- */
    (void *)lasx_cross3_batch,
    (void *)lasx_cross3_batch_checked,
    (void *)lasx_unitize3_batch,
    (void *)lasx_unitize3_batch_checked,
    (void *)lasx_mat3_mul_vec3_batch,
    (void *)lasx_mat3_mul_vec3_batch_checked,
    (void *)lasx_quat_normalize_batch,
    (void *)lasx_quat_normalize_batch_checked,
    (void *)lasx_quat_mul_batch,
    (void *)lasx_quat_mul_batch_checked,
    (void *)lasx_quat_rotate_batch,
    (void *)lasx_quat_rotate_batch_checked,
    (void *)lasx_quat_to_dcm_batch,
    (void *)lasx_quat_to_dcm_batch_checked,
    /* --- N1：NN 侧（§4.6） --- */
    (void *)lasx_softmax_rows,
    (void *)lasx_softmax_rows_checked,
    (void *)lasx_rms_norm,
    (void *)lasx_rms_norm_checked,
    (void *)lasx_silu,
    (void *)lasx_silu_checked,
    (void *)lasx_gelu_quick,
    (void *)lasx_gelu_quick_checked,
    (void *)lasx_gelu_erf,
    (void *)lasx_gelu_erf_checked,
    (void *)lasx_rope,
    (void *)lasx_rope_checked,
    (void *)lasx_dot_f16,
    (void *)lasx_dot_f16_checked,
    (void *)lasx_gemv_f16,
    (void *)lasx_gemv_f16_checked,
    /* --- N3：int8 推理（§4.3） --- */
    (void *)lasx_amax,
    (void *)lasx_amax_checked,
    (void *)lasx_absmax_rows,
    (void *)lasx_absmax_rows_checked,
    (void *)lasx_quantize_i8_per_tensor,
    (void *)lasx_quantize_i8_per_tensor_checked,
    (void *)lasx_quantize_i8_per_row,
    (void *)lasx_quantize_i8_per_row_checked,
    (void *)lasx_dequantize_i8,
    (void *)lasx_dequantize_i8_checked,
    (void *)lasx_dequantize_i8_rows,
    (void *)lasx_dequantize_i8_rows_checked,
    (void *)lasx_gemv_i8,
    (void *)lasx_gemv_i8_checked,
    (void *)lasx_matmul_i8,
    (void *)lasx_matmul_i8_checked,
    /* --- ONNX 契约补充（§4.6） --- */
    (void *)lasx_layer_norm,
    (void *)lasx_layer_norm_checked,
    (void *)lasx_gather_rows,
    (void *)lasx_gather_rows_checked,
    (void *)lasx_gather_rows_i8,
    (void *)lasx_gather_rows_i8_checked,
    /* --- 矩阵乘"计划句柄"（docs/c-abi.md 的"计划与句柄"一节）---
     * `lasx_matmul_plan` 是**不透明类型**：这里只取函数地址，绝不解引用句柄。 */
    (void *)lasx_matmul_prepare,
    (void *)lasx_matmul_prepare_checked,
    (void *)lasx_matmul_run,
    (void *)lasx_matmul_run_checked,
    (void *)lasx_matmul_run_threaded,
    (void *)lasx_matmul_run_threaded_checked,
    (void *)lasx_matmul_plan_destroy,
};

/* 数量在**编译期**钉住：漏写/多写一个都会在这里失败。 */
typedef char lasx_symbol_count_check
    [(sizeof(g_syms) / sizeof(g_syms[0]) == LASX_SYMBOL_COUNT) ? 1 : -1];

/* ------------------------------------------------------------------ */
/* 断言工具                                                            */
/* ------------------------------------------------------------------ */

static int g_failures = 0;

static void check(int ok, const char *what) {
    printf("%-58s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok) {
        g_failures++;
    }
}

/* ------------------------------------------------------------------ */
/* ② 41 个 _checked 符号：零长/NULL 调用 ⇒ 状态必须写进末尾的 status   */
/* ------------------------------------------------------------------ */

static int status_valid(int st) { return st >= LASX_OK && st <= LASX_NON_FINITE_CONSTANT; }

static void smoke_checked_symbols(void) {
    int st;

    st = 999; lasx_dot_checked(NULL, NULL, 0, &st);                          check(status_valid(st), "call lasx_dot_checked");
    st = 999; lasx_sum_checked(NULL, 0, &st);                                check(status_valid(st), "call lasx_sum_checked");
    st = 999; lasx_dot_f64_checked(NULL, NULL, 0, &st);                      check(status_valid(st), "call lasx_dot_f64_checked");
    st = 999; lasx_axpy_checked(0.0f, NULL, NULL, 0, &st);                   check(status_valid(st), "call lasx_axpy_checked");
    st = 999; lasx_matmul_checked(0, 0, 0, NULL, NULL, NULL, &st);           check(status_valid(st), "call lasx_matmul_checked");
    st = 999; lasx_matmul_scaled_checked(0, 0, 0, 0.0f, NULL, NULL, 0.0f, NULL, &st);
                                                                             check(status_valid(st), "call lasx_matmul_scaled_checked");
    st = 999; lasx_matmul_f64_checked(0, 0, 0, NULL, NULL, NULL, &st);       check(status_valid(st), "call lasx_matmul_f64_checked");
    st = 999; lasx_dot_i8_checked(NULL, NULL, 0, &st);                       check(status_valid(st), "call lasx_dot_i8_checked");
    st = 999; lasx_dot_q4_checked(NULL, NULL, NULL, NULL, 0, &st);           check(status_valid(st), "call lasx_dot_q4_checked");
    st = 999; lasx_norm3_batch_checked(NULL, NULL, NULL, NULL, 0, &st);      check(status_valid(st), "call lasx_norm3_batch_checked");
    st = 999; lasx_vec3_add_scaled_batch_checked(NULL, NULL, NULL, NULL, NULL, NULL, 0.0,
                                                 NULL, NULL, NULL, 0, &st);
                                                                             check(status_valid(st), "call lasx_vec3_add_scaled_batch_checked");
    st = 999; lasx_batch_distance2d_checked(0.0f, 0.0f, NULL, NULL, NULL, 0, &st);
                                                                             check(status_valid(st), "call lasx_batch_distance2d_checked");
    st = 999; lasx_ballistic_step_checked(NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0, 0.0f,
                                          0.0f, &st);
                                                                             check(status_valid(st), "call lasx_ballistic_step_checked");
    st = 999; lasx_j2_accel_batch_checked(NULL, NULL, NULL, 0.0, 0.0, 0.0, NULL, NULL, NULL, 0, &st);
                                                                             check(status_valid(st), "call lasx_j2_accel_batch_checked");
    st = 999; lasx_rk4_j2_step_batch_checked(NULL, NULL, NULL, NULL, NULL, NULL, 0.0, 0.0, 0.0,
                                             0.0, 0, &st);
                                                                             check(status_valid(st), "call lasx_rk4_j2_step_batch_checked");
    st = 999; lasx_cross3_batch_checked(NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0, &st);
                                                                             check(status_valid(st), "call lasx_cross3_batch_checked");
    st = 999; lasx_unitize3_batch_checked(NULL, NULL, NULL, NULL, NULL, NULL, 0, &st);
                                                                             check(status_valid(st), "call lasx_unitize3_batch_checked");
    st = 999; lasx_mat3_mul_vec3_batch_checked(NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                                               NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0, &st);
                                                                             check(status_valid(st), "call lasx_mat3_mul_vec3_batch_checked");
    st = 999; lasx_quat_normalize_batch_checked(NULL, NULL, NULL, NULL, 0, &st);
                                                                             check(status_valid(st), "call lasx_quat_normalize_batch_checked");
    st = 999; lasx_quat_mul_batch_checked(NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                                          NULL, NULL, NULL, NULL, 0, &st);
                                                                             check(status_valid(st), "call lasx_quat_mul_batch_checked");
    st = 999; lasx_quat_rotate_batch_checked(NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                                             NULL, NULL, NULL, 0, &st);
                                                                             check(status_valid(st), "call lasx_quat_rotate_batch_checked");
    st = 999; lasx_quat_to_dcm_batch_checked(NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                                             NULL, NULL, NULL, NULL, NULL, 0, &st);
                                                                             check(status_valid(st), "call lasx_quat_to_dcm_batch_checked");
    st = 999; lasx_softmax_rows_checked(NULL, NULL, NULL, 0, 0, 0.0f, &st);
                                                                             check(status_valid(st), "call lasx_softmax_rows_checked");
    st = 999; lasx_rms_norm_checked(NULL, NULL, NULL, 0, 0, 0.0f, &st);      check(status_valid(st), "call lasx_rms_norm_checked");
    st = 999; lasx_silu_checked(NULL, NULL, 0, &st);                         check(status_valid(st), "call lasx_silu_checked");
    st = 999; lasx_gelu_quick_checked(NULL, NULL, 0, &st);                   check(status_valid(st), "call lasx_gelu_quick_checked");
    st = 999; lasx_gelu_erf_checked(NULL, NULL, 0, &st);                     check(status_valid(st), "call lasx_gelu_erf_checked");
    st = 999; lasx_rope_checked(NULL, NULL, NULL, NULL, 0, 0, 0, 0, &st);    check(status_valid(st), "call lasx_rope_checked");
    st = 999; lasx_dot_f16_checked(NULL, NULL, 0, &st);                      check(status_valid(st), "call lasx_dot_f16_checked");
    st = 999; lasx_gemv_f16_checked(NULL, NULL, NULL, 0, 0, &st);            check(status_valid(st), "call lasx_gemv_f16_checked");
    st = 999; lasx_amax_checked(NULL, 0, &st);                               check(status_valid(st), "call lasx_amax_checked");
    st = 999; lasx_absmax_rows_checked(NULL, 0, 0, NULL, &st);               check(status_valid(st), "call lasx_absmax_rows_checked");
    st = 999; lasx_quantize_i8_per_tensor_checked(NULL, NULL, 0, &st);       check(status_valid(st), "call lasx_quantize_i8_per_tensor_checked");
    st = 999; lasx_quantize_i8_per_row_checked(NULL, NULL, NULL, 0, 0, &st); check(status_valid(st), "call lasx_quantize_i8_per_row_checked");
    st = 999; lasx_dequantize_i8_checked(NULL, 0.0f, NULL, 0, &st);          check(status_valid(st), "call lasx_dequantize_i8_checked");
    st = 999; lasx_dequantize_i8_rows_checked(NULL, NULL, NULL, 0, 0, &st);  check(status_valid(st), "call lasx_dequantize_i8_rows_checked");
    st = 999; lasx_gemv_i8_checked(NULL, NULL, NULL, 0.0f, NULL, 0, 0, &st); check(status_valid(st), "call lasx_gemv_i8_checked");
    st = 999; lasx_matmul_i8_checked(NULL, NULL, NULL, NULL, NULL, 0, 0, 0, &st);
                                                                             check(status_valid(st), "call lasx_matmul_i8_checked");
    st = 999; lasx_layer_norm_checked(NULL, NULL, NULL, NULL, 0, 0, 0.0f, &st);
                                                                             check(status_valid(st), "call lasx_layer_norm_checked");
    st = 999; lasx_gather_rows_checked(NULL, NULL, NULL, 0, 1, 1, &st);      check(status_valid(st), "call lasx_gather_rows_checked");
    st = 999; lasx_gather_rows_i8_checked(NULL, NULL, NULL, NULL, 0, 1, 1, &st);
                                                                             check(status_valid(st), "call lasx_gather_rows_i8_checked");
}

/* ------------------------------------------------------------------ */
/* ③ 行为核对：matmul / matmul_scaled / _checked                        */
/* ------------------------------------------------------------------ */

#define MM_M 5
#define MM_K 7
#define MM_N 6

static void behavior_checks(void) {
    float a[MM_M * MM_K];
    float b[MM_K * MM_N];
    float c_ref[MM_M * MM_N];
    float c_scaled[MM_M * MM_N];

    for (int i = 0; i < MM_M * MM_K; i++) a[i] = (float)i * 0.25f - 1.0f;
    for (int i = 0; i < MM_K * MM_N; i++) b[i] = 0.5f - (float)i * 0.125f;

    /* (a) alpha=1,beta=0 与 lasx_matmul 逐位一致（memcmp） */
    lasx_matmul(MM_M, MM_K, MM_N, a, b, c_ref);
    memset(c_scaled, 0xA5, sizeof(c_scaled)); /* 先弄脏，确认 beta=0 会整体覆盖 */
    lasx_matmul_scaled(MM_M, MM_K, MM_N, 1.0f, a, b, 0.0f, c_scaled);
    check(memcmp(c_ref, c_scaled, sizeof(c_ref)) == 0,
          "matmul_scaled(alpha=1,beta=0) bitwise == matmul");

    /* (b) beta=0 时**不读 c**：c 填 NaN，输出必须全有限 */
    for (int i = 0; i < MM_M * MM_N; i++) c_scaled[i] = NAN;
    lasx_matmul_scaled(MM_M, MM_K, MM_N, 1.5f, a, b, 0.0f, c_scaled);
    {
        int all_finite = 1;
        for (int i = 0; i < MM_M * MM_N; i++) {
            if (!isfinite(c_scaled[i])) all_finite = 0;
        }
        check(all_finite, "matmul_scaled(beta=0) does not read c (NaN probe)");
    }

    /* (c) _checked 空指针（长度非 0）⇒ 状态非 0，且**不触碰输出** */
    {
        int st = 999;
        int untouched = 1;
        for (int i = 0; i < MM_M * MM_N; i++) c_scaled[i] = 7.5f;
        lasx_matmul_scaled_checked(MM_M, MM_K, MM_N, 1.5f, NULL, b, 0.5f, c_scaled, &st);
        for (int i = 0; i < MM_M * MM_N; i++) {
            if (c_scaled[i] != 7.5f) untouched = 0;
        }
        check(st == LASX_NULL_POINTER, "matmul_scaled_checked(NULL a) -> LASX_NULL_POINTER");
        check(untouched, "matmul_scaled_checked(NULL a) leaves output untouched");
    }

    /* (d) _checked 成功路径与裸符号逐位一致，且写 Ok */
    {
        int st = 999;
        memset(c_ref, 0, sizeof(c_ref));
        for (int i = 0; i < MM_M * MM_N; i++) c_scaled[i] = 3.25f;
        for (int i = 0; i < MM_M * MM_N; i++) c_ref[i] = 3.25f;
        lasx_matmul_scaled(MM_M, MM_K, MM_N, 1.25f, a, b, 0.5f, c_ref);
        lasx_matmul_scaled_checked(MM_M, MM_K, MM_N, 1.25f, a, b, 0.5f, c_scaled, &st);
        check(st == LASX_OK, "matmul_scaled_checked(ok) -> LASX_OK");
        check(memcmp(c_ref, c_scaled, sizeof(c_ref)) == 0,
              "matmul_scaled_checked bitwise == matmul_scaled");
    }

    /* (e) status = NULL 合法（校验仍执行，不 deref 空指针） */
    {
        int untouched = 1;
        for (int i = 0; i < MM_M * MM_N; i++) c_scaled[i] = -1.0f;
        lasx_matmul_scaled_checked(MM_M, MM_K, MM_N, 1.0f, NULL, b, 0.0f, c_scaled, NULL);
        for (int i = 0; i < MM_M * MM_N; i++) {
            if (c_scaled[i] != -1.0f) untouched = 0;
        }
        check(untouched, "matmul_scaled_checked(status=NULL) still validates");
    }
}

/* ------------------------------------------------------------------ */
/* ④ 运行时元信息与内存归还（§4.8）                                     */
/* ------------------------------------------------------------------ */

static void meta_and_memory_checks(void) {
    /* (a) 版本：永不为 NULL、非空 */
    {
        const char *v = lasx_version();
        check(v != NULL, "lasx_version() != NULL");
        check(v != NULL && strlen(v) > 0, "lasx_version() non-empty");
        check(lasx_version() == v, "lasx_version() returns static data (stable pointer)");
    }

    /* (b) 能力查询：0/1 二值；本机（含 LASX 的 CI 机）应为 1 */
    {
        int h = lasx_has_lasx();
        check(h == 0 || h == 1, "lasx_has_lasx() in {0,1}");
        check(h == 1, "lasx_has_lasx() == 1 on this LASX machine");
    }

    /* (c) 状态串：0..6 + 未知码全部非 NULL、非空 */
    {
        int ok = 1;
        for (int st = -1; st <= 8; st++) {
            const char *m = lasx_status_message(st);
            if (m == NULL || strlen(m) == 0) ok = 0;
        }
        check(ok, "lasx_status_message(-1..8) never NULL / never empty");
        check(strcmp(lasx_status_message(LASX_OK), lasx_status_message(0)) == 0,
              "lasx_status_message is stable per code");
        check(strcmp(lasx_status_message(999), lasx_status_message(-7)) == 0,
              "unknown codes share one generic message");
    }

    /* (d) lasx_alloc → 写 → lasx_free 配对；两种 no-op 不得崩 */
    {
        const int n = 257;
        float *p = lasx_alloc(n);
        check(p != NULL && ((uintptr_t)p % 32) == 0, "lasx_alloc(257) 32-byte aligned");
        if (p != NULL) {
            for (int i = 0; i < n; i++) p[i] = (float)i * 0.5f;
            check(p[0] == 0.0f && p[n - 1] == (float)(n - 1) * 0.5f,
                  "lasx_alloc buffer is writable/readable");
            lasx_free(p, n); /* 配对归还（同一 layout） */
            check(1, "lasx_free(p, n) returns without crashing");
        }
        /* no-op：NULL（任意 n）与 n <= 0（含 alloc(0) 的对齐悬垂指针） */
        lasx_free(NULL, 0);
        lasx_free(NULL, 64);
        lasx_free(NULL, -3);
        float *dangling = lasx_alloc(0);
        check(dangling != NULL && ((uintptr_t)dangling % 32) == 0,
              "lasx_alloc(0) -> aligned dangling pointer (historical contract)");
        lasx_free(dangling, 0);
        lasx_free(dangling, -1);
        check(1, "lasx_free no-op paths (NULL / n<=0 / dangling) do not crash");
    }
}

/* ------------------------------------------------------------------ */
/* ④ 计划句柄：位精确 + 生命周期（docs/c-abi.md 的"计划与句柄"一节）  */
/* ------------------------------------------------------------------ */

/* 每个形状：参考（单线程整块） vs ①调用方按行切 ②便利层 threads=1 ③便利层 threads=0(自动) */
static void plan_checks_shape(int m, int k, int n, int block) {
    float *a = malloc(sizeof(float) * (size_t)m * k);
    float *b = malloc(sizeof(float) * (size_t)k * n);
    float *c_ref = malloc(sizeof(float) * (size_t)m * n);
    float *c_cur = malloc(sizeof(float) * (size_t)m * n);
    char what[128];
    if (!a || !b || !c_ref || !c_cur) {
        check(0, "plan_checks: malloc");
        free(a); free(b); free(c_ref); free(c_cur);
        return;
    }
    for (long i = 0; i < (long)m * k; i++) a[i] = (float)(i % 17) * 0.25f - 2.0f;
    for (long i = 0; i < (long)k * n; i++) b[i] = 0.5f - (float)(i % 13) * 0.125f;

    int st = 999;
    lasx_matmul(m, k, n, a, b, c_ref); /* 参考：单线程整块 */

    lasx_matmul_plan *plan = lasx_matmul_prepare_checked(k, n, b, &st);
    snprintf(what, sizeof(what), "plan prepare(%dx%dx%d) status==0", m, k, n);
    check(plan != NULL && st == LASX_OK, what);
    if (plan == NULL) {
        free(a); free(b); free(c_ref); free(c_cur);
        return;
    }

    /* ① 调用方按行切块：块大小故意不整除（块首指针由调用方算） */
    memset(c_cur, 0xA5, sizeof(float) * (size_t)m * n);
    for (int r0 = 0; r0 < m; r0 += block) {
        int rows = (m - r0 < block) ? (m - r0) : block;
        lasx_matmul_run(plan, rows, a + (size_t)r0 * k, c_cur + (size_t)r0 * n);
    }
    snprintf(what, sizeof(what), "plan caller-split %d rows vs matmul (%dx%dx%d)", block, m, k, n);
    check(memcmp(c_ref, c_cur, sizeof(float) * (size_t)m * n) == 0, what);

    /* ② 便利层 threads=1（串行、不建线程） */
    {
        int used = -1, chunks = -1;
        double mx = -1.0, mn = -1.0;
        memset(c_cur, 0xA5, sizeof(float) * (size_t)m * n);
        st = 999;
        lasx_matmul_run_threaded_checked(plan, m, a, c_cur, 1, &used, &chunks, &mx, &mn, &st);
        snprintf(what, sizeof(what), "plan threaded threads=1 status==0 && used==1 (%dx%dx%d)", m, k,
                 n);
        check(st == LASX_OK && used == 1, what);
        snprintf(what, sizeof(what), "plan threaded threads=1 vs matmul (%dx%dx%d)", m, k, n);
        check(memcmp(c_ref, c_cur, sizeof(float) * (size_t)m * n) == 0, what);
    }

    /* ③ 便利层 threads=0 = 自动（物理核）：结果仍须逐位一致，且要报告实际 worker 数 */
    {
        int used = -1, chunks = -1;
        double mx = -1.0, mn = -1.0;
        memset(c_cur, 0xA5, sizeof(float) * (size_t)m * n);
        st = 999;
        lasx_matmul_run_threaded_checked(plan, m, a, c_cur, 0, &used, &chunks, &mx, &mn, &st);
        snprintf(what, sizeof(what),
                 "plan threaded auto status==0 && used>=1 && chunks>=1 (%dx%dx%d)", m, k, n);
        check(st == LASX_OK && used >= 1 && chunks >= 1, what);
        printf("    [%dx%dx%d threads=0] used=%d chunks=%d ms_max=%.3f ms_min=%.3f\n", m, k, n, used,
               chunks, mx, mn);
        snprintf(what, sizeof(what), "plan threaded auto vs matmul (%dx%dx%d)", m, k, n);
        check(memcmp(c_ref, c_cur, sizeof(float) * (size_t)m * n) == 0, what);
    }

    /* 生命周期 + 边界状态码 */
    {
        int bad = 999;
        lasx_matmul_run_checked(NULL, 4, a, c_cur, &bad);
        check(bad == LASX_NULL_POINTER, "plan run_checked(plan=NULL) -> NullPointer");
        bad = 999;
        lasx_matmul_run_checked(plan, -1, a, c_cur, &bad);
        check(bad == LASX_NEGATIVE_LENGTH, "plan run_checked(rows=-1) -> NegativeLength");
        bad = 999;
        lasx_matmul_run_checked(plan, 4, NULL, c_cur, &bad);
        check(bad == LASX_NULL_POINTER, "plan run_checked(a=NULL) -> NullPointer");
        /* 失败不触碰输出：把 c 弄脏，长度错的那次调用之后应原样 */
        {
            float probe = c_cur[0];
            bad = 999;
            lasx_matmul_run_checked(plan, 4, a, NULL, &bad);
            check(bad == LASX_NULL_POINTER && c_cur[0] == probe,
                  "plan run_checked(c=NULL) fails without touching outputs");
        }
        bad = 999;
        lasx_matmul_prepare_checked(-1, 8, b, &bad);
        check(bad == LASX_NEGATIVE_LENGTH, "plan prepare_checked(k=-1) -> NegativeLength");
        bad = 999;
        lasx_matmul_prepare_checked(4, 8, NULL, &bad);
        check(bad == LASX_NULL_POINTER, "plan prepare_checked(b=NULL) -> NullPointer");
        /* 出参全 NULL + status=NULL 不得崩（裸入口语义） */
        lasx_matmul_run_threaded(plan, m, a, c_cur, 2);
        lasx_matmul_run_threaded_checked(plan, m, a, c_cur, 2, NULL, NULL, NULL, NULL, NULL);
        lasx_matmul_run_checked(plan, 2, a, c_cur, NULL);
        check(1, "plan NULL out-params / NULL status do not crash");
        lasx_matmul_plan_destroy(NULL);
        check(1, "plan destroy(NULL) is a no-op");
        lasx_matmul_plan_destroy(plan);
        check(1, "plan destroy(plan) returns without crashing");
    }

    free(a);
    free(b);
    free(c_ref);
    free(c_cur);
}

static void plan_checks(void) {
    /* k 取 384/385（打包路径的 k 分块边界两侧）；m=40 不被 7 整除；n=257 不是 32 的倍数
     * ⇒ 同时覆盖"列尾副本"那条路。 */
    plan_checks_shape(40, 384, 257, 7);
    plan_checks_shape(40, 385, 257, 7);
}

/* ------------------------------------------------------------------ */

int main(void) {
    const size_t n_syms = sizeof(g_syms) / sizeof(g_syms[0]);

    printf("lasx_rs C ABI check\n");
    printf("  头文件符号数（编译期钉住）：%d\n", LASX_SYMBOL_COUNT);
    printf("  数组实际条目数              ：%zu\n", n_syms);

    {
        int all_non_null = 1;
        for (size_t i = 0; i < n_syms; i++) {
            if (g_syms[i] == NULL) all_non_null = 0;
        }
        check(n_syms == LASX_SYMBOL_COUNT && all_non_null,
              "all 94 symbol addresses resolved at link time");
    }

    check(LASX_OK == 0 && LASX_NULL_POINTER == 1 && LASX_NEGATIVE_LENGTH == 2 &&
              LASX_BAD_SHAPE == 3 && LASX_SIZE_OVERFLOW == 4 &&
              LASX_NON_POSITIVE_CONSTANT == 5 && LASX_NON_FINITE_CONSTANT == 6,
          "lasx_status codes match docs/ops.md 4.5");

    smoke_checked_symbols();
    behavior_checks();
    plan_checks();
    meta_and_memory_checks();

    printf("\n%s：%d 项失败\n", g_failures == 0 ? "结论 OK" : "结论 FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}
