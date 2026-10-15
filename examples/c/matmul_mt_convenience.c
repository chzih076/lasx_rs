/*
 * matmul_mt_convenience.c — **便利层**的多核矩阵乘：一行搞定，并且**报告实际用了几个线程**。
 *
 * 编译/运行见同目录 Makefile。与 `matmul_mt_caller.c` 的分工：
 *
 * | | 线程由谁开 | 适合 |
 * |---|---|---|
 * | `matmul_mt_caller.c`（`lasx_matmul_run`） | **调用方**（OpenMP/线程池/协程都行） | 热循环、已有线程池、要和别的算子共用线程 |
 * | 本例（`lasx_matmul_run_threaded`） | 库内派发 | 一次性调用、只想一行搞定 |
 *
 * 便利层**不是唯一路径**，也不引入进程级状态：线程池是**每次调用现建现拆**的
 * （没有全局池、不读环境变量、没有 atexit）。所以热循环里请用 `lasx_matmul_run` + 你自己的
 * 线程；本例演示的是"图省事"的那一档。
 *
 * `threads` 语义：`<= 0` = 自动（物理核）、`1` = 串行（不建线程）、`N` = N 个 worker
 * （超过物理核会夹到物理核）。
 *
 * # 为什么一定要看 `threads_out`
 *
 * "开 12 个线程却只跑出单线程的速度"是本库踩过的真实坑（`lasx_matmul` 按行切、每线程各自
 * 重打包整个 B ⇒ 12 线程扩展比 1.08×）。现在这种退化可以直接看出来：
 * `_checked` 版会回填 **实际 worker 数**（`threads_out`）、**实际块数**（`chunks_out`）与
 * **最慢/最快一块的耗时**（`ms_max_out`/`ms_min_out`）——四个出参传 `NULL` 表示不关心。
 */
#include <lasx_rs.h>

#include <stdio.h>
#include <string.h>

#define M 192
#define K 768
#define N 3072

int main(void) {
    printf("lasx_rs 版本：%s\n", lasx_version());
    if (!lasx_has_lasx()) {
        fprintf(stderr, "lasx_has_lasx() == 0：本进程没有 LASX 路径，示例不跑。\n");
        return 2;
    }

    const size_t an = (size_t)M * K, bn = (size_t)K * N, cn = (size_t)M * N;
    float *a = lasx_alloc((int)an);
    float *b = lasx_alloc((int)bn);
    float *c = lasx_alloc((int)cn);
    float *c_ref = lasx_alloc((int)cn);
    for (size_t i = 0; i < an; i++) a[i] = (float)(i % 19) * 0.125f - 1.0f;
    for (size_t i = 0; i < bn; i++) b[i] = 0.5f - (float)(i % 23) * 0.0625f;

    int st = -1;
    lasx_matmul_plan *plan = lasx_matmul_prepare_checked(K, N, b, &st);
    if (plan == NULL) {
        fprintf(stderr, "lasx_matmul_prepare_checked 失败：%s\n", lasx_status_message(st));
        return 1;
    }

    /* ---------- 便利层：一行 + 四个自检出参 ---------- */
    int used = -1, chunks = -1;
    double ms_max = -1.0, ms_min = -1.0;
    lasx_matmul_run_threaded_checked(plan, M, a, c, 0 /* 0 = 自动（物理核） */, &used, &chunks,
                                     &ms_max, &ms_min, &st);
    if (st != 0) {
        fprintf(stderr, "lasx_matmul_run_threaded_checked 失败：%s\n", lasx_status_message(st));
        return 1;
    }
    printf("便利层（threads=0 ⇒ 自动）：实际 worker 数 = %d，块数 = %d\n", used, chunks);
    printf("每块耗时：最慢 %.3f ms / 最快 %.3f ms（块数 %d ⇒ 动态取块，不是固定等分）\n", ms_max,
           ms_min, chunks);
    if (used == 1 && chunks > 1) {
        printf("⚠️  worker 数是 1：这次是**串行**跑的（threads=0 在本机只探到 1 个物理核）。\n");
    }

    /* 显式指定线程数也试一次（1 = 串行、不建线程） */
    lasx_matmul_run_threaded_checked(plan, M, a, c, 1, &used, &chunks, &ms_max, &ms_min, &st);
    printf("便利层（threads=1）：实际 worker 数 = %d，块数 = %d（串行档不建线程）\n", used, chunks);

    /* ---------- 位精确自检 ---------- */
    lasx_matmul(M, K, N, a, b, c_ref);
    printf("位精确（vs 单线程 lasx_matmul）：%s\n",
           memcmp(c, c_ref, sizeof(float) * cn) == 0 ? "逐位一致" : "不一致（不该发生，请报 bug）");

    lasx_matmul_plan_destroy(plan);
    lasx_free(a, (int)an);
    lasx_free(b, (int)bn);
    lasx_free(c, (int)cn);
    lasx_free(c_ref, (int)cn);
    return 0;
}
