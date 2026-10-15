/*
 * matmul_mt_caller.c — **调用方驱动**的多核矩阵乘：计划句柄 + 你自己的线程。
 *
 * 编译/运行见同目录 Makefile（`make run` 会把它和便利层那个示例一起跑）。
 *
 * # 这个示例演示什么
 *
 * `lasx_matmul` 每次调用都要把**整个 B 重新打包**一遍。多核下如果你给每个线程各调一次
 * `lasx_matmul`，打包量就是 `线程数 × B`（`192×768×3072` 时 12 线程 ≈ 113 MB），带宽立刻
 * 打满——实测 12 线程扩展比只有 **1.08×**（`docs/dev.md` §7.10 ⑤）。
 *
 * 计划句柄把这个开销提到调用之前：
 *   ① `lasx_matmul_prepare(k, n, b)` 打包一次，返回**只读**句柄（它会复制 b，之后你可以
 *      释放 b）；
 *   ② 每个线程调 `lasx_matmul_run(plan, rows, a + r0*k, c + r0*n)` 算**自己那一段行**
 *      ——线程怎么开、块怎么切**由你决定**（这里是 pthread；换 OpenMP / 协程 / 单线程
 *      串行都一样，库不夺控制权）；
 *   ③ `lasx_matmul_plan_destroy(plan)` 释放。
 *
 * # 两个硬约束
 * - **块要落在整行上**：`a`/`c` 传的是"这一块自己的首行"（`a + r0*k`、`c + r0*n`），
 *   行数要对得上；按列或按 `k` 拆都不是这个 API 的用法。
 * - **同一个 plan 可以被多线程并发 `run`**（`prepare` 之后它只读、各线程写自己的 `c`）；
 *   但 `prepare` / `destroy` 不能和 `run` 并发。
 *
 * # 位精确
 * 结果与单线程 `lasx_matmul` **逐位一致**（每个输出元素仍沿 k 单累加器升序；切块位置、
 * 行数是否为 4 的倍数都不改变累加次序）。本例最后自己 `memcmp` 验一遍并打印结论。
 */
#include <lasx_rs.h>

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define M 192
#define K 768
#define N 3072
#define THREADS 12

typedef struct {
    const lasx_matmul_plan *plan;
    const float *a;
    float *c;
    int r0;   /* 本线程负责的首行 */
    int rows; /* 行数 */
    double ms;
} Job;

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

static void *worker(void *arg) {
    Job *j = (Job *)arg;
    const double t0 = now_ms();
    /* 关键就这一行：算自己那段行。块首指针由调用方算，rows 由调用方定。 */
    lasx_matmul_run(j->plan, j->rows, j->a + (size_t)j->r0 * K, j->c + (size_t)j->r0 * N);
    j->ms = now_ms() - t0;
    return NULL;
}

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

    /* ---------- ① 打包一次 ---------- */
    int st = -1;
    lasx_matmul_plan *plan = lasx_matmul_prepare_checked(K, N, b, &st);
    if (plan == NULL) {
        fprintf(stderr, "lasx_matmul_prepare_checked 失败：%s\n", lasx_status_message(st));
        return 1;
    }
    printf("计划已就绪：k=%d n=%d 打包面板 %.1f MB（B 本身 %.1f MB）\n", K, N,
           (double)K * N * 4 / 1048576.0, (double)K * N * 4 / 1048576.0);

    /* ---------- ② 自己的线程：按行切块 ---------- */
    pthread_t th[THREADS];
    Job jobs[THREADS];
    const int base = M / THREADS;      /* 每线程行数（余数并给最后一个线程） */
    const int tail = M - base * (THREADS - 1);
    const double t0 = now_ms();
    for (int t = 0; t < THREADS; t++) {
        jobs[t].plan = plan;
        jobs[t].a = a;
        jobs[t].c = c;
        jobs[t].r0 = t * base;
        jobs[t].rows = (t == THREADS - 1) ? tail : base;
        jobs[t].ms = 0.0;
        if (pthread_create(&th[t], NULL, worker, &jobs[t]) != 0) {
            fprintf(stderr, "pthread_create 失败（t=%d）\n", t);
            return 1;
        }
    }
    for (int t = 0; t < THREADS; t++) {
        pthread_join(th[t], NULL);
    }
    const double ms = now_ms() - t0;

    double mn = jobs[0].ms, mx = jobs[0].ms;
    for (int t = 1; t < THREADS; t++) {
        if (jobs[t].ms < mn) mn = jobs[t].ms;
        if (jobs[t].ms > mx) mx = jobs[t].ms;
    }
    const double gflops = 2.0 * M * K * N / ms / 1e6;
    printf("调用方线程：实际用了 %d 个线程（%d 块，每块 %d 行，末块 %d 行）\n", THREADS, THREADS,
           base, tail);
    printf("总耗时 %.3f ms = %.1f GFLOP/s；单块耗时 %.3f–%.3f ms（max/median 比 %.2f）\n", ms,
           gflops, mn, mx, mx / ((mn + mx) / 2));
    printf("提示：max/min 差距大 = 块负载不均或机器被抢；想知道库内派发的档位就用\n"
           "      matmul_mt_convenience（它会报告实际 worker 数）。\n");

    /* ---------- ③ 位精确自检：与单线程整块比 ---------- */
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
