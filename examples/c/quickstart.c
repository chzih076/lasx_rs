/*
 * quickstart.c — lasx_rs 的最小 C 用法，也是"调用体验"的样板。
 *
 * 编译/运行见同目录 Makefile（`make run`）。
 *
 * 调用顺序**本身就是心智模型**，三步别颠倒：
 *   ① 先问版本：日志里能指明"哪一版库出的问题"（`lasx_version()` 非 NULL、静态串）；
 *   ② 再问能力：`lasx_has_lasx()`。标了 [LASX-only] 的符号（lasx_dot / lasx_sum /
 *      lasx_matmul / 全部 int8 内核 …）在没有 LASX 路径的机型上会**执行非法指令
 *      (SIGILL)**，所以这一问是**安全前提**，不是可选优化；它问的是"当前生效路径"，
 *      与算子入口同源，因此不必（也不该）自己去读 cpucfg；
 *   ③ 然后才调算子，并用 `lasx_status_message()` 把 `_checked` 的 status 翻成人话
 *      ——不必把 0..6 硬编码进调用方。
 *
 * 内存：本例演示 `lasx_alloc` → 写 → `lasx_free` 的配对。`lasx_alloc` **不会**返回
 * NULL（n == 0 给"对齐但不可解引用"的悬垂指针，分配失败直接 abort），所以这里不写
 * 无意义的空指针分支；要写的是"归还"。
 */
#include <lasx_rs.h>

#include <stdio.h>

int main(void) {
    /* ---------- ① 版本：任何时候都能安全调用 ---------- */
    printf("lasx_rs 版本：%s\n", lasx_version());

    /* ---------- ② 能力：调 [LASX-only] 之前必须问 ---------- */
    if (!lasx_has_lasx()) {
        fprintf(stderr,
                "lasx_has_lasx() == 0：本进程当前没有 LASX 路径。\n"
                "  原因可能是：机型只实现 LSX(128 位)（如 2K3000/3B6000M），\n"
                "  或本线程被强制降级（lasx_force_lsx_thread，仅 Rust 侧）。\n"
                "  本例要用的 lasx_dot 是 [LASX-only]，在这里调用会收到 SIGILL，故退出。\n"
                "  可替代：改用标了 LASX+LSX 的符号，如 lasx_norm3_batch。\n");
        return 2; /* 明确的状态码：不是崩溃，是"环境不支持" */
    }
    printf("能力检查：lasx_has_lasx() = 1 ⇒ 可以调 [LASX-only] 符号\n");

    /* ---------- ③ 调一个算子（走 _checked 入口 + 状态串） ---------- */
    const int n = 8;
    float *a = lasx_alloc(n); /* 32 字节对齐、未初始化 */
    float *b = lasx_alloc(n);
    for (int i = 0; i < n; i++) {
        a[i] = (float)(i + 1);          /* 1,2,…,8 */
        b[i] = 1.0f / (float)(i + 1);   /* 1,1/2,…,1/8 ⇒ 点积应为 8 */
    }

    int st = -1;
    float dot = lasx_dot_checked(a, b, n, &st);
    printf("lasx_dot_checked(n=%d) = %.6f   status=%d (%s)\n", n, (double)dot, st,
           lasx_status_message(st));
    if (st != LASX_OK) {
        fprintf(stderr, "lasx_dot 失败：%s\n", lasx_status_message(st));
        lasx_free(a, n);
        lasx_free(b, n);
        return 1;
    }

    /* ---------- 归还：必须用 lasx_free（不要用 free()） ---------- */
    lasx_free(a, n);
    lasx_free(b, n);
    printf("已用 lasx_free 归还两块缓冲（free() 不是 ABI 契约）\n");
    return 0;
}
