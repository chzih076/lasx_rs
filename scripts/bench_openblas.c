/*
 * bench_openblas.c — **可复现的跨库对照装置**：同一个二进制里跑两条矩阵乘实现。
 *
 * 为什么要有它（`docs/dev.md` §7.10 的历史问题）：那一节的 OpenBLAS 数字来自 `/tmp` 里的
 * 一次性探针，探针不进仓库 ⇒ 数字永远只是"历史记录"、不可复现。这个文件把**装置**入库，
 * 于是"本库 vs OpenBLAS"随时可以重测。
 *
 * 编译（在仓库根目录；先 `cargo build --release`）：
 *     gcc -O2 -march=native -Iinclude scripts/bench_openblas.c \
 *         target/release/liblasx_rs.a $(pkg-config --cflags --libs openblas) \
 *         -lm -lpthread -ldl -o /tmp/bench_openblas
 *   （**静态**链 `liblasx_rs.a` 是故意的：免得踩 SONAME / `liblasx_rs.so.0` 符号链接那一脚。
 *     `pkg-config --cflags openblas` 在本机指向一个已失效的构建缓存目录，gcc 会忽略它，
 *     `cblas.h` 实际来自 `/usr/include`——留着这条命令是为了别处也能照抄。）
 *
 * 用法：
 *     /tmp/bench_openblas --impl=both     --dtype=f32 --m=256 --k=256 --n=256 --threads=1
 *     /tmp/bench_openblas --impl=lasx     --dtype=f64 --m=512 --k=512 --n=512 --threads=1
 *     /tmp/bench_openblas --impl=openblas --dtype=f32 --m=192 --k=768 --n=3072 --threads=12
 *   可选 `--iters=N`（默认自适应到 ~100 ms）、`--seed=<u64>`、`--tol=<f>`（先验一致性阈值）。
 *
 * 输出（机器可读、逐行；供 `scripts/bench_openblas.sh` 解析）：
 *     OB impl=lasx dtype=f32 m=256 k=256 n=256 threads=1 iters=9 ms_med=… ms_min=… ms_max=… gflops=…
 *     OBCHECK dtype=f32 m=256 k=256 n=256 threads=1 max_diff=… max_ref=… max_rel=… tol=… ok=yes
 *
 * ---- 公平性约定（每条都能在下面的代码里对上；`docs/dev.md` §6.4 引用同一份清单）----
 *
 * 1. **同一形状、同一确定性数据、同一预热**：数据由固定种子的 LCG 生成（`fill_rand_*`，
 *    范围 `[-1,1)`），两个实现读**同一份** `a`/`b`；计时前各自热身 1 次，再按同一迭代数取样。
 *    选 `[-1,1)` 是为了让 `|C| ~ O(√k)`：先验一致性比的是"相对量级"，不会被接近 0 的元素放大。
 * 2. **布局正确**：CBLAS 是**列主序**，但两边都用**行主序**缓冲，所以显式声明布局、按行主序填
 *    leading dimension：
 *        `C(m×n) = A(m×k) · B(k×n)`，A/B/C 行主序 ⇒ `lda=k, ldb=n, ldc=n`
 *    即 `cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A, k, B, n, 0, C, n)`
 *    （注意 CBLAS 的参数次序是 `m, n, k`）。**不做转置**：转置会把对手的数据布局与访问模式
 *    一起改掉，那就不是在比同一个东西了。
 * 3. **先验一致性**（`--impl=both` 时**强制**、且在**计时之前**）：两条实现的 C 逐元素比
 *    `max|C_lasx - C_ob| / max(1, max|C_ob|)`，阈值 f32 `1e-3`、f64 `1e-9`（f32 放宽是因为
 *    累加次序/FFMA 差异会被浮点舍入放大）。不通过 ⇒ 打印原因、`exit 1`、**不进入计时**
 *    （防止"比的其实是错的东西"；这条同时证明 CBLAS 调用没写错——布局错会得到完全不同的结果）。
 * 4. **线程数对齐**：单线程档两边都 1 线程；多线程档两边都 N 线程。OpenBLAS 用
 *    `openblas_set_num_threads(N)`（本机头文件没声明它 ⇒ 见下面的 extern）+ 环境变量
 *    `OPENBLAS_NUM_THREADS`（由 `.sh` 设，双保险）。本库这边：**C ABI 不导出线程池**
 *    （`docs/ops.md` §3.1/§9 写明"多核需调用方自行切子区间"）⇒ 多线程档按该约定用
 *    **调用方行切分**（`--threads=N` 起 N 个常驻 worker，各自 `lasx_matmul` 一段行）。
 *    **不对称如实标注**：OpenBLAS 是库内线程、本库是调用方线程——这是 ABI 事实，不修饰。
 *    worker 常驻 + barrier 复用（不在计时循环里反复 `pthread_create`），否则线程创建开销会
 *    稀释多线程档的读数。
 *    **两种 C 侧切分策略（`--split=`）**，因为"怎么切"决定能不能扩展：
 *      - `--split=rows`（默认，`ops.md` §9 的直白做法）：按输出**行**切，每线程一次
 *        `lasx_matmul(rows_i, k, n, …)`。**每次调用都会把整个 B 重新打包一遍** ⇒ B 的
 *        打包量 = 线程数 × B，带宽立刻打满、几乎不扩展（实测 12 线程 = 单线程）。
 *      - `--split=cols`（C 侧可行的"plan"做法）：先把 B **转置一次**（`Bt` 是 n×k 行主序，
 *        计时**不含**这次转置，成本单独打印成 `OBPLAN`——它就是"计划"该摊掉的那部分），
 *        然后按**列**切：第 i 个 worker 拿 `Bt` 里连续的 (c1-c0)×k 段调
 *        `lasx_matmul(m, k, c1-c0, A, Bt+c0*k, tmp_i)`，再把 `tmp_i`（m×(c1-c0)，连续）
 *        散射回 C 的列区间。打包总量 = B **一遍**（各线程只打包自己那部分）⇒ 这才是能扩展的
 *        切法。（C 是行主序 ⇒ 列区间在 C 里不连续，所以必须经 `tmp_i` 散射。）
 */
#define _GNU_SOURCE
#include <lasx_rs.h>

#include <cblas.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* 本机 `cblas.h` 没声明它（只有 Fortran 侧 `f77blas.h` 有），但符号在 libopenblas 里。 */
extern void openblas_set_num_threads(int);

/* ---- 形状与缓冲：做成文件级状态，让"一次计算"只有一处实现（少犯重复分支的错）---- */
static int G_m, G_k, G_n, G_is_f64, G_nt, G_rows_per_thread;
static int G_split_cols;      /* 0 = --split=rows（默认），1 = --split=cols */
static int G_cols_per_thread; /* cols 模式：每线程负责的输出列数 */
static void *G_bt;            /* cols 模式：B 的转置（n×k 行主序），只读 */
static void *G_tmp[64];       /* cols 模式：每线程的 w×m 连续输出块（= Cᵀ 的一段）*/
static void *G_at;            /* cols 模式：A 的转置（k×m 行主序），只读 */
static const void *G_a, *G_b;
static pthread_t G_th[64];
static pthread_barrier_t G_start, G_end;
static void *G_c; /* 多线程档的当前输出缓冲 */
static int G_stop;
/* 非 ABI 的测试模式（见下面 run_scan/run_verify 的注释）：scan / verify / no-scatter */
static int G_mode_scan, G_mode_verify, G_no_scatter;
static int G_ms[16], G_nms;

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* 确定性数据（固定种子 LCG；范围 [-1,1)）。 */
static void fill_rand_f32(float *p, size_t n, uint64_t seed) {
    uint64_t s = seed;
    for (size_t i = 0; i < n; i++) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        p[i] = (float)((double)(s >> 11) / 9007199254740992.0 * 2.0 - 1.0);
    }
}
static void fill_rand_f64(double *p, size_t n, uint64_t seed) {
    uint64_t s = seed ^ 0x9e3779b97f4a7c15ull;
    for (size_t i = 0; i < n; i++) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        p[i] = (double)(s >> 11) / 9007199254740992.0 * 2.0 - 1.0;
    }
}

/* ---- 一次计算：两条实现各一处（无重复分支）---- */
static void run_lasx(void *c) {
    if (G_nt > 1) { /* 多线程档：常驻 worker 行切分（见文件头第 4 条） */
        G_c = c;
        pthread_barrier_wait(&G_start);
        pthread_barrier_wait(&G_end);
    } else if (G_is_f64) {
        lasx_matmul_f64(G_m, G_k, G_n, (const double *)G_a, (const double *)G_b, (double *)c);
    } else {
        lasx_matmul(G_m, G_k, G_n, (const float *)G_a, (const float *)G_b, (float *)c);
    }
}

static void run_ob(void *c) {
    /* 行主序 + lda=k/ldb=n/ldc=n；CBLAS 的参数次序是 m, n, k（见文件头第 2 条）。 */
    if (G_is_f64) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, G_m, G_n, G_k, 1.0,
                    (const double *)G_a, G_k, (const double *)G_b, G_n, 0.0, (double *)c, G_n);
    } else {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, G_m, G_n, G_k, 1.0f,
                    (const float *)G_a, G_k, (const float *)G_b, G_n, 0.0f, (float *)c, G_n);
    }
}

static double time_once(int do_lasx, void *c) {
    double t0 = now_s();
    if (do_lasx) {
        run_lasx(c);
    } else {
        run_ob(c);
    }
    return now_s() - t0;
}

static void *worker_fn(void *arg) {
    const int idx = (int)(intptr_t)arg;
    for (;;) {
        pthread_barrier_wait(&G_start);
        if (G_stop) {
            break;
        }
        if (G_split_cols) {
            /* cols 模式：第 idx 个 worker 负责 C 的列区间 [c0,c1)。
             * 做法：把 A/B 都转置，算 **Cᵀ = Bᵀ·Aᵀ** 的这一段（Bt 的行区间连续 ⇒ 打包量
             * 只有 B 的一遍），再散射转置回 C。`G_bt` 的第 j 行 = B 的第 j 列。 */
            const int c0 = idx * G_cols_per_thread;
            int c1 = c0 + G_cols_per_thread;
            if (c1 > G_n) {
                c1 = G_n;
            }
            const int w = c1 - c0;
            if (w > 0) {
                void *ct = G_tmp[idx]; /* w × m（= Cᵀ 的这一段） */
                if (G_is_f64) {
                    lasx_matmul_f64(w, G_k, G_m, (const double *)G_bt + (size_t)c0 * G_k,
                                    (const double *)G_at, (double *)ct);
                } else {
                    lasx_matmul(w, G_k, G_m, (const float *)G_bt + (size_t)c0 * G_k,
                                (const float *)G_at, (float *)ct);
                }
                if (!G_no_scatter) { /* --no-scatter 是**只测时间**的测试模式：结果无效 */
                    if (G_is_f64) {
                        for (int r = 0; r < G_m; r++) {
                            for (int j = 0; j < w; j++) {
                                ((double *)G_c)[(size_t)r * G_n + c0 + j] =
                                    ((double *)ct)[(size_t)j * G_m + r];
                            }
                        }
                    } else {
                        for (int r = 0; r < G_m; r++) {
                            for (int j = 0; j < w; j++) {
                                ((float *)G_c)[(size_t)r * G_n + c0 + j] =
                                    ((float *)ct)[(size_t)j * G_m + r];
                            }
                        }
                    }
                }
            }
        } else {
            /* rows 模式（默认）：按输出行切；A 的 [r0,r1) 行与 C 的 [r0,r1) 行都是连续段 */
            int r0 = idx * G_rows_per_thread;
            int r1 = r0 + G_rows_per_thread;
            if (r1 > G_m) {
                r1 = G_m;
            }
            if (r0 < r1) {
                if (G_is_f64) {
                    lasx_matmul_f64(r1 - r0, G_k, G_n, (const double *)G_a + (size_t)r0 * G_k,
                                    (const double *)G_b, (double *)G_c + (size_t)r0 * G_n);
                } else {
                    lasx_matmul(r1 - r0, G_k, G_n, (const float *)G_a + (size_t)r0 * G_k,
                                (const float *)G_b, (float *)G_c + (size_t)r0 * G_n);
                }
            }
        }
        pthread_barrier_wait(&G_end);
    }
    return NULL;
}

/* ---- 非 ABI 的**测试模式**（只为"验穿根因"，不改变本装置测量的口径）----
 *   --mode=scan   ：单线程扫 m（`--ms=192,384,…`），把 `ms(m) = a + b·m` 的**截距 a** 当
 *                   "一次调用里打包 B + 固定开销"的量级（B 固定 = k×n），从而回答
 *                   "B 被打包 12 次"值多少时间。
 *   --mode=verify ：**位精确**：单线程 `lasx_matmul` 作参考，逐位比对"调用方 N 线程按行切"
 *                   与"预转置 plan + 按列切"的结果（`memcmp`，即 to_bits 全等）。不一致 ⇒ exit 1。
 *   --no-scatter  ：**只测时间、结果无效**：`--split=cols` 里跳过"散射回 C"那一步，
 *                   用来把 236 G 拆成"（打包+计算）"与"散射"两部分。绝不用它出数值。
 */

/* 单线程扫 m：打印每个 m 的中位数与 GFLOP/s（供拟合 ms(a+b·m)） */
static void run_scan(void) {
    const size_t esz = G_is_f64 ? 8 : 4;
    for (int i = 0; i < G_nms; i++) {
        const int m = G_ms[i];
        const size_t an = (size_t)m * G_k, cn = (size_t)m * G_n;
        void *A = malloc(esz * an), *C = calloc(cn, esz);
        if (!A || !C) {
            fprintf(stderr, "scan: 分配失败（m=%d）\n", m);
            exit(2);
        }
        if (G_is_f64) {
            fill_rand_f64((double *)A, an, 20261015ull + 7);
        } else {
            fill_rand_f32((float *)A, an, 20261015ull + 7);
        }
        G_m = m;
        G_a = A;
        /* 热身 1 次 + 自适应迭代到 ~100 ms，取中位数（口径同装置的默认路径） */
        run_lasx(C);
        double one = time_once(1, C);
        int it = (int)(0.100 / (one > 0 ? one : 1e-6));
        if (it < 1) {
            it = 1;
        }
        if (it > 200) {
            it = 200;
        }
        double *s = malloc(sizeof(double) * (size_t)it);
        for (int k = 0; k < it; k++) {
            s[k] = time_once(1, C);
        }
        qsort(s, (size_t)it, sizeof(double), cmp_double);
        const double med = (it % 2) ? s[it / 2] : 0.5 * (s[it / 2 - 1] + s[it / 2]);
        printf("OBSCAN dtype=%s k=%d n=%d m=%d iters=%d ms_med=%.4f ms_min=%.4f gflops=%.2f\n",
               G_is_f64 ? "f64" : "f32", G_k, G_n, m, it, med * 1e3, s[0] * 1e3,
               2.0 * (double)m * (double)G_k * (double)G_n / med / 1e9);
        fflush(stdout);
        free(s);
        free(A);
        free(C);
    }
}

/* 位精确：单线程参考 vs（rows 切分、cols 切分）。**要求 threads>1**（否则两条切法都退化成直接调用） */
static int run_verify(void) {
    const size_t esz = G_is_f64 ? 8 : 4, cn = (size_t)G_m * G_n;
    void *ref = malloc(esz * cn), *got = malloc(esz * cn);
    if (!ref || !got) {
        fprintf(stderr, "verify: 分配失败\n");
        return 2;
    }
    memset(ref, 0, esz * cn);
    if (G_is_f64) {
        lasx_matmul_f64(G_m, G_k, G_n, (const double *)G_a, (const double *)G_b, (double *)ref);
    } else {
        lasx_matmul(G_m, G_k, G_n, (const float *)G_a, (const float *)G_b, (float *)ref);
    }
    const int save = G_split_cols;
    /* 先隔离"m 分块本身"：**无线程**、顺序地把 m 切成 2/3 块分别调，再与整块比。
     * 如果这一步就 MISMATCH，说明差异与线程/并发无关，而是"每次调用的 m 不同会走不同内部路径"。 */
    int seq_ok = 1;
    for (int blocks = 2; blocks <= 3; blocks++) {
        memset(got, 0, esz * cn);
        for (int b = 0; b < blocks; b++) {
            const int r0 = (int)((long long)G_m * b / blocks);
            const int r1 = (int)((long long)G_m * (b + 1) / blocks);
            if (r1 > r0) {
                if (G_is_f64) {
                    lasx_matmul_f64(r1 - r0, G_k, G_n, (const double *)G_a + (size_t)r0 * G_k,
                                    (const double *)G_b, (double *)got + (size_t)r0 * G_n);
                } else {
                    lasx_matmul(r1 - r0, G_k, G_n, (const float *)G_a + (size_t)r0 * G_k,
                                (const float *)G_b, (float *)got + (size_t)r0 * G_n);
                }
            }
        }
        if (memcmp(ref, got, esz * cn) != 0) {
            seq_ok = 0;
        }
    }
    int rows_ok = 0, cols_ok = 0;
    for (int pass = 0; pass < 2; pass++) {
        G_split_cols = pass; /* 0 = rows，1 = cols */
        memset(got, 0xA5, esz * cn);
        if (pass == 0 && G_nt <= 1) { /* rows 档 threads=1 时退化为直接调用，跳过（无意义） */
            continue;
        }
        G_c = got;
        run_lasx(got);
        const int same = memcmp(ref, got, esz * cn) == 0;
        if (pass == 0) {
            rows_ok = same;
        } else {
            cols_ok = same;
        }
    }
    G_split_cols = save;
    printf("OBVERIFY dtype=%s m=%d k=%d n=%d threads=%d seq_split=%s rows=%s cols=%s\n",
           G_is_f64 ? "f64" : "f32", G_m, G_k, G_n, G_nt > 0 ? G_nt : 1,
           seq_ok ? "bitexact" : "MISMATCH", rows_ok ? "bitexact" : "MISMATCH",
           cols_ok ? "bitexact" : "MISMATCH");
    fflush(stdout);
    const int ok = seq_ok && rows_ok && cols_ok;
    if (!ok) {
        fprintf(stderr,
                "bench_openblas: 位精确**不通过**（m=%d k=%d n=%d threads=%d seq=%s rows=%s "
                "cols=%s）——切分与单线程参考不一致，拒绝进入计时。\n",
                G_m, G_k, G_n, G_nt, seq_ok ? "ok" : "MISMATCH", rows_ok ? "ok" : "MISMATCH",
                cols_ok ? "ok" : "MISMATCH");
    }
    free(ref);
    free(got);
    return ok ? 0 : 1;
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "用法：%s --impl=lasx|openblas|both [--dtype=f32|f64] --m=M --k=K --n=N [--threads=N] "
            "[--iters=N] [--seed=U64] [--tol=F] [--split=rows|cols] [--no-scatter]\n"
            "      %s --mode=scan --k=K --n=N --ms=192,384,768,1536,3072   # 单线程扫 m（拟合打包成本）\n"
            "      %s --mode=verify --m=M --k=K --n=N --threads=N          # 位精确自检（rows/cols）\n",
            argv0, argv0, argv0);
}

int main(int argc, char **argv) {
    const char *impl = NULL, *dtype = "f32";
    int iters = 0, threads = 1;
    double tol = -1.0;
    uint64_t seed = 20261015ull;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strncmp(a, "--impl=", 7)) {
            impl = a + 7;
        } else if (!strncmp(a, "--dtype=", 8)) {
            dtype = a + 8;
        } else if (!strncmp(a, "--m=", 4)) {
            G_m = atoi(a + 4);
        } else if (!strncmp(a, "--k=", 4)) {
            G_k = atoi(a + 4);
        } else if (!strncmp(a, "--n=", 4)) {
            G_n = atoi(a + 4);
        } else if (!strncmp(a, "--threads=", 10)) {
            threads = atoi(a + 10);
        } else if (!strncmp(a, "--split=", 8)) {
            const char *s = a + 8;
            if (!strcmp(s, "cols")) {
                G_split_cols = 1;
            } else if (!strcmp(s, "rows")) {
                G_split_cols = 0;
            } else {
                fprintf(stderr, "bench_openblas: --split 只支持 rows|cols（收到 %s）\n", s);
                return 2;
            }
        } else if (!strncmp(a, "--iters=", 8)) {
            iters = atoi(a + 8);
        } else if (!strncmp(a, "--seed=", 7)) {
            seed = strtoull(a + 7, NULL, 10);
        } else if (!strncmp(a, "--tol=", 6)) {
            tol = atof(a + 6);
        } else if (!strncmp(a, "--mode=", 7)) {
            const char *s = a + 7;
            if (!strcmp(s, "scan")) {
                G_mode_scan = 1;
            } else if (!strcmp(s, "verify")) {
                G_mode_verify = 1;
                G_split_cols = 1; /* 让下面的建池逻辑把 At/Bt 计划也建出来 */
            } else {
                fprintf(stderr, "bench_openblas: --mode 只支持 scan|verify（收到 %s）\n", s);
                return 2;
            }
        } else if (!strncmp(a, "--ms=", 5)) {
            const char *p = a + 5;
            G_nms = 0;
            while (*p && G_nms < 16) {
                G_ms[G_nms++] = atoi(p);
                while (*p && *p != ',') {
                    p++;
                }
                if (*p == ',') {
                    p++;
                }
            }
        } else if (!strcmp(a, "--no-scatter")) { /* 只测时间：cols 档跳过散射（结果无效） */
            G_no_scatter = 1;
        } else {
            fprintf(stderr, "bench_openblas: 不认识的参数 %s\n", a);
            usage(argv[0]);
            return 2;
        }
    }
    const int do_lasx = impl && (!strcmp(impl, "lasx") || !strcmp(impl, "both"));
    const int do_ob = impl && (!strcmp(impl, "openblas") || !strcmp(impl, "both"));
    const int do_check = impl && !strcmp(impl, "both");
    /* 测试模式（scan/verify）不需要 --impl，按各自的必填项校验 */
    if (G_mode_scan) {
        if (G_nms <= 0 || G_k <= 0 || G_n <= 0) {
            usage(argv[0]);
            return 2;
        }
    } else if (G_mode_verify) {
        if (G_m <= 0 || G_k <= 0 || G_n <= 0) {
            usage(argv[0]);
            return 2;
        }
    } else if ((!do_lasx && !do_ob) || G_m <= 0 || G_k <= 0 || G_n <= 0) {
        usage(argv[0]);
        return 2;
    }
    if (threads < 1) {
        threads = 1;
    }
    if (threads > 64) {
        threads = 64;
    }
    G_is_f64 = (strcmp(dtype, "f64") == 0);
    if (tol < 0) {
        tol = G_is_f64 ? 1e-9 : 1e-3;
    }

    /* 线程对齐（第 4 条）：两个实现都用同一个 N；这是"程序内"那一半，另一半是 .sh 设的
     * `OPENBLAS_NUM_THREADS`。 */
    openblas_set_num_threads(threads);

    const size_t an = (size_t)G_m * (size_t)G_k, bn = (size_t)G_k * (size_t)G_n,
                 cn = (size_t)G_m * (size_t)G_n;
    const size_t esz = G_is_f64 ? 8 : 4;
    void *A = malloc(esz * an), *B = malloc(esz * bn);
    void *C1 = calloc(cn, esz), *C2 = calloc(cn, esz);
    if (!A || !B || !C1 || !C2) {
        fprintf(stderr, "bench_openblas: 分配失败（m=%d k=%d n=%d）\n", G_m, G_k, G_n);
        return 2;
    }
    if (G_is_f64) {
        fill_rand_f64((double *)A, an, seed);
        fill_rand_f64((double *)B, bn, seed + 1);
    } else {
        fill_rand_f32((float *)A, an, seed);
        fill_rand_f32((float *)B, bn, seed + 1);
    }
    G_a = A;
    G_b = B;

    /* 多线程档：起常驻 worker（一次，不在计时循环里反复建线程） */
    if (threads > 1) {
        /* **两种切分的参数都要备好**：`--mode=verify` 会在同一个进程里来回切 rows/cols，
         * 只备一种会让另一种用到未初始化的 rows/cols_per_thread（本装置曾在 verify 里
         * 因此报出假 MISMATCH——记在这里当教训）。*/
        G_nt = threads < G_m ? threads : G_m;
        G_rows_per_thread = (G_m + G_nt - 1) / G_nt;
        if (G_split_cols) {
            G_nt = threads < G_n ? threads : G_n;
            G_cols_per_thread = (G_n + G_nt - 1) / G_nt;
            /* "计划"成本：把 A、B 都转置（`G_at` 是 k×m、`G_bt` 是 n×k，行主序）。
             * **不计入下面的计时**（plan 该摊掉的部分），成本单独打印成 OBPLAN。 */
            G_at = malloc(esz * an);
            G_bt = malloc(esz * bn);
            if (!G_at || !G_bt) {
                fprintf(stderr, "bench_openblas: 转置缓冲分配失败\n");
                return 2;
            }
            const double t0 = now_s();
            if (G_is_f64) {
                const double *a = (const double *)A, *b = (const double *)B;
                double *at = (double *)G_at, *bt = (double *)G_bt;
                for (int j = 0; j < G_k; j++) {
                    for (int i = 0; i < G_m; i++) {
                        at[(size_t)j * G_m + i] = a[(size_t)i * G_k + j];
                    }
                }
                for (int c = 0; c < G_n; c++) {
                    for (int r = 0; r < G_k; r++) {
                        bt[(size_t)c * G_k + r] = b[(size_t)r * G_n + c];
                    }
                }
            } else {
                const float *a = (const float *)A, *b = (const float *)B;
                float *at = (float *)G_at, *bt = (float *)G_bt;
                for (int j = 0; j < G_k; j++) {
                    for (int i = 0; i < G_m; i++) {
                        at[(size_t)j * G_m + i] = a[(size_t)i * G_k + j];
                    }
                }
                for (int c = 0; c < G_n; c++) {
                    for (int r = 0; r < G_k; r++) {
                        bt[(size_t)c * G_k + r] = b[(size_t)r * G_n + c];
                    }
                }
            }
            printf("OBPLAN dtype=%s m=%d k=%d n=%d threads=%d split=cols transpose_ms=%.4f\n", dtype,
                   G_m, G_k, G_n, threads, (now_s() - t0) * 1e3);
            fflush(stdout);
            for (int i = 0; i < G_nt; i++) {
                G_tmp[i] = malloc(esz * (size_t)G_m * (size_t)G_cols_per_thread);
                if (!G_tmp[i]) {
                    fprintf(stderr, "bench_openblas: tmp 分配失败\n");
                    return 2;
                }
            }
        } else if (G_rows_per_thread <= 0) { /* 兜底：正常路径上面已经算过 */
            G_rows_per_thread = 1;
        }
        pthread_barrier_init(&G_start, NULL, (unsigned)G_nt + 1);
        pthread_barrier_init(&G_end, NULL, (unsigned)G_nt + 1);
        for (int i = 0; i < G_nt; i++) {
            pthread_create(&G_th[i], NULL, worker_fn, (void *)(intptr_t)i);
        }
    }

/* 收工：放行并回收 worker（正常返回与一致性失败两条路都要走）。 */
#define JOIN_WORKERS()                                    \
    do {                                                  \
        if (G_nt > 1) {                                   \
            G_stop = 1;                                   \
            pthread_barrier_wait(&G_start);               \
            for (int i = 0; i < G_nt; i++) {              \
                pthread_join(G_th[i], NULL);              \
            }                                             \
            pthread_barrier_destroy(&G_start);            \
            pthread_barrier_destroy(&G_end);              \
        }                                                 \
    } while (0)

    /* ---- 非 ABI 的测试模式：跑完即返回（--mode=scan / --mode=verify）---- */
    if (G_mode_verify) {
        if (G_nt <= 1) {
            fprintf(stderr, "bench_openblas: --mode=verify 需要 --threads>1\n");
            JOIN_WORKERS();
            return 2;
        }
        const int rc = run_verify();
        JOIN_WORKERS();
        return rc;
    }
    if (G_mode_scan) {
        if (G_nms <= 0 || threads > 1) {
            fprintf(stderr, "bench_openblas: --mode=scan 需要 --ms=… 且 threads=1（单线程扫 m）\n");
            JOIN_WORKERS();
            return 2;
        }
        run_scan();
        JOIN_WORKERS();
        return 0;
    }

    /* ---- 先验一致性（第 3 条）：强制、且在计时之前 ---- */
    if (do_check) {
        run_lasx(C1);
        run_ob(C2);
        double max_diff = 0.0, max_ref = 0.0;
        for (size_t i = 0; i < cn; i++) {
            double x = G_is_f64 ? ((double *)C1)[i] : (double)((float *)C1)[i];
            double y = G_is_f64 ? ((double *)C2)[i] : (double)((float *)C2)[i];
            double d = fabs(x - y);
            if (d > max_diff) {
                max_diff = d;
            }
            if (fabs(y) > max_ref) {
                max_ref = fabs(y);
            }
        }
        /* 相对**量级**：|C| ~ O(√k)，用 max(1, max|C_ob|) 做分母，避免接近 0 的元素放大比值 */
        const double denom = max_ref > 1.0 ? max_ref : 1.0;
        const double rel = max_diff / denom;
        const int ok = rel <= tol;
        printf("OBCHECK dtype=%s m=%d k=%d n=%d threads=%d max_diff=%.6e max_ref=%.6e max_rel=%.3e "
               "tol=%.1e ok=%s\n",
               dtype, G_m, G_k, G_n, threads, max_diff, max_ref, rel, tol, ok ? "yes" : "no");
        fflush(stdout);
        if (!ok) {
            fprintf(stderr,
                    "bench_openblas: 先验一致性**不通过**（max_rel=%.3e > tol=%.1e）——两条实现算的"
                    "不是同一个东西，拒绝进入计时。先查 CBLAS 的布局参数（lda/ldb/ldc）与数据类型。\n",
                    rel, tol);
            JOIN_WORKERS();
            return 1;
        }
    }

    /* ---- 计时：先热身 1 次，再按同一迭代数取样（第 1 条）---- */
    for (int which = 0; which < 2; which++) {
        const int is_lasx = (which == 0);
        if ((is_lasx && !do_lasx) || (!is_lasx && !do_ob)) {
            continue;
        }
        void *CB = is_lasx ? C1 : C2;
        (void)time_once(is_lasx, CB); /* 热身 */

        int it = iters;
        if (it <= 0) { /* 自适应：目标 ~100 ms，限 1..200 次 */
            const double one = time_once(is_lasx, CB);
            it = (int)(0.100 / (one > 0 ? one : 1e-6));
            if (it < 1) {
                it = 1;
            }
            if (it > 200) {
                it = 200;
            }
        }
        double *samples = malloc(sizeof(double) * (size_t)it);
        if (!samples) {
            JOIN_WORKERS();
            return 2;
        }
        for (int i = 0; i < it; i++) {
            samples[i] = time_once(is_lasx, CB);
        }
        qsort(samples, (size_t)it, sizeof(double), cmp_double);
        const double med =
            (it % 2) ? samples[it / 2] : 0.5 * (samples[it / 2 - 1] + samples[it / 2]);
        const double gflops = 2.0 * (double)G_m * (double)G_k * (double)G_n / med / 1e9;
        printf("OB impl=%s dtype=%s m=%d k=%d n=%d threads=%d iters=%d ms_med=%.4f ms_min=%.4f "
               "ms_max=%.4f gflops=%.2f\n",
               is_lasx ? "lasx" : "openblas", dtype, G_m, G_k, G_n, threads, it, med * 1e3,
               samples[0] * 1e3, samples[it - 1] * 1e3, gflops);
        fflush(stdout);
        free(samples);
    }

    JOIN_WORKERS();
    free(A);
    free(B);
    free(C1);
    free(C2);
    free(G_at);
    free(G_bt);
    for (int i = 0; i < G_nt; i++) {
        free(G_tmp[i]);
    }
    return 0;
}
