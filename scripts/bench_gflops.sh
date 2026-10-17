#!/bin/bash
# **GFLOPS 薄封装**（`docs/dev.md §7.27` 的产数入口）。
#
# **底层复用 `scripts/bench_openblas.sh`**（它的 `gflops` 模式已输出 `ms_med(min-max)` 与
# `gflops(med)` 列、逐轮 loadavg、A/B/C 分级）—— 本脚本**只做三件事**，**不实现第二套口径**：
#   1) 打标签（`<label>` 进日志名）；
#   2) 把原始日志重定向到 `/tmp/bench_gflops_<label>_*.txt`（**用 `>`，不用 `tee`**：规则 1 黑名单）；
#   3) 打印"跑完后的 loadavg"与日志路径，便于把噪声写进报告。
#
# 用法：
#   bash scripts/bench_gflops.sh <label> [rounds] [case-filter]
#     例：bash scripts/bench_gflops.sh gflops12 5 s512
#         bash scripts/bench_gflops.sh gflops12 5 1024
#
# 口径要点（沿用 `bench_openblas.sh` / `bench_openblas.c` 头部注释）：
#   - **同一时刻只跑一个实现**（绝不同进程混跑 OpenBLAS + LASX）；
#   - `nice -n 10`；`SETTLE` 默认 60s（本机 GUI 基线噪声下可降到 30，见 §7.27）；
#   - GFLOPS = 2·m·n·k / t / 1e9（由底层脚本计算，本脚本不再重算）；
#   - 多线程档由底层脚本的 case 定义决定（`openblas_set_num_threads(12)` 档见 §7.27）。
set -u
label="${1:?用法: bench_gflops.sh <label> [rounds] [case-filter]}"
rounds="${2:-5}"
filter="${3:-}"
log="/tmp/bench_gflops_${label}_log.txt"
sum="/tmp/bench_gflops_${label}_summary.txt"

echo "[bench_gflops] label=${label} rounds=${rounds} filter=${filter:-<all>}"
echo "[bench_gflops] loadavg(before) = $(cat /proc/loadavg)"

# **单进程、串行**：底层脚本内部已是进程级交替（不并发起两个实现）。
nice -n 10 bash "$(dirname "$0")/bench_openblas.sh" "$label" "$rounds" "$filter" > >(cat > "$log") 2>&1

echo "[bench_gflops] loadavg(after)  = $(cat /proc/loadavg)"
echo "[bench_gflops] 原始日志: $log"
echo "[bench_gflops] 汇总(底层脚本产物): /tmp/bench_openblas_${label}_log.txt（逐轮 loadavg）"
echo "[bench_gflops] 提示：把 loadavg(before/after) 范围与 GUI 基线说明一并写进 §7.27。" > "$sum"
