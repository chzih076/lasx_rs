#!/usr/bin/env bash
# **性能比赛·测试台第一批**：DSL 算子（`ops_dsl::*` + `LasxBackend`）vs LASX 库算子（`lasx_rs::api::*`）。
#
# 做法：**进程级交替 A/B**——每个 (op,threads) 跑 ROUNDS 轮，每轮先后各起一个进程
# （`--arm=lasx` / `--arm=dsl`，顺序逐轮对调以抵消漂移）；**丢弃首轮**，其余取中位与极差。
# 每进程内再跑 `--iters` 次取中位（进程内抖动由 example 处理）。输出 markdown 表。
#
# 用法：
#   bash scripts/dsl_vs_lasx_ab.sh [--ops matmul,dot,...] [--threads 1,12] [--rounds 6] [--iters 3]
#
# 说明：
# * 是**趋势台架**：本机 loadavg > 3 时表头会标「趋势」，数字只用于同窗口比较、不与历史表横比；
# * 线程控制：LASX 臂 `pool::init_global(N)`；DSL 臂另设 `SIMD_DSL_ENGINE_THREADS=N`
#   （该 env 引擎只读一次 ⇒ example 每进程只跑一个 threads 值，脚本因此每轮各起一个进程）；
# * 公平性/等价类由 example 保证并在 `RESULT` 行打印（计时区外生成输入与校验、两臂预分配、
#   无 prepare/打包、cmatmul 两臂都走 Direct 路线）。
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || exit 1

OPS="matmul,dot,sum,softmax_rows,rms_norm,layer_norm,gelu_quick,axpy,absmax_rows,cmatmul"
THREADS="1,12"
ROUNDS=6
ITERS=3
BIN="target/release/examples/dsl_vs_lasx_ab"

while [ $# -gt 0 ]; do
    case "$1" in
    --ops) OPS="${2:-}"; shift 2 ;;
    --ops=*) OPS="${1#--ops=}"; shift ;;
    --threads) THREADS="${2:-}"; shift 2 ;;
    --threads=*) THREADS="${1#--threads=}"; shift ;;
    --rounds) ROUNDS="${2:-6}"; shift 2 ;;
    --rounds=*) ROUNDS="${1#--rounds=}"; shift ;;
    --iters) ITERS="${2:-3}"; shift 2 ;;
    --iters=*) ITERS="${1#--iters=}"; shift ;;
    -h | --help)
        echo "用法：bash scripts/dsl_vs_lasx_ab.sh [--ops a,b] [--threads 1,12] [--rounds 6] [--iters 3]"
        exit 0
        ;;
    *)
        echo "未知参数：$1" >&2
        exit 2
        ;;
    esac
done

if [ "$ROUNDS" -lt 5 ]; then
    echo "--rounds 必须 ≥5（要求：进程级交替 ≥5 轮且丢首轮）" >&2
    exit 2
fi
if [ "$ROUNDS" -le 6 ]; then
    : # 6 轮丢首轮后剩 5 个样本 ⇒ 中位 = 排序后第 3 个（无需浮点平均，脚本纯整数）
else
    echo "# 注意：--rounds=$ROUNDS > 6；本脚本中位取「排序后中间样本」，偶数样本时取左侧那个" >&2
fi

# shellcheck disable=SC2086
NUM_THREADS=$(printf '%s' "$THREADS" | tr ',' '\n' | grep -c '[0-9]')
if [ "$NUM_THREADS" -lt 1 ]; then
    echo "--threads 需要形如 1,12" >&2
    exit 2
fi

echo "== 构建 example（release + dsl-backend）=="
if ! cargo build --release --example dsl_vs_lasx_ab --features dsl-backend >/tmp/dsl_vs_lasx_build.log 2>&1; then
    echo "构建失败：见 /tmp/dsl_vs_lasx_build.log" >&2
    tail -20 /tmp/dsl_vs_lasx_build.log >&2
    exit 1
fi
if [ ! -x "$BIN" ]; then
    echo "构建产物不存在：$BIN" >&2
    exit 1
fi

read -r LOADAVG _ < /proc/loadavg || LOADAVG="?"
LOAD_INT=${LOADAVG%%.*}
TREND=""
if [ "${LOAD_INT:-0}" -gt 3 ] 2>/dev/null; then
    TREND="（**趋势**：loadavg=${LOADAVG} > 3）"
fi

# ---- 纯 bash 取值工具（不使用 awk/sed/python；只 grep/cut/sort）----
# 从 example 的 `RESULT ...` 行里取 key 的值（key 形如 median_ns / eq / work）
field() {
    # $1 = 输出文本，$2 = key
    printf '%s\n' "$1" | grep -o "$2=[^ ]*" | head -1 | cut -d= -f2
}
# ns → "1.234"（毫秒，3 位小数，纯整数运算）
ms3() {
    local ns=$1
    printf '%d.%03d' $((ns / 1000000)) $(((ns % 1000000) / 1000))
}
# ns → 微秒整数（极差用）
us0() {
    printf '%d' $(($1 / 1000))
}
# 中位：参数为已升序数组；取奇数样本的中间那个
median_of() {
    local -a v=("$@")
    local n=${#v[@]}
    printf '%d' "${v[$((n / 2))]}"
}
# 比值 b/a（3 位小数，纯整数；四舍五入到 0.001）
ratio3() {
    local a=$1 b=$2
    if [ "$a" -le 0 ]; then
        printf 'inf'
        return
    fi
    local thousandths=$(((b * 1000 + a / 2) / a))
    printf '%d.%03d' $((thousandths / 1000)) $((thousandths % 1000))
}

echo
echo "# DSL 算子 vs LASX 库算子（进程级交替 A/B，${ROUNDS} 轮丢首轮，进程内 ${ITERS} 次取中位）$TREND"
echo "- loadavg（开始）= \`${LOADAVG}\`；threads = \`${THREADS}\`；ops = \`${OPS}\`"
echo "- A = LASX 库算子（\`lasx_rs::api::*\`）；B = DSL 算子（\`ops_dsl::*\` + \`LasxBackend\`）；**B/A > 1 ⇒ DSL 更慢**"
echo
echo "| op | 规模 | threads | A=LASX 中位(极差) | B=DSL 中位(极差) | B/A | 等价类 | loadavg |"
echo "|---|---|---|---|---|---|---|---|"

IFS=',' read -r -a OP_LIST <<<"$OPS"
IFS=',' read -r -a TH_LIST <<<"$THREADS"

for t in "${TH_LIST[@]}"; do
    t=${t// /}
    [ -z "$t" ] && continue
    for op in "${OP_LIST[@]}"; do
        op=${op// /}
        [ -z "$op" ] && continue
        A_VALS=()
        B_VALS=()
        EQ="?"
        WORK="?"
        FAILED=0
        for ((r = 1; r <= ROUNDS; r++)); do
            if ((r % 2 == 1)); then
                A_OUT=$("$BIN" --op="$op" --arm=lasx --threads="$t" --iters="$ITERS" 2>&1) || FAILED=1
                B_OUT=$("$BIN" --op="$op" --arm=dsl --threads="$t" --iters="$ITERS" 2>&1) || FAILED=1
            else
                B_OUT=$("$BIN" --op="$op" --arm=dsl --threads="$t" --iters="$ITERS" 2>&1) || FAILED=1
                A_OUT=$("$BIN" --op="$op" --arm=lasx --threads="$t" --iters="$ITERS" 2>&1) || FAILED=1
            fi
            A_NS=$(field "$A_OUT" median_ns)
            B_NS=$(field "$B_OUT" median_ns)
            if [ -z "${A_NS:-}" ] || [ -z "${B_NS:-}" ]; then
                FAILED=1
                echo "| $op | - | $t | **失败** | - | - | - | - |" >&2
                break
            fi
            if [ "$r" -eq 1 ]; then
                EQ=$(field "$A_OUT" eq)
                WORK=$(field "$A_OUT" work)
                continue # 丢弃首轮（预热）
            fi
            A_VALS+=("$A_NS")
            B_VALS+=("$B_NS")
        done
        if [ "$FAILED" -ne 0 ] || [ "${#A_VALS[@]}" -eq 0 ]; then
            continue
        fi
        mapfile -t A_SORTED < <(printf '%s\n' "${A_VALS[@]}" | sort -n)
        mapfile -t B_SORTED < <(printf '%s\n' "${B_VALS[@]}" | sort -n)
        A_MED=$(median_of "${A_SORTED[@]}")
        B_MED=$(median_of "${B_SORTED[@]}")
        A_RANGE=$(us0 $((A_SORTED[$((${#A_SORTED[@]} - 1))] - A_SORTED[0])))
        B_RANGE=$(us0 $((B_SORTED[$((${#B_SORTED[@]} - 1))] - B_SORTED[0])))
        A_MS=$(ms3 "$A_MED")
        B_MS=$(ms3 "$B_MED")
        R=$(ratio3 "$A_MED" "$B_MED")
        echo "| $op | $WORK | $t | ${A_MS} ms (${A_RANGE} µs) | ${B_MS} ms (${B_RANGE} µs) | ${R}× | $EQ | ${LOADAVG} |"
    done
done

read -r LOADAVG_END _ < /proc/loadavg || LOADAVG_END="?"
echo
echo "- loadavg（结束）= \`${LOADAVG_END}\`；口径见 \`examples/dsl_vs_lasx_ab.rs\` 文件头（公平性/等价类/缺口）"
