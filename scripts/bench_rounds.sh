#!/bin/bash
# 进程级多轮测量 runner（方差测量法；配套换算见 scripts/benchvar.rs）
#
# 用法：bash scripts/bench_rounds.sh <label> <rounds> <pin|nopin> <g1,g2,...|ALL>
# 环境变量：
#   BIN      被测二进制（默认 /tmp/lasx_bench_frozen；务必是**已冻结**的产物，
#            并建议同时用 BIN_SRC 记下它的来源提交）
#   BIN_SRC  该二进制的来源提交（写进日志头，便于归因）
#   SETTLE   轮间等待秒数（默认 180）：让上一轮 bench 自身的 1 分钟 loadavg 衰减，
#            使本轮的 before 读数尽量反映外部基线
#
# 产物（**故意写到 /tmp**，原始日志不进仓库）：
#   /tmp/bench_<label>_log.txt    逐轮 before/after 的 uptime 原文 + A/B/C 分级 + 并发数
#   /tmp/bench_<label>_r<i>.txt   第 i 轮各组的原始输出（按组顺序拼接）
#
# 分级（见 docs/dev.md §6.4）：
#   A = 轮前 1min<=3 且 5min<=3 且无并发 bench/编译（桌面常驻负载下通常不可达）
#   B = 轮前 1min<=4.5 且无并发 bench/编译 ⇒ 可作正式数
#   C = 其余 ⇒ 只作趋势
# 并发判定：>=20% CPU 的 rustc/cargo 进程数 + 任何 lasx_bench 进程数（按 CPU 判定可排除
# 与本工作区无关、0% CPU 的僵留 cargo）；轮末再测一次，>0 则把该轮降为 C（可重跑）。
set -u
label="${1:?label}"; rounds="${2:?rounds}"; mode="${3:?pin|nopin}"; groups="${4:?groups|ALL}"
bin="${BIN:-/tmp/lasx_bench_frozen}"
log="/tmp/bench_${label}_log.txt"
settle="${SETTLE:-180}"

cmp_le() { # $1 <= $2 ?
  local a="$1" b="$2"
  case "$a" in *.*) ;; *) a="$a.0" ;; esac
  case "$b" in *.*) ;; *) b="$b.0" ;; esac
  [ "$(printf '%s\n%s\n' "$a" "$b" | sort -n | head -1)" = "$a" ]
}
hotc() { # 并发 bench/编译数
  local n=0 pc ip
  while read -r pc _; do
    case "$pc" in *.*) ip="${pc%%.*}" ;; *) ip="$pc" ;; esac
    [ "${ip:-0}" -ge 20 ] 2>/dev/null && n=$((n + 1))
  done < <(ps -eo pcpu,comm --no-headers | grep -E '(lasx_bench|rustc|cargo)[[:space:]]*$')
  n=$((n + $(ps -eo comm | grep -c '^lasx_bench$')))
  echo "$n"
}

{
  echo "# label=$label rounds=$rounds mode=$mode groups=$groups bin=$bin settle=${settle}s"
  echo "# bin sha256: $(sha256sum "$bin" | cut -d' ' -f1)"
  echo "# bin built from: ${BIN_SRC:-$(git rev-parse --short HEAD)} ($(date -r "$bin" '+%F %T'))"
  echo "# 分级：A=1min<=3 且 5min<=3 且 conc=0；B=1min<=4.5 且 conc=0（正式）；C=其余（趋势）"
} > "$log"

for i in $(seq 1 "$rounds"); do
  [ "$i" -gt 1 ] && sleep "$settle"
  before="$(uptime)"
  b_loads="${before##*load average: }"
  l1="${b_loads%%,*}"; mid="${b_loads#*, }"; l5="${mid%%,*}"
  cb="$(hotc)"
  if [ "$cb" -ne 0 ] || ! cmp_le "$l1" 4.5; then grade=C
  elif cmp_le "$l1" 3 && cmp_le "$l5" 3; then grade=A
  else grade=B; fi

  out="/tmp/bench_${label}_r${i}.txt"
  : > "$out"
  fails=0
  if [ "$groups" = "ALL" ]; then
    if [ "$mode" = pin ]; then taskset -c 4 "$bin" >> "$out" 2>&1; else "$bin" >> "$out" 2>&1; fi
    [ "$?" -ne 0 ] && fails=$((fails + 1))
  else
    IFS=',' read -r -a gs <<< "$groups"
    for g in "${gs[@]}"; do
      printf '===== GROUP %s =====\n' "$g" >> "$out"
      if [ "$mode" = pin ]; then
        taskset -c 4 "$bin" "$g" >> "$out" 2>&1
      else
        "$bin" "$g" >> "$out" 2>&1
      fi
      [ "$?" -ne 0 ] && fails=$((fails + 1))
    done
  fi
  after="$(uptime)"
  a_loads="${after##*load average: }"
  ca="$(hotc)"
  note=""
  if [ "$ca" -ne 0 ] && [ "$grade" != C ]; then grade=C; note=" mid_round_conc=$ca"; fi
  printf 'round %s grade=%s conc_before=%s conc_after=%s before_load=%s after_load=%s fails=%s%s\n  before: %s\n  after:  %s\n' \
    "$i" "$grade" "$cb" "$ca" "$b_loads" "$a_loads" "$fails" "$note" "$before" "$after" >> "$log"
done
echo "done: $log"
grep -c '^round' "$log"
