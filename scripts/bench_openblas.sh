#!/bin/bash
# 跨库对照 runner（lasx_rs vs OpenBLAS 0.3.34）——**可复现**：进程级交替 + 丢首轮 + 中位数/极差
# + 逐轮 loadavg + A/B/C 分级。
#
# 口径见 `docs/dev.md` §6.4；**跨库对照的公平性清单**（同一形状/同一确定性数据/同一预热、
# CBLAS 布局映射、先验一致性、线程对齐）写在 `scripts/bench_openblas.c` 头部注释里——那里是
# 权威，本脚本只负责"多轮 + 分级 + 汇总"。
#
# 用法：
#   bash scripts/bench_openblas.sh <label> [rounds] [case-filter]
#     例：bash scripts/bench_openblas.sh ob1 6 192x          # 只跑 case 名含 192x 的
#         bash scripts/bench_openblas.sh ob1                  # 全部（6 轮）
#
# 环境变量：
#   BIN      对照二进制（默认 /tmp/bench_openblas；不存在时本脚本按下面那条 gcc 命令构建）
#   SETTLE   轮间等待秒数（默认 60：让上一轮 bench 自身的 1min loadavg 衰减）
#   PIN_CPU  单线程档绑的核（默认 4；**只有单线程内核该绑核**，见 §6.4）
#
# 产物（**故意写 /tmp**，原始日志不进仓库）：
#   /tmp/bench_openblas_<label>_log.txt      逐轮 before/after loadavg + 分级 + 原始 OB 行
#   /tmp/bench_openblas_<label>_summary.txt  丢首轮后各 case 的中位数/极差/GFLOP/s/比值
#
# 分级（docs/dev.md §6.4）：A = 轮前 1min 且 5min ≤3 且无并发 bench/编译（本机通常不可达）；
#   B = 轮前 1min ≤4.5 且无并发 ⇒ 可作正式数；C = 其余 ⇒ 只作趋势。
set -u
label="${1:?用法: bench_openblas.sh <label> [rounds] [case-filter]}"
rounds="${2:-6}"
filter="${3:-}"
bin="${BIN:-/tmp/bench_openblas}"
settle="${SETTLE:-60}"
pin_cpu="${PIN_CPU:-4}"
log="/tmp/bench_openblas_${label}_log.txt"
sum="/tmp/bench_openblas_${label}_summary.txt"

# ---- case 表：name|dtype|m|k|n|threads|split|pin ----
# `split` 只对 threads>1 有意义（本库的 C ABI 不导出线程池 ⇒ 多线程由调用方切分，见 .c 头部）。
cases=(
  "s64^3-f32|f32|64|64|64|1|rows|pin"
  "s128^3-f32|f32|128|128|128|1|rows|pin"
  "s256^3-f32|f32|256|256|256|1|rows|pin"
  "s384^3-f32|f32|384|384|384|1|rows|pin"
  "s512^3-f32|f32|512|512|512|1|rows|pin"
  "s64^3-f64|f64|64|64|64|1|rows|pin"
  "s128^3-f64|f64|128|128|128|1|rows|pin"
  "s256^3-f64|f64|256|256|256|1|rows|pin"
  "s384^3-f64|f64|384|384|384|1|rows|pin"
  "s512^3-f64|f64|512|512|512|1|rows|pin"
  "m192x768x3072-f32-rows|f32|192|768|3072|12|rows|nopin"
  "m192x768x3072-f32-cols|f32|192|768|3072|12|cols|nopin"
  "w256x64x4096-f32-1t|f32|256|64|4096|1|rows|pin"
  "w256x64x4096-f32-12t-cols|f32|256|64|4096|12|cols|nopin"
)

cmp_le() { # $1 <= $2 ?
  local a="$1" b="$2"
  case "$a" in *.*) ;; *) a="$a.0" ;; esac
  case "$b" in *.*) ;; *) b="$b.0" ;; esac
  [ "$(printf '%s\n%s\n' "$a" "$b" | sort -n | head -1)" = "$a" ]
}
hotc() { # 并发 bench/编译进程数（判定与 bench_rounds.sh 同口径）
  local n=0 pc _ ip
  while read -r pc _; do
    case "$pc" in *.*) ip="${pc%%.*}" ;; *) ip="$pc" ;; esac
    [ "${ip:-0}" -ge 20 ] 2>/dev/null && n=$((n + 1))
  done < <(ps -eo pcpu,comm --no-headers | grep -E '(bench_openblas|lasx_bench|rustc|cargo)[[:space:]]*$')
  n=$((n + $(ps -eo comm | grep -c '^bench_openblas$')))
  echo "$n"
}
median_of() { # 读 stdin 的数字（每行一个），打印上中位数
  local n
  n="$(printf '%s\n' "$@" | grep -c .)"
  [ "$n" -gt 0 ] || return 0
  printf '%s\n' "$@" | sort -n | head -$(((n + 1) / 2)) | tail -1
}

# ---- 构建（缺失时）----
if [ ! -x "$bin" ]; then
  echo "[bench_openblas] 构建 $bin"
  gcc -O2 -march=native -Iinclude scripts/bench_openblas.c target/release/liblasx_rs.a \
      $(pkg-config --cflags --libs openblas) -lm -lpthread -ldl -o "$bin" || exit 1
fi

{
  echo "# label=$label rounds=$rounds filter=${filter:-ALL} bin=$bin settle=${settle}s pin_cpu=$pin_cpu"
  echo "# date: $(date '+%F %T')"
  echo "# openblas: $(pkg-config --modversion openblas 2>/dev/null || echo '?')"
  echo "# liblasx_rs.a sha256: $(sha256sum target/release/liblasx_rs.a | cut -d' ' -f1)"
  echo "# git: $(git rev-parse --short HEAD)  工作区: $(git status --porcelain | wc -l) 处未提交"
  echo "# 分级：A=1min<=3 且 5min<=3 且 conc=0；B=1min<=4.5 且 conc=0（正式）；C=其余（趋势）"
} > "$log"

# ---- 先验一致性（每个 case 一次，失败即中止：不进入计时）----
echo "[bench_openblas] 先验一致性检查（--impl=both）"
for c in "${cases[@]}"; do
  IFS='|' read -r name dtype m k n threads split pin <<<"$c"
  [ -z "$filter" ] || case $name in *"$filter"*) ;; *) continue ;; esac
  out=$(OPENBLAS_NUM_THREADS=$threads "$bin" --impl=both --dtype="$dtype" --m="$m" --k="$k" \
        --n="$n" --threads="$threads" --split="$split" 2>&1 | grep '^OBCHECK')
  printf '%s\n' "$out"
  printf '%s\n' "$out" >> "$log"
  case "$out" in
    *"ok=yes"*) ;;
    *)
      printf '[bench_openblas] 先验一致性**未通过**，中止（不进入计时）：%s\n' "$out" >&2
      exit 1
      ;;
  esac
done

# ---- 多轮：进程级交替（A/B 两个进程，顺序逐轮翻转）----
for i in $(seq 1 "$rounds"); do
  [ "$i" -gt 1 ] && sleep "$settle"
  before="$(uptime)"
  b_loads="${before##*load average: }"
  l1="${b_loads%%,*}"; mid="${b_loads#*, }"; l5="${mid%%,*}"
  cb="$(hotc)"
  if [ "$cb" -ne 0 ] || ! cmp_le "$l1" 4.5; then grade=C
  elif cmp_le "$l1" 3 && cmp_le "$l5" 3; then grade=A
  else grade=B; fi

  echo "## round $i before: $before  conc=$cb  grade=$grade" >> "$log"
  for c in "${cases[@]}"; do
    IFS='|' read -r name dtype m k n threads split pin <<<"$c"
    [ -z "$filter" ] || case $name in *"$filter"*) ;; *) continue ;; esac
    pre=""
    [ "$pin" = pin ] && pre="taskset -c $pin_cpu"
    args=(--dtype="$dtype" --m="$m" --k="$k" --n="$n" --threads="$threads" --split="$split")
    # 每条实现各起一个进程；两个进程的先后顺序逐轮翻转 ⇒ 同窗口成对交替
    if [ $((i % 2)) -eq 1 ]; then order="lasx openblas"; else order="openblas lasx"; fi
    for impl in $order; do
      line=$(OPENBLAS_NUM_THREADS=$threads $pre "$bin" --impl="$impl" "${args[@]}" 2>&1 | grep '^OB ')
      echo "r$i case=$name $line" >> "$log"
    done
  done
  echo "## round $i after:  $(uptime)" >> "$log"
done

# ---- 汇总：丢首轮 + 上中位数 + 极差 + 比值 ----
{
  echo "# label=$label rounds=$rounds（丢弃首轮，用后 $((rounds - 1)) 轮取上中位数）"
  echo "# 逐轮原文与 loadavg 见 ${log}"
  echo "# case impl ms_med(min-max) gflops(med) | openblas/lasx 比值"
  for c in "${cases[@]}"; do
    IFS='|' read -r name dtype m k n threads split pin <<<"$c"
    [ -z "$filter" ] || case $name in *"$filter"*) ;; *) continue ;; esac
    med_lasx=""; med_openblas=""
    for impl in lasx openblas; do
      vals=(); gfs=()
      for i in $(seq 2 "$rounds"); do
        v=$(grep -m1 "^r$i case=$name OB impl=$impl " "$log" | grep -o 'ms_med=[0-9.]*' | cut -d= -f2)
        g=$(grep -m1 "^r$i case=$name OB impl=$impl " "$log" | grep -o 'gflops=[0-9.]*' | cut -d= -f2)
        [ -n "$v" ] && vals+=("$v") && gfs+=("$g")
      done
      if [ "${#vals[@]}" -eq 0 ]; then
        echo "$name $impl 无样本（该轮缺失？）"
        continue
      fi
      med=$(median_of "${vals[@]}")
      gmed=$(median_of "${gfs[@]}")
      mn=$(printf '%s\n' "${vals[@]}" | sort -n | head -1)
      mx=$(printf '%s\n' "${vals[@]}" | sort -n | tail -1)
      echo "$name $impl ${med}(${mn}-${mx}) gflops=${gmed}"
      if [ "$impl" = lasx ]; then med_lasx=$med; else med_openblas=$med; fi
    done
    if [ -n "$med_lasx" ] && [ -n "$med_openblas" ]; then
      ratio=$(echo "scale=3; $med_openblas / $med_lasx" | bc -l 2>/dev/null)
      echo "$name RATIO openblas_med/lasx_med = ${ratio}（>1 = 本库更快；<1 = OpenBLAS 更快）"
    fi
  done
} > "$sum"

echo "[bench_openblas] 汇总（丢首轮、上中位数、极差）："
cat "$sum"
echo "[bench_openblas] 原始逐轮日志：$log"
