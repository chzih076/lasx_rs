#!/bin/bash
# lasx_rs 性能剖析 runner（perf stat 计数 + 用户态调用图）。
#
# 目标（`docs/dev.md` §7.10 ⑧）：把"C 侧 245.8 G 与结构上界 346–355 G 之间的差距"查清——
# 到底是**带宽（DRAM/L3）**、**同步/调度抖动**、**TLB**，还是**窗口**？
#
# 环境约束（本机实测，**perf_event_paranoid = 2**）：
#   * 只能**用户态**采样/计数（`:u`）：系统级 `-a` 与内核态不可用；
#   * `perf record` 的调用图**看不到内核帧**——"看不见"只能如实写，不能当归因；
#   * **一次挂太多事件会 "Some events weren't counted"（NMI watchdog）** ⇒ 本脚本**分批**，
#     并且**先跑事件自检**（每个分批单独 `perf stat … -- true`）把"哪些真能计数"打出来。
#
# 用法：
#   bash scripts/profile_lasx.sh [label] [reps]
#     例：bash scripts/profile_lasx.sh p1 3
#
# 产物（**故意写 /tmp**，原始 perf 输出不进仓库）：
#   /tmp/profile_lasx_<label>_log.txt      事件自检 + 每次运行的 perf 原文与 loadavg
#   /tmp/profile_lasx_<label>_summary.txt  各档各计数的**中位数（极差）** + 派生量
#   /tmp/profile_lasx_<label>_perf.data    `perf record` 的样本（plan-tight 档）
#   /tmp/profile_lasx_<label>_top.txt      `perf report` 的用户态热点（前若干函数）
#
# 口径（`docs/dev.md` §6.4）：每档每个分批 **≥3 次重复**；报**中位数 + 极差**；
#   逐次记录轮前 `loadavg`；并同时报装置自己的 `iters`/`ms_max`（窗口质量）。
#   容器里没有 `perf` 时脚本会明确说"本机不可用"并退出（**不编数据**）。
set -u
label="${1:-p1}"
reps="${2:-3}"
root="$(cd "$(dirname "$0")/.." && pwd)"
bin_prof="/tmp/bench_openblas_prof"
log="/tmp/profile_lasx_${label}_log.txt"
sum="/tmp/profile_lasx_${label}_summary.txt"
data="/tmp/profile_lasx_${label}_perf.data"
top="/tmp/profile_lasx_${label}_top.txt"

command -v perf >/dev/null 2>&1 || {
  printf 'profile_lasx: 本机没有 perf —— 该分析不可用（不编数据）\n' >&2
  exit 2
}
paranoid="$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo '?')"

# ---- 分批：都实测过"能计数"（自检会再打一遍）----
batches=(
  "ipc|cycles:u,instructions:u"
  "llc|cache-misses:u,llc-loads:u,llc-load-misses:u"
  "tlb|dTLB-loads:u,dTLB-load-misses:u"
  "sched|context-switches,cpu-migrations,page-faults"
  "br|branches:u,branch-misses:u"
)

all_events_of() { printf '%s\n' "${batches[@]}" | grep "^$1|" | cut -d'|' -f2; }

{
  echo "# profile_lasx label=$label reps=$reps"
  echo "# date: $(date '+%F %T')  perf: $(perf --version 2>/dev/null)"
  echo "# perf_event_paranoid=$paranoid（2 ⇒ 只能用户态；-a/内核态不可用）"
  echo "# 口径：每档每个分批 ${reps} 次重复，报中位数+极差；逐次记轮前 loadavg"
} > "$log"

# ------------------------------------------------------------------
# ① 事件自检：每个分批单独 `perf stat … -- true`，打"可用/不可用"
# ------------------------------------------------------------------
echo "== ① 事件自检（perf stat 分批 -- true） =="
for b in "${batches[@]}"; do
  name="${b%%|*}"
  evs="${b#*|}"
  out="$(perf stat -e "$evs" -- true 2>&1)"
  if printf '%s' "$out" | grep -q "weren't counted\|not supported\|No such file\|invalid"; then
    verdict="不可用"
  else
    verdict="可用"
  fi
  printf '  [%s] %-8s %s\n' "$verdict" "$name" "$evs" | tee -a "$log" >/dev/null
  printf '  [%s] %-8s %s\n' "$verdict" "$name" "$evs"
  printf -- '--- 自检原文（%s）---\n%s\n' "$name" "$out" >> "$log"
done
# 负对照：一次挂 8 个事件（说明"为什么必须分批"）
neg="$(perf stat -e cycles:u,instructions:u,cache-misses:u,llc-loads:u,llc-load-misses:u,dTLB-loads:u,dTLB-load-misses:u,branches:u -- true 2>&1)"
printf '  [负对照] 8 事件一次挂： %s\n' "$(printf '%s' "$neg" | grep -o "Some events weren't counted" || echo '（本次没触发）')"
printf -- '--- 负对照原文 ---\n%s\n' "$neg" >> "$log"

# ------------------------------------------------------------------
# ② 待剖析的"档"（同一形状 192×768×3072 f32 @12）
# ------------------------------------------------------------------
# 每项：名字|类型|参数。`bench` 用 /tmp/bench_openblas_prof；`rusttest` 用 Rust 侧诊断测试
# （池 vs 计划同进程交替，各 7 次调用 ⇒ 计数按 7 折算）。
shape="--dtype=f32 --m=192 --k=768 --n=3072 --threads=12"
arms=(
  "plan-tight|bench|--impl=lasx $shape --split=plan-tight"
  "pool(rust)|rusttest|-"
  "single-thread|bench|--impl=lasx --dtype=f32 --m=192 --k=768 --n=3072 --threads=1 --split=plan"
  "openblas|bench|--impl=openblas $shape"
  "rows(病理)|bench|--impl=lasx $shape --split=rows"
)

# ---- 编译带 -g/-fno-omit-frame-pointer 的剖析版装置（进 /tmp，不动仓库产物）----
echo "== ② 编译剖析版装置（-O2 -g -fno-omit-frame-pointer -fopenmp） =="
( cd "$root" && gcc -O2 -g -fno-omit-frame-pointer -march=native -fopenmp -Iinclude \
    scripts/bench_openblas.c target/release/liblasx_rs.a \
    $(pkg-config --cflags --libs openblas) -lm -lpthread -ldl -o "$bin_prof" ) || exit 1

# ---- Rust 侧诊断测试的二进制（池 vs 计划）----
# 注意：这个 crate 有 build script，测试二进制落在 `target/release/build/<crate>/<hash>/out/`，
# **不在** `target/release/deps/` ⇒ 从 `cargo test --no-run` 的输出里取路径（唯一可靠做法）。
rust_test_bin="$(cd "$root" && cargo test --release -p lasx_rs --lib --no-run 2>&1 |
  grep -oE 'Executable unittests src/lib.rs \([^)]+\)' | head -1 | cut -d'(' -f2 | tr -d ')')"
if [ -z "$rust_test_bin" ] || [ ! -x "$rust_test_bin" ]; then
  printf 'profile_lasx: 取不到 Rust 测试二进制（pool 档将记为不可用）\n' >&2
  rust_test_bin=""
fi

arm_cmd() { # $1 = 档名 → 打印命令（不含 perf）
  case "$1" in
    bench)
      printf '%s %s' "$bin_prof" "$2"
      ;;
    rusttest)
      # `--exact` 要**全路径名**（`parallel::diag_pool_vs_plan`）；写成短名会匹配 0 个测试，
      # 于是这一档只剩测试框架的计数（本脚本踩过一次，计数小到 5.7e5 才被发现）。
      printf '%s --ignored --exact parallel::diag_pool_vs_plan --nocapture' "$rust_test_bin"
      ;;
  esac
}

# ------------------------------------------------------------------
# ③ 逐档 × 逐分批 × reps 次：跑 perf stat，收计数 + 装置自报的窗口指标
# ------------------------------------------------------------------
echo "== ③ 逐档计数（每格 ${reps} 次，报中位数+极差） =="
d() { # $1 表达式（bc -l）。注意：**bc 不认 `1e9` 这类科学计数**（会让整行变空、静默丢数），
      # 所有常量都写全数字（本脚本踩过一次：派生量整列变成 "-"）。
  printf '%s\n' "$1" | bc -l 2>/dev/null
}
median_of() { # 从 stdin 的整数列表取中位数（偶数取上中位）
  local n
  n="$(printf '%s\n' "$@" | grep -c .)"
  [ "$n" -gt 0 ] || return 0
  printf '%s\n' "$@" | sort -n | head -$(((n + 1) / 2)) | tail -1
}
range_of() { # 从 stdin 的整数列表取 最小–最大
  local lo hi
  lo="$(printf '%s\n' "$@" | sort -n | head -1)"
  hi="$(printf '%s\n' "$@" | sort -n | tail -1)"
  printf '%s-%s' "$lo" "$hi"
}
extract() { # $1 = perf 原文, $2 = 事件名（带或不带 :u）→ 打印计数（去逗号；取不到打 n/a）
  local v
  # 注意：`perf stat` 输出里事件名**后面有尾随空格** ⇒ 不能用 `$` 锚定（早期版本因此全取成 n/a）
  v="$(printf '%s' "$1" |
    grep -oE "[0-9][0-9,]*(\.[0-9]+)? +$2(:u)?([[:space:]]|$)" |
    head -1 | cut -d' ' -f1 | tr -d ',')"
  printf '%s' "${v:-n/a}"
}

: > "$sum"
# 每个分批的"同源比率"规格：分子/分母**必须来自同一次运行**（同一个 `perf stat` 的同一批事件）。
# 格式：`分子下标 分母下标 比率名 上界`。**为什么必须同源**：上一版把"A 分批的中位数"与
# "B 分批的中位数"相除 ⇒ 出现 95%/435% 这种不可能的缺失率、DRAM 估计也对不上（Lead 复核抓到）。
# 比率型结果一律**逐轮算、再看中位数**，并且带机器自洽闸门（不合格只打 `!!`，不出数）。
ratio_spec() {
  case "$1" in
    ipc) printf '1 0 IPC_每条指令周期 20' ;; # IPC = instructions/cycles（上限 20 是防呆，不是百分比）
    llc) printf '2 1 LLC缺失率pct 110' ;;
    tlb) printf '1 0 TLB缺失率pct 110' ;;
    br) printf '1 0 分支缺失率pct 110' ;;
    *) printf '' ;;
  esac
}
for a in "${arms[@]}"; do
  aname="${a%%|*}"; rest="${a#*|}"; atype="${rest%%|*}"; aargs="${rest#*|}"
  for b in "${batches[@]}"; do
    bname="${b%%|*}"; evs="${b#*|}"
    vals_ms=(); vals_iters=(); vals_max=(); vals_calls=()
    vals_ratio=(); vals_bw=()
    # 注意：`declare -A` 对**已存在**的关联数组是"重新声明"而**不清空** ⇒ 必须显式 unset，
    # 否则上一档/上一分批的计数会留在里面，把中位数与极差污染掉（本脚本踩过一次）。
    unset per_event
    declare -A per_event
    for i in $(seq 1 "$reps"); do
      l1="$(cut -d' ' -f1 /proc/loadavg)"
      if [ "$atype" = bench ]; then
        cmd="$(arm_cmd bench "$aargs")"
      else
        cmd="$(arm_cmd rusttest)"
      fi
      out="$(perf stat -e "$evs" -- $cmd 2>&1)"
      printf -- '### arm=%s batch=%s rep=%d loadavg1=%s\n%s\n' "$aname" "$bname" "$i" "$l1" "$out" >> "$log"
      # 窗口指标：装置自己的 OB 行；Rust 诊断测试档读它自己打印的"池 … = X ms = Y GFLOP/s"
      if [ "$atype" = bench ]; then
        ob="$(printf '%s' "$out" | grep '^OB ' | head -1)"
        ms="$(printf '%s' "$ob" | grep -oE 'ms_med=[0-9.]+' | cut -d= -f2)"
        it="$(printf '%s' "$ob" | grep -oE 'iters=[0-9]+' | cut -d= -f2)"
        mx="$(printf '%s' "$ob" | grep -oE 'ms_max=[0-9.]+' | cut -d= -f2)"
      else
        ob="$(printf '%s' "$out" | grep -E '^池' | head -1)"
        ms="$(printf '%s' "$ob" | grep -oE '= [0-9.]+ ms' | grep -oE '[0-9.]+' | head -1)"
        it="7" # 诊断测试固定 7 次交替调用
        mx="$ms"
      fi
      vals_ms+=("${ms:-}"); vals_iters+=("${it:-}"); vals_max+=("${mx:-}")
      # 本次运行"每次调用折算"用的调用数：bench = iters + 2（1 热身 + 1 自适应探测 + iters 计时）
      if [ "$atype" = bench ] && [ -n "${it:-}" ]; then
        calls_i=$((it + 2))
      else
        calls_i=7
      fi
      vals_calls+=("$calls_i")
      # 计数：**逐事件、同一次运行**地取（ev_vals 与 ev_list 下标对齐）
      IFS=',' read -r -a ev_list <<<"$evs"
      ev_vals=()
      for e in "${ev_list[@]}"; do
        v="$(extract "$out" "$e")"
        ev_vals+=("$v")
        per_event["$e"]="${per_event[$e]:-} $v"
      done
      # ---- 同源比率（**逐轮**算；分子分母来自这一次运行）----
      case "$bname" in
        ipc)
          if [ "${ev_vals[0]:-n/a}" != n/a ] && [ "${ev_vals[1]:-n/a}" != n/a ] &&
            [ "${ev_vals[0]:-0}" != 0 ]; then
            vals_ratio+=("$(d "${ev_vals[1]}/${ev_vals[0]}")")
          fi
          ;;
        llc)
          if [ "${ev_vals[1]:-n/a}" != n/a ] && [ "${ev_vals[2]:-n/a}" != n/a ] &&
            [ "${ev_vals[1]:-0}" != 0 ]; then
            vals_ratio+=("$(d "100*${ev_vals[2]}/${ev_vals[1]}")")
            # 每调用的 **全线程合计** DRAM 估计 =（miss × 64 B ÷ 本次运行调用数）÷ 每次调用墙钟
            if [ -n "${ms:-}" ] && [ "$ms" != 0 ]; then
              vals_bw+=("$(d "${ev_vals[2]}*64*1000/($calls_i*$ms*1000000000)")")
            fi
          fi
          ;;
        tlb)
          if [ "${ev_vals[0]:-n/a}" != n/a ] && [ "${ev_vals[1]:-n/a}" != n/a ] &&
            [ "${ev_vals[0]:-0}" != 0 ]; then
            vals_ratio+=("$(d "100*${ev_vals[1]}/${ev_vals[0]}")")
          fi
          ;;
        br)
          if [ "${ev_vals[0]:-n/a}" != n/a ] && [ "${ev_vals[1]:-n/a}" != n/a ] &&
            [ "${ev_vals[0]:-0}" != 0 ]; then
            vals_ratio+=("$(d "100*${ev_vals[1]}/${ev_vals[0]}")")
          fi
          ;;
      esac
    done
    # 折算：bench 档的调用数 = iters + 2（1 热身 + 1 自适应探测 + iters 计时）；rusttest = 7
    calls=""
    calls="$(median_of "${vals_calls[@]}")"
    {
      printf 'arm=%s batch=%s shape=192x768x3072-f32-threads12 calls_per_run=%s\n' \
        "$aname" "$bname" "${calls:-n/a}"
      printf '  ms_med 中位数=%s 极差=%s | iters 中位数=%s | ms_max 中位数=%s\n' \
        "$(median_of "${vals_ms[@]}")" "$(range_of "${vals_ms[@]}")" \
        "$(median_of "${vals_iters[@]}")" "$(median_of "${vals_max[@]}")"
      # **原始分子/分母**都打出来（读者可手算复核）
      for e in "${ev_list[@]}"; do
        read -r -a vv <<<"${per_event[$e]}"
        printf '  %-22s 中位数=%-14s 极差=%s\n' "$e" "$(median_of "${vv[@]}")" "$(range_of "${vv[@]}")"
      done
      # 同源比率 + **自洽闸门**（不合格只打 `!!`，不出数）
      spec="$(ratio_spec "$bname")"
      if [ -n "$spec" ] && [ "${#vals_ratio[@]}" -gt 0 ]; then
        set -- $spec
        rname="$3"; rbound="$4"
        rmed="$(median_of "${vals_ratio[@]}")"
        ok=1
        for rv in "${vals_ratio[@]}"; do
          if [ "$bname" = ipc ]; then
            [ "$(d "$rv < 0.05")" = 1 ] && ok=0
            [ "$(d "$rv > 20")" = 1 ] && ok=0
          else
            [ "$(d "$rv > $rbound")" = 1 ] && ok=0
            [ "$(d "$rv < 0")" = 1 ] && ok=0
          fi
        done
        if [ "$ok" = 1 ]; then
          printf '  %s 逐轮中位数=%s 极差=%s（**同源**：分子/分母同一次运行；闸门 OK）\n' \
            "$rname" "$rmed" "$(range_of "${vals_ratio[@]}")"
        else
          printf '  !! 自洽失败：%s 逐轮=%s（闸门 %s）——**不出数**\n' \
            "$rname" "$(printf '%s ' "${vals_ratio[@]}")" "$rbound"
        fi
      fi
      if [ "${#vals_bw[@]}" -gt 0 ]; then
        printf '  DRAM估计GB/s 逐轮中位数=%s 极差=%s（**每调用、全线程合计**流量 ÷ 每调用墙钟）\n' \
          "$(median_of "${vals_bw[@]}")" "$(range_of "${vals_bw[@]}")"
      fi
      printf '\n'
    } >> "$sum"
    printf '  %-14s %-6s ms_med=%s iters=%s ms_max=%s\n' "$aname" "$bname" \
      "$(median_of "${vals_ms[@]}")" "$(median_of "${vals_iters[@]}")" "$(median_of "${vals_max[@]}")"
  done
done

# ------------------------------------------------------------------
# ④ 派生量（**全部读 ③ 里"逐轮同源"算好的比率**，不再跨分批相除）
# ------------------------------------------------------------------
ratio_of() { # $1 arm, $2 batch, $3 比率名 → 打该比率的"逐轮中位数"（取不到打空）
  grep -A 12 "^arm=$1 batch=$2 " "$sum" |
    grep -E "^  $3 逐轮中位数=" | head -1 | grep -oE '逐轮中位数=[0-9.]+' | cut -d= -f2
}
bw_of() { # $1 arm → DRAM 逐轮中位数
  grep -A 12 "^arm=$1 batch=llc " "$sum" |
    grep -E '^  DRAM估计GB/s 逐轮中位数=' | head -1 | grep -oE '=[0-9.]+' | head -1 | cut -d= -f2
}
cpc_of() { # $1 arm, $2 batch, $3 事件 → 该事件的**每次调用**中位数 = 中位数 ÷ calls_per_run
  local med calls
  med="$(grep -A 12 "^arm=$1 batch=$2 " "$sum" | grep -E "^  $3 +中位数=" | head -1 |
    grep -oE '中位数=[0-9]+' | cut -d= -f2)"
  calls="$(grep "^arm=$1 batch=$2 " "$sum" | head -1 | grep -oE 'calls_per_run=[0-9]+' | cut -d= -f2)"
  [ -n "$med" ] && [ -n "$calls" ] && [ "$calls" != 0 ] && d "$med/$calls"
}

{
  echo
  echo "== ④ 派生量总表 =="
  # 注意：这一段提示**全部用单引号**——双引号里的反引号会被 bash 当命令替换执行
  # （本脚本踩过两次：`--call-graph fp` 与 `perf stat` 都被当命令跑，后者没带参数直接挂住）。
  echo '# **口径（每列逐个写明）**：所有计数都是**整个进程之和**（12 个 worker + 主线程；'
  echo '#   perf stat 默认继承所有线程）；'
  echo '#   * X/调用 = 该次运行的计数 ÷ 该次运行的调用数（bench = iters+2：1 热身 + 1 自适应'
  echo '#     探测 + iters 计时；rust 测试档 = 7）——即**全线程合计**的每调用量；'
  echo '#   * IPC / 缺失率 / DRAM 都是**逐轮同源**算完再取中位数（不是两个中位数相除）；'
  echo '#   * DRAM 估计 =（llc-load-misses × 64 B ÷ 调用数）÷ 每次调用墙钟 = **每调用、全线程合计**；'
  echo '#   * ctx/迁移/页缺失 = 整轮计数 ÷ 调用数。'
  printf '%-14s %8s %6s %8s %10s %11s %12s %10s %10s %9s\n' \
    档 ms/调用 iters/ms_max IPC LLC缺失率 DRAM估GB/s cycles/调用 instr/调用 llc-loads/调用 dTLB缺失/调用
  for a in plan-tight "pool(rust)" single-thread openblas "rows(病理)"; do
    ms="$(grep -A 1 "^arm=$a batch=ipc " "$sum" | grep -oE 'ms_med 中位数=[0-9.]+' | head -1 | cut -d= -f2)"
    it="$(grep -A 1 "^arm=$a batch=ipc " "$sum" | grep -oE 'iters 中位数=[0-9]+' | head -1 | cut -d= -f2)"
    mx="$(grep -A 1 "^arm=$a batch=ipc " "$sum" | grep -oE 'ms_max 中位数=[0-9.]+' | head -1 | cut -d= -f2)"
    ipc="$(ratio_of "$a" ipc IPC_每条指令周期)"
    mr="$(ratio_of "$a" llc LLC缺失率pct)"
    bw="$(bw_of "$a")"
    cy="$(cpc_of "$a" ipc cycles:u)"
    ins="$(cpc_of "$a" ipc instructions:u)"
    ll="$(cpc_of "$a" llc llc-loads:u)"
    dm="$(cpc_of "$a" tlb dTLB-load-misses:u)"
    printf '%-14s %8s %6s %8s %10s %11s %12s %10s %10s %9s\n' "$a" "${ms:--}" "${it:-?}/${mx:-?}" \
      "${ipc:--}" "${mr:--}" "${bw:--}" "${cy:--}" "${ins:--}" "${ll:--}" "${dm:--}"
  done
  echo '# 表中"-"= 该量**本机没测到/闸门不过**（不编数）；比率列若出现"!! 自洽失败"，'
  echo '#   请直接看 ③ 里那一段的逐轮原值。'
  echo
  echo "# 读法（每个量支持 / 不支持什么）："
  echo "#  IPC = instructions/cycles（逐轮同源）：明显低于 1.5 说明**每条指令停顿多**；"
  echo "#  LLC 缺失率 = llc-load-misses/llc-loads（逐轮同源，闸门 ≤110%）：高说明 L3 装不住；"
  echo "#  DRAM 估GB/s = llc-load-misses×64B÷时间：**只是估计**（缺 uncore/内存控制器事件，"
  echo "#      也分不清预取/写回），**别当实测天花板**；文档里 ~100 GB/s 同样是估计值；"
  echo "#  cycles/调用 与 instr/调用 是**全线程合计**（12 线程之和，不是单线程口径）；"
  echo "#  ctx 切换/迁移/页缺失：**调度与同步抖动**的量级（自旋握手应当接近 0）；"
  echo "#  dTLB 缺失：**页表抖动**（大页是否可能有用）。"
} >> "$sum"

# ------------------------------------------------------------------
# ⑤ 用户态调用图（只看 plan-tight 一档；内核帧不可见是已知限制）
# ------------------------------------------------------------------
echo "== ⑤ perf record 用户态调用图（plan-tight） =="
for cg in fp dwarf; do
  rec="/tmp/profile_lasx_${label}_perf_${cg}.data"
  if ( cd "$root" && perf record -o "$rec" -e cycles:u -g --call-graph "$cg" -- \
      "$bin_prof" --impl=lasx $shape --split=plan-tight >/dev/null 2>&1 ) && [ -s "$rec" ]; then
    n="$(perf report -i "$rec" --stdio --header-only 2>/dev/null | grep -oE 'Samples: [0-9K]+' | head -1)"
    printf '  --call-graph %-5s 可采集（%s）→ %s\n' "$cg" "${n:-样本数未知}" "$rec"
    chosen="$rec"
    chosen_cg="$cg"
  else
    printf '  --call-graph %-5s 不可用（paranoid=%s）——如实记为不可用\n' "$cg" "$paranoid"
  fi
done
if [ -n "${chosen:-}" ]; then
  cp -f "$chosen" "$data" 2>/dev/null || true
  # 自耗时（--no-children）前若干行 + 关键符号的占比（打包 / 内核 / 自旋 / 握手）
  perf report -i "$chosen" --stdio --no-children --percent-limit 0.5 2>/dev/null |
    grep -E '^ +[0-9]+\.[0-9]+%' | head -25 > "$top"
  {
    echo "== ⑤ perf record 用户态热点（plan-tight；--call-graph ${chosen_cg:-?}） =="
    echo "# 自耗时前几名（--no-children）："
    cat "$top"
    echo "# 关键符号占比（按行尾匹配；没出现就是没进 top）："
    for sym in worker_fn lasx_matmul_run matmul_f32_packed_rows pack_b packed_rows \
      row_tail sched_yield pthread_barrier_wait lasx_matmul_prepare; do
      line="$(grep -E "$sym[[:space:]]*$" "$top" | head -1)"
      [ -n "$line" ] && printf '  %-32s %s\n' "$sym" "$(printf '%s' "$line" | grep -oE '^ +[0-9.]+%')"
    done
    echo '# **已知限制**：perf_event_paranoid=2 ⇒ 只能用户态采样，**内核帧不可见**；'
    # 注意：这几行提示里**不能用反引号**——双引号内的反引号会被 bash 当命令替换执行
    # （本脚本踩过一次：`--call-graph fp` 被当成命令跑，报"未找到命令"）。
    echo '#   而且本库内核是手写 LASX 汇编、不维护帧指针 ⇒ --call-graph fp 在汇编下方的'
    echo '#   展开不可靠（会看到一堆 0x… 地址，本脚本不把它们当归因）。'
    echo '#   所以这里只报"自耗时前几名"，**不做函数级精确归因**。'
  } >> "$sum"
else
  printf '  perf record 没拿到样本（本机限制：paranoid=%s）——如实记为"不可用"\n' "$paranoid" | tee -a "$sum"
fi

echo
echo "汇总 → $sum"
echo "原始 → $log"
