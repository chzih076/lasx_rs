#!/usr/bin/env bash
# lasx_rs C ABI **命名规范**门禁（规则的唯一权威：docs/c-abi.md §9.1）。
#
# 为什么是独立脚本、而不是并进 scripts/doccheck.sh：
#   ① **职责不同**——doccheck 管"文档彼此一致"（§ 引用、表格列数、文档里的计数），纯文本、
#      不需要工具链；本脚本管"**ABI 产物与 FFI 源码的符号名**是否符合命名规范"，数据源是
#      `nm` 读出的产物符号表，失败含义与修法都不一样；
#   ② **CI 里能单独列一步**：红了立刻知道是"名字/规范"问题，而不是文档漂移；
#   ③ 反过来也成立：没有工具链（或只想查文档）的环境仍能跑 doccheck。
#
# 用法：
#   bash scripts/namecheck.sh
#   NAMECHECK_LIB=target/release/liblasx_rs.a bash scripts/namecheck.sh   # 换数据源（.a 也行）
#   NAMECHECK_SYMS=/tmp/syms.txt bash scripts/namecheck.sh
#       ——**仅供测试/离线校验**（每行一个符号名）：CI 与日常都用产物本身，别把它当常规入口。
#
# 校验四条（全部出自 docs/c-abi.md §9.1「命名规范」）：
#   1 **形态**  `^lasx_[a-z0-9]+(_[a-z0-9]+)*$`：无大写、无连续下划线、无首尾下划线；
#   2 **后缀**  尾部 token 不得是"版本/口味"词（黑名单见 SUFFIX_BLACKLIST）；
#   3 **孪生**  每个 `_checked` 必须有同名裸孪生（反之不要求）；
#   4 **库级**  库级白名单恰好 5 个且在产物里；白名单之外不得出现库级词根（LIB_TOKENS）。
#
# 退出码：0 = 通过；1 = 有违规；2 = 用法/环境问题（产物不存在、符号表为空等）。
set -eu

# ---- 规范常量（改动这里必须同步 docs/c-abi.md §9.1 与 docs/c-abi.md §10 的守卫表）----
# 规则 2：只禁"版本/口味"后缀（语义/类型限定词如 batch/i8/q4/f16/per_row/scaled 是合法的）
SUFFIX_BLACKLIST="v1 v2 v3 v4 v5 v6 v7 v8 v9 ex fast slow new old alt lite pro plus tmp test bak experimental legacy"
# 规则 4：库级符号白名单（**恰好 5 个**；新增库级符号要同时改：这里、docs/c-abi.md §9.1、§3.6）
LIB_WHITELIST="lasx_alloc lasx_free lasx_has_lasx lasx_status_message lasx_version"
# 规则 4：被保留给库级的词根——白名单之外的符号不许用（`lasx_dot_alloc` 这类一律报错）
LIB_TOKENS="alloc free version has message init fini config"

# ---- 取符号表 ----
if [ -n "${NAMECHECK_SYMS:-}" ]; then
  [ -f "$NAMECHECK_SYMS" ] || { printf 'namecheck: 用法错误——NAMECHECK_SYMS 指向的文件不存在：%s\n' "$NAMECHECK_SYMS" >&2; exit 2; }
  SYMS=$(sort -u "$NAMECHECK_SYMS")
  SOURCE=$NAMECHECK_SYMS
else
  LIB=${NAMECHECK_LIB:-}
  if [ -z "$LIB" ]; then
    for c in target/release/liblasx_rs.so target/release/liblasx_rs.a; do
      if [ -f "$c" ]; then LIB=$c; break; fi
    done
  fi
  if [ -z "$LIB" ] || [ ! -f "$LIB" ]; then
    printf 'namecheck: 用法错误——找不到产物（默认找 target/release/liblasx_rs.{so,a}）。\n' >&2
    printf '           先跑 `cargo build --release`，或用 NAMECHECK_LIB=<产物> 指定。\n' >&2
    exit 2
  fi
  case $LIB in
    *.so) SYMS=$(nm -D --defined-only "$LIB" | grep ' T lasx_' | cut -d' ' -f3 | sort -u) ;;
    *)    SYMS=$(nm    --defined-only "$LIB" | grep ' T lasx_' | cut -d' ' -f3 | sort -u) ;;
  esac
  SOURCE=$LIB
fi

if [ -z "$SYMS" ]; then
  printf 'namecheck: 用法错误——%s 里没读到任何 `lasx_*` 符号（产物没构建好？）\n' "$SOURCE" >&2
  exit 2
fi

# ---- 四条校验 ----
RE='^lasx_[a-z0-9]+(_[a-z0-9]+)*$'
fails=0
bad() {
  printf 'namecheck: 违规[%s] %s\n' "$1" "$2" >&2
  fails=$((fails + 1))
}

while IFS= read -r s; do
  [ -n "$s" ] || continue

  # 1 形态
  if [[ ! $s =~ $RE ]]; then
    bad "形态" "$s 不匹配 $RE（大写/连续下划线/首尾下划线都算）"
  fi

  # 2 后缀（先剥掉合法的 `_checked`，再看末段）
  base=${s%_checked}
  last=${base##*_}
  case " $SUFFIX_BLACKLIST " in
    *" $last "*) bad "后缀" "$s 用了版本/口味后缀 _$last（新符号只允许尾部 _checked）" ;;
  esac

  # 3 孪生：`_checked` ⇒ 必须有同名裸符号（注意：提示串里别用反引号——双引号内会被当命令替换）
  if [ "$s" != "$base" ]; then
    printf '%s\n' "$SYMS" | grep -qx "$base" ||
      bad "孪生" "$s 没有同名裸孪生 $base（_checked 必须成对）"
  fi

  # 4 库级：白名单之外不得出现库级词根
  is_whitelisted=0
  for w in $LIB_WHITELIST; do
    if [ "$s" = "$w" ]; then is_whitelisted=1; fi
  done
  if [ "$is_whitelisted" = 0 ]; then
    for t in $LIB_TOKENS; do
      case $base in
        *_"$t" | *_"${t}"_*) bad "库级" "$s 含库级词根 _$t（库级符号必须进白名单）" ;;
      esac
    done
  fi
done < <(printf '%s\n' "$SYMS")

# 4（续）白名单 5 个必须都在产物里
for w in $LIB_WHITELIST; do
  printf '%s\n' "$SYMS" | grep -qx "$w" ||
    bad "库级" "白名单符号 $w 不在产物里（白名单固定 5 个）"
done

# ---- 汇总 ----
n=$(printf '%s\n' "$SYMS" | grep -c .)
n_checked=$(printf '%s\n' "$SYMS" | grep -c '_checked$' || true)
n_bare=$((n - n_checked))
# shellcheck disable=SC2086  # 这里就是要按空格分词数白名单条目
set -- $LIB_WHITELIST
n_wl=$#

if [ "$fails" -gt 0 ]; then
  printf 'namecheck: FAIL（%d 个符号，%d 处违规；规则见 docs/c-abi.md §9.1）\n' "$n" "$fails" >&2
  exit 1
fi

printf 'namecheck: %d 个符号 = %d 裸 + %d `_checked`（其中库级白名单 %d 个）；数据源 %s\n' \
  "$n" "$n_bare" "$n_checked" "$n_wl" "$SOURCE"
printf 'namecheck: OK（%d 个符号，0 例外）\n' "$n"
