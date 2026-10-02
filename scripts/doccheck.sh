#!/usr/bin/env bash
# 文档一致性校验（纯文本，无需 Rust 工具链；CI 的 fmt job 里跑同一条）。
#
# 校验三件事（都是本项目**实际踩过**的文档缺陷）：
#   ① 每个 `§N[.M]` 引用都能在**目标文档**里找到同名编号的标题；
#   ② 表格每行的列数与表头一致（Markdown 表格错列会静默渲染坏）；
#   ③ 文档里的"导出符号数"与源码里 `extern "C" fn lasx_*` 的实际数量一致。
#
# 引用解析规则（与仓库既有写法一致）：
#   - 同一句里点名了文档（`docs/dev.md §X`）⇒ 必须命中**那个**文档；
#   - `本文 §X` ⇒ 当前文档；
#   - 裸引用 ⇒ 任一文档里有这个编号即可（README/AGENTS 的既有写法）。
#
# 用法：bash scripts/doccheck.sh [仓库根目录]；有发现时以非零退出。
set -u
root="${1:-.}"
cd "$root" || exit 1

T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
findings=0
note() { echo "$1"; findings=$((findings + 1)); }

# ---- 收集各文档的编号集合 ----
sections() { grep -oE '^#{1,4} [0-9]+(\.[0-9]+)*' "$1" 2>/dev/null | sed 's/^#* //' | sort -u; }
for pair in "docs/dev.md dev" "docs/ops.md ops" "README.md root" "README.en.md rooten" "AGENTS.md agents"; do
  f="${pair%% *}"; k="${pair##* }"
  [ -f "$f" ] && sections "$f" > "$T/$k"
done
all_sections() { cat "$T"/* 2>/dev/null | sort -u; }

# ---- ① 引用 ----
check_refs() {
  local f="$1" line ref num prefix target
  while IFS= read -r line; do
    # `manual §9.6` 这类是**外部手册**的编号，不是本文档体系（docs/dev.md §16 有说明）
    case "$line" in *'manual §'*) continue ;; esac
    for ref in $(printf '%s' "$line" | grep -oE '§[0-9]+(\.[0-9]+)*'); do
      num="${ref#§}"
      prefix="${line%%"$ref"*}"
      target=""
      # 取**前缀里最后一次**出现的文档名（"就近"语义：一句里可以先提 dev 再提 ops）
      last_doc=$(printf '%s' "$prefix" |
        grep -oE 'docs/dev\.md|docs/ops\.md|README\.en\.md|README\.md|AGENTS\.md' | tail -1)
      case "$last_doc" in
        docs/dev.md) target=dev ;;
        docs/ops.md) target=ops ;;
        README.en.md) target=rooten ;;
        README.md) target=root ;;
        AGENTS.md) target=agents ;;
      esac
      case "$prefix" in
        *本文* | *本节*)
          case "$f" in
            docs/ops.md) target=ops ;;
            docs/dev.md) target=dev ;;
          esac
          ;;
      esac
      if [ -z "$target" ]; then
        all_sections | grep -qx "$num" || note "MISS $f: $ref（裸引用，任何文档都没有该编号）"
      else
        grep -qx "$num" "$T/$target" 2>/dev/null || note "MISS $f: $ref -> $target"
      fi
    done
  done < "$f"
}
for f in docs/dev.md docs/ops.md README.md README.en.md AGENTS.md; do
  [ -f "$f" ] && check_refs "$f"
done

# ---- ② 表格列数 ----
for f in docs/dev.md docs/ops.md README.md README.en.md; do
  [ -f "$f" ] || continue
  in_fence=0 expect=-1 lineno=0
  while IFS= read -r line; do
    lineno=$((lineno + 1))
    case "$line" in
      '```'*) [ "$in_fence" -eq 0 ] && in_fence=1 || in_fence=0; continue ;;
    esac
    [ "$in_fence" -eq 1 ] && continue
    case "$line" in
      '|'*'|'*)
        cnt=$(printf '%s' "$line" | sed 's/\\|//g; s/`[^`]*`//g' | tr -cd '|' | wc -c)
        [ "$expect" -eq -1 ] && expect="$cnt"
        [ "$cnt" -ne "$expect" ] && note "COLS $f:$lineno 期望 $expect 列分隔符，实际 $cnt"
        ;;
      *) expect=-1 ;;
    esac
  done < "$f"
done

# ---- ③ 导出符号数 vs 源码 ----
actual=$(grep -h -c 'pub extern "C" fn lasx_' src/ffi/*.rs 2>/dev/null | paste -sd+ - | bc 2>/dev/null)
if [ -n "${actual:-}" ]; then
  # 只取"数字 个导出符号"这一小段（带行号会先把行号当成数字，踩过一次）
  while IFS= read -r hit; do
    f="${hit%%:*}"
    n=$(printf '%s' "$hit" | sed 's/^[^:]*://; s/ .*//')
    [ "$n" = "$actual" ] || note "COUNT $f: 文档写 '$n 个导出符号'，源码实际 $actual"
  done < <(grep -HoE '[0-9]+ 个导出符号' docs/*.md README.md README.en.md 2>/dev/null)
fi

if [ "$findings" -eq 0 ]; then
  echo "doccheck: OK（引用 / 表格列数 / 符号数 一致）"
else
  echo "doccheck: $findings 处待修"
  exit 1
fi
