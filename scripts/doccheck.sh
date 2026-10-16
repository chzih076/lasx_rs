#!/usr/bin/env bash
# 文档一致性校验（纯文本，无需 Rust 工具链；CI 的 fmt job 里跑同一条）。
#
# 校验三件事（都是本项目**实际踩过**的文档缺陷）：
#   ① 每个 `§N[.M]` 引用都能在**目标文档**里找到同名编号的标题；
#   ② 表格每行的列数与表头一致（Markdown 表格错列会静默渲染坏）；
#   ③ 文档里的"导出符号数"与源码里 `extern "C" fn lasx_*` 的实际数量一致。
#
# 引用解析规则（**按文档解析**，2026-10-15 收紧）：
#   - 点名了文档（`docs/dev.md §X`、`docs/ops.md §X`）⇒ 必须命中**那个**文档（可跨文档）；
#   - `本文 §X` / `本节 §X` ⇒ 当前文档；
#   - **裸 `§X` ⇒ 必须在"引用所在文档"自己的编号集合里解析**，不能退而命中别的文档。
#
# 为什么裸引用必须按文档解析：编号集合若是"全部文档的并集"，一旦某个文档新增了同号小节
# （例如 `docs/c-abi.md` 的 §1–10），别处裸写的 `§7` 就会**退而命中它**——引用"能解析"，
# 但实际指向了错的文档 ⇒ 规则变松、漂移不会被发现。所以：裸 `§X` 只允许指本文件；
# 要指别处就点名文档（这也是 `docs/dev.md` 文档规范里那条"跨文档引用必须点名"的机器强制。
#
# 用法：bash scripts/doccheck.sh [仓库根目录]；有发现时以非零退出。
set -u
root="${1:-.}"
cd "$root" || exit 1

T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
findings=0
note() { echo "$1"; findings=$((findings + 1)); }

# 文件路径 → 编号集合的 key（"按文档解析"要用：裸引用只在本文档的集合里找）
doc_key() {
  case "$1" in
    docs/dev.md) echo dev ;;
    docs/ops.md) echo ops ;;
    docs/c-abi.md) echo cabi ;;
    docs/platform.md) echo platform ;;
    docs/dsl.md) echo dsl ;;
    README.md) echo root ;;
    README.en.md) echo rooten ;;
    AGENTS.md) echo agents ;;
  esac
}

# ---- 收集各文档的编号集合 ----
sections() { grep -oE '^#{1,4} [0-9]+(\.[0-9]+)*' "$1" 2>/dev/null | sed 's/^#* //' | sort -u; }
for pair in "docs/dev.md dev" "docs/ops.md ops" "docs/c-abi.md cabi" "docs/platform.md platform" "docs/dsl.md dsl" "README.md root" "README.en.md rooten" "AGENTS.md agents"; do
  f="${pair%% *}"; k="${pair##* }"
  [ -f "$f" ] && sections "$f" > "$T/$k"
done

# ---- ① 引用 ----
check_refs() {
  local f="$1" line ref num prefix target ln=0
  while IFS= read -r line; do
    ln=$((ln + 1))
    # `manual §9.6` 这类是**外部手册**的编号，不是本文档体系（docs/dev.md §16 有说明）
    case "$line" in *'manual §'*) continue ;; esac
    for ref in $(printf '%s' "$line" | grep -oE '§[0-9]+(\.[0-9]+)*'); do
      num="${ref#§}"
      prefix="${line%%"$ref"*}"
      target=""
      # 取**前缀里最后一次**出现的文档名（"就近"语义：一句里可以先提 dev 再提 ops）
      last_doc=$(printf '%s' "$prefix" |
        grep -oE 'docs/c-abi\.md|docs/dev\.md|docs/ops\.md|docs/platform\.md|docs/dsl\.md|README\.en\.md|README\.md|AGENTS\.md' | tail -1)
      case "$last_doc" in
        docs/c-abi.md) target=cabi ;;
        docs/dev.md) target=dev ;;
        docs/ops.md) target=ops ;;
        docs/platform.md) target=platform ;;
        docs/dsl.md) target=dsl ;;
        README.en.md) target=rooten ;;
        README.md) target=root ;;
        AGENTS.md) target=agents ;;
      esac
      # "本文/本节 §x" 指当前文件——但**只在它比最后一个文档名更靠后时**才成立：
      # 反例"判据不在本节（在 `docs/ops.md` §5.15）"里 §5.15 属于 ops（就近是 ops）。
      # 用字节偏移比较"最后一次出现"的位置。
      doc_off=$(printf '%s' "$prefix" |
        grep -bo 'docs/c-abi\.md\|docs/dev\.md\|docs/ops\.md\|docs/dsl\.md\|README\.en\.md\|README\.md\|AGENTS\.md' | tail -1 | cut -d: -f1)
      self_off=$(printf '%s' "$prefix" | grep -bo '本文\|本节' | tail -1 | cut -d: -f1)
      if [ -n "${self_off:-}" ] && { [ -z "${doc_off:-}" ] || [ "$self_off" -gt "$doc_off" ]; }; then
        target=$(doc_key "$f")
      fi
      if [ -z "$target" ]; then
        # 裸引用：**只在本文档自己的编号集合里**解析（见顶部"按文档解析"）
        self_key=$(doc_key "$f")
        grep -qx "$num" "$T/$self_key" 2>/dev/null ||
          note "MISS $f:$ln: $ref（裸引用，本文档没有该编号；指向别的文档请点名文档）"
      else
        grep -qx "$num" "$T/$target" 2>/dev/null || note "MISS $f:$ln: $ref -> $target"
      fi
    done
  done < "$f"
}
for f in docs/dev.md docs/ops.md docs/c-abi.md docs/platform.md docs/dsl.md README.md README.en.md AGENTS.md; do
  [ -f "$f" ] && check_refs "$f"
done

# ---- ② 表格列数 ----
for f in docs/dev.md docs/ops.md docs/c-abi.md README.md README.en.md; do
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
# 纯 shell 求和（不依赖 bc：CI runner 上缺它会让这一步**静默跳过**，等于白进 CI）
actual=0
for f in src/ffi/*.rs; do
  [ -f "$f" ] || continue
  actual=$((actual + $(grep -c 'pub extern "C" fn lasx_' "$f")))
done
if [ "$actual" -gt 0 ]; then
  # 只取"数字 个导出符号"这一小段（带行号会先把行号当成数字，踩过一次）
  while IFS= read -r hit; do
    f="${hit%%:*}"
    n=$(printf '%s' "$hit" | sed 's/^[^:]*://; s/ .*//')
    [ "$n" = "$actual" ] || note "COUNT $f: 文档写 '$n 个导出符号'，源码实际 $actual"
  done < <(grep -HoE '[0-9]+ 个导出符号' docs/*.md README.md README.en.md 2>/dev/null)
fi

# ---- ④ 符号数的**其它写法** ----
# 实例 1：`README.md` 的文档索引曾写"**75 个符号总表**"、`README.en.md` 写 "**the 75-symbol table**"。
# 实例 2（加这条英文写法的直接原因）：加 C 入口把计数改成 83 时，**`README.en.md` 的
# `81 exported symbols (41 plain + 40 checked)` 逃过了检查**（当时只覆盖"个符号…"与 `N-symbol table`），
# 是 `readme-platform` 人工核对时抓到的 ⇒ 这里补上 `N exported symbols` 这一写法。
# **不**用"个符号"泛匹配：文档里合法存在"N3 的 7 个符号"这类**子集**描述，泛匹配会误报
# （实测：加宽后立刻误报 `docs/ops.md` 的"N3 的 7 个符号"一条）。
if [ "$actual" -gt 0 ]; then
  while IFS= read -r hit; do
    f="${hit%%:*}"
    n=$(printf '%s' "$hit" | sed 's/^[^:]*://; s/ 个.*//; s/ exported.*//; s/-.*//')
    [ "$n" = "$actual" ] || note "COUNT $f: 文档写 '$n 个符号总表/汇总'（或英文 $n-symbol / $n exported symbols），源码实际 $actual"
  done < <(grep -HoE '[0-9]+ 个符号(总表|一览|汇总|总数|共)|[0-9]+-symbol (table|list|summary)|[0-9]+ exported symbols' \
             docs/*.md README.md README.en.md 2>/dev/null)
fi

# ---- ⑤ 引用噪音：同一文档里同一 `§X` 被引用 >12 次 ----
# 本次系统性整理的目标是"别到处引用"（用户要求）。实测：同文档内同一目标的重复指向会滚到
# 20+ 次（`docs/dev.md §7.9` 曾 **24** 次、`docs/ops.md §7.9` 17 次），修掉后再加门禁——
# 本仓库的既有教训是"**没有门禁的规范会复发**"（`doccheck` 抓过 4 次结构性破坏）。
# 阈值 12：整理后当前最高为 `docs/dev.md §6.3` 的 10 次，留 2 次余量。
for f in docs/dev.md docs/ops.md docs/c-abi.md docs/platform.md README.md README.en.md AGENTS.md; do
  [ -f "$f" ] || continue
  while IFS= read -r hit; do
    n="${hit%% *}"
    ref="${hit##* }"
    if [ "${n:-0}" -gt 12 ] 2>/dev/null; then
      note "NOISE $f: $ref 被引用 $n 次（>12）"
    fi
  done < <(grep -oE '§[0-9]+(\.[0-9]+)*' "$f" | sort | uniq -c | sed 's/^ *//')
done

if [ "$findings" -eq 0 ]; then
  echo "doccheck: OK（引用 / 表格列数 / 符号数 一致）"
else
  echo "doccheck: $findings 处待修"
  exit 1
fi
