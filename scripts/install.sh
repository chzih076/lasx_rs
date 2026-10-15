#!/usr/bin/env bash
# lasx_rs 安装脚本：把 `cargo build --release` 的产物装进 --prefix，并生成 pkg-config 文件。
#
# 用法：
#   scripts/install.sh --prefix=/usr/local
#   scripts/install.sh --prefix /tmp/stage            # 空格形式也接受（两者等价）
#   CARGO_TARGET_DIR=/tmp/build scripts/install.sh --prefix=/tmp/stage
#
# 装到 <prefix> 下的东西（相对路径）：
#   lib/liblasx_rs.so.<version>  动态库本体（SONAME = liblasx_rs.so.0，见 build.rs）
#   lib/liblasx_rs.so.0          符号链接 → 本体；**运行时**按 SONAME 找这个名字
#   lib/liblasx_rs.so            符号链接 → 本体；**链接期** `-llasx_rs` 找这个名字
#   lib/liblasx_rs.a             静态库（不想带 .so 的 C 消费者）
#   lib/pkgconfig/lasx_rs.pc     pkg-config 元数据（由 lasx_rs.pc.in 生成）
#   include/lasx_rs.h            C 头（由仓库提供；本脚本**不**生成头文件）
#
# 性质：
#   * `set -eu`；任何一步失败都立即退出（但校验都在写入之前，不会留半套安装）。
#   * **幂等**：重复执行只覆盖上面这几个文件、用 `ln -sf` 重建链接，不删别的东西。
#   * 装之前会核对**产物的 SONAME**（readelf 可用时）：不对就拒绝安装——把错误的 SONAME
#     写进系统 libdir 是真实的打包事故，不能静默放过。
#   * 缺产物/缺头文件会给出"该先跑什么"的错误，而不是继续装一半。
set -eu

SONAME=liblasx_rs.so.0

usage() {
  printf '%s\n' \
    '用法：scripts/install.sh --prefix=<安装根目录>' \
    '' \
    '  --prefix=<dir>  安装根目录（必填）；会生成 <dir>/lib、<dir>/include、<dir>/lib/pkgconfig' \
    '  -h, --help      显示本帮助' \
    '' \
    '前置：先在本仓库根目录跑过 `cargo build --release`（需要 liblasx_rs.so 与 liblasx_rs.a），' \
    '      且 include/lasx_rs.h 已存在（C 头由仓库提供；可用 LASX_RS_HEADER=<path> 指向别处）。'
}

# 仓库根 = 本脚本所在目录的上一级（不依赖调用时的 cwd）。
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TARGET_DIR=${CARGO_TARGET_DIR:-$ROOT/target}/release
PC_TEMPLATE=$ROOT/lasx_rs.pc.in
HEADER=${LASX_RS_HEADER:-$ROOT/include/lasx_rs.h}
PREFIX=

while [ $# -gt 0 ]; do
  case $1 in
    --prefix=*) PREFIX=${1#--prefix=} ;;
    --prefix)
      shift
      [ $# -gt 0 ] || { printf '错误：--prefix 后面缺少目录\n' >&2; exit 2; }
      PREFIX=$1
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      printf '错误：不认识的参数 `%s`\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

[ -n "$PREFIX" ] || { printf '错误：必须给 --prefix=<dir>（安装根目录）\n\n' >&2; usage >&2; exit 2; }

# 相对 prefix 会让生成的 .pc 指向错误位置，这里统一解析成绝对路径。
case $PREFIX in
  /*) ;;
  *) PREFIX=$PWD/$PREFIX ;;
esac

SO=$TARGET_DIR/liblasx_rs.so
A=$TARGET_DIR/liblasx_rs.a

# ---- 先在写入之前把所有前提校验完（失败不留半套安装）----
if [ ! -f "$SO" ] || [ ! -f "$A" ]; then
  printf '错误：缺少构建产物：\n' >&2
  [ -f "$SO" ] || printf '  %s\n' "$SO" >&2
  [ -f "$A" ] || printf '  %s\n' "$A" >&2
  printf '      先跑：cargo build --release\n      （找不到就检查 CARGO_TARGET_DIR；当前用的是 %s）\n' "$TARGET_DIR" >&2
  exit 1
fi
if [ ! -f "$PC_TEMPLATE" ]; then
  printf '错误：缺少 pkg-config 模板 %s（源码仓库根目录）\n' "$PC_TEMPLATE" >&2
  exit 1
fi
if [ ! -f "$HEADER" ]; then
  printf '错误：缺少 C 头 %s\n' "$HEADER" >&2
  printf '      头文件由仓库的 include/lasx_rs.h 提供（83 个原型），本脚本不生成它；\n' >&2
  printf '      若头文件放到了别处，用 LASX_RS_HEADER=<path> 指过来。\n' >&2
  exit 1
fi

VERSION=$(grep -m1 '^version' "$ROOT/Cargo.toml" | cut -d'"' -f2)
[ -n "$VERSION" ] || { printf '错误：读不出 Cargo.toml 的 version\n' >&2; exit 1; }

if command -v readelf >/dev/null 2>&1; then
  GOT=$(readelf -d "$SO" | grep -m1 'SONAME' || true)
  case $GOT in
    *"[$SONAME]"*) ;;
    *)
      printf '错误：%s 的 SONAME 不是 %s\n' "$SO" "$SONAME" >&2
      printf '      实际：%s\n' "${GOT:-（readelf 没读到 SONAME 条目）}" >&2
      printf '      检查 build.rs 的 `cargo:rustc-cdylib-link-arg=-Wl,-soname,%s` 是否生效。\n' "$SONAME" >&2
      exit 1
      ;;
  esac
fi

# ---- 安装（幂等）----
LIBDIR=$PREFIX/lib
INCDIR=$PREFIX/include
PCDIR=$LIBDIR/pkgconfig
install -d "$LIBDIR" "$INCDIR" "$PCDIR"

install -m 0755 "$SO" "$LIBDIR/liblasx_rs.so.$VERSION"
ln -sf "liblasx_rs.so.$VERSION" "$LIBDIR/$SONAME"
ln -sf "liblasx_rs.so.$VERSION" "$LIBDIR/liblasx_rs.so"
install -m 0644 "$A" "$LIBDIR/liblasx_rs.a"
install -m 0644 "$HEADER" "$INCDIR/lasx_rs.h"

# .pc：把 @PREFIX@ / @VERSION@ 替换掉（纯 shell 参数展开；先写临时文件再 rename，避免半截文件被读到）
PC=$PCDIR/lasx_rs.pc
while IFS= read -r line || [ -n "$line" ]; do
  line=${line//@PREFIX@/$PREFIX}
  line=${line//@VERSION@/$VERSION}
  printf '%s\n' "$line"
done < "$PC_TEMPLATE" > "$PC.tmp"
mv -f "$PC.tmp" "$PC"

# 自检：符号链接必须指到真文件，否则这个安装树是坏的
for f in "$LIBDIR/$SONAME" "$LIBDIR/liblasx_rs.so"; do
  [ -f "$f" ] || { printf '错误：%s 不是有效链接（目标不存在）\n' "$f" >&2; exit 1; }
done

# ---- 汇报装到哪 + 一行链接示例 ----
printf '已安装到 %s\n' "$PREFIX"
printf '  %s/liblasx_rs.so.%s（+ %s / liblasx_rs.so 两个符号链接）\n' "$LIBDIR" "$VERSION" "$SONAME"
printf '  %s/liblasx_rs.a\n' "$LIBDIR"
printf '  %s\n' "$PC"
printf '  %s/lasx_rs.h\n' "$INCDIR"
printf '\n链接示例（动态库；-Wl,-rpath 让运行期不必设 LD_LIBRARY_PATH）：\n'
printf '  cc example.c -o example -I%s -L%s -llasx_rs -Wl,-rpath,%s\n' "$INCDIR" "$LIBDIR" "$LIBDIR"
printf '未设 rpath 时运行前：\n'
printf '  LD_LIBRARY_PATH=%s ./example\n' "$LIBDIR"
printf '用 pkg-config：\n'
printf '  PKG_CONFIG_PATH=%s pkg-config --cflags --libs lasx_rs\n' "$PCDIR"
printf '静态链接（不带 .so）：\n'
printf '  cc example.c -o example -I%s %s/liblasx_rs.a -lm\n' "$INCDIR" "$LIBDIR"
if [ "$PREFIX" = /usr ] || [ "$PREFIX" = /usr/local ]; then
  printf '\n装到系统目录后建议刷新链接缓存：ldconfig（或把 %s 写进 /etc/ld.so.conf.d/）。\n' "$LIBDIR"
fi
