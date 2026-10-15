# lasx_rs 的 C 示例：编译 / 运行 quickstart.c
#
# 前置（二选一）：
#   ① 安装到某个前缀，让 pkg-config 能找到（推荐）：
#        cargo build --release
#        scripts/install.sh --prefix=$HOME/.local          # 或用 /usr/local
#        export PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig  # 或 make PKG_CONFIG_PATH=…
#   ② 不装，直接用仓库里的 target/release（本 Makefile 会自动兜底）：
#        cargo build --release && make run
#
# 常用：
#   make            # 编译出 ./quickstart
#   make run        # 编译并运行
#   make FALLBACK=1 # 强制走"直接链接 target/release"（忽略 pkg-config）
#   make clean
#
# pkg-config 元数据由 `lasx_rs.pc.in` 生成（scripts/install.sh），里面已经有
# `Libs: -L${libdir} -llasx_rs` 与 `Cflags: -I${includedir}`；本 Makefile 额外把 rpath
# 写进可执行文件，这样 `make run` 不需要设 LD_LIBRARY_PATH。
# （不想要 rpath 的话：`make RUN_ENV="LD_LIBRARY_PATH=…"`，或用 `readelf -d` 看依赖名。）

CC      ?= cc
PKG_CONFIG ?= pkg-config
CFLAGS  ?= -O2 -Wall -Wextra
REPO    := $(abspath $(CURDIR)/../..)

# pkg-config 找不到 lasx_rs 时的兜底：直接用仓库构建产物
FALLBACK ?=

PC_CFLAGS := $(shell $(PKG_CONFIG) --cflags lasx_rs 2>/dev/null)
PC_LIBS   := $(shell $(PKG_CONFIG) --libs lasx_rs 2>/dev/null)
PC_LIBDIR := $(shell $(PKG_CONFIG) --variable=libdir lasx_rs 2>/dev/null)

ifeq ($(FALLBACK),1)
  PC_CFLAGS :=
  PC_LIBS   :=
endif

ifeq ($(strip $(PC_CFLAGS)$(PC_LIBS)),)
  # 未安装 / 未设 PKG_CONFIG_PATH：兜底到仓库（输出一行说明，不静默）
  $(info [make] pkg-config 未找到 lasx_rs ⇒ 直接链接 $(REPO)/target/release)
  $(info [make] 想用安装树：先 scripts/install.sh --prefix=<dir> 并设 PKG_CONFIG_PATH=<dir>/lib/pkgconfig)
  CFLAGS_USE := -I$(REPO)/include
  LIBS_USE   := -L$(REPO)/target/release -llasx_rs -Wl,-rpath,$(REPO)/target/release
else
  CFLAGS_USE := $(PC_CFLAGS)
  LIBS_USE   := $(PC_LIBS)
  ifneq ($(strip $(PC_LIBDIR)),)
    LIBS_USE += -Wl,-rpath,$(PC_LIBDIR)
  endif
endif

all: quickstart

quickstart: quickstart.c
	$(CC) $(CFLAGS) $(CFLAGS_USE) -o $@ $< $(LIBS_USE) -lm

run: quickstart
	$(RUN_ENV) ./quickstart

clean:
	rm -f quickstart

.PHONY: all run clean
