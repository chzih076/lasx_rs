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
  # cargo 只产出 `liblasx_rs.so`，而产物的 **SONAME 是 `liblasx_rs.so.0`**：
  # 运行时 loader 按 SONAME 找文件，所以"直接在 target/release 上链接"必须先补这个符号链接
  # （这是真实会踩的坑；装到 prefix 时由 `scripts/install.sh` 负责建它）。
  SONAME_LINK := $(REPO)/target/release/liblasx_rs.so.0
else
  CFLAGS_USE := $(PC_CFLAGS)
  LIBS_USE   := $(PC_LIBS)
  ifneq ($(strip $(PC_LIBDIR)),)
    LIBS_USE += -Wl,-rpath,$(PC_LIBDIR)
  endif
endif

all: quickstart

quickstart: quickstart.c $(SONAME_LINK)
	$(CC) $(CFLAGS) $(CFLAGS_USE) -o $@ $< $(LIBS_USE) -lm

# 兜底分支才用得到：补齐 SONAME 需要的文件名（内容就是同一个文件）
$(SONAME_LINK): $(REPO)/target/release/liblasx_rs.so
	ln -sf liblasx_rs.so $@

# 没构建过时给一句人话，而不是让 make 报 "No rule to make target"
$(REPO)/target/release/liblasx_rs.so:
	@echo "[make] 缺少 $@：请先在仓库根目录跑 \`cargo build --release\`" >&2; exit 1

run: quickstart
	$(RUN_ENV) ./quickstart

clean:
	rm -f quickstart

.PHONY: all run clean
