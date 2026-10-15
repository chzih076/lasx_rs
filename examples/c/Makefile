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

# ---- 架构防呆：本示例只在 LoongArch64 上能编译/运行 ----
# 头文件自己也有 `#error`（见 include/lasx_rs.h 顶部），这里把话说得更早、更具体。
# FOREIGN=1：显式跳过检查（交叉编译 / 只做文档与语法检查的场景）——它会同时定义
# `LASX_RS_ALLOW_FOREIGN_ARCH` 关掉头文件那道 `#error`。**只关编译期检查，运行时仍不可用。**
HOST_ARCH := $(shell uname -m)
CFLAGS_ARCH :=
ifneq ($(HOST_ARCH),loongarch64)
  ifeq ($(FOREIGN),1)
    $(info [make] 非 LoongArch64（uname -m = $(HOST_ARCH)）但 FOREIGN=1：跳过架构防呆，仅用于交叉/文档场景)
    CFLAGS_ARCH := -DLASX_RS_ALLOW_FOREIGN_ARCH
  else
    $(error [make] 本机架构是 "$(HOST_ARCH)"，不是 loongarch64：lasx_rs 是 LoongArch LASX/LSX 指令集绑定，示例只能在 LoongArch64 上编译/运行。确需交叉编译或在别的架构上做语法/文档检查时，显式加 FOREIGN=1（只关掉编译期检查，不会让运行时可用）)
  endif
endif

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

all: quickstart matmul_mt_caller matmul_mt_convenience

quickstart: quickstart.c $(SONAME_LINK)
	$(CC) $(CFLAGS) $(CFLAGS_ARCH) $(CFLAGS_USE) -o $@ $< $(LIBS_USE) -lm

# 两个多核示例（计划句柄，docs/c-abi.md 的"计划与句柄"一节）：都要 -lpthread。
# ① 调用方自己开线程 + plan 复用；② 便利层一行（并报告实际 worker 数）。
matmul_mt_caller: matmul_mt_caller.c $(SONAME_LINK)
	$(CC) $(CFLAGS) $(CFLAGS_ARCH) $(CFLAGS_USE) -o $@ $< $(LIBS_USE) -lm -lpthread

matmul_mt_convenience: matmul_mt_convenience.c $(SONAME_LINK)
	$(CC) $(CFLAGS) $(CFLAGS_ARCH) $(CFLAGS_USE) -o $@ $< $(LIBS_USE) -lm -lpthread

# 兜底分支才用得到：补齐 SONAME 需要的文件名（内容就是同一个文件）
$(SONAME_LINK): $(REPO)/target/release/liblasx_rs.so
	ln -sf liblasx_rs.so $@

# 没构建过时给一句人话，而不是让 make 报 "No rule to make target"
$(REPO)/target/release/liblasx_rs.so:
	@echo "[make] 缺少 $@：请先在仓库根目录跑 \`cargo build --release\`" >&2; exit 1

run: all
	$(RUN_ENV) ./quickstart
	$(RUN_ENV) ./matmul_mt_caller
	$(RUN_ENV) ./matmul_mt_convenience

clean:
	rm -f quickstart matmul_mt_caller matmul_mt_convenience

.PHONY: all run clean
