# DynamicMxQuant solution test entries. 8 个验证通过的正式用例（tail/nontail ×
# OCP/cuBLAS × FP8/FP4，含 4-PE SPMD 与大 BlockSize 变体），映射到 6 个 kernel 实现。
# 注：两个 *_bigbs 用例是大 BlockSize 场景，现走 nontail plain 单块路径（方案A 切归约轴
# 的 _bigbs kernel 已退休——新工具链 tile 上限 256KB，单块可直接覆盖大 BS）。elf basename
# 固定为 dynamic_mx_quant_<variant>，使 dynamic_mx_quant_data_compare.py 按 basename 找
# compare/<basename> 金标目录。4-PE 变体运行须 gfrun -s softcore.multiThreadNum=4。

ifeq ($(TYPE), TAIL_OCP_FP8)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tail_ocp_fp8.cpp
   TARGET = $(ELF_DIR)/$(TESTCASE)_tail_ocp_fp8.elf
endif

ifeq ($(TYPE), TAIL_OCP_FP8_DYN)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tail_ocp_fp8_dyn.cpp
   TARGET = $(ELF_DIR)/$(TESTCASE)_tail_ocp_fp8_dyn.elf
endif

ifeq ($(TYPE), TAIL_OCP_FP4)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tail_ocp_fp4.cpp
   TARGET = $(ELF_DIR)/$(TESTCASE)_tail_ocp_fp4.elf
endif

# TAIL_OCP_FP4 大 shape 基准用例 [15360,1536] BS=32（专用 driver=专用 .o，避免与 512×256 共享）。
ifeq ($(TYPE), TAIL_OCP_FP4_BENCH)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tail_ocp_fp4_bench.cpp
   TARGET = $(ELF_DIR)/$(TESTCASE)_tail_ocp_fp4_bench.elf
endif

ifeq ($(TYPE), TAIL_CUBLAS_FP8_4PE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tail_cublas_fp8_4pe.cpp
   TARGET = $(ELF_DIR)/$(TESTCASE)_tail_cublas_fp8_4pe.elf
endif

ifeq ($(TYPE), NONTAIL_CUBLAS_FP8_4PE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_cublas_fp8_4pe.cpp
   TARGET = $(ELF_DIR)/$(TESTCASE)_nontail_cublas_fp8_4pe.elf
endif

ifeq ($(TYPE), NONTAIL_CUBLAS_FP8_BS128)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_cublas_fp8_bs128.cpp
   TARGET = $(ELF_DIR)/$(TESTCASE)_nontail_cublas_fp8_bs128.elf
endif

ifeq ($(TYPE), NONTAIL_OCP_FP4_4PE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_ocp_fp4_4pe.cpp
   TARGET = $(ELF_DIR)/$(TESTCASE)_nontail_ocp_fp4_4pe.elf
endif

ifeq ($(TYPE), NONTAIL_OCP_FP4_BS128)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_ocp_fp4_bs128.cpp
   TARGET = $(ELF_DIR)/$(TESTCASE)_nontail_ocp_fp4_bs128.elf
endif

# --- 运行期动态 shape（_dyn）家族 ---
# kernel 编译期不知 M/N（或 Axis/Post），靠运行期 tiling 传入。以下均支持任意 N。
# kPeNum=4 时必须用 4 线程跑：gfrun -s softcore.multiThreadNum=4。

# TAIL_OCP_FP4 动态 shape 版（M/N 运行期 tiling 传入；fp4 打包 + 三守卫 recip）。
ifeq ($(TYPE), TAIL_OCP_FP4_DYN)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tail_ocp_fp4_dyn.cpp
   TARGET = $(ELF_DIR)/$(TESTCASE)_tail_ocp_fp4_dyn.elf
endif

# TAIL_CUBLAS_FP8 动态 shape 版（M/K 运行期 tiling 传入；cuBLAS scale，fp16 in）。
ifeq ($(TYPE), TAIL_CUBLAS_FP8_DYN)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tail_cublas_fp8_dyn.cpp
   TARGET = $(ELF_DIR)/$(TESTCASE)_tail_cublas_fp8_dyn.elf
endif

# NONTAIL_CUBLAS_FP8 动态 shape 版（Axis/Post 运行期 tiling 传入；按块行 kb 切分 4-PE）。
ifeq ($(TYPE), NONTAIL_CUBLAS_FP8_DYN)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_cublas_fp8_dyn.cpp
   TARGET = $(ELF_DIR)/$(TESTCASE)_nontail_cublas_fp8_dyn.elf
endif

# NONTAIL_CUBLAS_FP8 动态 shape + 按自由轴 N（列 tile）切分 4-PE 版。
ifeq ($(TYPE), NONTAIL_CUBLAS_FP8_SPLITN_DYN)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_cublas_fp8_splitN_dyn.cpp
   TARGET = $(ELF_DIR)/$(TESTCASE)_nontail_cublas_fp8_splitN_dyn.elf
endif

# NONTAIL_OCP_FP4 动态 shape 版（Axis/Post 运行期 tiling 传入；按块行 kb 切分 4-PE）。
ifeq ($(TYPE), NONTAIL_OCP_FP4_DYN)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_ocp_fp4_dyn.cpp
   TARGET = $(ELF_DIR)/$(TESTCASE)_nontail_ocp_fp4_dyn.elf
endif

# NONTAIL_OCP_FP4 动态 shape + 按自由轴 N（列 tile）切分 4-PE 版。
ifeq ($(TYPE), NONTAIL_OCP_FP4_SPLITN_DYN)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_ocp_fp4_splitN_dyn.cpp
   TARGET = $(ELF_DIR)/$(TESTCASE)_nontail_ocp_fp4_splitN_dyn.elf
endif

# 命令行可覆盖的 driver 参数：**全部**转发为 -D 宏，driver 内 #ifndef 默认值兜底。
# 关键：凡 driver 里出现的 #ifndef 宏都必须在此转发，否则 `make VAR=x` 会被静默忽略
#       （历史教训：PTILEN 未转发导致覆盖失效）。任一新增 driver 宏须同步登记到这里。
#   形状：tail driver 用 PM/PN；nontail driver 用 PAXIS/PPOST。
#   结构常量（默认已正确）：PBS=BlockSize、PTILEN=physical TileN(=128, TCVT physical-Row 契约)、
#       PPE=kPeNum。覆盖为非法值会在编译期响亮报错（非静默），故转发安全。
ifdef PM
   DEFINES += -DPM=$(PM)
endif
ifdef PN
   DEFINES += -DPN=$(PN)
endif
ifdef PAXIS
   DEFINES += -DPAXIS=$(PAXIS)
endif
ifdef PPOST
   DEFINES += -DPPOST=$(PPOST)
endif
ifdef PBS
   DEFINES += -DPBS=$(PBS)
endif
ifdef PTILEN
   DEFINES += -DPTILEN=$(PTILEN)
endif
ifdef PPE
   DEFINES += -DPPE=$(PPE)
endif

include ../../../common/Makefile.common
