TESTCASE ?= transpose_v32

ifeq ($(TESTCASE),transpose_v32)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/transpose_v32.cpp
TARGET = $(ELF_HEAD)_transpose_v32_PE4.elf
V32_CASE ?= 0
DEFINES += -DV32_CASE=$(V32_CASE)
V32_TILE_ELEMENTS ?= 512
DEFINES += -DV32_TILE_ELEMENTS=$(V32_TILE_ELEMENTS)
else ifeq ($(TESTCASE),transpose_simt)
# element-if（SIMT/SMALL_SHAPE）路径：需含 element-if 的 Linx LLVM + generic-element lowering
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/transpose_simt.cpp
TARGET = $(ELF_HEAD)_transpose_simt_PE4.elf
SIMT_CASE ?= 0
SIMT_PE ?= 4
DEFINES += -DSIMT_CASE=$(SIMT_CASE) -DSIMT_PE=$(SIMT_PE)
# 必须保留（命令行传入 CC_OPTS 时普通赋值会被忽略，故用 override）
override CC_OPTS += -mllvm -linxv5-enable-generic-element=true
else
$(error Unsupported TESTCASE=$(TESTCASE); expected transpose_v32 | transpose_simt)
endif

include ../../common/Makefile.common
