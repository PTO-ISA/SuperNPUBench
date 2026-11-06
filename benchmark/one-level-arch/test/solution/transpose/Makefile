TESTCASE ?= transpose_v32

ifeq ($(TESTCASE),transpose_v31)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/transpose_v31.cpp
TARGET = $(ELF_HEAD)_transpose_v31_PE4.elf
V31_CASE ?= 0
DEFINES += -DV31_CASE=$(V31_CASE)
else ifeq ($(TESTCASE),transpose_v32)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/transpose_v32.cpp
TARGET = $(ELF_HEAD)_transpose_v32_PE4.elf
V32_CASE ?= 0
DEFINES += -DV32_CASE=$(V32_CASE)
V32_TILE_ELEMENTS ?= 512
else ifeq ($(TESTCASE),transpose_simt)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/transpose_simt.cpp
TARGET = $(ELF_HEAD)_transpose_simt_PE4.elf
SIMT_CASE ?= 0
SIMT_PE ?= 4
DEFINES += -DSIMT_CASE=$(SIMT_CASE) -DSIMT_PE=$(SIMT_PE) -DENABLE_BINARY_OUTPUT
CC_OPTS += -mllvm -linxv5-enable-generic-element=true
else
$(error Unsupported TESTCASE=$(TESTCASE); expected transpose_v31 | transpose_v32 | transpose_simt)
endif

include ../../common/Makefile.common
