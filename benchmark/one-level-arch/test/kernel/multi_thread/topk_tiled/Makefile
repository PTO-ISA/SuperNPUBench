TESTCASE ?= topk_tiled

ifeq ($(TESTCASE),topk_tiled)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/topk_tiled.cpp
TARGET = $(ELF_HEAD)_topk_tiled_PE4.elf
else
$(error Unsupported TESTCASE=$(TESTCASE); expected topk_tiled)
endif

ifeq ($(GFSIM),on)
DEFINES += -DGFSIM
endif

# Portable test ELFs read/write files relative to their run directory.
CHK_DIR ?= .

include ../../../common/Makefile.common

# The common rules do not track headers or mode flags. Order their clean
# prerequisite before compiling, so -j4 also rebuilds when GFSIM changes.
$(OBJ): clean
clean: pre_work
