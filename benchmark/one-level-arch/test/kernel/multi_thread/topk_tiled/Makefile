TESTCASE ?= topk_tiled

ifeq ($(TESTCASE),topk_tiled)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/topk_tiled.cpp
TARGET = $(ELF_HEAD)_topk_tiled_PE4.elf
else
$(error Unsupported TESTCASE=$(TESTCASE); expected topk_tiled)
endif

include ../../../common/Makefile.common
