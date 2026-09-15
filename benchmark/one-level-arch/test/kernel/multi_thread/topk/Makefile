TESTCASE ?= topk

ifeq ($(TESTCASE),topk)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/topk.cpp
TARGET = $(ELF_HEAD)_topk_PE4.elf
else
$(error Unsupported TESTCASE=$(TESTCASE); expected topk)
endif

include ../../../common/Makefile.common
