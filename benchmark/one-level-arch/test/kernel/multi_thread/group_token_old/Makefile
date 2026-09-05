# Multi-thread group_token_old operator.

TESTCASE ?= group_token_old

ifeq ($(TESTCASE),group_token_old)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/group_token_old.cpp
TARGET = $(ELF_HEAD)_$(TESTCASE).elf
else
$(error Unsupported TESTCASE=$(TESTCASE); expected group_token_old)
endif

include ../../../common/Makefile.common
