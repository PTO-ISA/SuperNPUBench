TESTCASE ?= concat_gather

ifeq ($(TESTCASE),concat_gather)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/concat_gather.cpp
else ifeq ($(TESTCASE),concat_scatter)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/concat_scatter.cpp
else
$(error Unsupported TESTCASE=$(TESTCASE); expected concat_gather or concat_scatter)
endif

TARGET = $(ELF_HEAD)_$(TESTCASE)_PE4.elf
include ../../../common/Makefile.common
