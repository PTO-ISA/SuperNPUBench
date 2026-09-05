TESTCASE ?= gelu

ifeq ($(TESTCASE),gelu)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/gelu.cpp
TARGET = $(ELF_HEAD)_gelu_PE4.elf
else
$(error Unsupported TESTCASE=$(TESTCASE); expected gelu)
endif

include ../../../../common/Makefile.common
