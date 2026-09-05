TESTCASE ?= rms_norm_binary

ifeq ($(TESTCASE),rms_norm_binary)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/rms_norm_binary.cpp
TARGET = $(ELF_HEAD)_rms_norm_binary_PE4.elf
else
$(error Unsupported TESTCASE=$(TESTCASE); expected rms_norm_binary)
endif

include ../../../../common/Makefile.common
