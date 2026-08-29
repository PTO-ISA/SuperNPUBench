TESTCASE ?= rms_norm

ifeq ($(TESTCASE),rms_norm)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/rms_norm.cpp
TARGET = $(ELF_HEAD)_rms_norm_PE4.elf
else
$(error Unsupported TESTCASE=$(TESTCASE); expected rms_norm)
endif

include ../../../../common/Makefile.common
