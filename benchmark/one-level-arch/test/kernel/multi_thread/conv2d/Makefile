TESTCASE ?= v300_conv2d

ifeq ($(TESTCASE),v300_conv2d)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/v300_conv2d.cpp
TARGET = $(ELF_HEAD)_v300_conv2d_PE4.elf
else
$(error Unsupported TESTCASE=$(TESTCASE); expected v300_conv2d)
endif

include ../../../common/Makefile.common
