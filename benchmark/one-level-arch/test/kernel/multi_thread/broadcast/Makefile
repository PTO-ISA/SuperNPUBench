TESTCASE ?= broadcast

ifeq ($(TESTCASE),broadcast)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/broadcast.cpp
TARGET = $(ELF_HEAD)_broadcast_PE4.elf
else
$(error Unsupported TESTCASE=$(TESTCASE); expected broadcast)
endif

include ../../../common/Makefile.common
