TESTCASE ?= gather

ifeq ($(TESTCASE),gather)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/gather.cpp
TARGET = $(ELF_HEAD)_gather_PE4.elf
else
$(error Unsupported TESTCASE=$(TESTCASE); expected gather)
endif

include ../../../common/Makefile.common
