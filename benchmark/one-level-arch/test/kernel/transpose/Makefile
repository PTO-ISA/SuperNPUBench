TESTCASE ?= transpose

ifeq ($(TESTCASE),transpose)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/transpose.cpp
TARGET = $(ELF_HEAD)_transpose_PE4.elf
else ifeq ($(TESTCASE),transpose_minrepro)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/transpose_minrepro.cpp
TARGET = $(ELF_HEAD)_transpose_minrepro_PE4.elf
else
$(error Unsupported TESTCASE=$(TESTCASE); expected transpose | transpose_minrepro)
endif

include ../../common/Makefile.common
