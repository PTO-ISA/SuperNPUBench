TESTCASE ?= hashtable_lookup
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/hashtable_lookup.cpp
TARGET = $(ELF_HEAD)_hashtable_lookup_PE4.elf
CFLAGS += -mllvm -linxv5-enable-generic-element=true
include ../../common/Makefile.common

# Smoke path: no libc/startup objects or compiler-generated libc calls.
ifneq ($(res_check),on)
CC_OPTS += -ffreestanding
CC_LINK = -nostdlib -nostartfiles $(ROOT)/test/common/_start.s
endif
