TESTCASE ?= hashtable_lookup
hash_mode ?= smoke
hash_perf_barrier ?= 0
# Default kernel is adaptive 2+6: the B phase reads up to slot o+7, so it
# needs a 7-entry (112-byte) guard tail.  Override with hash_table_guard=48
# when selecting the 2+2 (HASH_ADAPTIVE_TAIL=2) or stride4 implementation.
hash_table_guard ?= 112
DEFINES += -DHASH_PERF_BARRIER=$(hash_perf_barrier) -DHASH_TABLE_GUARD=$(hash_table_guard)
ifneq ($(filter $(hash_mode),perf verify),)
ifeq ($(res_check),on)
$(error hash_mode=$(hash_mode) uses the freestanding dataset harness; omit res_check=on)
endif
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/hashtable_lookup_dataset.cpp
TARGET = $(ELF_HEAD)_hashtable_lookup_6144_$(hash_mode)_PE4.elf
DEFINES += -DHASH_VERIFY=$(if $(filter verify,$(hash_mode)),1,0)
else ifeq ($(hash_mode),smoke)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/hashtable_lookup.cpp
TARGET = $(ELF_HEAD)_hashtable_lookup_PE4.elf
else
$(error hash_mode must be smoke, perf, or verify)
endif
CFLAGS += -mllvm -linxv5-enable-generic-element=true
ifneq ($(HASH_KERNEL_HEADER),)
DEFINES += -DHASH_KERNEL_HEADER='"$(HASH_KERNEL_HEADER)"'
endif
include ../../common/Makefile.common

# Smoke path: no libc/startup objects or compiler-generated libc calls.
ifneq ($(res_check),on)
CC_OPTS += -ffreestanding
CC_LINK = -nostdlib -nostartfiles $(ROOT)/test/common/_start.s
endif

# Mode changes are compile-time changes. Never reuse a perf object for verify
# (or vice versa) when callers share OBJ_ROOT across invocations.
ifneq ($(filter $(hash_mode),perf verify),)
.PHONY: hash_dataset_mode
hash_dataset_mode:
$(OBJ): hash_dataset_mode
endif
