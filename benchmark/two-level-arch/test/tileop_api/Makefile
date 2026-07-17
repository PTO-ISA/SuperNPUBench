ifneq ($(TESTCASE), )
TARGET = $(ELF_HEAD)_$(TESTCASE)_$(PLAT).elf
SRC_FILE +=  $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
endif

include ../common/Makefile.common

# The current direct-boot lane uses the scalar Linx implementations.  The
# retired tile-register constraint path (`Tr`) is intentionally not enabled.
DEFINES := $(filter-out -DENABLE_TENSOR_INSTR,$(DEFINES))

# Every migrated source provides its own freestanding `_start`; do not link the
# shared runtime entry point a second time.
ifeq ($(PLAT), linx)
CC_LINK := -target linx64-linx-none-elf -nostdlib -Wl,-e,_start
endif
