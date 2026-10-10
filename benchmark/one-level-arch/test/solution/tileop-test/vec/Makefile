ifeq ($(TESTCASE),elementwise_atomic_histogram)
# Reuse the fresh generic-element compiler/runtime lane for this migrated case.
SUITE_DIR := $(abspath ../../../kernel/element_wise/tile_element_suite)
.DEFAULT_GOAL := all
.PHONY: all clean diss ir
all clean diss ir:
	$(MAKE) -C $(SUITE_DIR) $@
else
SRC_FILE = src/$(TESTCASE).cpp
TARGET = $(ELF_HEAD)/$(TESTCASE).elf
include ../Makefile.common
endif
