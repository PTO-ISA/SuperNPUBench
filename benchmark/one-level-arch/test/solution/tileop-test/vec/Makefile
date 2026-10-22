ifneq ($(filter elementwise_if_for elementwise_if_for_strong elementwise_masked_gather_scatter elementwise_masked_gather_scatter_strong,$(TESTCASE)),)
# Reuse the fresh generic-element compiler/runtime lane for this migrated case.
SUITE_DIR := $(abspath ../../../kernel/element_wise/tile_element_suite)
.DEFAULT_GOAL := all
.PHONY: all clean diss ir
all clean diss ir:
	$(MAKE) -C $(SUITE_DIR) $@
else
SRC_FILE = src/$(TESTCASE).cpp
TARGET = $(ELF_HEAD)/$(TESTCASE).elf
ifneq ($(filter elementwise_atomic_histogram elementwise_if_predicates,$(TESTCASE)),)
CC_OPTS += -mllvm -linxv5-enable-generic-element=true \
	-fno-vectorize -fno-slp-vectorize -Werror=unknown-pragmas
endif
include ../Makefile.common
endif
