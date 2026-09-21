# Functional test entry for the 256-bin suffix cumsum operator
# (kernels/multi_thread/histogram_cumsum_m32/histogram_cumsum_m32.hpp).
# Each PE runs the operator exactly ONCE over its 256-bin histogram.
#
#   make TESTCASE=histogram_cumsum_m32 res_check=on diss
#
# The res_check=on build reads hist.bin and exports result.bin, compared by
# src/run_histogram_cumsum_m32_check.py. The default build is the artifact for
# the cycle model (same as test/kernel/multi_thread/suffix_cumsum_perf).

TESTCASE ?= histogram_cumsum_m32

ifeq ($(TESTCASE),histogram_cumsum_m32)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/histogram_cumsum_m32.cpp
TARGET = $(ELF_HEAD)_PE4.elf
else
$(error Unsupported TESTCASE=$(TESTCASE); expected histogram_cumsum_m32)
endif

include ../../../common/Makefile.common
