# Micro-benchmark isolating the histogram_cumsum_m32 operator
# (kernels/multi_thread/histogram_cumsum_m32/histogram_cumsum_m32.hpp).  One
# per-PE hist[384] is initialized once, then suffix_cumsum runs kIters times
# inside BENCHSTART/END (set CUMSUM_ITERS to change kIters).
#
# res_check=on is not meaningful here (no golden); the default build is the
# artifact to hand the cycle simulator (gfsim) / swimlane.

TESTCASE ?= suffix_cumsum_perf

ifeq ($(TESTCASE),suffix_cumsum_perf)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/suffix_cumsum_perf.cpp
TARGET = $(ELF_HEAD)_suffix_cumsum_perf_PE4.elf
else
$(error Unsupported TESTCASE=$(TESTCASE); expected suffix_cumsum_perf)
endif

include ../../../common/Makefile.common
