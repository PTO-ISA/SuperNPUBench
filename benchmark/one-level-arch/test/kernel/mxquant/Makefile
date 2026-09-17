# Multi-thread MX quantization (OCP MX v1.0 section 6.3): one [512, 256] BF16
# tensor quantized to E4M3 payload plus group-32 E8M0 scales, in CUBE_M32
# layout.  The four PEs slice the tensor by M via get_thread_idx().
#
# |x| is folded as CUBE-aware max(x, -x) instead of TABS (see
# kernels/basic_op/mxquant/mxquant.hpp and LinxISA/SuperScalarModel issue
# #678).  The kernel is TLSU bound until contiguous load + subview + assemble
# land.
#
# TESTCASE is the source basename: the `clean` rule in Makefile.common only
# removes ${TESTCASE}.o, and there is no header dependency tracking.
#
# Run with four PEs, e.g.
#   gfrun -s softcore.multiThreadNum=4 -f <elf>
#
# res_check=on builds the verifying variant: PE0 reads input.bin from CHK_DIR,
# derives the scalar golden from it, and exports both for src/run_mxquant_check.py
# to compare byte-for-byte.  The default build runs the tile kernel alone and is
# the one to hand a cycle simulator.
#
# Three TIMING-ONLY siblings exist for performance work (no output, so they are
# deliberately NOT in compile.all and must not be res_checked):
#   mxquant_compute       no TLOAD/TSTORE, input from TEXPANDS
#   mxquant_compute_tabs  same, but |x| = TABS (invalid on gfrun, issue #678)
#   mxquant_texpands      only the input TEXPANDS, to subtract its cost
# Build/compare them with src/../perf_compare.sh.

TESTCASE ?= mxquant

ifeq ($(TESTCASE),mxquant)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/mxquant.cpp
TARGET = $(ELF_HEAD)_mxquant_PE4.elf
else ifeq ($(TESTCASE),mxquant_compute)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/mxquant_compute.cpp
TARGET = $(ELF_HEAD)_mxquant_compute_PE4.elf
else ifeq ($(TESTCASE),mxquant_compute_tabs)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/mxquant_compute_tabs.cpp
TARGET = $(ELF_HEAD)_mxquant_compute_tabs_PE4.elf
else ifeq ($(TESTCASE),mxquant_texpands)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/mxquant_texpands.cpp
TARGET = $(ELF_HEAD)_mxquant_texpands_PE4.elf
else
$(error Unsupported TESTCASE=$(TESTCASE); expected mxquant, mxquant_compute, \
mxquant_compute_tabs, or mxquant_texpands)
endif

include ../../common/Makefile.common
