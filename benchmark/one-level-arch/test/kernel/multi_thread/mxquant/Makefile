# Multi-thread MX quantization (OCP MX v1.0 section 6.3) operators.
#
# Three variants share one kernel contract: a [128, kCols] BF16 tensor is
# quantized to E4M3 payload plus group-32 E8M0 scales, and each kernel slices
# the tensor by M across four PEs via get_thread_idx().
#
#   mxquant                  kCols=64,   two strided half-blocks
#   mxquant_1024_assembly    kCols=1024, 8-block batches (higher tile pressure)
#   mxquant_1024_streaming   kCols=1024, one block at a time (~5 KB peak)
#
# TESTCASE selects the variant and is deliberately the source basename: the
# `clean` rule in Makefile.common only removes ${TESTCASE}.o, so a per-variant
# TESTCASE is what keeps a rebuild from silently relinking another variant's
# stale object (there is no header dependency tracking).
#
# Run every variant with four PEs, e.g.
#   gfrun -s softcore.multiThreadNum=4 -f <elf>
#
# res_check=on builds the verifying variant: PE0 reads input.bin from CHK_DIR,
# derives the scalar golden from it, and exports both for src/run_mxquant_check.py
# to compare byte-for-byte.  The default build runs the tile kernel alone and is
# the one to hand a cycle simulator.

TESTCASE ?= mxquant

ifeq ($(TESTCASE),mxquant)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/mxquant.cpp
TARGET = $(ELF_HEAD)_mxquant_PE4.elf
else ifeq ($(TESTCASE),mxquant_1024_assembly)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/mxquant_1024_assembly.cpp
TARGET = $(ELF_HEAD)_mxquant_1024_assembly_PE4.elf
else ifeq ($(TESTCASE),mxquant_1024_streaming)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/mxquant_1024_streaming.cpp
TARGET = $(ELF_HEAD)_mxquant_1024_streaming_PE4.elf
else
$(error Unsupported TESTCASE=$(TESTCASE); expected mxquant, \
mxquant_1024_assembly, or mxquant_1024_streaming)
endif

include ../../../common/Makefile.common
