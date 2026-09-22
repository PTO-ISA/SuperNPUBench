# Regression microbenchmark for same-address GM atomic add serialization.
#
# Four PEs, each owning one counter word, issue kMaxChunks back-to-back full
# [32,1] I32 MGATHER.ADD tiles with index 0, so every lane of every chunk hits
# that single word.  kernels/multi_thread/topk/topk_tiled.hpp's slot allocator
# (H[bin+1]++) depends on the old values being a permutation of
# 0..chunks*32-1, which this case pins down.
#
#   make TESTCASE=gm_atom_dup res_check=on diss
#
# src/run_gm_atom_dup_check.py drives the shape through chunks.bin and checks
# both the old-value stream (out.bin) and the final counters (counter.bin).

TESTCASE ?= gm_atom_dup

ifeq ($(TESTCASE),gm_atom_dup)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/gm_atom_dup.cpp
TARGET = $(ELF_HEAD)_gm_atom_dup_PE4.elf
else
$(error Unsupported TESTCASE=$(TESTCASE); expected gm_atom_dup)
endif

include ../../../common/Makefile.common
