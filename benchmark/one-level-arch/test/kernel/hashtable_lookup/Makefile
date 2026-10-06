TESTCASE ?= hashtable_lookup_simd

ifeq ($(TESTCASE),hashtable_lookup_simd)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/hashtable_lookup_simd.cpp
TARGET = $(ELF_HEAD)_hashtable_lookup_simd_kNum6144_kMaxProbe8_knum_col256_PE4.elf
else ifeq ($(TESTCASE),hashtable_lookup_simt)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/hashtable_lookup_simt.cpp
TARGET = $(ELF_HEAD)_hashtable_lookup_simt_kNum6144_kNumThreads6144_kMaxProbe8_PE4.elf
else
$(error Unsupported TESTCASE=$(TESTCASE))
endif

include ../../common/Makefile.common
