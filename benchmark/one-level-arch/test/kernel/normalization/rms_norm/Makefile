DType ?= __half
G_A ?= 16
G_R ?= 512
TILE_A ?= 2
TILE_R ?= 512
# 1 = single PE (default). 4 = split A across PEs via get_thread_idx();
# gfrun needs -s softcore.multiThreadNum=4 (same as multi_thread/vec).
PE_NUM ?= 1

ifneq ($(PE_NUM),1)
START_FILE = $(ROOT)/test/common/_start_multipe.s
endif

ifeq ($(TESTCASE), rms_norm)
DEFINES += -DDType=$(DType) -DPE_NUM=$(PE_NUM)
ifeq ($(PE_NUM),1)
TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType).elf
else
TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType)_PE$(PE_NUM).elf
endif
endif

ifeq ($(TESTCASE), rms_norm_static)
DEFINES += -DDType=$(DType) -DPE_NUM=$(PE_NUM)
DEFINES += -DG_A=$(G_A) -DG_R=$(G_R) -DTILE_A=$(TILE_A) -DTILE_R=$(TILE_R)
ifeq ($(PE_NUM),1)
TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType)_gA$(G_A)_gR$(G_R)_tA$(TILE_A)_tR$(TILE_R).elf
else
TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType)_gA$(G_A)_gR$(G_R)_tA$(TILE_A)_tR$(TILE_R)_PE$(PE_NUM).elf
endif
endif

SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
include ../../../common/Makefile.common
