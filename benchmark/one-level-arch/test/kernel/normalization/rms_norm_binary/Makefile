DType ?= __half
G_A ?= 16
G_R ?= 16384
TILE_A ?= 1
TILE_R ?= 1024
POW_R ?= 4096
PE_NUM ?= 1

ifneq ($(PE_NUM),1)
START_FILE = $(ROOT)/test/common/_start_multipe.s
endif

ifeq ($(TESTCASE), rms_norm_binary)
DEFINES += -DDType=$(DType) -DG_A=$(G_A) -DG_R=$(G_R) -DPE_NUM=$(PE_NUM)
ifeq ($(PE_NUM),1)
TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType).elf
else
TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType)_gA$(G_A)_gR$(G_R)_PE$(PE_NUM).elf
endif
endif

ifeq ($(TESTCASE), rms_norm_binary_static)
DEFINES += -DDType=$(DType) -DPE_NUM=$(PE_NUM)
DEFINES += -DG_A=$(G_A) -DG_R=$(G_R) -DTILE_A=$(TILE_A) -DTILE_R=$(TILE_R)
DEFINES += -DPOW_R=$(POW_R)
ifeq ($(PE_NUM),1)
TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType)_gA$(G_A)_gR$(G_R)_tA$(TILE_A)_tR$(TILE_R)_powR$(POW_R).elf
else
TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType)_gA$(G_A)_gR$(G_R)_tA$(TILE_A)_tR$(TILE_R)_powR$(POW_R)_PE$(PE_NUM).elf
endif
endif

SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
include ../../../common/Makefile.common
