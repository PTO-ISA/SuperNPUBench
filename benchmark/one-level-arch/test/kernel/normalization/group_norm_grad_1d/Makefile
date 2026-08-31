N_BATCH ?= 8
C_CH ?= 64
G_GRP ?= 8
TILE_D ?= -1
DType ?= __half
PE_NUM ?= 1

ifneq ($(PE_NUM),1)
START_FILE = $(ROOT)/test/common/_start_multipe.s
endif

ifeq ($(TESTCASE), group_norm_grad_1d)
DEFINES += -DDType=$(DType)
DEFINES += -DN_BATCH=$(N_BATCH)
DEFINES += -DC_CH=$(C_CH)
DEFINES += -DG_GRP=$(G_GRP)
DEFINES += -DTILE_D=$(TILE_D)
DEFINES += -DPE_NUM=$(PE_NUM)
ifeq ($(PE_NUM),1)
TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType)_N$(N_BATCH)_C$(C_CH)_G$(G_GRP).elf
else
TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType)_N$(N_BATCH)_C$(C_CH)_G$(G_GRP)_PE$(PE_NUM).elf
endif
endif

ifeq ($(TESTCASE), group_norm_grad_1d_static)
GN1D_TILE_D := $(TILE_D)
ifeq ($(GN1D_TILE_D),-1)
GN1D_TILE_D := 8
endif
DEFINES += -DDType=$(DType)
DEFINES += -DN_BATCH=$(N_BATCH)
DEFINES += -DC_CH=$(C_CH)
DEFINES += -DG_GRP=$(G_GRP)
DEFINES += -DTILE_D=$(GN1D_TILE_D)
TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType)_N$(N_BATCH)_C$(C_CH)_G$(G_GRP)_tD$(GN1D_TILE_D).elf
endif

SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
include ../../../common/Makefile.common
