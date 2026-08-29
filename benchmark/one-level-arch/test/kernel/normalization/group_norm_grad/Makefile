N_BATCH ?= 2
C_CH ?= 16
G_GRP ?= 4
HxW_SZ ?= 16
TILE_HW ?= 8
DType ?= __half
PE_NUM ?= 1

ifneq ($(PE_NUM),1)
START_FILE = $(ROOT)/test/common/_start_multipe.s
endif

ifeq ($(TESTCASE), group_norm_grad)
DEFINES += -DDType=$(DType)
DEFINES += -DN_BATCH=$(N_BATCH)
DEFINES += -DC_CH=$(C_CH)
DEFINES += -DG_GRP=$(G_GRP)
DEFINES += -DHxW_SZ=$(HxW_SZ)
DEFINES += -DTILE_HW=$(TILE_HW)
DEFINES += -DPE_NUM=$(PE_NUM)
ifeq ($(PE_NUM),1)
TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType)_N$(N_BATCH)_C$(C_CH)_G$(G_GRP)_HxW$(HxW_SZ).elf
else
TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType)_N$(N_BATCH)_C$(C_CH)_G$(G_GRP)_HxW$(HxW_SZ)_PE$(PE_NUM).elf
endif
endif

ifeq ($(TESTCASE), group_norm_grad_static)
DEFINES += -DDType=$(DType)
DEFINES += -DN_BATCH=$(N_BATCH)
DEFINES += -DC_CH=$(C_CH)
DEFINES += -DG_GRP=$(G_GRP)
DEFINES += -DHxW_SZ=$(HxW_SZ)
DEFINES += -DTILE_HW=$(TILE_HW)
TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType)_N$(N_BATCH)_C$(C_CH)_G$(G_GRP)_HxW$(HxW_SZ)_tHW$(TILE_HW).elf
endif

SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
include ../../../common/Makefile.common
