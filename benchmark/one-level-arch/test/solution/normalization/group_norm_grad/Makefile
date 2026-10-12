DType ?= __half
PE_NUM ?= 4
ifneq ($(PE_NUM),4)
$(error group_norm_grad supports only the 4PE testcases)
endif
START_FILE = $(ROOT)/test/common/_start_multipe.s

# HxW > 1
ifneq ($(filter group_norm_grad_dynamic group_norm_grad_static,$(TESTCASE)),)
N_BATCH ?= 2
C_CH ?= 16
G_GRP ?= 8
HxW_SZ ?= 2048
DEFINES += -DDType=$(DType) -DN_BATCH=$(N_BATCH) -DC_CH=$(C_CH)
DEFINES += -DG_GRP=$(G_GRP) -DHxW_SZ=$(HxW_SZ) -DPE_NUM=$(PE_NUM)
TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType)_N$(N_BATCH)_C$(C_CH)_G$(G_GRP)_HxW$(HxW_SZ)_PE$(PE_NUM).elf
endif

# HxW == 1 (1D)
ifneq ($(filter group_norm_grad_1d_dynamic group_norm_grad_1d_static,$(TESTCASE)),)
N_BATCH ?= 256
C_CH ?= 512
G_GRP ?= 8
GB_TILE_D ?= 0
GB_TILE_G ?= 0
DEFINES += -DGB_TILE_D=$(GB_TILE_D) -DGB_TILE_G=$(GB_TILE_G)
DEFINES += -DDType=$(DType) -DN_BATCH=$(N_BATCH) -DC_CH=$(C_CH)
DEFINES += -DG_GRP=$(G_GRP) -DPE_NUM=$(PE_NUM)
TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType)_N$(N_BATCH)_C$(C_CH)_G$(G_GRP)_PE$(PE_NUM).elf
endif

SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
include ../../../common/Makefile.common
