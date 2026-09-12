N_BATCH ?= 256
C_CH ?= 256
G_GRP ?= 8
DType ?= __half
PE_NUM ?= 4
GB_TILE_D ?= 0
GB_TILE_G ?= 0
DEFINES += -DGB_TILE_D=$(GB_TILE_D) -DGB_TILE_G=$(GB_TILE_G)
ifneq ($(PE_NUM),4)
$(error group_norm_grad_1d supports only the dynamic 4PE testcase)
endif
START_FILE = $(ROOT)/test/common/_start_multipe.s
ifeq ($(TESTCASE), group_norm_grad_1d)
DEFINES += -DDType=$(DType) -DN_BATCH=$(N_BATCH) -DC_CH=$(C_CH)
DEFINES += -DG_GRP=$(G_GRP) -DPE_NUM=$(PE_NUM)
TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType)_N$(N_BATCH)_C$(C_CH)_G$(G_GRP)_PE$(PE_NUM).elf
endif
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
include ../../../common/Makefile.common
