N_BATCH ?= 32
C_CH ?= 16
G_GRP ?= 8
HxW_SZ ?= 8192
DType ?= __half
PE_NUM ?= 4
ifneq ($(PE_NUM),4)
$(error group_norm_grad supports only the dynamic 4PE testcase)
endif
START_FILE = $(ROOT)/test/common/_start_multipe.s
ifeq ($(TESTCASE), group_norm_grad)
DEFINES += -DDType=$(DType) -DN_BATCH=$(N_BATCH) -DC_CH=$(C_CH)
DEFINES += -DG_GRP=$(G_GRP) -DHxW_SZ=$(HxW_SZ) -DPE_NUM=$(PE_NUM)
TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType)_N$(N_BATCH)_C$(C_CH)_G$(G_GRP)_HxW$(HxW_SZ)_PE$(PE_NUM).elf
endif
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
include ../../../common/Makefile.common
