N_BATCH ?= 8
C_CH ?= 64
G_GRP ?= 8
TILE_D ?= -1
DType ?= __half

ifeq ($(TESTCASE), group_norm_grad_1d)
DEFINES += -DDType=$(DType)
DEFINES += -DN_BATCH=$(N_BATCH)
DEFINES += -DC_CH=$(C_CH)
DEFINES += -DG_GRP=$(G_GRP)
DEFINES += -DTILE_D=$(TILE_D)
TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType)_N$(N_BATCH)_C$(C_CH)_G$(G_GRP).elf
endif

SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
include ../../../common/Makefile.common
