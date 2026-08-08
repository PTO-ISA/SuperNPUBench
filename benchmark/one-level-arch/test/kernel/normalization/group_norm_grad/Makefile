N_BATCH ?= 2
C_CH ?= 16
G_GRP ?= 4
HxW_SZ ?= 16
TILE_HW ?= 8
DType ?= __half

ifeq ($(TESTCASE), group_norm_grad)
DEFINES += -DDType=$(DType)
DEFINES += -DN_BATCH=$(N_BATCH)
DEFINES += -DC_CH=$(C_CH)
DEFINES += -DG_GRP=$(G_GRP)
DEFINES += -DHxW_SZ=$(HxW_SZ)
DEFINES += -DTILE_HW=$(TILE_HW)
TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType)_N$(N_BATCH)_C$(C_CH)_G$(G_GRP)_HxW$(HxW_SZ).elf
endif

SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
include ../../../common/Makefile.common
