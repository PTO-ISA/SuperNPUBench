DType ?= __half
G_A ?= 512
G_R ?= 512
PE_NUM ?= 4
ifneq ($(PE_NUM),4)
$(error rms_norm supports only the dynamic 4PE testcase)
endif
START_FILE = $(ROOT)/test/common/_start_multipe.s
ifeq ($(TESTCASE), rms_norm)
DEFINES += -DDType=$(DType) -DPE_NUM=$(PE_NUM) -DG_A=$(G_A) -DG_R=$(G_R)
TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType)_gA$(G_A)_gR$(G_R)_PE$(PE_NUM).elf
endif
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
include ../../../common/Makefile.common
