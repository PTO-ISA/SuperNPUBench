DType ?= __half
G_A ?= 16
G_R ?= 16384
PE_NUM ?= 4
ifneq ($(PE_NUM),4)
$(error rms_norm_binary supports only the dynamic 4PE testcase)
endif
START_FILE = $(ROOT)/test/common/_start_multipe.s
ifeq ($(TESTCASE), rms_norm_binary)
DEFINES += -DDType=$(DType) -DG_A=$(G_A) -DG_R=$(G_R) -DPE_NUM=$(PE_NUM)
TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType)_gA$(G_A)_gR$(G_R)_PE$(PE_NUM).elf
endif
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
include ../../../common/Makefile.common
