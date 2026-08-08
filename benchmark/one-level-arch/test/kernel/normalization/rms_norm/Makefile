ifeq ($(TESTCASE), rms_norm)
DEFINES += -DDType=$(DType)
TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType).elf
endif

SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
include ../../../common/Makefile.common
