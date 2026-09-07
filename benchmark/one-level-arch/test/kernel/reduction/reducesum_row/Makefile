ifeq ($(TESTCASE), rowsum_subview)
ROWSUM_ROWS ?= 32
ROWSUM_COLS ?= 32
ROWSUM_PARTS ?= 4
DEFINES += -DROWSUM_ROWS=$(ROWSUM_ROWS) -DROWSUM_COLS=$(ROWSUM_COLS) \
           -DROWSUM_PARTS=$(ROWSUM_PARTS)
TARGET = $(ELF_HEAD)_$(TESTCASE)_Rows$(ROWSUM_ROWS)_Cols$(ROWSUM_COLS)_Parts$(ROWSUM_PARTS).elf
else ifeq ($(TESTCASE), reducesum_row)
DEFINES += -DRDType=$(DType) -DgIMs=$(gM_s) -DgINs=$(gN_s) -DtMs=$(tM_s) -DtNs=$(tN_s)
TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType)_tM$(tM_s)_tN$(tN_s)_GM$(gM_s)_GN$(gN_s).elf
endif


#TARGET = $(ELF_HEAD)_$(TESTCASE).elf
SRC_FILE +=  $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
include ../../../common/Makefile.common
