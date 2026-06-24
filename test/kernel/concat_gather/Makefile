ifeq ($(TESTCASE), concat_gather)
DEFINES += -DDType=$(DType) -DMAX_DIMs=$(MAX_DIM) -DtMs=$(tM) -DIN_SHAPEs=$(IN_SHAPE) -DOUT_SHAPEs=$(OUT_SHAPE) -DDATA_DIMs=$(DATA_DIM) -DCONCAT_DIMs=$(CONCAT_DIM)  -DgIMs=$(gIM) -DgOMs=$(gOM) 
TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType)_tM$(tM)_IN_SHAPE$(IN_SHAPE_NAME)_OUT_SHAPE$(OUT_SHAPE_NAME).elf
endif


#TARGET = $(ELF_HEAD)_$(TESTCASE).elf
SRC_FILE +=  $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
include ../../../common/Makefile.common
