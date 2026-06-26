ifeq ($(TESTCASE), reducemax_3dcol)
DEFINES += -DDType=$(DType) -DNums=$(Num) -DgIMs=$(gM_s) -DgINs=$(gN_s) -DtMs=$(tM_s) -DtNs=$(tN_s) -DtM_VLDs=$(tM_VLDs)
TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType)_tM$(tM_s)_tN$(tN_s)_tMVLD$(tM_VLDs)_Num$(Num)_GM$(gM_s)_GN$(gN_s).elf
endif


SRC_FILE +=  $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
include ../../../common/Makefile.common
