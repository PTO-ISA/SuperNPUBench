ifeq ($(TESTCASE), lat_bw)
TARGET = $(ELF_HEAD)_$(TESTCASE)_$(TEST_OP)_chain_$(CHAIN_NUM).elf
DEFINES += -DCHAIN_NUM=$(CHAIN_NUM) -D$(TEST_OP)
	ifeq ($(mt), on)
        DEFINES += -DMT
        TARGET = $(ELF_HEAD)_$(TESTCASE)_$(TEST_OP)_mt_chain_$(CHAIN_NUM).elf
	endif
endif

SRC_FILE +=  $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp

include ../common/Makefile.common