ifeq ($(TESTCASE), matop)
dtype = $(shell echo $(DTYPE) | tr 'A-Z' 'a-z')
DEFINES += -DTEST_OP=\"$(TEST_OP)\" -DDTYPE=$(DTYPE) -DtM=$(M) -DtN=$(N) -DtK=$(K)
TARGET = $(ELF_HEAD)_$(TESTCASE)_$(TEST_OP)_$(dtype)_M$(M)_N$(N)_K$(K).elf
endif

SRC_FILE +=  $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
include ../../common/Makefile.common
