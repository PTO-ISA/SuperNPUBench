TARGET = $(ELF_HEAD)_$(TESTCASE).elf
SRC_FILE +=  $(TEST_ROOT)/$(CATEGORY)/$(TESTCASE)/$(TESTCASE).cpp

ifeq ($(opt), on)
DEFINES += -DOPT
TARGET = $(ELF_HEAD)_$(TESTCASE)_OPT.elf
endif

include ../../common/Makefile.common
