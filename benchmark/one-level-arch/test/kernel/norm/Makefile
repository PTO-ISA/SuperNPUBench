ifeq ($(TESTCASE),rms_norm)
DEFINES += -DRMS_M=$(M) -DRMS_N=$(N) -DRMS_TM=$(tM) -DRMS_TN=$(tN)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/rms_norm.cpp
TARGET = $(ELF_HEAD)_$(TESTCASE)_M$(M)_N$(N)_tM$(tM)_tN$(tN).elf
else
$(error Unsupported TESTCASE=$(TESTCASE); expected rms_norm)
endif

include ../../common/Makefile.common
