TESTCASE ?= concat_gather

ifeq ($(TESTCASE),concat_gather)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/concat_gather.cpp
else ifeq ($(TESTCASE),concat_scatter)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/concat_scatter.cpp
else ifeq ($(TESTCASE),concat_two)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/concat_two.cpp
OUTER ?= 4
LEFT ?= 128
RIGHT ?= 128
INNER ?= 1
TILE ?= 128
DTYPE ?= float
DEFINES += -DCONCAT_OUTER=$(OUTER) -DCONCAT_LEFT=$(LEFT) -DCONCAT_RIGHT=$(RIGHT)
DEFINES += -DCONCAT_INNER=$(INNER) -DCONCAT_TILE=$(TILE) -DDTYPE=$(DTYPE)
else
$(error Unsupported TESTCASE=$(TESTCASE); expected concat_gather, concat_scatter or concat_two)
endif

ifeq ($(TESTCASE),concat_two)
TARGET = $(ELF_HEAD)_concat_two_O$(OUTER)_L$(LEFT)_R$(RIGHT)_I$(INNER)_T$(TILE)_$(DTYPE)_PE4.elf
else
TARGET = $(ELF_HEAD)_$(TESTCASE)_PE4.elf
endif
include ../../common/Makefile.common

.PHONY: check-concat-two
check-concat-two:
	python3 src/check_concat_two.py --compiler-dir "$(COMPILER_DIR)" $(if $(GFRUN),--gfrun "$(GFRUN)")
