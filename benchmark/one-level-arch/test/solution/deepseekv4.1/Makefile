TESTCASE ?= decode
SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
TARGET = $(ELF_HEAD)_$(TESTCASE).elf
EXTRA_OBJ_FILES += $(OBJ_DIR)/input.o
ifneq ($(TILEOP_INCLUDE),)
INCLUDE += -I$(TILEOP_INCLUDE)
endif

include ../../common/Makefile.common

ifeq ($(PLAT),linx)
CC_LINK += -nostdlib++
endif

$(OBJ): $(wildcard $(ROOT)/kernels/deepseekv4.1/*.hpp) $(CURDIR)/src/fixture.hpp

$(OBJ_DIR)/input.bin: cases.py reference.py layout.py
	@mkdir -p $(@D)
	python3 cases.py $@

$(OBJ_DIR)/input.o: src/input.s $(OBJ_DIR)/input.bin
	$(AS) -mlxbc -c -I$(OBJ_DIR) $< -o $@
