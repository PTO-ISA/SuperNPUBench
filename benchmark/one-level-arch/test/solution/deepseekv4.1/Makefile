TESTCASE ?= decode
SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
TARGET = $(ELF_HEAD)_$(TESTCASE).elf
ifneq ($(TESTCASE),tile_edges)
EXTRA_OBJ_FILES += $(OBJ_DIR)/input.o
endif
ifneq ($(TILEOP_INCLUDE),)
INCLUDE += -I$(TILEOP_INCLUDE)
endif

include ../../common/Makefile.common

ifeq ($(PLAT),linx)
CC_LINK += -nostdlib++
# 0.59 objdump can decode the physical bundles for dual-scale GEMV.
DUMP += --disassembler-options=no-tile-macros
endif

$(OBJ): $(wildcard $(ROOT)/kernels/deepseekv4.1/*.hpp) $(CURDIR)/src/fixture.hpp
ifeq ($(TESTCASE),wse)
$(OBJ): $(wildcard $(ROOT)/kernels/wse/*.hpp)
endif

$(OBJ_DIR)/input.bin: cases.py reference.py layout.py
	@mkdir -p $(@D)
	python3 cases.py $@

$(OBJ_DIR)/input.o: src/input.s $(OBJ_DIR)/input.bin
	$(AS) -mlxbc -c -I$(OBJ_DIR) $< -o $@

# Optional resident-worker profile from SuperScalarModel #1243 and TileOP #315.
# Its runtime owns the SRAM service endpoint on PE1, with the layer on PE0.
ifneq ($(filter msgb,$(MAKECMDGOALS)),)
ifndef MSGB_RUNTIME
$(error Set MSGB_RUNTIME to the linx-runtime source directory)
endif
ifndef MSGB_INCLUDE
$(error Set MSGB_INCLUDE to the TileOP include directory containing common/pto_message.hpp)
endif
MSGB_DIR := $(OBJ_DIR)/msgb
MSGB_ELF := $(ELF_HEAD)_wse_msgb.elf
.PHONY: msgb check-msgb-worker
msgb: $(MSGB_ELF)

$(MSGB_DIR)/layer_msgb.o: src/wse.cpp src/fixture.hpp $(wildcard $(ROOT)/kernels/deepseekv4.1/*.hpp $(ROOT)/kernels/wse/*.hpp)
	@mkdir -p $(@D)
	$(CXX) $(CXX_O_ALL) $(INCLUDE) -I$(MSGB_INCLUDE) $(DEFINES) -DV41_MSGB_RUNTIME \
		-fno-exceptions -fno-rtti -MMD -MP -c $< -o $@
-include $(MSGB_DIR)/layer_msgb.d

$(MSGB_DIR)/lxmsgb_worker.o: check-msgb-worker
	$(MAKE) -C $(MSGB_RUNTIME) msgb-worker OUTPUT=$(MSGB_DIR) LINX_CXX=$(abspath $(CXX)) TILEOP_INCLUDE=$(abspath $(MSGB_INCLUDE))

$(MSGB_ELF): $(MSGB_DIR)/lxmsgb_worker.o $(MSGB_DIR)/layer_msgb.o $(OBJ_DIR)/input.o $(MSGB_RUNTIME)/src/v5/worker.ld
	@mkdir -p $(@D)
	$(LINK) -static -nostartfiles -nostdlib++ -Wl,-T,$(MSGB_RUNTIME)/src/v5/worker.ld \
		$(filter %.o,$^) -lm -o $@
endif
