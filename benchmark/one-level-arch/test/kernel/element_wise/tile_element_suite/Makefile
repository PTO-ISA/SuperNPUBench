ifndef TESTCASE
$(error TESTCASE is required)
endif

SUPPORTED_CASES := histogram_tile_element selected_radix_tile_element \
	element_expression_chain signed_element_expression indexed_gather_tile_element \
	generic_predicated_cfg_i32 generic_typed_tile_cfg_i32 generic_atomic_cfg_i32 \
	topk_boundaries topk_boundaries_0 \
	topk_boundaries_1 topk_boundaries_2
ifeq ($(filter $(TESTCASE),$(SUPPORTED_CASES)),)
$(error unsupported TESTCASE=$(TESTCASE); choose $(SUPPORTED_CASES))
endif

ifndef SYSROOT
$(error SYSROOT is required)
endif
ifndef RESOURCE_DIR
$(error RESOURCE_DIR is required)
endif
ifndef API_INCLUDE
$(error API_INCLUDE is required)
endif

TARGET_TRIPLE ?= linx64v5-unknown-linux-musl
TARGET_FLAGS := --target=$(TARGET_TRIPLE) --sysroot=$(SYSROOT) \
	-resource-dir=$(RESOURCE_DIR)

SRC_FILE := $(CURDIR)/$(TESTCASE).cpp
ifneq ($(filter topk_boundaries_%,$(TESTCASE)),)
SRC_FILE := $(CURDIR)/topk_boundaries.cpp
endif
ifeq ($(TESTCASE),topk_boundaries_0)
DEFINES += -DTOPK_BOUNDARY_FIRST=0 -DTOPK_BOUNDARY_LAST=6
endif
ifeq ($(TESTCASE),topk_boundaries_1)
DEFINES += -DTOPK_BOUNDARY_FIRST=6 -DTOPK_BOUNDARY_LAST=13
endif
ifeq ($(TESTCASE),topk_boundaries_2)
DEFINES += -DTOPK_BOUNDARY_FIRST=13 -DTOPK_BOUNDARY_LAST=17
endif
ifeq ($(TESTCASE),topk_boundaries)
DEFINES += -DTOPK_BOUNDARY_SHARED_WORKSPACE=1
endif
TARGET = $(ELF_DIR)/$(TESTCASE).elf
DEFINES += -DSUPERNPUBENCH_LINX_TILE_SIZE -DLINX_ELEMENTWISE_AVAILABLE
CC_OPTS += $(TARGET_FLAGS) -fno-vectorize -fno-slp-vectorize \
	-Werror=unknown-pragmas
ifneq ($(filter element_expression_chain signed_element_expression \
	indexed_gather_tile_element histogram_tile_element \
	selected_radix_tile_element generic_predicated_cfg_i32 \
	generic_typed_tile_cfg_i32 generic_atomic_cfg_i32 topk_boundaries \
	topk_boundaries_0 topk_boundaries_1 topk_boundaries_2,$(TESTCASE)),)
CC_OPTS += -mllvm -linxv5-enable-generic-element=true
endif
CC_LINK += $(TARGET_FLAGS) -fuse-ld=lld
INCLUDE += -I$(API_INCLUDE)

include ../../../common/Makefile.common

ir: pre_work
	@test -n "$(IR_TARGET)" || \
		(echo "IR_TARGET is required for the ir target" >&2; exit 1)
	$(CXX) $(CXX_O_ALL) $(INCLUDE) $(DEFINES) -S -emit-llvm \
		$(SRC_FILE) -o $(IR_TARGET)

.PHONY: ir
