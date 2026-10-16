ifndef TESTCASE
$(error TESTCASE is required)
endif

SUPPORTED_CASES := gather_v2_f32_0_original gather_v2_s32_1_original gather_v2_s32_1_alias_probe gather_v2_f16_2_original view_copy_s32_0_original view_copy_s32_0_alias_probe view_copy_f16_1_original view_copy_f16_2_tail gather_f32_3_index_alias_probe broadcast_inline_s32_original broadcast_inline_s32_alias_probe \
	gather_f32_0_original gather_f32_0_alias_probe gather_f32_1_original gather_f32_1_alias_probe gather_f32_2_original gather_f32_2_alias_probe gather_f32_3_original gather_f32_3_alias_probe \
	transpose_f16_0_original transpose_f16_0_alias_probe transpose_f16_1_original transpose_f16_1_alias_probe transpose_f16_2_original transpose_f16_2_alias_probe transpose_f16_3_original transpose_f16_3_alias_probe \
	broadcast_f16_2d_original broadcast_f16_2d_alias_probe broadcast_f16_3d_original broadcast_f16_3d_alias_probe broadcast_f16_5d_original broadcast_f16_5d_alias_probe \
	histogram_tile_element selected_radix_tile_element \
	element_expression_chain signed_element_expression indexed_gather_tile_element \
	generic_predicated_cfg_i32 generic_typed_tile_cfg_i32 generic_atomic_cfg_i32 elementwise_atomic_histogram \
	concat_gather_s32_original concat_gather_s32_alias_probe \
	broadcast_fourpe gather_fourpe concat_gather_fourpe concat_scatter_fourpe concat_scatter_s32_original concat_scatter_s32_alias_probe \
	concat_scatter_f16_original concat_scatter_f16_alias_probe \
	concat_gather_f16_original concat_gather_f16_alias_probe \
	group_count_s32_original group_count_s32_strong \
	elementwise_if_for elementwise_if_for_strong \
	elementwise_masked_gather_scatter elementwise_masked_gather_scatter_strong \
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

# Several fixtures intentionally share one source file with different compile
# definitions. Keep their objects in testcase-specific directories so ordinary
# sequential make invocations cannot reuse an object from another profile.
override CASE_SRC_DIR = $(CATEGORY)/src/$(TESTCASE)

TARGET_TRIPLE ?= linx64v5-unknown-linux-musl
TARGET_FLAGS := --target=$(TARGET_TRIPLE) --sysroot=$(SYSROOT) \
	-resource-dir=$(RESOURCE_DIR)

SRC_FILE := $(CURDIR)/$(TESTCASE).cpp
ifneq ($(filter gather_v2_%,$(TESTCASE)),)
SRC_FILE := $(CURDIR)/gather_v2_original.cpp
CC_OPTS += -mllvm -linxv5-enable-generic-element=true
endif
ifneq ($(filter gather_v2_s32_1_%,$(TESTCASE)),)
DEFINES += -DGATHER_V2_PROFILE=1
endif
ifeq ($(TESTCASE),gather_v2_s32_1_alias_probe)
DEFINES += -DGATHER_V2_ALIAS_PROBE=1
endif
ifeq ($(TESTCASE),gather_v2_f16_2_original)
DEFINES += -DGATHER_V2_PROFILE=2
endif
ifneq ($(filter view_copy_%,$(TESTCASE)),)
SRC_FILE := $(CURDIR)/view_copy_original.cpp
CC_OPTS += -mllvm -linxv5-enable-generic-element=true
endif
ifeq ($(TESTCASE),view_copy_s32_0_alias_probe)
DEFINES += -DVIEW_COPY_ALIAS_PROBE=1
endif
ifeq ($(TESTCASE),view_copy_f16_1_original)
DEFINES += -DVIEW_COPY_PROFILE=1
endif
ifeq ($(TESTCASE),view_copy_f16_2_tail)
DEFINES += -DVIEW_COPY_PROFILE=2
endif
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
ifneq ($(filter concat_scatter_s32_original concat_scatter_s32_alias_probe concat_scatter_f16_original concat_scatter_f16_alias_probe,$(TESTCASE)),)
SRC_FILE := $(CURDIR)/concat_scatter_original.cpp
endif
ifneq ($(filter concat_scatter_f16_%,$(TESTCASE)),)
DEFINES += -DCONCAT_SCATTER_FP16=1
endif
ifneq ($(filter concat_scatter_s32_alias_probe concat_scatter_f16_alias_probe,$(TESTCASE)),)
DEFINES += -DCONCAT_SCATTER_ALIAS_PROBE=1
endif
ifneq ($(filter broadcast_f16_%,$(TESTCASE)),)
SRC_FILE := $(CURDIR)/broadcast_original.cpp
CC_OPTS += -mllvm -linxv5-enable-generic-element=true
endif
ifneq ($(filter broadcast_f16_3d_%,$(TESTCASE)),)
DEFINES += -DBROADCAST_RANK=3
endif
ifneq ($(filter broadcast_f16_5d_%,$(TESTCASE)),)
DEFINES += -DBROADCAST_RANK=5
endif
ifneq ($(filter broadcast_f16_2d_alias_probe broadcast_f16_3d_alias_probe broadcast_f16_5d_alias_probe,$(TESTCASE)),)
DEFINES += -DBROADCAST_ALIAS_PROBE=1
endif
ifneq ($(filter transpose_f16_%,$(TESTCASE)),)
SRC_FILE := $(CURDIR)/transpose_original.cpp
CC_OPTS += -mllvm -linxv5-enable-generic-element=true
endif
ifneq ($(filter transpose_f16_1_%,$(TESTCASE)),)
DEFINES += -DTRANSPOSE_PROFILE=1
endif
ifneq ($(filter transpose_f16_2_%,$(TESTCASE)),)
DEFINES += -DTRANSPOSE_PROFILE=2
endif
ifneq ($(filter transpose_f16_3_%,$(TESTCASE)),)
DEFINES += -DTRANSPOSE_PROFILE=3
endif
ifneq ($(filter transpose_f16_0_alias_probe transpose_f16_1_alias_probe transpose_f16_2_alias_probe transpose_f16_3_alias_probe,$(TESTCASE)),)
DEFINES += -DTRANSPOSE_ALIAS_PROBE=1
endif
ifneq ($(filter gather_f32_%,$(TESTCASE)),)
SRC_FILE := $(CURDIR)/gather_original.cpp
CC_OPTS += -mllvm -linxv5-enable-generic-element=true
endif
ifneq ($(filter gather_f32_1_%,$(TESTCASE)),)
DEFINES += -DGATHER_PROFILE=1
endif
ifneq ($(filter gather_f32_2_%,$(TESTCASE)),)
DEFINES += -DGATHER_PROFILE=2
endif
ifeq ($(TESTCASE),gather_f32_3_index_alias_probe)
DEFINES += -DGATHER_INDEX_ALIAS_PROBE=1
CC_OPTS += -fno-strict-aliasing
endif
ifneq ($(filter gather_f32_3_%,$(TESTCASE)),)
DEFINES += -DGATHER_PROFILE=3
endif
ifneq ($(filter gather_f32_0_alias_probe gather_f32_1_alias_probe gather_f32_2_alias_probe gather_f32_3_alias_probe,$(TESTCASE)),)
DEFINES += -DGATHER_ALIAS_PROBE=1
endif
ifneq ($(filter broadcast_inline_s32_%,$(TESTCASE)),)
SRC_FILE := $(CURDIR)/broadcast_original.cpp
DEFINES += -DBROADCAST_INLINE_I32=1 -DBROADCAST_RANK=3
CC_OPTS += -mllvm -linxv5-enable-generic-element=true
endif
ifeq ($(TESTCASE),broadcast_inline_s32_alias_probe)
DEFINES += -DBROADCAST_ALIAS_PROBE=1
endif
ifeq ($(TESTCASE),concat_gather_s32_alias_probe)
SRC_FILE := $(CURDIR)/concat_gather_s32_original.cpp
DEFINES += -DCONCAT_GATHER_S32_ALIAS_PROBE=1
endif
ifeq ($(TESTCASE),concat_gather_f16_alias_probe)
SRC_FILE := $(CURDIR)/concat_gather_f16_original.cpp
DEFINES += -DCONCAT_GATHER_F16_ALIAS_PROBE=1
endif
ifeq ($(TESTCASE),elementwise_atomic_histogram)
SRC_FILE := $(CURDIR)/../../../solution/tileop-test/vec/src/elementwise_atomic_histogram.cpp
endif
ifeq ($(TESTCASE),group_count_s32_strong)
SRC_FILE := $(CURDIR)/group_count_s32_original.cpp
DEFINES += -DGROUP_COUNT_S32_STRONG_CASE=1
endif
ifneq ($(filter elementwise_if_for elementwise_if_for_strong,$(TESTCASE)),)
SRC_FILE := $(CURDIR)/../../../solution/tileop-test/vec/src/elementwise_if_for.cpp
endif
ifeq ($(TESTCASE),elementwise_if_for_strong)
DEFINES += -DELEMENTWISE_IF_FOR_STRONG_CASE=1
endif
ifneq ($(filter elementwise_masked_gather_scatter elementwise_masked_gather_scatter_strong,$(TESTCASE)),)
SRC_FILE := $(CURDIR)/../../../solution/tileop-test/vec/src/elementwise_masked_gather_scatter.cpp
endif
ifeq ($(TESTCASE),elementwise_masked_gather_scatter_strong)
DEFINES += -DELEMENTWISE_MASKED_GATHER_SCATTER_STRONG_CASE=1
endif
TARGET = $(ELF_DIR)/$(TESTCASE).elf
DEFINES += -DSUPERNPUBENCH_LINX_TILE_SIZE -DLINX_ELEMENTWISE_AVAILABLE
CC_OPTS += $(TARGET_FLAGS) -fno-vectorize -fno-slp-vectorize \
	-Werror=unknown-pragmas
ifneq ($(filter element_expression_chain signed_element_expression \
	indexed_gather_tile_element histogram_tile_element \
	selected_radix_tile_element generic_predicated_cfg_i32 \
	generic_typed_tile_cfg_i32 generic_atomic_cfg_i32 elementwise_atomic_histogram \
	concat_gather_s32_original concat_gather_s32_alias_probe \
	broadcast_fourpe gather_fourpe concat_gather_fourpe concat_scatter_fourpe concat_scatter_s32_original concat_scatter_s32_alias_probe \
	concat_scatter_f16_original concat_scatter_f16_alias_probe \
	concat_gather_f16_original concat_gather_f16_alias_probe topk_boundaries \
	group_count_s32_original group_count_s32_strong \
	elementwise_if_for elementwise_if_for_strong \
	elementwise_masked_gather_scatter elementwise_masked_gather_scatter_strong \
	topk_boundaries_0 topk_boundaries_1 topk_boundaries_2,$(TESTCASE)),)
CC_OPTS += -mllvm -linxv5-enable-generic-element=true
endif
CC_LINK += $(TARGET_FLAGS) -fuse-ld=lld
ifneq ($(filter gather_f32_3_%,$(TESTCASE)),)
# Preserve the complete 128MiB input while leaving the fixed low stack clear
# of UART/finisher MMIO. Link this observation image above the MMIO window.
CC_LINK += -Wl,--image-base=0x20000000
endif
INCLUDE += -I$(API_INCLUDE)
ifneq ($(filter concat_gather_f16_original concat_gather_f16_alias_probe,$(TESTCASE)),)
ifneq ($(ELEMENT_FIXTURE_DIR),)
DEFINES += -DCONCAT_GATHER_F16_LINKED_INPUTS=1
INCLUDE += -I$(ELEMENT_FIXTURE_DIR)
endif
endif
ifneq ($(filter concat_scatter_f16_original concat_scatter_f16_alias_probe,$(TESTCASE)),)
ifneq ($(ELEMENT_FIXTURE_DIR),)
DEFINES += -DCONCAT_SCATTER_F16_LINKED_INPUTS=1
INCLUDE += -I$(ELEMENT_FIXTURE_DIR)
endif
endif

include ../../../common/Makefile.common

ir: pre_work
	@test -n "$(IR_TARGET)" || \
		(echo "IR_TARGET is required for the ir target" >&2; exit 1)
	$(CXX) $(CXX_O_ALL) $(INCLUDE) $(DEFINES) -S -emit-llvm \
		$(SRC_FILE) -o $(IR_TARGET)

.PHONY: ir
