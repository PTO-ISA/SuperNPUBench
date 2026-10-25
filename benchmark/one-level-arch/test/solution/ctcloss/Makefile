DType ?= float
TESTCASE ?= ctcloss_v2_single_pe
res_check ?= on
ifeq ($(TESTCASE),ctcloss_v2_single_pe)
PE_NUM ?= 1
ifneq ($(PE_NUM),1)
$(error ctcloss_v2_single_pe requires PE_NUM=1)
endif
else ifeq ($(TESTCASE),ctcloss_v2_four_pe)
PE_NUM ?= 4
ifneq ($(PE_NUM),4)
$(error ctcloss_v2_four_pe requires PE_NUM=4)
endif
else
$(error Unsupported TESTCASE; use ctcloss_v2_single_pe or ctcloss_v2_four_pe)
endif
ifneq ($(DType),float)
$(error This network fixture requires DType=float)
endif
ifneq ($(res_check),on)
ifneq ($(res_check),off)
$(error res_check must be on or off)
endif
endif

ifndef API_INCLUDE
ifdef TILEOP_API_ROOT
API_INCLUDE := $(TILEOP_API_ROOT)/include
else
$(error Set API_INCLUDE to the public TileOP include directory)
endif
endif
LINX_RUNTIME_ROOT ?= $(abspath $(COMPILER_DIR)/..)
LINX_TARGET_FLAGS = --target=linx64v5-unknown-linux-musl --sysroot=$(LINX_RUNTIME_ROOT)/sysroot -resource-dir=$(LINX_RUNTIME_ROOT)/lib/clang/15.0.4
INCLUDE += -I$(API_INCLUDE)
DEFINES += -DDType=$(DType) -DPE_NUM=$(PE_NUM) -DSUPERNPUBENCH_LINX_TILE_SIZE -DLINX_ELEMENTWISE_AVAILABLE
# Inputs are also required for timing: zero lengths would change the workload.
DEFINES += -DENABLE_BINARY_OUTPUT
CC_OPTS += $(LINX_TARGET_FLAGS) -fno-vectorize -fno-slp-vectorize -fno-fast-math -gline-tables-only -Werror=unknown-pragmas -mllvm -linxv5-enable-generic-element=true
TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType)_T195_N256_D41_U74_PE$(PE_NUM)_rescheck$(res_check).elf
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
include ../../common/Makefile.common

# Both modes use the repository's hosted musl / worker entry, never a private
# direct-boot start.s. PE0 alone owns file I/O and process exit.
CXX = $(COMPILER_DIR)/clang++ $(LINX_TARGET_FLAGS)
CC_LINK = $(LINX_TARGET_FLAGS) -fuse-ld=lld -static
EXTRA_OBJ_FILES := $(sort $(EXTRA_OBJ_FILES) $(OBJ_ROOT)/$(COMM_SRC_DIR)/group_worker_runtime.o)
CHK_DIR ?= $(ROOT)/compare/$(notdir $(basename $(TARGET)))
ifneq ($(res_check),on)
DEFINES += -DCHK_DIR=\"$(CHK_DIR)\"
endif
$(OBJ): $(wildcard $(ROOT)/kernels/solution/ctcloss/*.hpp) $(CURDIR)/src/ctcloss_v2_test.hpp
