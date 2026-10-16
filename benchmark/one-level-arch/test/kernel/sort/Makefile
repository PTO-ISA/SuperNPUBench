TESTCASE ?= element_atomic_topk
ifneq ($(TESTCASE),element_atomic_topk)
$(error This directory builds the element_atomic_topk example; tiled TopK lives in test/kernel/multi_thread/topk)
endif

ifeq ($(TESTCASE), element_atomic_topk)
TARGET = $(ELF_DIR)/element_atomic_topk.elf
SRC_FILE += $(TEST_ROOT)/$(CATEGORY)/src/element_atomic_topk.cpp
DEFINES += -DSUPERNPUBENCH_LINX_TILE_SIZE -DLINX_ELEMENTWISE_AVAILABLE
TARGET_TRIPLE ?= linx64v5-unknown-linux-musl
ifndef SYSROOT
$(error SYSROOT is required for element_atomic_topk)
endif
ifndef RESOURCE_DIR
$(error RESOURCE_DIR is required for element_atomic_topk)
endif
ifndef API_INCLUDE
$(error API_INCLUDE is required for element_atomic_topk)
endif
TLEA_TARGET_FLAGS := --target=$(TARGET_TRIPLE) --sysroot=$(SYSROOT) \
	-resource-dir=$(RESOURCE_DIR)
CC_OPTS += $(TLEA_TARGET_FLAGS)
# Keep ordinary setup/verification loops scalar.  The requested Tile work is
# explicit in the kernel; generic loop vectorization otherwise creates TMOV
# carrier blocks that obscure the TLEA/atomic path in both models.
CC_OPTS += -fno-vectorize -fno-slp-vectorize -Werror=unknown-pragmas
CC_OPTS += -mllvm -linxv5-enable-generic-element=true
CC_LINK += $(TLEA_TARGET_FLAGS) -fuse-ld=lld
INCLUDE += -I$(API_INCLUDE)
endif
include ../../common/Makefile.common

ir: pre_work
	@test -n "$(IR_TARGET)" || \
		(echo "IR_TARGET is required for the ir target" >&2; exit 1)
	$(CXX) $(CXX_O_ALL) $(INCLUDE) $(DEFINES) -S -emit-llvm \
		$(SRC_FILE) -o $(IR_TARGET)

.PHONY: ir
