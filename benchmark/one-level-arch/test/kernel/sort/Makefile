TARGET = $(ELF_HEAD)_$(TESTCASE).elf

# Override target name for topk
ifeq ($(TESTCASE), topk)
TARGET = $(ELF_DIR)/topk.elf
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
ifneq ($(TESTCASE), element_atomic_topk)
SRC_FILE +=  $(TEST_ROOT)/$(CATEGORY)/$(TESTCASE)/$(TESTCASE).cpp
endif

# Special handling for topk - embed data as object files
EXTRA_OBJ_FILES :=
EXTRA_OBJ_DEPS :=
DATA_OBJ_DIR := topk/data_obj
OUTPUT_DATA_OBJ_DIR := ../../../output/kernel/sort/topk/data_obj

# Generate .data files from Python script if missing
gen_topk_data:
	@if [ ! -f $(DATA_OBJ_DIR)/input_131072.data ] || [ ! -f $(DATA_OBJ_DIR)/top_2048_out.data ]; then \
		echo "Generating topk data..."; \
		cd topk && python3 gen_topk_data.py; \
	fi

ifeq ($(TESTCASE), topk)
EXTRA_OBJ_FILES += $(OUTPUT_DATA_OBJ_DIR)/input_131072.o
EXTRA_OBJ_FILES += $(OUTPUT_DATA_OBJ_DIR)/top_2048_out.o

pre_work: build_data_objs

build_data_objs: gen_topk_data
	@COMPILER_DIR="$(COMPILER_DIR)" $(DATA_OBJ_DIR)/build_data_obj.sh $(DATA_OBJ_DIR) $(OUTPUT_DATA_OBJ_DIR)

$(OUTPUT_DATA_OBJ_DIR)/%.o: $(DATA_OBJ_DIR)/%.s pre_work
	@mkdir -p $(shell dirname $@)
	$(AS) $(CC_O_ALL) $(INCLUDE) $(DEFINES) $< -o $@
endif

ifeq ($(opt), on)
DEFINES += -DOPT
TARGET = $(ELF_HEAD)_$(TESTCASE)_OPT.elf
endif

include ../../common/Makefile.common

ir: pre_work
	@test -n "$(IR_TARGET)" || \
		(echo "IR_TARGET is required for the ir target" >&2; exit 1)
	$(CXX) $(CXX_O_ALL) $(INCLUDE) $(DEFINES) -S -emit-llvm \
		$(SRC_FILE) -o $(IR_TARGET)

.PHONY: ir
