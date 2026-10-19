# Group Token Old (three-phase MoE token grouping) — single-PE and 4-PE SPMD.
# Single-PE testcase:  group_token_old       (gfrun default)
# 4-PE testcase:       group_token_old_mt    (gfrun -s softcore.multiThreadNum=4)

ifeq ($(TESTCASE),group_token_old)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
TARGET = $(ELF_HEAD)_$(TESTCASE).elf
else ifeq ($(TESTCASE),group_token_old_mt)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
TARGET = $(ELF_HEAD)_$(TESTCASE).elf
else ifeq ($(TESTCASE),group_token_histogram_element)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
TARGET = $(ELF_HEAD)_$(TESTCASE).elf
else
$(error Unsupported TESTCASE=$(TESTCASE); expected group_token_old, group_token_old_mt, or group_token_histogram_element)
endif

include ../../common/Makefile.common

ifeq ($(PLAT),linx)
ELEMENT_TARGET_FLAGS = $(if $(TARGET_TRIPLE),--target=$(TARGET_TRIPLE)) \
	$(if $(SYSROOT),--sysroot=$(SYSROOT)) \
	$(if $(RESOURCE_DIR),-resource-dir=$(RESOURCE_DIR))
INCLUDE += $(if $(API_INCLUDE),-I$(API_INCLUDE))
CC_OPTS += $(ELEMENT_TARGET_FLAGS) -mllvm -linxv5-enable-generic-element=true \
	-fno-vectorize -fno-slp-vectorize -Werror=unknown-pragmas
CC_LINK += $(ELEMENT_TARGET_FLAGS) -fuse-ld=lld
endif

ir: pre_work
	@test -n "$(IR_TARGET)" || \
		(echo "IR_TARGET is required for the ir target" >&2; exit 1)
	$(CXX) $(CXX_O_ALL) $(INCLUDE) $(DEFINES) -S -emit-llvm \
		$(SRC_FILE) -o $(IR_TARGET)

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)

.PHONY: ir
