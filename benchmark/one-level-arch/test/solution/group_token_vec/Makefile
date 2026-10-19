# Group Token Vec (vectorized MoE token grouping) — single-PE, 4-PE SPMD and
# 4-PE dynamic-shape SPMD.
# Single-PE testcase:  group_token_vec         (gfrun default)
# 4-PE testcase:       group_token_vec_mt      (gfrun -s softcore.multiThreadNum=4)
# 4-PE dyn testcase:   group_token_vec_mt_dyn  (gfrun -s softcore.multiThreadNum=4;
#                       runtime tiling = {bs,topK,expertPerRank,expertPerPod,
#                       superPodNum}, 2 shape configs)

ifeq ($(TESTCASE),group_token_vec)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
TARGET = $(ELF_HEAD)_$(TESTCASE).elf
else ifeq ($(TESTCASE),group_token_vec_mt)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
TARGET = $(ELF_HEAD)_$(TESTCASE).elf
else ifeq ($(TESTCASE),group_token_vec_mt_dyn)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
TARGET = $(ELF_HEAD)_$(TESTCASE).elf
else
$(error Unsupported TESTCASE=$(TESTCASE); expected group_token_vec, group_token_vec_mt or group_token_vec_mt_dyn)
endif

include ../../common/Makefile.common

ifeq ($(PLAT),linx)
ELEMENT_TARGET_FLAGS = $(if $(TARGET_TRIPLE),--target=$(TARGET_TRIPLE)) \
	$(if $(SYSROOT),--sysroot=$(SYSROOT)) \
	$(if $(RESOURCE_DIR),-resource-dir=$(RESOURCE_DIR))
INCLUDE += $(if $(API_INCLUDE),-I$(API_INCLUDE))
CC_OPTS += $(ELEMENT_TARGET_FLAGS)
CC_LINK += $(ELEMENT_TARGET_FLAGS) -fuse-ld=lld

ifneq ($(filter group_token_vec_mt_dyn,$(TESTCASE)),)
CC_OPTS += -mllvm -linxv5-enable-generic-element=true \
	-fno-vectorize -fno-slp-vectorize -Werror=unknown-pragmas

histogram-ir: pre_work
	@test -n "$(IR_TARGET)" || \
		(echo "IR_TARGET is required for histogram-ir" >&2; exit 1)
	$(CXX) $(CXX_O_ALL) $(INCLUDE) $(DEFINES) \
		-mllvm -linxv5-enable-generic-element=true \
		-fno-vectorize -fno-slp-vectorize -Werror=unknown-pragmas \
		-S -emit-llvm $(SRC_FILE) -o $(IR_TARGET)
endif
endif

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)

.PHONY: histogram-ir
