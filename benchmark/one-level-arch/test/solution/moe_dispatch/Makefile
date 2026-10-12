# MoE Dispatch — single-PE V2, 4-PE SPMD and 4-PE dynamic-shape SPMD.
# Single-PE testcase:  moe_dispatch_v2      (gfrun default)
# 4-PE testcase:       moe_dispatch_mt      (gfrun -s softcore.multiThreadNum=4)
# 4-PE dyn testcase:   moe_dispatch_mt_dyn  (gfrun -s softcore.multiThreadNum=4;
#                       runtime tiling = {BS,H,K,MoeExpertNum}, 2 shape configs)

ifeq ($(TESTCASE),moe_dispatch_v2)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
TARGET = $(ELF_HEAD)_$(TESTCASE).elf
else ifeq ($(TESTCASE),moe_dispatch_mt)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
TARGET = $(ELF_HEAD)_$(TESTCASE).elf
else ifeq ($(TESTCASE),moe_dispatch_mt_dyn)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
TARGET = $(ELF_HEAD)_$(TESTCASE).elf
else
$(error Unsupported TESTCASE=$(TESTCASE); expected moe_dispatch_v2, moe_dispatch_mt or moe_dispatch_mt_dyn)
endif

include ../../common/Makefile.common

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)
