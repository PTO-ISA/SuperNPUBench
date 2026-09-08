# MoE Combine — single-PE V2 and 4-PE SPMD.
# Single-PE testcase:  moe_combine_v2   (gfrun default)
# 4-PE testcase:       moe_combine_mt   (gfrun -s softcore.multiThreadNum=4)

ifeq ($(TESTCASE),moe_combine_v2)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
TARGET = $(ELF_HEAD)_$(TESTCASE).elf
else ifeq ($(TESTCASE),moe_combine_mt)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
TARGET = $(ELF_HEAD)_$(TESTCASE).elf
else
$(error Unsupported TESTCASE=$(TESTCASE); expected moe_combine_v2 or moe_combine_mt)
endif

include ../../common/Makefile.common

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)
