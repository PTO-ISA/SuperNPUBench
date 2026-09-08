# Group Token Old (three-phase MoE token grouping) — single-PE and 4-PE SPMD.
# Single-PE testcase:  group_token_old       (gfrun default)
# 4-PE testcase:       group_token_old_mt    (gfrun -s softcore.multiThreadNum=4)

ifeq ($(TESTCASE),group_token_old)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
TARGET = $(ELF_HEAD)_$(TESTCASE).elf
else ifeq ($(TESTCASE),group_token_old_mt)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
TARGET = $(ELF_HEAD)_$(TESTCASE).elf
else
$(error Unsupported TESTCASE=$(TESTCASE); expected group_token_old or group_token_old_mt)
endif

include ../../common/Makefile.common

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)
