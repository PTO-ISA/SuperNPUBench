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

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)
