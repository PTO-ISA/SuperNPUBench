# DynamicMxQuant test entries — 精度看护第一批（batch1）。
# 本分支仅收 tail_cublas_fp8（8×32 基础场景）作为组件合入前的功能/精度看护基线；
# 其余 driver（tail/nontail 各 dtype、各 probe）待后续批次逐个迁入并对齐 AscendC 后再加 TYPE 块。

ifeq ($(TYPE), TAIL_CUBLAS_FP8)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tail_cublas_fp8.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tail_cublas_fp8.elf
endif

include ../../../common/Makefile.common

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)
