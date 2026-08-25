# DynamicMxQuant test entries — 组件合入前的功能/精度看护基线。
# 逐批迁入已对齐 AscendC 的 driver：tail_cublas_fp8、tail_ocp_fp4、nontail_cublas_fp8。
# 其余 driver（各 dtype、各 probe）待后续批次迁入后再加对应 TYPE 块。

ifeq ($(TYPE), TAIL_CUBLAS_FP8)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tail_cublas_fp8.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tail_cublas_fp8.elf
endif

ifeq ($(TYPE), TAIL_OCP_FP4)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tail_ocp_fp4.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tail_ocp_fp4.elf
endif

ifeq ($(TYPE), NONTAIL_CUBLAS_FP8)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_cublas_fp8.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_nontail_cublas_fp8.elf
endif

include ../../../common/Makefile.common

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)
