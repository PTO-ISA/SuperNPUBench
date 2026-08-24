# qli testcases: Quant Lightning Indexer
#
# =============================================================================
# ★ QLI demo 主线（已提交，已验证精度）：
#     qli_check_opt  — qli_pto_opt.hpp 多轮 MSD radix-select TopK（推荐）
#       * 关键 Case：Sq=64 Skv=128 topk=128 / Sq=4 Skv=2048 topk=512
#                     Sq=4 Skv=8192 topk=512 / Sq=4 Skv=2080 topk=512
#       * 数据：gen_qli_golden.py 生成 compare/<cfg>，再按 test/kernel/qli/README.md
#               流程更新 qli_check_data.s + clang 生成 .o + fix_cpp_addrs.py
# =============================================================================
# 其余条目（qli / qli_tail / qli_bucket / qli_fused / qli_bucket_check /
# qli_check / qli_check_opt_mt / qli_check_bf16 / cube_mini / cube_mini8）
# 保留用于向后兼容：对应 src/*.cpp（bucket/bf16/fused/tail 等历史实验）未随
# 本 demo 提交，需自行补充源码后方可编译。
# =============================================================================

ifeq ($(TESTCASE), qli)
   SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/qli.cpp
   B ?= 1
   Sq ?= 64
   Skv ?= 128
   g ?= 64
   Tm ?= 16
   Tk ?= 32
   topk ?= 8
   DEFINES += -DBatch=$(B)
   DEFINES += -DTsq=$(Sq)
   DEFINES += -DTskv=$(Skv)
   DEFINES += -DTg=$(g)
   DEFINES += -DTm=$(Tm)
   DEFINES += -DTk=$(Tk)
   DEFINES += -DTtopk=$(topk)
   ifeq ($(QLI_DTYPE), FP8)
      DEFINES += -DQLI_DTYPE_FP8
      DT_TAG = _fp8
   else ifeq ($(QLI_DTYPE), INT8)
      DEFINES += -DQLI_INT8
      DT_TAG = _int8
   else
      DT_TAG = _fp16
   endif
   ifeq ($(QLI_WTYPE), FP32)
      DEFINES += -DQLI_WTYPE_FP32
      W_DT_TAG = _fp32w
   else
      W_DT_TAG =
   endif
   TARGET = $(ELF_HEAD)/$(TESTCASE)$(DT_TAG)$(W_DT_TAG)_B$(B)_Sq$(Sq)_Skv$(Skv)_g$(g)_Tm$(Tm)_Tk$(Tk).elf
endif

ifeq ($(TESTCASE), qli_tail)
   SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/qli_tail.cpp
   B ?= 1
   Sq ?= 66
   Skv ?= 128
   g ?= 64
   Tm ?= 16
   Tk ?= 32
   topk ?= 8
   DEFINES += -DBatch=$(B)
   DEFINES += -DTsq=$(Sq)
   DEFINES += -DTskv=$(Skv)
   DEFINES += -DTg=$(g)
   DEFINES += -DTm=$(Tm)
   DEFINES += -DTk=$(Tk)
   DEFINES += -DTtopk=$(topk)
   ifeq ($(QLI_DTYPE), FP8)
      DEFINES += -DQLI_DTYPE_FP8
      DT_TAG = _fp8
   else ifeq ($(QLI_DTYPE), INT8)
      DEFINES += -DQLI_INT8
      DT_TAG = _int8
   else
      DT_TAG = _fp16
   endif
   ifeq ($(QLI_WTYPE), FP32)
      DEFINES += -DQLI_WTYPE_FP32
      W_DT_TAG = _fp32w
   else
      W_DT_TAG =
   endif
   TARGET = $(ELF_HEAD)/$(TESTCASE)$(DT_TAG)$(W_DT_TAG)_B$(B)_Sq$(Sq)_Skv$(Skv)_g$(g)_Tm$(Tm)_Tk$(Tk).elf
endif

ifeq ($(TESTCASE), qli_bucket)
   SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/qli_bucket.cpp
   B ?= 1
   Sq ?= 128
   Skv ?= 256
   g ?= 64
   Tm ?= 16
   Tk ?= 32
   topk ?= 8
   DEFINES += -DBatch=$(B)
   DEFINES += -DTsq=$(Sq)
   DEFINES += -DTskv=$(Skv)
   DEFINES += -DTg=$(g)
   DEFINES += -DTm=$(Tm)
   DEFINES += -DTk=$(Tk)
   DEFINES += -DTtopk=$(topk)
   ifeq ($(QLI_DTYPE), INT8)
      DEFINES += -DQLI_INT8
      DT_TAG = _int8
   else
      DEFINES += -DQLI_DTYPE_FP8
      DT_TAG = _fp8
   endif
   ifeq ($(QLI_SCORE), BF16)
      DEFINES += -DQLI_SCORE_BF16
      S_TAG = _bf16s
   else
      S_TAG = _fp32s
   endif
   TARGET = $(ELF_HEAD)/$(TESTCASE)$(DT_TAG)$(S_TAG)_B$(B)_Sq$(Sq)_Skv$(Skv)_g$(g)_Tm$(Tm)_Tk$(Tk).elf
endif

ifeq ($(TESTCASE), qli_fused)
   SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/qli_fused.cpp
   B ?= 1
   Sq ?= 64
   Skv ?= 128
   g ?= 64
   Tm ?= 16
   Tk ?= 32
   topk ?= 8
   DEFINES += -DBatch=$(B)
   DEFINES += -DTsq=$(Sq)
   DEFINES += -DTskv=$(Skv)
   DEFINES += -DTg=$(g)
   DEFINES += -DTm=$(Tm)
   DEFINES += -DTk=$(Tk)
   DEFINES += -DTtopk=$(topk)
   ifeq ($(QLI_DTYPE), FP8)
      DEFINES += -DQLI_DTYPE_FP8
      DT_TAG = _fp8
   else ifeq ($(QLI_DTYPE), INT8)
      DEFINES += -DQLI_INT8
      DT_TAG = _int8
   else
      DT_TAG = _fp16
   endif
   ifeq ($(QLI_WTYPE), FP32)
      DEFINES += -DQLI_WTYPE_FP32
      W_DT_TAG = _fp32w
   else
      W_DT_TAG =
   endif
   TARGET = $(ELF_HEAD)/$(TESTCASE)$(DT_TAG)$(W_DT_TAG)_B$(B)_Sq$(Sq)_Skv$(Skv)_g$(g)_Tm$(Tm)_Tk$(Tk).elf
endif

ifeq ($(TESTCASE), qli_bucket_check)
   SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/qli_bucket_check.cpp
   # .incbin 数据对象（输入 srcq/srck/srcw/srcsq/srcsk）
   EXTRA_OBJ_FILES += $(TEST_ROOT)/$(CASE_SRC_DIR)/qli_bucket_data.o
   B ?= 1
   Sq ?= 128
   Skv ?= 256
   g ?= 64
   Tm ?= 16
   Tk ?= 32
   topk ?= 8
   DEFINES += -DBatch=$(B)
   DEFINES += -DTsq=$(Sq)
   DEFINES += -DTskv=$(Skv)
   DEFINES += -DTg=$(g)
   DEFINES += -DTm=$(Tm)
   DEFINES += -DTk=$(Tk)
   DEFINES += -DTtopk=$(topk)
   ifeq ($(QLI_DTYPE), INT8)
      DEFINES += -DQLI_INT8
      DT_TAG = _int8
   else
      DEFINES += -DQLI_DTYPE_FP8
      DT_TAG = _fp8
   endif
   ifeq ($(QLI_SCORE), BF16)
      DEFINES += -DQLI_SCORE_BF16
      S_TAG = _bf16s
   else
      S_TAG = _fp32s
   endif
   TARGET = $(ELF_HEAD)/$(TESTCASE)$(DT_TAG)$(S_TAG)_B$(B)_Sq$(Sq)_Skv$(Skv)_g$(g)_Tm$(Tm)_Tk$(Tk).elf
endif

ifeq ($(TESTCASE), qli_check)
   SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/qli_check.cpp
   EXTRA_OBJ_FILES += $(TEST_ROOT)/$(CASE_SRC_DIR)/qli_check_data.o
   B ?= 1
   Sq ?= 64
   Skv ?= 128
   g ?= 64
   Tm ?= 16
   Tk ?= 32
   topk ?= 8
   DEFINES += -DBatch=$(B)
   DEFINES += -DTsq=$(Sq)
   DEFINES += -DTskv=$(Skv)
   DEFINES += -DTg=$(g)
   DEFINES += -DTm=$(Tm)
   DEFINES += -DTk=$(Tk)
   DEFINES += -DTtopk=$(topk)
   ifeq ($(QLI_DTYPE), INT8)
      DEFINES += -DQLI_INT8
      DT_TAG = _int8
   else
      DEFINES += -DQLI_DTYPE_FP8
      DT_TAG = _fp8
   endif
   TARGET = $(ELF_HEAD)/$(TESTCASE)$(DT_TAG)_B$(B)_Sq$(Sq)_Skv$(Skv)_g$(g)_Tm$(Tm)_Tk$(Tk).elf
endif

ifeq ($(TESTCASE), qli_check_opt)
   SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/qli_check_opt.cpp
   EXTRA_OBJ_FILES += $(TEST_ROOT)/$(CASE_SRC_DIR)/qli_check_data.o
   B ?= 1
   Sq ?= 64
   Skv ?= 128
   g ?= 64
   Tm ?= 16
   Tk ?= 32
   topk ?= 8
   DEFINES += -DBatch=$(B)
   DEFINES += -DTsq=$(Sq)
   DEFINES += -DTskv=$(Skv)
   DEFINES += -DTg=$(g)
   DEFINES += -DTm=$(Tm)
   DEFINES += -DTk=$(Tk)
   DEFINES += -DTtopk=$(topk)
   ifeq ($(QLI_DTYPE), INT8)
      DEFINES += -DQLI_INT8
      DT_TAG = _int8
   else
      DEFINES += -DQLI_DTYPE_FP8
      DT_TAG = _fp8
   endif
   TARGET = $(ELF_HEAD)/$(TESTCASE)$(DT_TAG)_B$(B)_Sq$(Sq)_Skv$(Skv)_g$(g)_Tm$(Tm)_Tk$(Tk).elf
endif

ifeq ($(TESTCASE), qli_check_opt_mt)
   SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/qli_check_opt_mt.cpp
   EXTRA_OBJ_FILES += $(TEST_ROOT)/$(CASE_SRC_DIR)/qli_check_data.o
   B ?= 1
   Sq ?= 64
   Skv ?= 128
   g ?= 64
   Tm ?= 16
   Tk ?= 32
   topk ?= 8
   DEFINES += -DBatch=$(B)
   DEFINES += -DTsq=$(Sq)
   DEFINES += -DTskv=$(Skv)
   DEFINES += -DTg=$(g)
   DEFINES += -DTm=$(Tm)
   DEFINES += -DTk=$(Tk)
   DEFINES += -DTtopk=$(topk)
   ifeq ($(QLI_DTYPE), INT8)
      DEFINES += -DQLI_INT8
      DT_TAG = _int8
   else
      DEFINES += -DQLI_DTYPE_FP8
      DT_TAG = _fp8
   endif
   TARGET = $(ELF_HEAD)/$(TESTCASE)$(DT_TAG)_B$(B)_Sq$(Sq)_Skv$(Skv)_g$(g)_Tm$(Tm)_Tk$(Tk).elf
endif

ifeq ($(TESTCASE), qli_check_bf16)
   SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/qli_check_bf16.cpp
   EXTRA_OBJ_FILES += $(TEST_ROOT)/$(CASE_SRC_DIR)/qli_check_data.o
   B ?= 1
   Sq ?= 64
   Skv ?= 128
   g ?= 64
   Tm ?= 16
   Tk ?= 32
   topk ?= 8
   DEFINES += -DBatch=$(B)
   DEFINES += -DTsq=$(Sq)
   DEFINES += -DTskv=$(Skv)
   DEFINES += -DTg=$(g)
   DEFINES += -DTm=$(Tm)
   DEFINES += -DTk=$(Tk)
   DEFINES += -DTtopk=$(topk)
   ifeq ($(QLI_DTYPE), INT8)
      DEFINES += -DQLI_INT8
      DT_TAG = _int8
   else
      DEFINES += -DQLI_DTYPE_FP8
      DT_TAG = _fp8
   endif
   TARGET = $(ELF_HEAD)/$(TESTCASE)$(DT_TAG)_B$(B)_Sq$(Sq)_Skv$(Skv)_g$(g)_Tm$(Tm)_Tk$(Tk).elf
endif

ifeq ($(TESTCASE), cube_mini)
   SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/cube_mini.cpp
   DEFINES += -D__linx
   TARGET = $(ELF_HEAD)/$(TESTCASE).elf
endif

ifeq ($(TESTCASE), cube_mini8)
   SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/cube_mini8.cpp
   DEFINES += -D__linx
   TARGET = $(ELF_HEAD)/$(TESTCASE).elf
endif

include ../../common/Makefile.common

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)
