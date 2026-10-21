# qli testcases: Quant Lightning Indexer
#
# =============================================================================
# ★ QLI 主线（已验证精度）：
#     qli_check_opt  — qli_pto_opt.hpp 精简 MSD radix-select TopK（推荐）
#       * 关键 Case：Sq=64 Skv=128 topk=128 / Sq=4 Skv=2048 topk=512
#                     Sq=4 Skv=8192 topk=512 / Sq=4 Skv=2080 topk=512
#       * 数据：gen_qli_golden.py 生成 compare/<cfg>，再按 test/kernel/qli/README.md
#               流程更新 qli_check_data.s + clang 生成 .o + fix_cpp_addrs.py
# =============================================================================

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
   # 多 PE 测试路径（仅此路径）：wbb/K^T 转置按块拆到 NPE 个 PE，分数+TopK
   # 仍由 PE0 串行（kernel 行级并行待后续）。NPE 默认 4。
   # 运行须 -s softcore.multiThreadNum=$(NPE)。ELF 名固定带 _npeN 后缀。
   NPE ?= 4
   DEFINES += -DTNPE=$(NPE)
   NPE_TAG = _npe$(NPE)
   # topk 入名（与 qli_check_opt_dynamic 的命名约定一致）：同 (Sq,Skv) 不同
   # topk 的变体（如真机对照的 topk512/topk1024）产物不再互相覆盖
   TARGET = $(ELF_HEAD)/$(TESTCASE)$(DT_TAG)_B$(B)_Sq$(Sq)_Skv$(Skv)_topk$(topk)$(NPE_TAG)_g$(g)_Tm$(Tm)_Tk$(Tk).elf
endif

ifeq ($(TESTCASE), qli_check_opt_dynamic)
   SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/qli_check_opt_dynamic.cpp
   EXTRA_OBJ_FILES += $(TEST_ROOT)/$(CASE_SRC_DIR)/qli_check_data.o
   B ?= 1
   Sq ?= 4
   Skv ?= 128
   topk ?= 8
   NPE ?= 1
   DEFINES += -DBatch=$(B) -DTsq=$(Sq) -DTskv=$(Skv) -DTtopk=$(topk) -DTNPE=$(NPE)
   DEFINES += -DQLI_DTYPE_FP8
   DT_TAG = _fp8
   TARGET = $(ELF_HEAD)/$(TESTCASE)$(DT_TAG)_B$(B)_Sq$(Sq)_Skv$(Skv)_topk$(topk)_npe$(NPE).elf
endif

include ../../common/Makefile.common

# qli_check_data.o：纯数据目标文件（.incbin 嵌入输入），必须 -c 只汇编不链接。
# make 内建 %.o:%.s 规则对 AS=clang 漏了 -c，clang 会尝试把无 main 的 .o 链接成可执行
# 而报 "undefined symbol: main"，故显式覆盖。incbin 的 compare/current/*.bin 相对 make
# 工作目录（本 test 目录）解析，由 compile.all 逐变体用 gen_qli_golden.py 预生成。
$(TEST_ROOT)/$(CASE_SRC_DIR)/qli_check_data.o: $(TEST_ROOT)/$(CASE_SRC_DIR)/qli_check_data.s
	$(AS) -c $< -o $@

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)
