# qli testcases: Quant Lightning Indexer
#
# =============================================================================
# ★ QLI 主线（已验证精度）：
#     qli_check_opt  — qli_pto_opt.hpp 精简 MSD radix-select TopK（推荐）
#       * 关键 Case：Sq=64 Skv=128 topk=128 / Sq=4 Skv=2048 topk=512
#                     Sq=4 Skv=8192 topk=512 / Sq=4 Skv=2080 topk=512
#       * 数据：gen_qli_golden.py 生成 compare/<cfg>，再按 test/kernel/qli/README.md
#               编译时自动生成符号数据对象，无需修补 C++ 地址
# =============================================================================

QLI_DTYPE ?= FP8

ifeq ($(TESTCASE), qli_check_opt)
   SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/qli_check_opt.cpp
   EXTRA_OBJ_FILES += $(QLI_DATA_DIR)/qli_data.o
   B ?= 1
   Sq ?= 64
   Skv ?= 128
   g ?= 64
   Tm ?= 16
   Tk ?= 32
   topk ?= 8
   ifneq ($(B),1)
      $(error qli_check_opt currently supports exactly B=1)
   endif
   DEFINES += -DBatch=1
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
   EXTRA_OBJ_FILES += $(QLI_DATA_DIR)/qli_data.o
   B ?= 1
   Sq ?= 4
   Skv ?= 128
   topk ?= 8
   NPE ?= 1
   g ?= 64
   ifneq ($(B),1)
      $(error qli_check_opt_dynamic currently supports exactly B=1)
   endif
   ifneq ($(QLI_DTYPE),FP8)
      $(error qli_check_opt_dynamic supports only QLI_DTYPE=FP8)
   endif
   ifneq ($(g),64)
      $(error qli_check_opt_dynamic requires g=64)
   endif
   DEFINES += -DBatch=1 -DTsq=$(Sq) -DTskv=$(Skv) -DTtopk=$(topk) -DTNPE=$(NPE)
   DEFINES += -DQLI_DTYPE_FP8
   DT_TAG = _fp8
   TARGET = $(ELF_HEAD)/$(TESTCASE)$(DT_TAG)_B$(B)_Sq$(Sq)_Skv$(Skv)_topk$(topk)_npe$(NPE).elf
endif

PYTHON ?= python3
QLI_DATA_DIR ?= $(OBJ_ROOT)/$(CATEGORY)/data/$(DT_TAG)_Sq$(Sq)_Skv$(Skv)_g$(g)_topk$(topk)
g ?= 64

include ../../common/Makefile.common

ifeq ($(PLAT),linx)
ELEMENT_TARGET_FLAGS = $(if $(TARGET_TRIPLE),--target=$(TARGET_TRIPLE)) \
    $(if $(SYSROOT),--sysroot=$(SYSROOT)) \
    $(if $(RESOURCE_DIR),-resource-dir=$(RESOURCE_DIR))
INCLUDE += $(if $(API_INCLUDE),-I$(API_INCLUDE))
CC_OPTS += $(ELEMENT_TARGET_FLAGS) -mllvm -linxv5-enable-generic-element=true \
    -fno-vectorize -fno-slp-vectorize -Werror=unknown-pragmas
CC_LINK += $(ELEMENT_TARGET_FLAGS)
endif

.PHONY: qli_prepare_data
qli_prepare_data: src/prepare_data.py src/gen_qli_golden.py
	$(PYTHON) src/prepare_data.py --out "$(QLI_DATA_DIR)" --sq $(Sq) --skv $(Skv) \
	    --topk $(topk) --heads $(g) --batch $(B) \
	    --tile-k $(if $(filter qli_check_opt_dynamic,$(TESTCASE)),32,$(Tk)) \
	    --npe $(if $(filter qli_check_opt_dynamic,$(TESTCASE)),$(NPE),1) \
	    --dtype $(if $(filter INT8,$(QLI_DTYPE)),int8,fp8)

$(QLI_DATA_DIR)/qli_data.o: qli_prepare_data
	$(AS) $(ELEMENT_TARGET_FLAGS) -c "$(QLI_DATA_DIR)/qli_data.s" -o "$@"


clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)
