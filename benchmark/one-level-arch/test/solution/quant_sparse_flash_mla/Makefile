# QSMLA solution: single-PE TADD and unified five-mode four-PE kernels.
# Keep the existing IMPL / QSMLA_DTYPE / shape options; see README.md.

ifeq ($(TESTCASE), quant_sparse_flash_mla)
   SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/quant_sparse_flash_mla.cpp
   B ?= 1
   s1 ?= 64
   s2 ?= 128
   N1 ?= 1
   N2 ?= 1
   D ?= 512
   QSMLA_DTYPE ?= HIF8
   # Single-PE compatibility: IMPL=tadd QSMLA_DTYPE=FP16.
   # HIF8 remains the default dtype for the unified four-PE implementations.
   IMPL ?= tadd
   Tm ?= 32
   ifeq ($(IMPL), tadd)
      # Single-PE tadd requires square B tiles (Tk == Td_block, unpatched
      # Local-B contract) and a row-reduction source <= 2048 bytes, so its
      # defaults follow the validated Tk16 square profile instead of the
      # legacy Tk=32/Td_block=64 pair (which now fails the kernel asserts).
      Tk ?= 16
      Td_block ?= 16
   else
      Tk ?= 32
      Td_block ?= 64
   endif
   softmax_scale ?= 0.125
   wleft ?= 1
   wright ?= 1
   cmp_s2 ?= 64
   ori_topk ?= 40
   cmp_topk ?= 40
   cmp_ratio ?= 4
   QSMLA_DTYPE ?= HIF8
   ifeq ($(filter $(QSMLA_DTYPE),HIF8 FP16),)
      $(error QSMLA_DTYPE must be HIF8 or FP16)
   endif
   ifeq ($(QSMLA_DTYPE), HIF8)
      q_descale ?= 0.5
      ori_kv_descale ?= 0.25
      cmp_kv_descale ?= 0.125
   else
      q_descale ?= 1.0
      ori_kv_descale ?= 1.0
      cmp_kv_descale ?= 1.0
   endif
   # Single-PE compatibility: IMPL=tadd QSMLA_DTYPE=FP16.
   # HIF8 remains the default dtype for the unified four-PE implementations.
   QSMLA_UNIFIED_IMPLS := swa_tadd_4pe hca_tadd_4pe csa_tadd_4pe ori_sparse_tadd_4pe ori_cmp_sparse_tadd_4pe
    QSMLA_1PE_IMPLS := csa_tadd_1pe
    QSMLA_V3_IMPLS := csa_v3_4pe csa_v3_1_4pe csa_v3_0_1_4pe csa_v3_2_4pe csa_v3_1_1_4pe csa_v3_3_4pe csa_v3_4_1_4pe csa_v3_3_1_4pe csa_v3_3_2_4pe csa_v3_3_3_4pe csa_v3_5_4pe
    # PA_BBND variants (v2.1 unified / v3.1 ops-flow) + block tables.
    QSMLA_V2_1_IMPLS := csa_v2_1_4pe
    # Dynamic-shape variant (v2): Vec tiles use DYNAMIC ValidRow/ValidCol.
    # Only valid for the unified four-PE IMPLs.
    ifeq ($(QSMLA_DYNAMIC_SHAPE), on)
       ifeq ($(filter $(IMPL),$(QSMLA_UNIFIED_IMPLS)),)
          $(error QSMLA_DYNAMIC_SHAPE requires a unified four-PE IMPL)
       endif
       DEFINES += -DQSMLA_USE_DYNAMIC_SHAPE
    endif
   ifeq ($(filter $(IMPL),tadd tadd_4pe $(QSMLA_UNIFIED_IMPLS) $(QSMLA_1PE_IMPLS) $(QSMLA_V3_IMPLS) $(QSMLA_V2_1_IMPLS)),)
      $(error unsupported QSMLA IMPL=$(IMPL); use tadd or a unified four-PE implementation)
   endif
    ifeq ($(IMPL), tadd)
       ifneq ($(QSMLA_DTYPE), FP16)
          $(error single-PE tadd requires QSMLA_DTYPE=FP16; HIF8 uses a unified four-PE IMPL)
       endif
    endif
   DEFINES += -DTbatch=$(B)
   DEFINES += -DTs1=$(s1)
   DEFINES += -DTs2=$(s2)
   DEFINES += -DTn1=$(N1)
   DEFINES += -DTn2=$(N2)
   DEFINES += -DTd=$(D)
   DEFINES += -DTm=$(Tm)
   DEFINES += -DTk=$(Tk)
   DEFINES += -DTd_block=$(Td_block)
   DEFINES += -DTsoftmax_scale=$(softmax_scale)
   DEFINES += -DTwleft=$(wleft)
   DEFINES += -DTwright=$(wright)
   DEFINES += -DTcmp_s2=$(cmp_s2)
   DEFINES += -DTori_topk=$(ori_topk)
   DEFINES += -DTcmp_topk=$(cmp_topk)
   DEFINES += -DTcmp_ratio=$(cmp_ratio)
   DEFINES += -DTq_descale=$(q_descale)
   DEFINES += -DTori_kv_descale=$(ori_kv_descale)
   DEFINES += -DTcmp_kv_descale=$(cmp_kv_descale)
   ifeq ($(QSMLA_DTYPE), HIF8)
      DEFINES += -DQSMLA_USE_HIF8
      QSMLA_INPUT_EXT := hif8
      QSMLA_INPUT_BYTES := 1
   else
      QSMLA_INPUT_EXT := fp16
      QSMLA_INPUT_BYTES := 2
   endif
   ifeq ($(IMPL), tadd)
      DEFINES += -DQSMLA_USE_TADD
   endif
   ifeq ($(IMPL), tadd_4pe)
      DEFINES += -DQSMLA_USE_TADD_4PE
      TARGET = $(ELF_HEAD)/$(TESTCASE)_B$(B)_s1$(s1)_s2$(s2)_N1$(N1)_N2$(N2)_D$(D)_Tm$(Tm)_Tk$(Tk)_Td$(Td_block)_IMPL$(IMPL).elf
   else
      TARGET = $(ELF_HEAD)/$(TESTCASE)_B$(B)_s1$(s1)_s2$(s2)_N1$(N1)_N2$(N2)_D$(D)_Tm$(Tm)_Tk$(Tk)_Td$(Td_block).elf
   endif

   ifneq ($(filter $(IMPL),$(QSMLA_UNIFIED_IMPLS)),)
      ifeq ($(IMPL), swa_tadd_4pe)
         DEFINES += -DQSMLA_USE_TADD_4PE
         QSMLA_SPARSE_CASE := swa_small
      endif
      ifeq ($(IMPL), hca_tadd_4pe)
         DEFINES += -DQSMLA_USE_HCA_TADD_4PE
         QSMLA_SPARSE_CASE := hca_small
      endif
      ifeq ($(IMPL), csa_tadd_4pe)
         DEFINES += -DQSMLA_USE_CSA_TADD_4PE
         QSMLA_SPARSE_CASE := csa_small
      endif
      ifeq ($(IMPL), ori_sparse_tadd_4pe)
         DEFINES += -DQSMLA_USE_ORI_SPARSE_TADD_4PE
         QSMLA_SPARSE_CASE := ori_sparse_small
      endif
      ifeq ($(IMPL), ori_cmp_sparse_tadd_4pe)
         DEFINES += -DQSMLA_USE_ORI_CMP_SPARSE_TADD_4PE
         QSMLA_SPARSE_CASE := ori_cmp_sparse_small
      endif
      TARGET = $(ELF_HEAD)/$(TESTCASE)_B$(B)_s1$(s1)_oriS2$(s2)_cmpS2$(cmp_s2)_N1$(N1)_N2$(N2)_D$(D)_oriK$(ori_topk)_cmpK$(cmp_topk)_ratio$(cmp_ratio)_Tm$(Tm)_Tk$(Tk)_Td$(Td_block)_IMPL$(IMPL).elf
   endif

   # v3: ops-arch35-flow 4-PE CSA port (single-pass online softmax,
   # 128-token blocks). Tm/Tk/Td_block only feed the ELF name / Config.
   ifeq ($(IMPL), csa_v3_4pe)
      DEFINES += -DQSMLA_USE_CSA_V3_4PE
      QSMLA_SPARSE_CASE := csa_small
      ifneq ($(QSMLA_DTYPE), HIF8)
         $(error csa_v3_4pe requires QSMLA_DTYPE=HIF8)
      endif
      TARGET = $(ELF_HEAD)/$(TESTCASE)_B$(B)_s1$(s1)_oriS2$(s2)_cmpS2$(cmp_s2)_N1$(N1)_N2$(N2)_D$(D)_oriK$(ori_topk)_cmpK$(cmp_topk)_ratio$(cmp_ratio)_Tm$(Tm)_Tk$(Tk)_Td$(Td_block)_IMPL$(IMPL)_DTYPE$(QSMLA_DTYPE)_INPUTembedded.elf
   endif

   # v3.0.1: v3 body + G-slice (N1 > 64 / network-case experiments). Same
   # ops-arch35 flow as v3; only the work decomposition adds G slices.
   ifeq ($(IMPL), csa_v3_0_1_4pe)
      DEFINES += -DQSMLA_USE_CSA_V3_0_1_4PE
      QSMLA_SPARSE_CASE := csa_small
      ifneq ($(QSMLA_DTYPE), HIF8)
         $(error csa_v3_0_1_4pe requires QSMLA_DTYPE=HIF8)
      endif
      TARGET = $(ELF_HEAD)/$(TESTCASE)_B$(B)_s1$(s1)_oriS2$(s2)_cmpS2$(cmp_s2)_N1$(N1)_N2$(N2)_D$(D)_oriK$(ori_topk)_cmpK$(cmp_topk)_ratio$(cmp_ratio)_Tm$(Tm)_Tk$(Tk)_Td$(Td_block)_IMPL$(IMPL)_DTYPE$(QSMLA_DTYPE)_INPUTembedded.elf
   endif

   # v3.1: v3 + PA_BBND block-table KV addressing (paged KV + identity block
   # tables generated in the case data dir).
   ifeq ($(IMPL), csa_v3_1_4pe)
      DEFINES += -DQSMLA_USE_CSA_V3_1_4PE
      DEFINES += -DQSMLA_PA_BLOCK_SIZE=128
      QSMLA_SPARSE_CASE := csa_small
      QSMLA_SPARSE_PA := on
      ifeq ($(QSMLA_PA_ISOLATE), on)
         DEFINES += -DQSMLA_PA_ISOLATE_NULL_TABLES
      endif
      ifneq ($(QSMLA_DTYPE), HIF8)
         $(error csa_v3_1_4pe requires QSMLA_DTYPE=HIF8)
      endif
   TARGET = $(ELF_HEAD)/$(TESTCASE)_B$(B)_s1$(s1)_oriS2$(s2)_cmpS2$(cmp_s2)_N1$(N1)_N2$(N2)_D$(D)_oriK$(ori_topk)_cmpK$(cmp_topk)_ratio$(cmp_ratio)_Tm$(Tm)_Tk$(Tk)_Td$(Td_block)_IMPL$(IMPL)_DTYPE$(QSMLA_DTYPE)_KVPABBND_INPUTembedded.elf
   endif

   # v3.4: O residency + inline staged-KV prefetch (no lambda).
   ifeq ($(IMPL), csa_v3_4_1_4pe)
      DEFINES += -DQSMLA_USE_CSA_V3_4_1_4PE
      DEFINES += -DQSMLA_PA_BLOCK_SIZE=128
      QSMLA_SPARSE_CASE := csa_small
      QSMLA_SPARSE_PA := on
      ifneq ($(QSMLA_DTYPE), HIF8)
         $(error csa_v3_4_1_4pe requires QSMLA_DTYPE=HIF8)
      endif
   TARGET = $(ELF_HEAD)/$(TESTCASE)_B$(B)_s1$(s1)_oriS2$(s2)_cmpS2$(cmp_s2)_N1$(N1)_N2$(N2)_D$(D)_oriK$(ori_topk)_cmpK$(cmp_topk)_ratio$(cmp_ratio)_Tm$(Tm)_Tk$(Tk)_Td$(Td_block)_IMPL$(IMPL)_DTYPE$(QSMLA_DTYPE)_KVPABBND_INPUTembedded.elf
   endif

   # v3.5: v3.3.3 + L1 侧 KV 双缓冲（kvA/kvB，深度 1）。
   ifeq ($(IMPL), csa_v3_5_4pe)
      DEFINES += -DQSMLA_USE_CSA_V3_5_4PE
      DEFINES += -DQSMLA_PRELOAD=$(QSMLA_PRELOAD)
      DEFINES += -DQSMLA_PA_BLOCK_SIZE=128
      QSMLA_SPARSE_CASE := csa_small
      QSMLA_SPARSE_PA := on
      ifneq ($(QSMLA_DTYPE), HIF8)
         $(error csa_v3_5_4pe requires QSMLA_DTYPE=HIF8)
      endif
   TARGET = $(ELF_HEAD)/$(TESTCASE)_B$(B)_s1$(s1)_oriS2$(s2)_cmpS2$(cmp_s2)_N1$(N1)_N2$(N2)_D$(D)_oriK$(ori_topk)_cmpK$(cmp_topk)_ratio$(cmp_ratio)_Tm$(Tm)_Tk$(Tk)_Td$(Td_block)_IMPL$(IMPL)_DTYPE$(QSMLA_DTYPE)_KVPABBND_INPUTembedded.elf
   endif

   # v3.3.3: 可配置预取深度（QSMLA_PRELOAD=x，默认 2）。
   QSMLA_PRELOAD ?= 2
   ifeq ($(IMPL), csa_v3_3_3_4pe)
      DEFINES += -DQSMLA_USE_CSA_V3_3_3_4PE
      DEFINES += -DQSMLA_PRELOAD=$(QSMLA_PRELOAD)
      DEFINES += -DQSMLA_PA_BLOCK_SIZE=128
      QSMLA_SPARSE_CASE := csa_small
      QSMLA_SPARSE_PA := on
      ifneq ($(QSMLA_DTYPE), HIF8)
         $(error csa_v3_3_3_4pe requires QSMLA_DTYPE=HIF8)
      endif
   TARGET = $(ELF_HEAD)/$(TESTCASE)_B$(B)_s1$(s1)_oriS2$(s2)_cmpS2$(cmp_s2)_N1$(N1)_N2$(N2)_D$(D)_oriK$(ori_topk)_cmpK$(cmp_topk)_ratio$(cmp_ratio)_Tm$(Tm)_Tk$(Tk)_Td$(Td_block)_IMPL$(IMPL)_P$(QSMLA_PRELOAD)_DTYPE$(QSMLA_DTYPE)_KVPABBND_INPUTembedded.elf
   endif

   # v3.3.2: v3.3.1 + preload=3（3 份 staging slice、预取距离 2）。
   ifeq ($(IMPL), csa_v3_3_2_4pe)
      DEFINES += -DQSMLA_USE_CSA_V3_3_2_4PE
      DEFINES += -DQSMLA_PA_BLOCK_SIZE=128
      QSMLA_SPARSE_CASE := csa_small
      QSMLA_SPARSE_PA := on
      ifneq ($(QSMLA_DTYPE), HIF8)
         $(error csa_v3_3_2_4pe requires QSMLA_DTYPE=HIF8)
      endif
   TARGET = $(ELF_HEAD)/$(TESTCASE)_B$(B)_s1$(s1)_oriS2$(s2)_cmpS2$(cmp_s2)_N1$(N1)_N2$(N2)_D$(D)_oriK$(ori_topk)_cmpK$(cmp_topk)_ratio$(cmp_ratio)_Tm$(Tm)_Tk$(Tk)_Td$(Td_block)_IMPL$(IMPL)_DTYPE$(QSMLA_DTYPE)_KVPABBND_INPUTembedded.elf
   endif

   # v3.3.1: v3.3 + staging 内部行级 ping-pong（双行 tile 交替）。
   ifeq ($(IMPL), csa_v3_3_1_4pe)
      DEFINES += -DQSMLA_USE_CSA_V3_3_1_4PE
      DEFINES += -DQSMLA_PA_BLOCK_SIZE=128
      QSMLA_SPARSE_CASE := csa_small
      QSMLA_SPARSE_PA := on
      ifneq ($(QSMLA_DTYPE), HIF8)
         $(error csa_v3_3_1_4pe requires QSMLA_DTYPE=HIF8)
      endif
   TARGET = $(ELF_HEAD)/$(TESTCASE)_B$(B)_s1$(s1)_oriS2$(s2)_cmpS2$(cmp_s2)_N1$(N1)_N2$(N2)_D$(D)_oriK$(ori_topk)_cmpK$(cmp_topk)_ratio$(cmp_ratio)_Tm$(Tm)_Tk$(Tk)_Td$(Td_block)_IMPL$(IMPL)_DTYPE$(QSMLA_DTYPE)_KVPABBND_INPUTembedded.elf
   endif

   # v3.3: O full UB residency + staged-KV prefetch.
   ifeq ($(IMPL), csa_v3_3_4pe)
      DEFINES += -DQSMLA_USE_CSA_V3_3_4PE
      DEFINES += -DQSMLA_PA_BLOCK_SIZE=128
      QSMLA_SPARSE_CASE := csa_small
      QSMLA_SPARSE_PA := on
      ifneq ($(QSMLA_DTYPE), HIF8)
         $(error csa_v3_3_4pe requires QSMLA_DTYPE=HIF8)
      endif
   TARGET = $(ELF_HEAD)/$(TESTCASE)_B$(B)_s1$(s1)_oriS2$(s2)_cmpS2$(cmp_s2)_N1$(N1)_N2$(N2)_D$(D)_oriK$(ori_topk)_cmpK$(cmp_topk)_ratio$(cmp_ratio)_Tm$(Tm)_Tk$(Tk)_Td$(Td_block)_IMPL$(IMPL)_DTYPE$(QSMLA_DTYPE)_KVPABBND_INPUTembedded.elf
   endif

   # v3.1.1: v3.1 + full UB-resident O accumulator.
   ifeq ($(IMPL), csa_v3_1_1_4pe)
      DEFINES += -DQSMLA_USE_CSA_V3_1_1_4PE
      DEFINES += -DQSMLA_PA_BLOCK_SIZE=128
      QSMLA_SPARSE_CASE := csa_small
      QSMLA_SPARSE_PA := on
      ifeq ($(QSMLA_PA_ISOLATE), on)
         DEFINES += -DQSMLA_PA_ISOLATE_NULL_TABLES
      endif
      ifneq ($(QSMLA_DTYPE), HIF8)
         $(error csa_v3_1_1_4pe requires QSMLA_DTYPE=HIF8)
      endif
   TARGET = $(ELF_HEAD)/$(TESTCASE)_B$(B)_s1$(s1)_oriS2$(s2)_cmpS2$(cmp_s2)_N1$(N1)_N2$(N2)_D$(D)_oriK$(ori_topk)_cmpK$(cmp_topk)_ratio$(cmp_ratio)_Tm$(Tm)_Tk$(Tk)_Td$(Td_block)_IMPL$(IMPL)_DTYPE$(QSMLA_DTYPE)_KVPABBND_INPUTembedded.elf
   endif

   # v3.2: v3.1 + KV ping-pong double buffering (prefetch next block's KV).
   ifeq ($(IMPL), csa_v3_2_4pe)
      DEFINES += -DQSMLA_USE_CSA_V3_2_4PE
      DEFINES += -DQSMLA_PA_BLOCK_SIZE=128
      QSMLA_SPARSE_CASE := csa_small
      QSMLA_SPARSE_PA := on
      ifeq ($(QSMLA_PA_ISOLATE), on)
         DEFINES += -DQSMLA_PA_ISOLATE_NULL_TABLES
      endif
      ifneq ($(QSMLA_DTYPE), HIF8)
         $(error csa_v3_2_4pe requires QSMLA_DTYPE=HIF8)
      endif
   TARGET = $(ELF_HEAD)/$(TESTCASE)_B$(B)_s1$(s1)_oriS2$(s2)_cmpS2$(cmp_s2)_N1$(N1)_N2$(N2)_D$(D)_oriK$(ori_topk)_cmpK$(cmp_topk)_ratio$(cmp_ratio)_Tm$(Tm)_Tk$(Tk)_Td$(Td_block)_IMPL$(IMPL)_DTYPE$(QSMLA_DTYPE)_KVPABBND_INPUTembedded.elf
   endif

   # v2.1: unified five-mode tadd + PA_BBND block-table KV addressing.
   ifeq ($(IMPL), csa_v2_1_4pe)
      DEFINES += -DQSMLA_USE_CSA_V2_1_4PE
      DEFINES += -DQSMLA_PA_BLOCK_SIZE=128
      QSMLA_SPARSE_CASE := csa_small
      QSMLA_SPARSE_PA := on
      ifeq ($(QSMLA_PA_ISOLATE), on)
         DEFINES += -DQSMLA_PA_ISOLATE_NULL_TABLES
      endif
      ifneq ($(QSMLA_DTYPE), HIF8)
         $(error csa_v2_1_4pe requires QSMLA_DTYPE=HIF8)
      endif
   TARGET = $(ELF_HEAD)/$(TESTCASE)_B$(B)_s1$(s1)_oriS2$(s2)_cmpS2$(cmp_s2)_N1$(N1)_N2$(N2)_D$(D)_oriK$(ori_topk)_cmpK$(cmp_topk)_ratio$(cmp_ratio)_Tm$(Tm)_Tk$(Tk)_Td$(Td_block)_IMPL$(IMPL)_DTYPE$(QSMLA_DTYPE)_KVPABBND_INPUTembedded.elf
   endif

   # Single-PE CSA (Local CUBE path, FP16, embedded input only).
   ifeq ($(IMPL), csa_tadd_1pe)
      ifneq ($(QSMLA_SPARSE_EMBED_INPUT), on)
         $(error csa_tadd_1pe requires QSMLA_SPARSE_EMBED_INPUT=on (FP16 csa_small data))
      endif
      DEFINES += -DQSMLA_USE_CSA_TADD_1PE
      QSMLA_SPARSE_CASE := csa_small
      TARGET = $(ELF_HEAD)/$(TESTCASE)_B$(B)_s1$(s1)_oriS2$(s2)_cmpS2$(cmp_s2)_N1$(N1)_N2$(N2)_D$(D)_oriK$(ori_topk)_cmpK$(cmp_topk)_ratio$(cmp_ratio)_Tm$(Tm)_Tk$(Tk)_Td$(Td_block)_IMPL$(IMPL)_DTYPE$(QSMLA_DTYPE)_INPUTembedded.elf
   endif

   ifeq ($(QSMLA_EMBED_INPUT), on)
      ifneq ($(B), 8)
         $(error QSMLA_EMBED_INPUT supports only B=8,S1=4,N1=128,N2=1,D=512)
      endif
      ifneq ($(s1), 4)
         $(error QSMLA_EMBED_INPUT supports only B=8,S1=4,N1=128,N2=1,D=512)
      endif
      ifneq ($(N1), 128)
         $(error QSMLA_EMBED_INPUT supports only B=8,S1=4,N1=128,N2=1,D=512)
      endif
      ifneq ($(N2), 1)
         $(error QSMLA_EMBED_INPUT supports only B=8,S1=4,N1=128,N2=1,D=512)
      endif
      ifneq ($(D), 512)
         $(error QSMLA_EMBED_INPUT supports only B=8,S1=4,N1=128,N2=1,D=512)
      endif
      ifeq ($(filter $(s2),128 1024 4096),)
         $(error QSMLA_EMBED_INPUT supports only S2=128,1024,4096)
      endif

      QSMLA_DATA_DIR ?= $(abspath $(CURDIR)/../../../../../..)/qsmla_testdata/typical_bsnd_swa
      QSMLA_DATA_OBJ_DIR := $(abspath $(CURDIR)/../../../output/solution/quant_sparse_flash_mla/qsmla_data)
      QSMLA_Q_OBJ := $(QSMLA_DATA_OBJ_DIR)/qsmla_q_input.o
      QSMLA_KV_OBJ := $(QSMLA_DATA_OBJ_DIR)/qsmla_kv_input_s2_$(s2).o
      QSMLA_Q_BYTES := 4194304
      QSMLA_KV_BYTES := $(shell expr $(B) \* $(s2) \* $(N2) \* $(D) \* 2)
      QSMLA_INPUT_BUILDER := $(CURDIR)/build_qsmla_input_object.py

      DEFINES += -DQSMLA_USE_EMBEDDED_INPUT
      EXTRA_OBJ_FILES += $(QSMLA_Q_OBJ) $(QSMLA_KV_OBJ)
      TARGET = $(ELF_HEAD)/$(TESTCASE)_B$(B)_s1$(s1)_s2$(s2)_N1$(N1)_N2$(N2)_D$(D)_Tm$(Tm)_Tk$(Tk)_Td$(Td_block)_IMPL$(IMPL)_INPUTembedded.elf

      $(QSMLA_Q_OBJ): $(QSMLA_DATA_DIR)/q.fp16.bin $(QSMLA_INPUT_BUILDER)
	@mkdir -p $(QSMLA_DATA_OBJ_DIR)
	python3 $(QSMLA_INPUT_BUILDER) --compiler $(CXX) --input $< --output $@ --symbol q_fp16 --bytes $(QSMLA_Q_BYTES)

      $(QSMLA_KV_OBJ): $(QSMLA_DATA_DIR)/kv_s2_4096.fp16.bin $(QSMLA_INPUT_BUILDER)
	@mkdir -p $(QSMLA_DATA_OBJ_DIR)
	python3 $(QSMLA_INPUT_BUILDER) --compiler $(CXX) --input $< --output $@ --symbol kv_fp16 --bytes $(QSMLA_KV_BYTES)
   endif

   ifeq ($(QSMLA_SPARSE_EMBED_INPUT), on)
      ifeq ($(filter $(IMPL),$(QSMLA_UNIFIED_IMPLS) $(QSMLA_1PE_IMPLS) $(QSMLA_V3_IMPLS) $(QSMLA_V2_1_IMPLS)),)
         $(error QSMLA_SPARSE_EMBED_INPUT requires a unified four-PE IMPL or csa_tadd_1pe)
      endif
      QSMLA_SPARSE_DATA_ROOT ?= /tmp/qsmla-sparse
      QSMLA_SPARSE_DATA_DIR := $(QSMLA_SPARSE_DATA_ROOT)/$(QSMLA_SPARSE_CASE)
      QSMLA_SPARSE_OBJ_DIR := $(abspath $(CURDIR)/../../../output/solution/quant_sparse_flash_mla/qsmla_sparse_data/$(QSMLA_SPARSE_CASE)_$(QSMLA_DTYPE))
      QSMLA_INPUT_BUILDER := $(CURDIR)/build_qsmla_input_object.py
      QSMLA_Q_BYTES := $(shell expr $(B) \* $(s1) \* $(N1) \* $(D) \* $(QSMLA_INPUT_BYTES))
      QSMLA_ORI_KV_BYTES := $(shell expr $(B) \* $(s2) \* $(N2) \* $(D) \* $(QSMLA_INPUT_BYTES))
      QSMLA_CMP_KV_BYTES := $(shell expr $(B) \* $(cmp_s2) \* $(N2) \* $(D) \* $(QSMLA_INPUT_BYTES))
      # PA_BBND pool sizing (block_size = 128).
      QSMLA_PA_BLOCK := 128
      QSMLA_ORI_ROWS_PA := $(shell expr \( \( $(s2) + $(QSMLA_PA_BLOCK) - 1 \) / $(QSMLA_PA_BLOCK) \) \* $(QSMLA_PA_BLOCK))
      QSMLA_CMP_ROWS_PA := $(shell expr \( \( $(cmp_s2) + $(QSMLA_PA_BLOCK) - 1 \) / $(QSMLA_PA_BLOCK) \) \* $(QSMLA_PA_BLOCK))
      QSMLA_CMP_BYTES_PA := $(shell expr $(B) \* $(QSMLA_CMP_ROWS_PA) \* $(N2) \* $(D) \* $(QSMLA_INPUT_BYTES))
      QSMLA_PA_MAX_BLOCKS := $(shell expr \( $(s2) + $(QSMLA_PA_BLOCK) - 1 \) / $(QSMLA_PA_BLOCK))
      QSMLA_PA_TABLE_BYTES := $(shell expr $(B) \* $(QSMLA_PA_MAX_BLOCKS) \* 4)
      # CMP block table is its own width (ceil(cmp_s2 / block)); for cmp_s2 > s2
      # it is wider than the ORI table (e.g. the network CSA case).
      QSMLA_PA_CMP_MAX_BLOCKS := $(shell expr \( $(cmp_s2) + $(QSMLA_PA_BLOCK) - 1 \) / $(QSMLA_PA_BLOCK))
      QSMLA_PA_CMP_TABLE_BYTES := $(shell expr $(B) \* $(QSMLA_PA_CMP_MAX_BLOCKS) \* 4)
      QSMLA_ORI_INDEX_BYTES := $(shell expr $(B) \* $(s1) \* $(N2) \* $(ori_topk) \* 4)
      QSMLA_CMP_INDEX_BYTES := $(shell expr $(B) \* $(s1) \* $(N2) \* $(cmp_topk) \* 4)
      QSMLA_LENGTH_BYTES := $(shell expr $(B) \* $(s1) \* $(N2) \* 4)

      QSMLA_Q_OBJ := $(QSMLA_SPARSE_OBJ_DIR)/q.o
      QSMLA_ORI_KV_OBJ := $(QSMLA_SPARSE_OBJ_DIR)/ori_kv.o
      EXTRA_OBJ_FILES += $(QSMLA_Q_OBJ) $(QSMLA_ORI_KV_OBJ)
      DEFINES += -DQSMLA_USE_SPARSE_EMBEDDED_INPUT

      ifneq ($(filter $(IMPL),hca_tadd_4pe csa_tadd_4pe ori_cmp_sparse_tadd_4pe csa_tadd_1pe csa_v3_4pe csa_v3_1_4pe csa_v3_0_1_4pe csa_v2_1_4pe csa_v3_2_4pe csa_v3_1_1_4pe csa_v3_3_4pe csa_v3_4_1_4pe csa_v3_3_1_4pe csa_v3_3_2_4pe csa_v3_3_3_4pe csa_v3_5_4pe),)
         QSMLA_CMP_KV_OBJ := $(QSMLA_SPARSE_OBJ_DIR)/cmp_kv.o
         EXTRA_OBJ_FILES += $(QSMLA_CMP_KV_OBJ)
      endif
       ifneq ($(filter $(IMPL),csa_tadd_4pe ori_cmp_sparse_tadd_4pe csa_tadd_1pe csa_v3_4pe csa_v3_1_4pe csa_v3_0_1_4pe csa_v2_1_4pe csa_v3_2_4pe csa_v3_1_1_4pe csa_v3_3_4pe csa_v3_4_1_4pe csa_v3_3_1_4pe csa_v3_3_2_4pe csa_v3_3_3_4pe csa_v3_5_4pe),)
          QSMLA_CMP_INDEX_OBJ := $(QSMLA_SPARSE_OBJ_DIR)/cmp_indices.o
          EXTRA_OBJ_FILES += $(QSMLA_CMP_INDEX_OBJ)
       endif
       ifeq ($(IMPL), csa_tadd_1pe)
          # qli-style pre-transposed ORI KV view [D, S2] per batch: the
          # single-PE CSA BMM1 B-tile (CUBE_N8 ND source) consumes K^T
          # directly, so the transpose moves out of the kernel into the
          # input layout (same contract as the qli test harness).
          QSMLA_ORI_KV_T_OBJ := $(QSMLA_SPARSE_OBJ_DIR)/ori_kv_t.o
          EXTRA_OBJ_FILES += $(QSMLA_ORI_KV_T_OBJ)
       endif
      ifneq ($(filter $(IMPL),ori_sparse_tadd_4pe ori_cmp_sparse_tadd_4pe),)
         QSMLA_ORI_INDEX_OBJ := $(QSMLA_SPARSE_OBJ_DIR)/ori_indices.o
         QSMLA_ORI_LENGTH_OBJ := $(QSMLA_SPARSE_OBJ_DIR)/ori_length.o
         EXTRA_OBJ_FILES += $(QSMLA_ORI_INDEX_OBJ) $(QSMLA_ORI_LENGTH_OBJ)
      endif
      ifeq ($(IMPL), ori_cmp_sparse_tadd_4pe)
         QSMLA_CMP_LENGTH_OBJ := $(QSMLA_SPARSE_OBJ_DIR)/cmp_length.o
         EXTRA_OBJ_FILES += $(QSMLA_CMP_LENGTH_OBJ)
      endif

      # PA_BBND variants: paged KV copies + [B, max_blocks] identity block
      # tables (block_size = 128), generated beside the case data.
      ifeq ($(QSMLA_SPARSE_PA), on)
         QSMLA_ORI_KV_PA_OBJ := $(QSMLA_SPARSE_OBJ_DIR)/ori_kv_pa.o
         QSMLA_CMP_KV_PA_OBJ := $(QSMLA_SPARSE_OBJ_DIR)/cmp_kv_pa.o
         QSMLA_ORI_BT_OBJ := $(QSMLA_SPARSE_OBJ_DIR)/ori_block_table.o
         QSMLA_CMP_BT_OBJ := $(QSMLA_SPARSE_OBJ_DIR)/cmp_block_table.o
         EXTRA_OBJ_FILES += $(QSMLA_ORI_KV_PA_OBJ) $(QSMLA_CMP_KV_PA_OBJ) \
                            $(QSMLA_ORI_BT_OBJ) $(QSMLA_CMP_BT_OBJ)
         DEFINES += -DQSMLA_SPARSE_KV_PA_BBND=1
      endif

      TARGET = $(ELF_HEAD)/$(TESTCASE)_B$(B)_s1$(s1)_oriS2$(s2)_cmpS2$(cmp_s2)_N1$(N1)_N2$(N2)_D$(D)_oriK$(ori_topk)_cmpK$(cmp_topk)_ratio$(cmp_ratio)_Tm$(Tm)_Tk$(Tk)_Td$(Td_block)_IMPL$(IMPL)_DTYPE$(QSMLA_DTYPE)_INPUTembedded.elf

      $(QSMLA_Q_OBJ): $(QSMLA_SPARSE_DATA_DIR)/q.$(QSMLA_INPUT_EXT).bin $(QSMLA_INPUT_BUILDER)
	@mkdir -p $(QSMLA_SPARSE_OBJ_DIR)
	python3 $(QSMLA_INPUT_BUILDER) --compiler $(CXX) --input $< --output $@ --symbol q_$(QSMLA_INPUT_EXT) --bytes $(QSMLA_Q_BYTES) --exact-bytes

       $(QSMLA_ORI_KV_OBJ): $(QSMLA_SPARSE_DATA_DIR)/ori_kv.$(QSMLA_INPUT_EXT).bin $(QSMLA_INPUT_BUILDER)
	@mkdir -p $(QSMLA_SPARSE_OBJ_DIR)
	python3 $(QSMLA_INPUT_BUILDER) --compiler $(CXX) --input $< --output $@ --symbol ori_kv_$(QSMLA_INPUT_EXT) --bytes $(QSMLA_ORI_KV_BYTES) --exact-bytes

       $(QSMLA_ORI_KV_T_OBJ): $(QSMLA_SPARSE_DATA_DIR)/ori_kv_t.$(QSMLA_INPUT_EXT).bin $(QSMLA_INPUT_BUILDER)
	@mkdir -p $(QSMLA_SPARSE_OBJ_DIR)
	python3 $(QSMLA_INPUT_BUILDER) --compiler $(CXX) --input $< --output $@ --symbol ori_kv_t_$(QSMLA_INPUT_EXT) --bytes $(QSMLA_ORI_KV_BYTES) --exact-bytes

      $(QSMLA_CMP_KV_OBJ): $(QSMLA_SPARSE_DATA_DIR)/cmp_kv.$(QSMLA_INPUT_EXT).bin $(QSMLA_INPUT_BUILDER)
	@mkdir -p $(QSMLA_SPARSE_OBJ_DIR)
	python3 $(QSMLA_INPUT_BUILDER) --compiler $(CXX) --input $< --output $@ --symbol cmp_kv_$(QSMLA_INPUT_EXT) --bytes $(QSMLA_CMP_KV_BYTES) --exact-bytes

      $(QSMLA_ORI_INDEX_OBJ): $(QSMLA_SPARSE_DATA_DIR)/ori_sparse_indices.int32.bin $(QSMLA_INPUT_BUILDER)
	@mkdir -p $(QSMLA_SPARSE_OBJ_DIR)
	python3 $(QSMLA_INPUT_BUILDER) --compiler $(CXX) --input $< --output $@ --symbol ori_sparse_indices_int32 --bytes $(QSMLA_ORI_INDEX_BYTES) --exact-bytes

      $(QSMLA_CMP_INDEX_OBJ): $(QSMLA_SPARSE_DATA_DIR)/cmp_sparse_indices.int32.bin $(QSMLA_INPUT_BUILDER)
	@mkdir -p $(QSMLA_SPARSE_OBJ_DIR)
	python3 $(QSMLA_INPUT_BUILDER) --compiler $(CXX) --input $< --output $@ --symbol cmp_sparse_indices_int32 --bytes $(QSMLA_CMP_INDEX_BYTES) --exact-bytes

      $(QSMLA_ORI_LENGTH_OBJ): $(QSMLA_SPARSE_DATA_DIR)/ori_topk_length.int32.bin $(QSMLA_INPUT_BUILDER)
	@mkdir -p $(QSMLA_SPARSE_OBJ_DIR)
	python3 $(QSMLA_INPUT_BUILDER) --compiler $(CXX) --input $< --output $@ --symbol ori_topk_length_int32 --bytes $(QSMLA_LENGTH_BYTES) --exact-bytes

      $(QSMLA_CMP_LENGTH_OBJ): $(QSMLA_SPARSE_DATA_DIR)/cmp_topk_length.int32.bin $(QSMLA_INPUT_BUILDER)
	@mkdir -p $(QSMLA_SPARSE_OBJ_DIR)
	python3 $(QSMLA_INPUT_BUILDER) --compiler $(CXX) --input $< --output $@ --symbol cmp_topk_length_int32 --bytes $(QSMLA_LENGTH_BYTES) --exact-bytes

      ifeq ($(QSMLA_SPARSE_PA), on)
      $(QSMLA_ORI_KV_PA_OBJ): $(QSMLA_SPARSE_DATA_DIR)/ori_kv_pa.$(QSMLA_INPUT_EXT).bin $(QSMLA_INPUT_BUILDER)
	@mkdir -p $(QSMLA_SPARSE_OBJ_DIR)
	python3 $(QSMLA_INPUT_BUILDER) --compiler $(CXX) --input $< --output $@ --symbol ori_kv_pa_$(QSMLA_INPUT_EXT) --bytes $(QSMLA_ORI_KV_BYTES) --exact-bytes

      $(QSMLA_CMP_KV_PA_OBJ): $(QSMLA_SPARSE_DATA_DIR)/cmp_kv_pa.$(QSMLA_INPUT_EXT).bin $(QSMLA_INPUT_BUILDER)
	@mkdir -p $(QSMLA_SPARSE_OBJ_DIR)
	python3 $(QSMLA_INPUT_BUILDER) --compiler $(CXX) --input $< --output $@ --symbol cmp_kv_pa_$(QSMLA_INPUT_EXT) --bytes $(QSMLA_CMP_BYTES_PA) --exact-bytes

      $(QSMLA_ORI_BT_OBJ): $(QSMLA_SPARSE_DATA_DIR)/ori_block_table.int32.bin $(QSMLA_INPUT_BUILDER)
	@mkdir -p $(QSMLA_SPARSE_OBJ_DIR)
	python3 $(QSMLA_INPUT_BUILDER) --compiler $(CXX) --input $< --output $@ --symbol ori_block_table_int32 --bytes $(QSMLA_PA_TABLE_BYTES) --exact-bytes

      $(QSMLA_CMP_BT_OBJ): $(QSMLA_SPARSE_DATA_DIR)/cmp_block_table.int32.bin $(QSMLA_INPUT_BUILDER)
	@mkdir -p $(QSMLA_SPARSE_OBJ_DIR)
	python3 $(QSMLA_INPUT_BUILDER) --compiler $(CXX) --input $< --output $@ --symbol cmp_block_table_int32 --bytes $(QSMLA_PA_CMP_TABLE_BYTES) --exact-bytes
      endif
   endif
endif

include ../../common/Makefile.common
