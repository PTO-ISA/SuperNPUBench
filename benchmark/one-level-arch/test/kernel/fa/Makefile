# fa testcases: fa_HIF4_HIF4, fa_2d_unroll, sfa

ifeq ($(TESTCASE), sfa)
   SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/sfa.cpp
   Sq ?= 256
   Skv ?= 512
   Tm ?= 8
   Tk ?= 16
   DEFINES += -DTsq=$(Sq)
   DEFINES += -DTskv=$(Skv)
   DEFINES += -DTm=$(Tm)
   DEFINES += -DTk=$(Tk)
   TARGET = $(ELF_HEAD)/$(TESTCASE)_Sq$(Sq)_Skv$(Skv)_Tm$(Tm)_Tk$(Tk).elf
endif

ifeq ($(TESTCASE), fa_HIF4_HIF4)
   SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/fa_hif4.cpp
   Sq ?= 512
   Skv ?= 512
   Tm ?= 8
   Tk ?= 32
   X ?= 2
   Y ?= 4
   DEFINES += -DTsq=$(Sq)
   DEFINES += -DTskv=$(Skv)
   DEFINES += -DTm=$(Tm)
   DEFINES += -DTk=$(Tk)
   DEFINES += -DXdim=$(X)
   DEFINES += -DYdim=$(Y)
   ifneq ($(MODE), )
      DEFINES += -D$(MODE)
   endif
   ifneq ($(NAME), )
      TARGET = $(ELF_HEAD)/$(NAME)_$(MODE)_Sq$(Sq)_Skv$(Skv)_Tm$(Tm)_Tk$(Tk)_X$(X)_Y$(Y).elf
   else
      TARGET = $(ELF_HEAD)/$(TESTCASE)_$(MODE)_Sq$(Sq)_Skv$(Skv)_Tm$(Tm)_Tk$(Tk)_X$(X)_Y$(Y).elf
   endif
endif

ifeq ($(TESTCASE), fa_2d_unroll)
   SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/fa_2d_unroll.cpp
   Sq ?= 512
   Skv ?= 512
   Tm ?= 8
   Tk ?= 16
   X ?= 1
   Y ?= 1
   DEFINES += -DTsq=$(Sq)
   DEFINES += -DTskv=$(Skv)
   DEFINES += -DTm=$(Tm)
   DEFINES += -DTk=$(Tk)
   DEFINES += -DXdim=$(X)
   DEFINES += -DYdim=$(Y)
   TARGET = $(ELF_HEAD)/$(TESTCASE)_Sq$(Sq)_Skv$(Skv)_Tm$(Tm)_Tk$(Tk)_X$(X)_Y$(Y).elf
endif

ifeq ($(TESTCASE), fa_softmax_pto)
   SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/fa_softmax_pto.cpp
   Sq ?= 512
   Skv ?= 512
   Tm ?= 8
   Tk ?= 16
   DEFINES += -DTsq=$(Sq)
   DEFINES += -DTskv=$(Skv)
   DEFINES += -DTm=$(Tm)
   DEFINES += -DTk=$(Tk)
   TARGET = $(ELF_HEAD)/$(TESTCASE)_Sq$(Sq)_Skv$(Skv)_Tm$(Tm)_Tk$(Tk).elf
endif

ifeq ($(TESTCASE), fa_2d_unroll_gmma)
   SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/fa_2d_unroll_gmma.cpp
   Sq ?= 512
   Skv ?= 512
   Tm ?= 8
   Tk ?= 16
   X ?= 1
   Y ?= 1
   DEFINES += -DTsq=$(Sq)
   DEFINES += -DTskv=$(Skv)
   DEFINES += -DTm=$(Tm)
   DEFINES += -DTk=$(Tk)
   DEFINES += -DXdim=$(X)
   DEFINES += -DYdim=$(Y)
   TARGET = $(ELF_HEAD)/$(TESTCASE)_Sq$(Sq)_Skv$(Skv)_Tm$(Tm)_Tk$(Tk)_X$(X)_Y$(Y).elf
endif

ifeq ($(TESTCASE), quant_sparse_flash_mla)
   SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/quant_sparse_flash_mla.cpp
   B ?= 1
   s1 ?= 64
   s2 ?= 128
   N1 ?= 1
   N2 ?= 1
   D ?= 512
   Tm ?= 32
   Tk ?= 32
   Td_block ?= 64
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
   IMPL ?= onepass
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

   QSMLA_UNIFIED_IMPLS := swa_tadd_4pe hca_tadd_4pe csa_tadd_4pe ori_sparse_tadd_4pe ori_cmp_sparse_tadd_4pe
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
      QSMLA_DATA_OBJ_DIR := $(abspath $(CURDIR)/../../../output/kernel/fa/qsmla_data)
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
      ifeq ($(filter $(IMPL),$(QSMLA_UNIFIED_IMPLS)),)
         $(error QSMLA_SPARSE_EMBED_INPUT requires a unified four-PE IMPL)
      endif
      QSMLA_SPARSE_DATA_ROOT ?= /tmp/qsmla-sparse
      QSMLA_SPARSE_DATA_DIR := $(QSMLA_SPARSE_DATA_ROOT)/$(QSMLA_SPARSE_CASE)
      QSMLA_SPARSE_OBJ_DIR := $(abspath $(CURDIR)/../../../output/kernel/fa/qsmla_sparse_data/$(QSMLA_SPARSE_CASE)_$(QSMLA_DTYPE))
      QSMLA_INPUT_BUILDER := $(CURDIR)/build_qsmla_input_object.py
      QSMLA_Q_BYTES := $(shell expr $(B) \* $(s1) \* $(N1) \* $(D) \* $(QSMLA_INPUT_BYTES))
      QSMLA_ORI_KV_BYTES := $(shell expr $(B) \* $(s2) \* $(N2) \* $(D) \* $(QSMLA_INPUT_BYTES))
      QSMLA_CMP_KV_BYTES := $(shell expr $(B) \* $(cmp_s2) \* $(N2) \* $(D) \* $(QSMLA_INPUT_BYTES))
      QSMLA_ORI_INDEX_BYTES := $(shell expr $(B) \* $(s1) \* $(N2) \* $(ori_topk) \* 4)
      QSMLA_CMP_INDEX_BYTES := $(shell expr $(B) \* $(s1) \* $(N2) \* $(cmp_topk) \* 4)
      QSMLA_LENGTH_BYTES := $(shell expr $(B) \* $(s1) \* $(N2) \* 4)

      QSMLA_Q_OBJ := $(QSMLA_SPARSE_OBJ_DIR)/q.o
      QSMLA_ORI_KV_OBJ := $(QSMLA_SPARSE_OBJ_DIR)/ori_kv.o
      EXTRA_OBJ_FILES += $(QSMLA_Q_OBJ) $(QSMLA_ORI_KV_OBJ)
      DEFINES += -DQSMLA_USE_SPARSE_EMBEDDED_INPUT

      ifneq ($(filter $(IMPL),hca_tadd_4pe csa_tadd_4pe ori_cmp_sparse_tadd_4pe),)
         QSMLA_CMP_KV_OBJ := $(QSMLA_SPARSE_OBJ_DIR)/cmp_kv.o
         EXTRA_OBJ_FILES += $(QSMLA_CMP_KV_OBJ)
      endif
      ifneq ($(filter $(IMPL),csa_tadd_4pe ori_cmp_sparse_tadd_4pe),)
         QSMLA_CMP_INDEX_OBJ := $(QSMLA_SPARSE_OBJ_DIR)/cmp_indices.o
         EXTRA_OBJ_FILES += $(QSMLA_CMP_INDEX_OBJ)
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

      TARGET = $(ELF_HEAD)/$(TESTCASE)_B$(B)_s1$(s1)_oriS2$(s2)_cmpS2$(cmp_s2)_N1$(N1)_N2$(N2)_D$(D)_oriK$(ori_topk)_cmpK$(cmp_topk)_ratio$(cmp_ratio)_Tm$(Tm)_Tk$(Tk)_Td$(Td_block)_IMPL$(IMPL)_DTYPE$(QSMLA_DTYPE)_INPUTembedded.elf

      $(QSMLA_Q_OBJ): $(QSMLA_SPARSE_DATA_DIR)/q.$(QSMLA_INPUT_EXT).bin $(QSMLA_INPUT_BUILDER)
	@mkdir -p $(QSMLA_SPARSE_OBJ_DIR)
	python3 $(QSMLA_INPUT_BUILDER) --compiler $(CXX) --input $< --output $@ --symbol q_$(QSMLA_INPUT_EXT) --bytes $(QSMLA_Q_BYTES) --exact-bytes

      $(QSMLA_ORI_KV_OBJ): $(QSMLA_SPARSE_DATA_DIR)/ori_kv.$(QSMLA_INPUT_EXT).bin $(QSMLA_INPUT_BUILDER)
	@mkdir -p $(QSMLA_SPARSE_OBJ_DIR)
	python3 $(QSMLA_INPUT_BUILDER) --compiler $(CXX) --input $< --output $@ --symbol ori_kv_$(QSMLA_INPUT_EXT) --bytes $(QSMLA_ORI_KV_BYTES) --exact-bytes

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
   endif
endif

include ../../common/Makefile.common
