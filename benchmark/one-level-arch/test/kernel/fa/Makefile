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
   ifeq ($(IMPL), tadd)
      DEFINES += -DQSMLA_USE_TADD
   endif
   ifeq ($(IMPL), tadd_4pe)
      DEFINES += -DQSMLA_USE_TADD_4PE
      TARGET = $(ELF_HEAD)/$(TESTCASE)_B$(B)_s1$(s1)_s2$(s2)_N1$(N1)_N2$(N2)_D$(D)_Tm$(Tm)_Tk$(Tk)_Td$(Td_block)_IMPL$(IMPL).elf
   else
      TARGET = $(ELF_HEAD)/$(TESTCASE)_B$(B)_s1$(s1)_s2$(s2)_N1$(N1)_N2$(N2)_D$(D)_Tm$(Tm)_Tk$(Tk)_Td$(Td_block).elf
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
endif

include ../../common/Makefile.common
