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
endif

include ../../common/Makefile.common

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)
