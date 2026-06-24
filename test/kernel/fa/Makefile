# fa testcases: flash_attention, fa, fa_HIF4_HIF4

ifeq ($(TESTCASE), flash_attention)
   SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/fa_basic.cpp
   DEFINES += -DBatSz=$(B) -DglobS=$(S) -DglobD=$(D) -DtilM=$(tM) -DtilK=$(tK)
   ifneq ($(MODE), )
      DEFINES += -DMODE=\"$(MODE)\"
   else
      DEFINES += -DMODE=\"ALL\"
   endif
   TARGET = $(ELF_HEAD)/$(TESTCASE)_$(MODE)_B$(B)_S$(S)_D$(D)_tM$(tM)_tK$(tK).elf
endif

ifeq ($(TESTCASE), fa)
   SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/fa_fusion.cpp
   ifeq ($(dtype), fp16)
      DEFINES += -DFA_DTYPE=__half
   endif
   ifeq ($(dtype), fp32)
      DEFINES += -DFA_DTYPE=float
   endif
   ifeq ($(dtype), fp8)
      DEFINES += -DFA_DTYPE=__fp8_e4m3
   endif
   ifeq ($(dtype), fp4)
      DEFINES += -DFA_DTYPE=__fp4_e2m1x2
   endif
   ifneq ($(qD), )
      DEFINES += -DFA_QD=$(qD)
   endif
   ifneq ($(vD), )
      DEFINES += -DFA_VD=$(vD)
   endif
   ifeq ($(opt1), on)
      DEFINES += -DOPT1
      TARGET = $(ELF_HEAD)/$(TESTCASE)_OPT1.elf
   else ifeq ($(opt2), on)
      DEFINES += -DOPT2
      TARGET = $(ELF_HEAD)/$(TESTCASE)_OPT2.elf
   else ifeq ($(opt3), on)
      DEFINES += -DOPT3
      TARGET = $(ELF_HEAD)/$(TESTCASE)_OPT3.elf
   else ifeq ($(opt4), on)
      DEFINES += -DOPT4
      TARGET = $(ELF_HEAD)/$(TESTCASE)_OPT4.elf
   else
      TARGET = $(ELF_HEAD)/$(TESTCASE).elf
   endif
endif

ifeq ($(TESTCASE), fa_HIF4_HIF4)
   SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/fa_hif4.cpp
   Sq = 512
   Skv = 512
   ifneq ($(Sq), )
      Sq = $(Sq)
   endif
   ifneq ($(Skv), )
      Skv = $(Skv)
   endif
   Tm = 128
   Tk = 128
   X = 2
   Y = 4
   DEFINES += -DTsq=$(Sq)
   DEFINES += -DTskv=$(Skv)
   DEFINES += -DTm=$(Tm)
   DEFINES += -DTk=$(Tk)
   DEFINES += -DXdim=$(X)
   DEFINES += -DYdim=$(Y)
   ifneq ($(MODE), )
      DEFINES += -DMODE=\"$(MODE)\"
   endif
   ifneq ($(NAME), )
      TARGET = $(ELF_HEAD)/$(NAME)_$(MODE)_Sq$(Sq)_Skv$(Skv)_Tm$(Tm)_Tk$(Tk)_X$(X)_Y$(Y).elf
   else
      TARGET = $(ELF_HEAD)/$(TESTCASE)_$(MODE)_Sq$(Sq)_Skv$(Skv)_Tm$(Tm)_Tk$(Tk)_X$(X)_Y$(Y).elf
   endif
endif

include ../../common/Makefile.common

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)
