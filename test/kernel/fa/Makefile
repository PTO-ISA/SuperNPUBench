# fa testcases: fa_HIF4_HIF4, fa_mask

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

ifeq ($(TESTCASE), fa_mask)
   SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/fa_mask.cpp
   DEFINES += -DBatSz=$(B) -DglobS=$(S) -DglobD=$(D) -DtilM=$(tM) -DtilK=$(tK)
   ifneq ($(MODE), )
      DEFINES += -DMODE=\"$(MODE)\"
   else
      DEFINES += -DMODE=\"ALL\"
   endif
   TARGET = $(ELF_HEAD)/$(TESTCASE)_$(MODE)_B$(B)_S$(S)_D$(D)_tM$(tM)_tK$(tK).elf
endif

include ../../common/Makefile.common

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)
