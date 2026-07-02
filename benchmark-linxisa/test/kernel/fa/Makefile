# fa testcases: fa_HIF4_HIF4, fa_2d_unroll

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
   X = 1
   Y = 1
   ifneq ($(X), )
      X = $(X)
   endif
   ifneq ($(Y), )
      Y = $(Y)
   endif
   DEFINES += -DTsq=$(Sq)
   DEFINES += -DTskv=$(Skv)
   DEFINES += -DTm=$(Tm)
   DEFINES += -DTk=$(Tk)
   DEFINES += -DXdim=$(X)
   DEFINES += -DYdim=$(Y)
   TARGET = $(ELF_HEAD)/$(TESTCASE)_Sq$(Sq)_Skv$(Skv)_Tm$(Tm)_Tk$(Tk)_X$(X)_Y$(Y).elf
endif

include ../../common/Makefile.common

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)
