# fa testcases: fa_HIF4_HIF4, fa_2d_unroll, sfa

ifeq ($(TESTCASE), sfa)
   SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/sfa.cpp
   Sq ?= 256
   Skv ?= 512
   Tm ?= 16
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
   Tm ?= 16
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
   Tm ?= 16
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
   Tm ?= 16
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

include ../../common/Makefile.common

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)
