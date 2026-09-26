TESTCASE := fa_lowp_algB
Sq ?= 256
Skv ?= 256
Tm ?= 128
Tk ?= 128
ALGO_C ?= 1
GM_FUSED ?= 0
DEFINES += -DTsq=$(Sq) -DTskv=$(Skv) -DTm=$(Tm) -DTk=$(Tk)
DEFINES += -DFA_ALGB_ALGO_C=$(ALGO_C) -DFA_ALGB_GM_FUSED=$(GM_FUSED)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/fa_lowp_algB.cpp
TARGET = $(ELF_HEAD)_C$(ALGO_C)_GM$(GM_FUSED)_Sq$(Sq)_Skv$(Skv)_Tk$(Tk)_check$(res_check).elf
include ../../../common/Makefile.common
