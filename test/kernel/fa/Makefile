ifeq ($(TESTCASE), flash_attention)
DEFINES += -DBatSz=$(B) -DglobS=$(S) -DglobD=$(D) -DtilM=$(tM) -DtilK=$(tK)
TARGET = $(ELF_HEAD)_$(TESTCASE)_$(MODE)_B$(B)_S$(S)_D$(D)_tM$(tM)_tK$(tK).elf
endif

ifeq ($(TESTCASE), fa)
    TARGET = $(ELF_HEAD)_$(TESTCASE).elf

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
    TARGET = $(ELF_HEAD)_$(TESTCASE)_OPT1.elf
    endif

    ifeq ($(opt2), on)
    DEFINES += -DOPT2
    TARGET = $(ELF_HEAD)_$(TESTCASE)_OPT2.elf
    endif

    ifeq ($(opt3), on)
    DEFINES += -DOPT3
    TARGET = $(ELF_HEAD)_$(TESTCASE)_OPT3.elf
    endif

    ifeq ($(opt4), on)
    DEFINES += -DOPT4
    TARGET = $(ELF_HEAD)_$(TESTCASE)_OPT4.elf
    endif
endif

ifeq ($(TESTCASE), fa_HIF4_HIF4)
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
    DEFINES += -D$(MODE)
	CPPFILES = $(wildcard $(TEST_ROOT)/$(CASE_SRC_DIR)/*.cpp)
	OPPATH = $(shell find src -name "*.cpp" | grep -i hif4)
	OPFILE = $(notdir $(OPPATH))
	OPNAME = $(patsubst %.cpp,%, $(OPFILE))
    ifneq ($(NAME), )
        TARGET = $(ELF_HEAD)/$(NAME)_$(MODE)_Sq$(Sq)_Skv$(Skv)_Tm$(Tm)_Tk$(Tk)_X$(X)_Y$(Y).elf
    else
        TARGET = $(ELF_HEAD)/$(TESTCASE)_$(MODE)_Sq$(Sq)_Skv$(Skv)_Tm$(Tm)_Tk$(Tk)_X$(X)_Y$(Y).elf
    endif
endif

ifneq ($(MODE), )
   DEFINES += -D$(MODE)
endif

CP = cp
DEST_DIR = ~/elf_subset/subset_fa_reuse/
COMM_SRC_FILE += $(abspath ../../common/_start.s)
COMM_SRC_DIR = $(shell dirname $(COMM_SRC_FILE))
COMM_OBJ = $(patsubst %.s, %.o, $(subst $(COMM_SRC_DIR), $(OBJ_DIR), $(COMM_SRC_FILE)))
SRC_FILE +=  $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
include ../../common/Makefile.common

.PHONY: fa_force
$(OBJ): fa_force

.PHONY: copy_target
all: copy_target

copy_target: $(TARGET)
	@mkdir -p $(DEST_DIR)
	$(CP) $(TARGET) $(DEST_DIR)
