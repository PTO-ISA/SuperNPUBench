TARGET = $(ELF_HEAD)_$(TESTCASE).elf
SRC_FILE +=  $(TEST_ROOT)/$(CATEGORY)/$(TESTCASE)/$(TESTCASE).cpp

ifeq ($(opt1), on)
DEFINES += -DOPT1
TARGET = $(ELF_HEAD)_$(TESTCASE)_OPT1.elf
endif

ifeq ($(opt2), on)
DEFINES += -DOPT2
TARGET = $(ELF_HEAD)_$(TESTCASE)_OPT2.elf
endif

ifeq ($(opt3), on)
	ifeq ($(multitile), on)
        DEFINES += -DMULTI_TILE_OPT3 -DMULTI_REUSE
        TARGET = $(ELF_HEAD)_$(TESTCASE)_MULTI_TILE_OPT3.elf
        ifeq ($(multi_ldst), on)
        DEFINES += -DMULTI_LDST
        TARGET = $(ELF_HEAD)_$(TESTCASE)_MULTI_TILE_MULTI_LDST_OPT3.elf
        endif
		ifeq ($(template), on)
            DEFINES += -DTEMPLATE
            TARGET = $(ELF_HEAD)_$(TESTCASE)_MULTI_TILE_TEMPLATE_OPT3.elf
            ifeq ($(multi_ldst), on)
            TARGET = $(ELF_HEAD)_$(TESTCASE)_MULTI_TILE_MULTI_LDST_TEMPLATE_OPT3.elf
            endif
		endif
	else
        DEFINES += -DOPT3
        TARGET = $(ELF_HEAD)_$(TESTCASE)_OPT3.elf
		ifeq ($(template), on)
            DEFINES += -DTEMPLATE
            TARGET = $(ELF_HEAD)_$(TESTCASE)_TEMPLATE_OPT3.elf
		endif
	endif
endif

ifeq ($(opt4), on)
    DEFINES += -DOPT4
    TARGET = $(ELF_HEAD)_$(TESTCASE)_OPT4.elf
	ifeq ($(template), on)
    DEFINES += -DTEMPLATE
    TARGET = $(ELF_HEAD)_$(TESTCASE)_TEMPLATE_OPT4.elf
	endif
endif

ifeq ($(dcore), on)
    Sq = 512
    Skv = 512
    ifneq ($(S), )
    Sq = $(S)
    Skv = $(S)
    endif
    Tm = 128
    Tk = 128
    DEFINES += -DTsq=$(Sq)
    DEFINES += -DTskv=$(Skv)
    DEFINES += -DTm=$(Tm)
    DEFINES += -DTk=$(Tk)
    DEFINES += -DDCORE -DMULTI_REUSE
    TARGET = $(ELF_HEAD)_$(TESTCASE)_DCORE_Sq$(Sq)_Skv$(Skv)_Tm$(Tm)_Tk$(Tk).elf
	ifeq ($(multi_ldst), on)
    DEFINES += -DMULTI_LDST
    TARGET = $(ELF_HEAD)_$(TESTCASE)_MULTI_LDST_DCORE_Sq$(Sq)_Skv$(Skv)_Tm$(Tm)_Tk$(Tk).elf
	endif
endif

ifeq ($(2d_unroll), on)
    Sq = 512
    Skv = 512
    ifneq ($(S), )
    Sq = $(S)
    Skv = $(S)
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
    DEFINES += -D_2D_UNROLL
    TARGET = $(ELF_HEAD)_$(TESTCASE)_2D_UNROLL_Sq$(Sq)_Skv$(Skv)_Tm$(Tm)_Tk$(Tk)_X$(X)_Y$(Y).elf
    ifeq ($(multi_ldst), on)
    DEFINES += -DMULTI_LDST
    TARGET = $(ELF_HEAD)_$(TESTCASE)_MULTI_LDST_2D_UNROLL_Sq$(Sq)_Skv$(Skv)_Tm$(Tm)_Tk$(Tk)_X$(X)_Y$(Y).elf
    endif
endif

ifeq ($(unalign_2d_unroll), on)
    Sq = 517
    Skv = 517
    ifneq ($(S), )
    Sq = $(S)
    Skv = $(S)
    endif
    Tm = 128
    Tk = 128
    X = 1
    Y = 1
    DEFINES += -DTsq=$(Sq)
    DEFINES += -DTskv=$(Skv)
    DEFINES += -DTm=$(Tm)
    DEFINES += -DTk=$(Tk)
    DEFINES += -DXdim=$(X)
    DEFINES += -DYdim=$(Y)
    DEFINES += -D_UNALIGN_2D_UNROLL
    TARGET = $(ELF_HEAD)_$(TESTCASE)_UNALIGN_2D_UNROLL_Sq$(Sq)_Skv$(Skv)_Tm$(Tm)_Tk$(Tk)_X$(X)_Y$(Y).elf
endif

ifeq ($(template_2d_unroll), on)
    Sq = 512
    Skv = 512
    ifneq ($(S), )
    Sq = $(S)
    Skv = $(S)
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
    DEFINES += -D_TEMPLATE_2D_UNROLL
    TARGET = $(ELF_HEAD)_$(TESTCASE)_TEMPLATE_2D_UNROLL_Sq$(Sq)_Skv$(Skv)_Tm$(Tm)_Tk$(Tk)_X$(X)_Y$(Y).elf
    ifeq ($(multi_ldst), on)
    DEFINES += -DMULTI_LDST
    TARGET = $(ELF_HEAD)_$(TESTCASE)_MULTI_LDST_TEMPLATE_2D_UNROLL_Sq$(Sq)_Skv$(Skv)_Tm$(Tm)_Tk$(Tk)_X$(X)_Y$(Y).elf
	endif
endif

ifeq ($(2d_unroll_pto), on)
    Sq = 512
    Skv = 512
    ifneq ($(S), )
    Sq = $(S)
    Skv = $(S)
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
    DEFINES += -D_2D_UNROLL_PTO
    TARGET = $(ELF_HEAD)_$(TESTCASE)_2D_UNROLL_PTO_Sq$(Sq)_Skv$(Skv)_Tm$(Tm)_Tk$(Tk)_X$(X)_Y$(Y).elf
endif

ifeq ($(dynamic), on)
    Sq = 512
    Skv = 512
    ifneq ($(S), )
    Sq = $(S)
    Skv = $(S)
    endif
    Tm = 128
    Tk = 128
    X = 1
    Y = 1
    DEFINES += -DTsq=$(Sq)
    DEFINES += -DTskv=$(Skv)
    DEFINES += -DTm=$(Tm)
    DEFINES += -DTk=$(Tk)
    DEFINES += -DXdim=$(X)
    DEFINES += -DYdim=$(Y)
    DEFINES += -DFA_DYNAMIC
    TARGET = $(ELF_HEAD)_$(TESTCASE)_DYNAMIC_Sq$(Sq)_Skv$(Skv)_Tq$(Tm)_Tk$(Tk)_Q$(X)_K$(Y).elf
	ifeq ($(multi_ldst), on)
    DEFINES += -DMULTI_LDST
    TARGET = $(ELF_HEAD)_$(TESTCASE)_MULTI_LDST_DYNAMIC_Sq$(Sq)_Skv$(Skv)_Tq$(Tm)_Tk$(Tk)_Q$(X)_K$(Y).elf
	endif
endif

ifeq ($(dynamic_unroll), on)
    Sq = 512
    Skv = 512
    ifneq ($(S), )
    Sq = $(S)
    Skv = $(S)
    endif
    Tm = 128
    Tk = 128
    X = 1
    Y = 1
    DEFINES += -DTsq=$(Sq)
    DEFINES += -DTskv=$(Skv)
    DEFINES += -DTm=$(Tm)
    DEFINES += -DTk=$(Tk)
    DEFINES += -DXdim=$(X)
    DEFINES += -DYdim=$(Y)
    DEFINES += -DFA_DYNAMIC_UNROLL
    TARGET = $(ELF_HEAD)_$(TESTCASE)_DYNAMIC_UNROLL_Sq$(Sq)_Skv$(Skv)_Tq$(Tm)_Tk$(Tk)_Q$(X)_K$(Y).elf
	ifeq ($(multi_ldst), on)
    DEFINES += -DMULTI_LDST
    TARGET = $(ELF_HEAD)_$(TESTCASE)_MULTI_LDST_DYNAMIC_Sq$(Sq)_Skv$(Skv)_Tq$(Tm)_Tk$(Tk)_Q$(X)_K$(Y).elf
	endif
endif

ifeq ($(manual), on)
    Sq = 512
    Skv = 512
    ifneq ($(S), )
    Sq = $(S)
    Skv = $(S)
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
    DEFINES += -DMANUAL
    TARGET = $(ELF_HEAD)_$(TESTCASE)_MANUAL_Sq$(Sq)_Skv$(Skv)_Tm$(Tm)_Tk$(Tk)_X$(X)_Y$(Y).elf
endif

target ?= auto
prefix ?= $(CURDIR)

ifneq ($(target), auto)
    TARGET=$(prefix)/$(target)
endif

include ../../common/Makefile.common