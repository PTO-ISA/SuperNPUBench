
ifneq ($(GROW), )
   DEFINES += -DGROW=$(GROW)
endif

ifneq ($(GCOL), )
   DEFINES += -DGCOL=$(GCOL)
endif

ifneq ($(TROW), )
   DEFINES += -DTROW=$(TROW)
endif

ifneq ($(TCOL), )
   DEFINES += -DTCOL=$(TCOL)
endif

ifneq ($(STRD), )
   DEFINES += -DSTRD=$(STRD)
endif

ifneq ($(LOOP), )
   DEFINES += -DLOOP=$(LOOP)
endif

ifneq ($(MODE), )
   DEFINES += -DMODE=\"$(MODE)\"
else
   DEFINES += -DMODE=\"gm_rd\"
endif

ifneq ($(DTYPE), )
   DEFINES += -DDTYPE=\"$(DTYPE)\"
else
   DEFINES += -DDTYPE=\"float\"
endif

ifeq ($(TESTCASE), mem)
   ifeq ($(filter gm_f%,$(MODE)),$(MODE))
      TARGET = $(ELF_HEAD)_$(TESTCASE)_$(MODE)_grow$(GROW)_gcol$(GCOL).elf
   else ifeq ($(filter gm%,$(MODE)),$(MODE))
      TARGET = $(ELF_HEAD)_$(TESTCASE)_$(MODE)_grow$(GROW)_gcol$(GCOL)_strd$(STRD).elf
   else ifeq ($(filter tile_f%,$(MODE)),$(MODE))
      TARGET = $(ELF_HEAD)_$(TESTCASE)_$(MODE)_trow$(TROW)_tcol$(TCOL).elf
   else ifeq ($(filter tile%,$(MODE)),$(MODE))
      TARGET = $(ELF_HEAD)_$(TESTCASE)_$(MODE)_trow$(TROW)_tcol$(TCOL)_strd$(STRD).elf
   else
      TARGET = $(ELF_HEAD)_$(TESTCASE)_$(MODE)_gr$(GROW)_gc$(GCOL)_tr$(TROW)_tc$(TCOL).elf
   endif
endif

SRC_FILE +=  $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
include ../common/Makefile.common