# TMATMUL fixp microbench.
#
# One source (src/fixp_tmatmul.cpp), many binaries. Each build selects a single
# fixp mode via -DFIXP_MODE; the define name is uppercase, the ELF suffix lower.
#
# Build one variant, e.g.:
#   make FIXP_MODE=S_QF_S8          # QF322S8Pre scalar quant
#   make FIXP_MODE=ROWGROUP_MAXABS  # row_max + group_max<8> + max_abs
#   make FIXP_MODE=S_QF_S8 diss     # also dump .diss
#
# Build every variant with: bash compile.all

TESTCASE ?= fixp_tmatmul

M ?= 32
N ?= 32
K ?= 32
TM ?= 32
TN ?= 32
TK ?= 32

DEFINES += -DTM=$(TM) -DTN=$(TN) -DTK=$(TK)

FIXP_MODE ?= KEEP_ACC
MODE_LABEL := $(shell echo $(FIXP_MODE) | tr 'A-Z' 'a-z')
DEFINES += -D$(FIXP_MODE)

SRC_FILE = src/$(TESTCASE).cpp
TARGET = $(ELF_HEAD)/$(TESTCASE)_$(MODE_LABEL)_M$(M)_N$(N)_K$(K)_tM$(TM)_tN$(TN)_tK$(TK).elf

include ../Makefile.common

# fixp_tmatmul.o is mode-specific in content (-D$(FIXP_MODE)) but shares one
# .o name, so force a recompile on every build — otherwise each FIXP_MODE just
# relinks the previous mode's stale .o and emits an ELF carrying the wrong
# mode. (The one-level Makefile.common did this via `clean` as a prerequisite
# of the link rule; microbench's Makefile.common has no such prerequisite.)
.PHONY: _force_rebuild
$(OBJ): _force_rebuild