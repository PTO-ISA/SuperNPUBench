# Multi-thread GMMA matmul operators.

TESTCASE ?= matmul

B ?= 1
M ?= 256
N ?= 256
K ?= 256
tM ?= 32
tN ?= 32
tK ?= 32

DEFINES += -DBatch=$(B)
DEFINES += -DglobM=$(M)
DEFINES += -DglobN=$(N)
DEFINES += -DglobK=$(K)
DEFINES += -DtilM=$(tM)
DEFINES += -DtilN=$(tN)
DEFINES += -DtilK=$(tK)

ifeq ($(TESTCASE),matmul_shared)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/matmul_shared.cpp
TARGET = $(ELF_HEAD)_$(TESTCASE)_B$(B)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK).elf
else
$(error Unsupported TESTCASE=$(TESTCASE); expected matmul_shared)
endif

include ../../../common/Makefile.common
