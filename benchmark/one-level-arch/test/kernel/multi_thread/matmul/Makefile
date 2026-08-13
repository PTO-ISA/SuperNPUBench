# Multi-thread GMMA matmul operators.

TESTCASE ?= matmul

B ?= 1
M ?= 256
N ?= 256
K ?= 256
tM ?= 32
tN ?= 32
tK ?= 32

DTYPE_BYTES := 4
TILE_BYTE_LIMIT := 8192
TILE_SHAPE_VALID := $(shell \
	test $(tM) -gt 0 && test $(tN) -gt 0 && test $(tK) -gt 0 && \
	test $$(( $(tM) * $(tK) * $(DTYPE_BYTES) )) -lt $(TILE_BYTE_LIMIT) && \
	test $$(( $(tK) * $(tN) * $(DTYPE_BYTES) )) -lt $(TILE_BYTE_LIMIT) && \
	test $$(( $(tM) * $(tN) * $(DTYPE_BYTES) )) -lt $(TILE_BYTE_LIMIT) && \
	echo yes)

ifneq ($(TILE_SHAPE_VALID),yes)
$(error Invalid FP32 matmul tile shape: require tM*tK*4, tK*tN*4, and tM*tN*4 to each be < 8192 bytes; got tM=$(tM), tN=$(tN), tK=$(tK))
endif

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
