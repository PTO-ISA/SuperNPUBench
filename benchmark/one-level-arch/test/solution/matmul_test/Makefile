# matmul_test — 4-PE cooperative fp16 GEMM (A Local CellReg / B Shared).

ifeq ($(TESTCASE),matmul_test)
else
$(error Unsupported TESTCASE=$(TESTCASE); expected matmul_test)
endif

M ?= 256
N ?= 256
K ?= 256
tM ?= 128
tN ?= 64
tK ?= 64
B ?= 1

DEFINES += -DglobM=$(M) -DglobN=$(N) -DglobK=$(K) \
	-DtilM=$(tM) -DtilN=$(tN) -DtilK=$(tK) -DBatch=$(B)

EXTRA_DEFINES ?=
DEFINES += $(EXTRA_DEFINES)

SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
TARGET = $(ELF_HEAD)_$(TESTCASE)_B$(B)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK).elf

include ../../common/Makefile.common

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)
