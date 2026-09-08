# matmul_test (dynamic-shape fp16 GEMM: fp16 in / fp32 acc / fp16 out) —
# single-PE and 4-PE cooperative.
# Single-PE testcase:  matmul_test      (gfrun -t 1 -s softcore.multiThreadNum=1 -f)
# 4-PE testcase:       matmul_test_mt   (gfrun -t 1 -f; the model defaults to 4 PEs)

ifeq ($(filter $(TESTCASE),matmul_test matmul_test_mt),)
$(error Unsupported TESTCASE=$(TESTCASE); expected matmul_test or matmul_test_mt)
endif

M ?= 256
N ?= 256
K ?= 256
tM ?= 16
tN ?= 16
tK ?= 16
B ?= 1

DEFINES += -DglobM=$(M) -DglobN=$(N) -DglobK=$(K) \
	-DtilM=$(tM) -DtilN=$(tN) -DtilK=$(tK) -DBatch=$(B)

SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
TARGET = $(ELF_HEAD)_$(TESTCASE)_B$(B)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK).elf

include ../../common/Makefile.common

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)
