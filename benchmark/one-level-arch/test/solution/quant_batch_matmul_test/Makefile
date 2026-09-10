# quant_batch_matmul_test — e2m1x2 / hif4x2 fp4 MX multi-block matmul 测试
#
# TESTCASE 取值:
#   quant_batch_matmul_test_mxfp4 — e2m1x2 fp4 + e8m0 scale (group-32)
#   quant_batch_matmul_test_hif4  — hif4x2 fp4 + U32 scale (group-64)
#
# 使用方式:
#   make TESTCASE=quant_batch_matmul_test_mxfp4 M=8192 N=4096 K=1600 ...
#   make TESTCASE=quant_batch_matmul_test_hif4  M=8192 N=4096 K=1600 ...

ifeq ($(filter $(TESTCASE),quant_batch_matmul_test_mxfp4 quant_batch_matmul_test_hif4),)
$(error Unsupported TESTCASE=$(TESTCASE); expected mxfp4 or hif4)
endif

M ?= 8192
N ?= 4096
K ?= 1600
tM ?= 32
tN ?= 32
tK ?= 64
B ?= 1
NBLOCKS ?= 32

DEFINES += -DglobM=$(M) -DglobN=$(N) -DglobK=$(K) \
	-DtilM=$(tM) -DtilN=$(tN) -DtilK=$(tK) \
	-DBatch=$(B) -DNBLOCKS=$(NBLOCKS)

SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
TARGET = $(ELF_HEAD)_$(TESTCASE)_B$(B)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK).elf

EXTRA_INCLUDE = -I$(ROOT)/../two-level-arch/include
CC_OPTS += $(EXTRA_INCLUDE)

include ../../common/Makefile.common

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)
