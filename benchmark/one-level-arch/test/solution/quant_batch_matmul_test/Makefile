# quant_batch_matmul_test — e2m1x2 / hif4x2 fp4 MX multi-block matmul 测试
#
# TESTCASE 取值:
#   quant_batch_matmul_test_mxfp4     — e2m1x2 fp4 + e8m0 scale (group-32), single-PE per block
#   quant_batch_matmul_test_mxfp4_mt  — e2m1x2 fp4 + e8m0 scale (group-32), 4-PE cooperative
#   quant_batch_matmul_test_hif4      — hif4x2 fp4 + U32 scale (group-64), 4-PE cooperative
#
# 使用方式:
#   make TESTCASE=quant_batch_matmul_test_mxfp4 M=8192 N=4096 K=1600 ...
#   make TESTCASE=quant_batch_matmul_test_mxfp4_mt M=256 N=256 K=256 tM=64 tN=32 tK=64 ...
#   make TESTCASE=quant_batch_matmul_test_hif4  M=256 N=256 K=256 tM=64 tN=32 tK=64 ...

ifeq ($(filter $(TESTCASE),quant_batch_matmul_test_mxfp4 quant_batch_matmul_test_mxfp4_mt quant_batch_matmul_test_hif4),)
$(error Unsupported TESTCASE=$(TESTCASE); expected mxfp4, mxfp4_mt or hif4)
endif

# hif4 uses 4-PE cooperative mode (tM=64/128, tK multiple of 64)
ifeq ($(TESTCASE),quant_batch_matmul_test_hif4)
M ?= 256
N ?= 256
K ?= 256
tM ?= 64
tN ?= 32
tK ?= 64
else
M ?= 8192
N ?= 4096
K ?= 1600
tM ?= 32
tN ?= 32
tK ?= 64
endif
B ?= 1
NBLOCKS ?= 32

DEFINES += -DglobM=$(M) -DglobN=$(N) -DglobK=$(K) \
	-DtilM=$(tM) -DtilN=$(tN) -DtilK=$(tK) \
	-DBatch=$(B) -DNBLOCKS=$(NBLOCKS)

SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
TARGET = $(ELF_HEAD)_$(TESTCASE)_B$(B)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK).elf

EXTRA_INCLUDE = -I$(ROOT)/../two-level-arch/include
# Prioritize toolchain tileop-api headers (newer) over the bundled two-level-arch copies
TOOLCHAIN_TILEOP_API := $(shell $(COMPILER_DIR)/clang -print-resource-dir 2>/dev/null)/include/tileop-api
ifneq ($(wildcard $(TOOLCHAIN_TILEOP_API)),)
EXTRA_INCLUDE := -I$(TOOLCHAIN_TILEOP_API) $(EXTRA_INCLUDE)
endif
CC_OPTS += $(EXTRA_INCLUDE)

include ../../common/Makefile.common

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)
