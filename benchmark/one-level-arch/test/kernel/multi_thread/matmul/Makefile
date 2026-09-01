# Multi-thread GMMA matmul operators.

TESTCASE ?= matmul_shared

B ?= 1
M ?= 256
N ?= 256
K ?= 256
tM ?= 32
tN ?= 32
tK ?= 32
GOLDEN_ARGS ?= --ones

# Element data type for the A/B input tiles. The output C tile is always FP32
# (see matmul_shared.hpp: gmC/tileC use float). Supported: float, __bf16, __half.
DTYPE ?= float
LP_MODE ?= FP8

ifeq ($(filter $(DTYPE),float __bf16 __half),)
$(error Unsupported DTYPE '$(DTYPE)'; supported: float, __bf16, __half)
endif

DEFINES += -DDTYPE=$(DTYPE)
DEFINES += -DBatch=$(B)
DEFINES += -DglobM=$(M)
DEFINES += -DglobN=$(N)
DEFINES += -DglobK=$(K)
DEFINES += -DtilM=$(tM)
DEFINES += -DtilN=$(tN)
DEFINES += -DtilK=$(tK)

ifeq ($(TESTCASE),matmul_shared)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/matmul_shared.cpp
else ifeq ($(TESTCASE),matmul_reuseB)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/matmul_reuseB.cpp
else ifeq ($(TESTCASE),matmul_lowp)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/matmul_lowp.cpp
ifeq ($(LP_MODE),FP8)
LOWP_DTYPE := __fp8_e4m3
PACKED_FACTOR := 1
USE_MX := 0
else ifeq ($(LP_MODE),HIFP8)
LOWP_DTYPE := __hif8
PACKED_FACTOR := 1
USE_MX := 0
else ifeq ($(LP_MODE),HIF4)
LOWP_DTYPE := __fp4_e1m2x2
PACKED_FACTOR := 2
USE_MX := 0
else ifeq ($(LP_MODE),HIF4X2)
LOWP_DTYPE := __fp4_e1m2x2
PACKED_FACTOR := 2
USE_MX := 0
else ifeq ($(LP_MODE),MXFP8)
LOWP_DTYPE := __fp8_e4m3
PACKED_FACTOR := 1
USE_MX := 1
else ifeq ($(LP_MODE),MXFP4)
LOWP_DTYPE := __fp4_e2m1x2
PACKED_FACTOR := 2
USE_MX := 1
else
$(error Unsupported LP_MODE=$(LP_MODE); expected FP8, HIFP8, HIF4, HIF4X2, MXFP8, or MXFP4)
endif
DEFINES += -DLOWP_DTYPE=$(LOWP_DTYPE)
DEFINES += -DPACKED_FACTOR=$(PACKED_FACTOR)
DEFINES += -DUSE_MX=$(USE_MX)
else
$(error Unsupported TESTCASE=$(TESTCASE); expected matmul_shared, matmul_reuseB, or matmul_lowp)
endif

# Keep the default (FP32) ELF name unchanged for backward compatibility;
# non-default dtypes get a _DType<dtype> suffix so binaries don't collide.
ifeq ($(TESTCASE),matmul_lowp)
TARGET = $(ELF_HEAD)_$(TESTCASE)_$(LP_MODE)_B$(B)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK).elf
else ifeq ($(DTYPE),float)
TARGET = $(ELF_HEAD)_$(TESTCASE)_B$(B)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK).elf
else
TARGET = $(ELF_HEAD)_$(TESTCASE)_B$(B)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK)_DType$(DTYPE).elf
endif

include ../../../common/Makefile.common

.PHONY: golden-check
golden-check:
	$(MAKE) -B TESTCASE=$(TESTCASE) DTYPE=$(DTYPE) B=$(B) M=$(M) N=$(N) K=$(K) \
		tM=$(tM) tN=$(tN) tK=$(tK) \
		COMPILER_DIR=$(COMPILER_DIR) \
		res_check=on all
	python3 $(TEST_ROOT)/$(CASE_SRC_DIR)/golden_cmp.py $(GOLDEN_ARGS) -d $(TARGET)
