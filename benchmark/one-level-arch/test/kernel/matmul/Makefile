# Multi-thread GMMA matmul operators.

TESTCASE ?= matmul_shared

B ?= 1
M ?= 256
N ?= 256
K ?= 256
tM ?= 32
tN ?= 32
tK ?= 32
group_runtime := on

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
SCALE_DTYPE := __fp8_e8m0
SCALE_GROUP := 32
else ifeq ($(LP_MODE),HIFP8)
LOWP_DTYPE := __hif8
PACKED_FACTOR := 1
USE_MX := 0
SCALE_DTYPE := __fp8_e8m0
SCALE_GROUP := 32
else ifeq ($(LP_MODE),HIF4)
LOWP_DTYPE := __fp4_hif4x2
PACKED_FACTOR := 2
USE_MX := 1
SCALE_DTYPE := uint32_t
SCALE_GROUP := 64
else ifeq ($(LP_MODE),HIF4X2)
LOWP_DTYPE := __fp4_hif4x2
PACKED_FACTOR := 2
USE_MX := 1
SCALE_DTYPE := uint32_t
SCALE_GROUP := 64
else ifeq ($(LP_MODE),MXFP8)
LOWP_DTYPE := __fp8_e4m3
PACKED_FACTOR := 1
USE_MX := 1
SCALE_DTYPE := __fp8_e8m0
SCALE_GROUP := 32
else ifeq ($(LP_MODE),MXFP4)
LOWP_DTYPE := __fp4_e2m1x2
PACKED_FACTOR := 2
USE_MX := 1
SCALE_DTYPE := __fp8_e8m0
SCALE_GROUP := 32
else
$(error Unsupported LP_MODE=$(LP_MODE); expected FP8, HIFP8, HIF4, HIF4X2, MXFP8, or MXFP4)
endif
DEFINES += -DLOWP_DTYPE=$(LOWP_DTYPE)
DEFINES += -DPACKED_FACTOR=$(PACKED_FACTOR)
DEFINES += -DUSE_MX=$(USE_MX)
DEFINES += -DSCALE_DTYPE=$(SCALE_DTYPE)
DEFINES += -DSCALE_GROUP=$(SCALE_GROUP)
else ifeq ($(TESTCASE),matmul_hif4_l1_quantize)
# Fused matmul -> first-level HiF4 quantization. Output data is packed
# HiF4X2; scale is one raw E6M2 byte per row and 64-column group.
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/matmul_hif4_l1_quantize.cpp
else
$(error Unsupported TESTCASE=$(TESTCASE); expected matmul_shared, matmul_reuseB, matmul_lowp, or matmul_hif4_l1_quantize)
endif

# Keep the default (FP32) ELF name unchanged for backward compatibility;
# non-default dtypes get a _DType<dtype> suffix so binaries don't collide.
ifeq ($(TESTCASE),matmul_lowp)
TARGET = $(ELF_HEAD)_$(TESTCASE)_$(LP_MODE)_B$(B)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK).elf
else ifeq ($(TESTCASE),matmul_hif4_l1_quantize)
TARGET = $(ELF_HEAD)_$(TESTCASE)_B$(B)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK).elf
else ifeq ($(DTYPE),float)
TARGET = $(ELF_HEAD)_$(TESTCASE)_B$(B)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK).elf
else
TARGET = $(ELF_HEAD)_$(TESTCASE)_B$(B)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK)_DType$(DTYPE).elf
endif

include ../../common/Makefile.common

.PHONY: golden-check precision-check
ifeq ($(TESTCASE),matmul_lowp)
golden-check:
	$(MAKE) -B TESTCASE=$(TESTCASE) LP_MODE=$(LP_MODE) \
		B=$(B) M=$(M) N=$(N) K=$(K) tM=$(tM) tN=$(tN) tK=$(tK) \
		COMPILER_DIR=$(COMPILER_DIR) OBJ_ROOT=$(OBJ_ROOT) \
		res_check=on all
	python3 $(TEST_ROOT)/$(CASE_SRC_DIR)/gfrun_matmul.py -d $(TARGET)
else
golden-check:
	$(MAKE) -B TESTCASE=$(TESTCASE) DTYPE=$(DTYPE) B=$(B) M=$(M) N=$(N) K=$(K) \
		tM=$(tM) tN=$(tN) tK=$(tK) \
		COMPILER_DIR=$(COMPILER_DIR) OBJ_ROOT=$(OBJ_ROOT) \
		res_check=on all
	python3 $(TEST_ROOT)/$(CASE_SRC_DIR)/gfrun_matmul.py -d $(TARGET)
endif

# Compact numerical portfolio. FP32/FP16/BF16 use K/tK=2 to exercise the
# internal-accumulator chain. MX/HiF4 use one K tile, keeping this target
# focused on payload/scale/matmul numerics (the MX accumulator-chain path is a
# separate execution-path concern). All six cases use the same host checker
# and PyTorch FP32 matmul oracle.
precision-check:
	@status=0; pass=0; fail=0; \
	for config in \
		"TESTCASE=matmul_shared DTYPE=float K=128" \
		"TESTCASE=matmul_shared DTYPE=__half K=128" \
		"TESTCASE=matmul_shared DTYPE=__bf16 K=128" \
		"TESTCASE=matmul_lowp LP_MODE=MXFP8 K=64" \
		"TESTCASE=matmul_lowp LP_MODE=MXFP4 K=64" \
		"TESTCASE=matmul_lowp LP_MODE=HIF4X2 K=64"; do \
		if $(MAKE) golden-check $$config \
			B=1 M=64 N=64 tM=64 tN=64 tK=64; then \
			pass=$$((pass + 1)); \
		else \
			fail=$$((fail + 1)); status=1; \
		fi; \
	done; \
	echo "matmul precision summary: PASS=$$pass FAIL=$$fail"; \
	exit $$status
