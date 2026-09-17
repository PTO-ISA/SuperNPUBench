# Multi-thread GMMA matmul operators.

TESTCASE ?= matmul_shared

B ?= 1
M ?= 256
N ?= 256
K ?= 256
tM ?= 32
tN ?= 32
tK ?= 32
ChainK ?= 32
BLOCKM ?= 1
BLOCKN ?= 1
GOLDEN_ARGS ?= --ones
group_runtime := on

# Element data type for the A/B input tiles. The output C tile is always FP32
# (see matmul_shared.hpp: gmC/tileC use float). Supported: float, __bf16, __half.
DTYPE ?= float
LP_MODE ?= FP8

# matmul_quantize: fused matmul -> block quantize -> (data + scale).
# QUANT_MODE selects the quantized data dtype; USE_ASSEMBLE toggles the
# TASSEMBLY (B.ASSEMBLE) pack path (compile/disasm only -- gfsim does not
# model B.ASSEMBLE, so functional runs use USE_ASSEMBLE=0).
QUANT_MODE ?= FP8
USE_ASSEMBLE ?= 1

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
DEFINES += -DCHAIN_K=$(ChainK)
DEFINES += -DblkM=$(BLOCKM)
DEFINES += -DblkN=$(BLOCKN)

ifeq ($(TESTCASE),matmul_shared)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/matmul_shared.cpp
else ifeq ($(TESTCASE),matmul_reuseB)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/matmul_reuseB.cpp
else ifeq ($(TESTCASE),matmul_blockM)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/matmul_blockM.cpp
else ifneq ($(filter matmul_lowp%,$(TESTCASE)),)
# Low-precision variants: matmul_lowp and matmul_lowp_blockM share the
# same LP_MODE routing; only the source file differs.
ifeq ($(TESTCASE),matmul_lowp)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/matmul_lowp.cpp
else
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/matmul_lowp_blockM.cpp
endif
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
else ifeq ($(TESTCASE),matmul_kchains)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/matmul_kchains.cpp
else ifeq ($(TESTCASE),matmul_quantize)
# Fused matmul -> block quantize -> (data + scale). QUANT_MODE picks the
# quantized data dtype (the per-row scale is always E8M0, group 32);
# USE_ASSEMBLE=1 enables the TASSEMBLY (B.ASSEMBLE) pack path.
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/matmul_quantize.cpp
ifeq ($(QUANT_MODE),FP8)
QUANT_OUT_DTYPE := __fp8_e4m3
else ifeq ($(QUANT_MODE),FP4)
QUANT_OUT_DTYPE := __fp4_e2m1x2
else
$(error Unsupported QUANT_MODE=$(QUANT_MODE); expected FP8 or FP4)
endif
DEFINES += -DOUT_DTYPE=$(QUANT_OUT_DTYPE)
DEFINES += -DUSE_ASSEMBLE=$(USE_ASSEMBLE)
else ifeq ($(TESTCASE),matmul_hif4_l1_quantize)
# Fused matmul -> first-level HiF4 quantization. Output data is packed
# HiF4X2; scale is one raw E6M2 byte per row and 64-column group.
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/matmul_hif4_l1_quantize.cpp
else
$(error Unsupported TESTCASE=$(TESTCASE); expected matmul_shared, matmul_reuseB, matmul_blockM, matmul_lowp, matmul_lowp_blockM, matmul_kchains, matmul_quantize, or matmul_hif4_l1_quantize)
endif

# Keep the default (FP32) ELF name unchanged for backward compatibility;
# non-default dtypes get a _DType<dtype> suffix so binaries don't collide.
# Block-tiled variants append _bM<blockM> so different batch sizes don't
# collide.
ifneq ($(filter matmul_lowp%,$(TESTCASE)),)
ifeq ($(TESTCASE),matmul_lowp_blockM)
TARGET = $(ELF_HEAD)_$(TESTCASE)_$(LP_MODE)_B$(B)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK)_bM$(BLOCKM)_bN$(BLOCKN).elf
else
TARGET = $(ELF_HEAD)_$(TESTCASE)_$(LP_MODE)_B$(B)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK).elf
endif
else ifeq ($(TESTCASE),matmul_blockM)
ifeq ($(DTYPE),float)
TARGET = $(ELF_HEAD)_$(TESTCASE)_B$(B)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK)_bM$(BLOCKM)_bN$(BLOCKN).elf
else
TARGET = $(ELF_HEAD)_$(TESTCASE)_B$(B)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK)_DType$(DTYPE)_bM$(BLOCKM)_bN$(BLOCKN).elf
endif
else ifeq ($(TESTCASE),matmul_kchains)
ifeq ($(DTYPE),float)
TARGET = $(ELF_HEAD)_$(TESTCASE)_B$(B)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_ChainK$(ChainK).elf
else
TARGET = $(ELF_HEAD)_$(TESTCASE)_B$(B)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_ChainK$(ChainK)_DType$(DTYPE).elf
endif
else ifeq ($(TESTCASE),matmul_quantize)
TARGET = $(ELF_HEAD)_$(TESTCASE)_$(QUANT_MODE)_ASM$(USE_ASSEMBLE)_B$(B)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK).elf
else ifeq ($(TESTCASE),matmul_hif4_l1_quantize)
TARGET = $(ELF_HEAD)_$(TESTCASE)_B$(B)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK).elf
else ifeq ($(DTYPE),float)
TARGET = $(ELF_HEAD)_$(TESTCASE)_B$(B)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK).elf
else
TARGET = $(ELF_HEAD)_$(TESTCASE)_B$(B)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK)_DType$(DTYPE).elf
endif

include ../../common/Makefile.common

.PHONY: golden-check
ifneq ($(filter matmul_lowp%,$(TESTCASE)),)
golden-check:
	$(MAKE) -B TESTCASE=$(TESTCASE) LP_MODE=$(LP_MODE) \
		B=$(B) M=$(M) N=$(N) K=$(K) tM=$(tM) tN=$(tN) tK=$(tK) \
		BLOCKM=$(BLOCKM) BLOCKN=$(BLOCKN) \
		COMPILER_DIR=$(COMPILER_DIR) OBJ_ROOT=$(OBJ_ROOT) \
		res_check=on all
	python3 $(TEST_ROOT)/$(CASE_SRC_DIR)/gfrun_matmul.py -d $(TARGET)
else ifeq ($(TESTCASE),matmul_blockM)
golden-check:
	$(MAKE) -B TESTCASE=$(TESTCASE) DTYPE=$(DTYPE) B=$(B) M=$(M) N=$(N) K=$(K) \
		tM=$(tM) tN=$(tN) tK=$(tK) \
		BLOCKM=$(BLOCKM) BLOCKN=$(BLOCKN) \
		COMPILER_DIR=$(COMPILER_DIR) OBJ_ROOT=$(OBJ_ROOT) \
		res_check=on all
	python3 $(TEST_ROOT)/$(CASE_SRC_DIR)/golden_cmp.py $(GOLDEN_ARGS) -d $(TARGET)
else
golden-check:
	$(MAKE) -B TESTCASE=$(TESTCASE) DTYPE=$(DTYPE) B=$(B) M=$(M) N=$(N) K=$(K) \
		tM=$(tM) tN=$(tN) tK=$(tK) \
		COMPILER_DIR=$(COMPILER_DIR) OBJ_ROOT=$(OBJ_ROOT) \
		res_check=on all
	python3 $(TEST_ROOT)/$(CASE_SRC_DIR)/golden_cmp.py $(GOLDEN_ARGS) -d $(TARGET)
endif
