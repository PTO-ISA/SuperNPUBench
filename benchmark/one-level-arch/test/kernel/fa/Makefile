# Multi-thread GMMA FlashAttention operator.

TESTCASE ?= fa_2d_unroll_gmma

Sq ?= 512
Skv ?= 512
Tm ?= 16
Tk ?= 16
QD ?= 128
VD ?= 128
MaxSq ?= 256
MaxSkv ?= 256
X_dim ?= 1
Y_dim ?= 1

# Matrix-input precision (Q/K/V) and vector precision (softmax, P, online
# accumulation and O) are selected independently through FA_MODE. Both
# TMATMUL destinations remain FP32.
FA_MODE ?= FP32_VECFP32
FA_TRACE_ONLY ?= 0

ifeq ($(FA_TRACE_ONLY),1)
DEFINES += -DFA_TRACE_ONLY=1
endif

ifeq ($(FA_MODE),FP32_VECFP32)
MATRIX_DTYPE := float
VECTOR_DTYPE := float
PACKED_FACTOR := 1
USE_MX := 0
else ifeq ($(FA_MODE),FP32_VECBF16)
MATRIX_DTYPE := float
VECTOR_DTYPE := __bf16
PACKED_FACTOR := 1
USE_MX := 0
else ifeq ($(FA_MODE),BF16_VECFP32)
MATRIX_DTYPE := __bf16
VECTOR_DTYPE := float
PACKED_FACTOR := 1
USE_MX := 0
else ifeq ($(FA_MODE),BF16_VECBF16)
MATRIX_DTYPE := __bf16
VECTOR_DTYPE := __bf16
PACKED_FACTOR := 1
USE_MX := 0
else ifeq ($(FA_MODE),MXFP8_VECFP32)
MATRIX_DTYPE := __fp8_e4m3
VECTOR_DTYPE := float
PACKED_FACTOR := 1
USE_MX := 1
else ifeq ($(FA_MODE),MXFP4_VECBF16)
MATRIX_DTYPE := __fp4_e2m1x2
VECTOR_DTYPE := __bf16
PACKED_FACTOR := 2
USE_MX := 1
else ifeq ($(FA_MODE),HIF8_VECFP32)
MATRIX_DTYPE := __hif8
VECTOR_DTYPE := float
PACKED_FACTOR := 1
USE_MX := 0
else ifeq ($(FA_MODE),FP8_VECFP32)
MATRIX_DTYPE := __fp8_e4m3
VECTOR_DTYPE := float
PACKED_FACTOR := 1
USE_MX := 0
else ifeq ($(FA_MODE),FP8_VECBF16)
MATRIX_DTYPE := __fp8_e4m3
VECTOR_DTYPE := __bf16
PACKED_FACTOR := 1
USE_MX := 0
else ifeq ($(FA_MODE),HIF4_VECBF16)
MATRIX_DTYPE := __fp4_hif4x2
VECTOR_DTYPE := __bf16
PACKED_FACTOR := 2
USE_MX := 0
else
$(error Unsupported FA_MODE=$(FA_MODE); expected FP32_VECFP32, FP32_VECBF16, BF16_VECFP32, BF16_VECBF16, MXFP8_VECFP32, MXFP4_VECBF16, HIF8_VECFP32, FP8_VECFP32, FP8_VECBF16, or HIF4_VECBF16)
endif

# Spell out both execution data types in artifact names. FA_MODE keeps its
# compact command-line spelling (for example FP32_VECFP32), while generated
# ELF/diss/trace names use CubeFP32_VectorFP32 so the two fields are explicit.
FA_MODE_PARTS := $(subst _, ,$(FA_MODE))
CUBE_DTYPE_TAG := $(word 1,$(FA_MODE_PARTS))
VECTOR_DTYPE_TAG := $(patsubst VEC%,%,$(word 2,$(FA_MODE_PARTS)))
DTYPE_SUFFIX := _Cube$(CUBE_DTYPE_TAG)_Vector$(VECTOR_DTYPE_TAG)

DEFINES += -DMATRIX_DTYPE=$(MATRIX_DTYPE)
DEFINES += -DVECTOR_DTYPE=$(VECTOR_DTYPE)
DEFINES += -DPACKED_FACTOR=$(PACKED_FACTOR)
DEFINES += -DUSE_MX=$(USE_MX)
DEFINES += -DTsq=$(Sq)
DEFINES += -DTskv=$(Skv)
DEFINES += -DTm=$(Tm)
DEFINES += -DTk=$(Tk)
DEFINES += -DFA_QD=$(QD)
DEFINES += -DFA_VD=$(VD)
DEFINES += -DFA_MAX_SQ=$(MaxSq)
DEFINES += -DFA_MAX_SKV=$(MaxSkv)
DEFINES += -DFA_X_DIM=$(X_dim)
DEFINES += -DFA_Y_DIM=$(Y_dim)

ifeq ($(TESTCASE),fa_2d_unroll_gmma)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/fa_2d_unroll_gmma.cpp
else ifeq ($(TESTCASE),fa_gmma_kchains)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/fa_gmma_kchains.cpp
else ifeq ($(TESTCASE),fa_gmma_opt)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/fa_gmma_opt.cpp
else ifeq ($(TESTCASE),fa_gmma_dynamic)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/fa_gmma_dynamic.cpp
else ifeq ($(TESTCASE),fa_2d_unroll_gmma_no_tstore)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/fa_2d_unroll_gmma_no_tstore.cpp
else ifeq ($(TESTCASE),fa_fixpipe)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/fa_fixpipe.cpp
else ifeq ($(TESTCASE),fa_lowp)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/fa_lowp.cpp
else ifeq ($(TESTCASE),fa_lowp_recip)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/fa_lowp_recip.cpp
DEFINES += -DFA_LOWP_RECIP=1
else
$(error Unsupported TESTCASE=$(TESTCASE); expected fa_2d_unroll_gmma, fa_gmma_opt, fa_gmma_dynamic, fa_gmma_kchains, fa_2d_unroll_gmma_no_tstore, fa_fixpipe, fa_lowp, or fa_lowp_recip)
endif

ifeq ($(TESTCASE),fa_gmma_dynamic)
FA_DYNAMIC_MODES := FP32_VECFP32 FP32_VECBF16 BF16_VECFP32 BF16_VECBF16 FP8_VECFP32 FP8_VECBF16
ifeq ($(filter $(FA_MODE),$(FA_DYNAMIC_MODES)),)
$(error fa_gmma_dynamic supports FA_MODE=$(FA_DYNAMIC_MODES))
endif
DYNAMIC_SUFFIX = _MaxSq$(MaxSq)_MaxSkv$(MaxSkv)
endif

ifeq ($(QD)_$(VD),128_128)
TARGET = $(ELF_HEAD)_$(TESTCASE)_Sq$(Sq)_Skv$(Skv)$(DYNAMIC_SUFFIX)_Tm$(Tm)_Tk$(Tk)_X$(X_dim)_Y$(Y_dim)$(DTYPE_SUFFIX).elf
else
TARGET = $(ELF_HEAD)_$(TESTCASE)_Sq$(Sq)_Skv$(Skv)$(DYNAMIC_SUFFIX)_Tm$(Tm)_Tk$(Tk)_qD$(QD)_vD$(VD)_X$(X_dim)_Y$(Y_dim)$(DTYPE_SUFFIX).elf
endif

include ../../common/Makefile.common
