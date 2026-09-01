# Multi-thread GMMA FlashAttention operator.

TESTCASE ?= fa_2d_unroll_gmma

Sq ?= 512
Skv ?= 512
Tm ?= 16
Tk ?= 16
QD ?= 128
VD ?= 128
X_dim ?= 1
Y_dim ?= 1

# Matrix-input precision (Q/K/V) and vector precision (softmax, P, online
# accumulation and O) are selected independently through FA_MODE. Both
# TMATMUL destinations remain FP32.
FA_MODE ?= FP32_VECFP32

ifeq ($(FA_MODE),FP32_VECFP32)
MATRIX_DTYPE := float
VECTOR_DTYPE := float
PACKED_FACTOR := 1
USE_MX := 0
else ifeq ($(FA_MODE),BF16_VECFP32)
MATRIX_DTYPE := __bf16
VECTOR_DTYPE := float
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
MATRIX_DTYPE := __fp4_e1m2x2
VECTOR_DTYPE := __bf16
PACKED_FACTOR := 2
USE_MX := 0
else
$(error Unsupported FA_MODE=$(FA_MODE); expected FP32_VECFP32, BF16_VECFP32, MXFP8_VECFP32, MXFP4_VECBF16, HIF8_VECFP32, FP8_VECFP32, FP8_VECBF16, or HIF4_VECBF16)
endif

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
DEFINES += -DFA_X_DIM=$(X_dim)
DEFINES += -DFA_Y_DIM=$(Y_dim)

ifeq ($(TESTCASE),fa_2d_unroll_gmma)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/fa_2d_unroll_gmma.cpp
else
$(error Unsupported TESTCASE=$(TESTCASE); expected fa_2d_unroll_gmma)
endif

ifeq ($(QD)_$(VD),128_128)
TARGET = $(ELF_HEAD)_Sq$(Sq)_Skv$(Skv)_Tm$(Tm)_Tk$(Tk)_X$(X_dim)_Y$(Y_dim)_$(FA_MODE).elf
else
TARGET = $(ELF_HEAD)_Sq$(Sq)_Skv$(Skv)_Tm$(Tm)_Tk$(Tk)_qD$(QD)_vD$(VD)_X$(X_dim)_Y$(Y_dim)_$(FA_MODE).elf
endif

include ../../../common/Makefile.common
