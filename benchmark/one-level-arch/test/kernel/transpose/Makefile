# Use the same shape family as compile.all.u16; case 2 is [8,4096,3].
TESTCASE ?= transpose_u16_shapes

ifeq ($(TESTCASE),transpose)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/transpose.cpp
TARGET = $(ELF_HEAD)_transpose_U16_PE4.elf
else ifeq ($(TESTCASE),transpose_u16_shapes)
SHAPE_CASE ?= 2
TRANSPOSE_GATHER ?= 0
TRANSPOSE_CONTIGUOUS ?= 0
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/transpose_u16_shapes.cpp
TARGET = $(ELF_HEAD)_transpose_case$(SHAPE_CASE)_U16_G$(TRANSPOSE_GATHER)_PE4$(if $(filter 1,$(TRANSPOSE_CHECK)),_check,).elf
DEFINES += -DTRANSPOSE_SHAPE_CASE=$(SHAPE_CASE) -DTRANSPOSE_GATHER=$(TRANSPOSE_GATHER) -DTRANSPOSE_CONTIGUOUS=$(TRANSPOSE_CONTIGUOUS)
ifeq ($(TRANSPOSE_CHECK),1)
DEFINES += -DTRANSPOSE_CHECK
endif
else ifeq ($(TESTCASE),transpose_gather_plane)
GENERIC_CASE ?= 0
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/transpose_gather_plane.cpp
TARGET = $(ELF_HEAD)_transpose_gather_plane_case$(GENERIC_CASE)_U16_PE4$(if $(filter 1,$(TRANSPOSE_CHECK)),_check,).elf
DEFINES += -DTRANSPOSE_GENERIC_CASE=$(GENERIC_CASE)
ifeq ($(TRANSPOSE_CHECK),1)
DEFINES += -DTRANSPOSE_CHECK
endif
else ifeq ($(TESTCASE),transpose_tload)
GENERIC_CASE ?= 0
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/transpose_tload.cpp
TARGET = $(ELF_HEAD)_transpose_tload_case$(GENERIC_CASE)_U16_PE4$(if $(filter 1,$(TRANSPOSE_CHECK)),_check,).elf
DEFINES += -DTRANSPOSE_GENERIC_CASE=$(GENERIC_CASE)
ifeq ($(TRANSPOSE_CHECK),1)
DEFINES += -DTRANSPOSE_CHECK
endif
else
$(error Unsupported TESTCASE=$(TESTCASE); expected transpose, transpose_u16_shapes, transpose_gather_plane or transpose_tload)
endif

# Preserve existing compile.all check mode, using the repository file-I/O path.
ifeq ($(TRANSPOSE_CHECK),1)
res_check = on
endif

include ../../common/Makefile.common
