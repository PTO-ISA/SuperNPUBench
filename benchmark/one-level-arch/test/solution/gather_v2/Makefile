ifeq ($(TESTCASE),gather_v2)
ABS_TOL ?= 0.001f
REL_TOL ?= 0.001f

DEFINES += -DGATHER_DTYPE=$(DType) \
	-DGATHER_ITYPE=$(IType) \
	-DRANKs=$(RANK) \
	-DGATHER_DIMs=$(GATHER_DIM) \
	-DINPUT_ELEMENTSs=$(INPUT_ELEMENTS) \
	-DOUTPUT_ELEMENTSs=$(OUTPUT_ELEMENTS) \
	-DINDEX_ELEMENTSs=$(INDEX_ELEMENTS) \
	-DTILE_ELEMENTSs=$(TILE_ELEMENTS) \
	-DINPUT_SHAPEs=$(INPUT_SHAPE) \
	-DOUTPUT_SHAPEs=$(OUTPUT_SHAPE) \
	-DABS_TOLs=$(ABS_TOL) \
	-DREL_TOLs=$(REL_TOL)

TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType)_IType$(IType)_RANK$(RANK)_DIM$(GATHER_DIM)_IN$(INPUT_SHAPE_NAME)_OUT$(OUTPUT_SHAPE_NAME)_TILE$(TILE_ELEMENTS).elf
endif

SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
include ../../common/Makefile.common

# These tests use the direct-entry four-PE runtime, not hosted libc startup.
ifeq ($(PLAT),linx)
# Optional overrides support an in-development compiler outside its runtime.
ELEMENT_TARGET_FLAGS = $(if $(TARGET_TRIPLE),--target=$(TARGET_TRIPLE)) \
	$(if $(SYSROOT),--sysroot=$(SYSROOT)) \
	$(if $(RESOURCE_DIR),-resource-dir=$(RESOURCE_DIR))
INCLUDE += $(if $(API_INCLUDE),-I$(API_INCLUDE))

# The kernel's pragma loops require generic element lowering.
CC_OPTS += $(ELEMENT_TARGET_FLAGS) -mllvm -linxv5-enable-generic-element=true \
	-fno-vectorize -fno-slp-vectorize -Werror=unknown-pragmas
# The bundled compiler's BFI constant synthesis miscompiles repeated words.
CC_OPTS += -mllvm -linxv5-enable-bfi-opt=false
CC_LINK = $(ELEMENT_TARGET_FLAGS) -fuse-ld=lld -nostartfiles $(TEST_ROOT)/solution/common/start.s
endif
