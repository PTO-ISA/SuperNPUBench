TARGET = $(ELF_HEAD)_$(TESTCASE)$(SUFFIX).elf

# Override target names
ifeq ($(TESTCASE), hashtable_lookup_simt)
TARGET = $(ELF_DIR)/hashtable_lookup_simt$(SUFFIX).elf
endif

ifeq ($(TESTCASE), hashtable_lookup_simt_v2)
TARGET = $(ELF_DIR)/hashtable_lookup_simt_v2$(SUFFIX).elf
endif

ifeq ($(TESTCASE), hashtable_lookup_simd)
TARGET = $(ELF_DIR)/hashtable_lookup_simd$(SUFFIX).elf
endif

ifeq ($(TESTCASE), hkv)
TARGET = $(ELF_DIR)/hkv_lookup.elf
endif

ifeq ($(TESTCASE), simple_rdadd)
TARGET = $(ELF_DIR)/simple_rdadd$(SUFFIX).elf
endif

ifeq ($(TESTCASE), mgather_with_mask)
TARGET = $(ELF_DIR)/mgather_with_mask$(SUFFIX).elf
endif

# Source file: default is TESTCASE/TESTCASE.cpp
# hashtable_lookup_simt_v2 uses a different filename
ifeq ($(TESTCASE), hashtable_lookup_simt_v2)
SRC_FILE += $(TEST_ROOT)/$(CATEGORY)/hashtable_lookup_simt/hashtable_lookup_simt_v2.cpp
else
SRC_FILE +=  $(TEST_ROOT)/$(CATEGORY)/$(TESTCASE)/$(TESTCASE).cpp
endif

# Special handling for hashtable_lookup_simd - embed data as object files
EXTRA_OBJ_FILES :=
EXTRA_OBJ_DEPS :=

# Data object files location (relative paths)
DATA_OBJ_DIR := hashtable_lookup_simd/data_obj
OUTPUT_DATA_OBJ_DIR := ../../../output/kernel/control/hashtable_lookup_simd/data_obj

# hashtable_lookup_simd uses embedded data (large dataset for 2.55M-entry table)
ifeq ($(TESTCASE), hashtable_lookup_simd)
EXTRA_OBJ_FILES += $(OUTPUT_DATA_OBJ_DIR)/inserted_slot.o
EXTRA_OBJ_FILES += $(OUTPUT_DATA_OBJ_DIR)/lookup_keys.o
EXTRA_OBJ_FILES += $(OUTPUT_DATA_OBJ_DIR)/lookup_values.o

pre_work: build_sim_data_objs

build_sim_data_objs:
	@COMPILER_DIR="$(COMPILER_DIR)" $(DATA_OBJ_DIR)/build_data_obj.sh $(DATA_OBJ_DIR) $(OUTPUT_DATA_OBJ_DIR)

$(OUTPUT_DATA_OBJ_DIR)/%.o: $(DATA_OBJ_DIR)/%.s pre_work
	@mkdir -p $(shell dirname $@)
	$(AS) $(CC_O_ALL) $(INCLUDE) $(DEFINES) $< -o $@

endif

# hashtable_lookup_simt uses embedded data (same hashtable_lookup_simd data)
# Pre-build hashtable_lookup_simd data_objs first, then use them
ifeq ($(TESTCASE), hashtable_lookup_simt)
EXTRA_OBJ_FILES += $(OUTPUT_DATA_OBJ_DIR)/inserted_slot.o
EXTRA_OBJ_FILES += $(OUTPUT_DATA_OBJ_DIR)/lookup_keys.o
EXTRA_OBJ_FILES += $(OUTPUT_DATA_OBJ_DIR)/lookup_values.o

pre_work: build_simt_data_objs

build_simt_data_objs:
	@COMPILER_DIR="$(COMPILER_DIR)" $(DATA_OBJ_DIR)/build_data_obj.sh $(DATA_OBJ_DIR) $(OUTPUT_DATA_OBJ_DIR)

$(OUTPUT_DATA_OBJ_DIR)/%.o: $(DATA_OBJ_DIR)/%.s pre_work
	@mkdir -p $(shell dirname $@)
	$(AS) $(CC_O_ALL) $(INCLUDE) $(DEFINES) $< -o $@

endif

# hashtable_lookup_simt_v2 uses the same embedded data
ifeq ($(TESTCASE), hashtable_lookup_simt_v2)
EXTRA_OBJ_FILES += $(OUTPUT_DATA_OBJ_DIR)/inserted_slot.o
EXTRA_OBJ_FILES += $(OUTPUT_DATA_OBJ_DIR)/lookup_keys.o
EXTRA_OBJ_FILES += $(OUTPUT_DATA_OBJ_DIR)/lookup_values.o

pre_work: build_simt_v2_data_objs

build_simt_v2_data_objs:
	@COMPILER_DIR="$(COMPILER_DIR)" $(DATA_OBJ_DIR)/build_data_obj.sh $(DATA_OBJ_DIR) $(OUTPUT_DATA_OBJ_DIR)

$(OUTPUT_DATA_OBJ_DIR)/%.o: $(DATA_OBJ_DIR)/%.s pre_work
	@mkdir -p $(shell dirname $@)
	$(AS) $(CC_O_ALL) $(INCLUDE) $(DEFINES) $< -o $@

endif

# hkv uses embedded data
HKV_DATA_OBJ_DIR := hkv/data_obj
HKV_OUTPUT_DATA_OBJ_DIR := ../../../output/kernel/control/hkv/data_obj

ifeq ($(TESTCASE), hkv)
EXTRA_OBJ_FILES += $(HKV_OUTPUT_DATA_OBJ_DIR)/buckets.bin.o
EXTRA_OBJ_FILES += $(HKV_OUTPUT_DATA_OBJ_DIR)/buckets_size.bin.o
EXTRA_OBJ_FILES += $(HKV_OUTPUT_DATA_OBJ_DIR)/lookup_keys.bin.o
EXTRA_OBJ_FILES += $(HKV_OUTPUT_DATA_OBJ_DIR)/lookedup_values.bin.o
EXTRA_OBJ_FILES += $(HKV_OUTPUT_DATA_OBJ_DIR)/key_score_digest.bin.o

pre_work: build_hkv_data_objs

build_hkv_data_objs:
	@COMPILER_DIR="$(COMPILER_DIR)" $(HKV_DATA_OBJ_DIR)/build_data_obj.sh $(HKV_DATA_OBJ_DIR) $(HKV_OUTPUT_DATA_OBJ_DIR)

$(HKV_OUTPUT_DATA_OBJ_DIR)/%.o: $(HKV_DATA_OBJ_DIR)/%.s pre_work
	@mkdir -p $(shell dirname $@)
	$(AS) $(CC_O_ALL) $(INCLUDE) $(DEFINES) $< -o $@

endif

include ../../common/Makefile.common

DEFINES += $(EXTRA_DEFINES)
