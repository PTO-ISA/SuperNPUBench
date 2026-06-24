.DEFAULT_GOAL := all

TARGET = $(ELF_HEAD)_$(TESTCASE).elf
SRC_FILE +=  $(TEST_ROOT)/$(CATEGORY)/$(TESTCASE)/$(TESTCASE).cpp

# Special handling for hashfind - embed data as object files
EXTRA_OBJ_FILES =
EXTRA_OBJ_DEPS =

# Data object files location (relative paths)
DATA_OBJ_DIR := hashfind/data_obj
OUTPUT_DATA_OBJ_DIR = $(OBJ_ROOT)/$(CATEGORY)/hashfind/data_obj

# hashfind uses embedded data (simple_ dataset)
ifeq ($(TESTCASE), hashfind)
EXTRA_OBJ_FILES += $(OUTPUT_DATA_OBJ_DIR)/simple_inserted_slot.o
EXTRA_OBJ_FILES += $(OUTPUT_DATA_OBJ_DIR)/simple_lookup_keys.o
EXTRA_OBJ_FILES += $(OUTPUT_DATA_OBJ_DIR)/simple_lookup_values.o

pre_work: build_data_objs

build_data_objs:
	@COMPILER_DIR="$(COMPILER_DIR)" $(DATA_OBJ_DIR)/build_data_obj.sh $(DATA_OBJ_DIR) $(OUTPUT_DATA_OBJ_DIR)

endif

# hashfind_simple also uses embedded data (simple_ prefixed)
ifeq ($(TESTCASE), hashfind_simple)
EXTRA_OBJ_FILES += $(OUTPUT_DATA_OBJ_DIR)/simple_inserted_slot.o
EXTRA_OBJ_FILES += $(OUTPUT_DATA_OBJ_DIR)/simple_lookup_keys.o
EXTRA_OBJ_FILES += $(OUTPUT_DATA_OBJ_DIR)/simple_lookup_values.o

pre_work: build_data_objs

build_data_objs:
	@COMPILER_DIR="$(COMPILER_DIR)" $(DATA_OBJ_DIR)/build_data_obj.sh $(DATA_OBJ_DIR) $(OUTPUT_DATA_OBJ_DIR)

endif

include ../../common/Makefile.common

ifneq ($(EXTRA_OBJ_FILES),)
$(EXTRA_OBJ_FILES): pre_work
	@true
endif
