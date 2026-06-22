.DEFAULT_GOAL := all

TARGET = $(ELF_HEAD)_$(TESTCASE).elf

# Override target name for topk
ifeq ($(TESTCASE), topk)
TARGET = $(ELF_DIR)/topk.elf
endif
SRC_FILE +=  $(TEST_ROOT)/$(CATEGORY)/$(TESTCASE)/$(TESTCASE).cpp

# Special handling for topk - embed data as object files
EXTRA_OBJ_FILES :=
EXTRA_OBJ_DEPS :=
DATA_OBJ_DIR := topk/data_obj
OUTPUT_DATA_OBJ_DIR = $(OBJ_ROOT)/$(CATEGORY)/topk/data_obj

ifeq ($(TESTCASE), topk)
EXTRA_OBJ_FILES += $(OUTPUT_DATA_OBJ_DIR)/input_131072.o
EXTRA_OBJ_FILES += $(OUTPUT_DATA_OBJ_DIR)/top_2048_out.o

pre_work: build_data_objs

build_data_objs:
	@COMPILER_DIR="$(COMPILER_DIR)" $(DATA_OBJ_DIR)/build_data_obj.sh $(DATA_OBJ_DIR) $(OUTPUT_DATA_OBJ_DIR)

$(EXTRA_OBJ_FILES): pre_work
endif

ifeq ($(opt), on)
DEFINES += -DOPT
TARGET = $(ELF_HEAD)_$(TESTCASE)_OPT.elf
endif

include ../../common/Makefile.common
