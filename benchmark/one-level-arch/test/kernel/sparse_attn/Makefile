TESTCASE ?= sparse_attn
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/sparse_attn.cpp
TARGET = $(ELF_HEAD)_sparse_attn.elf
NUM_HEADS ?= 65
HEADS_PER_GROUP ?= 64
DEFINES += -DSPARSE_NUM_HEADS=$(NUM_HEADS) -DSPARSE_HEADS_PER_GROUP=$(HEADS_PER_GROUP)
include ../../common/Makefile.common
