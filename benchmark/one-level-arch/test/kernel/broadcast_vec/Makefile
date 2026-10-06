TESTCASE ?= broadcast_vec
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/broadcast_vec.cpp
ifeq ($(TESTCASE),broadcast_vec_2d_1334_129)
DEFINES += -DBROADCAST_VEC_2D_1334_129
TARGET = $(ELF_HEAD)_broadcast_vec_2d_1334_129_PE4.elf
else ifeq ($(TESTCASE),broadcast_vec)
TARGET = $(ELF_HEAD)_broadcast_vec_PE4.elf
else
$(error Unsupported TESTCASE=$(TESTCASE))
endif
include ../../common/Makefile.common
