ifeq ($(TESTCASE),v300_conv2d)
VERIFY ?= on
PATTERN ?= off
DEFINES += -DCONV_IN_C=$(IN_C) -DCONV_IN_H=$(IN_H) -DCONV_IN_W=$(IN_W)
DEFINES += -DCONV_OUT_C=$(OUT_C)
DEFINES += -DCONV_TILE_M=$(tM) -DCONV_TILE_N=$(tN) -DCONV_TILE_K=$(tK)
ifeq ($(VERIFY),on)
DEFINES += -DCONV2D_TMUL_VERIFY
endif
ifeq ($(PATTERN),on)
DEFINES += -DCONV_PATTERN_VERIFY
endif
SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/v300_conv2d.cpp
TARGET = $(ELF_HEAD)_$(TESTCASE)_verify$(VERIFY)_pattern$(PATTERN)_IC$(IN_C)_H$(IN_H)_W$(IN_W)_OC$(OUT_C)_tM$(tM)_tN$(tN)_tK$(tK).elf
else
$(error Unsupported TESTCASE=$(TESTCASE); expected v300_conv2d)
endif

include ../../common/Makefile.common
