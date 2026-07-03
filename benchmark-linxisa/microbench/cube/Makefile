# Microbench Makefile for Cube operations

TESTCASE ?= matmul_fp32_16x16

# Include common build system
include ../../test/common/Makefile.common

# Source files
SRC_FILE = src/$(TESTCASE).cpp

# Compiler flags
CC_O += -I$(ROOT)/microbench/cube
