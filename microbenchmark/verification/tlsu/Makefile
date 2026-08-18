SRC_FILE = src/$(TESTCASE).cpp
TARGET = $(ELF_HEAD)/$(TESTCASE).elf
# 本目录嵌在 verification/ 下一层，Makefile.common 的 -I$(MICROBENCH_ROOT)/$(CATEGORY)
# 只到 microbenchmark/verification，补上本目录以便 src/*.cpp 找到 tlsu_bench.hpp
INCLUDE += -I$(CURDIR)
include ../../Makefile.common
