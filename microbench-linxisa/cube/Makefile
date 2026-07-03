# Cube microbench Makefile

# Set source file based on TESTCASE
SRC_FILE = src/$(TESTCASE).cpp

# Set target path
TARGET = $(ELF_HEAD)/$(TESTCASE).elf

# Include common build rules
include ../Makefile.common

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
