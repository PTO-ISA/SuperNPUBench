ifeq ($(TYPE), TAIL_OCP)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tail_ocp.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tail_ocp.elf
endif

ifeq ($(TYPE), TAIL_CUBLAS)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tail_cublas.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tail_cublas.elf
endif

ifeq ($(TYPE), TAIL_DYNAMIC_RANGE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tail_dynamic_range.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tail_dynamic_range.elf
endif

ifeq ($(TYPE), NONTAIL_OCP)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_ocp.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_nontail_ocp.elf
endif

ifeq ($(TYPE), NONTAIL_CUBLAS)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_cublas.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_nontail_cublas.elf
endif

ifeq ($(TYPE), NONTAIL_DYNAMIC_RANGE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_dynamic_range.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_nontail_dynamic_range.elf
endif

include ../../../common/Makefile.common

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)
