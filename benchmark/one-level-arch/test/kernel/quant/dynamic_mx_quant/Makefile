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

ifeq ($(TYPE), BS32)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/bs32.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_bs32.elf
endif

ifeq ($(TYPE), BS64)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/bs64.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_bs64.elf
endif

ifeq ($(TYPE), BS256)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/bs256.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_bs256.elf
endif

ifeq ($(TYPE), BS512)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/bs512.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_bs512.elf
endif

ifeq ($(TYPE), BS512_CUBLAS)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/bs512_cublas.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_bs512_cublas.elf
endif

ifeq ($(TYPE), TM4_BS32)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tm4_bs32.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tm4_bs32.elf
endif

ifeq ($(TYPE), TM4_BS64)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tm4_bs64.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tm4_bs64.elf
endif

ifeq ($(TYPE), TM4_BS128)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tm4_bs128.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tm4_bs128.elf
endif

ifeq ($(TYPE), TM4_BS256)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tm4_bs256.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tm4_bs256.elf
endif

ifeq ($(TYPE), TM4_BS512)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tm4_bs512.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tm4_bs512.elf
endif

ifeq ($(TYPE), BS1024_TM4)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/bs1024_tm4.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_bs1024_tm4.elf
endif

ifeq ($(TYPE), BS1024)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/bs1024.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_bs1024.elf
endif

include ../../../common/Makefile.common

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)
