# Matmul test Makefile
#
# Supported TYPE values:
#   - HIF4_HIF4: FP4 x FP4 matmul with microscaling (uses HiF4_HiF4.cpp)
#   - A16W4: BF16 x FP4 mixed precision (uses A16W4.cpp)
#   - MASK: Generic matmul with mask (uses matmul.cpp)
#
# For TYPE=MASK, use MODE to select variant:
#   - VEC, FRAC
#   - MASK_FP32, MASK_FP32_REUSEA, MASK_FP32_REUSEB, MASK_FP32_REUSEAB
#   - MASK_FP32_DYNAMIC, MASK_FP32_DYNAMIC_REUSE
#   - MASK_FP16, MASK_FP16_REUSEA, MASK_FP16_REUSEA_OPT, MASK_FP16_REUSEB
#   - MASK_FP8, MASK_FP8_2LVL, MASK_FP8_REUSEA, MASK_FP8_REUSEB
#   - MX_FP8

ifeq ($(TYPE), HIF4_HIF4)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/HiF4_HiF4.cpp
   DEFINES += -DglobM=$(M) -DglobN=$(N) -DglobK=$(K) -DtilM=$(tM) -DtilN=$(tN) -DtilK=$(tK)
   Batch = 1
   ifneq ($(B), )
      Batch = $(B)
      DEFINES += -DBatch=$(B)
   endif
   ifneq ($(VER), )
      DEFINES += -D$(VER)
   else
      VER=BASE
      DEFINES += -D$(VER)
   endif
   ifneq ($(NAME), )
      TARGET = $(ELF_HEAD)/$(NAME)_$(VER)_B$(Batch)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK).elf
   else
      TARGET = $(ELF_HEAD)/$(TESTCASE)_$(TYPE)_$(VER)_B$(Batch)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK).elf
   endif
endif

ifeq ($(TYPE), A16W4)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/A16W4.cpp
   DEFINES += -DglobM=$(M) -DglobN=$(N) -DglobK=$(K) -DtilM=$(tM) -DtilN=$(tN) -DtilK=$(tK)
   Batch = 1
   ifneq ($(B), )
      Batch = $(B)
      DEFINES += -DBatch=$(B)
   endif
   TARGET = $(ELF_HEAD)/$(TESTCASE)_$(TYPE)_B$(Batch)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK).elf
endif

ifeq ($(TYPE), MASK)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/matmul.cpp
   DEFINES += -DglobM=$(M) -DglobN=$(N) -DglobK=$(K) -DtilM=$(tM) -DtilN=$(tN) -DtilK=$(tK)
   ifneq ($(MODE), )
      DEFINES += -D$(MODE)
   endif
   ifneq ($(NAME), )
      TARGET = $(ELF_HEAD)/$(NAME)_$(MODE)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK).elf
   else
      TARGET = $(ELF_HEAD)/$(TESTCASE)_$(TYPE)_$(MODE)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK).elf
   endif
endif

include ../../common/Makefile.common

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)
