ifeq ($(TYPE), HIF4_HIF4)
   CPPFILES = $(wildcard $(TEST_ROOT)/$(CASE_SRC_DIR)/*.cpp)
   OPPATH = $(shell find src -name "*.cpp" | grep -i $(TYPE))
   OPFILE = $(notdir $(OPPATH))
   OPNAME = $(patsubst %.cpp,%, $(OPFILE))
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
   # TARGET = $(OPNAME)_B$(Batch)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK).elf
   ifneq ($(NAME), )
      TARGET = $(ELF_HEAD)/$(NAME)_$(VER)_B$(Batch)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK).elf
   else
      TARGET = $(ELF_HEAD)/$(TESTCASE)_$(TYPE)_$(VER)_B$(Batch)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK).elf
   endif
endif
# TARGET = $(ELF_HEAD)_$(TESTCASE)_$(MODE)_B$(Batch)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK).elf
ifeq ($(TYPE), A16W4)
   CPPFILES = $(wildcard $(TEST_ROOT)/$(CASE_SRC_DIR)/*.cpp)
   OPPATH = $(shell find src -name "*.cpp" | grep -i $(TYPE))
   OPFILE = $(notdir $(OPPATH))
   OPNAME = $(patsubst %.cpp,%, $(OPFILE))
   DEFINES += -DglobM=$(M) -DglobN=$(N) -DglobK=$(K) -DtilM=$(tM) -DtilN=$(tN) -DtilK=$(tK)
      Batch = 1
      ifneq ($(B), )
      Batch = $(B)
      endif
   DEFINES += -DBatch=$(B)
   # TARGET = $(OPNAME)_B$(Batch)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK).elf
   TARGET = $(ELF_HEAD)/$(TESTCASE)_$(TYPE)_$(MODE)_B$(Batch)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK).elf
endif

SRC_FILE +=  $(TEST_ROOT)/$(CASE_SRC_DIR)/$(OPNAME).cpp
include ../../common/Makefile.common
# $(info "CPPFILES is $(CPPFILES)")
# $(info "CPPFILES is $(OPPATH)")
# $(info "OPPATH is $(OPNAME)")
# $(info "mode is $(MODE)")
clean:
#	rm -rf $(OBJ_ROOT)/*/*.o
	@find $(OBJ_ROOT) -type f -name "${OPNAME}.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)


CP = cp
DEST_DIR = ~/elf_subset/subset_matmul_reuse/

$(TARGET): clean $(LINK_SCRIPT) $(COMM_OBJ) $(OBJ) $(EXTRA_OBJ_FILES)
	@mkdir -p $(shell dirname $@)
	$(LINK) $(CC_LINK) $(OBJ) $(COMM_OBJ) $(EXTRA_OBJ_FILES) -o $@
	$(CP) $(TARGET) $(DEST_DIR)