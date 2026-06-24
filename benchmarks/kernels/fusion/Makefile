
ifeq ($(TESTCASE), fa_HIF4_HIF4)
    Sq = 512
    Skv = 512
    ifneq ($(Sq), )
    Sq = $(Sq)
    endif
    ifneq ($(Skv), )
    Skv = $(Skv)
    endif
    Tm = 128
    Tk = 128
    X = 2
    Y = 4
    DEFINES += -DTsq=$(Sq)
    DEFINES += -DTskv=$(Skv)
    DEFINES += -DTm=$(Tm)
    DEFINES += -DTk=$(Tk)
    DEFINES += -DXdim=$(X)
    DEFINES += -DYdim=$(Y)
    DEFINES += -D$(MODE)
	CPPFILES = $(wildcard $(TEST_ROOT)/$(CASE_SRC_DIR)/*.cpp)
	OPPATH = $(shell find src -name "*.cpp" | grep -i hif4)
	OPFILE = $(notdir $(OPPATH))
	OPNAME = $(patsubst %.cpp,%, $(OPFILE))
    # TARGET = $(ELF_HEAD)/$(TESTCASE)_$(MODE)_Sq$(Sq)_Skv$(Skv)_Tm$(Tm)_Tk$(Tk)_X$(X)_Y$(Y).elf
    ifneq ($(NAME), )
        TARGET = $(ELF_HEAD)/$(NAME)_$(MODE)_Sq$(Sq)_Skv$(Skv)_Tm$(Tm)_Tk$(Tk)_X$(X)_Y$(Y).elf
    else
        TARGET = $(ELF_HEAD)/$(TESTCASE)_$(MODE)_Sq$(Sq)_Skv$(Skv)_Tm$(Tm)_Tk$(Tk)_X$(X)_Y$(Y).elf
    endif
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