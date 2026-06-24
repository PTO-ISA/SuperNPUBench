ifeq ($(TESTCASE), flash_attention)
DEFINES += -DBatSz=$(B) -DglobS=$(S) -DglobD=$(D) -DtilM=$(tM) -DtilK=$(tK)
TARGET = $(ELF_HEAD)_$(TESTCASE)_$(MODE)_B$(B)_S$(S)_D$(D)_tM$(tM)_tK$(tK).elf
endif

ifneq ($(MODE), )
   DEFINES += -D$(MODE)
endif

CP = cp
DEST_DIR = ~/elf_subset/subset_fa_reuse/
SRC_FILE +=  $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
include ../../common/Makefile.common

$(TARGET): clean $(LINK_SCRIPT) $(COMM_OBJ) $(OBJ) $(EXTRA_OBJ_FILES)
	@mkdir -p $(shell dirname $@)
	$(LINK) $(CC_LINK) $(OBJ) $(COMM_OBJ) $(EXTRA_OBJ_FILES) -o $@
	$(CP) $(TARGET) $(DEST_DIR)
