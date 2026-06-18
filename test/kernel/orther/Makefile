ifeq ($(TESTCASE), matmul)
DEFINES += -DglobM=$(M) -DglobN=$(N) -DglobK=$(K) -DtilM=$(tM) -DtilN=$(tN) -DtilK=$(tK)
   Batch = 1
   ifneq ($(B), )
   Batch = $(B)
   DEFINES += -DBatch=$(B)
   endif
ifneq ($(NAME), )
TARGET = $(ELF_DIR)/$(NAME)_B$(Batch)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK).elf
else
TARGET = $(ELF_HEAD)_$(TESTCASE)_$(MODE)_B$(Batch)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK).elf
endif
endif

ifeq ($(TESTCASE), flash_attention)
DEFINES += -DBatSz=$(B) -DglobS=$(S) -DglobD=$(D) -DtilM=$(tM) -DtilK=$(tK)
TARGET = $(ELF_HEAD)_$(TESTCASE)_$(MODE)_B$(B)_S$(S)_D$(D)_tM$(tM)_tK$(tK).elf
endif

ifeq ($(TESTCASE), softmax)
DEFINES += -DglobM=$(M) -DglobN=$(N) -DtilM=$(tM) -DtilN=$(tN)
TARGET = $(ELF_HEAD)_$(TESTCASE)_M$(M)_N$(N)_tM$(tM)_tN$(tN).elf
endif

ifeq ($(TESTCASE), onlinesoftmax)
DEFINES += -DglobM=$(M) -DglobN=$(N) -DtilM=$(tM) -DtilN=$(tN)
TARGET = $(ELF_HEAD)_$(TESTCASE)_M$(M)_N$(N)_tM$(tM)_tN$(tN).elf
endif

ifeq ($(TESTCASE), normalization)
DEFINES += -DglobM=$(M) -DglobN=$(N) -DtilM=$(tM) -DtilN=$(tN)
TARGET = $(ELF_HEAD)_$(TESTCASE)_$(MODE)_M$(M)_N$(N)_tM$(tM)_tN$(tN).elf
endif

ifeq ($(TESTCASE), gemm)
   ifeq ($(relu), on)
   DEFINES += -DwithRelu=true
   else
   relu = off
   DEFINES += -DwithRelu=false
   endif
   ifeq ($(bias), on)
   DEFINES += -DhasBias=true
   else
   bias = off
   DEFINES += -DhasBias=false
   endif
DEFINES += -DglobM=$(M) -DglobN=$(N) -DglobK=$(K) -DtilM=$(tM) -DtilN=$(tN) -DtilK=$(tK)
TARGET = $(ELF_HEAD)_$(TESTCASE)_M$(M)_N$(N)_K$(K)_tM$(tM)_tK$(tK)_tN$(tN)_relu_$(relu)_bias_$(bias).elf
endif

ifeq ($(TESTCASE), linear)
   ifeq ($(bias), on)
   DEFINES += -DhasBias=true
   else
   bias = off
   DEFINES += -DhasBias=false
   endif
DEFINES += -DDIN1=$(DIN1) -DDIN2=$(DIN2) -DDOUT=$(DOUT)
TARGET = $(ELF_HEAD)_$(TESTCASE)_$(MODE)_DIN1_$(DIN1)_DIN2_$(DIN2)_DOUT_$(DOUT)_bias_$(bias).elf
endif

ifneq ($(MODE), )
   DEFINES += -D$(MODE)
endif

CP = cp
DEST_DIR = ~/elf_subset/subset_matmul_reuse/
SRC_FILE +=  $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
include ../../common/Makefile.common

$(TARGET): clean $(LINK_SCRIPT) $(COMM_OBJ) $(OBJ) $(EXTRA_OBJ_FILES)
	@mkdir -p $(shell dirname $@)
	$(LINK) $(CC_LINK) $(OBJ) $(COMM_OBJ) $(EXTRA_OBJ_FILES) -o $@
	$(CP) $(TARGET) $(DEST_DIR)
