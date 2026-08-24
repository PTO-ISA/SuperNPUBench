# Conv2d 1x1 test Makefile
#
# 1x1 convolution = matmul:
#   input  NCHW (in_c, in_h*in_w)  -> A(gM, gK) ColMajor
#   weight      (out_c, in_c)      -> B(gK, gN) ColMajor
#   output      (out_c, in_h*in_w) -> C(gM, gN) RowMajor
#
# Usage:
#   make TYPE=FP32 IN_H=14 IN_W=14 IN_C=64 OUT_C=64 tilM=16 tilN=16 tilK=16 TESTCASE=conv2d
#   make TYPE=FP32 IN_H=4 IN_W=4 IN_C=16 OUT_C=16 tilM=16 tilN=16 tilK=16 TESTCASE=conv2d TMUL_VERIFY=1

ifeq ($(TMUL_VERIFY), 1)
   DEFINES += -DCONV2D_TMUL_VERIFY
   TMUL_SUFFIX = _tmulverify
else
   TMUL_SUFFIX =
endif

ifeq ($(EMBED_DATA), 1)
   DEFINES += -DEMBED_DATA
   EMBED_SUFFIX = _embed
else
   EMBED_SUFFIX =
endif

ifeq ($(TYPE), FP32)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/conv2d.cpp
   DEFINES += -DIN_H=$(IN_H) -DIN_W=$(IN_W) -DIN_C=$(IN_C) -DOUT_C=$(OUT_C) \
              -DtilM=$(tilM) -DtilN=$(tilN) -DtilK=$(tilK)
   DEFINES += -DCONV_FP32
   TARGET = $(ELF_HEAD)/$(TESTCASE)_$(TYPE)_IN$(IN_H)x$(IN_W)x$(IN_C)_OUT$(OUT_C)_tM$(tilM)_tN$(tilN)_tK$(tilK)$(TMUL_SUFFIX)$(EMBED_SUFFIX).elf
endif

ifeq ($(TYPE), FP16)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/conv2d.cpp
   DEFINES += -DIN_H=$(IN_H) -DIN_W=$(IN_W) -DIN_C=$(IN_C) -DOUT_C=$(OUT_C) \
              -DtilM=$(tilM) -DtilN=$(tilN) -DtilK=$(tilK)
   DEFINES += -DCONV_FP16
   TARGET = $(ELF_HEAD)/$(TESTCASE)_$(TYPE)_IN$(IN_H)x$(IN_W)x$(IN_C)_OUT$(OUT_C)_tM$(tilM)_tN$(tilN)_tK$(tilK)$(TMUL_SUFFIX)$(EMBED_SUFFIX).elf
endif

include ../../common/Makefile.common

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)