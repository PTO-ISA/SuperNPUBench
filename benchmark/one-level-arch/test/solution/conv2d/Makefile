# Conv2d 1x1 test Makefile (all variants merged)
#
# TESTCASE selects the variant (source file src/$(TESTCASE).cpp):
#   - conv2d:            legacy TileLeft/TileRight kernel (IsCubeLayout assert under TileOP f94bc12, docs/conv2d_accuracy_verification.md §6.4)
#   - conv2d_cm:         ColMajor GM variant (compiles; numeric FAIL repro, docs/conv2d_accuracy_verification.md §6.1)
#   - conv2d_rm:         RowMajor single-PE CubeTile kernel (recommended, verified PASS)
#   - conv2d_rm_dyn:     RowMajor dynamic-shape variant (IsCubeLayout assert under f94bc12, docs/conv2d_accuracy_verification.md §6.4)
#   - conv2d_rm_mt:      RowMajor multi-PE batch-of-4 (recommended, verified PASS)
#   - conv2d_rm_shared:  RowMajor multi-PE SharedTile variant (not migrated, docs/conv2d_accuracy_verification.md §6.3)
#
# Usage:
#   make TYPE=FP32 IN_H=4 IN_W=4 IN_C=16 OUT_C=16 tilM=16 tilN=16 tilK=16 TESTCASE=conv2d_rm PLAT=linx
#   make TYPE=FP32 IN_H=4 IN_W=4 IN_C=16 OUT_C=16 tilM=16 tilN=16 tilK=16 TESTCASE=conv2d_rm_mt PLAT=linx
#   make TYPE=FP32 IN_H=14 IN_W=14 IN_C=64 OUT_C=64 tilM=16 tilN=16 tilK=16 TESTCASE=conv2d TMUL_VERIFY=1

TESTCASE ?= conv2d_rm

SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp

ifeq ($(TESTCASE), conv2d)
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
endif

ifeq ($(TYPE), FP32)
    DEFINES += -DIN_H=$(IN_H) -DIN_W=$(IN_W) -DIN_C=$(IN_C) -DOUT_C=$(OUT_C) \
               -DtilM=$(tilM) -DtilN=$(tilN) -DtilK=$(tilK)
    DEFINES += -DCONV_FP32
    TARGET = $(ELF_HEAD)/$(TESTCASE)_$(TYPE)_IN$(IN_H)x$(IN_W)x$(IN_C)_OUT$(OUT_C)_tM$(tilM)_tN$(tilN)_tK$(tilK)$(TMUL_SUFFIX)$(EMBED_SUFFIX).elf
endif

ifeq ($(TYPE), FP16)
    DEFINES += -DIN_H=$(IN_H) -DIN_W=$(IN_W) -DIN_C=$(IN_C) -DOUT_C=$(OUT_C) \
               -DtilM=$(tilM) -DtilN=$(tilN) -DtilK=$(tilK)
    DEFINES += -DCONV_FP16
    TARGET = $(ELF_HEAD)/$(TESTCASE)_$(TYPE)_IN$(IN_H)x$(IN_W)x$(IN_C)_OUT$(OUT_C)_tM$(tilM)_tN$(tilN)_tK$(tilK)$(TMUL_SUFFIX)$(EMBED_SUFFIX).elf
endif

include ../../common/Makefile.common

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)
