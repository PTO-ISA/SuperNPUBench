ifeq ($(TESTCASE), broadcast_vec_07)
DEFINES += -DDType=$(DType) -DtMs=$(tMs) -DMAX_DIMs=$(MAX_DIMs) -DIN_SHAPEs=$(IN_SHAPEs) -DOUT_SHAPEs=$(OUT_SHAPEs) -DIN_DIMs=$(IN_DIMs) -DOUT_DIMs=$(OUT_DIMs) -DgIMs=$(gIMs) -DgOMs=$(gOMs)
TARGET = $(ELF_HEAD)_$(TESTCASE)_$(MODE)_DType$(DType)_tM$(tMs)_IN_SHAPE$(IN_SHAPE_NAME)_OUT_SHAPE$(OUT_SHAPE_NAME).elf
endif

ifeq ($(TESTCASE), broadcast_vec_019)
DEFINES += -DDType=$(DType) -DtMs=$(tMs) -DkInners=$(kInners) -DMAX_DIMs=$(MAX_DIMs) -DIN_SHAPEs=$(IN_SHAPEs) -DOUT_SHAPEs=$(OUT_SHAPEs) -DIN_DIMs=$(IN_DIMs) -DOUT_DIMs=$(OUT_DIMs) -DgIMs=$(gIMs) -DgOMs=$(gOMs)
TARGET = $(ELF_HEAD)_$(TESTCASE)_$(MODE)_DType$(DType)_tM$(tMs)_kInner$(kInners)_IN_SHAPE$(IN_SHAPE_NAME)_OUT_SHAPE$(OUT_SHAPE_NAME).elf
endif

ifeq ($(TESTCASE), broadcast_vec_039)
DEFINES += -DDType=$(DType) -DtMs=$(tMs) -DkInners=$(kInners) -DMAX_DIMs=$(MAX_DIMs) -DIN_SHAPEs=$(IN_SHAPEs) -DOUT_SHAPEs=$(OUT_SHAPEs) -DIN_DIMs=$(IN_DIMs) -DOUT_DIMs=$(OUT_DIMs) -DgIMs=$(gIMs) -DgOMs=$(gOMs)
TARGET = $(ELF_HEAD)_$(TESTCASE)_$(MODE)_DType$(DType)_tM$(tMs)_kInner$(kInners)_IN_SHAPE$(IN_SHAPE_NAME)_OUT_SHAPE$(OUT_SHAPE_NAME).elf
endif

# DEFINES += -DRES_CHECK
# DEFINES += -DCHK_DIR="/remote/lms01/l00948608/project/jcore_benchmark/JanusCoreBench/test/broadcast_vec/src/data"

SRC_FILE +=  $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
include ../../../common/Makefile.common