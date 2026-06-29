ifeq ($(TESTCASE), gather)
DEFINES += -DDType=$(DType) -DOType=$(OType) -DgKs=$(gKs) -DgMs=$(gMs) -DgNs=$(gNs) -DtMs=$(tMs) -DtNs=$(tNs)
TARGET = $(ELF_HEAD)_$(TESTCASE)_DType$(DType)_OType$(OType)_gKs$(gKs)_gMs$(gMs)_gNs$(gNs)_tMs$(tMs)_tNs$(tNs).elf
endif

# DEFINES += -DRES_CHECK
# DEFINES += -DCHK_DIR="/remote/lms01/l00948608/project/jcore_benchmark/JanusCoreBench/test/broadcast/src/data"

SRC_FILE +=  $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
$(info  ===> SRC_FILE = $(SRC_FILE))
include ../../common/Makefile.common