ifeq ($(TESTCASE), split)
DEFINES += -DROW=$(row) -DDIM1=$(dim1) -DDIM2=$(dim2)
TARGET = $(ELF_HEAD)_$(TESTCASE)_r$(row)_c1_$(dim1)_c2_$(dim2).elf
endif

ifeq ($(TESTCASE), concat)
DEFINES += -DROW=$(row) -DDIM1=$(dim1) -DDIM2=$(dim2)
TARGET = $(ELF_HEAD)_$(TESTCASE)_r$(row)_c1_$(dim1)_c2_$(dim2).elf
endif

ifeq ($(TESTCASE), permute)
DEFINES += -DDIM0=$(dim0) -DDIM1=$(dim1) -DDIM2=$(dim2) -DDIM3=$(dim3)
TARGET = $(ELF_HEAD)_$(TESTCASE)_D0_$(dim0)_D1_$(dim1)_D2_$(dim2)_D3_$(dim3).elf
endif

ifeq ($(TESTCASE), expand)
DEFINES += -DROW=$(row) -DDIM=$(dim) -DEXT_DIM=$(ext_dim)
TARGET = $(ELF_HEAD)_$(TESTCASE)_row$(row)_dim$(dim)_extdim$(ext_dim).elf
endif

ifeq ($(TESTCASE), rmsnorm)
DEFINES += -DglobM=$(M) -DglobN=$(N) -DtilM=$(tM) -DtilN=$(tN)
TARGET = $(ELF_HEAD)_$(TESTCASE)_M$(M)_N$(N)_tM$(tM)_tN$(tN).elf
endif

ifeq ($(TESTCASE), projection)
DEFINES += -DBSZ=$(BSZ) -DSLEN=$(SLEN) -DDIM_IN=$(DIM_IN) -DDIM_OUT=$(DIM_OUT)
TARGET = $(ELF_HEAD)_$(TESTCASE)_BSZ$(BSZ)_S$(SLEN)_DIN$(DIM_IN)_DOUT$(DIM_OUT).elf
endif

ifeq ($(TESTCASE), rope)
DEFINES += -DROW=$(ROW) -DPE_DIM=$(PE_DIM)
TARGET = $(ELF_HEAD)_$(TESTCASE)_ROW$(ROW)_PeDim$(PE_DIM).elf
endif

ifeq ($(TESTCASE), mla)
TARGET = $(ELF_HEAD)_$(TESTCASE).elf
endif

ifeq ($(TESTCASE), topk)
DEFINES += -DROW=$(ROW) -DDIM=$(DIM) -DTOPK=$(TOPK)
TARGET = $(ELF_HEAD)_$(TESTCASE)_ROW$(ROW)_DIM$(DIM)_TOPK$(TOPK).elf
endif

ifeq ($(TESTCASE), gate)
DEFINES += -DBSZ=$(bsz) -DSLEN=$(slen)
TARGET = $(ELF_HEAD)_$(TESTCASE)_BSZ$(bsz)_SLEN$(slen).elf
endif

ifeq ($(TESTCASE), mlp)
DEFINES += -DROW=$(ROW) -DDIM=$(DIM) -DINTER_DIM=$(INTER_DIM)
TARGET = $(ELF_HEAD)_$(TESTCASE)_ROW$(ROW)_DIM$(DIM)_INTERDIM$(INTER_DIM).elf
endif

ifeq ($(TESTCASE), mask)
DEFINES += -DROW=$(ROW) -DDIM=$(DIM) -DIDX_DIM=$(IDX_DIM) -DEXT_DIM=$(EXT_DIM)
TARGET = $(ELF_HEAD)_$(TESTCASE)_ROW$(ROW)_DIM$(DIM)_IDXDIM$(IDX_DIM)_EXTDIM$(EXT_DIM).elf
endif

ifeq ($(TESTCASE), moe)
TARGET = $(ELF_HEAD)_$(TESTCASE).elf
endif

ifeq ($(TESTCASE), transformer)
TARGET = $(ELF_HEAD)_$(TESTCASE).elf
endif

ifneq ($(MODE), )
   DEFINES += -DMODE=\"$(MODE)\"
endif

SRC_FILE +=  $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
include ../../common/Makefile.common
