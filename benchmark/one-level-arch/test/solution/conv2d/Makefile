TESTCASE ?= conv2d_img2col

# conv2d_img2col: NCHW in / NHWC out conv driven by TIMG2COL + weight TLOAD.
# Golden: src/golden_conv2d_img2col.py (torch.nn.functional.conv2d).
Batch ?= 2
Cin ?= 8
Hin ?= 32
Win ?= 32
Cout ?= 32
Kh ?= 2
Kw ?= 2
Sh ?= 2
Sw ?= 2
Pt ?= 0
Pb ?= 0
Pl ?= 0
Pr ?= 0
Dh ?= 1
Dw ?= 1

ifeq ($(TESTCASE),conv2d_img2col)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/conv2d_img2col.cpp
TARGET = $(ELF_HEAD)_conv2d_img2col_PE4.elf
DEFINES += -DBatch=$(Batch) -DCin=$(Cin) -DHin=$(Hin) -DWin=$(Win)
DEFINES += -DCout=$(Cout) -DKh=$(Kh) -DKw=$(Kw) -DSh=$(Sh) -DSw=$(Sw)
DEFINES += -DPt=$(Pt) -DPb=$(Pb) -DPl=$(Pl) -DPr=$(Pr)

else ifeq ($(TESTCASE),conv2d_img2col_dyn)
# Dynamic-shape variant: all shape numbers reach the kernel through the
# runtime tiling descriptor; the macros below only size the host buffers
# and pick the KPhys bucket. Golden: src/golden_conv2d_img2col_dyn.py.
# Default shape shares the static test's KPhys=32 bucket but otherwise
# differs (Cin=16, 1x2 kernel, 32x64 map, asymmetric pads, Co=48).
Batch = 2
Cin = 16
Hin = 32
Win = 64
Cout = 48
Kh = 1
Kw = 2
Sh = 2
Sw = 2
Pt = 2
Pb = 1
Pl = 0
Pr = 1
Dh = 1
Dw = 1
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/conv2d_img2col_dyn.cpp
TARGET = $(ELF_HEAD)_conv2d_img2col_dyn_PE4.elf
DEFINES += -DBatch=$(Batch) -DCin=$(Cin) -DHin=$(Hin) -DWin=$(Win)
DEFINES += -DCout=$(Cout) -DKh=$(Kh) -DKw=$(Kw) -DSh=$(Sh) -DSw=$(Sw)
DEFINES += -DPt=$(Pt) -DPb=$(Pb) -DPl=$(Pl) -DPr=$(Pr)
DEFINES += -DDh=$(Dh) -DDw=$(Dw)

else
$(error Unsupported TESTCASE=$(TESTCASE); expected conv2d_img2col or conv2d_img2col_dyn)
endif

include ../../common/Makefile.common
