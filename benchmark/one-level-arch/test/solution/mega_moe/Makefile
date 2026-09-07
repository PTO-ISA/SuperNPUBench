# MegaMoe A8W8 (full device pipeline) — single-PE sim drivers and true 4-PE variant.
# Default sim spec 16x32x64 (matches compile.all; verified on gfsim).
# Single-PE:     mega_moe_sim / mega_moe_full_sim   (gfrun default)
# 4-PE:          mega_moe_sim_mt                    (gfrun -s softcore.multiThreadNum=4)

TESTCASE ?= mega_moe_sim

BS ?= 16
H ?= 32
HIDDEN ?= 64

# PLAT=cpu (host simulation) needs the tileop-api headers from the toolchain.
ifeq ($(PLAT), cpu)
INCLUDE += -I$(COMPILER_DIR)/../lib/clang/15.0.4/include/tileop-api
endif

DEFINES += -DkBS=$(BS) -DkH=$(H) -DkHidden=$(HIDDEN)

# SIM_FAKE mode: sigmoid-only fast path (aligned with A5 MEGA_MOE_SIM_FAKE)
ifeq ($(SIM_FAKE), 1)
DEFINES += -DMEGA_MOE_SIM_FAKE
endif

ifeq ($(TESTCASE),mega_moe_sim)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/mega_moe_sim.cpp
TARGET = $(ELF_HEAD)_$(TESTCASE)_BS$(BS)_H$(H)_HD$(HIDDEN).elf
else ifeq ($(TESTCASE),mega_moe_full_sim)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/mega_moe_full_sim.cpp
TARGET = $(ELF_HEAD)_$(TESTCASE).elf
else ifeq ($(TESTCASE),mega_moe_sim_mt)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/mega_moe_sim_mt.cpp
TARGET = $(ELF_HEAD)_$(TESTCASE)_BS$(BS)_H$(H)_HD$(HIDDEN).elf
else
$(error Unsupported TESTCASE=$(TESTCASE); expected mega_moe_sim, mega_moe_full_sim or mega_moe_sim_mt)
endif

include ../../common/Makefile.common

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)
