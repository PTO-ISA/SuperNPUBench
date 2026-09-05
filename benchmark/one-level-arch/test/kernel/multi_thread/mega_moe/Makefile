# MegaMoe A8W8 — 真 4PE (multi-thread) 版 Makefile
# 规格与单线程版一致 (默认 16×32×64); gfrun 运行需 softcore.multiThreadNum=4

TESTCASE ?= mega_moe_sim_mt

BS ?= 16
H ?= 32
HIDDEN ?= 64

DEFINES += -DkBS=$(BS) -DkH=$(H) -DkHidden=$(HIDDEN)

# SIM_FAKE mode: sigmoid-only fast path (aligned with A5 MEGA_MOE_SIM_FAKE)
ifeq ($(SIM_FAKE), 1)
DEFINES += -DMEGA_MOE_SIM_FAKE
endif

TARGET = $(ELF_DIR)/$(TESTCASE)_BS$(BS)_H$(H)_HD$(HIDDEN).elf
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp

include ../../../common/Makefile.common

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)
