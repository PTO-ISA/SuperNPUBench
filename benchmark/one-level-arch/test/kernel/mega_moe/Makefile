# MegaMoe A8W8 (完整真机流水) 迁移样例 Makefile
# 仿真规格 16×32×64 (与 compile.all 一致, 已跑通 gfsim); 可通过 BS/H/HIDDEN 参数化调整

TESTCASE ?= mega_moe_sim

BS ?= 16
H ?= 32
HIDDEN ?= 64

LINX_TOOLCHAIN ?= /mnt/workspace/gitCode/cann/Dev-experience/v300_tag_0824/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin

# PLAT=cpu (host 仿真) 时 g++ 不自动注入 tileop-api 头路径, 需显式指定
ifeq ($(PLAT), cpu)
INCLUDE += -I$(LINX_TOOLCHAIN)/../lib/clang/15.0.4/include/tileop-api
endif

DEFINES += -DkBS=$(BS) -DkH=$(H) -DkHidden=$(HIDDEN)

# SIM_FAKE mode: sigmoid-only fast path (aligned with A5 MEGA_MOE_SIM_FAKE)
ifeq ($(SIM_FAKE), 1)
DEFINES += -DMEGA_MOE_SIM_FAKE
endif

ifeq ($(TESTCASE),mega_moe_sim)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/mega_moe_sim.cpp
TARGET = $(ELF_HEAD)/$(TESTCASE)_BS$(BS)_H$(H)_HD$(HIDDEN).elf
else ifeq ($(TESTCASE),mega_moe_full_sim)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/mega_moe_full_sim.cpp
TARGET = $(ELF_HEAD)/$(TESTCASE).elf
else
$(error Unsupported TESTCASE=$(TESTCASE); expected mega_moe_sim or mega_moe_full_sim)
endif

include ../../common/Makefile.common

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)