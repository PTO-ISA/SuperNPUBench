# DynamicHiF4Quant test entries. The operator itself is still in design (see
# kernels/multi_thread/quant/dynamic_hi_f4_quant/DESIGN.md + RECORD.md); only emit-capability
# PROBES have build entries so far. Probes are EMIT/diss checks (runtime blocked
# by toolchain<->emulator skew).

# e6m2 (L1 base scale) cast emit probe. Stage 1 (default) exercises the honest
# one-level path Tile<__fp8_e6m2> + TCVT/TSTORE and is EXPECTED TO FAIL COMPILE
# (type_traits<__fp8_e6m2> unregistered — RECORD 问题3). Stage 2 injects a
# placeholder type_traits to prove the dst type is mis-tagged (e8m0). Build with:
#   make TESTCASE=dynamic_hi_f4_quant TYPE=E6M2_PROBE diss            # stage 1
#   make TESTCASE=dynamic_hi_f4_quant TYPE=E6M2_PROBE_S2 diss         # stage 2
ifeq ($(TYPE), E6M2_PROBE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/e6m2_probe.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_e6m2_probe.elf
endif

ifeq ($(TYPE), E6M2_PROBE_S2)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/e6m2_probe.cpp
   DEFINES += -DSTAGE2
   TARGET = $(ELF_HEAD)/$(TESTCASE)_e6m2_probe_s2.elf
endif

# Segmented-max + grouped-broadcast-mul emit probe (DESIGN §3.1 问题1/2). Verifies
# TRESHAPE + TROWMAX (segmented max 64->16->8->1) and TRESHAPE + TROWEXPANDMUL
# (grouped broadcast-mul 16->64) EMIT on the one-level toolchain. Build with:
#   make TESTCASE=dynamic_hi_f4_quant TYPE=SEGREDUCE_PROBE diss
ifeq ($(TYPE), SEGREDUCE_PROBE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/segreduce_probe.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_segreduce_probe.elf
endif

# Route-A framework emit witness. Instantiates dynamic_hi_f4_quant_tail (bf16 +
# half input paths, full+tail pass, multi-block) and checks it COMPILES CLEAN and
# emits real BSTART.TEPL blocks. Framework carries PLACEHOLDER gaps (RECORD 问题
# 3/4/10) so numerics are NOT trusted; this is an emit/diss witness only (runtime
# blocked by toolchain<->emulator skew). Build with:
#   make TESTCASE=dynamic_hi_f4_quant TYPE=FRAMEWORK diss
ifeq ($(TYPE), FRAMEWORK)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/framework_emit.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_framework_emit.elf
endif

include ../../../../common/Makefile.common

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)
