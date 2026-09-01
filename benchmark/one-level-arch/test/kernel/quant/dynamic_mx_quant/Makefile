# DynamicMxQuant test entries. Only DEBUGGED configs (op-aligned to AscendC, all
# known issues recorded in RECORD.md) keep a build entry here. UNDEBUGGED kernels
# (tail/nontail OCP-FP8, tail/nontail DynRange-FP4) have had their driver cpp and
# TYPE blocks removed on purpose — see README.md「状态总览」. FP4_PROBE and
# BS512_CUBLAS (a TileSize probe over the debugged cublas kernel) are kept as tooling.

ifeq ($(TYPE), TAIL_CUBLAS_FP8)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tail_cublas_fp8.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tail_cublas_fp8.elf
endif

ifeq ($(TYPE), TAIL_CUBLAS_FP8_FP16)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tail_cublas_fp8_fp16.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tail_cublas_fp8_fp16.elf
endif

ifeq ($(TYPE), NONTAIL_CUBLAS_FP8_FP16)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_cublas_fp8_fp16.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_nontail_cublas_fp8_fp16.elf
endif

ifeq ($(TYPE), TAIL_OCP_FP4)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tail_ocp_fp4.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tail_ocp_fp4.elf
endif

ifeq ($(TYPE), NONTAIL_CUBLAS_FP8)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_cublas_fp8.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_nontail_cublas_fp8.elf
endif

ifeq ($(TYPE), NONTAIL_CUBLAS_FP8_BIGBS)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_cublas_fp8_bigbs.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_nontail_cublas_fp8_bigbs.elf
endif

ifeq ($(TYPE), NONTAIL_OCP_FP4)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_ocp_fp4.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_nontail_ocp_fp4.elf
endif

ifeq ($(TYPE), FP4_PROBE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/fp4_probe.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_fp4_probe.elf
endif

ifeq ($(TYPE), DTYPE_PROBE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/dtype_probe.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_dtype_probe.elf
endif

ifeq ($(TYPE), BS512_CUBLAS)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/bs512_cublas.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_bs512_cublas.elf
endif

ifeq ($(TYPE), PROBE_OCP_FP8)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/probe_ocp_fp8.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_probe_ocp_fp8.elf
endif

ifeq ($(TYPE), PROBE_OCP_FP8_NEWCALC)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/probe_ocp_fp8_newcalc.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_probe_ocp_fp8_newcalc.elf
endif

# 正式 TAIL_OCP_FP8 kernel（固定 SPMD 4-PE，去 probe 化自 newcalc_mt 母本）。
# 必须用 4 线程跑：gfrun -s softcore.multiThreadNum=4 / gfsim --conf fourpe。
ifeq ($(TYPE), TAIL_OCP_FP8)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tail_ocp_fp8.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tail_ocp_fp8.elf
endif

# 正式 TAIL_CUBLAS_FP8 kernel（固定 SPMD 4-PE，kPeNum=4，fp16 in）。
# 必须用 4 线程跑：gfrun -s softcore.multiThreadNum=4。
ifeq ($(TYPE), TAIL_CUBLAS_FP8_4PE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tail_cublas_fp8_4pe.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tail_cublas_fp8_4pe.elf
endif

# 正式 NONTAIL_CUBLAS_FP8 kernel（固定 SPMD 4-PE，kPeNum=4，按块行切分，fp16 in）。
# 必须用 4 线程跑：gfrun -s softcore.multiThreadNum=4。
ifeq ($(TYPE), NONTAIL_CUBLAS_FP8_4PE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_cublas_fp8_4pe.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_nontail_cublas_fp8_4pe.elf
endif

ifeq ($(TYPE), ALIGN32_PROBE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/align32_probe.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_align32_probe.elf
endif

ifeq ($(TYPE), ROW32_PROBE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/row32_probe.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_row32_probe.elf
endif

ifeq ($(TYPE), TSEL_INPLACE_PROBE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tsel_inplace_probe.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tsel_inplace_probe.elf
endif

ifeq ($(TYPE), TSEL_INPLACE_FALSESRC_PROBE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tsel_inplace_falsesrc_probe.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tsel_inplace_falsesrc_probe.elf
endif

ifeq ($(TYPE), FP4_SHAPE_PROBE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/fp4_shape_probe.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_fp4_shape_probe.elf
   ifeq ($(WIDEN), on)
     DEFINES += -DWIDEN
   endif
endif

ifeq ($(TYPE), TABS_TROWMAX_PROBE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tabs_trowmax_probe.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tabs_trowmax_probe.elf
endif

ifeq ($(TYPE), TCVT_REDUCE_SHAPE_PROBE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tcvt_reduce_shape_probe.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tcvt_reduce_shape_probe.elf
endif

include ../../../common/Makefile.common

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)
