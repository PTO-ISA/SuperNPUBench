# DynamicMxQuant solution test entries. 8 个验证通过的正式用例（tail/nontail ×
# OCP/cuBLAS × FP8/FP4，含 4-PE SPMD 与大 BlockSize 变体），映射到 6 个 kernel 实现。
# 注：两个 *_bigbs 用例是大 BlockSize 场景，现走 nontail plain 单块路径（方案A 切归约轴
# 的 _bigbs kernel 已退休——新工具链 tile 上限 256KB，单块可直接覆盖大 BS）。elf basename
# 固定为 dynamic_mx_quant_<variant>，使 dynamic_mx_quant_data_compare.py 按 basename 找
# compare/<basename> 金标目录。4-PE 变体运行须 gfrun -s softcore.multiThreadNum=4。

ifeq ($(TYPE), TAIL_OCP_FP8)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tail_ocp_fp8.cpp
   TARGET = $(ELF_DIR)/$(TESTCASE)_tail_ocp_fp8.elf
endif

ifeq ($(TYPE), TAIL_OCP_FP8_DYN)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tail_ocp_fp8_dyn.cpp
   TARGET = $(ELF_DIR)/$(TESTCASE)_tail_ocp_fp8_dyn.elf
endif

ifeq ($(TYPE), TAIL_OCP_FP4)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tail_ocp_fp4.cpp
   TARGET = $(ELF_DIR)/$(TESTCASE)_tail_ocp_fp4.elf
endif

ifeq ($(TYPE), TAIL_CUBLAS_FP8_4PE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tail_cublas_fp8_4pe.cpp
   TARGET = $(ELF_DIR)/$(TESTCASE)_tail_cublas_fp8_4pe.elf
endif

ifeq ($(TYPE), NONTAIL_CUBLAS_FP8_4PE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_cublas_fp8_4pe.cpp
   TARGET = $(ELF_DIR)/$(TESTCASE)_nontail_cublas_fp8_4pe.elf
endif

ifeq ($(TYPE), NONTAIL_CUBLAS_FP8_BIGBS)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_cublas_fp8_bigbs.cpp
   TARGET = $(ELF_DIR)/$(TESTCASE)_nontail_cublas_fp8_bigbs.elf
endif

ifeq ($(TYPE), NONTAIL_OCP_FP4_4PE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_ocp_fp4_4pe.cpp
   TARGET = $(ELF_DIR)/$(TESTCASE)_nontail_ocp_fp4_4pe.elf
endif

ifeq ($(TYPE), NONTAIL_OCP_FP4_BIGBS)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_ocp_fp4_bigbs.cpp
   TARGET = $(ELF_DIR)/$(TESTCASE)_nontail_ocp_fp4_bigbs.elf
endif

# 可选 shape 覆盖（4-PE res_check 多 Post 一致性）：命令行提供时转发为 -D 宏。
# tail driver 用 PM/PN；nontail driver 用 PAXIS/PPOST。driver 内 #ifndef 默认值不受影响。
ifdef PM
   DEFINES += -DPM=$(PM)
endif
ifdef PN
   DEFINES += -DPN=$(PN)
endif
ifdef PAXIS
   DEFINES += -DPAXIS=$(PAXIS)
endif
ifdef PPOST
   DEFINES += -DPPOST=$(PPOST)
endif

include ../../../common/Makefile.common
