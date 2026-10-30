# ArgMaxV2 (CANN ArgMaxV2 / aclnnArgMax 语义) 四 PE 测试。
#
# 本目录为 ArgMaxV2 唯一测试目录，四套件：
#   1. 代表例（下方 2 例，compile.all 逐例构建）——切 R 路径与动态版，
#      run_res_check.py host-golden 零容差比对（per-operator 脚本范式，
#      对齐 matmul golden_cmp.py / mxquant run_mxquant_check.py）。
#   2. 单核缩放用例 argmax_net_sc_1..8（run_net_sc.py）——网络形状单核
#      等比缩放，与 V121 逐例对比的性能追踪套件（自带精度比对）。
#   3. fuzz 用例 f000..（run_fuzz.py 生成源码后构建）——200 例随机形状
#      回归；run_gfsim_batch.py 为 gfsim 定时批量复验。
# 通用 make 入口：make TESTCASE=<名> COMPILER_DIR=<...> [res_check=on |
# SUFFIX=_t]，2-4 套件用例走文件尾通用映射分支。
#
# 代表例（金标准注册于 res_check_all.py；更广覆盖见 fuzz 200 随机形状、
# argmax_net_* 网络形状、argmax_net_sc_* 单核缩放套件，由同目录 runner
# 自包含生成数据并比对）：
#   argmaxv2          fp32 dim=1 [7,3333]  -> int32[7]   （切 R 路径代表例：
#     末段 tailB=5 + 哨兵 + 屏障合并全链；输入注入 NaN/±inf/平局/±0 边界）
#   argmaxv2_dynamic  fp32 dim=1 [133,1000] -> int32[133]（动态版代表例：
#     运行期 tiling、ceil 分区 + 尾数）
#
TESTCASE ?= argmaxv2
SUFFIX ?=

ifeq ($(TESTCASE),argmaxv2)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/argmaxv2.cpp
TARGET = $(ELF_HEAD)_argmaxv2_fp32_dim1_PE4.elf
else ifeq ($(TESTCASE),argmaxv2_dynamic)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/argmaxv2_dynamic.cpp
TARGET = $(ELF_HEAD)_argmaxv2_dynamic_fp32_dim1_PE4.elf
else
# 通用映射：单核缩放用例（argmax_net_sc_*）与 fuzz 生成用例（f*）——
# 源文件与 TESTCASE 同名，ELF 名亦随 TESTCASE。
# SUFFIX 分离 timing ELF（res_check 关闭时传 SUFFIX=_t，产物带 _t 后缀）。
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
SUFFIX ?=
TARGET = $(ELF_HEAD)_$(TESTCASE)_PE4$(SUFFIX).elf
endif

include ../../common/Makefile.common
