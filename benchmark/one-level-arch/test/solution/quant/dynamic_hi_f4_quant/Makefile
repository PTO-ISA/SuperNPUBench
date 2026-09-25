# DynamicHiF4Quant 数值验证 build 入口。

# 数值验证 driver（单-PE）。res_check=on 注入 -DRES_CHECK -DCHK_DIR + group_worker_runtime.o；
# 由 run_hif4_precision.py 驱动:gen golden -> make res_check=on -> gfrun multiThreadNum=1 ->
# hif4_compare.py 重建 MSE。PM/PN 通过 make PM=.. PN=.. 覆盖(默认 32/128)。Build:
#   make TESTCASE=dynamic_hi_f4_quant TYPE=HIF4_RES_CHECK res_check=on PM=32 PN=128
ifeq ($(TYPE), HIF4_RES_CHECK)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/hif4_res_check.cpp
   ifdef PM
     DEFINES += -DPM=$(PM)
   endif
   ifdef PN
     DEFINES += -DPN=$(PN)
   endif
   TARGET = $(ELF_HEAD)/$(TESTCASE)_hif4_res_check.elf
endif

include ../../../common/Makefile.common

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)
