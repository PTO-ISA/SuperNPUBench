#!/usr/bin/python3

import os
import sys
import argparse
import json
import math
import subprocess
import re
import torch
import torch.nn as nn
import numpy as np
import signal 
from concurrent.futures import ThreadPoolExecutor, as_completed

#1 read elf list
#2 分析elf name: matmul MNK 
#3 产生随机src0, src1 -> golden_dst 放到 kernel/golden/kernel_matmul_MASK_M128_N1024_K128_tM32_tK32_tN32/src0.bin, src1.bin golden.bin
#4 跑qemu 读取src0, src1 产生dst -> /remote/lms01/c00622284/janus/JanusCoreBench/compare/kernel_matmul_MASK_M128_N1024_K128_tM32_tK32_tN32/res.bin
#5 跑res_compare.py
#6 输出一个报告.

MAX_WORKERS = 20  #parallel thread num depend on your machine
qemu = "/remote/lms01/l00948608/project/jcore_benchmark/qemu_code/LinxBlockModel/build/qemu-linx"
qemu_args = " -blk_optimize force_tb_chained -s 4096M "

cmp_root = os.path.abspath(os.path.dirname(__file__)+"/../../../compare")
script_path = os.path.abspath(os.path.dirname(__file__))
DType = torch.int32
DType_np = np.int32

statics = {"pass":[], "fail":[]}

def map_rule(elf_name):
    pass

def read_bf16_bin(file_path):
    with open(file_path, "rb") as f:
        buffer = f.read()
    tensor = torch.frombuffer(buffer, dtype=torch.uint8).view(torch.bfloat16)
    return tensor
    
def result_compare(cmp_data, golden_data):
    print("cmp data: ",cmp_data)
    if "__bf16" in cmp_data:
        res = read_bf16_bin(cmp_data)
        res_ref = read_bf16_bin(golden_data)   
        loss = torch.mean((res - res_ref) ** 2)      
    else:
        res = np.fromfile(cmp_data, dtype=DType_np)
        res_ref = np.fromfile(golden_data, dtype=DType_np)
        loss = np.mean((res - res_ref) ** 2).item()
    if loss < 0.1:
        return "pass", loss
    else:
        return "fail", loss

def check_elf(elf):
    print('Start ro check elf----------------')
    elf_name = os.path.basename(elf).replace(".elf", "").strip()
    cmp_data_path = cmp_root+"/"+elf_name

    # os.system(f"rm -rf {cmp_data_path};mkdir -p {cmp_data_path}")

    os.makedirs(cmp_data_path, exist_ok=True)
    print('com_data path', cmp_data_path)
    cmp_data = cmp_data_path+"/output.bin"
    golden_data = cmp_data_path+"/golden.bin"
    if os.path.exists(cmp_data):
        chk_status, loss = result_compare(cmp_data, golden_data)
        print('loss = {loss}')
        return elf_name, chk_status, loss
    else:
        return elf_name, "output not exist", "NaN"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Compile_all Scripts")
    parser.add_argument("-l", dest="elf_list", default="./tmp.list", type=str, help="elf list contain elf absolute path")
    parser.add_argument("-d", dest="dbg_elf", default=None, type=str, help="debug single elf")
    parser.add_argument("-o", dest="res_log", default="result_check.log", type=str, help="result_log name")
    parser.add_argument("-plat", dest="plat", default="linx", type=str, help="PLATFORM")
    args = parser.parse_args()

    if args.dbg_elf is not None:
        elf_paths = [args.dbg_elf]
    else:
        with open(args.elf_list, "r") as f:
            elf_paths = [line.strip() for line in f if line.strip()]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(check_elf, elf): elf for elf in elf_paths}
        for future in as_completed(futures):
            elf, chk_status, loss = future.result()
            print(f"{elf} → chk_status: {chk_status.upper()} loss:{loss}")

    if args.dbg_elf is None:
        with open(cmp_root+"/"+args.res_log, "w") as f:
            pass_num = len(statics["pass"])
            fail_num = len(statics["fail"])
            f.write(f"\nResult_Check Summary: \n")
            f.write(f"\npass : {pass_num}\n")
            f.write(f"\nfail : {fail_num}\n")
            f.write(f"\npass list:\n")
            for elf in statics["pass"]:
                f.write(f"{elf}\n")

            f.write(f"\n\n\n")

            f.write(f"fail list:\n")
            for elf in statics["fail"]:
                f.write(f"{elf}\n")



