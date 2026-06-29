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

cmp_root = os.path.abspath(os.path.dirname(__file__)+"/../../../../../compare")
script_path = os.path.abspath(os.path.dirname(__file__))
DType = torch.int32
DType_np = np.int32
OType = torch.uint32
OType_np = np.uint32

statics = {"pass":[], "fail":[]}

def map_rule(elf_name):
    pass

def extract_shape(shape_name, elf_name):
    pattern = rf'{shape_name}((?:\d+_)*\d+)'
    match = re.search(pattern, elf_name)
    shape_str = match.group(1)
    return tuple(map(int, shape_str.split('_')))

def gen_input_and_golden(elf_name, path):
    print('Start to gen input data & golden data:',path,elf_name)
    if "gather" in elf_name:
        gKs  = extract_shape("gKs", elf_name)[0]
        gMs = extract_shape("gMs", elf_name)[0]
        gNs = extract_shape("gNs", elf_name)[0]
        input_data_shape = (gKs, gNs)
        input_offset_shape = (gMs, 1)
        print(f"✅ input_data_shape: {input_data_shape}")
        print(f"✅ input_offset_shape: {input_offset_shape}")
        torch.manual_seed(123)
        if "DTypeint32_t" in elf_name:
            DType = torch.int32
            DType_np = np.int32
        if "DTypeuint32_t" in elf_name:
            DType = torch.uint32
            DType_np = np.uint32
        elif "DTypeFP16" in elf_name:
            DType = torch.float16
            DType_np = np.float16
        elif "DTypeFP8" in elf_name:
            DType = torch.float8_e4m3fn
            DType_np = np.float8_e4m3fn
        else:
            DType = torch.float32
            DType_np = np.float32

        if "OTypeint32_t" in elf_name:
            OType = torch.int32
            OType_np = np.int32
        elif "OTypeFP16" in elf_name:
            OType = torch.float16
            OType_np = np.float16
        elif "OTypeFP8" in elf_name:
            OType = torch.float8_e4m3fn
            OType_np = np.float8_e4m3fn
        else:
            OType = torch.uint32
            OType_np = np.uint32

        input_data = torch.randn(input_data_shape, dtype=torch.float32) * 10.0 - 5.0
        offset_data = torch.randint(0, gKs, size=input_offset_shape, dtype=torch.long)
        
        golden = torch.zeros((gMs,gNs))
        # np.savetxt(path+"/"+"golden.bin", golden.numpy(), fmt="%.6f", delimiter=" ")
        input_data = input_data.to(DType)
        golden = golden.to(DType)
        for i in range(gMs) :
            golden[i] = input_data[offset_data[i][0]]
        offset_data = offset_data.to(OType)
        print(f"✅ offset_data[0:5]: {offset_data[0:5]}")
        print(f"✅ input_data[0:5]: {input_data[317][0:5]}")
        print(f" input_data size: {input_data.size()}")
        print(f" offset_data size: {offset_data.size()}")
        print(f" golden size: {golden.size()}")
        print('Start to store input data & golden data into: ', path)
        input_data.numpy().tofile(path+"/"+"input.bin")
        offset_data.numpy().tofile(path+"/"+"input_offset.bin")
        golden.numpy().tofile(path+"/"+"golden.bin")

def run_qemu(elf):
    print('Start to run qemu----------')
    try:  
        if os.path.exists(elf):
            if args.plat == "cpu":
                print('plat == "cpu"')
                proc = subprocess.Popen([elf], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True, preexec_fn=os.setsid)
            else:
                proc = subprocess.Popen([qemu + qemu_args + elf], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True, preexec_fn=os.setsid)
            stdout, stderr = proc.communicate(timeout=1200)
            output = stdout + stderr
            # status = "pass" if "test pass" in output else "fail"
            status = "pass" if proc.returncode == 0 else "fail"
            print('qemu status:', status)
            return (elf, status, output)
        else:
            print('elf not exist')
            return (elf, "not exist", "")
    
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.communicate()  # 确保进程资源被正确清理
        return (elf, "timeout", "Timeout expired")
    
    except Exception as e:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.communicate()  # 确保进程资源被正确清理
        return (elf, "error", str(e))

def result_compare(cmp_data, golden_data):
    print("cmp data: ",cmp_data)
    if "accelerator_fusion_fa1" in cmp_data:
        res = np.fromfile(cmp_data, dtype=DType_np)
        res_ref = np.fromfile(golden_data, dtype=DType_np)        
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

    os.system(f"rm -rf {cmp_data_path};mkdir -p {cmp_data_path}")

    os.makedirs(cmp_data_path, exist_ok=True)
    gen_input_and_golden(elf_name, cmp_data_path)
    _ , status, _ = run_qemu(elf)
    if status == "pass":
        cmp_data = cmp_data_path+"/output.bin"
        golden_data = cmp_data_path+"/golden.bin"
        if os.path.exists(cmp_data):
            chk_status, loss = result_compare(cmp_data, golden_data)
            return elf_name, status, chk_status, loss
        else:
            return elf_name, status, "output not exist", "NaN"
    else:
        return elf_name, status, "not chk", "NaN"

def log_result(elf, status, chk_status, loss):
    if status=="pass" and chk_status=="pass":
        statics["pass"].append(f"{elf} → loss:{loss}")
    else:
        statics["fail"].append(f"{elf} → run_status: {status.upper()} chk_status: {chk_status.upper()} loss:{loss}")

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
            elf, status, chk_status, loss = future.result()
            print(f"{elf} → run_status: {status.upper()} chk_status: {chk_status.upper()} loss:{loss}")
            log_result(elf, status, chk_status, loss)

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



