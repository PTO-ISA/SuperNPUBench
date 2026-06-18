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
#4 跑qemu 读取src0, src1 产生dst -> /remote/lms60/c00622284/janus/JanusCoreBench/compare/kernel_matmul_MASK_M128_N1024_K128_tM32_tK32_tN32/res.bin
#5 跑res_compare.py
#6 输出一个报告.

MAX_WORKERS = 20  #parallel thread num depend on your machine
qemu = "/remote/lms60/c00622284/qemu/LinxBlockModel/build/qemu-linx"
qemu_args = " -blk_optimize force_tb_chained -s 4096M "

cmp_root = os.path.abspath(os.path.dirname(__file__)+"../../../compare")
script_path = os.path.abspath(os.path.dirname(__file__))

statics = {"pass":[], "fail":[]}

def map_rule(elf_name):
    pass

def gen_src_and_golden(elf_name, path):
    # print(f"gen golden data: {elf_name}")
    if "matmul_MASK" in elf_name:
        match = re.search(r'_M(\d+)_N(\d+)_K(\d+)', elf_name)
        M = int(match.group(1))
        N = int(match.group(2))
        K = int(match.group(3))
        # print(f"M = {M}, N = {N}, K = {K}")
        torch.manual_seed(123)
        if "FP8" in elf_name:
            src0 = torch.randn((M,K), dtype=torch.float8_e4m3fn) * 10.0 + 10.0
            src1 = torch.randn((K,N), dtype=torch.float8_e4m3fn) * 10.0 + 10.0
        elif "FP16" in elf_name:
            src0 = torch.randn((M,K), dtype=torch.float16) * 10.0 + 10.0
            src1 = torch.randn((K,N), dtype=torch.float16) * 10.0 + 10.0
        else:
            src0 = torch.randn((M,K), dtype=torch.float32) * 10.0 + 10.0
            src1 = torch.randn((K,N), dtype=torch.float32) * 10.0 + 10.0
        # src0 = torch.ones((M,K), dtype=torch.float32)
        # src1 = torch.ones((K,N), dtype=torch.float32)
        src0.numpy().tofile(path+"/"+"src0.bin")
        src1.numpy().tofile(path+"/"+"src1.bin")
        golden = torch.matmul(src0, src1)
        # np.savetxt(path+"/"+"golden.bin", golden.numpy(), fmt="%.6f", delimiter=" ")
        golden.numpy().tofile(path+"/"+"golden.bin")
    elif "flash_attention" in elf_name:
        match = re.search(r'_B(\d+)_S(\d+)_D(\d+)', elf_name)
        B = int(match.group(1))
        S = int(match.group(2))
        D = int(match.group(3))
        # print(f"B = {B}, S = {S}, D = {D}")
        torch.manual_seed(210)
        q = torch.randn((B*S,D), dtype=torch.float32) * 10.0 + 10.0
        k = torch.randn((B*S,D), dtype=torch.float32) * 10.0 + 10.0
        v = torch.randn((B*S,D), dtype=torch.float32) * 10.0 + 10.0
        # q = torch.ones((B*S,D), dtype=torch.float32) * 0.1
        # k = torch.ones((B*S,D), dtype=torch.float32) * 0.1
        # v = torch.ones((B*S,D), dtype=torch.float32) * 0.1
        q.numpy().tofile(path+"/"+"srcq.bin")
        k.numpy().tofile(path+"/"+"srck.bin")
        v.numpy().tofile(path+"/"+"srcv.bin")
        scale = 1.0 / math.sqrt(D)
        softmax = nn.Softmax(dim=1)
        golden = torch.mm(softmax(torch.mm(q, torch.t(k)) * scale), v)
        golden = golden.contiguous()
        golden.numpy().tofile(path+"/"+"golden.bin")
    elif "kernel_softmax" in elf_name:
        match = re.search(r'_M(\d+)_N(\d+)_', elf_name)
        M = int(match.group(1))
        N = int(match.group(2))
        torch.manual_seed(379)
        src = torch.randn((M,N), dtype=torch.float32) * 10.0 + 10.0
        src.numpy().tofile(path+"/"+"src.bin")
        softmax = nn.Softmax(dim=1)
        golden = softmax(src)
        golden.numpy().tofile(path+"/"+"golden.bin")
    elif "kernel_gemm" in elf_name:
        match = re.search(r'_M(\d+)_N(\d+)_K(\d+)', elf_name)
        M = int(match.group(1))
        N = int(match.group(2))
        K = int(match.group(3))
        torch.manual_seed(127)
        src0 = torch.randn((M,K), dtype=torch.float32) * 10.0 + 10.0
        src1 = torch.randn((K,N), dtype=torch.float32) * 10.0 + 10.0 
        src2 = torch.randn((1,N), dtype=torch.float32) * 10.0 + 10.0
        src2 = src2.expand(M,N)

        golden = torch.mm(src0, src1)
        if "bias_on" in elf_name:
            golden = golden + src2

        if "relu_on" in elf_name:
            golden = torch.relu(golden)

        golden = golden.contiguous()
        src0.numpy().tofile(path+"/"+"src0.bin")
        src1.numpy().tofile(path+"/"+"src1.bin")
        src2.numpy().tofile(path+"/"+"src2.bin")
        golden.numpy().tofile(path+"/"+"golden.bin")

    elif "rms_norm" in elf_name:
        match = re.search(r'_M(\d+)_N(\d+)_', elf_name)
        M = int(match.group(1))
        N = int(match.group(2))
        torch.manual_seed(477)
        src = torch.randn((M,N), dtype=torch.float32) * 10.0 + 10.0
        src.numpy().tofile(path+"/"+"src.bin")
        softmax = nn.Softmax(dim=1)
        golden = softmax(src)
        golden.numpy().tofile(path+"/"+"golden.bin")
    
    elif "accelerator_fusion_fa1" in elf_name:
        B = 1
        S = 512
        D = 64
        # print(f"B = {B}, S = {S}, D = {D}")
        torch.manual_seed(210)
        q = torch.randn((B*S,D), dtype=torch.float16) * 10.0 + 10.0
        k = torch.randn((B*S,D), dtype=torch.float16) * 10.0 + 10.0
        v = torch.randn((B*S,D), dtype=torch.float16) * 10.0 + 10.0
        q.numpy().tofile(path+"/"+"srcq.bin")
        k.numpy().tofile(path+"/"+"srck.bin")
        v.numpy().tofile(path+"/"+"srcv.bin")
        scale = 1.0 / math.sqrt(D)
        softmax = nn.Softmax(dim=1)
        golden = torch.mm(softmax(torch.mm(q, torch.t(k)) * scale), v)
        print(elf_name)
        if "OPT3" in elf_name or "OPT4" in elf_name or "OPT5" in elf_name or "DCORE" in elf_name or "2D_UNROLL" in elf_name:
            golden = golden.t().contiguous()
        else:
            golden = golden.contiguous()
        golden.numpy().tofile(path+"/"+"golden.bin") 

def run_qemu(elf):
    try:  
        if os.path.exists(elf):
            if args.plat == "cpu":
                proc = subprocess.Popen([elf], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True, preexec_fn=os.setsid)
            else:
                proc = subprocess.Popen([qemu + qemu_args + elf], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True, preexec_fn=os.setsid)
            stdout, stderr = proc.communicate(timeout=1200)
            output = stdout + stderr
            # status = "pass" if "test pass" in output else "fail"
            status = "pass" if proc.returncode == 0 else "fail"
            return (elf, status, output)
        else:
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
    if "accelerator_fusion_fa1" in cmp_data:
        res = np.fromfile(cmp_data, dtype=np.float16)
        res_ref = np.fromfile(golden_data, dtype=np.float16)        
    else:
        res = np.fromfile(cmp_data, dtype=np.float32)
        res_ref = np.fromfile(golden_data, dtype=np.float32)
    loss = np.mean((res - res_ref) ** 2).item()
    if loss < 0.1:
        return "pass", loss
    else:
        return "fail", loss

def check_elf(elf):
    elf_name = os.path.basename(elf).replace(".elf", "").strip()
    cmp_data_path = cmp_root+"/"+elf_name

    os.system(f"rm -rf {cmp_data_path};mkdir -p {cmp_data_path}")

    os.makedirs(cmp_data_path, exist_ok=True)
    gen_src_and_golden(elf_name, cmp_data_path)
    _ , status, _ = run_qemu(elf)
    if status == "pass":
        cmp_data = cmp_data_path+"/res.bin"
        golden_data = cmp_data_path+"/golden.bin"
        if os.path.exists(cmp_data):
            chk_status, loss = result_compare(cmp_data, golden_data)
            return elf_name, status, chk_status, loss
        else:
            return elf_name, status, "res not exist", "NaN"
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



