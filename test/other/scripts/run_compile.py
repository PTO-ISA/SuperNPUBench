#!/usr/bin/python3

import os
import sys
import argparse
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

MAX_WORKERS = 20  #parallel thread num depend on your machine

root = "JanusCoreBench/test/"

compile_result = {"pass":[], "fail":[], "timeout":[]}

compile_list = [
    "./tileop_test/compile.all",
    "./kernel/compile_softmax.all",
    "./kernel/compile_gemm.all",
    "./kernel/compile_linear.all",
    "./kernel/compile_matmul.all",
    "./kernel/compile_flash_attention.all",
    "./lmbench/compile_mem.all",
    "./vec/compile_lat_bw.all",
    "./cube/compile.all",
    "./deepseek/compile.all",
    "./accelerator/vec_simd/compile.all",
    "./accelerator/fusion/compile.all",
]

def cmd_config_parse(list):
    pass

def compile_elf(list):
    cmd_path = list
    cmd_dir = os.path.dirname(os.path.abspath(cmd_path)).split(root, 1)[-1]
    cmd = os.path.basename(os.path.abspath(cmd_path)).split(".")[0] + "_" + cmd_dir.replace("/", "_")
    print(f"processing {cmd}...")

    compile_out = ""
    with open(f"../cm_log/{cmd}.log", "a+") as f:
        with open(cmd_path, "r") as f2:
            for line in f2:
                line = line.strip()
                if "make TESTCASE" in line and not line.startswith("#"):
                    try:
                        result = subprocess.run(f"cd {cmd_dir};{line};", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60, universal_newlines=True)
                        compile_out = result.stdout
                        f.writelines(compile_out)
                        if ".elf" in compile_out and "error" not in compile_out:
                            elf_paths = re.findall(r'\S+\.elf', compile_out)
                            elf_name = [os.path.basename(path) for path in elf_paths]
                            elf_name = ''.join(elf_name)
                            compile_result["pass"].append(elf_name) 
                        else:
                            compile_result["fail"].append(f"{cmd_dir}:{line}")
                    except subprocess.TimeoutExpired:
                            compile_result["timeout"].append(f"{cmd_dir}:{line}")
                        
    print(f"{cmd} finished...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Compile_all Scripts")
    parser.add_argument("-plat", dest="plat", default="linx", type=str, help="set platform for compile: jcore/cpu_sim/arm_sme")
    args = parser.parse_args()

    os.chdir(os.path.dirname(__file__)+"/../")
    print(f"cur_dir is {os.getcwd()}")
    os.system("rm -rf ../cm_log;mkdir ../cm_log")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(compile_elf, list): list for list in compile_list}
        for future in as_completed(futures):
            pass
            #path, status, output = future.result()
    
    #Summary
    with open(f"../cm_log/compile_summary.log", "w") as f:
        f.write(f"Compile Summary:\n")
        f.write(f"pass: {len(compile_result['pass'])}\n")
        f.write(f"fail: {len(compile_result['fail'])}\n")

        pass_matmul_num = sum("matmul" in s for s in compile_result["pass"])
        fail_matmul_num = sum("TESTCASE=matmul" in s for s in compile_result["fail"])
        pass_fa_frac_num = sum("flash_attention_FRAC" in s for s in compile_result["pass"])
        fail_fa_frac_num = sum("TESTCASE=flash_attention MODE=FRAC" in s for s in compile_result["fail"])
        pass_fa_all_num = sum("flash_attention_ALL" in s for s in compile_result["pass"])
        fail_fa_all_num = sum("TESTCASE=flash_attention MODE=ALL" in s for s in compile_result["fail"])
        f.write(f"pass_matmul_num: {pass_matmul_num}\n")
        f.write(f"fail_matmul_num: {fail_matmul_num}\n")
        f.write(f"pass_flash_attention_frac_num: {pass_fa_frac_num}\n")
        f.write(f"fail_flash_attention_frac_num: {fail_fa_frac_num}\n")
        f.write(f"pass_flash_attention_all_num: {pass_fa_all_num}\n")
        f.write(f"fail_flash_attention_all_num: {fail_fa_all_num}\n")

        f.write(f"timeout: {len(compile_result['timeout'])}\n")
        
        f.write(f"pass case:\n")
        for case in compile_result['pass']:
            f.write(f"{case}\n")
        f.write(f"\n")
        f.write(f"fail case:\n")
        for case in compile_result['fail']:
            f.write(f"{case}\n")
        f.write(f"\n")
        f.write(f"timeout 60s case:\n")
        for case in compile_result['timeout']:
            f.write(f"{case}\n")
