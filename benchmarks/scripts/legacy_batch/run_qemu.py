#!/usr/bin/python3

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import os
import signal 
import psutil

MAX_WORKERS = 20  #parallel thread num depend on your machine

# /remote/lms60/c00622284/qemu/LinxBlockModel/build/qemu-lin -run-supertest -blk_optimize force_tb_chained -s 128M 
# -d cpu,in_asm,exec,nochain,linxmem -D run.log
qemu = "/remote/lms60/c00622284/qemu/LinxBlockModel/build/qemu-linx"
qemu_args = " -blk_optimize force_tb_chained -s 4096M "

def run_elf(path):
    try:        
        proc = subprocess.Popen([qemu + qemu_args + path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True, preexec_fn=os.setsid)

        stdout, stderr = proc.communicate(timeout=1200)
        
        output = stdout + stderr
        #status = "pass" if "test pass" in output else "fail"
        status = "pass" if proc.returncode == 0 else "fail"
        return (path, status, output)
    
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        #proc.communicate()  # 确保进程资源被正确清理
        return (path, "timeout", "Timeout expired")
    
    except Exception as e:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        #proc.communicate()  # 确保进程资源被正确清理
        return (path, "error", str(e))
    except KeyboardInterrupt:
        os.system("ps -u c00622284 | grep qemu-linx | awk '{print $1}' | xargs kill -9")

def log_result(path, status, out_list, output):
    if status == "pass":
        out_list["pass"].append(path)
    elif status == "fail":
        out_list["fail"].append(path+" "+output)
    elif status == "timeout":
        out_list["fail"].append(path+" "+"time out")
    elif status == "error":
        out_list["fail"].append(path+" "+"exec error")

def kill_process_by_keyword(keyword):
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # 检查进程名称或命令行中是否包含关键字
            if keyword in proc.info['name'] or any(keyword in arg for arg in proc.info['cmdline']):
                print(f"Killing process {proc.info['name']} (PID: {proc.info['pid']})")
                proc.terminate()
        except (psutil.NoSuchProcess, psutil.AaccelssDenied, psutil.ZombieProcess):
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QEMU Scripts")
    parser.add_argument("-i", dest="list", default="all.list", type=str, help="elf list path")
    parser.add_argument("-o", dest="out",  default="qemu_run.log", type=str, help="out log name")
    parser.add_argument("-m", dest="qemu", default="", type=str, help="qemu-linx path")
    args = parser.parse_args()

    out_list = {"pass":[], "fail":[]}
    if args.qemu != "":
        qemu = args.qemu
    with open(args.list, "r") as f:
        elf_paths = [line.strip() for line in f if line.strip()]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_elf, path): path for path in elf_paths}
        for future in as_completed(futures):
            path, status, output = future.result()
            print(f"{path} → {status.upper()} {output}")
            log_result(path, status, out_list, output)

    with open(args.out, "w") as f:
        f.write(f"run cmd:\n")
        f.write(f"{qemu+qemu_args}\n\n")
        f.write(f"pass list:\n")
        for path in out_list["pass"]:
            f.write(f"{path}\n")

        f.write(f"\n\n\n")

        f.write(f"fail list:\n")
        for path in out_list["fail"]:
            f.write(f"{path}\n")

        pass_num = len(out_list["pass"])
        fail_num = len(out_list["fail"])
        f.write(f"\nQemu Run Summary: \n")
        f.write(f"\npass num: {pass_num}\n")
        f.write(f"\nfail num: {fail_num}\n")
        #tmp add
        pass_matmul_frac_num = sum("matmul_FRAC" in s for s in out_list["pass"])
        fail_matmul_frac_num = sum("matmul_FRAC" in s for s in out_list["fail"])
        pass_matmul_mask_num = sum("matmul_MASK" in s for s in out_list["pass"])
        fail_matmul_mask_num = sum("matmul_MASK" in s for s in out_list["fail"])
        pass_fa_frac_num = sum("flash_attention_FRAC" in s for s in out_list["pass"])
        fail_fa_frac_num = sum("flash_attention_FRAC" in s for s in out_list["fail"])
        pass_fa_all_num = sum("flash_attention_ALL" in s for s in out_list["pass"])
        fail_fa_all_num = sum("flash_attention_ALL" in s for s in out_list["fail"])
        pass_fa_all_frac_num = 0
        fail_fa_all_frac_num = 0
        f.write(f"\npass matmul_frac num: {pass_matmul_frac_num}\n")
        f.write(f"\nfail matmul_frac num: {fail_matmul_frac_num}\n")
        f.write(f"\npass matmul_mask num: {pass_matmul_mask_num}\n")
        f.write(f"\nfail matmul_mask num: {fail_matmul_mask_num}\n")
        f.write(f"\npass flash_attention_frac num: {pass_fa_frac_num}\n")
        f.write(f"\nfail flash_attention_frac num: {fail_fa_frac_num}\n")
        f.write(f"\npass flash_attention_all num: {pass_fa_all_num}\n")
        f.write(f"\nfail flash_attention_all num: {fail_fa_all_num}\n")
