#!/usr/bin/python3

import os
import sys
import argparse
import json
import math
import subprocess
from pathlib import Path

OTHER_ROOT = Path(__file__).resolve().parent.parent

def compile():
    compile_dir = OTHER_ROOT / "tileop_api"
    print(compile_dir)
    subprocess.run("./compile.all", cwd=compile_dir, shell=True)
    return

def run():

    return

def verify():
    
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run CI Scripts")
    parser.add_argument("-plat", dest="plat", default="linx", type=str, help="set platform for compile: jcore/cpu_sim/arm_sme")
    parser.add_argument("-cmp_dir", dest="cmp_dir", default="/usr/bin", type=str, help="set compiler: default /usr/bin")
    parser.add_argument("-cxx_ver", dest="cxx_ver", default="c++20", type=str, help="set cxx version")
    parser.add_argument("-cc", dest="cc", default="clang", type=str, help="set c compiler")
    parser.add_argument("-cxx", dest="cxx", default="clang++", type=str, help="set c++ compiler")
    args = parser.parse_args()

    os.environ["PLAT"] = args.plat
    os.environ["COMPILER_DIR"] = args.cmp_dir
    os.environ["CXX_VER"] = "-std="+args.cxx_ver
    os.environ["CC"] = args.cmp_dir+"/"+args.cc
    os.environ["CXX"] = args.cmp_dir+"/"+args.cxx

    compile()
    run()
    verify()
