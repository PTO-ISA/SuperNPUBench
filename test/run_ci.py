#!/usr/bin/python3

import os
import sys
import argparse
import json
import math
import subprocess

def compile():
    os.chdir(os.path.dirname(__file__)+"/tileop_api")
    print(os.getcwd())
    subprocess.run("./compile.all", shell=True)
    return

def run():

    return

def verify():
    
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run CI Scripts")
    parser.add_argument("-plat", dest="plat", default="linx", type=str, help="set platform for compile: jcore/cpu_sim/arm_sme")
    parser.add_argument("-cmp_dir", dest="cmp_dir", default="/usr/bin", type=str, help="set compiler: default /usr/bin")
    parser.add_argument("-cc_ver", dest="cc_ver", default="c++20", type=str, help="set cxx version")
    parser.add_argument("-cc", dest="cc", default="clang", type=str, help="set c compiler")
    parser.add_argument("-cxx", dest="cxx", default="clang++", type=str, help="set c++ compiler")
    args = parser.parse_args()

    os.environ["PLAT"] = args.plat
    os.environ["COMPILER_DIR"] = args.cmp_dir
    os.environ["CC_VER"] = "-std="+args.cc_ver
    os.environ["CC"] = args.cmp_dir+"/"+args.cc
    os.environ["CXX"] = args.cmp_dir+"/"+args.cxx

    compile()
    run()
    verify()

    