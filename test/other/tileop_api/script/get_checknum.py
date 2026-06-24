#!/usr/bin/python3
import os
import sys
import argparse
import test as test_py


class GetChecknum:

    def __init__(self):

        self.pwd_path = os.getcwd()

    def get_checknum(self, src_dir, log_dir):

        subName_elfs = self.cmp(src_dir)
        self.run(subName_elfs)

    def cmp(self, BenchMark_dir):

        subName_elfs = {}

        Process = test_py.Process()
        test_lists = Process.process(BenchMark_dir)
        Compile = test_py.Compile()
        subName_cFiles = Compile.get_file(test_lists, BenchMark_dir, ".cpp")
        include_dir = os.path.join(BenchMark_dir, "include")

        for sub_name, c_file in subName_cFiles.items():
            elf_file = os.path.join(self.pwd_path, sub_name + ".bin")
            shell = "g++ -std=c++20 -O2 -D__cpu_sim__ {} -I{} -o {}".format(c_file, include_dir, elf_file)
            print(shell)
            os.system(shell)
            subName_elfs[sub_name] = elf_file

        return subName_elfs
      
    def run(self, subName_elfs):

        for sub_name, elf_path in subName_elfs.items():
            log_path = os.path.join(log_dir, sub_name + ".log")
            shell = "{} > {}".format(elf_path, log_path)
            print(shell)
            os.system(shell)

def main(src_dir, log_dir):

    GetChecknum_m = GetChecknum()
    GetChecknum_m.get_checknum(src_dir, log_dir)

if __name__ == "__main__":
    argParser = argparse.ArgumentParser(description="xx")
    argParser.add_argument("-i", type=str, help="src dir")
    argParser.add_argument("-o", type=str, help="out dir")

    args = argParser.parse_args()

    src_dir = args.i
    log_dir = args.o

    main(src_dir, log_dir)
