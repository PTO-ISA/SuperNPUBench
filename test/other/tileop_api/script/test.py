#!/usr/bin/python3
import os
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor
import shutil

# 最大并行数
g_max_threads = 10
# 全局默认目录名称
g_tmp_name = "tmp"
g_elf_dir_name = "elf_linx"
g_run_dir_name = "run_log"
g_error_dir_name = "error_log"


class Process:

    def __init__(self):

        self.unit_lists = []
        self.asc_lists = []

    def process(self, BenchMark_dir):

        self.get_unit_lists(BenchMark_dir)
        test_lists = self.unit_lists + self.asc_lists
        return test_lists

    def get_unit_lists(self, BenchMark_dir):

        kerword = ".cpp"
        src_dir = os.path.join(BenchMark_dir, "test/tileop_api/src")
        for root, dirs, files in os.walk(src_dir):
            for file_name in files:
                if kerword in file_name:
                    sub_name = file_name.split(kerword)[0]
                    self.unit_lists.append(sub_name)
        
        self.unit_lists.sort()

class Compile:

    def __init__(self):

        self.pwd_path = os.getcwd()
        self.elf_dir = os.path.join(self.pwd_path, g_elf_dir_name)
        self.tmp_dir = os.path.join(self.pwd_path, g_tmp_name)
        self.cmp_paras = "-std=c++20 -fenable-matrix -O2"

    def compile(self, BenchMark_dir, llvm_dir, pmc_able):

        if not os.path.exists(llvm_dir):
            sys.exit("Error! {} not exist.".format(llvm_dir))
        if not os.path.exists(BenchMark_dir):
            sys.exit("Error! {} not exist.".format(BenchMark_dir))

        testProcess = Process()
        test_lists = testProcess.process(BenchMark_dir)
        llvm, pmc_paras = self.readying(llvm_dir, BenchMark_dir, pmc_able)

        # cpp -> elf
        src_dir = os.path.join(BenchMark_dir, "test/")
        subName_cFiles = self.get_file(test_lists, src_dir, ".cpp")
        self.cmp_elf(llvm, subName_cFiles, pmc_paras)
        check_cmps = self.check_cmp(test_lists, self.elf_dir, ".bin")
        # elf -> asm
        subName_elfs = self.get_file(test_lists, self.elf_dir, ".bin")
        self.objdump_elf(llvm_dir, subName_elfs)

        self.out_result(test_lists, check_cmps)

        return self.elf_dir

    def readying(self, llvm_dir, src_dir, pmc_able):

        self.cp_hpp_to_llvmlib(llvm_dir, src_dir)

        # mkdir elf dir
        if os.path.exists(self.elf_dir):
            shutil.rmtree(self.elf_dir)
            os.mkdir(self.elf_dir)
        else:
            os.mkdir(self.elf_dir)

        # init tmp dir
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
            os.mkdir(self.tmp_dir)
        else:
            os.mkdir(self.tmp_dir)

        llvm = os.path.join(llvm_dir, "bin/clang++")
        pmc_paras = ""
        if pmc_able != "N":
            pmc_paras = "-DLINX_PMC"

        return llvm, pmc_paras

    def cp_hpp_to_llvmlib(self, llvm_dir, src_dir):

        include_dir = os.path.join(src_dir, "include")
        llvm_lib = os.path.join(llvm_dir, "lib/clang/15.0.4/include/tileop-api")
        shell = "rm -r {}/*; cp -r {}/* {}".format(llvm_lib, include_dir, llvm_lib)
        print(shell)
        os.system(shell)

    def run(self, shell):
        os.system(shell)

    def cmp_elf(self, llvm, subName_cFiles, pmc_paras):

        tasks = []
        with ThreadPoolExecutor(max_workers=g_max_threads) as executor:
            for sub_name, file in subName_cFiles.items():
                elf_file = os.path.join(self.elf_dir, sub_name + ".bin")
                shell = "{} {} -mlxbc {} {} -o {}".format(
                    llvm, self.cmp_paras, pmc_paras, file, elf_file
                )
                print(shell)
                task = executor.submit(self.run, shell)
                tasks.append(task)

        for it in tasks:
            it.result()

    def objdump_elf(self, llvm_dir, subName_elfs):

        objdump_path = os.path.join(llvm_dir, "bin/llvm-objdump")
        tasks = []
        with ThreadPoolExecutor(max_workers=g_max_threads) as executor:
            for sub_name, file in subName_elfs.items():
                asm_file = os.path.join(self.elf_dir, sub_name + ".asm")
                shell = "{} -d {} > {}".format(objdump_path, file, asm_file)
                print(shell)
                task = executor.submit(self.run, shell)
                tasks.append(task)

        for it in tasks:
            it.result()

    def get_file(self, test_cases, src_dir, kerword):

        subName_files = {}

        for root, dirs, files in os.walk(src_dir):
            for file_name in files:
                if kerword in file_name:
                    file_path = os.path.join(root, file_name)
                    sub_name = file_path.split("/")[-1].split(kerword)[0]
                    if sub_name in test_cases:
                        subName_files[sub_name] = file_path

        return subName_files

    def check_cmp(self, test_cases, dir_path, kerword):

        check_cmps = {}

        subName_files = {}
        for root, dirs, files in os.walk(dir_path):
            for file_name in files:
                if kerword in file_name:
                    sub_name = file_name.split(kerword)[0]
                    file_path = os.path.join(root, file_name)
                    subName_files[sub_name] = file_path

        # 检查错误
        for sub_name in test_cases:
            # 编译错误
            if subName_files.get(sub_name):
                check_cmps[sub_name] = "True"
            else:
                check_cmps[sub_name] = "False"

        return check_cmps

    def if_all_error(self, subName_results):

        for sub_name, result in subName_results.items():
            if result == "True":
                return True

        return False

    def out_result(self, test_cases, check_cmps):

        outCsv_name = "tileop_complie.csv"
        fd = open(outCsv_name, "w", encoding="utf8")
        fd.write("tileop,Compile\n")

        num_sum = 0
        cmp_err = 0

        for sub_name in test_cases:
            num_sum += 1
            if check_cmps[sub_name] != "True":
                cmp_err += 1
            fd.write(sub_name + "," + check_cmps[sub_name] + "\n")

        fd.close()
        print("***************************************************************")
        print("For details, see: {}".format(outCsv_name))
        print("Total:           ", num_sum)
        print("Compile PASS:    ", num_sum - cmp_err)
        print("***************************************************************")


class RUN_LINX:

    def __init__(self):

        self.qemu_timeout = "20m"
        self.gfrun_timeout = "1h"
        self.pwd_path = os.getcwd()
        self.run_log_dir = ""
        self.error_log_dir = ""

    def run_linx(self, qemu, gfrun, elf_dir):

        if qemu == "None" and gfrun == "None":
            sys.exit("Error! Both -qemu and -gfrun cannot be empty!")

        run_paras = ""
        model = ""
        if qemu != "None":
            model = qemu
            time_out = self.qemu_timeout
            run_paras = "-blk_optimize force_tb_chained"
            self.run_log_dir = os.path.join(self.pwd_path, "qemu_" + g_run_dir_name)
            self.error_log_dir = os.path.join(self.pwd_path, "qemu_" + g_error_dir_name)

            self.run_model(time_out, model, run_paras, elf_dir)

        if gfrun != "None":
            model = gfrun
            time_out = self.gfrun_timeout
            run_paras = "-f"
            self.run_log_dir = os.path.join(self.pwd_path, "gfrun_" + g_run_dir_name)
            self.error_log_dir = os.path.join(
                self.pwd_path, "gfrun_" + g_error_dir_name
            )

            self.run_model(time_out, model, run_paras, elf_dir)

    def run_model(self, time_out, model, run_paras, elf_dir):

        self.mkdir(self.run_log_dir)
        self.mkdir(self.error_log_dir)

        elf_paths = self.get_elf(elf_dir)

        tasks = []
        with ThreadPoolExecutor(max_workers=g_max_threads) as executor:
            for elf_path in elf_paths:
                file_name = elf_path.split("/")[-1].split(".bin")[0] + ".log"
                run_log_path = os.path.join(self.run_log_dir, file_name)
                file_name = elf_path.split("/")[-1].split(".bin")[0] + ".err"
                error_log_path = os.path.join(self.error_log_dir, file_name)

                shell = "timeout {time} {m} {rp} {elf} 1> {rl} 2> {err}".format(
                    time=time_out,
                    m=model,
                    rp=run_paras,
                    elf=elf_path,
                    rl=run_log_path,
                    err=error_log_path,
                )

                print(shell)
                task = executor.submit(self.run, shell)
                tasks.append(task)

            for it in tasks:
                it.result()

    def run(self, shell):
        os.system(shell)

    def mkdir(self, dir_path):

        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)
        else:
            os.mkdir(dir_path)

    def get_elf(self, elf_dir):

        elf_paths = []

        for root, dirs, files in os.walk(elf_dir):
            for file_name in files:
                if ".bin" in file_name:
                    file_path = os.path.join(root, file_name)
                    elf_paths.append(file_path)

        return elf_paths


class Checknum:

    def __init__(self):

        self.pwd_path = os.getcwd()

    def checknum(self, qemu, gfrun, elf_dir, BenchMark_dir):

        testProcess = Process()
        test_lists = testProcess.process(BenchMark_dir)

        compile = Compile()
        check_links = compile.check_cmp(test_lists, elf_dir, ".bin")

        qemu_check_runs, gfrun_check_runs = self.check_run(qemu, gfrun)
        qemu_check_checknums, gfrun_check_checknums = self.check_checknum(qemu, gfrun)

        self.out_result(
            test_lists,
            qemu,
            gfrun,
            check_links,
            qemu_check_runs,
            gfrun_check_runs,
            qemu_check_checknums,
            gfrun_check_checknums,
        )

    def check_run(self, qemu, gfrun):

        qemu_check_runs = {}
        gfrun_check_runs = {}

        log_dir = ""
        if qemu != "None":
            log_dir = os.path.join(self.pwd_path, "qemu_" + g_run_dir_name)
            qemu_check_runs = self.check_run_help_qemu(log_dir)

        if gfrun != "None":
            error_log_dir = os.path.join(self.pwd_path, "gfrun_" + g_error_dir_name)
            run_log_dir = os.path.join(self.pwd_path, "gfrun_" + g_run_dir_name)
            # gfrun 既需要检查err_log, 还需要检查run_log，run_log的结果可以覆盖err_log
            gfrun_check_runs = self.check_run_help_gfrun(error_log_dir)
            gfrun_check_runs = self.check_run_help_gfrun(run_log_dir)

        return qemu_check_runs, gfrun_check_runs

    def check_run_help_qemu(self, error_log_dir):

        check_runs = {}

        for root, dirs, files in os.walk(error_log_dir):
            for file_name in files:
                sub_name = file_name.split(".")[0]
                file_path = os.path.join(root, file_name)
                if os.path.getsize(file_path) == 0:
                    check_runs[sub_name] = "Pass"
                else:
                    try:
                        with open(file_path) as f:
                            for line in f.readlines():
                                if "mask error" in line:
                                    check_runs[sub_name] = "mask error"
                                elif "Illegal instruction" in line:
                                    check_runs[sub_name] = "Illegal instruction"
                                elif "Segmentation fault" in line:
                                    check_runs[sub_name] = "Segmentation fault"
                                elif "core dumped" in line:
                                    check_runs[sub_name] = "core dumped"
                                else:
                                    check_runs[sub_name] = "Pass"
                    except UnicodeDecodeError:
                        check_runs[sub_name] = "False"
                
        return check_runs

    def check_run_help_gfrun(self, error_log_dir):

        check_runs = {}

        for root, dirs, files in os.walk(error_log_dir):
            for file_name in files:
                sub_name = file_name.split(".")[0]
                file_path = os.path.join(root, file_name)
                with open(file_path) as f:
                    for line in f.readlines():
                        if "mask error" in line:
                            check_runs[sub_name] = "mask error"
                        elif "Illegal instruction" in line:
                            check_runs[sub_name] = "Illegal instruction"
                        elif "Segmentation fault" in line:
                            check_runs[sub_name] = "Segmentation fault"
                        elif "core dumped" in line:
                            check_runs[sub_name] = "core dumped"
                        elif "assert" in line or "Assert" in line or "ASSERT" in line:
                            check_runs[sub_name] = "assert"
                        elif (
                            "Suaccelss to Reach the End of Benchmark!" in line
                        ):  # gfrun运行成功的标志
                            check_runs[sub_name] = "Pass"
                        else:
                            check_runs[sub_name] = "timeout or unknown error"

        return check_runs

    def check_checknum(self, qemu, gfrun):

        qemu_check_checknums = {}
        gfrun_check_checknums = {}

        # get true checknum
        py_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        checknum_true_dir = os.path.join(py_dir, "checknum_true")
        subName_checknums_true = self.get_all_checknum(checknum_true_dir)

        run_linx = RUN_LINX()
        # get qemu checknum
        if qemu != "None":
            qemu_checknum_dir = os.path.join(self.pwd_path, "qemu_" + g_run_dir_name)
            subName_checknums_qemu = self.get_all_checknum(qemu_checknum_dir)

            qemu_check_checknums = self.check_checknum_help(
                subName_checknums_true, subName_checknums_qemu
            )

        if gfrun != "None":
            gfrun_checknum_dir = os.path.join(self.pwd_path, "gfrun_" + g_run_dir_name)
            gfrun_checknum_files = self.get_files(gfrun_checknum_dir)
            # 需要去掉gfrun本身的输出
            self.del_line_files(gfrun_checknum_files)
            subName_checknums_gfrun = self.get_all_checknum(gfrun_checknum_dir)

            gfrun_check_checknums = self.check_checknum_help(
                subName_checknums_true, subName_checknums_gfrun
            )

        return qemu_check_checknums, gfrun_check_checknums

    def check_checknum_help(self, subName_checknums_true, subName_checknums_linx):

        check_checknums = {}

        for sub_name, chechnums_true in subName_checknums_true.items():
            if subName_checknums_linx.get(sub_name):
                check_checknums[sub_name] = "True"
                checknums_linx = subName_checknums_linx[sub_name]
                # 如果结果的精度误差小于等于0.02，则认为功能验证通过
                if len(checknums_linx) != len(chechnums_true):
                    check_checknums[sub_name] = "False"
                else:
                    for i in range(len(chechnums_true)):
                        sub_num = chechnums_true[i] - checknums_linx[i]
                        if sub_num > 0.02:
                            check_checknums[sub_name] = "False"
                            break

        return check_checknums

    def get_all_checknum(self, dir_path):

        subName_checknums = {}

        subName_files = self.get_files(dir_path)
        for sub_name, file_path in subName_files.items():
            chechnum_lists = self.get_checknum(file_path)
            subName_checknums[sub_name] = chechnum_lists

        return subName_checknums

    # 获取checknum，输入日志文件，输出数字列表（日志文件通过空格和换行分割的所有数字内容）
    def get_checknum(self, file_path):

        chechnum_lists = []

        try:
          with open(file_path, "r", encoding="utf8") as f:
              for line in f.readlines():
                  ret = line.split()
                  for str_i in ret:
                      try:
                          num = float(str_i)
                          chechnum_lists.append(num)
                      except ValueError:
                          pass
        except UnicodeDecodeError:
            pass

        return chechnum_lists

    # 递归的获取目录下的所有文件
    def get_files(self, dir_path):

        subName_files = {}

        for root, dirs, files in os.walk(dir_path):
            for file_name in files:
                if ".log" in file_name or ".err" in file_name:
                    sub_name = file_name.split(".")[0]
                    file_path = os.path.join(root, file_name)
                    subName_files[sub_name] = file_path

        return subName_files

    def del_line_files(self, file_paths):

        for file_path in file_paths.values():
            self.del_line_file(file_path)

    def del_line_file(self, file_path):

        file_line = ""
        with open(file_path, "r", encoding="utf8") as f:
            for line in f.readlines():
                if "Starting from" in line or "=" in line:
                    continue
                if ":" in line and "Result" not in line:
                    continue
                file_line += line
        with open(file_path, "w", encoding="utf8") as f:
            f.write(file_line)

        f.close()

    def out_result(
        self,
        test_cases,
        qemu,
        gfrun,
        check_links,
        qemu_check_runs,
        gfrun_check_runs,
        qemu_check_checknums,
        gfrun_check_checknums,
    ):

        check_runs = {}
        check_checknums = {}
        if qemu != "None":
            outCsv_name = "qemu_test.csv"
            check_runs = qemu_check_runs
            check_checknums = qemu_check_checknums

            self.out_result_help(
                outCsv_name, test_cases, check_links, check_runs, check_checknums
            )

        if gfrun != "None":
            outCsv_name = "gfrun_test.csv"
            check_runs = gfrun_check_runs
            check_checknums = gfrun_check_checknums

            self.out_result_help(
                outCsv_name, test_cases, check_links, check_runs, check_checknums
            )

    def out_result_help(
        self, outCsv_name, test_cases, check_links, check_runs, check_checknums
    ):

        fd = open(outCsv_name, "w", encoding="utf8")
        fd.write("TileOP,Compile,Run,Function Verification\n")

        num_sum = 0
        link_err = 0
        run_err = 0
        check_err = 0

        for sub_name in test_cases:
            num_sum += 1
            # 如果不存在，则错误
            if not check_links.get(sub_name):
                check_links[sub_name] = "False"
            if not check_runs.get(sub_name):
                check_runs[sub_name] = "False"
            if not check_checknums.get(sub_name):
                check_checknums[sub_name] = "False"

            # 如果编译失败，则运行是Nan
            if check_links[sub_name] != "True":
                link_err += 1
                check_runs[sub_name] = "/"
            # 如果运行失败，则功能验证是Nan
            if check_runs[sub_name] != "Pass":
                run_err += 1
                check_checknums[sub_name] = "/"
            if check_checknums[sub_name] != "True":
                check_err += 1
            fd.write(
                sub_name
                + ","
                + check_links[sub_name]
                + ","
                + check_runs[sub_name]
                + ","
                + check_checknums[sub_name]
                + "\n"
            )

        fd.close()
        print("***************************************************************")
        print("For details, see:  {}".format(outCsv_name))
        print("Total:        ", num_sum)
        print("Compile PASS: ", num_sum - link_err)
        print("Run PASS:     ", num_sum - run_err)
        print("Verify PASS:  ", num_sum - check_err)
        print("***************************************************************")


# 输入源码和编译工具链测试
def run_cmp_run(src_dir, test_model, llvm_dir, qemu, gfrun, pmc_able):

    if test_model != "cmp" and test_model != "run":
        sys.exit("Error! Unrecognized operation for -m.")

    compile = Compile()
    elf_dir = compile.compile(src_dir, llvm_dir, pmc_able)

    if test_model == "run":
        run_linx = RUN_LINX()
        run_linx.run_linx(qemu, gfrun, elf_dir)

        checknum = Checknum()
        checknum.checknum(qemu, gfrun, elf_dir, src_dir)


# 输入二进制测试
def run_input_elf(src_dir, input_elf_dir, qemu, gfrun):

    if not os.path.exists(input_elf_dir):
        sys.exit("Error! Not find dir: {}".format(input_elf_dir))

    elf_dir = input_elf_dir

    run_linx = RUN_LINX()
    run_linx.run_linx(qemu, gfrun, elf_dir)

    checknum = Checknum()
    checknum.checknum(qemu, gfrun, elf_dir, src_dir)


def main(src_dir, test_model, llvm_dir, qemu, gfrun, pmc_able, input_elf_dir):

    if input_elf_dir == "None":
        # 输入源码和编译工具链测试
        run_cmp_run(src_dir, test_model, llvm_dir, qemu, gfrun, pmc_able)
    else:
        # 输入二进制测试
        run_input_elf(src_dir, input_elf_dir, qemu, gfrun)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser(description="tileop test")
    argParser.add_argument("-src", type=str, help="src dir path, case: /xx/TileOP_Lib/")
    argParser.add_argument(
        "-m", type=str, default="cmp", help="test model: cmp or run, default cmp"
    )
    argParser.add_argument(
        "-pmc", default="N", type=str, help="if open ckpt PMC, N(default) or Y"
    )
    argParser.add_argument(
        "-llvm", type=str, help="llvm dir path, case: /xx/linx_blockisa_llvm"
    )
    argParser.add_argument("-qemu", default="None", type=str, help="qemu-linx path")
    argParser.add_argument("-gfrun", default="None", type=str, help="gfrun path")

    argParser.add_argument(
        "-i",
        default="None",
        type=str,
        help="elf dir, Directly input the binary package without compilation. elf name: xx.c.bin",
    )

    args = argParser.parse_args()

    src_dir = args.src
    test_model = args.m
    llvm_dir = args.llvm
    qemu = args.qemu
    gfrun = args.gfrun
    pmc_able = args.pmc
    input_elf_dir = args.i

    main(src_dir, test_model, llvm_dir, qemu, gfrun, pmc_able, input_elf_dir)
