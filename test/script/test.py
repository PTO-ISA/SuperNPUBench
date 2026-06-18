#!/usr/bin/python3
import os
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor
import shutil

# 最大并行数
g_max_threads = 10


class Process:

    def process(self, src_dir, single_case):
        test_lists = self.get_test_lists(src_dir, single_case)
        return test_lists

    def get_test_lists(self, src_dir, single_case):

        if single_case != "None":
            test_lists = [single_case]
            return test_lists
        test_lists = []
        skip_file = ["MatMul_e4m3.cpp", "TDivs.cpp", "TRowMaxExpand.cpp", "TRowSumExpand.cpp", "TRSqrt.cpp"]
        kerword = ".cpp"
        for root, dirs, files in os.walk(src_dir):
            for file_name in files:
                if file_name in skip_file:
                    continue
                if kerword in file_name and "_py" not in file_name:
                    sub_name = file_name.split(kerword)[0]
                    test_lists.append(sub_name)

        test_lists.sort()

        return test_lists


class Compile:

    def __init__(self):

        self.elf_dir_name = "elf_linx"
        self.cmp_paras = "-std=c++20 -O2"

    def compile(self, BenchMark_dir, src_dir, LLVM_LINX, pmc_able, LLVM_HOST, single_case):

        if not os.path.exists(BenchMark_dir):
            sys.exit("Error! {} not exist.".format(BenchMark_dir))
        if not os.path.exists(src_dir):
            sys.exit("Error! {} not exist.".format(src_dir))

        Process_T = Process()
        test_lists = Process_T.process(src_dir, single_case)

        elf_dir_linx = "None"
        elf_dir_host = "None"
        if LLVM_LINX != "None":
            Compile_Linx_T = Compile_Linx()
            elf_dir_linx = Compile_Linx_T.compile_linx(
                BenchMark_dir, test_lists, LLVM_LINX, pmc_able
            )
        if LLVM_HOST != "None":
            Compile_Host_T = Compile_Host()
            elf_dir_host = Compile_Host_T.compile_host(
                BenchMark_dir, test_lists, LLVM_HOST
            )

        return elf_dir_linx, elf_dir_host

    def cmp_elf(self, LLVM_LINX, subName_cFiles, cmp_paras, elf_dir):

        tasks = []
        with ThreadPoolExecutor(max_workers=g_max_threads) as executor:
            for sub_name, file in subName_cFiles.items():
                elf_file = os.path.join(elf_dir, sub_name + ".bin")
                shell = "{} {} {} -o {}".format(LLVM_LINX, cmp_paras, file, elf_file)
                print(shell)
                task = executor.submit(self.run, shell)
                tasks.append(task)

        for it in tasks:
            it.result()

    def run(self, shell):
        os.system(shell)

    def get_file(self, test_lists, src_dir, kerword):

        subName_files = {}
        skip_dirs = ["accelerator"]
        for root, dirs, files in os.walk(src_dir):
            if any(d in root.split(os.sep) for d in skip_dirs):
                continue
            for file_name in files:
                if kerword in file_name:
                    file_path = os.path.join(root, file_name)
                    sub_name = file_path.split("/")[-1].split(kerword)[0]
                    if sub_name in test_lists:
                        subName_files[sub_name] = file_path

        return subName_files

    def check_cmp(self, test_lists, dir_path, kerword):

        check_cmps = {}

        subName_files = {}
        for root, dirs, files in os.walk(dir_path):
            for file_name in files:
                if kerword in file_name:
                    sub_name = file_name.split(kerword)[0]
                    file_path = os.path.join(root, file_name)
                    subName_files[sub_name] = file_path

        # 检查错误
        for sub_name in test_lists:
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

    def out_result(self, test_lists, check_cmps, out_filename):

        fd = open(out_filename, "w", encoding="utf8")
        fd.write("tileop,Compile\n")

        num_sum = 0
        cmp_err = 0

        for sub_name in test_lists:
            num_sum += 1
            if check_cmps[sub_name] != "True":
                cmp_err += 1
            fd.write(sub_name + "," + check_cmps[sub_name] + "\n")

        fd.close()
        print("***************************************************************")
        print("For details, see: {}".format(out_filename))
        print("Total:           ", num_sum)
        print("Compile PASS:    ", num_sum - cmp_err)
        print("***************************************************************")


class Compile_Linx:

    def __init__(self):

        self.Compile_T = Compile()
        self.elf_dir = os.path.join(os.getcwd(), self.Compile_T.elf_dir_name + "_linx")

    def compile_linx(self, BenchMark_dir, test_lists, LLVM, pmc_able):

        if not os.path.exists(LLVM_LINX):
            sys.exit("Error! {} not exist.".format(LLVM_LINX))

        pmc_paras = self.readying(BenchMark_dir, LLVM_LINX, pmc_able)
        cmp_paras = (
            self.Compile_T.cmp_paras + " " + "-fenable-matrix -mlxbc" + " " + pmc_paras
        )
        # cpp -> elf
        subName_cFiles = self.Compile_T.get_file(test_lists, BenchMark_dir, ".cpp")
        self.Compile_T.cmp_elf(LLVM, subName_cFiles, cmp_paras, self.elf_dir)
        check_cmps = self.Compile_T.check_cmp(test_lists, self.elf_dir, ".bin")
        # cpp -> i
        self.cmp_i(LLVM, subName_cFiles, cmp_paras, pmc_paras)
        # elf -> asm
        subName_elfs = self.Compile_T.get_file(test_lists, self.elf_dir, ".bin")
        self.objdump_elf(LLVM, subName_elfs)
        outCsv_name = "compile_linx.csv"
        self.Compile_T.out_result(test_lists, check_cmps, outCsv_name)

        return self.elf_dir

    def readying(self, BenchMark_dir, LLVM_LINX, pmc_able):

        self.cp_hpp_to_llvmlib(LLVM_LINX, BenchMark_dir)

        # mkdir elf dir
        if os.path.exists(self.elf_dir):
            shutil.rmtree(self.elf_dir)
            os.mkdir(self.elf_dir)
        else:
            os.mkdir(self.elf_dir)

        pmc_paras = ""
        if pmc_able != "N":
            pmc_paras = "-DLINX_PMC"

        return pmc_paras

    def cp_hpp_to_llvmlib(self, LLVM_LINX, BenchMark_dir):

        include_dir = os.path.join(BenchMark_dir, "include")
        llvm_bin_dir = LLVM_LINX.replace(LLVM_LINX.split("/")[-1], "")
        llvm_lib = os.path.join(llvm_bin_dir, "../lib/clang/15.0.4/include/tileop-api")
        shell = "rm -r {}/*; cp -r {}/* {}".format(llvm_lib, include_dir, llvm_lib)
        print(shell)
        os.system(shell)

    def cmp_i(self, LLVM_LINX, subName_cFiles, cmp_paras, pmc_paras):

        tasks = []
        with ThreadPoolExecutor(max_workers=g_max_threads) as executor:
            for sub_name, file in subName_cFiles.items():
                elf_file = os.path.join(self.elf_dir, sub_name + ".i")
                shell = "{} {} -mlxbc -E {} {} -o {}".format(
                    LLVM_LINX, cmp_paras, pmc_paras, file, elf_file
                )
                task = executor.submit(self.Compile_T.run, shell)
                tasks.append(task)

        for it in tasks:
            it.result()

    def objdump_elf(self, LLVM_LINX, subName_elfs):

        llvm_bin_dir = LLVM_LINX.replace(LLVM_LINX.split("/")[-1], "")
        objdump_path = os.path.join(llvm_bin_dir, "llvm-objdump")
        tasks = []
        with ThreadPoolExecutor(max_workers=g_max_threads) as executor:
            for sub_name, file in subName_elfs.items():
                asm_file = os.path.join(self.elf_dir, sub_name + ".asm")
                shell = "{} -d {} > {}".format(objdump_path, file, asm_file)
                task = executor.submit(self.Compile_T.run, shell)
                tasks.append(task)

        for it in tasks:
            it.result()


class Compile_Host:

    def __init__(self):

        self.Compile_T = Compile()
        self.elf_dir = os.path.join(os.getcwd(), self.Compile_T.elf_dir_name + "_host")

    def compile_host(self, BenchMark_dir, test_lists, LLVM):

        include_dir = self.readying(BenchMark_dir)
        cmp_paras = (
            self.Compile_T.cmp_paras + " " + "-D__cpu_sim__" + " " + "-I" + include_dir
        )
        # cpp -> elf
        subName_cFiles = self.Compile_T.get_file(test_lists, BenchMark_dir, ".cpp")
        self.Compile_T.cmp_elf(LLVM, subName_cFiles, cmp_paras, self.elf_dir)
        check_cmps = self.Compile_T.check_cmp(test_lists, self.elf_dir, ".bin")

        outCsv_name = "compile_host.csv"
        self.Compile_T.out_result(test_lists, check_cmps, outCsv_name)

        return self.elf_dir

    def readying(self, BenchMark_dir):

        # mkdir elf dir
        if os.path.exists(self.elf_dir):
            shutil.rmtree(self.elf_dir)
            os.mkdir(self.elf_dir)
        else:
            os.mkdir(self.elf_dir)

        include_dir = os.path.join(BenchMark_dir, "include")

        return include_dir


class Exec:

    def __init__(self):

        self.run_dir_name = "log_stdout"
        self.error_dir_name = "log_error"

    def exec(self, elf_dir_linx, elf_dir_host, qemu, gfrun):

        if elf_dir_linx != "None":
            Exec_Linx_T = Exec_Linx()
            Exec_Linx_T.exec_linx(elf_dir_linx, qemu, gfrun)
        if elf_dir_host != "None":
            Exec_Host_T = Exec_Host()
            Exec_Host_T.exec_host(elf_dir_host)

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

    def run(self, shell):
        os.system(shell)


class Exec_Linx:

    def __init__(self):

        self.qemu_timeout = "20m"
        self.gfrun_timeout = "1h"
        self.Exec_T = Exec()
        self.qemu_run_log_dir = os.path.join(
            os.getcwd(), self.Exec_T.run_dir_name + "_qemu"
        )
        self.qemu_error_log_dir = os.path.join(
            os.getcwd(), self.Exec_T.error_dir_name + "_qemu"
        )
        self.gfrun_run_log_dir = os.path.join(
            os.getcwd(), self.Exec_T.run_dir_name + "_gfrun"
        )
        self.gfrun_error_log_dir = os.path.join(
            os.getcwd(), self.Exec_T.error_dir_name + "_gfrun"
        )

    def exec_linx(self, elf_dir, qemu, gfrun):

        if qemu == "None" and gfrun == "None":
            sys.exit("Error! Both -qemu and -gfrun cannot be empty!")

        run_paras = ""
        model = ""
        if qemu != "None":
            model = qemu
            time_out = self.qemu_timeout
            run_paras = "-blk_optimize force_tb_chained"
            self.Exec_T.mkdir(self.qemu_run_log_dir)
            self.Exec_T.mkdir(self.qemu_error_log_dir)

            self.run_model(
                time_out,
                model,
                run_paras,
                elf_dir,
                self.qemu_run_log_dir,
                self.qemu_error_log_dir,
            )

        if gfrun != "None":
            model = gfrun
            time_out = self.gfrun_timeout
            run_paras = "-f"
            self.Exec_T.mkdir(self.gfrun_run_log_dir)
            self.Exec_T.mkdir(self.gfrun_error_log_dir)
            self.run_model(
                time_out,
                model,
                run_paras,
                elf_dir,
                self.gfrun_run_log_dir,
                self.gfrun_error_log_dir,
            )

    def run_model(
        self, time_out, model, run_paras, elf_dir, run_log_dir, error_log_dir
    ):

        elf_paths = self.Exec_T.get_elf(elf_dir)

        tasks = []
        with ThreadPoolExecutor(max_workers=g_max_threads) as executor:
            for elf_path in elf_paths:
                file_name = elf_path.split("/")[-1].split(".bin")[0] + ".log"
                run_log_path = os.path.join(run_log_dir, file_name)
                file_name = elf_path.split("/")[-1].split(".bin")[0] + ".err"
                error_log_path = os.path.join(error_log_dir, file_name)

                shell = "timeout {time} {m} {rp} {elf} 1> {rl} 2> {err}".format(
                    time=time_out,
                    m=model,
                    rp=run_paras,
                    elf=elf_path,
                    rl=run_log_path,
                    err=error_log_path,
                )

                print(shell)
                task = executor.submit(self.Exec_T.run, shell)
                tasks.append(task)

            for it in tasks:
                it.result()


class Exec_Host:

    def __init__(self):

        self.Exec_T = Exec()
        self.run_log_dir = os.path.join(os.getcwd(), self.Exec_T.run_dir_name + "_host")
        self.error_log_dir = os.path.join(
            os.getcwd(), self.Exec_T.error_dir_name + "_host"
        )

    def exec_host(self, elf_dir):

        self.Exec_T.mkdir(self.run_log_dir)
        self.Exec_T.mkdir(self.error_log_dir)
        elf_paths = self.Exec_T.get_elf(elf_dir)

        tasks = []
        with ThreadPoolExecutor(max_workers=g_max_threads) as executor:
            for elf_path in elf_paths:
                file_name = elf_path.split("/")[-1].split(".bin")[0] + ".log"
                run_log_path = os.path.join(self.run_log_dir, file_name)
                file_name = elf_path.split("/")[-1].split(".bin")[0] + ".err"
                error_log_path = os.path.join(self.error_log_dir, file_name)
                shell = "{} 1> {} 2> {}".format(elf_path, run_log_path, error_log_path)
                print(shell)
                task = executor.submit(self.Exec_T.run, shell)
                tasks.append(task)

            for it in tasks:
                it.result()


class Check:

    def check(self, qemu, gfrun, src_dir, elf_dir_linx, elf_dir_host, single_case):

        Process_T = Process()
        test_lists = Process_T.process(src_dir, single_case)

        Check_Host_T = Check_Host()
        Check_Host_T.check_host(test_lists, elf_dir_host)
        if qemu != "None":
            Check_QEMU_T = Check_QEMU()
            Check_QEMU_T.check_qemu(test_lists, elf_dir_linx)
        if gfrun != "None":
            Check_Gfrun_T = Check_Gfrun()
            Check_Gfrun_T.check_gfrun(test_lists, elf_dir_linx)
        # check and assert
        failed = {}

        # 收集 host 结果
        host_csv = "host_test.csv"
        if os.path.exists(host_csv):
            with open(host_csv, encoding="utf-8") as f:
                for line in f.readlines()[1:]:
                    parts = line.strip().split(",")
                    if len(parts) >= 3:
                        case, comp, run = parts[:3]
                        if comp != "True" or run != "Pass":
                            failed.setdefault(case, []).append(f"host({comp}/{run})")

        # 收集 qemu 结果
        qemu_csv = "qemu_test.csv"
        if os.path.exists(qemu_csv):
            with open(qemu_csv, encoding="utf-8") as f:
                for line in f.readlines()[1:]:
                    parts = line.strip().split(",")
                    if len(parts) >= 4:
                        case, comp, run, chk = parts[:4]
                        if comp != "True" or run != "Pass" or chk != "True":
                            failed.setdefault(case, []).append(f"qemu({comp}/{run}/{chk})")

        if failed:
            import sys
            sys.exit("some tileop test case failed")

    def out_result(
        self, outCsv_name, test_lists, check_cmps, check_runs, check_checknums
    ):

        fd = open(outCsv_name, "w", encoding="utf8")
        fd.write("TileOP,Compile,Run,Function Verification\n")

        num_sum = 0
        cmp_err = 0
        run_err = 0
        check_err = 0

        for sub_name in test_lists:
            num_sum += 1
            # 如果不存在，则错误
            if not check_cmps.get(sub_name):
                check_cmps[sub_name] = "False"
            if not check_runs.get(sub_name):
                check_runs[sub_name] = "False"
            if not check_checknums.get(sub_name):
                check_checknums[sub_name] = "False"

            # 如果编译失败，则运行是Nan
            if check_cmps[sub_name] != "True":
                cmp_err += 1
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
                + check_cmps[sub_name]
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
        print("Compile PASS: ", num_sum - cmp_err)
        print("Run PASS:     ", num_sum - run_err)
        print("***************************************************************")
        print("Verify PASS:  ", num_sum - check_err)
        if check_err > 0:
            print("Verify Failed Case: {}".format(", ".join([n for n in test_lists if check_checknums.get(n) != "True"])))


    def check_checknum(self, subName_checknums_true, subName_checknums_linx):

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


class Check_Host:

    def check_host(self, test_lists, elf_dir):

        Exec_Host_T = Exec_Host()
        if not os.path.exists(Exec_Host_T.error_log_dir):
            return
        Compile_T = Compile()
        check_cmps = Compile_T.check_cmp(test_lists, elf_dir, ".bin")
        Check_QEMU_T = Check_QEMU()
        check_runs = Check_QEMU_T.check_qemu_help(Exec_Host_T.error_log_dir)

        outCsv_name = "host_test.csv"
        self.out_result(outCsv_name, test_lists, check_cmps, check_runs)

    def out_result(self, outCsv_name, test_lists, check_cmps, check_runs):

        fd = open(outCsv_name, "w", encoding="utf8")
        fd.write("TileOP,Compile,Run\n")

        num_sum = 0
        cmp_err = 0
        run_err = 0

        for sub_name in test_lists:
            num_sum += 1
            # 如果不存在，则错误
            if not check_cmps.get(sub_name):
                check_cmps[sub_name] = "False"
            if not check_runs.get(sub_name):
                check_runs[sub_name] = "False"

            # 如果编译失败，则运行是Nan
            if check_cmps[sub_name] != "True":
                cmp_err += 1
                check_runs[sub_name] = "/"
            if check_runs[sub_name] != "Pass":
                run_err += 1
            fd.write(
                sub_name
                + ","
                + check_cmps[sub_name]
                + ","
                + check_runs[sub_name]
                + "\n"
            )

        fd.close()
        print("***************************************************************")
        print("For details, see:  {}".format(outCsv_name))
        print("Total:        ", num_sum)
        print("Compile PASS: ", num_sum - cmp_err)
        print("Run PASS:     ", num_sum - run_err)
        print("***************************************************************")


class Check_QEMU:

    def __init__(self):

        self.Check = Check()
        self.Exec_Linx = Exec_Linx()
        self.Exec_Host = Exec_Host()

    def check_qemu(self, test_lists, elf_dir):

        Compile_T = Compile()
        check_cmps = Compile_T.check_cmp(test_lists, elf_dir, ".bin")

        check_runs = self.check_qemu_help(self.Exec_Linx.qemu_error_log_dir)
        check_checknums = self.check_checknum()

        self.out_result(test_lists, check_cmps, check_runs, check_checknums)

    def check_qemu_help(self, log_dir):

        check_runs = {}

        for root, dirs, files in os.walk(log_dir):
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

    def check_checknum(self):

        check_checknums = {}

        # get true checknum
        subName_checknums_true = self.Check.get_all_checknum(self.Exec_Host.run_log_dir)

        subName_checknums = self.Check.get_all_checknum(self.Exec_Linx.qemu_run_log_dir)
        check_checknums = self.Check.check_checknum(
            subName_checknums_true, subName_checknums
        )

        return check_checknums

    def out_result(self, test_lists, check_cmps, check_runs, check_checknums):

        outCsv_name = "qemu_test.csv"
        self.Check.out_result(
            outCsv_name, test_lists, check_cmps, check_runs, check_checknums
        )


class Check_Gfrun:

    def __init__(self):

        self.Check = Check()
        self.Exec_Linx = Exec_Linx()
        self.Exec_Host = Exec_Host()

    def check_gfrun(self, test_lists, elf_dir):

        Compile_T = Compile()
        check_cmps = Compile_T.check_cmp(test_lists, elf_dir, ".bin")

        # gfrun 既需要检查err_log, 还需要检查run_log，run_log的结果可以覆盖err_log
        check_runs = self.check_gfrun_help(self.Exec_Linx.gfrun_error_log_dir)
        check_runs = self.check_gfrun_help(self.Exec_Linx.gfrun_run_log_dir)
        check_checknums = self.check_checknum()

        self.out_result(test_lists, check_cmps, check_runs, check_checknums)

    def check_gfrun_help(self, log_dir):

        check_runs = {}

        for root, dirs, files in os.walk(log_dir):
            for file_name in files:
                sub_name = file_name.split(".")[0]
                file_path = os.path.join(root, file_name)
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
                            elif (
                                "assert" in line or "Assert" in line or "ASSERT" in line
                            ):
                                check_runs[sub_name] = "assert"
                            elif (
                                "Success to Reach the End of Benchmark!" in line
                            ):  # gfrun运行成功的标志
                                check_runs[sub_name] = "Pass"
                            else:
                                check_runs[sub_name] = "timeout or unknown error"
                except UnicodeDecodeError:
                    check_runs[sub_name] = "False"

        return check_runs

    def check_checknum(self):

        check_checknums = {}

        # get true checknum
        subName_checknums_true = self.Check.get_all_checknum(self.Exec_Host.run_log_dir)

        checknum_files = self.Check.get_files(self.Exec_Linx.gfrun_run_log_dir)
        # 需要去掉gfrun本身的输出
        self.del_line_files(checknum_files)
        subName_checknums = self.Check.get_all_checknum(
            self.Exec_Linx.gfrun_run_log_dir
        )
        check_checknums = self.Check.check_checknum(
            subName_checknums_true, subName_checknums
        )

        return check_checknums

    def del_line_files(self, file_paths):

        for file_path in file_paths.values():
            self.del_line_file(file_path)

    def del_line_file(self, file_path):

        file_line = ""
        try:
            with open(file_path, "r", encoding="utf8") as f:
                for line in f.readlines():
                    if "Starting from" in line or "=" in line:
                        continue
                    if ":" in line and "Result" not in line:
                        continue
                    file_line += line
        except UnicodeDecodeError:
            pass

        with open(file_path, "w", encoding="utf8") as f:
            f.write(file_line)

        f.close()

    def out_result(self, test_lists, check_cmps, check_runs, check_checknums):

        outCsv_name = "gfrun_test.csv"
        self.Check.out_result(
            outCsv_name, test_lists, check_cmps, check_runs, check_checknums
        )


# 输入源码和编译工具链测试
def run_cmp_run(
    BenchMark_dir, src_dir, test_model, pmc_able, LLVM_LINX, LLVM_HOST, qemu, gfrun, single_case
):

    if test_model != "cmp" and test_model != "run":
        sys.exit("Error! Unrecognized operation for -m.")

    compile = Compile()
    elf_dir_linx, elf_dir_host = compile.compile(
        BenchMark_dir, src_dir, LLVM_LINX, pmc_able, LLVM_HOST, single_case
    )

    if test_model == "run":
        Exec_T = Exec()
        Exec_T.exec(elf_dir_linx, elf_dir_host, qemu, gfrun)

        Check_T = Check()
        Check_T.check(qemu, gfrun, src_dir, elf_dir_linx, elf_dir_host, single_case)


def main(
    BenchMark_dir, src_dir, test_model, pmc_able, LLVM_LINX, LLVM_HOST, qemu, gfrun, single_case
):

    run_cmp_run(
        BenchMark_dir, src_dir, test_model, pmc_able, LLVM_LINX, LLVM_HOST, qemu, gfrun, single_case
    )


if __name__ == "__main__":
    argParser = argparse.ArgumentParser(description="tileop test")
    argParser.add_argument("-lib", type=str, help="tile lib dir, case: /xx/Linx-TileOP-API/")
    argParser.add_argument(
        "-src",
        type=str,
        default="None",
        help="input test dir, case: /xx/Linx-TileOP-API/test/tileop_api/src",
    )
    argParser.add_argument(
        "-m", type=str, default="cmp", help="test model: cmp or run, default cmp"
    )
    argParser.add_argument(
        "-pmc", default="N", type=str, help="if open ckpt PMC, N(default) or Y"
    )
    argParser.add_argument(
        "-lc",
        type=str,
        default="None",
        help="linx clang++ dir, case: /xx/linx_blockisa_llvm/bin/clang++",
    )
    argParser.add_argument(
        "-hc",
        type=str,
        default="None",
        help="linx clang++ dir, case: /xx/llvm-15.0.4/bin/clang++",
    )
    argParser.add_argument("-qemu", default="None", type=str, help="qemu-linx path")
    argParser.add_argument("-gfrun", default="None", type=str, help="gfrun path")
    argParser.add_argument("-case", default="None", type=str, help="single test case")

    args = argParser.parse_args()

    BenchMark_dir = args.lib
    src_dir = args.src
    # 默认编译库下所有.cpp文件
    if src_dir == "None":
        src_dir = BenchMark_dir
    test_model = args.m
    pmc_able = args.pmc
    LLVM_LINX = args.lc
    LLVM_HOST = args.hc
    qemu = args.qemu
    gfrun = args.gfrun
    single_case = args.case

    main(
        BenchMark_dir, src_dir, test_model, pmc_able, LLVM_LINX, LLVM_HOST, qemu, gfrun, single_case
    )
