#!/bin/python

import os
import sys
import argparse
import subprocess
import shlex
import shutil

def gettarget(cmd):
  try:
    parts = shlex.split(cmd, posix=True)
    for part in parts:
      if part.startswith("target="):
        key, value = part.split("=", 1)
        return value
  except Exception as e:
    print(f"gettarget error: {e}")

def gensim(target, args):
  SMT = args.smt
  stash = args.stash
  tilemem = args.tilemem
  perfectmem = ""
  if args.perfectmem:
    perfectmem = "vectorcore.perfect_load_enable=true vectorcore.perfect_store_enable=true"
  rob = ""
  if args.largerob:
    rob = "pe.ope_rob_depth=512 core.preg_count=512 core.local_reg_t=256 core.local_reg_u=128 bctrl.tile_tag_num=128"
  elif args.smallrob:
    rob = "bctrl.vecIssueQDepth=8"
  isq = ""
  if args.largeisq:
    isq = "bctrl.bissq_depth=64 iex.iexCmdIqDepth=64"
  icache = ""
  if args.perfect_icache:
    icache = "ifu.icache_perfect=true"
  dbg = ""
  if args.debug:
    dbg = "-t 3"

  file = f"{target}"
  swimfile = f"swimlane.log"
  pipefile = f"pipe.log"

  confs = f"core.simtEnable=true ifu.ifu_thread_num=1 core.legalCheckEnable=false core.simtLane=64 core.deadlock_cycles=50000 core.soc_enable=true iex.simt_iex_alu_iq_count=1 vectorcore.ta_cache_set=256 bctrl.tile_t_hand_size=25 bctrl.tile_u_hand_size=25 bctrl.tile_m_hand_size=25 bctrl.tile_n_hand_size=25 core.threadCount={SMT} vectorcore.vector_core_thread_num={SMT} vectorcore.stash_mode={stash} tmu.tile_reg_size={tilemem} {rob} {isq} {icache} {dbg} {perfectmem}"
  extras = f"--swimlane 1 --swimfile {swimfile} -p 1 --pipefile {pipefile}"
  # extras = ""
  cmd = f"{args.gfsim} -s {confs} {extras} -f {file}"
  return cmd

def showpipe(pipe):
  for num, line in enumerate(pipe, 1):
    line = line[:-1]
    print(line)

def socfromenv():
  ld_libs = os.environ.get("LD_LIBRARY_PATH", "")
  paths = [p for p in ld_libs.split(os.pathsep) if p]
  for p in paths:
    if os.path.abspath(p).endswith("soc/lib"):
      return os.path.abspath(p + "/..")

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Run Compile_all Scripts")
  parser.add_argument("-l", "--program-list", type=str, default="program.list")
  parser.add_argument("-M", "--gfsim", type=str, default="gfsim")
  parser.add_argument("-p", "--prefix", type=str, default=os.getcwd())
  parser.add_argument("-c", "--clear", default=False, action="store_true")
  parser.add_argument("-t", "--smt", type=int, default=4)
  parser.add_argument("-s", "--stash", type=str, default="false")
  parser.add_argument("-m", "--tilemem", type=int, default=2560)
  parser.add_argument("-P", "--perfectmem", type=bool, default=False)
  parser.add_argument("-r", "--largerob", type=bool, default=False)
  parser.add_argument("-R", "--smallrob", type=bool, default=False)
  parser.add_argument("-i", "--largeisq", type=bool, default=False)
  parser.add_argument("--perfect-icache", type=bool, default=False)
  parser.add_argument("-g", "--debug", default=False, action="store_true")
  parser.add_argument("--soc", type=str, default=socfromenv())
  args = parser.parse_args()

  toolchain = os.environ.get("COMPILER_DIR", "")
  if not toolchain:
    print("can not find toolchain, please setup COMPILER_DIR")
    exit(1)
  toolchain = os.path.abspath(toolchain + "/..")
  objdump = toolchain + "/bin/llvm-objdump"

  if not args.soc:
    print("can not find soc path, please setup LD_LIBRARY_PATH or set --soc")
    exit(1)

  if not os.path.exists(args.prefix):
    os.mkdir(args.prefix)
  elif args.clear:
    shutil.rmtree(args.prefix)
    os.mkdir(args.prefix)
  else:
    print(f"{args.prefix} already exists! sim in new prefix or clear with --clear/-c")
    exit(1)

  cmds = []
  with open(args.program_list, "r") as f:
    for num, line in enumerate(f, 1):
      line = line.strip()
      if not line or line.startswith("#"):
        continue
      cmds.append(line)

  procs = {}
  for cmd in cmds:
    target = gettarget(cmd)
    prefix = f"{args.prefix}/{target}"
    cmd = f"{cmd} prefix={prefix}"
    os.mkdir(prefix)
    print(f"{cmd}")
    env = os.environ.copy()
    ofile = f"{prefix}/compile.out"
    efile = f"{prefix}/compile.err"
    process = subprocess.Popen(cmd, shell=True, env=env, universal_newlines=True, stdout=open(ofile, "w"), stderr=open(efile, "w"), text=True)
    process.wait()
    if process.returncode != 0:
      showpipe(open(efile, "r"))
      exit(process.returncode)
    showpipe(open(ofile, "r"))

    disasm = f"{objdump} -d {prefix}/{target}"
    process = subprocess.Popen(disasm, shell=True, env=env, universal_newlines=True, stdout=open(f"{prefix}/{target}.asm", "w"), stderr=subprocess.STDOUT, text=True)
    process.wait()
    if process.returncode != 0:
      print("objdump failed")
      exit(process.returncode)

    os.symlink(f"{args.soc}/parameter", f"{prefix}/parameter")
    os.mkdir(f"{prefix}/log")
    sim = gensim(target, args)
    print(f"{sim}")
    ofile = f"{prefix}/sim.out"
    efile = f"{prefix}/sim.err"
    process = subprocess.Popen(sim, shell=True, cwd=prefix, env=env, universal_newlines=True, stdout=open(ofile, "w"), stderr=open(efile, "w"), text=True)
    procs[process] = prefix

  print("")
  print("------------------")
  print("")

  errors = 0
  for process, prefix in procs.items():
    process.wait()
    print(prefix)
    if process.returncode != 0:
      showpipe(open(f"{prefix}/sim.err", "r"))
      print(f"details see {prefix}/sim.out")
      errors += 1

    for line in open(f"{prefix}/sim.out", "r"):
      line = line.strip()
      # print(line)
      if line.startswith("Total Cycles"):
        print(line)

  if errors != 0:
    exit(1)

