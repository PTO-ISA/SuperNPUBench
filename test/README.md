Method 1: in Makefile.common to specify COMPILER DIRECTORY(COMPILER_DIR) in your own environment

Method 2: invoke the make command with COMPILER_DIR in parameter e.g.:

`make TESTCASE=TAdd PLAT=linx COMPILER_DIR=/path-to-compiler-bin-dir`

## x86 plat
`make clean;make TESTCASE=TAdd PLAT=cpu`
`make clean;make TESTCASE=attention PLAT=cpu`

## arm plat
`make clean;make TESTCASE=TAdd PLAT=arm_sme`
`make clean;make TESTCASE=attention PLAT=arm_sme`
## linxISA plat
`make clean;make TESTCASE=TAdd PLAT=linx`
`make clean;make TESTCASE=attention PLAT=linx`

## build all cases

`make PLAT=linx build_all`

## output
- the elf will be in output dir/case_name

# compile py lib
- make clean;make TESTCASE=tileop_py PLAT=cpu PY_LIB=on

It will generate tileop_py.so in output/tileop_py

using golden cmp.py to compare python result with tileop_c result

- python3 golden_cmp/golden_cmp.py -i tadd(for example)

- for how to add new case, please ref to golden_cmp/README.md

# run case

## set qemu simulator binary
Method 1: change the `QEMU` variable in Makefile.common to the qemu-linx binary location

Method 2: pass the qemu-linx binary location through make command parameter e.g.:
`make TESTCASE=TAdd QEMU=path-to-qemu-linx-bin-exe`

## run a single case

`make TESTCASE=TAdd sim`

## run all cases
`make sim_all`

