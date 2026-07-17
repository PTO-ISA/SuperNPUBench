# SuperNPUBench

SuperNPUBench is the LinxISA/PTO operator and microbenchmark workload suite.
The checked-in Linx target surface follows the current v0.57 contract:

- target triple: `linx64-linx-none-elf`
- compiler: the superproject LLVM build in `compiler/llvm`
- C library/sysroot: the superproject musl build in `out/libc/musl`
- tile assembly: named v0.57 TMA, CUBE, and TEPL block headers
- tile operands: canonical `B.IOT` descriptors with ordinary `r` constraints

Retired target triples, numeric PTO selectors, vendor-only register
constraints, and compatibility block spellings are not supported.

## Layout

```text
SuperNPUBench/
├── benchmark/
│   ├── two-level-arch/   # LinxISA block templates and PTO kernels
│   └── one-level-arch/   # PTO-oriented companion kernels
├── microbenchmark/       # scalar, memory, vector, and cube cases
├── architecture/         # workload-oriented programming notes
└── compile_all.sh
```

Build outputs under `output/` are ignored.

## Toolchain setup

SuperNPUBench is a submodule of the LinxISA superproject. Build LLVM and musl
from the superproject root, then export their locations:

```bash
cmake --build compiler/llvm/build-linxisa-clang \
  --target clang clang++ llvm-objdump llvm-objcopy -j8

export COMPILER_DIR="$PWD/compiler/llvm/build-linxisa-clang/bin"
export LINX_SYSROOT="$PWD/out/libc/musl/install/phase-b"
```

The Makefiles add:

```text
-target linx64-linx-none-elf
--sysroot=$LINX_SYSROOT
-fenable-matrix
-O2
-std=c++20
```

`COMPILER_DIR` must contain `clang`, `clang++`, `llvm-objdump`, and
`llvm-objcopy`. The workload does not download or select an external compiler.

## Build examples

Compile one two-level operator:

```bash
cd benchmark/two-level-arch/test/kernel/matmul
make TESTCASE=matmul TYPE=MASK MODE=MASK_FP32 \
  M=256 N=256 K=256 tM=64 tN=64 tK=64
```

Compile the matching one-level operator:

```bash
cd benchmark/one-level-arch/test/kernel/matmul
make TESTCASE=matmul TYPE=HIF4_HIF4 \
  M=256 N=2048 K=2048 tM=128 tN=128 tK=128
```

Run batch builds:

```bash
./compile_all.sh two-level
./compile_all.sh one-level
./compile_all.sh all
```

Artifacts are written below `benchmark/<arch>/output/`.

## Representative v0.57 hardware-path check

The hashtable kernel exercises named TEPL blocks plus TLOAD/TSTORE:

```bash
"$COMPILER_DIR/clang++" -c \
  -target linx64-linx-none-elf \
  --sysroot="$LINX_SYSROOT" \
  -idirafter "$LINX_SYSROOT/usr/include" \
  -fenable-matrix -O2 -std=c++20 \
  -Ibenchmark/two-level-arch/include \
  -Ibenchmark/two-level-arch/test/common \
  -Ibenchmark/two-level-arch/test/common/src \
  -Ibenchmark/two-level-arch/kernels \
  -Ibenchmark/two-level-arch/models \
  -D__linx -DENABLE_TENSOR_INSTR \
  -DkNum=6144 -DMAX_PROBE=512 -DNUM_COL=256 \
  benchmark/two-level-arch/test/kernel/control/hashtable_lookup_simd/hashtable_lookup_simd.cpp \
  -o /tmp/supernpubench-v057.o

"$COMPILER_DIR/llvm-objdump" -d /tmp/supernpubench-v057.o
```

The disassembly must contain named blocks such as `BSTART.TLOAD`,
`BSTART.TSTORE`, `BSTART.TSEL`, and `BSTART.TCMP`; numeric TEPL selectors are
not an acceptable result.

## Microbenchmarks

The microbenchmark tree covers:

| Family | Representative operations |
| --- | --- |
| scalar | integer, floating-point, conversion, and memory latency/throughput |
| memory | TLOAD, TSTORE, TMOV, gather, and scatter |
| vector | named TEPL elementwise, scalar-tile, reduction, and expansion ops |
| cube | TMATMUL, bias/accumulate variants, and ACCCVT |

```bash
cd microbenchmark
make TESTCASE=tmatmul_fp16_64x64x64
bash compile_all.sh all
```

See [microbenchmark/README.md](microbenchmark/README.md) for case-generation
details.

## Models

ELF files can be passed to the superproject model or emulator appropriate for
the test lane. Do not hard-code developer-local simulator paths in scripts.

## Development rules

1. Keep the two architecture trees structurally aligned where they expose the
   same operator.
2. Use named v0.57 block headers; do not introduce raw numeric PTO selectors.
3. Use standard inline-assembly constraints. Tile values use `r`, and B.IOT
   destinations use the Linx `%q` operand modifier to print only the queue bank.
4. Keep host fallbacks behind `!defined(__linx)`; Linx builds must compile the
   architectural tile path.
5. Run `git diff --check`, the representative hardware-path compile above, and
   the relevant `compile.all` scripts before submitting changes.

## Documentation

- [LinxISA workload reference](architecture/linxisa-reference/)
- [PTO workload reference](architecture/ptoisa-reference/)
- [Canonical LinxISA v0.57 manual](../../docs/isa/)

## License

See `LICENSE`.
