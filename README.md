# SuperNPUBench

SuperNPUBench is a high-performance operator library and benchmark platform for
NPU tile-programming ISA. It ships **two architecture backends** under `benchmark/`
(two-level-arch = LinxISA, one-level-arch = PTO ISA) plus an instruction-level
**microbenchmark** suite, all driven by the same Linx toolchain.

## Repository Structure

```
SuperNPUBench/
├── benchmark/
│   ├── two-level-arch/      # Linx two-level block ISA (was benchmark-linxisa)
│   │   ├── kernels/         # header-only operator implementations
│   │   ├── test/            # test suites + build system
│   │   │   ├── common/      # shared Makefile.common, _start.s, benchmark.h
│   │   │   └── kernel/      # per-operator test cases
│   │   └── compile_all.sh
│   ├── one-level-arch/      # PTO one-level tile ISA (was benchmark-ptoisa)
│   │   ├── kernels/
│   │   ├── test/
│   │   └── compile_all.sh
│   └── (no docs/ — comparison docs were removed)
├── microbenchmark/          # instruction-level micro-bench (cube/vector/memory/scalar)
├── architecture/            # ISA reference docs
│   ├── linxisa-reference/   # LinxISA programming guide + ISA reference
│   └── ptoisa-reference/    # PTO ISA programming guide
└── compile_all.sh           # top-level: two-level | one-level | all
```

> Build outputs (`output/`, `**/output/`) and `.DS_Store` are gitignored.

## Architecture Backends

### two-level-arch (LinxISA)
- Block-structured ISA with heterogeneous cores: BCC (main), Cube (matrix), Vector, MTC/TMA (data transfer).
- Programming model: block instructions (VPAR/VSEQ, CUBE, TMA, TEPL).
- Reference: [`architecture/linxisa-reference/`](architecture/linxisa-reference/).

### one-level-arch (PTO ISA)
- Tile-centric ISA with explicit memory hierarchy: Vec (UB), Mat (L1), Left (L0A), Right (L0B), Acc (L0C).
- Programming model: tile operations, Auto/Manual modes.
- Reference: [`architecture/ptoisa-reference/`](architecture/ptoisa-reference/).

Both backends share the same operator set and test layout; their kernel
implementations differ in ISA style.

## Operator Overview

Each backend implements 10 operator categories:

| Operator | Description |
|----------|-------------|
| **matmul** | FP4/BF16/FP32/FP16/FP8 matrix multiply; quantization, mixed precision, A/B reuse |
| **fa** | Flash Attention; 2D unroll, HIF4 quantization, unaligned boundary |
| **transpose** | 3D~6D tensor transpose; multiple dtypes |
| **reduction** | Row/column max & sum; single-tree, unaligned, cumsum, reduceprod |
| **gelu** | GELU activation; exact (erf) and tanh approximation |
| **broadcast** | 2D~5D broadcast; vectorized variants |
| **gather** | Data gathering; large-scale, power-of-2 dims |
| **concat** | Concatenation; gather/scatter modes |
| **control** | `hashtable_lookup_simd` (pure tile-op, single-tier gfsim) |
| **sort** | `topk` |

## Setup Environment

SuperNPUBench compiles with the **Linx toolchain** (`linx_blockisa_llvm_musl`,
clang-15, target `linx64v5-unknown-linux-musl`). Build it once from the
[`linx-toolchain-build`](https://github.com/LinxISA/linx-toolchain-build) repo,
which clones the matching ISA sources and produces the `linx_blockisa_llvm_musl`
install tree that `COMPILER_DIR` points at.

### 1. Clone the build repo

```bash
git clone https://github.com/LinxISA/linx-toolchain-build.git
cd linx-toolchain-build
```

### 2. Install host build tools

```bash
sudo apt-get install -y git make cmake ninja-build gcc g++ python3 autoconf m4
```

### 3. Initialize component sources

`make init-src` clones the five component repos under `src/` on their pinned
branches (run it again any time to fetch updates):

| Directory | Repository | Branch |
| --- | --- | --- |
| `src/llvm-project` | `LinxISA/llvm-project` | `dev-llvm15_56` |
| `src/musl` | `LinxISA/linx-musl` | `linx` |
| `src/jemalloc` | `LinxISA/jemalloc` | `linx` |
| `src/linux-linxisa` | `LinxISA/linux` | `main` |
| `src/Linx-TileOP-API` | `LinxISA/Linx-TileOP-API` | `linx` |

```bash
make init-src
```

### 4. Build the toolchain

Only `linx64v5-linux-musl` is supported by the top-level Makefile:

```bash
make WITH_TARGET=linx64v5-linux-musl
```

This builds, in order: LLVM/clang/lld → kernel headers → musl → compiler-rt →
libc++/libc++abi/libunwind → jemalloc → Linx-TileOP-API headers. Progress is
tracked by stamp files under `stamps/`, so re-running `make` resumes from the
last completed step; `make clean` rebuilds from scratch. The install tree is
written to `output/linx_blockisa_llvm_musl/`:

```
output/linx_blockisa_llvm_musl/
├── bin/        # clang, clang++, ld.lld, llvm-ar/nm/ranlib,
│              # linx64v5-linux-musl-clang(++) symlinks
├── lib/        # clang runtime, libc++, ...
└── sysroot/    # musl + kernel headers + runtime libs
```

### 5. Point SuperNPUBench at the toolchain

```bash
export COMPILER_DIR=$(pwd)/output/linx_blockisa_llvm_musl/bin
$COMPILER_DIR/clang --version
# clang version 15.0.4 (linx64v5-musl-local ...)
# Target: linx64v5-unknown-linux-musl
```

Then proceed to [Quick Start](#quick-start).

### (Optional) Package

```bash
make package     # -> output/linx_blockisa_llvm_musl.tar.gz
```

### Platform notes (macOS)

The build also works on Apple Silicon, but the system `make` (GNU 3.81) and
`tar` (libarchive) differ from the Linux defaults, so two extra steps are
needed:

- **GNU make >= 4** is required by the kernel-headers step
  (`GNU Make >= 4.0 is required`). `brew install make` (provides `gmake`) and
  run the build with `gmake`, or put `gmake` on `PATH` ahead of `/usr/bin/make`
  (the kernel Makefile invokes `make` literally).
- **GNU tar** is required by `make package` (`tar --format=gnu`). `brew install
  gnu-tar` and prepend `$(brew --prefix)/opt/gnu-tar/libexec/gnubin` to `PATH`.
- **sancov host-compile fix**: Apple clang rejects an initializer-list in
  `llvm/tools/sancov/sancov.cpp` (`chosen constructor is explicit in
  copy-initialization`). If the LLVM build fails there, change the two
  `SpecialCaseList::createOrDie({{...}}, ...)` call sites to pass an explicit
  `std::vector<std::string>{...}`, then resume with
  `ninja -C build/build-llvm-musl`.

## Quick Start

### 1. Environment

Build the Linx toolchain once (see [Setup Environment](#setup-environment)), then
point `COMPILER_DIR` at it:

```bash
export COMPILER_DIR=/path/to/linx_blockisa_llvm_musl/bin
```

### 2. Compile an operator

```bash
# two-level-arch (LinxISA)
cd benchmark/two-level-arch/test/kernel/matmul
make TESTCASE=matmul TYPE=MASK MODE=MASK_FP32 M=256 N=256 K=256 tM=64 tN=64 tK=64

# one-level-arch (PTO ISA)
cd benchmark/one-level-arch/test/kernel/matmul
make TESTCASE=matmul TYPE=HIF4_HIF4 M=256 N=2048 K=2048 tM=128 tN=128 tK=128
```

### 3. Batch / full compilation

```bash
# per-operator batch
cd benchmark/two-level-arch/test/kernel/matmul && bash compile.all

# all operators of one backend
./compile_all.sh two-level      # two-level-arch only
./compile_all.sh one-level      # one-level-arch only
./compile_all.sh all            # both (default)
```

Artifacts land in `benchmark/<arch>/output/kernel/<operator>/elf/`.

## Microbenchmark

`microbenchmark/` is an instruction-level bench organized by ISA family,
generated by `gen_cases.py`. All 231 cases compile & link on the Linx toolchain.

| family | covers | cases |
| --- | --- | ---: |
| cube (CUBE) | TMATMUL / TMATMUL_BIAS / TMATMUL_MX / ACCCVT | 9 |
| vector (TEPL) | elementwise / tile-scalar / reduce / expand (toolchain-exposed subset) | 73 |
| memory (TLSU) | TLOAD / TSTORE / TMOV / MGATHER / MSCATTER (+mask, layout) | 25 |
| scalar (GPR) | int ALU / load-store / float / conversion × throughput+latency | 124 |
| **total** | | **231** |

```bash
cd microbenchmark && make TESTCASE=tmatmul_fp16_64x64x64   # one case
cd microbenchmark && bash compile_all.sh all               # all families
```

See [`microbenchmark/README.md`](microbenchmark/README.md) for details.

## Running on the Models

Compiled ELF binaries run on the **SuperScalarModel** simulator suite (no longer
on LinxCoreModel). Build `gfrun`/`gfsim` from the
[SuperScalarModel](../SuperScalarModel) repo, then point them at the ELF:

- `gfrun` — functional model (correctness)
- `gfsim` — cycle-accurate model (timing)

```bash
# from the SuperScalarModel repo root (where bin/ lives)
bin/gfrun -f /path/to/SuperNPUBench/benchmark/two-level-arch/output/kernel/<op>/elf/<name>.elf
bin/gfsim -f /path/to/SuperNPUBench/benchmark/two-level-arch/output/kernel/<op>/elf/<name>.elf
```

### Tile-op kernels: single-tier gfsim mode

Kernels written purely with tile ops using TEPL template instructions (e.g.
`control/hashtable_lookup_simd`) run on the VectorLite engine, which gfsim only
steps in **single-tier mode**:

```bash
bin/gfsim -f <elf> -s core.singleTierMode=true
```

Without this flag the engine is inert and the run deadlocks. `gfrun` does not
need the flag.

## Build System

### Makefile parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `TESTCASE` | Test case name | `matmul`, `fa_2d_unroll` |
| `TYPE` | Operator type (matmul) | `HIF4_HIF4`, `A16W4`, `MASK` |
| `MODE` | Operator mode | `MASK_FP32`, `BF16x2_NOGATHER` |
| `M/N/K` | Matrix dimensions | `M=256 N=2048 K=2048` |
| `tM/tN/tK` | Tile sizes | `tM=128 tN=128 tK=128` |
| `COMPILER_DIR` | Compiler path | `/path/to/linx/bin` |
| `PLAT` | Platform | `linx` (default), `cpu` |

### Build targets

```bash
make TESTCASE=<case> all      # compile
make TESTCASE=<case> diss     # disassembly
make TESTCASE=<case> sim      # run in simulator
make TESTCASE=<case> debug    # debug mode
make clean                    # clean current operator
make clean_all                # clean all
```

## Documentation

- **LinxISA**: [programming guide (zh)](architecture/linxisa-reference/programming_guide.md) | [programming guide (en)](architecture/linxisa-reference/programming_guide_en.md) | [ISA reference](architecture/linxisa-reference/isa_reference.md)
- **PTO ISA**: [programming guide (zh)](architecture/ptoisa-reference/programming_guide.md) | [programming guide (en)](architecture/ptoisa-reference/programming_guide_en.md)

## Known Issues

- **`fa_2d_unroll` compiler crash** (Issue #6): `X=1,Y=1` / `X=2,Y=1` trigger
  `Assertion failed: (Reg != 0 && "LinxV5 CallingConv Fail!")`. Avoid `Ydim=1`.
- **`control/hashtable_lookup_simd`**: pure tile-op kernel; run gfsim with
  `-s core.singleTierMode=true` (see above). Its `.data` files are generated by
  the Makefile (`gen_data.py`) on first build.
- **microbenchmark `tmatmul_acc`**: skipped — toolchain `matmul.ac` backend crash.
- **microbenchmark vector subset**: only toolchain-exposed TEPL opcodes are
  generated; `TGEMV*`/`TMOV` not yet exposed (TCOPY fallback). See
  [`microbenchmark/README.md`](microbenchmark/README.md).

## Toolchain

- Compiler: `linx_blockisa_llvm_musl` (clang-15, linx64v5-musl)
- Flags: `-mlxbc -fenable-matrix -O2 -mllvm -enable-all-vector-as-tilereg=true -std=c++20`
- Target: Linx64 V5

## Development Guide

### Adding an operator

1. Add header-only kernel under both `benchmark/<arch>/kernels/<operator>/`.
2. Create test dir under both `benchmark/<arch>/test/kernel/<operator>/` with
   `Makefile`, `compile.all`, `src/`.
3. Mirror the two backends.

### Conventions

- Header-only kernels; PTO tile-programming paradigm.
- Build artifacts not tracked (`.gitignore`).
- Keep both backends in parallel.

## Related Links

- [LinxISA](https://linxisa.github.io/linx-isa/) · [PTO ISA](https://pto-isa.github.io/docs/isa/tile/)

## License

See LICENSE.
