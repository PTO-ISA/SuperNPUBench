# Microbenchmarks

Instruction-level benchmarks organized by ISA family. The tile families
(cube/vector/memory) use PTO 0.57.1 intrinsic naming (`TLOAD/TSTORE/TMOV`,
`TMATMUL/TGEMV`, TEPL set) via `<common/pto_tileop.hpp>`; the scalar
family uses plain C + volatile to drive the GPR micro-ISA. The generator emits
281 cases; the active toolchain determines which structural cases compile.

## Directory Structure

```
microbenchmark/
├── Makefile.common          # shared build (PLAT=linx, linx_blockisa_llvm_musl)
├── compile_all.sh           # top-level: cube / vector / memory / scalar
├── gen_cases.py             # table-driven case generator
├── common/
│   └── bench_utils.hpp      # data init / verify helpers
├── cube/                    # named TMA/CUBE direct operations
│   ├── cube_bench.hpp       # bench_matmul / bench_gemv ...
│   ├── Makefile / compile.all / src/*.cpp
├── vector/                  # TEPL family (BSTART.TEPL)
│   ├── vector_bench.hpp     # bench_binary / unary / ternary / reduce / scalar ...
│   ├── Makefile / compile.all / src/*.cpp
├── memory/                  # TLSU family (BSTART.TLSU)
│   ├── memory_bench.hpp     # bench_load / mov / gather / scatter (+mask)
│   ├── Makefile / compile.all / src/*.cpp
├── scalar/                  # GPR scalar family (BSTART.STD / FP)
│   ├── scalar_bench.hpp     # bench_latency / bench_throughput / bench_store / bench_cv
│   ├── Makefile / compile.all / src/*.cpp
└── verification/            # NOT a benchmark family — correctness cases (see below)
    └── tlsu/                # TLSU end-to-end functional / cross-model diff cases
```

## Coverage

| family | covers | cases |
| --- | --- | ---: |
| matrix (TMA/CUBE direct) | TMATMUL / TMATMUL_BIAS / TMATMUL_MX | 6 |
| vector (TEPL) | elementwise / tile-scalar / reduce / expand | 126 |
| memory (TLSU) | TLOAD / TSTORE / TMOV / MGATHER / MSCATTER / *_MASK / layout | 25 |
| scalar (GPR) | int ALU / load-store / float / conversion × throughput+latency | 124 |
| **total** | | **281** |

- tile dtypes: `fp16 / fp32 / i8 / i16 / i32`; scalar dtypes: `i32 / i64 / f32 / f64`.
- tile sizes: vector/memory 16×16 (some 32×32); matrix 64³ (fp32 32³, with an 8 KiB benchmark working-set choice).
- scalar: 1024-iter loop; throughput = 8 independent accumulators, latency = chain dependency.

`verification/` is **not** counted above and is **not** part of the benchmark
suite: it holds correctness cases (result dumps + independent checkers +
cross-model diffing), not cycle measurements. `compile_all.sh` deliberately
skips it — build those cases by hand, e.g.
`cd verification/tlsu && make TESTCASE=s1_copy_i32_32x32`. See
[`verification/tlsu/README.md`](verification/tlsu/README.md).

## Build

```bash
export COMPILER_DIR=/path/to/linx_blockisa_llvm_musl/bin

# single case
cd scalar && make TESTCASE=add_i32_lat
cd cube   && make TESTCASE=tmatmul_fp16_64x64x64

# one family
cd scalar && bash compile.all

# all families
./compile_all.sh              # default = all
./compile_all.sh scalar
./compile_all.sh cube vector
```

Artifacts: `output/microbenchmark/<family>/elf/<family>/<case>.elf`.

## Run

ELF binaries run on the **SuperScalarModel** `gfsim`/`gfrun` (build them from
the SuperScalarModel repo, then point at the ELF path here):

```bash
bin/gfsim -f output/microbenchmark/scalar/elf/scalar/add_i32_lat.elf
bin/gfsim -f output/microbenchmark/cube/elf/cube/tmatmul_fp16_64x64x64.elf
```

`BENCHSTART/BENCHEND` (`B.HINT TRACE.begin/end`) bracket the measured region;
gfsim reports cycles. For scalar: latency = cycles/1024, throughput = cycles/(8×1024).

## Regenerate cases

Edit the case tables at the top of `gen_cases.py` (opcode / dtype / size), then:

```bash
python3 gen_cases.py   # rewrites all four src/ trees + compile.all
```

## Status & Adaptation Notes

The structural corpus is generated from the PTO 0.57.1 operation set. The
vector family includes adaptations for the currently exposed intrinsic surface:

- **Name shims** in `vector_bench.hpp`: `TSEL→TSELECT`, `TEXPANDS→TEXPANDSCALAR`,
  `TROW/COLEXPAND→TEXPANDROW/COL`, plus a 3-arg `TCMP` overload (defaults `CmpMode::EQ`).
- **`bench_reduce`** uses a 1-column output tile (`ValidCol==1`).
- **dtype fixes**: `TABS` float-only, `TREM` int-only, `TRSQRT` fp32-only.
- **`VECTOR_SKIP`**: opcodes needing unexposed names or special fractal/NZ
  layout are skipped (`TPRELU/TRELU/TNEG/TNOT/TLOG/TPART*/TCOL*/
  expand-arith/TCONCAT/TGATHERB/TCMP/TRSQRT/TROWMAX/TROWSUM/TCMPS/…`), with
  TODO comments — re-enable when `pto_tileop.hpp` fully aligns to PTO naming.
- **memory**: `TMOV` is not yet exposed → `TCOPY` fallback.
- **cube**: `tmatmul_acc` skipped (toolchain `matmul.ac` backend crash); `TGEMV*`
  not exposed (templates kept under `#if 0`).
- scalar disassembly confirms lowering to scalar micro-ISA (e.g. `addw` chain).
