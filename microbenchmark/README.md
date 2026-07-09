# Microbenchmarks

Instruction-level benchmarks organized by ISA family. The tile families
(cube/vector/memory) use DavinciOO/PTO intrinsic naming (`TLOAD/TSTORE/TMOV`,
`TMATMUL/TGEMV/ACCCVT`, TEPL set) via `<common/pto_tileop.hpp>`; the scalar
family uses plain C + volatile to drive the GPR micro-ISA. All 231 cases
compile & link on the Linx toolchain (`linx_blockisa_llvm_musl`).

## Directory Structure

```
microbenchmark/
├── Makefile.common          # shared build (PLAT=linx, linx_blockisa_llvm_musl)
├── compile_all.sh           # top-level: cube / vector / memory / scalar
├── gen_cases.py             # table-driven case generator
├── common/
│   └── bench_utils.hpp      # data init / verify helpers
├── cube/                    # CUBE family (BSTART.CUBE)
│   ├── cube_bench.hpp       # bench_matmul / bench_gemv / bench_acccvt ...
│   ├── Makefile / compile.all / src/*.cpp
├── vector/                  # TEPL family (BSTART.TEPL)
│   ├── vector_bench.hpp     # bench_binary / unary / ternary / reduce / scalar ...
│   ├── Makefile / compile.all / src/*.cpp
├── memory/                  # TLSU family (BSTART.TLSU)
│   ├── memory_bench.hpp     # bench_load / mov / gather / scatter (+mask)
│   ├── Makefile / compile.all / src/*.cpp
└── scalar/                  # GPR scalar family (BSTART.STD / FP)
    ├── scalar_bench.hpp     # bench_latency / bench_throughput / bench_store / bench_cv
    ├── Makefile / compile.all / src/*.cpp
```

## Coverage

| family | covers | cases |
| --- | --- | ---: |
| cube (CUBE) | TMATMUL / TMATMUL_BIAS / TMATMUL_MX / ACCCVT | 9 |
| vector (TEPL) | elementwise / tile-scalar / reduce / expand (toolchain-exposed subset) | 73 |
| memory (TLSU) | TLOAD / TSTORE / TMOV / MGATHER / MSCATTER / *_MASK / layout | 25 |
| scalar (GPR) | int ALU / load-store / float / conversion × throughput+latency | 124 |
| **total** | | **231** |

- tile dtypes: `fp16 / fp32 / i8 / i16 / i32`; scalar dtypes: `i32 / i64 / f32 / f64`.
- tile sizes: vector/memory 16×16 (some 32×32); cube 64³ (fp32 32³, capped by the 8 KB tile-allocation limit).
- scalar: 1024-iter loop; throughput = 8 independent accumulators, latency = chain dependency.

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

All 231 cases compile & link on the Linx toolchain. The vector family was
adapted to the toolchain-exposed intrinsic surface:

- **Name shims** in `vector_bench.hpp`: `TSEL→TSELECT`, `TEXPANDS→TEXPANDSCALAR`,
  `TROW/COLEXPAND→TEXPANDROW/COL`, plus a 3-arg `TCMP` overload (defaults `CmpMode::EQ`).
- **`bench_reduce`** uses a 1-column output tile (`ValidCol==1`).
- **dtype fixes**: `TABS` float-only, `TREM` int-only, `TRSQRT` fp32-only.
- **`VECTOR_SKIP`**: opcodes needing unexposed names or special fractal/NZ
  layout are skipped (`TPRELU/TADDC/TRELU/TNEG/TNOT/TLOG/TPART*/TCOL*/
  expand-arith/TCONCAT/TGATHERB/TCMP/TRSQRT/TROWMAX/TROWSUM/TCMPS/…`), with
  TODO comments — re-enable when `pto_tileop.hpp` fully aligns to PTO naming.
- **memory**: `TMOV` is not yet exposed → `TCOPY` fallback.
- **cube**: `tmatmul_acc` skipped (toolchain `matmul.ac` backend crash); `TGEMV*`
  not exposed (templates kept under `#if 0`).
- scalar disassembly confirms lowering to scalar micro-ISA (e.g. `addw` chain).
