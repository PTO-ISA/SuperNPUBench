# Microbenchmarks

Instruction-level benchmarks organized by ISA family. The tile families use
the PTO v0.58 TileOP surface (`TLOAD/TSTORE`, `TLOAD_CUBE/TSTORE_CUBE`,
`TMATMUL*`, and the TEPL set) via `<common/pto_tileop.hpp>`; the scalar
family uses plain C + volatile to drive the GPR micro-ISA; the `fixp`
family exercises the `B.FPATR` quantization/PostProcess options
(scalar/vector quant, ReLU/PReLU, row/group-max, shared-right) on
`TMATMUL`/`TGEMV`. The four generated families currently emit 278 unique active
cases; the hand-maintained `fixp` family adds 94 active modes (372 configurations).
`coverage.json` records active and unsupported cases.

## Directory Structure

```
microbenchmark/
├── Makefile.common          # shared build (PLAT=linx, linx_blockisa_llvm_musl)
├── compile_all.sh           # top-level: cube / vector / memory / scalar / fixp
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
└── fixp/                    # hand-maintained FPATR/quant TMATMUL (one src, 94 active modes)
    ├── src/fixp_tmatmul.cpp
    ├── Makefile / compile.all / report_fixp.py / fixp_report.md
```

## Coverage

| family | covers | cases |
| --- | --- | ---: |
| matrix (CUBE direct) | TMATMUL / TMATMUL_ACC / TMATMUL_BIAS | 11 |
| vector (TEPL) | elementwise / tile-scalar / expand / TCI sequence | 129 |
| memory (TLSU) | TLOAD / TSTORE / MGATHER / MSCATTER | 14 |
| scalar (GPR) | int ALU / load-store / float / conversion × throughput+latency | 124 |
| fixp (FPATR/quant) | TMATMUL/TGEMV × `fixp::Options` (scalar/vector quant, relu/prelu, row/group-max, shared-right) | 94 |
| **total** | | **372** |

- tile dtypes: `bf16 / fp16 / fp32 / i8 / i16 / i32`; scalar dtypes: `i32 / i64 / fp32 / f64`.
- tile sizes: vector/memory 16×16 (some 32×32); CUBE uses M16/M32 and N8 CELL layouts.
- scalar: 1024-iter loop; throughput = 8 independent accumulators, latency = chain dependency.

## Build

```bash
export COMPILER_DIR=/path/to/linx_blockisa_llvm_musl/bin

# single case
cd scalar && make TESTCASE=add_i32_lat
cd cube   && make TESTCASE=tmatmul_fp16_32x64x64
cd fixp   && make FIXP_MODE=S_QF_S8 diss      # fixp uses FIXP_MODE, not TESTCASE

# one family
cd scalar && bash compile.all
cd fixp   && bash compile.all

# all families
./compile_all.sh              # default = all
./compile_all.sh scalar
./compile_all.sh cube
```

Artifacts: `output/microbenchmark/<family>/elf/<family>/<case>.elf`.

## Numerical validation

Set `res_check=on` for an individual Make build, or use `run_all.py
--res-check` for the complete corpus. When `res_check=on`, the Makefile adds
`-DRES_CHECK` and redirects objects to `output/res_check/microbenchmark/`, so a
normal performance build cannot reuse an object compiled with validation code.

### Execution steps

```bash
export COMPILER_DIR=/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin

# One case: return value 0 means the in-kernel reference matched.
make -C vector TESTCASE=tadd_fp32_16x16 res_check=on diss

# Build, disassemble, run, and classify all active cases.
python3 run_all.py --res-check
python3 run_all.py --res-check --category memory
```

`run_all.py --res-check` performs three phases:

1. **Build** — invokes `compile_all.sh` with `res_check=on`, which runs each
   family's `compile.all`. Each case is compiled (`-DRES_CHECK`), linked with
   the toolchain sysroot, disassembled (`llvm-objdump -dl`), and placed under
   `output/res_check/microbenchmark/<family>/elf/`.
2. **Run** — for every `.elf` found, runs `gfrun -t 1 -f <elf>` (fixp
   Shared/transpose modes add `-s softcore.multiThreadNum=4`). A gfrun log is
   written per case under `output/microbenchmark/report/logs/`.
3. **Classify** — `PASS` = gfrun rc 0 + `Reach the End of Benchmark` + `R2 = 0`;
   `NUMERIC_FAIL` = ended but `R2 ≠ 0`; `RUN_FAIL` = everything else (gfrun
   crash, timeout, non-zero rc). Results are written to
   `output/microbenchmark/report/result.json` and `result.md`.

### Validation method

The CUBE, vector, memory, and scalar families compare tile/element output
against an untimed scalar reference computed in the same kernel (`verify()` in
`common/bench_utils.hpp`). Fixp uses a zero-input invariant: inputs and
auxiliary operands are zero, the destination begins with a nonzero sentinel,
and every output byte must become zero.

### Regression results

Last full run: 2026-09-01. Toolchain: clang 15.0.4 (`0f878a871`,
linx64v5-musl-local), gfrun `762a72c3` (Tag_0817-459). Repo HEAD: `58d436c`.

| family | total | PASS | NUMERIC_FAIL | RUN_FAIL | COMPILE_FAIL |
| --- | ---: | ---: | ---: | ---: | ---: |
| cube | 11 | 9 | 2 | 0 | 0 |
| vector | 129 | 93 | 27 | 7 | 2 |
| memory | 14 | 14 | 0 | 0 | 0 |
| scalar | 124 | 91 | 19 | 14 | 0 |
| fixp | 94 | 92 | 2 | 0 | 0 |
| **total** | **372** | **299** | **50** | **21** | **2** |

**Pass rate: 80.4 % (299/372).** Memory (14/14) and fixp (92/94) are
near-clean; failures concentrate in vector and scalar.

### Issue analysis

#### Sysroot link fix (prerequisite)

The sysroot libraries (`libc.a`, `libc++.a`) were originally built with ELF
Machine `0x105` (261); the compiler later switched `EM_LinxV5` to `0xE9` (233)
in commit `0f878a871`. This mismatch caused **all** links to fail with
`is incompatible with elf64llinxv5`. Rebuilding musl and libc++ from the
toolchain `Makefile` (`make build-musl build-libcxx build-libcxxabi build-libunwind`)
after cleaning stale build objects resolved the issue: all 372 cases now link
with the default toolchain sysroot — no `-nostdlib` bypass or custom `memops.o`
is needed. Math functions (`exp`, `sqrt`, `log`, `fmodf` …) resolve from
`libc.a` (musl does not populate `libm.a` separately).

#### COMPILE_FAIL — 2 cases

`trem_fp16_16x16`, `trems_fp16_16x16` — the compiler (clang-15) crashes during
instruction selection when `-DRES_CHECK` activates the in-kernel `fmodf`
reference path for FP16. This is a **compiler bug**, not a source error; the
FP32 variants (`trem_fp32`, `trems_fp32`) compile and pass. The crash produces
a `DiagnosticReports/clang-15_*.crash` file.

#### RUN_FAIL — 21 cases (gfrun model gap)

All 21 share one crash signature:

```
gfrun: illegal instruction: ASSERTION FAILED: m_handlers.find(grp) != m_handlers.cend()
missing MInst handler … file MInstCalculator.cpp:85
```

The gfrun `SoftCore` model lacks a handler for the instruction group emitted
by these operations:

| pattern | cases | operations |
| --- | ---: | --- |
| scalar shift | 12 | `sll/sra/srl` × `{i32, i64}` × `{lat, thr}` |
| vector shift | 4 | `tshl/tshr` × `{i16, i32}` |
| scalar sqrt | 2 | `sqrt_f64_{lat,thr}` |
| vector sqrt/rsqrt | 3 | `tsqrt_fp16`, `trsqrt_{fp16,fp32}` |

This is a **gfrun model limitation** (no `MInst` handler for shift/sqrt
groups), not an operator or compiler issue.

#### NUMERIC_FAIL — 50 cases (R2 ≠ 0)

All 50 reach `End of Benchmark` with `R2 = 1` — the computed output differs
from the in-kernel reference. Grouped by suspected root cause:

| group | count | cases | likely cause |
| --- | ---: | --- | --- |
| vec min/neg/not/concat | 15 | `tmin`, `tneg`, `tnot`, `tpartmin`, `tmins`, `tconcat` (fp16/fp32/i16/i32) | verify reference or epsilon bug — trivial ops should not mismatch |
| vec expand-min/expdif | 8 | `trowexpandmin`, `tcolexpandmin`, `trowexpandexpdif`, `tcolexpandexpdif` | expand + min/expdif reference path |
| vec math (exp/log) | 4 | `texp`, `tlog` (fp16/fp32) | musl `exp`/`log` precision vs reference |
| scalar ld/st | 7 | `ld/st` × `{f64, fp32, i32, i64}` | verify reference bug — load/store has no computation |
| scalar abs/neg | 5 | `abs`, `neg` × `{f64, fp32, i64}` | verify reference bug — trivial unary ops |
| scalar div/mod | 4 | `div`, `mod` × `{f64, fp32, i32, i64}` | precision or signed-remainder semantics |
| scalar other | 3 | `max_i32_thr`, `not_i64_lat`, `popc_i32_thr` | mixed |
| cube bf16 | 2 | `tmatmul_bf16`, `tmatmul_bias_bf16` | bf16 matmul precision |
| fixp s4 | 2 | `s_qf_s4`, `v_qf_s4` | 4-bit quantization precision |

The `ld/st` and `abs/neg` groups (12 cases) are the strongest candidates for
**verify-reference bugs** rather than real operator defects — a load/store or
unary-negation kernel should produce bit-identical output, so a mismatch most
likely indicates the in-kernel `verify()` golden path or epsilon threshold is
wrong. Investigating these first would likely clear 12 of the 50 failures.

## Run

ELF binaries run on the **SuperScalarModel** `gfsim`/`gfrun` (build them from
the SuperScalarModel repo, then point at the ELF path here):

```bash
bin/gfsim -f output/microbenchmark/scalar/elf/scalar/add_i32_lat.elf
bin/gfsim -f output/microbenchmark/cube/elf/cube/tmatmul_fp16_32x64x64.elf
bin/gfsim -f output/microbenchmark/fixp/elf/fixp/fixp_tmatmul_s_qf_s8_M32_N32_K32_tM32_tN32_tK32.elf
```

`BENCHSTART/BENCHEND` (`B.HINT TRACE.begin/end`) bracket the measured region;
gfsim reports cycles. For scalar: latency = cycles/1024, throughput = cycles/(8×1024).

## Regenerate cases

Edit the case tables at the top of `gen_cases.py` (opcode / dtype / size), then:

```bash
python3 gen_cases.py   # rewrites the four generated src/ trees + compile.all
```

`fixp/` is hand-maintained (not regenerated by `gen_cases.py`): its modes
live in `fixp/src/fixp_tmatmul.cpp`'s `#if` ladder, `fixp/compile.all`'s
`MODES` array, and `fixp/report_fixp.py`'s `MODES` table — keep them in sync.

## Status & Adaptation Notes

The generated corpus follows the PTO v0.58 operation surface exposed by the
mandated main `linx-toolchain-build` checkout:

- **`bench_reduce`** uses a 1-column output tile (`ValidCol==1`).
- **dtype fixes**: active `TABS` covers FP16/FP32; BF16 remains recorded but is
  unsupported because the main compiler crashes during instruction selection.
- **unsupported cases** are explicit in `coverage.json`. TSELECT is absent from
  the installed compiler headers, and generic TMOV currently reaches an
  assembler `B.DATR` matcher error. TCMP/TCMPS/THISTOGRAM and masked gather/
  scatter likewise remain explicit while their emitted encodings are rejected.
  None of these operations is replaced by a fallback.
- **memory layout**: there is no synthetic `TLOAD_ND2NZ` operation. CUBE layout
  transport is tested through the real CUBE tile types and TLOAD_CUBE/TSTORE_CUBE.
- **cube**: A uses CUBE_M16/M32, B uses CUBE_N8, and accumulator/output uses
  CUBE_M16/M32. M=64 is an operator tiling problem rather than one CUBE tile.
- **MX** is not represented by an FP16 placeholder. It remains unsupported here
  until a real MX input and E8M0-scale fixture with an oracle is added.
- **fixp**: 94 modes are active; `LRELU_ONLY` is blocked by a toolchain `B.IOR`
  matcher gap (see `fixp/issues/`). `report_fixp.py` decodes `B.FPATR`/`B.IOR`/
  `B.IOT`/`B.IOS` from each `.diss` into `fixp_report.md`.
- scalar disassembly confirms lowering to scalar micro-ISA (e.g. `addw` chain).
