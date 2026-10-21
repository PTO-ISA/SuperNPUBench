# Portable topk RES_CHECK fixture

This bundle includes the gfsim lockstep RES_CHECK ELF, matching disassembly,
all input files, independent FP32 golden data, a runner and a SwimLane helper.
Copy this entire directory to another checkout or machine; no rebuild is needed.
The ELF reads `./shape.bin`, `./input.bin`, `./starts.bin`, `./ends.bin` and
writes results in its current working directory. The runner creates that
working directory, copies verified inputs into it, and leaves this bundle intact.
Model paths are supplied by the collaborator at runtime.

```bash
python3 eval/topk/run.py --gfsim /path/to/SuperScalarModel/bin/gfsim \
    --outdir ./perf_runs/topk --timeout 120
```

The supplied model must provide `configs/fourpe.conf` beside its `bin/` directory.
The runner passes its full path, so neither the caller's cwd nor the author's
model directory is required. gfsim must include the per-thread age-queue fix
from SuperScalarModel PR #951. Each gfsim run requests a SwimLane trace;
on successful completion `topk_swim.log`, `topk.json` and
`topk_nocounters.json` are saved in the chosen output directory. A model abort
may prevent trace finalization; such a run fails and retains its log.

## Fixture format

All data is little-endian. Seed 12345; shape batch=4, cols=8192, topk=512.

| File | Format | Bytes |
| --- | --- | ---: |
| inputs/shape.bin | int64[3]: batch, cols, topk | 24 |
| inputs/input.bin | float32[4*8192+32], including tail padding | 131200 |
| inputs/starts.bin | int32[4]: 0, 123, 7, 31 | 16 |
| inputs/ends.bin | int32[4]: 8192, 6000, 1531, 8191 | 16 |
| inputs/golden.bin | int32[4*512], ascending selected indices per row | 8192 |

Golden indices are generated on the host by sorting each row's input values,
independently of model output. The runner recomputes and verifies them before
execution, compares input readback, checks error flags and compares the complete
output index sets. SHA256 and byte counts of shipped payloads are recorded in
`manifest.json`. Missing ELF/input/output, corrupt payloads, failed model exits
and comparisons are failures.

## Build

From `benchmark/one-level-arch/test/kernel/multi_thread/topk_tiled`:

```bash
make res_check=on GFSIM=on -j4 diss
```

This testcase defaults to `CHK_DIR=.`; its checker runs the model from the
isolated case directory. The common Makefile now accepts a `CHK_DIR` override;
other testcases retain their previous default. Published ELFs must use relative
paths. Replacing this ELF requires regenerating the matching disassembly and
manifest, and validating a relocated copy of the bundle.

## Validation limits

The original test entry's isolated I/O smoke test passed 22 four-PE lockstep
syscalls with zero lockstep timeouts. The separately built default gfrun entry has passed functional comparison;
this GFSIM ELF is for the gfsim lockstep ABI and must not be treated as a
gfrun-compatible fixture. Full gfsim topk remains blocked in computation at MGATHER_ADD;
this bundle is a reproducible failing model test, not a gfsim performance or
numerical PASS. See SuperScalarModel issue #959 for the distinction between
file-I/O protocol compatibility and the remaining computation blocker.
