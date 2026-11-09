# RMSNorm single-buffer timing snapshot, 2026-11-09

This separate source archive preserves the historical **43,694 Total Cycles**
baseline. It adds no default operator or test entry and does not replace the
IO ping-pong kernel merged in [PR271](https://github.com/PTO-ISA/SuperNPUBench/pull/271).
It contains no continuous8, prefetch-store-priority or dual-output variant.

## Frozen source

- Kernel: SuperNPUBench `e7a4a761b880b9ea820e5106b64756ebbc6da998`, identical
  to the parent of IO ping-pong commit `92ecf231270dcfdbca975580d9644c7e68eb0ed6`.
- Driver: the same historical driver, with only input/gamma/output declarations
  changed to `alignas(512)`. The R=8192 path does not use partial_workspace;
  its declaration remains historical to preserve the measured source bytes.
- Alignment reference: wangyu's already merged
  [PR267](https://github.com/PTO-ISA/SuperNPUBench/pull/267), commit
  `68451bf75414d397c4bcede7a00581ed73585e09`. Current main also aligns workspace.
- Support headers and startup assembly come from the same historical commit.

Complete source and historical artifact hashes are in [identity.json](identity.json).
No personal ELF paths or unpublished input files are required by this zero-input
build. Input and gamma are static zero buffers, A=128/R=8192/FP16/4PE;
RES_CHECK is deliberately disabled. No nonzero correctness check or negative
control has been performed; this is a scheduling observation, not acceptance.

## Build and optional timing reproduction

From any working directory, set `ARCHIVE_DIR` to this directory and run:

```bash
python3 "$ARCHIVE_DIR/verify_sources.py"
python3 "$ARCHIVE_DIR/build_snapshot.py" \
  --toolchain-bin "$LINX_TOOLCHAIN_BIN" --outdir "$NEW_RUN_DIR"
```

The build requires the frozen clang identified by SHA256 in identity.json,
with its matching TileOP API headers, sysroot and linker. It emits an ELF,
matching disassembly, build logs and command/hash manifest to a new output
directory outside this archive. Prefix maps remove build-source/toolchain paths.
It refuses to overwrite an output directory or silently substitute a compiler.

To also run timing, append `--gfsim "$HISTORICAL_GFSIM"` to the build command.
The supplied runtime must match the recorded historical model hash. The runner
copies the native configuration into the output directory's `configs/`, uses
that directory as cwd for the old configuration parser, and sets
`RMS_TAIL_BYPASS=1`. Python 3 and Linux `resource` support are required; the
SwimLane helper uses a 6GiB host address-space ceiling for JSON export.
It produces `perf_run/rmsnorm_swim.log`, `rmsnorm.json` and
`rmsnorm_nocounters.json` and fails on simulator errors or missing/invalid trace.

The historical timing conditions are PR1145 lazy allocation/first-data-ready,
tail bypass, 256KiB CellReg/PE, read/write 128 B/c each with aggregate 128 B/c,
random SoC uniform[300,400], seed=2. The native conf is shipped byte-for-byte.
The helper and streaming JSON reader preserve instruction/flow events while
filtering PMU counters from the nocounters file.

The compiler, historical runtime, ELF and full traces are **not hosted in this
source archive**. Their hashes identify the author-retained artifacts; obtaining
the matching tools remains necessary for independent timing reproduction.
The source recipe is not a complete NDF result bundle. Rebuilt ELF hashes may
differ after path remapping; compare source/tool/config identities and actual
cycles rather than assuming binary equality.

## Scope of the result

The same fixed historical runtime/config measured this single-buffer version at
43,694 cycles and the exploratory dual-output version at 44,246 cycles.
No valid same-configuration comparison against current main's IO ping-pong
version exists: it hit a descriptor assertion with that historical tool
combination. This archive does not claim IO ping-pong is slower or propose
reverting it. The PR271-reported improvement belongs to its own configuration.

The archived recipe was also rebuilt from a copied source directory and run
from an unrelated cwd: **43,694 Total Cycles**, rc=0, with full/nocounters
SwimLane produced. The portable ELF instruction listing matches the historical
baseline; ELF and disassembly personal-path scans pass. This check does not add
nonzero numerical validation.
