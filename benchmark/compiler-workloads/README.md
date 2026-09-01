# Compiler workloads

This layer validates ordinary C/C++ compilation, ELF/link contracts, runtime
semantics, and auto-vectorization. It is deliberately separate from explicit
TileOP microbenchmarks and one-level operators.

The source bundle is pinned in `sources.lock.json`. Fetch it without vendoring
third-party source into this repository:

```bash
python3 benchmark/compiler-workloads/fetch_sources.py
```

The command installs a locked LinxISA snapshot plus PolyBench/C and cTuning
codelets under `.cache/`; TSVC is already pinned and vendored by that LinxISA
snapshot. Mutable upstream branch archives are guarded by SHA-256 in the lock
file, so an upstream change fails closed instead of silently changing coverage.

The command prints the locked `workloads/` directory. Run the default portfolio
with the mandated main compiler checkout:

```bash
python3 benchmark/compiler-workloads/run_portfolio.py \
  --compile-only
```

The default portfolio contains:

- CoreMark and Dhrystone for ordinary C, link, and libc coverage;
- PolyBench `gemm` and `jacobi-2d` for affine numerical loops;
- TSVC `off` and `auto` builds for scalar/autovectorization coverage;
- all 44 cTuning MILEPOST codelets for varied small-program control/data flow.

`run_portfolio.py` defaults to the locked source path produced by
`fetch_sources.py`. Use `--workloads-root` only to override that location.
PolyBench and TSVC can be disabled with `--no-polybench` and `--no-tsvc`;
`--ctuning-limit 0` disables cTuning, while a smaller positive limit selects a
smoke subset of its 44-codelet corpus. `--tsvc-modes` selects any of
`off,mseq,mpar,auto`.

Add `--run-command '<runtime> {exe}'` for hosted CoreMark/Dhrystone/PolyBench
execution. Add `--qemu /path/to/qemu-system-linx64` for TSVC and cTuning
freestanding execution. Without those runtime adapters each suite still builds
and emits ELF/object, disassembly, vectorization, and JSON evidence.

Compile-only results are build evidence, not runtime correctness. Full PASS
requires the upstream workload runner's semantic markers/checksum checks.

Generated artifacts and consolidated JSON/Markdown reports are written under
`output/compiler-workloads/`. The generated compatibility tree adapts the
locked LinxISA runner to the external `linx-toolchain-build` compiler required
by this repository; locked source files remain unchanged. In compile-only mode
it omits the legacy soft-float runtime (the current compiler asserts while
building that support file) and maps unavailable LinxISA-private TSVC autovec
switches to Clang's public vectorization on/off flags. Runtime mode retains the
soft-float runtime so an unsupported runtime build is reported rather than
silently producing an incomplete image.
