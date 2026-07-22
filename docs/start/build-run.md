# Build, run, and disassemble

Use three separate checks: compile the source, inspect the emitted block
templates, then execute a linked benchmark ELF. A successful compile alone does
not prove runtime behavior.

## 1. Compile the documentation examples

```bash
cd "$SUPERNPU_ROOT"
KEEP_DOC_OBJECTS=1 bash docs/examples/check.sh
```

This preserves objects under `docs/examples/output/` for the disassembly step.
For a compile-only check that cleans up automatically, omit
`KEEP_DOC_OBJECTS=1`:

```bash
bash docs/examples/check.sh tile-add
```

## 2. Inspect named block templates

```bash
"$COMPILER_DIR/llvm-objdump" -dr \
  docs/examples/output/tile_add.o | less
```

For the active hardware path, look for named block headers associated with the
operations in the source, such as `BSTART.TLOAD`, `BSTART.TADD`, and
`BSTART.TSTORE`. Exact scheduling and surrounding scalar code may change with
compiler revisions; the named v0.57 operation identities must remain visible.

## 3. Build a linked benchmark

The test harness supplies startup, linker, runtime, and generated-data objects.
For a small matrix-multiply manifest case:

```bash
cd "$SUPERNPU_ROOT/benchmark/one-level-arch/test/kernel/matmul"
make TESTCASE=matmul TYPE=MASK MODE=MASK_FP32 \
  M=256 N=256 K=256 tM=32 tN=32 tK=64
```

The Makefile reads `COMPILER_DIR` and `LINX_SYSROOT` from the environment and
writes the ELF beneath `benchmark/one-level-arch/output/`.

!!! note "Revision alignment"
    The documentation examples in step 1 are compile-checked independently.
    Linked benchmark cases also depend on matching benchmark headers, runtime,
    and compiler revisions. Treat a template diagnostic here as a compatibility
    issue until those revisions are confirmed.

## 4. Generate benchmark disassembly

Repeat the same variables and select the `diss` target:

```bash
make diss TESTCASE=matmul TYPE=MASK MODE=MASK_FP32 \
  M=256 N=256 K=256 tM=32 tN=32 tK=64
```

The harness writes `<target>.diss` next to the ELF.

## 5. Run on QEMU

```bash
make sim QEMU="$QEMU" \
  TESTCASE=matmul TYPE=MASK MODE=MASK_FP32 \
  M=256 N=256 K=256 tM=32 tN=32 tK=64
```

The `sim` target passes the ELF to `qemu-system-linx64` with the benchmark's
block-optimization and memory-size options.

!!! tip "Debug lane"
    `make debug QEMU="$QEMU" ...` writes QEMU CPU, assembly, execution, and
    Linx-memory traces to `run.log`. Use it after a normal `sim` failure; trace
    output is intentionally much larger.

## Interpret failures by stage

| Stage | Typical signal | First check |
| --- | --- | --- |
| C++ frontend | type or template diagnostic | tile element types, shapes, valid extents |
| Linx lowering | compiler crash or illegal operation | minimal source and exact compiler revision |
| Link | missing symbol or relocation | `LINX_SYSROOT`, startup objects, selected build mode |
| QEMU load | ELF or target error | target triple and QEMU revision |
| Runtime | trap, wrong result, timeout | disassembly, generated data, `make debug` trace |
