# Build, Run, and Disassemble

Use separate checks for source compilation, object inspection, linked
benchmark construction, and simulator execution. A successful compile proves
the C++ and tile API surface; it does not prove runtime behavior.

## 1. Compile the Documentation Examples

```bash
cd "$SUPERNPU_ROOT"
KEEP_DOC_OBJECTS=1 bash docs/examples/check.sh
```

This preserves objects under `docs/examples/output/`. For a compile-only check
that removes temporary objects, omit `KEEP_DOC_OBJECTS=1`:

```bash
bash docs/examples/check.sh tile-add
```

## 2. Inspect Named Operation Blocks

```bash
"$COMPILER_DIR/llvm-objdump" -dr \
  docs/examples/output/tile_add.o | less
```

Look for named operation block headers associated with the source operations,
such as `BSTART.TLOAD`, `BSTART.TADD`, and `BSTART.TSTORE`. Surrounding scalar
code and exact scheduling may change with compiler revisions.

## 3. Build a Linked Benchmark

The benchmark harness supplies startup code, linker inputs, and runtime
objects. Start with the compile-checked tile-add pipeline:

```bash
cd "$SUPERNPU_ROOT/benchmark/one-level-arch/test/kernel/pto_kernels"
make TESTCASE=pto_add
```

The Makefile reads `COMPILER_DIR` and `LINX_SYSROOT` from the environment and
writes the ELF under `benchmark/one-level-arch/output/`.

## 4. Generate Disassembly

Select the `diss` target:

```bash
make TESTCASE=pto_add diss
```

The harness writes `<target>.diss` next to the ELF.

## 5. Run on the Simulator

```bash
ELF="$SUPERNPU_ROOT/benchmark/one-level-arch/output/kernel/pto_kernels/elf/kernel_pto_kernels_pto_add.elf"
"$QEMU" -M virt -nographic -kernel "$ELF"
```

The small bare-metal benchmark returns to its startup loop after the kernel.
Use ++ctrl+a++, then ++x++, to leave the QEMU console. A silent console is not
a correctness result; use a result-checking harness or trace when validating
tensor values.

## Interpret Failures by Stage

| Stage | Typical signal | First check |
| --- | --- | --- |
| C++ frontend | type or template diagnostic | element types, shapes, valid extents |
| Tile lowering | compiler crash or illegal operation | minimal source and compiler revision |
| Link | missing symbol or relocation | sysroot, startup objects, selected build mode |
| Simulator load | ELF or target error | target triple and simulator revision |
| Runtime | trap, wrong result, timeout | disassembly, generated data, debug trace |

For instruction tracing, add `-d in_asm,exec -D run.log` to the QEMU command.
Use it after a normal launch because the trace is intentionally much larger.
