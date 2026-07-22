# Troubleshooting Builds and Runs

## Compiler Directory Is Rejected

`COMPILER_DIR` must point to the directory containing the Linx binaries, not the
LLVM build root:

```bash
export COMPILER_DIR="$LINXISA_ROOT/compiler/llvm/build-linxisa-clang/bin"
test -x "$COMPILER_DIR/clang"
test -x "$COMPILER_DIR/ld.lld"
test -x "$COMPILER_DIR/llvm-objdump"
```

## Headers Are Missing

Build from the benchmark directory shown on its catalog page. The Make harness
calculates include roots relative to that directory. Avoid invoking a raw
compiler command until the Make dry run shows the complete include and define
set.

```bash
make -n TESTCASE=<case> PLAT=linx COMPILER_DIR="$COMPILER_DIR"
```

## Sysroot or Runtime Symbols Are Missing

Export the phase-b musl sysroot and confirm it contains target headers and
libraries:

```bash
export LINX_SYSROOT="$LINXISA_ROOT/out/libc/musl/install/phase-b"
test -d "$LINX_SYSROOT/usr/include"
find "$LINX_SYSROOT" -name 'libc.a' -o -name 'crt1.o'
```

Bare-metal cases may use a repository linker script and freestanding runtime
instead. Follow the exact benchmark command.

## Unsupported Tile Type or Layout

Read the selected intrinsic page. Compare all source and destination element
types, location roles, full/valid shapes, and layouts. An element count match is
not sufficient for layout compatibility.

## Expected Block Is Missing

Disassemble the object or ELF and search for the named block:

```bash
"$COMPILER_DIR/llvm-objdump" -d output.elf | less
```

Check that the build targets `linx64-linx-none-elf`, that `__linx` is defined by
the target compiler, and that the intended compatibility/current PTO path was
selected.

## QEMU Fails Before the Kernel

Confirm the QEMU executable, machine/CPU selection, target ELF ABI, and runtime
image are from compatible LinxISA revisions. A userspace ELF and a bare-metal
ELF require different launch paths.

## Wrong Numeric Result

Work outward from the first mismatching tile:

1. tensor shape/stride and iterator offset;
2. load layout and valid region;
3. matrix left/right role or vector layout;
4. reduction axis and expand direction;
5. conversion rounding/packing/scaling;
6. store layout and output extent;
7. ownership, overlap, and synchronization.

## Data-Object Link Failure

Control and sort cases generate or assemble data objects before linking. Run the
manifest command from its own directory and do not bypass `pre_work`. The
benchmark page lists every checked-in generator/data input.

## Catalog Looks Stale

Regenerate both authoritative surfaces:

```bash
python3 scripts/sync_golden_manual.py --davincioo-root "$DAVINCIOO_ROOT"
python3 scripts/generate_benchmark_manual.py
python3 -m mkdocs build --strict
```
