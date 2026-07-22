# Build a One-Level Tile Kernel

## Select an Entry Point

The GELU entry is a compact one-level example:

```text
benchmark/one-level-arch/test/kernel/element_wise/gelu/src/gelu.cpp
```

Its implementation is in:

```text
benchmark/one-level-arch/kernels/element_wise/gelu_pto.hpp
```

## Dataflow

The kernel follows a load-compute-store value graph:

```cpp
TLOAD(input_tile, input_global);
TMULS(scaled, input_tile, scale);
TEXP(exponential, scaled);
TRECIP(normalized, exponential);
TMUL(output_tile, input_tile, normalized);
TSTORE(output_global, output_tile);
```

The exact implementation page in the [benchmark catalog](../benchmarks/index.md)
shows the complete source closure and the 0.57 intrinsics reached by the active
build.

For additional source-backed examples covering memory movement, tile addition,
six GEMM forms, and a two-stage attention kernel, see
[Migrated PTO Tile Kernels](migrated-pto-kernels.md).

## Compile

```bash
cd benchmark/one-level-arch/test/kernel/element_wise/gelu
PLAT=linx COMPILER_DIR=/path/to/linx-isa/compiler/llvm/build-linxisa-clang/bin \
  make TESTCASE=gelu
```

Set `LINX_SYSROOT` when the compiler cannot infer the Linx musl installation.

## Partition Across Cores

Use `get_block_idx()` to choose the core slice, then `get_thread_idx()` to
choose a quarter inside the core.

```cpp
const uint32_t block = get_block_idx();
const uint32_t pe = get_thread_idx();
auto slice = input.slice_rows(block * rows_per_core + pe * rows_per_pe,
                              rows_per_pe);
TLOAD(input_tile, slice);
```

## Verify

Disassemble the built object or ELF and confirm that logical tile producers and
consumers use the intended relative tile IDs. Independent chains may be
interleaved by the superscalar backend.
