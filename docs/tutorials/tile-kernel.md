# Build a Tile Kernel

This tutorial follows a compact elementwise kernel. The same structure applies
to larger kernels: select a tensor slice, load local tiles, compute tile
versions, and store the result.

## Select an Entry Point

The GELU entry is a compact checked example:

```text
benchmark/one-level-arch/test/kernel/element_wise/gelu/src/gelu.cpp
```

Its implementation is in:

```text
benchmark/one-level-arch/kernels/element_wise/gelu_pto.hpp
```

The `pto` string is part of a literal path in the repository.

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

Each destination is a new tile version. The exponential depends on `scaled`,
the reciprocal depends on `exponential`, and the final multiply depends on
`input_tile` and `normalized`.

## Compile

```bash
cd benchmark/one-level-arch/test/kernel/element_wise/gelu
PLAT=linx COMPILER_DIR=/path/to/linx-isa/compiler/llvm/build-linxisa-clang/bin \
  make TESTCASE=gelu
```

Set `LINX_SYSROOT` when the compiler cannot infer the target sysroot.

## Partition Rows

Use block identity for the coarse slice and thread identity for the fragment:

```cpp
const uint32_t block = get_block_idx();
const uint32_t thread = get_thread_idx();

RowSlice rows = partition_rows(total_rows, block, block_count, thread);
auto slice = input.slice_rows(rows.begin, rows.count);

TLOAD(input_tile, slice);
```

Confirm the local fragment is at least `kMinimumThreadFragmentBytes`, or route
small tails through padding, packing, or a scalar path.

## Verify the Object

Disassemble the built object or ELF and confirm that logical tile producers and
consumers use the intended tile IDs. Independent chains may be interleaved by
the superscalar backend as long as version dependencies are preserved.

The generated [benchmark catalog](../benchmarks/index.md) shows complete
source closure and supported intrinsic surface for checked builds.
