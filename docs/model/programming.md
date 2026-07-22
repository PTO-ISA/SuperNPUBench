# C++ Programming Model

Write kernels as ordinary C++ functions specialized by templates and launch
metadata. Use block/thread identity to select data, tile types to describe
storage, and tile operations to express movement and compute.

## Kernel Skeleton

```cpp
template <uint32_t kRows, uint32_t kCols>
void kernel(GlobalTensor<float> input,
            GlobalTensor<float> output,
            uint32_t block_count) {
  const uint32_t block = get_block_idx();
  const uint32_t thread = get_thread_idx();

  RowSlice rows = partition_rows(kRows, block, block_count, thread);

  LocalTile<float, kRowsPerThread, kCols> in;
  LocalTile<float, kRowsPerThread, kCols> out;

  TLOAD(in, input.slice_rows(rows.begin, rows.count));
  compute(out, in);
  TSTORE(output.slice_rows(rows.begin, rows.count), out);
}
```

The kernel is still C++. Use templates for static shapes, `if constexpr` for
compile-time selection, and normal branches for runtime choices.

## Shape Rules

Prefer compile-time tile shapes. A static shape lets the compiler assign tile
storage, validate layout compatibility, and form named operation blocks.

Runtime extents are still useful for tensor bounds and tails. Keep runtime
extents in scalar variables and keep tile payload shapes in named constants:

```cpp
static constexpr uint32_t kTileRows = 16;
static constexpr uint32_t kTileCols = 16;
static constexpr uint32_t kTileBytes =
    kTileRows * kTileCols * sizeof(float);
```

If a thread fragment would be smaller than
`kMinimumThreadFragmentBytes`, pad the tile storage or choose a scalar/tail
path. Do not read or store padded elements as if they were valid data.

## Value Definitions

Every tile destination defines a new version:

```cpp
TLOAD(a0, a_global);
TLOAD(b0, b_global);
TADD(sum0, a0, b0);
TMUL(scale0, sum0, alpha);
TSTORE(out_global, scale0);
```

The compiler may schedule `TLOAD(a0, ...)` and `TLOAD(b0, ...)` in either order
because neither consumes the other. `TADD` must wait for both load versions,
and `TMUL` must wait for the `TADD` version.

## Branches

Branches may define different tile versions on different thread paths:

```cpp
if (thread == 0) {
  TLOAD(metadata, metadata_view);
} else {
  TLOAD(payload, payload_view);
}
```

A later operation must consume only versions that are defined on the paths that
reach it. When a grouped operation requires all participating threads, every
participant must reach the same dynamic operation instance with compatible
source versions.

## Dependency Ordering

Express ordering through tile values:

- pass the produced tile to the consumer that needs it;
- use a shared tile version when threads must exchange tile payload;
- store to global memory only after the producing tile version exists;
- rely on the runtime only for launch-level ordering outside the kernel.

Do not write code whose correctness depends on instruction spacing, assumed
latency, or physical tile-register reuse.
