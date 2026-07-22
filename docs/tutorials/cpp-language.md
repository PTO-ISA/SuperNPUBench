# C++ for One-Level PTO Kernels

## Static Shapes

Tile shapes should be compile-time constants so the compiler can assign tile
IDs, storage sizes, and block dimensions.

```cpp
template <int Rows, int Cols>
struct TileShape {
  static_assert(Rows > 0 && Cols > 0);
  static constexpr int rows = Rows;
  static constexpr int cols = Cols;
};
```

## Identity-Based Branches

```cpp
const uint32_t pe = get_thread_idx();
if (pe == 0) {
  producer();
} else {
  consumer(pe);
}
```

The compiler may lower a statically analyzable PE branch to a PE mask. A
data-dependent branch remains ordinary divergent control flow.

## Core Tensor Slices

```cpp
const uint32_t core = get_block_idx();
const uint32_t begin = rows * core / core_count;
const uint32_t end = rows * (core + 1) / core_count;
auto slice = tensor.slice_rows(begin, end - begin);
```

Pass `core_count` and global shape through kernel arguments or launch metadata.

## Tile Values

```cpp
PETile<float, 16, 16> lhs;
PETile<float, 16, 16> rhs;
PETile<float, 16, 16> out;

TLOAD(lhs, lhs_global);
TLOAD(rhs, rhs_global);
TADD(out, lhs, rhs);
TSTORE(out_global, out);
```

Each destination call defines a new logical tile version. Keep dependencies in
ordinary C++ dataflow rather than relying on instruction spacing.

## Architecture-Level Shared Types

```cpp
SharedTile<half, 64, 64> shared_rhs;
```

Shared tile-register types are part of the architecture contract but are not
yet required to compile. See [group programming](group-programming.md) for the
complete examples.

## Current Source Examples

Compile-checked C++ is available in:

- `benchmark/one-level-arch/test/kernel/broadcast/`
- `benchmark/one-level-arch/test/kernel/element_wise/gelu/`
- `benchmark/one-level-arch/test/kernel/matmul/`
- `benchmark/one-level-arch/test/kernel/fa/`
