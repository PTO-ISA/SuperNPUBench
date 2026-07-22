# C++ Language Patterns for NPU Kernels

Use C++ to make tile programs explicit: static shapes, named constants,
ordinary branches, and typed values.

## Static Shapes

Tile shapes should be compile-time constants so the compiler can assign tile
IDs, storage sizes, and block dimensions.

```cpp
template <uint32_t Rows, uint32_t Cols>
struct TileShape {
  static_assert(Rows > 0);
  static_assert(Cols > 0);
  static constexpr uint32_t rows = Rows;
  static constexpr uint32_t cols = Cols;
};
```

Use descriptive constants rather than repeating profile values:

```cpp
static constexpr uint32_t kThreadsPerBlock = /* profile value */;
static constexpr uint32_t kMinimumThreadFragmentBytes = /* profile value */;
static constexpr uint32_t kTileRows = 16;
static constexpr uint32_t kTileCols = 16;
```

## Identity-Based Branches

```cpp
const uint32_t thread = get_thread_idx();

if (thread == 0) {
  produce_shared_value();
} else {
  consume_local_fragment(thread);
}
```

A statically analyzable identity branch may become a masked tile operation.
A data-dependent branch remains ordinary divergent C++ control flow.

## Block Tensor Slices

```cpp
const uint32_t block = get_block_idx();
const uint32_t begin = rows * block / block_count;
const uint32_t end = rows * (block + 1) / block_count;
auto block_rows = tensor.slice_rows(begin, end - begin);
```

Pass `block_count` and global shape through kernel arguments or launch
metadata. Then split `block_rows` by `thread` when the tile operation expects
per-thread fragments.

## Tile Values

```cpp
LocalTile<float, kTileRows, kTileCols> lhs;
LocalTile<float, kTileRows, kTileCols> rhs;
LocalTile<float, kTileRows, kTileCols> out;

TLOAD(lhs, lhs_global);
TLOAD(rhs, rhs_global);
TADD(out, lhs, rhs);
TSTORE(out_global, out);
```

Each destination call defines a new logical tile version. Keep dependencies in
ordinary C++ dataflow rather than relying on instruction spacing.

## Shared Tile Types

```cpp
SharedTile<half, kK, kN> shared_rhs;
```

Shared tile types represent block-local versions. Use them for register-sized
exchange, shared global loads, shared stores, and the right operand of grouped
matrix operations.

## Source Examples

Compile-checked C++ examples are available in:

- `benchmark/one-level-arch/test/kernel/broadcast/`
- `benchmark/one-level-arch/test/kernel/element_wise/gelu/`
- `benchmark/one-level-arch/test/kernel/matmul/`
- `benchmark/one-level-arch/test/kernel/fa/`
