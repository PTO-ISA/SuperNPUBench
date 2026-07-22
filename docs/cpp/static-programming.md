# Compile-Time Shapes and Templates

Tile code works best when payload shape, element type, layout, and operation
mode are compile-time properties.

## Shape Constants

```cpp
static constexpr uint32_t kTileRows = 16;
static constexpr uint32_t kTileCols = 16;
static constexpr uint32_t kTileBytes =
    kTileRows * kTileCols * sizeof(float);
```

Use `static_assert` to guard operation-specific requirements:

```cpp
static_assert(kTileBytes >= kMinimumThreadFragmentBytes);
static_assert(kTileRows % kThreadsPerBlock == 0);
```

## Template Parameters

```cpp
template <typename Element,
          uint32_t kRows,
          uint32_t kCols,
          Layout kLayout>
using Fragment = LocalTile<Element, kRows, kCols, kLayout>;
```

Templates keep the tile type precise without duplicating operation code for
every shape.

## Compile-Time Branches

Use `if constexpr` for mode choices that should disappear before lowering:

```cpp
if constexpr (kUseAccumulation) {
  TMATMUL_ACC(out, lhs, rhs, accum);
} else {
  TMATMUL(out, lhs, rhs);
}
```

Use runtime branches only when the decision depends on runtime data.
