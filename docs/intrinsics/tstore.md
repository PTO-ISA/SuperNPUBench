# TSTORE: Store a tile into a global tensor view

## Purpose / When to use

Use `TSTORE` to write a tile valid region back to a `GlobalTensor` or iterator-produced global view. Use the quantized or atomic template forms only when the selected backend supports the requested store mode.

## C++ Declaration

```cpp
template <typename TileData, typename GlobalData,
          AtomicType atomicType = AtomicType::AtomicNone>
PTO_INST void TSTORE(GlobalData &dst, TileData &src);

template <typename TileData, typename GlobalData,
          AtomicType atomicType = AtomicType::AtomicNone>
PTO_INST void TSTORE(GlobalData &dst, TileData &src, uint64_t preQuantScalar);

template <typename GlobalData, typename SharedTileData>
PTO_INST void TSTORE(GlobalData &dst, const SharedTileData &shared_src);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `dst` | Global tensor destination view. |
| `src` | Source tile whose valid rows and columns define the write region. |
| `preQuantScalar` | Optional scalar used by quantized accumulator-store paths. |
| `atomicType` | Optional atomic write mode. |
| `shared_src` | Ready shared tile version used as the source for a group store. |

## Operation / Semantics

- Copies `src` into `dst` over `src.GetValidRow()` by `src.GetValidCol()`.
- Atomic forms combine the source values with existing global values according to `atomicType`.
- Quantized forms apply the selected pre-quantization scaling before writing.
- For shared tile sources, a statically selected thread may store the full shared value, or each thread may store its defined fragment to non-overlapping global regions.

## Constraints

- Source tile location is `Vec`, `Mat`, or `Acc` where supported by the implementation.
- Element sizes normally match for non-quantized stores.
- Destination shape and source valid extents must be greater than zero.
- Vector stores that need dtype conversion should materialize the converted tile first, usually with `TCVT`.
- Shared tile stores are an architectural contract; compiler lowering support may lag the C++ surface.

## Example

```cpp
TSTORE(global_tile, tile);
```

## Common mistakes

- Reversing operands; `TSTORE` takes the global destination first.
- Storing more elements than the source tile valid region describes.
- Assuming same-address overlapping writes are ordered unless using a supported atomic form.

## Used by kernels

- Memory microbenchmarks use `TSTORE` in `microbenchmark/memory/memory_bench.hpp`.
- Most benchmark kernels pair `TSTORE` with `TLOAD` for final output writes.
- DeepSeek expansion and reduction kernels use repeated `TSTORE` calls to materialize tiled outputs.
