# TMOV: Move data between tile registers

## Purpose / When to use

Use `TMOV` for tile-to-tile copies and implementation-supported tile-register transfers. Use the shared forms to insert, publish, extract, or broadcast versioned shared tile values without changing the byte representation.

## C++ Declaration

```cpp
template <typename DstTileData, typename SrcTileData>
PTO_INST void TMOV(DstTileData &dst, SrcTileData &src);

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
PTO_INST void TMOV(DstTileData &dst, SrcTileData &src);

template <typename DstTileData, typename SrcTileData, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INST void TMOV(DstTileData &dst, SrcTileData &src);

template <typename DstTileData, typename SrcTileData, typename FpTileData,
          AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INST void TMOV(DstTileData &dst, SrcTileData &src, FpTileData &fp);

template <typename DstTileData, typename SrcTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INST void TMOV(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar);

template <typename DstTileData, typename SrcTileData, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INST void TMOV(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar);

TMOV<SharedMove::Insert>(shared, local_fragment);
TMOV<SharedMove::Publish>(shared, full_local_value);
TMOV<SharedMove::Extract>(local_fragment, shared);
TMOV<SharedMove::Broadcast>(full_local_copy, shared);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `dst` | Destination tile for local moves, extracts, or broadcasts. |
| `src` | Source tile for local moves, publishes, or inserts. |
| `fp` | Auxiliary floating-point tile used by selected accumulator-to-vector modes. |
| `preQuantScalar` | Optional scalar used by quantized move paths. |
| `mode` | Move or accumulator conversion mode. |
| `reluMode` | Optional ReLU preprocessing mode. |
| `shared` | Versioned shared tile register used by `SharedMove` forms. |

## Operation / Semantics

- The pure copy form preserves element bytes over the effective source/destination valid region.
- `Mat`, `Vec`, and `Acc` location pairs are limited to supported implementation paths.
- `SharedMove::Insert` and `SharedMove::Extract` address the current thread's shared fragment.
- `SharedMove::Publish` writes a complete shared value from one selected PE, and `SharedMove::Broadcast` copies a complete shared value into every participating PE local tile.
- Every shared destination write creates a new shared tile version.

## Constraints

- Shapes must match for fixed-shape local moves unless the selected mode documents a narrower effective region.
- Shared `TMOV` does not transpose, reshape, convert, or reinterpret element layout.
- Accumulator conversion modes must use supported source and destination dtypes.
- Compiler lowering for shared forms is an architectural contract and may lag the C++ surface.

## Example

```cpp
TMOV(dst, src);
TMOV<SharedMove::Broadcast>(local_copy, shared_value);
```

## Common mistakes

- Using `TMOV` when `TRESHAPE`, `TTRANS`, or `TCVT` is the operation actually needed.
- Publishing a shared version from multiple PEs.
- Expecting shared insert/extract to address arbitrary rows instead of the current thread fragment.

## Used by kernels

- Memory microbenchmarks describe `TMOV` coverage in `microbenchmark/memory/memory_bench.hpp`.
- Group-programming tutorials use shared `TMOV` forms for PE-local and shared-tile exchange.
- Some current benchmark paths use `TCVT` as a compatibility fallback when `TMOV` lowering is unavailable.
