# Kernel Functions and Entry Points

A kernel entry point is a C++ function template or specialization compiled for
the target. It receives global tensor views, scalar shape values, and any mode
parameters needed by the tile operations.

## Entry Shape

```cpp
template <uint32_t kRows, uint32_t kCols>
void tile_add(GlobalTensor<float> a,
              GlobalTensor<float> b,
              GlobalTensor<float> c,
              uint32_t block_count) {
  const uint32_t block = get_block_idx();
  const uint32_t thread = get_thread_idx();

  RowSlice rows = partition_rows(kRows, block, block_count, thread);
  compute_tile(rows, a, b, c);
}
```

Keep launch-dependent values such as `block_count` explicit. Keep tile payload
sizes in named constants so the compiler can validate tile types.

## Specialization

Documentation examples often end with an explicit template instantiation. That
forces object generation for a concrete shape even when the file has no host
`main`.

```cpp
template void tile_add<16, 16>(GlobalTensor<float>,
                               GlobalTensor<float>,
                               GlobalTensor<float>,
                               uint32_t);
```

Benchmark entries use the same idea with larger shape matrices and harness
provided data.

## Entry Rules

- Do not hide tile operations behind untyped pointer arithmetic.
- Pass global tensor views with shape, stride, layout, and valid-region data.
- Use block/thread identity only to select ownership and control paths.
- Store only ranges owned by the current block/thread pair.
