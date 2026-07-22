# get_block_idx() - Read the block index

## Purpose and when to use

`get_block_idx()` returns the logical block assigned to the current kernel
instance. Use it to partition a global tensor into independent block-sized
regions before threads divide the region further.

## C++ declaration

```cpp
PTO_INST uint32_t get_block_idx();
```

## Parameters and result

| Value | Meaning |
| --- | --- |
| Return value | The current block index in the launch grid. |

## Execution semantics

All threads in one block observe the same value. The launch configuration defines
the valid range. The function identifies work; it does not imply ordering,
communication, or memory visibility between different blocks.

## Example

```cpp
const uint32_t block = get_block_idx();
const uint32_t thread = get_thread_idx();

const uint32_t block_row = block * kRowsPerBlock;
const uint32_t thread_row = block_row + thread * kRowsPerThread;
auto input_fragment = input.slice_rows(thread_row, kRowsPerThread);
TLOAD(local_input, input_fragment);
```

## Constraints

- The launch grid must contain enough blocks to cover the requested tensor.
- Different blocks must use disjoint output ranges unless the algorithm requires
  an atomic memory operation.
- Do not use the block index as a synchronization mechanism.

## Common mistakes

- Applying a byte offset where the tensor view expects an element or row offset.
- Forgetting the second partition performed by `get_thread_idx()`.
- Assuming blocks execute in increasing index order.

See [mapping blocks and threads](../model/group-execution.md) for the logical and
physical hierarchy.

## Used by kernels

- GEMM and FlashAttention kernels use the block index to select an output tile.
- Model kernels use it to partition token, expert, or channel dimensions.
