# Block and Thread Mapping

This is the only page that maps the generic C++ programming terms in this
manual to the current implementation profile.

## Generic Source Terms

| Generic C++ term | Intrinsic | Meaning |
| --- | --- | --- |
| Block | `get_block_idx()` | Work-unit ID used to select a global tensor slice |
| Thread | `get_thread_idx()` | Lane ID inside the block used to select a fragment |

Kernel prose should use **block** and **thread**. The implementation currently
maps a block to one NPU core and maps each thread to one processing element.
The current profile has exactly four threads per block:

```cpp
static constexpr uint32_t kThreadsPerBlock = 4;
static constexpr uint32_t kThreadIndexMax = kThreadsPerBlock - 1;
static constexpr uint32_t kMinimumThreadFragmentBytes = 128;
```

Use these constants in examples outside this page instead of restating the
profile.

## Stable Identity

`get_block_idx()` returns the block identity for the current kernel instance.
`get_thread_idx()` returns the thread identity inside that block. Both values
are immutable for the lifetime of the kernel instance.

All threads in the block start at the same program entry and initial program
counter. C++ branches may then select different paths by thread identity. A
later grouped operation is valid only when every required participant reaches
the same dynamic operation instance with compatible tile versions.

## Partition Global Rows

Partition in two steps: first by block, then by thread.

```cpp
struct RowSlice {
  uint32_t begin;
  uint32_t count;
};

RowSlice partition_rows(uint32_t rows,
                        uint32_t block,
                        uint32_t block_count,
                        uint32_t thread) {
  const uint32_t block_begin = rows * block / block_count;
  const uint32_t block_end = rows * (block + 1) / block_count;
  const uint32_t block_rows = block_end - block_begin;

  const uint32_t thread_begin =
      block_begin + block_rows * thread / kThreadsPerBlock;
  const uint32_t thread_end =
      block_begin + block_rows * (thread + 1) / kThreadsPerBlock;

  return {thread_begin, thread_end - thread_begin};
}
```

This quotient split handles tails without overlap. It also keeps every global
element owned by exactly one block/thread pair when `block_count` covers the
whole tensor.

## Minimum Fragment Size

A per-thread tile fragment must represent at least
`kMinimumThreadFragmentBytes` of payload for the current profile. Smaller
logical work should be padded, packed with multiple elements, or handled by a
scalar path before forming a tile operation.

The minimum applies to the thread-local fragment. A block-wide logical tile
therefore has at least:

```cpp
static constexpr uint32_t kMinimumBlockTileBytes =
    kThreadsPerBlock * kMinimumThreadFragmentBytes;
```

Padding does not make padded elements valid data. Use valid-region metadata or
explicit tail logic so stores write only the intended output range.
