# get_thread_idx() - Read the thread index within a block

## Purpose and when to use

`get_thread_idx()` returns the immutable logical thread index for the current
kernel invocation. Use it to choose a disjoint tensor fragment or a thread-specific
control path.

## C++ declaration

```cpp
PTO_INST uint32_t get_thread_idx();
```

## Parameters and result

| Value | Meaning |
| --- | --- |
| Return value | An index in `[0, threads_per_block)`. |

## Execution semantics

Every thread in a block begins at the same kernel entry. Each thread has its own
program counter after entry, so branches on the returned value may diverge.
The value remains constant for the lifetime of the invocation and does not grant
access to another thread's local tiles.

## Example

```cpp
constexpr uint32_t kThreadsPerBlock = 4;
constexpr uint32_t kBytesPerFragment = 128;

const uint32_t thread = get_thread_idx();
const uint32_t byte_offset = thread * kBytesPerFragment;
auto fragment = output.byte_slice(byte_offset, kBytesPerFragment);
TSTORE(fragment, local_result);
```

Keep the active group size in launch/profile code rather than scattering a
hardware cardinality through kernel algorithms.

## Constraints

- Compare the result only with indices valid for the active block profile.
- Threads must write disjoint global ranges unless an operation is atomic.
- A branch that performs a group operation must satisfy that operation's
  participation rules.

## Common mistakes

- Treating the index as a global block identifier.
- Using the index to address another thread's local tile register.
- Letting two threads store the same output fragment accidentally.

See [block and thread indexing](../model/core-indexing.md) for multidimensional
partitioning patterns.

## Used by kernels

- Grouped matrix multiplication uses the thread index to select a disjoint
  output fragment.
- Reduction and exchange examples use it to select a producer or consumer role.
