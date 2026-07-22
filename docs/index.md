---
hide:
  - toc
---

# Superscalar NPU C++ Programming Manual

Write C++ kernels for a superscalar NPU using typed global tensors, local
tiles, shared tiles, and explicit tile operations. The source model is a
normal C++ program: use block and thread identity to select tensor slices,
define tile values, and let tile-version dependencies order the work.

## Reading Path

1. [Orientation](start/overview.md) explains the C++ programming stack.
2. [Block and thread mapping](model/core-indexing.md) defines the only place
   where this guide maps generic C++ terms to implementation terms.
3. [Types and values](model/types-and-values.md) introduces global tensors,
   local tiles, shared tiles, and named constants.
4. [Multidimensional tiling](tutorials/multidimensional-tiling.md) shows how
   C++ loops walk 2D, 3D, and batched tensor regions.
5. [Tile execution](model/execution.md) explains tile IDs, versions, and
   superscalar issue.
6. [Shared tile operations](model/shared-tile-registers.md) covers shared
   `TLOAD`, `TSTORE`, `TMOV`, and shared-right `TMATMUL`.

## Kernel Shape

Every kernel starts from a block and a thread. The block selects a large,
non-overlapping tensor region. The thread selects a fragment inside that
region. Tile operations then move data through three spaces:

| Space | C++ role | Common operations |
| --- | --- | --- |
| Global tensor | Typed view of external memory | `TLOAD`, `TSTORE` |
| Local tile | Per-thread tile value | arithmetic, reductions, `TMATMUL` left and result |
| Shared tile | Block-local tile value visible to participating threads | shared `TLOAD`, shared `TSTORE`, `TMOV`, `TMATMUL` right |

Use named constants for hardware-shaped quantities:

```cpp
static constexpr uint32_t kThreadsPerBlock = /* profile value */;
static constexpr uint32_t kMinimumThreadFragmentBytes = /* profile value */;
```

The mapping page explains the current values. Other pages use the names so the
programming model stays stable when a profile changes.

## Minimal Dataflow

```cpp
const uint32_t block = get_block_idx();
const uint32_t thread = get_thread_idx();

auto rows = partition_rows(total_rows, block, block_count,
                           thread, kThreadsPerBlock);

LocalTile<float, kRowsPerThread, kCols> a;
LocalTile<float, kRowsPerThread, kCols> b;
LocalTile<float, kRowsPerThread, kCols> c;

TLOAD(a, input_a.slice_rows(rows.begin, rows.count));
TLOAD(b, input_b.slice_rows(rows.begin, rows.count));
TADD(c, a, b);
TSTORE(output.slice_rows(rows.begin, rows.count), c);
```

Each destination operation defines a new tile version. Consumers bind to the
version they use; correctness does not depend on instruction spacing, cycle
counts or separate completion tokens.

The public examples include only `pto_kernel/tile.hpp`. Backend-specific
namespaces stay behind that header so kernel code reads as tensor regions,
tiles, and operations.

## Current Compiler Boundary

The compile-checked repository examples and benchmark catalog are the active
source surface. Shared tile-register examples in the model pages describe the
NPU C++ contract that lowering must implement; treat them as contract examples
unless a benchmark page shows the exact source compiling in the current tree.
