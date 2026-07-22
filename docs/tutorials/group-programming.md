# Group Programming Patterns

Grouped kernels use block/thread identity, shared tile versions, and
convergent grouped operations. These examples show the source contract. Check
the benchmark catalog for the source forms that compile in the current tree.

## Thread-Specific Control Paths

All threads start at the same entry, then branch using immutable thread
identity.

```cpp
void grouped_kernel(GlobalTensor input, GlobalTensor output) {
  const uint32_t thread = get_thread_idx();

  if (thread == 0) {
    metadata_path(input);
  } else if (thread == 1) {
    transform_path(input);
  } else {
    compute_path(input, output, thread);
  }
}
```

Each branch may define different local tile versions. Paths must reconverge
with compatible state before a later grouped operation.

## Tensor Split by Block and Thread

```cpp
const uint32_t block = get_block_idx();
const uint32_t thread = get_thread_idx();

RowSlice rows = partition_rows(total_rows, block, block_count, thread);
auto input_rows = input.slice_rows(rows.begin, rows.count);

TLOAD(local_input, input_rows);
```

The quotient split handles tails without overlapping global slices.

## Multidimensional Group Partition

Flatten independent batch/head planes, distribute them across the execution
grid, and tile each plane locally:

```cpp
const uint32_t worker =
    get_block_idx() * kThreadsPerBlock + get_thread_idx();
const uint32_t worker_count = block_count * kThreadsPerBlock;

for (uint32_t plane = worker; plane < kBatch * kHeads;
     plane += worker_count) {
  const uint32_t batch = plane / kHeads;
  const uint32_t head = plane % kHeads;

  for (int tile_row = 0; tile_row < kRowTiles; ++tile_row) {
    for (int tile_col = 0; tile_col < kColTiles; ++tile_col) {
      TLOAD(local_input, input(batch, head, tile_row, tile_col));
      TRELU(local_output, local_input);
      TSTORE(output(batch, head, tile_row, tile_col), local_output);
    }
  }
}
```

Each worker owns complete output planes in this example, so no shared value is
needed. Use shared tiles only when several threads collaborate on one plane.

## Grouped Matrix Multiply

Load the right matrix once into a shared tile version. Keep the left matrix
and result as local row fragments.

```cpp
template <uint32_t kM, uint32_t kN, uint32_t kK>
void grouped_matmul(GlobalTensor a, GlobalTensor b, GlobalTensor c) {
  const uint32_t thread = get_thread_idx();

  static constexpr uint32_t kRowsPerThread = kM / kThreadsPerBlock;

  LocalTile<half, kRowsPerThread, kK> local_a;
  LocalTile<float, kRowsPerThread, kN> local_c;
  SharedTile<half, kK, kN> shared_b;

  TLOAD(local_a, a.slice_rows(thread * kRowsPerThread, kRowsPerThread));

  if (thread == 0) {
    TLOAD(shared_b, b);
  }

  TMATMUL(local_c, local_a, shared_b);
  TSTORE(c.slice_rows(thread * kRowsPerThread, kRowsPerThread), local_c);
}
```

Every participant reaches the same dynamic `TMATMUL`. The shared right operand
is a tile-version dependency, so participants do not need cycle-aligned
arrival.

## Reduce to One

Each thread reduces its local input, inserts the scalar into its fixed shared
region, and one selected thread performs the final reduction.

```cpp
LocalTile<float, 1, kLocalWidth> local_input;
LocalTile<float, 1, 1> local_partial;
SharedTile<float, kThreadsPerBlock, 1> shared_partials;

TROWSUM(local_partial, local_input);
TMOV<SharedMove::Insert>(shared_partials, local_partial);

if (thread == 0) {
  LocalTile<float, kThreadsPerBlock, 1> all_partials;
  LocalTile<float, 1, 1> total;
  TMOV<SharedMove::Broadcast>(all_partials, shared_partials);
  TCOLSUM(total, all_partials);
  TSTORE(global_total, total);
}
```

The shared definition publishes only after all defined regions are ready. The
selected thread reads one coherent shared version.

## All-Reduce

Extend reduce-to-one by publishing the final scalar and broadcasting it back to
every thread.

```cpp
SharedTile<float, 1, 1> shared_total;

if (thread == 0) {
  TMOV<SharedMove::Publish>(shared_total, total);
}

LocalTile<float, 1, 1> local_total;
TMOV<SharedMove::Broadcast>(local_total, shared_total);
```

Every participant receives the same shared version.

## Neighbor Exchange

Local physical registers are never addressed directly. Publish fragments,
broadcast the complete shared value locally, then extract the desired neighbor
fragment.

```cpp
const uint32_t thread = get_thread_idx();
const uint32_t peer = (thread + 1) % kThreadsPerBlock;

SharedTile<float, kThreadsPerBlock, kWidth> shared_rows;
TMOV<SharedMove::Insert>(shared_rows, local_row);

LocalTile<float, kThreadsPerBlock, kWidth> all_rows;
LocalTile<float, 1, kWidth> peer_row;
TMOV<SharedMove::Broadcast>(all_rows, shared_rows);
TEXTRACT(peer_row, all_rows, peer, 0);
```

This pattern supports ring shifts, halo exchange, and small permutations.

## Partial Producer Set

```cpp
if (thread < kActiveProducerCount) {
  TMOV<SharedMove::Insert>(shared_partial, local_value);
}
```

The resulting shared version has only the selected regions defined. It may be
stored by partition or extracted by defined owners. It cannot be used as a
complete shared right matrix.
