# Group Programming Examples

These examples define the architecture-level four-PE source model. Shared tile
register lowering is not required from the current compiler yet.

## PE-Specific Control Paths

All four PEs start at the same entry, then branch using their immutable PE ID.

```cpp
void group_kernel(GlobalTensor input, GlobalTensor output) {
  const uint32_t pe = get_thread_idx();

  if (pe == 0) {
    metadata_path(input);
  } else if (pe == 1) {
    transform_path(input);
  } else {
    compute_path(input, output, pe);
  }
}
```

Each branch may define different PE-local tile versions. The paths must
reconverge with compatible state before a later four-PE group operation.

## Tensor Split by Core and PE

```cpp
const uint32_t core = get_block_idx();
const uint32_t pe = get_thread_idx();

const uint32_t core_begin = rows * core / core_count;
const uint32_t core_end = rows * (core + 1) / core_count;
const uint32_t core_rows = core_end - core_begin;

const uint32_t pe_begin = core_begin + core_rows * pe / 4;
const uint32_t pe_end = core_begin + core_rows * (pe + 1) / 4;
auto input_rows = input.slice_rows(pe_begin, pe_end - pe_begin);
TLOAD(local_input, input_rows);
```

The quotient split handles tails without overlapping core or PE slices.

## Four-PE Grouped Matrix Multiply

The B tile is loaded once into a shared tile version. A and C are PE-local
M-axis quarters.

```cpp
template <int M, int N, int K>
void grouped_matmul(GlobalTensor a, GlobalTensor b, GlobalTensor c) {
  const uint32_t pe = get_thread_idx();

  PETile<half, M / 4, K> local_a;
  PETile<float, M / 4, N> local_c;
  SharedTile<half, K, N> shared_b;

  TLOAD(local_a, a.slice_rows(pe * (M / 4), M / 4));

  if (pe == 0) {
    TLOAD(shared_b, b);
  }

  TMATMUL(local_c, local_a, shared_b);
  TSTORE(c.slice_rows(pe * (M / 4), M / 4), local_c);
}
```

All four PEs reach the same dynamic `TMATMUL`. Readiness of `shared_b` is a
tile-version dependency, so PE arrival does not need to be cycle aligned.

## Four-PE Reduce-to-One

Each PE reduces its local input, inserts the result into its fixed shared
region, and PE0 performs the final reduction.

```cpp
PETile<float, 1, LocalWidth> local_input;
PETile<float, 1, 1> local_partial;
SharedTile<float, 4, 1> shared_partials;

TROWSUM(local_partial, local_input);
TMOV<SharedMove::Insert>(shared_partials, local_partial);

if (get_thread_idx() == 0) {
  PETile<float, 4, 1> all_partials;
  PETile<float, 1, 1> total;
  TMOV<SharedMove::Broadcast>(all_partials, shared_partials);
  TCOLSUM(total, all_partials);
  TSTORE(global_total, total);
}
```

The shared definition publishes only after all four inserted regions are
ready. PE0 therefore reads one coherent version.

## Four-PE All-Reduce

Extend reduce-to-one by publishing the final scalar and broadcasting it back
to every PE.

```cpp
SharedTile<float, 1, 1> shared_total;

if (get_thread_idx() == 0) {
  TMOV<SharedMove::Publish>(shared_total, total);
}

PETile<float, 1, 1> local_total;
TMOV<SharedMove::Broadcast>(local_total, shared_total);
```

Every PE receives the same shared version.

## Neighbor Exchange

PE-local physical registers are never addressed directly. Publish all four
quarters, broadcast the complete shared value locally, then extract the desired
neighbor quarter.

```cpp
const uint32_t pe = get_thread_idx();
const uint32_t peer = (pe + 1) & 3;

SharedTile<float, 4, Width> shared_rows;
TMOV<SharedMove::Insert>(shared_rows, local_row);

PETile<float, 4, Width> all_rows;
PETile<float, 1, Width> peer_row;
TMOV<SharedMove::Broadcast>(all_rows, shared_rows);
TEXTRACT(peer_row, all_rows, peer, 0);
```

This pattern supports ring shifts, halo exchange, and small permutations.

## Partial Producer Set

```cpp
if (get_thread_idx() < 2) {
  TMOV<SharedMove::Insert>(shared_partial, local_value);
}
```

The resulting shared version has only PE0 and PE1 regions defined. It may be
stored by partition or extracted by those owners. It cannot be used as a
matrix RHS.
