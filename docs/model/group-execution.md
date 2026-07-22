# Grouped Execution

Grouped operations are dynamic operation instances reached by all required
participating threads. They combine per-thread local operands with shared tile
operands and define per-thread or shared results.

## Entry and Divergence

All threads start at the same program entry. A branch may select paths by
thread identity:

```cpp
const uint32_t thread = get_thread_idx();

if (thread == 0) {
  load_metadata();
} else {
  compute_payload(thread);
}
```

Different paths may define different local tile versions. Before a grouped
operation, the paths must reconverge with compatible versions for every source
the operation reads.

## Dynamic Operation Order

Grouped operations are matched by dynamic order, not by physical arrival time.
Every participant must reach the same grouped operation instance in the same
relative order.

```cpp
TMATMUL(c0, a0, shared_b0);  // grouped instance 0
TMATMUL(c1, a1, shared_b1);  // grouped instance 1
```

A program is invalid if one participant reaches instance 1 while another
participant is still expecting instance 0.

## Shared Operand Readiness

A grouped operation that reads a shared operand waits for the bound shared
version. The producing thread does not need to arrive in the same cycle as the
consuming threads. The data dependency is the shared tile version.

```cpp
if (thread == 0) {
  TLOAD(shared_b, global_b);
}

TMATMUL(local_c, local_a, shared_b);
```

The shared right operand must be fully defined before `TMATMUL` can consume it.

## Local Result Ownership

Grouped matrix operations define local result fragments. Each thread owns its
own result fragment and stores only its assigned global range:

```cpp
TSTORE(global_c.slice_rows(rows.begin, rows.count), local_c);
```

Avoid overlapping stores unless the selected operation and runtime explicitly
define the conflict behavior.

## Partial Groups

Some movement patterns intentionally publish only a subset of shared regions:

```cpp
if (thread < kActiveProducerCount) {
  TMOV<SharedMove::Insert>(shared_partial, local_value);
}
```

The resulting shared version may be consumed only by operations that accept the
defined-region mask. It cannot be used as a complete shared right matrix.

## Common Mistakes

- Letting one participant skip a required grouped operation.
- Reaching grouped operations in different dynamic orders.
- Treating shared tile readiness as a cycle-timing property.
- Reading a shared region that no participating thread defined.
- Storing thread fragments to overlapping global ranges.
