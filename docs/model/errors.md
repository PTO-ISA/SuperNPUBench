# Error Model

This page lists common invalid programs and the first correction to check.

## Grouped Operation Errors

- Fewer than the required participants reach a grouped operation.
- Participants reach grouped operations in different dynamic orders.
- A branch leaves a required participant behind a path that never reconverges.

First check the `get_thread_idx()` branches and confirm every participant
reaches the same dynamic operation sequence.

## Shared Tile Errors

- A program reads a shared region that was never defined.
- A program treats a shared tile ID as an address.
- A program relies on arrival time to select the latest shared version.
- A program mutates a published shared version in place.

First check the shared definition that should produce the consumed version.
Each new payload or defined-region mask requires a new shared version.

## Matrix Errors

- The right operand of `TMATMUL` is local rather than shared.
- The shared right operand is only partially defined.
- The left or result operand is not distributed as required by the operation.
- Accumulator and input element types do not match the selected intrinsic.

First check the tile types in the call signature and the intrinsic page for
the exact element, layout, and accumulator rules.

## Partition Errors

- A computed block or thread slice falls outside the tensor's valid region.
- Two participants store overlapping output ranges without a defined conflict
  rule.
- A tail path reads padding as input or stores padding as output.
- A per-thread fragment is smaller than `kMinimumThreadFragmentBytes`.

First check quotient partitioning, valid-region metadata, and fragment byte
size.

## Tile-ID Errors

- Source-level correctness depends on physical tile-register numbers.
- A source refers to a tile version outside the target's live producer window.
- A later write is expected to change a consumer that already bound an older
  version.

First check C++ value flow. The producer tile value should be passed directly
to the consumer that needs it.

## Dependency Errors

- Code introduces a dependency mechanism that bypasses tile versions.
- Code assumes a fixed operation latency.
- Code uses global memory as an implicit cross-block barrier.

First replace the ordering assumption with a tile-version dependency or move
the handoff into the runtime protocol.
