# TFREE: Release a pipe resource

## Purpose / When to use

Use `TFREE` when a pipe or FIFO resource bound by `TALLOC` is no longer needed.

## C++ Declaration

```cpp
template <typename Pipe>
PTO_INST void TFREE(Pipe &pipe);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `pipe` | Pipe object to release. |

## Operation / Semantics

- Releases the pipe association created by `TALLOC`.
- Returns the currently held slot or resource to the producer/consumer protocol.
- Does not modify tile contents directly.

## Constraints

- Only release pipes that have completed their producer/consumer protocol.
- Do not use after the pipe has already been released.
- Shared tile register lifetime is not managed by `TFREE`.
- Outstanding consumers or producers must be resolved by the kernel protocol.

## Example

```cpp
TFREE(pipe);
```

## Common mistakes

- Releasing a pipe before the last consumer has popped the payload.
- Calling `TFREE` twice for one allocation.
- Expecting `TFREE` to clear global memory or tile data.

## Used by kernels

- No direct repository kernel use was found; retained for pipe-lifetime management.
