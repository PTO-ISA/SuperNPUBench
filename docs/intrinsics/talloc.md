# TALLOC: Bind a pipe to global tensor storage

## Purpose / When to use

Use `TALLOC` before pipe or FIFO transfers that need a global tensor resource backing or describing the transfer lifetime.

## C++ Declaration

```cpp
template <typename Pipe, typename GlobalData>
PTO_INST void TALLOC(Pipe &pipe, GlobalData &global_tensor);
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `pipe` | Pipe object to bind. |
| `global_tensor` | Global tensor resource associated with the pipe. |

## Operation / Semantics

- Associates `pipe` with `global_tensor` for the lifetime managed by `TFREE`.
- Does not allocate or version a shared tile register.
- Prepares subsequent `TPUSH` and `TPOP` operations that use the pipe protocol.

## Constraints

- `pipe` and `global_tensor` must describe compatible transfer resources.
- Call `TFREE` when the pipe lifetime ends.
- Shared tile register allocation is separate from pipe allocation.
- CPU simulation support may be absent for this operation.

## Example

```cpp
TALLOC(pipe, tensor);
TPUSH(pipe, tile);
TFREE(pipe);
```

## Common mistakes

- Using `TALLOC` for shared tile registers.
- Forgetting to release the pipe with `TFREE`.
- Binding a pipe to a tensor with incompatible shape or dtype.

## Used by kernels

- No direct repository kernel use was found; retained for pipe-lifetime management.
