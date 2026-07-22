# TALLOC

## Introduction

Associate a pipeline object with a global-tensor resource for the lifetime
managed by the corresponding `TFREE` operation.

## C++ Intrinsic

```cpp
template <typename Pipe, typename GlobalData>
PTO_INST void TALLOC(Pipe &pipe, GlobalData &global_tensor);
```

## Constraints

- `pipe` and `global_tensor` must describe a compatible transfer resource.
- `TALLOC` does not allocate a versioned shared tile register. Shared tile
  versions are created by definitions such as shared-destination `TLOAD` or
  local-to-shared `TMOV`.
- The 0.57 workbook marks CPU simulation as unsupported and A2A3/A5 as
  supported.

## Examples

```cpp
TALLOC(pipe, tensor);
// Issue the pipeline operations that use tensor.
TFREE(pipe);
```

## Assembly Status

The 0.57 workbook defines the PTO interface. The selected backend owns the
resource-management lowering; no portable standalone block template is
specified here.
