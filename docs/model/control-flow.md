# Divergent and Convergent Control

C++ control flow may diverge by thread identity, scalar data, or template
selection. Tile correctness depends on which tile versions are defined on the
paths that reach each consumer.

## Compile-Time Selection

Use templates and `if constexpr` when the choice is part of the kernel
specialization:

```cpp
template <bool kUseBias>
void matmul_path(...) {
  if constexpr (kUseBias) {
    TMATMUL_BIAS(out, lhs, rhs, bias);
  } else {
    TMATMUL(out, lhs, rhs);
  }
}
```

Only the selected branch participates in code generation for that
specialization.

## Thread-Identity Branches

Use `get_thread_idx()` when one participant must produce or consume a shared
tile value:

```cpp
const uint32_t thread = get_thread_idx();

if (thread == 0) {
  TLOAD(shared_rhs, rhs_view);
}

TMATMUL(out_fragment, lhs_fragment, shared_rhs);
```

The grouped matrix operation is convergent: every required participant reaches
the same dynamic instance and binds the same shared right version.

## Data-Dependent Branches

A runtime scalar branch may create values on only some paths:

```cpp
if (needs_scale) {
  TMULS(scaled, input, alpha);
  TSTORE(output, scaled);
} else {
  TSTORE(output, input);
}
```

This is valid because each store consumes a value defined on its own path. It
would be invalid to consume `scaled` after the branch unless every reaching
path defines it.

## Reconvergence Rule

At a merge point, every tile consumer must have a defined source version for
each path that reaches it. For grouped operations, all required participants
must also reach the same dynamic instance in the same relative order.

## No Timing-Based Control

Do not use operation latency or arrival order as the control mechanism. Use C++
values and shared tile versions to express dependencies.
