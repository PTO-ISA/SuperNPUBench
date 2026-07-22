# Supported C++ Patterns

The examples use a conservative subset of C++ that lowers predictably to tile
operations.

## Prefer

- C++20 templates for static shape and mode selection.
- `static constexpr` constants for tile dimensions.
- Plain structs for shape and slice metadata.
- Typed global tensor views instead of raw pointer arithmetic at call sites.
- Direct tile-operation calls with explicit destination values.
- `if constexpr` for compile-time feature selection.

## Use Carefully

- Runtime branches that define tile values on only some paths.
- Loops with tile accumulators, because they create loop-carried versions.
- Helper functions, because inlining and specialization must preserve precise
  tile types.

## Avoid

- Source assumptions about physical tile-register numbers.
- Timing assumptions between independent tile operations.
- Dependency objects that bypass tile values and tile versions.
- Casting shared tiles to pointers or global tensor views.
- Overlapping global stores without defined conflict behavior.

## Verification

Compile the smallest specialization first, then inspect disassembly for the
expected named operation blocks. Add the full benchmark harness only after the
source-level tile graph is clear.
