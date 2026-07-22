# First tile program

This kernel loads two 16 × 16 global tensor views, adds their tiles, and stores
the result. It is intentionally small enough to expose the complete programming
shape without hiding the PTO calls behind a framework.

```cpp title="docs/examples/tile_add.cpp" linenums="1" hl_lines="10-12 14-16 18-20"
--8<-- "docs/examples/tile_add.cpp"
```

## Read the types first

`global_tensor<__fp32, RowMajor<Rows, Cols>>` describes a global-memory view.
`Tile<Location::Vec, ...>` describes tile-local vector storage. `Rows`, `Cols`,
element type, and layout are compile-time properties.

`__fp32` is the current direct-v0.57 PTO storage wrapper for a 32-bit floating
element. Unlike the C++ `float` type, it carries the tile descriptor's
compile-time `TypeCode` in the checked-in `jcore/type.hpp` specialization.

<dl class="snpu-contract">
  <dt>G</dt><dd>Global tensor view with row-major element addressing.</dd>
  <dt>T</dt><dd>Vector tile with a static 16 × 16 shape in this instantiation.</dd>
  <dt>TLOAD</dt><dd>Move each global view into its tile representation.</dd>
  <dt>TADD</dt><dd>Produce a new tile from two shape-compatible input tiles.</dd>
  <dt>TSTORE</dt><dd>Write the result tile through the output global view.</dd>
</dl>

## Compile it

```bash
cd "$SUPERNPU_ROOT"
bash docs/examples/check.sh tile-add
```

The gate invokes Linx `clang++` with C++20, `-fenable-matrix`, the PTO include
root, the active target triple, and `__linx`/tensor-instruction defines.

## Why the template is explicit

The explicit instantiation at the bottom forces LLVM to compile one concrete
kernel even though this example has no host `main`. Production benchmarks use
the same specialization mechanism at larger shapes and connect it to test
entrypoints and generated data.

Next, [inspect the object](build-run.md) or move directly to the
[GEMM tutorial](../tutorials/gemm.md).
