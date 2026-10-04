# Tile/element-wise benchmark suite

These executable examples show the intended source-level programming model:
load and transform a logical Tile, use ordinary C++ element expressions for
data-dependent work, then continue with TileOps. Application code includes only
the public `<common/pto_tileop.hpp>` API and names logical element counts.
Physical layouts and execution lanes stay inside the TileOp API and compiler.

## Benchmarks

[`histogram_tile_element.cpp`](histogram_tile_element.cpp) computes a 256-bin
histogram over 263 elements. `TLOAD`, `TPARTVIEW`, `TADDS`, and `TANDS` prepare
the data. Its element-wise loop uses a normal tail condition and relaxed atomic
fetch-add. `TADDS` then converts each returned old value from `[0,n)` to
`[1,n+1)` before `TSTORE`.

[`selected_radix_tile_element.cpp`](selected_radix_tile_element.cpp) extracts
high and low radix digits with `TSHRS` and `TANDS`. Its element-wise `if` updates
the low-digit histogram only when the high digit matches `selected_high`.
Afterward, `TADDS` adds seven to the atomic return values and `TSTORE` writes
the diagnostic Tile.

Both diagnostic output arrays preserve logical input element order, so
`output[element]` always describes `input[element]`; application indexing does
not expose a physical partition mapping.

[`topk_boundaries.cpp`](topk_boundaries.cpp) invokes the actual `topk16`
kernel across empty input, K=0/1/N/>N, counts around 32- and 128-element
boundaries, cutoff ties, and all-equal keys. It uses fixed native arrays and
independently checks histograms, Top-K multisets, output guards, and per-bin
atomic old-value permutations after every runtime-selected case. The harness
splits the runtime case table across three bounded regression ELFs; every case
runs, and every ELF contains the same two static atomic sites from the actual
kernel. It also requires the unsharded 17-call ELF to pass both models, which
reuses one workspace across all calls and keeps repeated caller behavior plus
block-ID wrap covered end to end.

The two standalone histogram inputs deliberately contain repeated hot bins,
bins 0 and 255, two full 128-element blocks, and a seven-element tail. Every
old-value checker accepts all legal atomic orders by checking each bin's
results as a permutation.

## End-to-end verification

Run from the `benchmark/one-level-arch` directory with fresh tool paths:

```bash
COMPILER_DIR=/path/to/fresh/compiler/bin \
SSM=/path/to/SuperScalarModel \
LINX_RUNTIME_ROOT=/path/to/linx/runtime \
API_INCLUDE=/path/to/Linx-TileOP-API/include \
RESOURCE_DIR=/path/to/task-owned/clang-resource \
bash verification/run_tile_element_suite.sh
```

The script first installs the selected public API branch into the isolated
resource directory with its official `make install` target. It then builds six
ELFs: two standalone kernels, one required unsharded 17-call Top-K boundary
benchmark, and three focused shards of that boundary table. The standalone
ELFs each contain one TLEA and one masked `MGATHER.ADD` static site; every Top-K
ELF contains exactly two of each.

Every ELF runs unchanged on `gfrun` and `gfsim`. Both memory dumps are checked
against separately generated input, histogram, output, and status goldens plus
the order-independent atomic-return contract. A gfsim result is accepted only
when its final invariant line reports `stq_conservation=OK` and
`a3_tile_violation=0`, with no assertion or `LOG_ERROR` in the log. The artifact
directory records the commands, repository heads, installed API tree, tool
hashes, ELFs, logs, dumps, and golden hashes.
