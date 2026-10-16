# Element-wise atomic Top-K histogram

This test covers the generic CFG lowering path requested for TLEA. Input values
are carried as `uint32_t` but are restricted to the unsigned 16-bit key domain.
The representation avoids an unrelated narrow-integer conversion dependency
while keeping the two-level radix algorithm identical to a `uint16_t` Top-K.

The complete `topk16` kernel is in
[`element_atomic_topk.hpp`](../../../../../kernels/basic_op/topk/element_atomic_topk.hpp).
It contains histogram construction, cutoff selection and collection in one
function. It includes only the public `<common/pto_tileop.hpp>` API and uses
`ElementTile<uint32_t, Elements>`, `TPARTVIEW`, `TPARTELEMENT`, ordinary element
indices and relaxed atomic fetch-add. The API owns physical storage, safe
partial loads and logical-order stores; the benchmark defines no private Tile
classes or copied arithmetic backend. The requested public extension is kept
in existing TileOp API headers and requires `PTO_TILEOP_API_HAS_ELEMENT_TILE`.

Tile loads and views prepare a bounded input snapshot. Ordinary scalar shifts,
comparisons and atomic updates run inside each marked element loop, then Tile
stores return old values in logical input element order. The coherence probe in
the runnable driver follows the same model; the source gate checks both files. The compiler
converts each logical U32 histogram index stream through one B32-to-B64 TLEA
before masked MGATHER.ADD. Top-K results are an unordered multiset; the
independent verification stage sorts them for comparison only. Optimized IR
must contain the three semantic atomic sites and no residual region/view,
`atomicrmw`, scalar extract/insert, or GM roundtrip.

The four independent workspace clears also use element loops, with one write
stream per region. The cutoff scans and compact output collection retain normal
sequential C++ loops because their counters depend on preceding elements. The
core header therefore has six marked loops; its driver adds the coherence loop.

The target test validates all 256 bins at both radix levels, exactly 37 output
elements, the complete Top-K multiset across a tied cutoff, and the returned
old value from every atomic. The old values for each bin must be exactly the
permutation `0..count-1`; unused and padded tail elements must remain zero. The
777-element input crosses six full 128-element parent Tiles
and ends in a partial parent Tile whose inactive storage
contains values outside the documented key domain.

The same ELF also runs a dedicated coherence probe. Four additional exported
golden segments contain its input, final histogram, old-value array, and scalar
observations `[initial, after_atomic, after_scalar, failures] = [0, 9, 10, 0]`. A volatile scalar load first caches a zero histogram
line, a volatile scalar store makes bin zero dirty with value 7, and a
two-element Tile/element-wise region performs two native masked atomic adds.
Their old-value Tile stores must contain the relaxed-atomic permutation
`[7, 8]` or `[8, 7]`, every inactive old-value element must remain zero, and the histogram
must become 9. A later volatile scalar store/load pair must observe 10, proving
that the later scalar writer supersedes the queued tile commit. Any mismatch
sets status failure bit `0x20`.

The portable reference test also covers `K=0`, `K=1`, `K=N`, duplicate hot
bins, cutoff ties, an all-equal input, and partial-tile tails:

```sh
benchmark/one-level-arch/test/kernel/sort/element_atomic_topk/check_reference.sh
```

`generate_case.py --out <dir>` independently emits little-endian U32 input,
both histogram levels, sorted Top-K output, status, the four coherence probe
goldens, and a manifest mapping all nine segments to stable ELF symbols. Atomic old-value arrays remain runtime-order
dependent and are verified inside the ELF as exact per-bin permutations.

The end-to-end harness combines a freshly built compiler/backend with an
installed Linx musl runtime and current TileOP headers. It rejects an ELF
whose three static atomic sites lack B.SUBVIEW, Tile digit operations,
TCI, GPR predicate production, exactly one TLEA, masked MGATHER.ADD and Tile
stores. It also checks optimized generic IR and rejects narrow index fallback
or residual scalar element lowering. Full histograms use a tail predicate; the
selected histogram combines tail and digit predicates through generic CFG mask
construction. Negative disassembly and IR canaries prove these checks reject
corrupted instructions. Current acceptance runs gfrun as one logical thread/PE
and compares the independently generated input, histogram, Top-K, status and
coherence memory segments. Queue capacities and the 120-second timeout retain
their defaults. Earlier dual-model evidence remains historical; gfsim changes
and qualification belong to a separate PR. UART text is diagnostic only:

```sh
MODELS=gfrun \
COMPILER_DIR=/path/to/fresh-llvm-build/bin \
LINX_RUNTIME_ROOT=/path/to/linx_blockisa_llvm_musl \
API_INCLUDE=/path/to/Linx-TileOP-API/include \
SSM=/path/to/SuperScalarModel \
benchmark/one-level-arch/verification/run_element_atomic_topk.sh
```

`TARGET_TRIPLE`, `SYSROOT`, and `RESOURCE_DIR` can be overridden. Their
defaults are respectively `linx64v5-unknown-linux-musl`,
`$LINX_RUNTIME_ROOT/sysroot`, and
`$LINX_RUNTIME_ROOT/lib/clang/15.0.4`.

Each run writes a unique evidence directory with a `content_id` under
`output/verification/element_atomic_topk/`. It contains the frozen ELF, disassembly,
symbol table, compiler/model logs, both memory dumps, independent goldens, and
`provenance.json` with commands, configs, tool and ELF hashes, repository
revisions/diff hashes, runtime-library hashes, and the TileOP header hash.

This directory's `compile.all` builds the canonical element-for example using
`LINX_RUNTIME_ROOT` and `API_INCLUDE`. The distinct upstream Tile-op TopK lives
under `kernels/multi_thread/topk/`; it is not counted as element-for migrated.

The additional native [Tile/element-wise suite](../../element_wise/tile_element_suite/README.md)
covers standalone histogram, selected radix histogram, and actual Top-K boundary
calls with independent goldens on gfrun and gfsim. Host reference tests are an
independent oracle, not an alternative TileOp implementation.

应用中的 element array 下标表示逻辑元素。`#pragma pto element for` 要求编译器分析紧随其后的 for 循环；不能忽略 pragma 并退化成普通标量代码。


## Alternating Tile operations and element loops

A Tile value remains usable by subsequent Tile operations after an element
loop updates it. The public views describe logical elements; the compiler owns
the SSA versions, masks and Local storage. The programmer keeps normal C++
statement order and does not insert barriers or expose physical layouts.

QLI's `RadixPopN` is the repeated alternation gate: it calls `TROWARGMAX`, stores
the selected logical index, clears that lane through `TPARTELEMENT` in a marked
loop, then calls `TROWARGMAX` again on the updated Tile. A real outlined Tile
helper returns the updated carrier to its caller. The supported helper boundary
is an internal direct void helper with one mutable NORM row; arbitrary nested
or aliased Tile helpers are not qualified by this example.

QLI's static TopK returns an unordered set of unique indices. Its transformed
U32 score keys define the raw-bit ordering, including signed zeros and special
FP32 encodings. The GT stage selects every strictly larger key; the EQ stage
fills any required tied subset. Candidate Tiles carry nonzero logical lane tags,
so a valid key zero is distinct from an inactive or cleared candidate. Dynamic
QLI also predicates candidates by the actual tail width. Inputs and the outer
TopK output/workspace allocations are disjoint; the inner key/histogram helpers
separately provide their documented complete-chunk alias snapshots.

## Failed runs and external build directories

Each fresh run directory records `provenance.json` even when setup, compilation,
model execution or comparison fails. Missing artifacts have null hashes and a
`provenance_errors` entry; the original failure status is retained. An incomplete
record cannot turn a successful exit into a qualified result. Reusing an existing
artifact directory is rejected.

The primary runner copies `RESOURCE_DIR` into its own `clang-resource` directory
and installs the selected `API_INCLUDE` there before compilation. This ensures
`-mlxbc` uses the requested API and leaves the caller's resource tree intact.
Runtime archives are fingerprinted from the actual `SYSROOT` override.

`COMPILER_DIR` and `SSM` may be external build directories. Set
`COMPILER_SOURCE_ROOT`, `MODEL_SOURCE_ROOT` and optionally `API_SOURCE_ROOT` to
record their source checkouts. Without an available checkout, binary hashes are
still recorded and repository identity is explicitly marked unavailable.
