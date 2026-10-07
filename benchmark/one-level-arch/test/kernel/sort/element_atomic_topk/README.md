# Element-wise atomic Top-K histogram

This test covers the standard-C lowering path requested for TLEA. Input values
are carried as `uint32_t` but are restricted to the unsigned 16-bit key domain.
The representation avoids an unrelated narrow-integer conversion dependency
while keeping the two-level radix algorithm identical to a `uint16_t` Top-K.

The complete `topk16` kernel is in
[`element_atomic_topk.hpp`](../../../../../kernels/single_thread/sort/element_atomic_topk.hpp).
It contains histogram construction, cutoff selection and collection in one
function. It includes only the public `<common/pto_tileop.hpp>` API and uses
`ElementTile<uint32_t, Elements>`, `TPARTVIEW`, `TPARTELEMENT`, ordinary element
indices and relaxed atomic fetch-add. The API owns physical storage, safe
partial loads and logical-order stores; the benchmark defines no private Tile
classes or copied arithmetic backend. The requested public extension is kept
in existing TileOp API headers and requires `PTO_TILEOP_API_HAS_ELEMENT_TILE`.

TileOps prepare radix digits. The element-wise loop performs histogram updates,
then TileOp stores return old values in logical input element order. The compiler
converts logical U32 histogram indices through exactly one TLEA to U64 byte
offsets before masked MGATHER.ADD. Top-K results are an unordered multiset; the
independent verification stage sorts them for comparison only.

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
Their old-value Tile stores must contain 7 at logical element 0 and 8 at
logical element 1, every inactive old-value element must remain zero, and the histogram
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
stores. Full histograms use a tail predicate; the selected histogram combines
tail and digit predicates with scalar AND. Four negative disassembly canaries
prove these checks can reject corrupted instructions. Both gfrun and gfsim must then run as one logical
thread/PE and match the independently generated input, histogram, Top-K, and
status and coherence memory segments. UART text is diagnostic only and is not used as the
correctness oracle:

```sh
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

The general sort `compile.all` and repository compile smoke keep their legacy
behavior. Set `ELEMENT_ATOMIC_TOPK=on` together with `LINX_RUNTIME_ROOT` and
`API_INCLUDE` to opt this draft PTO 0.59 case into those broader scripts.

The additional native [Tile/element-wise suite](../../element_wise/tile_element_suite/README.md)
covers standalone histogram, selected radix histogram, and actual Top-K boundary
calls with independent goldens on gfrun and gfsim. Host reference tests are an
independent oracle, not an alternative TileOp implementation.

应用中的 element array 下标表示逻辑元素。`#pragma pto element for` 要求编译器分析紧随其后的 for 循环；不能忽略 pragma 并退化成普通标量代码。
