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

[`element_expression_chain.cpp`](element_expression_chain.cpp) is one complete
TileOp → element expression → TileOp kernel over 263 elements. It loads a
logical 128-element Tile, visits four logical 32-element partitions, and uses
three `#pragma pto element for` loops to cover `+`, `-`, `*`, `/`, `%`, `&`,
`|`, `^`, `<<`, `>>`, unary `-`, and unary `~`. Each loop writes one logical
result Tile, which is stored before the next element-expression region. The
source contains no physical layout type and no helper implementing the
operations. The bitwise region reuses an early shift result after four later
temporaries, so the compiler must preserve a non-linear SSA value while it
allocates the intermediate Tiles. This case now opts into the generic CFG
compiler: every ordinary LLVM operation becomes a masked Tile operation, and
all three typed publications return to the public TileOp dataflow.

三个表达式循环里的 `element` 是当前分区中的逻辑元素编号，不是硬件 lane。
局部临时变量由编译器转换为 SSA 并分配 intermediate Tile；程序员不需要声明
物理 Tile 布局。最后一个不足 32 个元素的分区由 `TLOAD` 补零，三个输出仍显式
写满 384 个位置，因此尾部语义也包含在独立 golden 中。

[`signed_element_expression.cpp`](signed_element_expression.cpp) runs a complete
S32 TileOp/element/TileOp kernel over 263 input elements with padded 384 output
and 16-element guards. Three marked loops cover signed arithmetic/division,
C++ signed remainder and right-shift/bitwise/negation. The independent oracle
locks `-7 % 3 == -1` and `7 % 3 == 1`; direct PTO TREM would produce the wrong
negative-input result. Source arithmetic stays within int32 limits. Compiler
IR must implement remainder as dividend minus truncating quotient times divisor;
the generic path uses signed `TDIV`, raw-bit `TMUL`/`TSUB`, then an explicit S32
typed publication. The checker proves that dataflow, all execution masks, and
exact S32 transport. LLVM's legal logical-right-shift fold under a low-bit mask
uses a U32 operation view while retaining S32 storage.
The default suite includes this case; focus it with
`--case signed_element_expression`.

[`generic_predicated_cfg_i32.cpp`](generic_predicated_cfg_i32.cpp) exercises
the compiler's GM-array generic element path without typed Tile views. Two
ordinary C++ kernels implement the same three-way S32 expression with different
CFG shapes: a nested branch and an inverted guard with `continue`. Active elements
read `a[element]`, `control[element]`, and the division-only
`divisor[element]`, then write `output[element]` in source order. Non-division
lanes carry a zero divisor, while a separate all-inactive call passes null GM
pointers. The observable outputs therefore prove masked signed division and
that an empty domain performs no GM load, division, or store. A third branch-only
kernel writes the signed division result directly, without a PHI merge, so its
output independently requires an S32 MSCATTER descriptor. This case is
explicitly the GM-array compiler path; the following benchmark separately
tests the typed-view bridge between a public `ElementTile` and generic regions.

[`generic_typed_tile_cfg_i32.cpp`](generic_typed_tile_cfg_i32.cpp) exercises
that typed bridge in one complete, readable S32 kernel. `TLOAD` loads a logical
128-element Tile and `TPARTVIEW<32>` presents logical 32-element partitions.
After `TADDS`, the first marked loop uses ordinary nested `if`, signed division,
and XOR. Its result flows through `TADDS` and `TMOV` into a second marked loop.
That loop conditionally replaces only some elements; all other elements retain
the nonzero value copied by `TMOV`. A final `TSTORE` publishes the same carrier.
The source exposes no physical layout, lane identifier, mask Tile, or compiler
temporary. Exact input, padded output, guards, checksums, and retained-element
counts come from an independent Python golden.

[`indexed_gather_tile_element.cpp`](indexed_gather_tile_element.cpp) demonstrates
an ordinary indexed U32 load rather than an atomic update. It loads all 384
indices, including `UINT32_MAX` poison in the padded tail, copies each logical
part with `TADDS`, and writes `table[index_elements[element]]` only when the
element is active. The compiler converts element indices to byte offsets with
`TLEA` and applies the tail as a GPR execution mask to `MGATHER`. A second
`TADDS` adds three to the gathered Tile before `TSTORE`; therefore inactive
elements have the observable padded value three. The source includes only
`<common/pto_tileop.hpp>` and uses the public `ElementTile`, `TPARTVIEW`, and
`TPARTELEMENT` interfaces. Before the final 263-element observation, the same
kernel runs with count 257. The last block then contains three wholly inactive
32-element parts whose indices are all `UINT32_MAX`; exact status fields require
that empty-mask probe and both output guards to pass.

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

The generic CFG compiler now covers the complete U32 expression kernel, the
three-region signed expression kernel, the GM-array predication kernel, and the
two-region typed-carrier kernel. Each selected gate freshly compiles its source,
rejects residual region/view/extract/insert IR, and runs the same ELF on both
models against independent goldens. The indexed gather remains a separate
standard CFG checkpoint. Atomic histogram and Top-K are still being moved to
the same compiler; the earlier nine-ELF AST checkpoint does not establish a
current all-suite pass. All 21 original application entries and their
applicable configurations remain open.

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
resource directory with its official `make install` target. It then builds
eleven ELFs: seven standalone kernels, one required unsharded 17-call Top-K
boundary benchmark, and three focused shards of that boundary table. The two
atomic standalone ELFs each contain one TLEA and one masked `MGATHER.ADD`
static site; every Top-K ELF contains exactly two of each. The expression ELF
must contain all ten native Tile binary selectors, one extra XOR that reuses
an early SSA value, three native `TSTORE` sites, a native execution mask on
every compiler-generated operation, and no scalar
`extractelement`/`insertelement` fallback in compiler IR.
The checker accepts two exact forms: 13 operations when unary expressions stay
separate, or 12 after standard LLVM folds U32 `~(-x)` to `x + UINT32_MAX`.
The optimized form must prove that exact input and constant in IR. Both forms
retain their own long-lived SSA checks and corruption canaries; independent
goldens verify the arithmetic rather than forcing the compiler to undo a valid
optimization.

The signed-expression ELF must contain twelve masked generic operations. Its
checker locks signed division, the quotient/multiply/subtract remainder graph,
the two S32 publication retags, typed `TLOAD`/`TSTORE`, and the logical-shift
fold. Negative canaries reject unsigned division, direct PTO floor remainder,
wrong shift typing, broken remainder dataflow, or mixed signed/unsigned memory
transport.

The indexed-gather ELF must contain one ordinary masked `MGATHER` and no
`MGATHER.ADD`, plus one U32 `TLEA` whose IR contract scales 32-bit element
indices to byte offsets. Its negative canaries reject a lost execution mask,
layout, scaling operand, scalar vector-element fallback, or atomic opcode
substitution. The default command runs the seven pre-existing ELFs plus the
signed-expression, indexed-gather, generic GM-CFG, and generic typed-CFG ELFs
(eleven total). During
focused development, append `--case indexed_gather_tile_element` or
`--case generic_predicated_cfg_i32` or `--case generic_typed_tile_cfg_i32` to
run one new case.

Every ELF runs unchanged on `gfrun` and `gfsim`. Both memory dumps are checked
against separately generated input, histogram, output, and status goldens plus
the order-independent atomic-return contract. A gfsim result is accepted only
when its final invariant line reports `stq_conservation=OK` and
`a3_tile_violation=0`, with no assertion or `LOG_ERROR` in the log. The artifact
directory records the commands, repository heads, installed API tree, tool
hashes, ELFs, logs, dumps, and golden hashes.

应用中的 element array 下标表示逻辑元素。`#pragma pto element for`
要求编译器分析紧随其后的 for 循环；不能忽略 pragma 并退化成普通标量代码。
