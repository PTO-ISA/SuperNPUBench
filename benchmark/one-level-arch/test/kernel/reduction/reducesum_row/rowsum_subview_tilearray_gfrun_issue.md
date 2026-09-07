# [TileOP API] `TPARTVIEW` accepts a Vector/NORM parent and emits an ASL-illegal Local `B.SUBVIEW`

## Relationship to SuperNPUBench Issue #98

This testcase reproduces the same core problem tracked by
[SuperNPUBench #98](https://github.com/PTO-ISA/SuperNPUBench/issues/98): a
RowMajor Vector/NORM local Tile is partitioned with `B.SUBVIEW`, then consumed
by `TROWSUM`.

Do **not** report this as a duplicate gfrun defect. The follow-up analysis in
Issue #98 corrected the original diagnosis:

- gfrun's premature binder validation was fixed in SuperScalarModel commit
  `1ccac152`;
- the current PTO ASL defines Local `B.SUBVIEW` only for a CUBE/Matrix parent;
- therefore rejecting a Vector/NORM parent is the expected fail-closed model
  behavior.

The remaining actionable problem is the compiler/TileOP interface: the public
`TPARTVIEW` API accepts this unsupported type combination and generates an ELF
that is illegal under the current ASL, instead of rejecting it at compile time
or lowering it through a legal representation.

## Environment

| Component | Branch | Commit |
|---|---|---|
| SuperNPUBench | current working tree | testcase below |
| linx-toolchain-build | `main` | `e6a31efb4cfb17f1f1c33265cbf6dbb61bbba156` |
| SuperScalarModel / gfrun | `codex/pr-0.58.4-shared-model` | `762a72c34305f7f1df6964e7dfe202bd3e63a951` |
| pto-spec | `main` | `961fa81eb7f39b095b3173e4b4efb7299c45186d` |

Testcase:

```text
benchmark/one-level-arch/test/kernel/reduction/reducesum_row/
  src/rowsum_subview.cpp
```

Configuration:

```text
ROWSUM_ROWS=32
ROWSUM_COLS=32
ROWSUM_PARTS=4
dtype=FP32
```

## Reproducer

The parent and subview types are both Vector/RowMajor:

```cpp
using tileIn = Tile<Location::Vec, float, 32, 32, BLayout::RowMajor>;
using tileInPart = Tile<Location::Vec, float, 8, 32, BLayout::RowMajor>;

tileIn input_tile;
TLOAD(input_tile, input_ptr);

auto input_parts = TPARTVIEW<tileInPart, 4, 1>(input_tile);

TROWSUM(partial_sum0, input_parts[0][0]);
TROWSUM(partial_sum1, input_parts[1][0]);
TROWSUM(partial_sum2, input_parts[2][0]);
TROWSUM(partial_sum3, input_parts[3][0]);
```

The compiler accepts the program and emits local `B.SUBVIEW` modifiers for a
parent descriptor whose layout is NORM/RowMajor.

## Observed result

Compilation, linking, and disassembly succeed. gfrun rejects the first
`TROWSUM` block:

```text
gfrun: illegal instruction at 0x0: illegal TROWSUM operand or descriptor contract
```

This rejection is consistent with the current ASL. The Local CUBE subview
descriptor requires the parent to use Matrix location and CUBE layout; the
testcase instead supplies a Vector/NORM parent.

## Expected compiler/API behavior

Under the current ISA specification, one of the following should happen:

1. `TPARTVIEW` rejects a Vector/NORM local parent at compile time with a clear
   diagnostic; or
2. TileOP/compiler lowers the operation through an ISA-defined representation
   whose parent satisfies the CUBE/Matrix subview contract.

If Vector/NORM local subviews are intended ISA functionality, PTO ASL must
first define their descriptor, legality, mapping, and overlap semantics. Only
after that specification change should the compiler and gfrun implement the
extension.

## Separate code-generation observation

Using the public `TCVT(TileArrayOutputRef, source)` interface produces 12
runtime parent-size alternatives per call. Four calls therefore leave 48
static `B.ASSEMBLE` candidates in the disassembly.

This is independent of the illegal NORM `B.SUBVIEW` failure. The program is
rejected before assembly executes. Constant propagation or a compile-time
specialized TileArray output path could reduce this code-size overhead.

## Conclusion

- Same core testcase and legality conflict as Issue #98.
- Not a new gfrun issue under the current PTO ASL.
- Current issue ownership: TileOP API/compiler validation or lowering.
- Alternative ownership, only if Vector/NORM subviews are required: ISA/ASL
  extension followed by compiler and model support.
