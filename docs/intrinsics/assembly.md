# Read generated block disassembly

Kernel authors work in C++. Disassembly is a debugging view used to confirm that
loads, tile computation, matrix operations, and stores were lowered into the
expected block templates. It is not a second source language and it does not
change intrinsic semantics.

## Recognize a block

A generated tile block starts with `BSTART.<operation>`, contains descriptors,
and ends when the next block begins. Common descriptors include:

| Descriptor | What to inspect |
| --- | --- |
| `B.DIM` / `C.B.DIMI` | Dynamic or immediate tile dimensions |
| `B.DATR` | Element representation and conversion attributes |
| `B.IOR` | Global-memory base and stride operands |
| `B.IOT` | Source and destination tile IDs |
| `C.B.IOS` | A block-shared tile ID |

## Follow tile dependencies

Relative tile IDs connect a consumer to earlier producers. In this tile-add
excerpt, the add reads the two load results and the store reads the add result:

```asm
BSTART.TLOAD   FP32
B.IOT          last, ->t<4KB>
B.IOR          [a1,a3],[]

BSTART.TLOAD   FP32
B.IOT          last, ->t<4KB>
B.IOR          [a2,a3],[]

BSTART.TADD    FP32
B.IOT          t#1, t#2, last, ->t<4KB>

BSTART.TSTORE  FP32
B.IOT          t#1, last
B.IOR          [a0,a3],[]
```

The numbers are producer-relative references, not C++ variable names or physical
register assignments. Independent producer chains may be interleaved without
changing the C++ value graph.

## Identify a shared matrix operand

A group matrix operation binds its shared right-hand matrix with `C.B.IOS`. The
thread-local left operand and result remain on `B.IOT`:

```asm
BSTART.TLOAD    INT32
B.IOR           [a2,a4],[]
C.B.IOS         S#12

BSTART.TMATMUL  INT32
C.B.DIMI        8, ->lb0
C.B.DIMI        8, ->lb1
C.B.DIMI        8, ->lb2
C.B.IOS         S#12
B.IOT           t#8, last, ->acc<4KB>
```

The shared ID must name the version defined by the corresponding shared load.
Readiness follows the shared tile version shown by the producer and consumer.

## FlashAttention excerpt

The complete checked-in excerpt shows the Q/K product, accumulator conversion,
the probability/V product, and the final store:

```asm
--8<-- "docs/examples/disassembly/flash_attention.s"
```

## Generate disassembly

Use the benchmark's `diss` target so compilation and object inspection use the
same build parameters:

```bash
make TESTCASE=<case> PLAT=linx COMPILER_DIR="$COMPILER_DIR" diss
```

Compare the output with the intrinsic sequence in the kernel page. A missing
block usually indicates unsupported types, an unexpected scalar fallback, or an
optimization that folded the operation into a neighboring block.
