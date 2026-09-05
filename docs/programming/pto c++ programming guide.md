# PTO C++ Programming Guide

> **Version**: 2026.08 | **ISA**: PTO-ISA v0.57.1 | **Toolchain**: linx_blockisa_llvm_musl (clang-15)

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Programming Model](#2-programming-model)
- [3. C++ Programming Interface](#3-c-programming-interface)
- [4. Writing Kernels](#4-writing-kernels)
- [5. Compilation & Toolchain](#5-compilation--toolchain)
- [6. Best Practices & Optimization](#6-best-practices--optimization)
- [7. Appendix](#7-appendix)

---

## 1. Introduction

### 1.1 What Is PTO-ISA?

PTO-ISA (Parallel Tile Operator ISA) is a block-structured Instruction Set
Architecture designed for tile-programming on NPU hardware. It organizes
computation into **block instructions** that operate on **tile registers** —
fixed-size 2D data blocks resident in on-chip register files — rather than
individual scalars.

The programming model is exposed to C++ via **Linx-TileOP-API**, a header-only
template library that maps high-level tile operations (`TADD`, `TMATMUL`,
`TLOAD`, etc.) to inline-assembly block instructions (`BSTART.TEPL`,
`BSTART.TLSU`, `BSTART.CUBE`).

### 1.2 PTO ISA v0.57.1 Operation Set

The ISA defines exactly **120 operations**: 98 TEPL + 9 TMA + 13 CUBE.

| Family | Count | Key Operations |
|--------|-------|----------------|
| **TMA** (memory) | 9 | `TLOAD`, `TSTORE`, `TMOV`, `TPREFETCH`, `MGATHER`, `MSCATTER`, `MGATHER_MASK`, `MSCATTER_MASK`, `MGATHER_CAS` |
| **CUBE** (matrix) | 13 | `TMATMUL`, `TMATMUL_BIAS`, `TMATMUL_ACC`, `TMATMUL_MX` (+Bias/Acc variants), `ACCCVT`, `TGEMV` (+variants) |
| **TEPL** (elementwise) | 98 | `TADD`, `TSUB`, `TMUL`, `TCMP`, `TSEL`, `TROWSUM`, `TCOLMAX`, `TEXPANDS`, `TCONCAT`, `TTRANS`, ... |

### 1.3 Target Architecture

The **DavinciOO v4 core** consists of 4 Processing Elements (PEs):

| Component | Per-PE | Per-Core (4 PE) |
|-----------|--------|-----------------|
| OoO front-end | independent | — |
| CELL register file | 2048 × 128 B = 256 KB | 1 MB |
| TileReg namespace | 4 queues × 16 entries = 64 | 256 |
| TLSU (memory unit) | private | — |
| CUBE (matrix unit) | private | — |
| TEPL (elementwise unit) | private | — |

Each PE executes tile/block intrinsics as **thread-local** instructions — no
implicit cross-PE register access. Global memory is reachable only through TLSU
blocks (`TLOAD`, `TSTORE`, `MGATHER`, `MSCATTER`).

### 1.4 v5 Breaking Changes

PTO ISA v0.57.1 introduces breaking changes from earlier versions:

| Change | Old | New |
|--------|-----|-----|
| `Location::Acc` | Removed | Use `Location::Vec` for accumulator tiles |
| `TileAcc` alias | `Tile<Location::Acc, ...>` | `Tile<Location::Vec, float, R, C, RowMajor>` |
| `ACCCVT` | `ACCCVT(dst, acc)` export | Removed — `TMATMUL*` writes directly to output tile |
| `TMATMUL_ACC` | 3-arg: `(c, a, b)` implicit ACC | 4-arg: `(d, c, a, b)` explicit D/C tiles |
| `TCOPYIN`/`TCOPYOUT` | Tile-to-GM wrappers | Renamed to `TLOAD`/`TSTORE` |
| `TCOPY` (tile copy) | `TCOPY(dst, src)` | Use `TCVT(dst, src)` (same-type = copy) |
| `mask=` keyword in B.IOT | `B.IOT ..., mask=15, ...` | Mask is positional source operand via `B.IOT` |
| `_FIXP` opcodes | `TMATMUL_ACC_FIXP` etc. | Unified via `B.FPATR` header |

### 1.5 Document Scope

This guide covers:
- The C++ tile-programming API (`pto_tileop.hpp`)
- Kernel writing patterns (from SuperNPUBench)
- Compilation with the Linx toolchain

For ISA-level encoding and hardware microarchitecture, see the DavinciOO ISA
intrinsic documentation and [Linx-TileOP-API tileop-usage](https://github.com/LinxISA/Linx-TileOP-API/tree/linx/docs/tileop-usage).

---

## 2. Programming Model

### 2.1 Tile-Centric Execution

Unlike scalar ISAs where each instruction processes one element, PTO-ISA
**block instructions** process entire tiles in one operation:

```cpp
// One TADD call adds two 16×16 tiles (256 elements) in a single block instruction
Tile<Vec, half, 16, 16> a, b, c;
TADD(c, a, b);   // c[i][j] = a[i][j] + b[i][j]  for all 256 elements
```

### 2.2 Memory Hierarchy

```
Global Memory (DRAM)
    ↕  TLOAD / TSTORE / MGATHER / MSCATTER / TMOV  (TLSU blocks)
Tile Register Files (on-chip, per-PE)
    ├── Location::Vec      — general-purpose elementwise tiles
    ├── Location::Left     — GEMM A-operand tiles (boxed, 512B fractal)
    ├── Location::Right    — GEMM B-operand tiles (boxed, 512B fractal)
    ├── Location::Bias     — bias tiles
    └── Location::Scaling  — scaling factor tiles
```

> **v5 change**: `Location::Acc` is removed. Accumulator tiles are now regular
> `Location::Vec` tiles. `TMATMUL*` writes directly to the output tile — no
> separate `ACCCVT` export step needed.

There is **no pointer** to tile registers. Data must be explicitly moved between
global memory and tile registers via `TLOAD`/`TSTORE`.

### 2.3 Tile Register Namespace

Each PE has 4 independent register queues, each with 16 entries:

| Queue | Name | Typical Use |
|-------|------|-------------|
| `T` | `T#1..T#16` | General tile result / temporary |
| `U` | `U#1..U#16` | Second general stream (separate lifetime) |
| `M` | `M#1..M#16` | Extra stream (mask / index / data movement) |
| `N` | `N#1..N#16` | Extra stream (isolated lifetime) |

Queues use **relative indexing**: `#1` = newest live value, `#2` = one older.
When a new tile is produced, it appends to the queue tail. When a source is
consumed without `.reuse`, it may be released.

In C++, the compiler manages register allocation automatically — the programmer
declares C++ `Tile` variables and the lowering pass maps them to TReg queues.

### 2.4 Tile Size Classes

| `imm4` | Tile Bytes | CELLs | Max Live Tiles (capacity-limited) |
|--------|-----------|-------|-----------------------------------|
| 3 | 128 B | 1 | 2048 |
| 4 | 256 B | 2 | 1024 |
| 5 | 512 B | 4 | 512 |
| 6 | 1 KB | 8 | 256 |
| 7 | 2 KB | 16 | 128 |
| 8 | 4 KB | 32 | 64 |
| 9 | 8 KB | 64 | 32 |

Active profile restricts tile allocation to **128 B – 8 KB** (`imm4 = 3..9`).
Total live tile payload per PE must not exceed **256 KB** (2048 CELLs).

### 2.5 Block Instruction Structure

Each tile operation lowers to a **block** described by a header chain:

```
BSTART.<family> <opcode>, <dtype>    ← select block class + opcode + main dtype
B.DATR  <dtype_ext>, <pad>, ...      ← data attributes (optional)
B.DIM   <reg>, <imm>, ->LBx          ← dimensions / loop bounds
B.IOT   <src1>, <src2>, last, -><dst><size>  ← tile operand binding
BSTOP                               ← block end
```

Families:
- **TEPL** — elementwise, tile-scalar, reduce, expand, compare, select (98 ops)
- **TLSU** — TLOAD, TSTORE, TMOV, MGATHER, MSCATTER (+mask, +CAS) (9 ops)
- **CUBE** — TMATMUL, TGEMV, ACCCVT (13 ops)

In C++, these are emitted automatically by the template intrinsics — the
programmer never writes raw assembly.

### 2.6 B.IOT Operand Binding

`B.IOT` encodes at most **2 source TileRegs + 1 destination queue** per line.
Multi-source intrinsics use multiple sequential `B.IOT`s; the last must set
`last`.

Source TileReg encoding (6-bit):

| Encoding | Source |
|----------|--------|
| `0..15` | `T#1..T#16` |
| `16..31` | `U#1..U#16` |
| `32..47` | `M#1..M#16` |
| `48..63` | `N#1..N#16` |

> **v5 change**: Mask for `MGATHER_MASK`/`MSCATTER_MASK` is now a **positional
> source operand** bound via `B.IOT` (not a `mask=` keyword).

### 2.7 Data Types

| Category | Types |
|----------|-------|
| Float | `FP64`, `FP32`, `FP16`, `BF16`, `HiF8`, `e4m3`, `e5m2`, `HiF4x2` |
| Signed int | `S64`, `S32`, `S16`, `S8`, `S4x2` |
| Unsigned int | `U64`, `U32`, `U16`, `U8`, `U4x2` |

In C++, these map to: `float`, `__half`, `__bf16`, `int8_t`, `int16_t`,
`int32_t`, `uint8_t`, `uint16_t`, `uint32_t`, `__fp8_e4m3`, etc.

---

## 3. C++ Programming Interface

### 3.1 Include Strategy

The single entry point header:

```cpp
#include <common/pto_tileop.hpp>
using namespace pto;
```

This transitively includes:
1. `pto_tile.hpp` — Tile / GlobalTensor type system + concepts
2. `tileop_api.hpp` — public API wrappers (TLOAD, TSTORE, TADD, ...)
3. `global_iterator.hpp` — DRAM tile-stepping iterator
4. `tile_tensor_impl.hpp` — out-of-line Tile constructors

Backend is selected by defining exactly one of: `__linx`, `__ARM_FEATURE_SME`,
or `__cpu_sim__` at compile time.

> Under `__linx`, the inline-asm operations are provided directly (no `_Impl`
> wrappers). Some API wrappers (e.g. `TCOPY`) are only available under
> non-`__linx` backends; use `TCVT` as a tile-to-tile copy alternative.

### 3.2 Tile Types

#### Tile

The on-chip tile type, statically shaped:

```cpp
template <Location Loc, typename DType, int Rows, int Cols,
          BLayout BFractal = BLayout::RowMajor,
          int ValidRow = Rows, int ValidCol = Cols,
          SLayout SFractal = SLayout::NoneBox,
          int SFractalSize = 512,
          PadValue PadVal = PadValue::Null>
struct Tile;
```

| Parameter | Meaning |
|-----------|---------|
| `Loc` | Pipeline stage: `Vec`, `Left`, `Right`, `Mat`, `Bias`, `Scaling`, `Shared` |
| `DType` | Element type (`half`, `float`, `int32_t`, ...) |
| `Rows`, `Cols` | Compile-time tile footprint |
| `ValidRow`, `ValidCol` | Active (unpadded) extent; `-1` = runtime dynamic |
| `BFractal` | Block-level layout: `RowMajor` or `ColMajor` |
| `SFractal` | Inner fractal layout: `NoneBox`, `RowMajor`, `ColMajor` |
| `SFractalSize` | `512` for input tiles, `1024` for accumulator tiles |

#### Convenience Aliases

```cpp
// GEMM A-operand: ColMajor outer, RowMajor fractal, 512B block
template <typename E, int R, int C, int VR = R, int VC = C>
using TileLeft  = Tile<Location::Left,  E, R, C, BLayout::ColMajor, VR, VC, SLayout::RowMajor, 512>;

// GEMM B-operand: RowMajor outer, ColMajor fractal, 512B block
template <typename E, int R, int C, int VR = R, int VC = C>
using TileRight = Tile<Location::Right, E, R, C, BLayout::RowMajor, VR, VC, SLayout::ColMajor, 512>;
```

> **v5 change**: `TileAcc` (was `Tile<Location::Acc, ...>`) is removed. Use
> `Tile<Location::Vec, float, R, C, BLayout::RowMajor>` for accumulator tiles.
> `TMATMUL*` writes directly to this tile — no separate ACC export needed.

#### General-Purpose Tile (elementwise / reduction / load-store)

```cpp
Tile<Vec, half, 16, 16, BLayout::RowMajor> a;
Tile<Vec, float, 16, 16, BLayout::ColMajor> b;
```

#### Constructors

```cpp
TileLeft<half, 128, 128> a;                    // default (uninitialized)
TileLeft<half, 128, 128> a(0.0_h);              // fill with scalar
TileLeft<half, 128, 128, -1, -1> a(96, 64);    // dynamic valid extents
TileLeft<half, 128, 128, -1, -1> a(0.0_h, 96, 64);  // fill + dynamic extents
```

### 3.3 Global Memory Types

#### global_tensor (convenience wrapper)

```cpp
using GM = global_tensor<half, RowMajor<256, 256>>;
GM gA(dram_ptr);                    // both dims static
GM gA(dram_ptr, dynamic_cols);      // one dim dynamic
GM gA(dram_ptr, dynamic_rows, dynamic_cols);  // both dynamic
```

#### global_iterator (tile-stepping)

```cpp
using GM = global_tensor<half, RowMajor<256, 256>>;
GM gA(dram_ptr);
global_iterator<GM, TileLeft<half, 128, 128>> it(gA.data());

for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) {
        TileLeft<half, 128, 128> a;
        auto view = it(i, j);       // returns a GlobalTensor view of one 128×128 tile
        TLOAD(a, view);             // load that tile
    }
```

### 3.4 Tile Operations — C++ API Reference

#### Memory Operations (TLSU family)

| Function | Signature | Description |
|----------|-----------|-------------|
| `TLOAD` | `(tile& dst, gm& src)` | Load tile from global memory |
| `TSTORE` | `(gm& dst, tile& src)` | Store tile to global memory |
| `TCVT` | `(tile& dst, tile& src)` | Type/layout conversion (same-type = copy) |
| `MGATHER` | `(tile& dst, gm& src, tile& offsets)` | Gather by per-element byte offsets |
| `MSCATTER` | `(gm& dst, tile& src, tile& offsets)` | Scatter by per-element byte offsets |

> **v5 changes**: `TCOPYIN`→`TLOAD`, `TCOPYOUT`→`TSTORE`. `TCOPY` (tile-to-tile
> copy) is not available under `__linx`; use `TCVT` with same-type tiles. `TMOV`
> (PTO ISA TMA Function 2) performs tile-to-tile move with layout transform.
> `MGATHER_MASK`/`MSCATTER_MASK` bind mask as positional source (no `mask=` keyword).

#### Matrix Operations (CUBE family)

| Function | Signature | Description |
|----------|-----------|-------------|
| `TMATMUL` | `(C&, A&, B&)` | `C = A × B` (write-initialize) |
| `TMATMUL_ACC` | `(D&, C&, A&, B&)` | `D = C + A × B` (read-write accumulate) |
| `TMATMUL_BIAS` | `(C&, A&, B&, Bias&)` | `C = A × B + Bias` |
| `TMATMUL_MX` | `(C&, A&, AX&, B&, BX&)` | MX mixed-precision (A+B scaling) |
| `ACCCVT` | removed | TMATMUL* writes directly to output tile |

> **v5 changes**: ACC is no longer implicit. `TMATMUL_ACC` takes 4 explicit
> arguments `(d, c, a, b)` where D=output, C=accumulator input. For in-place
> accumulation: `TMATMUL_ACC(acc, acc, a, b)`. `_FIXP` variants replaced by
> unified `B.FPATR` header.

#### Elementwise Binary (TEPL family)

| Function | Description |
|----------|-------------|
| `TADD(dst, a, b)` | `dst = a + b` |
| `TSUB(dst, a, b)` | `dst = a - b` |
| `TMUL(dst, a, b)` | `dst = a * b` |
| `TDIV(dst, a, b)` | `dst = a / b` |
| `TMAX(dst, a, b)` | `dst = max(a, b)` |
| `TMIN(dst, a, b)` | `dst = min(a, b)` |
| `TAND(dst, a, b)` | Bitwise AND |
| `TOR(dst, a, b)` | Bitwise OR |
| `TXOR(dst, a, b)` | Bitwise XOR |
| `TCMP(dst, a, b)` | Compare (produces mask) |

#### Tile-Scalar Operations

| Function | Description |
|----------|-------------|
| `TADDs(dst, src, scalar)` | `dst = src + scalar` |
| `TSUBs(dst, src, scalar)` | `dst = src - scalar` |
| `TMULs(dst, src, scalar)` | `dst = src * scalar` |
| `TDIVs(dst, src, scalar)` | `dst = src / scalar` |
| `TMAXs(dst, src, scalar)` | `dst = max(src, scalar)` |
| `TMINs(dst, src, scalar)` | `dst = min(src, scalar)` |

#### Broadcast / Fill

| Function | Description |
|----------|-------------|
| `TEXPANDS(tile, scalar)` | Fill entire tile with scalar value |
| `TCI(tile, scalar)` | Constant-inject ramp `[base, base+1, ...]` |
| `TROWEXPANDMUL(dst, mat, vec)` | Row-broadcast multiply |
| `TCOLEXPANDMUL(dst, mat, vec)` | Column-broadcast multiply |
| `TROWEXPANDADD`, `TCOLEXPANDADD`, ... | Broadcast add/sub/div/max/min variants |

#### Reductions

| Function | Description |
|----------|-------------|
| `TROWSUM(dst, src)` | Row-wise sum → 1-element-per-row vector |
| `TCOLSUM(dst, src)` | Column-wise sum → 1-element-per-col vector |
| `TROWMAX(dst, src)` | Row-wise max |
| `TCOLMAX(dst, src)` | Column-wise max |
| `TROWMIN(dst, src)` | Row-wise min |
| `TCOLMIN(dst, src)` | Column-wise min |

#### Shift / Special

| Function | Description |
|----------|-------------|
| `TSHL(dst, src, shift)` | Left shift |
| `TSHR(dst, src, shift)` | Right shift |
| `TTRANS(dst, src)` | Hardware 2D transpose |
| `TRESHAPE(dst, src)` | Reshape (same numel) |
| `TEXTRACT(dst, src)` | Extract sub-tile |
| `TINSERT(dst, src)` | Insert sub-tile |
| `TCONCAT(dst, src0, src1)` | Concatenate two tiles |
| `TSELECT(dst, mask, a, b)` | Select by mask |

### 3.5 Concepts and Constraints

```cpp
template <typename T> concept is_tile_data_v;    // matches Tile<...>
template <typename T> concept is_global_data_v;  // matches GlobalTensor / global_tensor
template <typename T> concept is_boxed_data_v;   // boxed (fractal) tile
```

---

## 4. Writing Kernels

### 4.1 Kernel Structure

A PTO kernel is a C++ template function. Shapes are compile-time template
parameters enabling full unrolling:

```cpp
template <typename dtype, int gM, int gN, int gK,
          int tM, int tN, int tK>
void my_kernel(float *out_ptr, dtype *a_ptr, dtype *b_ptr) {
    using namespace pto;
    // ... tile declarations and operations ...
}
```

### 4.2 Example: Matrix Multiply

```cpp
#include <common/pto_tileop.hpp>
using namespace pto;

template <typename dtype, int gM, int gN, int gK, int tM, int tN, int tK>
void matmul(float *c_ptr, dtype *a_ptr, dtype *b_ptr) {
    using gmA = global_tensor<dtype, RowMajor<gM, gK>>;
    using gmB = global_tensor<dtype, RowMajor<gK, gN>>;
    using gmC = global_tensor<float, RowMajor<gM, gN>>;

    using TileA = TileLeft<dtype, tM, tK>;
    using TileB = TileRight<dtype, tK, tN>;
    // v5: accumulator is now a regular Vec tile (no TileAcc)
    using TileC = Tile<Vec, float, tM, tN, BLayout::RowMajor>;

    gmA gA(a_ptr); gmB gB(b_ptr); gmC gC(c_ptr);
    global_iterator<gmA, TileA> itA(gA.data());
    global_iterator<gmB, TileB> itB(gB.data());
    global_iterator<gmC, TileC> itC(gC.data());

    constexpr int Mb = gM / tM, Nb = gN / tN, Kb = gK / tK;

    for (int i = 0; i < Mb; ++i) {
        for (int j = 0; j < Nb; ++j) {
            TileC acc(0.0f);                    // zero-initialize
            for (int k = 0; k < Kb; ++k) {
                TileA tA;  TileB tB;
                TLOAD(tA, itA(i, k));
                TLOAD(tB, itB(k, j));
                if (k == 0)
                    TMATMUL(acc, tA, tB);        // first: acc = A × B
                else
                    TMATMUL_ACC(acc, acc, tA, tB);  // v5: 4-arg, in-place accumulate
            }
            TSTORE(itC(i, j), acc);              // v5: store directly (no ACCCVT)
        }
    }
}
```

### 4.3 Example: Elementwise + Reduction

```cpp
template <typename dtype, int gM, int gN, int tM, int tN>
void reducesum(dtype *out_ptr, dtype *in_ptr) {
    using namespace pto;
    using gmIn  = global_tensor<dtype, RowMajor<gM, gN>>;
    using gmOut = global_tensor<dtype, RowMajor<1, gN>>;
    using TileData = Tile<Vec, dtype, tM, tN, BLayout::RowMajor>;
    using TileSum  = Tile<Vec, dtype, 1, tN, BLayout::RowMajor>;

    gmIn gIn(in_ptr); gmOut gOut(out_ptr);
    global_iterator<gmIn, TileData> itIn(gIn.data());
    global_iterator<gmOut, TileSum> itOut(gOut.data());

    constexpr int Mb = gM / tM, Nb = gN / tN;
    for (int j = 0; j < Nb; ++j) {
        TileSum sum;
        TEXPANDS(sum, static_cast<dtype>(0));
        for (int i = 0; i < Mb; ++i) {
            TileData data;
            TLOAD(data, itIn(i, j));
            TileSum partial;
            TCOLSUM(partial, data);
            TADD(sum, sum, partial);
        }
        TSTORE(itOut(0, j), sum);
    }
}
```

### 4.4 Example: Type Conversion (bf16 ↔ fp32)

```cpp
// TCVT performs type conversion; with same-type tiles it acts as a copy
tile_x xq;       // Tile<Vec, __bf16, ...>
tile_f xf;       // Tile<Vec, float, ...>

TLOAD(xq, g_x);
TCVT(xf, xq);                   // bf16 → fp32 (upcast for computation)
// ... compute in fp32 ...
TCVT(xq, outf);                 // fp32 → bf16 (downcast for storage)
TSTORE(g_out, xq);
```

### 4.5 Tail / Boundary Handling

For dimensions not evenly divisible by tile size, use the `ValidRow`/`ValidCol`
template parameters:

```cpp
constexpr int rmd_M = gM % tM;
constexpr int rmd_N = gN % tN;

using TileData     = Tile<Vec, dtype, tM, tN, BLayout::RowMajor>;
using TileDataRow  = Tile<Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, tN>;
using TileDataCol  = Tile<Vec, dtype, tM, tN, BLayout::RowMajor, tM, rmd_N>;
using TileDataCor  = Tile<Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, rmd_N>;

for (int i = 0; i < Mb; ++i) {
    TileData data;
    // ... process full tile ...
}
if constexpr (rmd_M > 0) {
    TileDataRow data;    // valid region = rmd_M × tN
    // ... process tail ...
}
```

### 4.6 Test Driver Pattern

```cpp
#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "single_thread/matmul/matmul.hpp"

#define globM 256
#define globN 256
#define globK 256
#define tilM  16
#define tilN  16
#define tilK  16

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN 4096

int main() {
    uint8_t a_buf[globM * globK * sizeof(__half) + 2 * ALIGN];
    __half *a = (__half *)(((uint64_t)a_buf & ALIGN_MASK) + ALIGN);

    BENCHSTART;
    matmul<__half, globM, globN, globK, tilM, tilN, tilK>(c, a, b);
    BENCHEND;
    return 0;
}
```

---

## 5. Compilation & Toolchain

### 5.1 Toolchain

Build the Linx toolchain from `linx-toolchain-build`:

```bash
cd linx-toolchain-build
make init-src
make WITH_TARGET=linx64v5-linux-musl
export COMPILER_DIR=$(pwd)/output/linx_blockisa_llvm_musl/bin
```

### 5.2 Compiler Flags

```bash
$COMPILER_DIR/clang++ \
    -mlxbc -fenable-matrix -O2 \
    -mllvm -enable-all-vector-as-tilereg=true \
    -std=c++20 \
    -D__linx -DENABLE_TENSOR_INSTR \
    -I<repo>/benchmark/one-level-arch/include \
    -I<repo>/benchmark/one-level-arch/test/common \
    -I<repo>/benchmark/one-level-arch/kernels \
    kernel.cpp -nostartfiles _start.s -o kernel.elf
```

| Flag | Purpose |
|------|---------|
| `-mlxbc` | Enable BlockISA code generation |
| `-fenable-matrix` | Enable matrix/tile intrinsics |
| `-std=c++20` | Required for concepts |
| `-D__linx` | Select Linx backend |
| `-DENABLE_TENSOR_INSTR` | Enable tensor instruction definitions |
| `-O2` | Optimization level |

### 5.3 Makefile Build System

```bash
cd benchmark/one-level-arch/test/kernel/matmul
make TESTCASE=matmul TYPE=MASK MODE=MASK_FP32 M=256 N=256 K=256 tM=16 tN=16 tK=16
make TESTCASE=matmul ... diss    # also generate disassembly
```

### 5.4 Running on Simulator

```bash
# Functional model (correctness)
bin/gfrun -f kernel.elf

# Cycle-accurate timing model
bin/gfsim -f kernel.elf
```

---

## 6. Best Practices & Optimization

### 6.1 Register Pressure Management

- Use `TMATMUL_ACC(acc, acc, a, b)` for fused accumulate (4-arg v5 form)
- Use `Tile<Vec, float, R, C, RowMajor>` for accumulation — FP32 avoids precision loss
- Keep live tile count within the 64-entry naming window and 256 KB capacity
- Distribute long-lived tiles across T/U/M/N queues to avoid window overflow

### 6.2 Tile Reuse

- Load A tiles once and reuse across N-dimension iterations (A-tile reuse)
- Use `TCVT` to copy tiles when the source is needed by multiple consumers
- Minimize `TLOAD`/`TSTORE` — register-resident compute is much faster than DMA

### 6.3 Tail Handling

- Use `if constexpr (rmd > 0)` to elide empty tail paths at compile time
- Declare dedicated `ValidRow`/`ValidCol` tile types for each tail quadrant
- The 4-quadrant pattern: full / col-tail / row-tail / corner

### 6.4 Layout Selection

- GEMM operands must use `TileLeft`/`TileRight` (boxed fractal, 512B)
- Accumulators: `Tile<Vec, float, R, C, RowMajor>` (no boxed layout needed)
- Elementwise/reduction/load-store: plain `Tile<Vec, ...>` with `NoneBox`
- Match global memory layout (`RowMajor`/`ColMajor`) to avoid implicit transposes

### 6.5 Alignment

- Global memory buffers must be **4 KB aligned** for DMA tile loads
- Tile columns must be 32-byte aligned (e.g. `Npc % 16 == 0` for bf16)
- Use the `ALIGN_MASK + ALIGN` idiom for stack buffers

### 6.6 Static Shapes

- Pass all matrix dimensions and tile sizes as **template parameters**
- This enables full loop unrolling and compile-time register allocation
- Use `constexpr` arithmetic for derived values (`Mb = gM / tM`, `rmd = gM % tM`)

### 6.7 v5 Migration Guide

| Old pattern | New pattern |
|------------|------------|
| `TileAcc<float, R, C>` | `Tile<Vec, float, R, C, RowMajor>` |
| `TMATMUL_ACC(acc, a, b)` | `TMATMUL_ACC(acc, acc, a, b)` |
| `ACCCVT(out, acc)` + `TSTORE(gm, out)` | `TSTORE(gm, acc)` (direct) |
| `TCOPYIN(tile, gm)` | `TLOAD(tile, gm)` |
| `TCOPYOUT(gm, tile)` | `TSTORE(gm, tile)` |
| `TCOPY(dst, src)` | `TCVT(dst, src)` (same-type = copy) |
| `Location::Acc` | `Location::Vec` |
| `template_asm.h` _TEPL variants | Standard tileop-api functions |

---

## 7. Appendix

### 7.1 PTO ISA v0.57.1 Operation Index

#### TMA (9 operations)

| Operation | Function | Description |
|-----------|----------|-------------|
| `TLOAD` | 0 | Global → Tile |
| `TSTORE` | 1 | Tile → Global |
| `TMOV` | 2 | Tile → Tile (layout transform) |
| `TPREFETCH` | 3 | Prefetch (reserved) |
| `MGATHER` | 4 | Gather by offsets |
| `MSCATTER` | 5 | Scatter by offsets |
| `MGATHER_MASK` | 6 | Masked gather (mask = positional source) |
| `MSCATTER_MASK` | 7 | Masked scatter (mask = positional source) |
| `MGATHER_CAS` | 8 | Compare-and-swap gather |

#### CUBE (13 operations)

| Operation | ACC Effect | Description |
|-----------|-----------|-------------|
| `TMATMUL` | write-initialize | `D = A × B` |
| `TMATMUL_BIAS` | write-initialize | `D = A × B + Bias` |
| `TMATMUL_ACC` | read-write | `D = C + A × B` (4-arg: D, C, A, B) |
| `TMATMUL_MX` | write-initialize | MX (A+B scaling) |
| `TMATMUL_MX_BIAS` | write-initialize | MX with bias |
| `TMATMUL_MX_ACC` | read-write | MX accumulate |
| `TMATMUL_MX_BIAS_ACC` | read-write | MX bias accumulate |
| `ACCCVT` | — | Removed in v5 (TMATMUL writes directly) |
| `TGEMV` | write-initialize | Vector GEMV |
| `TGEMV_BIAS`/`TGEMV_ACC`/`TGEMV_MX`/... | various | TGEMV variants |

#### TEPL (98 operations)

`TADD`, `TSUB`, `TMUL`, `TDIV`, `TMAX`, `TMIN`, `TAND`, `TOR`, `TXOR`,
`TCMP`, `TADDS`, `TSUBS`, `TMULS`, `TDIVS`, `TMAXS`, `TMINS`,
`TABS`, `TEXP`, `TLOG`, `TSQRT`, `TRSqrt`, `TRECIP`, `TNEG`, `TNOT`,
`TCVT`, `TCAST`, `TRESHAPE`, `TTRANS`, `TSELECT`,
`TEXPANDS`, `TCI`, `TROWEXPAND{ADD,SUB,MUL,DIV,MAX,MIN,EXPDIF}`,
`TCOLEXPAND{ADD,SUB,MUL,DIV,MAX,MIN,EXPDIF}`, `TCONCAT`,
`TROWSUM`, `TCOLSUM`, `TROWMAX`, `TCOLMAX`, `TROWMIN`, `TCOLMIN`,
`TROWARGMAX`, `TROWARGMIN`, `TCOLARGMAX`, `TCOLARGMIN`,
`TPARTADD`, `TPARTMUL`, `TPARTMAX`, `TPARTMIN`,
`TSHL`, `TSHR`, `TSHLS`, `TSHRS`,
`TEXTRACT`, `TINSERT`, `TFILLPAD`,
`TSEL`, `TSELS`, `THISTOGRAM`, `TMRGSORT`, `TIMG2COL`, ...

### 7.2 Location Enum

```cpp
enum class Location {
    Vec,       // General-purpose elementwise tile
    Mat,       // Matrix tile (L1)
    Left,      // GEMM A-operand (L0A)
    Right,     // GEMM B-operand (L0B)
    Bias,      // Bias tile
    Scaling,   // Scaling factor tile
    Shared,    // v5: storage-class marker for SharedTile<LocalTile>
};
```

### 7.3 Layout Enums

```cpp
enum class BLayout  { RowMajor, ColMajor };
enum class SLayout  { NoneBox, RowMajor, ColMajor };
enum class PadValue { Zero=0, Max=1, Min=2, Null=3 };
enum class CmpMode  { EQ, NE, GT, LT, GE, LE };
```

### 7.4 Tile Layout Cheat Sheet

| Tile Alias | Location | BFractal | SFractal | SFractalSize | Use |
|------------|----------|----------|----------|---------------|-----|
| `TileLeft` | Left | ColMajor | RowMajor | 512 | GEMM A |
| `TileRight` | Right | RowMajor | ColMajor | 512 | GEMM B |
| `Tile<Vec, float, ..., RowMajor>` | Vec | RowMajor | NoneBox | 512 | Accumulator (v5) |
| `Tile<Vec, dtype, ..., RowMajor>` | Vec | RowMajor | NoneBox | 512 | Elementwise |

### 7.5 Capacity Quick Reference

```
Per-PE:  256 KB CELL register file (2048 × 128 B)
         64 named TReg entries (4 queues × 16)
Per-core: 1 MB aggregate (4 PE)

Tile size: 128 B – 8 KB (imm4 = 3..9)
CELL:     128 B minimum granularity

Constraint: sum(live tile bytes) ≤ 256 KB AND live named entries ≤ 64
```

### 7.6 File Organization

```
benchmark/one-level-arch/
├── include/common/pto_tileop.hpp    ← Public API (include this)
├── kernels/                          ← Header-only kernel implementations
│   ├── matmul/matmul.hpp
│   ├── fa/sfa_pto.hpp
│   ├── reduction/reducesum_colvec_pto.hpp
│   ├── transpose/transpose_pto.hpp
│   └── deepseek/                     ← 22 migrated kernels
├── test/kernel/                      ← Test drivers + build system
│   ├── common/Makefile.common        ← Shared build rules
│   ├── matmul/{Makefile, compile.all, src/}
│   ├── fa/{Makefile, compile.all, src/}
│   └── deepseek/{Makefile, compile.all, src/}
└── compile_all.sh                    ← Top-level: compiles all operators
```

### 7.7 References

- [DavinciOO ISA Intrinsic Documentation](https://github.com/PTO-ISA/DavinciOO/tree/main/isa/intrinsic)
- [Linx-TileOP-API tileop-usage](https://github.com/LinxISA/Linx-TileOP-API/tree/linx/docs/tileop-usage)
- [SuperScalarModel Simulator](https://github.com/LinxISA/SuperScalarModel)
- [Linx Toolchain Build](https://github.com/LinxISA/linx-toolchain-build)
