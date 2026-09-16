# Histogram suffix-cumsum M32 TileOP listing

```text
function HISTOGRAM_SUFFIX_CUMSUM_M32(Hrow: S32[257])
    -> Hrow: S32[257]

// Hrow[0:256] contains 256 atomic-add histogram bins;
// Hrow[256] is the zero sentinel and remains unchanged.
// One S32 M32 CellReg is <1,32,1,S32,M32><128B>.
// Eight CellRegs form the grouped histogram tile <8,32,8,S32,M32><1KB>.

// A. Load 32 shifted windows.
// valid_col is the flattened number of valid S32 elements. The hardware
// derives complete/partial CellRegs and zero-fills the remaining positions.
k = 0..31
TLOAD      <8,32,8,S32,M32> GM[Hrow + 4*k], valid_col=256-k, pad=0
           -> T(50+k)<8,32,8,S32,M32><1KB>

// B. On-tile binary reduction: 32 -> 16 -> 8 -> 4 -> 2 -> 1.
pair = 0..15
TADD     <8,32,8,S32,M32> T(50+2*pair),T(50+2*pair+1)
             -> T(82+pair)<8,32,8,S32,M32><1KB>

pair = 0..7
TADD     <8,32,8,S32,M32> T(82+2*pair),T(82+2*pair+1)
             -> T(98+pair)<8,32,8,S32,M32><1KB>

pair = 0..3
TADD     <8,32,8,S32,M32> T(98+2*pair),T(98+2*pair+1)
             -> T(106+pair)<8,32,8,S32,M32><1KB>

pair = 0..1
TADD     <8,32,8,S32,M32> T(106+2*pair),T(106+2*pair+1)
             -> T(110+pair)<8,32,8,S32,M32><1KB>

TADD     <8,32,8,S32,M32> T110,T111
             -> T112<8,32,8,S32,M32><1KB>

// T112[:,q:q+1] is the q-th logical CellReg component of T112.
// The slice notation is a logical range operand, not a new VIEW opcode.

// C. Accumulate CellReg components from high bins to low bins.
T112[:,7:8]<1,32,1,S32,M32> -> T120<1,32,1,S32,M32><128B>
TADD     <1,32,1,S32,M32> T112[:,6:7],T120
             -> T121<1,32,1,S32,M32><128B>
TADD     <1,32,1,S32,M32> T112[:,5:6],T121
             -> T122<1,32,1,S32,M32><128B>
TADD     <1,32,1,S32,M32> T112[:,4:5],T122
             -> T123<1,32,1,S32,M32><128B>
TADD     <1,32,1,S32,M32> T112[:,3:4],T123
             -> T124<1,32,1,S32,M32><128B>
TADD     <1,32,1,S32,M32> T112[:,2:3],T124
             -> T125<1,32,1,S32,M32><128B>
TADD     <1,32,1,S32,M32> T112[:,1:2],T125
             -> T126<1,32,1,S32,M32><128B>
TADD     <1,32,1,S32,M32> T112[:,0:1],T126
             -> T127<1,32,1,S32,M32><128B>

// D. Publish the eight completed CellRegs. No intermediate GM stores.
TSTORE   <1,32,1,S32,M32> T120 -> GM[Hrow + 4*224]
TSTORE   <1,32,1,S32,M32> T121 -> GM[Hrow + 4*192]
TSTORE   <1,32,1,S32,M32> T122 -> GM[Hrow + 4*160]
TSTORE   <1,32,1,S32,M32> T123 -> GM[Hrow + 4*128]
TSTORE   <1,32,1,S32,M32> T124 -> GM[Hrow + 4*96]
TSTORE   <1,32,1,S32,M32> T125 -> GM[Hrow + 4*64]
TSTORE   <1,32,1,S32,M32> T126 -> GM[Hrow + 4*32]
TSTORE   <1,32,1,S32,M32> T127 -> GM[Hrow + 0]

return Hrow
```

## Dataflow and cost

```text
T112[:,q:q+1][i] = sum(k=0..31, H[32*q+i+k])
R7 = T120
R6 = T121 = T112[:,6:7] + R7
R5 = T122 = T112[:,5:6] + R6
...
R0 = T127 = T112[:,0:1] + R1
```

After the eight stores, `Hrow[i] = sum(j=i..255, H[j])`.

```text
TLOAD: 32 × 8 CellReg × 128B = 32KiB
TADD:  31 grouped reductions + 7 CellReg accumulations = 38 TADD 指令
       31 grouped TADD × 8 CellReg + 7 single-Cell TADD = 255 CellReg adds
TSTORE: 8 × 128B = 1KiB
```

`T(50+k)`, `T(82+pair)` and the other `T#` values are logical TileOP names;
physical T/U/M/N queue allocation is compiler-managed. `TLOAD` and
`TSTORE` assume the profile performs GM-contiguous ↔ M32 layout conversion
on the load/store path. No `TSUFFIX_SCAN`, cross-Cell `TSHUF`, intermediate GM
store, or explicit barrier/fence is used.

## Profile assumption

This listing assumes `valid_col=256-k` is accepted as the flattened logical
element count for the grouped M32 load. Hardware derives the number of full
and partial CellRegs and zero-fills the tail. If the profile defines
`valid_col` strictly as the number of M32 logical columns, the load line needs
a profile-specific lowering, but the dataflow remains unchanged.

## Diagram

![Histogram suffix-cumsum dataflow](histogram_cumsum_m32.svg)
