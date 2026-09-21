#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>

// HISTOGRAM_CUMSUM_M32: in-place 256-bin suffix sum
//     hist[i] = sum_{j = i .. 255} hist_orig[j]
// on a per-PE GM histogram, as tile ops.
//
// Layout: the 256 S32 bins are one 1KB S32 CUBE_M32 grouped tile. A single
// grouped ND2M32 TLOAD (ValidCol=8, ValidRow=32) packs the dense stream so that
// cell_q[r] = bin[8r+q]; the eight physical CELLs sit along the logical columns
// and are addressable with B.SUBVIEW. hist[256:384] is the zero pad the load
// reads past the sentinel.
//
// Dataflow:
//   1. group_tload_832          one 1KB TLOAD of the 256 bins
//   2. within-group suffix      parallel pair tree of binary TADDs over
//                               B.SUBVIEW CELL reads (depth 3); cell_q becomes
//                               sum_{j>=q} bin[8r+j], cell_0 the group total
//   3. across-group suffix      a 32-lane TSHUF scan of cell_0 (the group
//                               totals), then the shifted-by-one scan added
//                               back to all eight cells
//   4. eight strided TSTOREs    scatter cell_q[r] -> hist[8r+q]
//
// Output cost / future work: step 4 is a stride-8 scatter done as eight
// [32,1] strided TSTOREs, and it dominates the kernel wall time (per byte a
// grouped store is several times cheaper). Two ways to collapse it into one
// large-packet 1KB write:
//   * TADD + assemble - have the final broadcast TADD write the eight CELLs
//     into a [32,8] parent via a destination-side B.ASSEMBLE, then issue one
//     grouped TSTORE. Not available today: the TEPL `_ASS` producer path
//     crashes the LinxV5 backend, and the working region assemble route only
//     accepts a RowMajor destination with CUBE subview sources, so a CUBE
//     parent cannot be built from computed CELLs.
//   * TSTORE microarchitecture coalescing - merge the eight strided
//     [32,1] stores into a single large-packet write in the TLSU.
// Until one of those lands, the strided stores are kept.

namespace histogram_cumsum_m32 {

using namespace pto;

constexpr int kLane = 32;

using I32Tile = VecTileM32<int32_t, kLane, 1>;
using U32Tile = VecTileM32<uint32_t, kLane, 1>;

struct GroupTileS32 {
    linx_tile_carrier<1024> carrier;
    linx_tile_carrier<1024>::RegisterType &data() { return carrier.Register; }
};

// One 1KB grouped ND2M32 load: T(r,q) = bin[8r+q], i.e. the eight physical
// CELLs sit along the 8 logical columns so B.SUBVIEW can address CELL q.
inline void group_tload_832(GroupTileS32 &dst, const int32_t *base) {
    asm volatile(
        "BSTART.TLSU TLOAD, %D[DataType]\n"
        "B.DATR ND2M32, Zero\n"
        "B.DIM zero, %c[ValidCol], ->lb0\n"
        "B.DIM zero, %c[ValidRow], ->lb1\n"
        "B.IOT mask=1111, last, ->%[dst]<%Z[TileSize]>\n"
        "B.IOR [%[base], %[stride]], []\n"
        : [dst] "=Tr"(dst.data())
        : [base] "r"(base), [stride] "r"(8 * sizeof(int32_t)),
          [DataType] "i"(type_traits<int32_t>::TypeCode),
          [TileSize] "i"(tile_type_traits<linx_tile_carrier<1024>>::TilesizeCode),
          [ValidCol] "i"(8), [ValidRow] "i"(kLane)
        : "memory");
}

// Grouped contiguous store of a 1KB tile: writes the 256 words flat.
inline void group_tstore(int32_t *base, GroupTileS32 &src) {
    asm volatile(
        "BSTART.TLSU TSTORE, %D[DataType]\n"
        "B.DATR M322ND, Null\n"
        "B.DIM zero, %c[ValidCol], ->lb0\n"
        "B.DIM zero, %c[ValidRow], ->lb1\n"
        "B.IOT %[src], mask=1111, last\n"
        "B.IOR [%[base], %[stride]], []\n"
        :
        : [base] "r"(base), [stride] "r"(sizeof(int32_t)),
          [src] "Tr"(src.data()),
          [DataType] "i"(type_traits<int32_t>::TypeCode),
          [ValidCol] "i"(1), [ValidRow] "i"(8 * kLane)
        : "memory");
}

// Broadcast a scalar into a grouped 1KB tile (used by topk's GM zeroing
// helper). TEXPANDS (TEPL 59), raw block form.
inline void group_texpands(GroupTileS32 &dst, int32_t scalar) {
    // Anti-fold: keep a compile-time-constant scalar off the zero register so
    // B.IOR [reg],[] still matches an instruction (wrapper convention).
    asm("" : "+r"(scalar));
    asm volatile(
        "BSTART.TEPL 59, %D[DataType]\n"
        "B.DATR CUBE_M32, Null\n"
        "B.DIM zero, %c[ValidCol], ->lb0\n"
        "B.DIM zero, %c[ValidRow], ->lb1\n"
        "B.DIM zero, %c[Col], ->lb2\n"
        "B.IOT mask=1111, last, ->%[dst]<%Z[TileSize]>\n"
        "B.IOR [%[scalar]],[]\n"
        : [dst] "=Tr"(dst.data())
        : [scalar] "r"(scalar),
          [DataType] "i"(type_traits<int32_t>::TypeCode),
          [TileSize] "i"(tile_type_traits<
                           linx_tile_carrier<1024>>::TilesizeCode),
          [ValidCol] "i"(1), [ValidRow] "i"(8 * kLane), [Col] "i"(1)
        : "memory");
}

// Pure copy of CELL `cell` out of the grouped parent: dst = parent.cell[cell].
inline void group_tmuls_cell(I32Tile &dst, GroupTileS32 &parent, uint32_t cell,
                             int32_t scalar) {
    asm volatile(
        "BSTART.TEPL 34, %D[DataType]\n"
        "B.DATR CUBE_M32, Null\n"
        "B.DIM zero, 1, ->lb0\n"
        "B.DIM zero, %c[ValidRow], ->lb1\n"
        "B.DIM zero, 1, ->lb2\n"
        "B.IOT %[parent], mask=1111, last, ->%[dst]<%Z[DstSize]>\n"
        "B.SUBVIEW 0, %[cell], 0, %c[SrcSize]\n"
        "B.IOR [%[scalar]],[]\n"
        : [dst] "=Tr"(dst.data())
        : [parent] "Tr"(parent.data()), [cell] "r"(cell), [scalar] "r"(scalar),
          [DataType] "i"(type_traits<int32_t>::TypeCode),
          [DstSize] "i"(tile_type_traits<I32Tile::TileDType>::TilesizeCode),
          [SrcSize] "i"(1), [ValidRow] "i"(kLane)
        : "memory");
}

// Binary TADD with two B.SUBVIEW sources of one CUBE_M32 parent:
// dst = parent.cell[cell0] + parent.cell[cell1].
inline void group_tadd_cell_pair(I32Tile &dst, GroupTileS32 &parent,
                                 uint32_t cell0, uint32_t cell1) {
    asm volatile(
        "BSTART.TEPL 0, %D[DataType]\n"
        "B.DATR CUBE_M32, Null\n"
        "B.DIM zero, 1, ->lb0\n"
        "B.DIM zero, %c[ValidRow], ->lb1\n"
        "B.DIM zero, 1, ->lb2\n"
        "B.IOT %[parent], %[parent], mask=1111, last, ->%[dst]<%Z[DstSize]>\n"
        "B.SUBVIEW 0, %[c0], 0, %c[SrcSize]\n"
        "B.SUBVIEW 1, %[c1], 0, %c[SrcSize]\n"
        : [dst] "=Tr"(dst.data())
        : [parent] "Tr"(parent.data()), [c0] "r"(cell0), [c1] "r"(cell1),
          [DataType] "i"(type_traits<int32_t>::TypeCode),
          [DstSize] "i"(tile_type_traits<I32Tile::TileDType>::TilesizeCode),
          [SrcSize] "i"(1), [ValidRow] "i"(kLane)
        : "memory");
}

// Binary TADD with one B.SUBVIEW source: dst = parent.cell[cell0] + addend.
inline void group_tadd_cell_tail(I32Tile &dst, GroupTileS32 &parent,
                                 uint32_t cell0, I32Tile &addend) {
    asm volatile(
        "BSTART.TEPL 0, %D[DataType]\n"
        "B.DATR CUBE_M32, Null\n"
        "B.DIM zero, 1, ->lb0\n"
        "B.DIM zero, %c[ValidRow], ->lb1\n"
        "B.DIM zero, 1, ->lb2\n"
        "B.IOT %[parent], %[add], mask=1111, last, ->%[dst]<%Z[DstSize]>\n"
        "B.SUBVIEW 0, %[c0], 0, %c[SrcSize]\n"
        : [dst] "=Tr"(dst.data())
        : [parent] "Tr"(parent.data()), [add] "Tr"(addend.data()),
          [c0] "r"(cell0),
          [DataType] "i"(type_traits<int32_t>::TypeCode),
          [DstSize] "i"(tile_type_traits<I32Tile::TileDType>::TilesizeCode),
          [SrcSize] "i"(1), [ValidRow] "i"(kLane)
        : "memory");
}

// TSHUF row shift (TEPL 118). mode 1 shifts lane r <- lane r+b inside each
// segment of segmentWidth = 2 << segmentCode lanes; boundary != 0 zero-fills
// the out-of-range tail. control = mode | (segmentCode<<8) | (boundary<<16).
inline void tshuf_shift_i32(I32Tile &dst, I32Tile &src, U32Tile &controls,
                            uint64_t control) {
    asm("" : "+r"(control));
    asm volatile(
        "BSTART.TEPL 118, %D[DataType]\n"
        "B.DATR CUBE_M32, Zero\n"
        "B.DIM zero, %c[ValidCol], ->lb0\n"
        "B.DIM zero, %c[ValidRow], ->lb1\n"
        "B.DIM zero, %c[Cols], ->lb2\n"
        "B.IOT %[src], %[ctrl], mask=1111, last, ->%[dst]<%Z[DstSize]>\n"
        "B.IOR [%[control]],[]\n"
        : [dst] "=Tr"(dst.data())
        : [src] "Tr"(src.data()), [ctrl] "Tr"(controls.data()),
          [control] "r"(control),
          [DataType] "i"(type_traits<int32_t>::TypeCode),
          [DstSize] "i"(tile_type_traits<I32Tile::TileDType>::TilesizeCode),
          [ValidCol] "i"(1), [ValidRow] "i"(kLane), [Cols] "i"(1)
        : "memory");
}

// Strided [32,1] S32 TSTORE: writes src[r] to base + r*strideBytes. With
// base = hist+q and strideBytes = 8*sizeof(int32_t) it lands on hist[8r+q].
inline void tstore_i32_strided(int32_t *base, I32Tile &src,
                               uint32_t strideBytes) {
    asm volatile(
        "BSTART.TLSU TSTORE, %D[DataType]\n"
        "B.DATR M322ND, Null\n"
        "B.DIM zero, %c[ValidCol], ->lb0\n"
        "B.DIM zero, %c[ValidRow], ->lb1\n"
        "B.IOT %[src], mask=1111, last\n"
        "B.IOR [%[base], %[stride]], []\n"
        :
        : [base] "r"(base), [stride] "r"(strideBytes),
          [src] "Tr"(src.data()),
          [DataType] "i"(type_traits<int32_t>::TypeCode),
          [ValidCol] "i"(1), [ValidRow] "i"(kLane)
        : "memory");
}

// Within-group suffix as a parallel pair tree. B.SUBVIEW makes every CELL of
// the parent freely readable, so the four adjacent pairs issue together and
// the running chain is only 3 TADDs deep instead of the 7-deep serial
// "c[q] = cell_q + c[q+1]" chain:
//   p67=X6+X7  p45=X4+X5  p23=X2+X3  p01=X0+X1
//   S5 = X5+p67           S1 = X1+p23
//   s47 = p45+p67         s03 = p01+p23
//   S3 = X3+s47  S2 = p23+s47  S1 += s47  S0 = s03+s47
//   S4 = s47  S6 = p67  S7 = X7
// The eight-cell suffix plus the TSHUF across-group scan are inlined (no
// shared helper) so the c[8] tile array is never address-taken and stays in
// tile registers.
inline void suffix_cumsum(int32_t *hist) {
    GroupTileS32 acc;
    group_tload_832(acc, hist);
    I32Tile c[8];
    // level 1: four independent pair sums
    group_tmuls_cell(c[7], acc, 7, 1);         // S7 = X7
    group_tadd_cell_pair(c[6], acc, 6, 7);     // p67
    group_tadd_cell_pair(c[4], acc, 4, 5);     // p45
    group_tadd_cell_pair(c[2], acc, 2, 3);     // p23
    group_tadd_cell_pair(c[0], acc, 0, 1);     // p01
    // level 2: odd-position tails + one-cell-offset block sums
    group_tadd_cell_tail(c[5], acc, 5, c[6]);  // S5 = X5+p67
    group_tadd_cell_tail(c[1], acc, 1, c[2]);  // q1 = X1+p23
    TADD(c[4], c[4], c[6]);                    // s47 = p45+p67
    TADD(c[0], c[0], c[2]);                    // s03 = p01+p23
    // level 3: fold the right block into every cell
    group_tadd_cell_tail(c[3], acc, 3, c[4]);  // S3 = X3+s47
    TADD(c[2], c[2], c[4]);                    // S2 = p23+s47
    TADD(c[1], c[1], c[4]);                    // S1 = q1+s47
    TADD(c[0], c[0], c[4]);                    // S0 = s03+s47

    // Across-group suffix: 32-lane Hillis-Steele scan of cell_0 (the group
    // totals) with mode-1 TSHUF row shifts, then the shifted-by-one scan is
    // broadcast back to all eight cells.
    const uint64_t kShufMode1Zero = 1ull | (4ull << 8) | (1ull << 16);
    I32Tile G;
    TADDS(G, c[0], 0);
    U32Tile ctrl;
#pragma clang loop unroll(full)
    for (int s = 1; s <= 16; s <<= 1) {
        TEXPANDS(ctrl, static_cast<uint32_t>(s));
        I32Tile sh;
        tshuf_shift_i32(sh, G, ctrl, kShufMode1Zero);
        TADD(G, G, sh);
    }
    TEXPANDS(ctrl, 1u);
    I32Tile S;
    tshuf_shift_i32(S, G, ctrl, kShufMode1Zero);
#pragma clang loop unroll(full)
    for (int q = 0; q < 8; ++q) TADD(c[q], c[q], S);
#pragma clang loop unroll(full)
    for (int q = 0; q < 8; ++q) {
        tstore_i32_strided(hist + q, c[q], 8 * sizeof(int32_t));
    }
}

}  // namespace histogram_cumsum_m32
