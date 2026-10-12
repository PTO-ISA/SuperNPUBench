#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>

// HISTOGRAM_CUMSUM_M32: in-place 256-bin suffix sum
//     hist[i] = sum_{j = i .. 255} hist_orig[j]
// on a per-PE GM histogram, as tile ops.
//
// Layout: the 256 S32 bins are one 1KB S32 CUBE_M32 grouped tile. A single
// grouped ND2M32 TLOAD packs the dense stream so that cell_q[r] = bin[8r+q];
// the eight physical CELLs sit along the logical columns and are addressable
// with B.SUBVIEW. hist[256:384] is the zero pad the load reads past the
// sentinel.
//
// Dataflow:
//   1. TLOAD                    one 1KB ND2M32 load of the 256 bins (API)
//   2. within-group suffix      parallel pair tree of binary TADDs over
//                               B.SUBVIEW CELL reads (depth 3); cell_q becomes
//                               sum_{j>=q} bin[8r+j], cell_0 the group total
//   3. across-group suffix      a 32-lane TSHUF scan of cell_0 (API), then the
//                               shifted-by-one scan added back to all cells
//   4. eight strided TSTOREs    scatter cell_q[r] -> hist[8r+q]
//
// Status: the grouped load/store/zero-fill (TLOAD / TSTORE / TEXPANDS) and the
// TSHUF scan go through the TileOP API.  Two hand-written blocks remain, both
// blocked on TileOP API gaps (LinxISA/Linx-TileOP-API#211):
//   * the B.SUBVIEW CELL reads in step 2 -- TEPL elementwise does not accept
//     subview sources (only TOR_ASS does);
//   * the step-4 strided stores -- the intended destination-side B.ASSEMBLE
//     collapse needs an INIT-capable *computed* producer, which does not exist
//     (TEPL _ASS cannot write the INIT slot; llvm-project#103 is fixed but not
//     sufficient).  Until then step 4 keeps the eight strided [32,1] stores,
//     which dominate the kernel wall time.

namespace histogram_cumsum_m32 {

using namespace pto;

constexpr int kLane = 32;

using I32Tile = VecTileM32<int32_t, kLane, 1>;
using U32Tile = VecTileM32<uint32_t, kLane, 1>;

// Typed 1KB CUBE_M32 parent (8 cells): the 256 S32 bins as one grouped tile.
using GroupTileS32 = VecTileM32<int32_t, kLane, 8>;

// One 1KB grouped ND2M32 load: cell_q[r] = bin[8r+q], i.e. the eight physical
// CELLs sit along the 8 logical columns so B.SUBVIEW can address CELL q.
inline void group_tload_832(GroupTileS32 &dst, const int32_t *base) {
    global_tensor<int32_t, RowMajor<kLane, 8>> g(base);
    TLOAD(dst, g);
}

// Grouped contiguous store of the 1KB tile: writes the 256 words flat.
inline void group_tstore(int32_t *base, GroupTileS32 &src) {
    global_tensor<int32_t, RowMajor<kLane, 8>> g(base);
    TSTORE(g, src);
}

// Broadcast a scalar into the grouped 1KB tile (topk's GM zeroing helper).
inline void group_texpands(GroupTileS32 &dst, int32_t scalar) {
    TEXPANDS(dst, scalar);
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

// TSHUF row shift. mode 1 shifts lane r <- lane r+b inside each segment of
// segmentWidth = 2 << segmentCode lanes; boundary != 0 zero-fills the
// out-of-range tail. control = mode | (segmentCode<<8) | (boundary<<16).
inline void tshuf_shift_i32(I32Tile &dst, I32Tile &src, U32Tile &controls,
                            uint64_t control) {
    TSHUF(dst, src, controls, control);
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
