#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>

// Top-K radix select: FP16 coarse pass + four FP32 byte passes, per
// topk_scatter_atomic_add_tileop.md (same directory).
//
// Semantics: for each batch row bx, select the kTopK largest values of
// input[bx, start:end) and write their in-row element indices to
// output[bx, 0:kTopK]. The output is an unordered set (source note 11).
//
// SPMD: 4 PEs, PE tid handles rows bx = tid, tid+4, ...  All per-row state
// (histogram, ping-pong candidates, chunk scratch) is private to one PE, so
// the listing's cross-thread barriers collapse to program order.
//
// This kernel targets the indexed-TLSU contract of PTO-ISA PR #313
// (issue #301, "Restore indexed TLSU byte displacement and unify CUBE
// layouts", open at time of writing) rather than the superseded v0.58 text:
//  - IndexTile entries are byte displacements, never scaled or decomposed,
//    and B.IOR is BaseGPR-only (RegSrc1/RegSrc2/RegDst zero). gfrun's
//    TMAEngine already executes exactly this form; the toolchain wrappers
//    still emit the legacy nonzero-stride B.IOR, so MGATHER/MSCATTER_MASK
//    are hand-written asm here.
//  - ROWMAJOR/CUBE_M16/CUBE_M32 form one indexed-layout class, so the whole
//    kernel uses CUBE_M32 vector tiles (VecTileM32), including the index and
//    data tiles of the indexed transfers. For these single-cell shapes the
//    M32 payload is byte-identical to RowMajor, which is why current gfrun
//    (layout-agnostic index reads) executes it unchanged.
//  - The MSCATTER_MASK predicate is an ordinary Local U8 tile with canonical
//    0x00/0x01 values and the same 32x1 valid shape as data/index,
//    materialized through GM scratch. A CUBE_M32 U8 cell keeps element
//    (row, 0) at byte row*4, so this requires gfrun's ExecuteMSCATTER_MASK to
//    resolve CUBE operands through the CELL payload index (model repo
//    TMAEngine.cpp; legacy dense reads only matched RowMajor tiles).
//  - PR #310 (CUBE_M16/M32 2D TCI) is not used: index sequences here come
//    from scalar bookkeeping, not TCI.
//
// Remaining model gaps that shape the implementation:
//  - GM atomics (MGATHER_ADD/MSCATTER_ADD, TLSU function 12/21) are still
//    not implemented in gfrun (functions 4..8 only). Histogram accumulation
//    and the output slot allocation (hist[bin+1]++ RMW, including the round-3
//    direct output) therefore run on the PE's scalar core over the
//    PE-private GM histogram; per-chunk bins are produced by tile ops,
//    TSTOREd to scratch, then accumulated scalar. Program order serializes
//    duplicate bins exactly like the atomic RMW. The 256-bin suffix cumsum,
//    the threshold search and the threshold-bin candidate compaction do NOT
//    need atomics and run as tile ops (see suffix_cumsum, find_threshold and
//    compact_candidates below).
//  - The FP16 payload is reinterpreted via a GM round trip
//    (TSTORE fp16 -> TLOAD u16) instead of reinterpret_tile: the model tags
//    each tile register with the dtype of its last writer and validates it
//    on TSTORE (AccumulateBlockInfo.cpp ValidateLocalTlsu); the round trip
//    keeps every block's declared dtype equal to the tag.
//  - find_threshold uses TCMPS/TSELS with the PredicateCell carrier; the
//    source 3.4 GPR-predicate carrier is not implemented in gfrun, and the
//    wrappers emit no CUBE layout selector for these ops. The scan key sign
//    masks still come from bit arithmetic
//    (key = bits ^ (sign ? ~0 : signbit)).

namespace topk_radix {

using namespace pto;

constexpr int kBatch = 4;
constexpr int kCols = 8192;
constexpr int kTopK = 512;
constexpr int kCandCap = 4096;  // candidate workspace bound (source note 12)
constexpr int kLane = 32;

// All vector tiles are CUBE_M32 single-cell column vectors (ValidCol=1,
// ValidRow=32). Sub-32-bit dtypes pad the physical Col to keep 128 B.
using F32Tile = VecTileM32<float, kLane, 1>;
using F16Tile = VecTileM32<__half, kLane, 2, kLane, 1>;
using U16Tile = VecTileM32<uint16_t, kLane, 2, kLane, 1>;
using U32Tile = VecTileM32<uint32_t, kLane, 1>;
using I32Tile = VecTileM32<int32_t, kLane, 1>;
using U8Tile = VecTileM32<uint8_t, kLane, 4, kLane, 1>;

struct Scratch {
    int32_t hist[288];      // [0,256) bins; [256,288) zero pad for the
                            // shifted cumsum window TLOADs, never written
    int32_t num[2];         // ping-pong candidate counts
    int32_t error;          // 0 ok, 1 candidate overflow, 2 empty range
    int32_t reduce[kLane];  // TCOLSUM landing pad for find_threshold
    int32_t lane[kLane];    // constant 0..31 lane-index sequence, TLOAD'd
                            // for the compaction keep mask (a TCI-produced
                            // tile does not carry the <32,1> tileInfo the
                            // compare validator requires)
    uint32_t scan[2 * kLane];  // zero pad + publish area for the candidate
                               // compaction prefix sum; the pad half stays
                               // zero after the per-row clear
    uint16_t bin16[kLane];
    uint32_t word[kLane];
    uint32_t off[kLane];    // byte displacements for the indexed transfers
    int32_t idx[kLane];
    uint8_t msk[kLane];
    int32_t cand[2][kCandCap];
};

inline int32_t clamp_lo(int32_t v, int32_t lo) { return v < lo ? lo : v; }
inline int32_t clamp_hi(int32_t v, int32_t hi) { return v > hi ? hi : v; }
inline int32_t min_i32(int32_t a, int32_t b) { return a < b ? a : b; }

// 256-bin suffix cumsum as pure tile ops, per histogram_cumsum_m32.md
// (same directory): 32 shifted window TLOADs, a 31-TADD
// on-tile binary reduction (32 -> 16 -> ... -> 1), then a 7-step cross-cell
// suffix accumulate from high bins to low bins and 8 cell TSTOREs.
//
// The grouped <8,32> histogram tile (eight S32 M32 CELLs, 1KB) cannot be
// expressed with wrapper tiles: pto_tile.hpp caps CubeM32 Vec tiles at 32
// rows, and its <32,8> alternative transposes the GM<->CELL mapping, which
// breaks the shifted windows. The listing's T112[:,q:q+1] logical range
// operand has no realization either (B.SUBVIEW parents must be Matrix
// tiles; M32 ranges advance along columns). So the grouped tile is bound
// through a raw 1KB linx_tile_carrier with hand-written blocks (mirroring
// the disassembled wrapper forms), and the eight result cells are
// materialized through hist itself: one grouped TSTORE + eight cell
// TLOADs. An S32 <256,1> M32 tile is byte-contiguous (cellColumns=1, so
// payload element e sits at byte 4e), which is exactly what makes the
// shifted-window dataflow work. hist[256:288] is zero padding consumed by
// the shifted windows in place of the listing's valid_col=256-k partial
// fill; it is never written.
//
// Correctness: window k holds hist[k+e] at flattened position e (zero pad
// past 255), so after the reduction T112[e] = sum_{k=0..31} hist[e+k]. Cell
// q of T112 covers bins [32q, 32q+31] and spills into cell q+1's first i
// bins at lane i, which exactly complements the clamped suffix of the
// accumulated higher cells: Rq[i] = T112[q][i] + R(q+1)[i] gives
// hist[bin] = sum_{j=bin..255} original_hist[j]. hist[256] stays 0.
struct GroupTileS32 {
    linx_tile_carrier<1024> carrier;
    linx_tile_carrier<1024>::RegisterType &data() { return carrier.Register; }
};

// Block forms mirror the disassembled wrapper CUBE blocks, with
// ValidRow=256 / 1KB SizeCode for the grouped tile.
inline void group_tload(GroupTileS32 &dst, const int32_t *base) {
    asm volatile(
        "BSTART.TLSU TLOAD, %D[DataType]\n"
        "B.DATR ND2M32, Zero\n"
        "B.DIM zero, %c[ValidCol], ->lb0\n"
        "B.DIM zero, %c[ValidRow], ->lb1\n"
        "B.IOT mask=1111, last, ->%[dst]<%Z[TileSize]>\n"
        "B.IOR [%[base], %[stride]], []\n"
        : [dst] "=Tr"(dst.data())
        : [base] "r"(base), [stride] "r"(sizeof(int32_t)),
          [DataType] "i"(type_traits<int32_t>::TypeCode),
          [TileSize] "i"(tile_type_traits<
                           linx_tile_carrier<1024>>::TilesizeCode),
          [ValidCol] "i"(1), [ValidRow] "i"(8 * kLane)
        : "memory");
}

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

inline void group_tadd(GroupTileS32 &dst, GroupTileS32 &a, GroupTileS32 &b) {
    asm volatile(
        "BSTART.TEPL 0, %D[DataType]\n"
        "B.DATR CUBE_M32, Null\n"
        "B.DIM zero, %c[ValidCol], ->lb0\n"
        "B.DIM zero, %c[ValidRow], ->lb1\n"
        "B.DIM zero, %c[Col], ->lb2\n"
        "B.IOT %[a], %[b], mask=1111, last, ->%[dst]<%Z[TileSize]>\n"
        : [dst] "=Tr"(dst.data())
        : [a] "Tr"(a.data()), [b] "Tr"(b.data()),
          [DataType] "i"(type_traits<int32_t>::TypeCode),
          [TileSize] "i"(tile_type_traits<
                           linx_tile_carrier<1024>>::TilesizeCode),
          [ValidCol] "i"(1), [ValidRow] "i"(8 * kLane), [Col] "i"(1)
        : "memory");
}

// find_threshold helpers, same raw-carrier convention as the cumsum blocks.
// TCMPS (TEPL 45) compares against a scalar into a PredicateCell destination;
// TEXPANDS (TEPL 59) broadcasts a scalar; TSELS (TEPL 58) selects tile/scalar
// under a predicate; TCOLSUM (TEPL 80) folds the 256 rows into one. Block
// forms mirror the disassembled wrappers (CUBE_M32 elementwise DATR, no DATR
// for TSELS, PadValue+CMode DATR for TCMPS).
inline void group_tcmps_ge(GroupTileS32 &pred, GroupTileS32 &src,
                           int32_t scalar) {
    asm volatile(
        "BSTART.TEPL 45, %D[DataType]\n"
        "B.DATR Zero, GE\n"
        "B.DIM zero, %c[ValidCol], ->lb0\n"
        "B.DIM zero, %c[ValidRow], ->lb1\n"
        "B.DIM zero, %c[Col], ->lb2\n"
        "B.IOT %[src], mask=1111, last, ->%[pred]<%Z[TileSize]>\n"
        "B.IOR [%[scalar]],[]\n"
        : [pred] "=Tr"(pred.data())
        : [src] "Tr"(src.data()), [scalar] "r"(scalar),
          [DataType] "i"(type_traits<int32_t>::TypeCode),
          [TileSize] "i"(tile_type_traits<
                           linx_tile_carrier<1024>>::TilesizeCode),
          [ValidCol] "i"(1), [ValidRow] "i"(8 * kLane), [Col] "i"(1)
        : "memory");
}

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

inline void group_tsels(GroupTileS32 &dst, GroupTileS32 &pred,
                        GroupTileS32 &src, int32_t scalar) {
    asm("" : "+r"(scalar));
    asm volatile(
        "BSTART.TEPL 58, %D[DataType]\n"
        "B.DIM zero, %c[ValidCol], ->lb0\n"
        "B.DIM zero, %c[ValidRow], ->lb1\n"
        "B.DIM zero, %c[Col], ->lb2\n"
        "B.IOT %[pred], %[src], mask=1111, last, ->%[dst]<%Z[TileSize]>\n"
        "B.IOR [%[scalar]],[]\n"
        : [dst] "=Tr"(dst.data())
        : [pred] "Tr"(pred.data()), [src] "Tr"(src.data()),
          [scalar] "r"(scalar),
          [DataType] "i"(type_traits<int32_t>::TypeCode),
          [TileSize] "i"(tile_type_traits<
                           linx_tile_carrier<1024>>::TilesizeCode),
          [ValidCol] "i"(1), [ValidRow] "i"(8 * kLane), [Col] "i"(1)
        : "memory");
}

inline void group_tcolsum(I32Tile &dst, GroupTileS32 &src) {
    asm volatile(
        "BSTART.TEPL 80, %D[DataType]\n"
        "B.DATR CUBE_M32, Null\n"
        "B.DIM zero, %c[ValidCol], ->lb0\n"
        "B.DIM zero, %c[ValidRow], ->lb1\n"
        "B.DIM zero, %c[Col], ->lb2\n"
        "B.IOT %[src], mask=1111, last, ->%[dst]<%Z[DstSize]>\n"
        : [dst] "=Tr"(dst.data())
        : [src] "Tr"(src.data()),
          [DataType] "i"(type_traits<int32_t>::TypeCode),
          [DstSize] "i"(tile_type_traits<I32Tile::TileDType>::TilesizeCode),
          [ValidCol] "i"(1), [ValidRow] "i"(8 * kLane), [Col] "i"(1)
        : "memory");
}

inline void suffix_cumsum(int32_t *hist) {
        // Tile arrays only stay in registers when every index is constant, so
        // all loops are fully unrolled (dynamic indexing demotes the array to
        // the stack and each tile access becomes a 1KB spill/reload).
        GroupTileS32 w[32];
    #pragma clang loop unroll(full)
        for (int k = 0; k < 32; ++k) group_tload(w[k], hist + k);
    #pragma clang loop unroll(full)
        for (int p = 0; p < 16; ++p) group_tadd(w[p], w[2 * p], w[2 * p + 1]);
    #pragma clang loop unroll(full)
        for (int p = 0; p < 8; ++p) group_tadd(w[p], w[2 * p], w[2 * p + 1]);
    #pragma clang loop unroll(full)
        for (int p = 0; p < 4; ++p) group_tadd(w[p], w[2 * p], w[2 * p + 1]);
    #pragma clang loop unroll(full)
        for (int p = 0; p < 2; ++p) group_tadd(w[p], w[2 * p], w[2 * p + 1]);
        group_tadd(w[0], w[0], w[1]);
    
        group_tstore(hist, w[0]);
        I32Tile c[8];
    #pragma clang loop unroll(full)
        for (int q = 0; q < 8; ++q) {
            global_tensor<int32_t, RowMajor<kLane, 1>> gq(hist + kLane * q);
            TLOAD(c[q], gq);
        }
    #pragma clang loop unroll(full)
        for (int q = 6; q >= 0; --q) TADD(c[q], c[q], c[q + 1]);
    #pragma clang loop unroll(full)
        for (int q = 0; q < 8; ++q) {
            global_tensor<int32_t, RowMajor<kLane, 1>> gq(hist + kLane * q);
            TSTORE(gq, c[q]);
    }
}

// Zero a GM word range with tile ops instead of scalar stores: one broadcast
// zero tile, then grouped 1KB TSTOREs plus a 32-word cell TSTORE tail.
// `words` must be a multiple of kLane.
inline void tile_zero_i32(int32_t *base, int32_t words) {
    GroupTileS32 z;
    group_texpands(z, 0);
    int32_t off = 0;
    for (; off + 8 * kLane <= words; off += 8 * kLane) {
        group_tstore(base + off, z);
    }
    if (off < words) {
        I32Tile c;
        TEXPANDS(c, 0);
        global_tensor<int32_t, RowMajor<kLane, 1>> g(base + off);
        TSTORE(g, c);
    }
}

// Threshold search as tile ops (source 3.4). The source uses the TCMPS CUBE
// GPR-predicate carrier (per-64-bin predicate bits merged in scalar GPRs with
// AND/CTZ); gfrun only implements the PredicateCell carrier, so the crossing
// is counted on tiles instead: TCMPS forms the hist[bin] >= rem predicate,
// TSELS materializes it as 0/1, TCOLSUM sums it, and one cell TSTORE returns
// the count to the scalar core. hist is non-increasing after suffix_cumsum
// with hist[0] = n >= rem, so
//   thr = #{bin : hist[bin] >= rem} - 1
// is exactly the first bin with hist[bin] >= rem && hist[bin+1] < rem (the
// source's no-crossing default of 0 is unreachable: bin 0 always counts).
inline int find_threshold(Scratch &sc, int32_t rem) {
    GroupTileS32 h, ones, pred, cond;
    I32Tile sum;
    group_tload(h, sc.hist);
    group_texpands(ones, 1);
    group_tcmps_ge(pred, h, rem);
    group_tsels(cond, pred, ones, 0);
    group_tcolsum(sum, cond);
    global_tensor<int32_t, RowMajor<kLane, 1>> gred(sc.reduce);
    TSTORE(gred, sum);
    return sc.reduce[0] - 1;
}

// PR #313 form of MGATHER: byte-displacement IndexTile, BaseGPR-only B.IOR,
// CUBE_M32 layout class. The toolchain wrapper still emits the legacy
// nonzero-stride B.IOR, hence hand-written asm (operand style mirrors
// mxquant.hpp's hand-written TCVT).
inline void mgather_u32_m32(U32Tile &dst, const void *base, U32Tile &index) {
    asm volatile(
        "BSTART.TLSU MGATHER, %D[DataType]\n"
        "B.DATR CUBE_M32, Zero\n"
        "B.DIM zero, %c[ValidCol], ->lb0\n"
        "B.DIM zero, %c[ValidRow], ->lb1\n"
        "B.DIM zero, %c[Col], ->lb2\n"
        "B.IOT %[off], mask=1111, last, ->%[dst]<%Z[TileSize]>\n"
        "B.IOR [%[base]], []\n"
        : [dst] "=Tr"(dst.data())
        : [base] "r"(base), [off] "Tr"(index.data()),
          [DataType] "i"(type_traits<uint32_t>::TypeCode),
          [TileSize] "i"(U32Tile::TilesizeCode),
          [ValidCol] "i"(U32Tile::ValidCol),
          [ValidRow] "i"(U32Tile::ValidRow),
          [Col] "i"(U32Tile::Cols)
        : "memory");
}

// PR #313 form of MSCATTER_MASK: DataTile/IndexTile/MaskTile share the
// CUBE_M32 layout class, U8 canonical 0/1 predicate, BaseGPR-only B.IOR.
inline void mscatter_mask_i32_m32(void *base, I32Tile &data, U32Tile &index,
                                  U8Tile &mask) {
    asm volatile(
        "BSTART.TLSU MSCATTER.MASK, %D[DataType]\n"
        "B.DATR CUBE_M32, Zero\n"
        "B.DIM zero, %c[ValidCol], ->lb0\n"
        "B.DIM zero, %c[ValidRow], ->lb1\n"
        "B.DIM zero, %c[Col], ->lb2\n"
        "B.IOT %[src], %[off], mask=1111\n"
        "B.IOT %[msk], mask=1111, last\n"
        "B.IOR [%[base]], []\n"
        :
        : [base] "r"(base), [src] "Tr"(data.data()),
          [off] "Tr"(index.data()), [msk] "Tr"(mask.data()),
          [DataType] "i"(type_traits<int32_t>::TypeCode),
          [ValidCol] "i"(I32Tile::ValidCol),
          [ValidRow] "i"(I32Tile::ValidRow),
          [Col] "i"(I32Tile::Cols)
        : "memory");
}

// Write the masked lanes of sc.idx to out[sc.off/4] via MSCATTER_MASK.
// sc.off holds byte displacements (PR #313: never scaled by the hardware).
inline void scatter_masked(int32_t *out, Scratch &sc) {
    bool any = false;
    for (int l = 0; l < kLane; ++l) any |= sc.msk[l] != 0;
    if (!any) return;
    global_tensor<uint32_t, RowMajor<kLane, 1>> goff(sc.off);
    global_tensor<int32_t, RowMajor<kLane, 1>> gidx(sc.idx);
    global_tensor<uint8_t, RowMajor<kLane, 1>> gmsk(sc.msk);
    U32Tile off;
    I32Tile idx;
    U8Tile msk;
    TLOAD(off, goff);
    TLOAD(idx, gidx);
    TLOAD(msk, gmsk);
    mscatter_mask_i32_m32(out, idx, off, msk);
}

// Hillis-Steele inclusive scan of acc over 32 lanes, published through
// scan[0:2*kLane]: the pad half scan[0:kLane] stays zero, so windows shifted
// past lane 0 read zeros. After the final TSTORE the inclusive totals live
// in scan[kLane:2*kLane], so one TLOAD from scan+kLane-1 then yields the
// exclusive scan and scan[2*kLane-1] is the grand total. Fully unrolled: a
// loop-carried tile can make the register allocator emit TLSU TMOVs at the
// back-edge, and gfrun rejects those.
inline __attribute__((always_inline)) void scan_publish_u32(U32Tile &acc, uint32_t *scan) {
    global_tensor<uint32_t, RowMajor<kLane, 1>> g(scan + kLane);
    TSTORE(g, acc);
#pragma clang loop unroll(full)
    for (uint32_t s = 1; s < kLane; s <<= 1) {
        global_tensor<uint32_t, RowMajor<kLane, 1>> gs(scan + kLane - s);
        U32Tile w;
        TLOAD(w, gs);
        TADD(acc, acc, w);
        TSTORE(g, acc);
    }
}

// Append the lanes whose bin equals thr to cand, compacted in lane order,
// as tile ops: eq predicate -> 0/1 mask -> exclusive prefix sum for the
// byte displacements -> MSCATTER_MASK. Compaction needs no GM atomics; the
// prefix sum replaces the scalar num++ per append, and MSCATTER_MASK only
// writes where the mask is set, so the compacted positions must (and do)
// come from the scan. Overflow keeps the source semantics: flag sc.error
// and drop the chunk's appends by retargeting the scatter to the idx sink
// with a zero bias (test inputs never overflow). Returns the chunk's
// candidate count.
//
// The body is deliberately straight-line and keeps no tile alive across the
// caller's chunk-loop back-edge: control-flow-dependent tile live ranges
// make the register allocator emit TLSU TMOVs that gfrun rejects (Local TMOV
// legality). All tile dims are the static 32-lane shape — gfrun's compare/
// select validator requires the block dims to equal the tile metadata, so a
// runtime vc in lb1 is rejected; the tail is masked explicitly instead by a
// keep predicate (lane < vc) folded into the 0/1 mask with TMUL. mu8 is
// converted before the publish consumes m.
inline __attribute__((always_inline)) int32_t compact_candidates(int32_t *cand, U32Tile &bins, I32Tile &data,
                                  uint32_t thr, int32_t vc, int32_t num,
                                  Scratch &sc) {
    I32Tile lane;
    global_tensor<int32_t, RowMajor<kLane, 1>> gl(sc.lane);
    TLOAD(lane, gl);  // 0..31, well-formed <32,1> tileInfo
    U32Tile pred_eq, pred_keep, ones, m_eq, m_keep, m;
    TCMPS<CmpMode::EQ>(pred_eq, bins, thr);    // bin == thr
    TCMPS<CmpMode::LT>(pred_keep, lane, vc);   // valid-lane keep mask
    TEXPANDS(ones, 1u);
    TSELS(m_eq, pred_eq, 0u, ones);            // 1 where eq, else 0
    TSELS(m_keep, pred_keep, 0u, ones);        // 1 in valid lanes, else 0
    TMUL(m, m_eq, m_keep);                     // 0/1 mask, tail zeroed
    U8Tile mu8;
    TCVT(mu8, m);
    scan_publish_u32(m, sc.scan);
    const int32_t count = static_cast<int32_t>(sc.scan[2 * kLane - 1]);
    const bool ok = num + count <= kCandCap;
    if (!ok) sc.error = 1;  // scalar-only branch; no tile crosses the join
    int32_t *dst = ok ? cand : sc.idx;
    const uint32_t bias = ok ? static_cast<uint32_t>(num) * 4u : 0u;
    U32Tile off;
    global_tensor<uint32_t, RowMajor<kLane, 1>> ge(sc.scan + kLane - 1);
    TLOAD(off, ge);  // exclusive prefix sums
    TMULS(off, off, 4u);
    TADDS(off, off, bias);
    mscatter_mask_i32_m32(dst, data, off, mu8);  // all-zero mask: no writes
    return count;
}

// Stage 1: bin = high 8 bits of the FP16 sortable key of each input element.
// key16 = bits16 ^ (sign ? 0xFFFF : 0x8000), bin = key16 >> 8.
// collect=false: histogram. collect=true: bin>thr -> out slot hist[bin+1]++
// (scalar, needs the atomic RMW); bin==thr -> append to cand[0], compacted
// by compact_candidates above.
inline void stage1_scan(const float *row, int32_t base0, int32_t n, bool collect,
                        int32_t thr, int32_t *out, Scratch &sc) {
    for (int32_t ch = 0; ch * kLane < n; ++ch) {
        const int32_t base = base0 + ch * kLane;
        const int32_t vc = min_i32(kLane, n - ch * kLane);

        global_tensor<float, RowMajor<kLane, 1>> gsrc(row + base);
        F32Tile f32;
        TLOAD(f32, gsrc);
        F16Tile f16;
        TCVT(f16, f32);  // RNE
        global_tensor<__half, RowMajor<kLane, 1>> gh(reinterpret_cast<__half *>(sc.bin16));
        TSTORE(gh, f16);  // dtype tag stays FP16: legal store
        global_tensor<uint16_t, RowMajor<kLane, 1>> gu(sc.bin16);
        U16Tile bits, sign, scaled, mask, key, bin;
        TLOAD(bits, gu);  // same payload, u16 tag from here on
        TSHRS(sign, bits, static_cast<uint16_t>(15));
        TMULS(scaled, sign, static_cast<uint16_t>(0x7FFF));
        TADDS(mask, scaled, static_cast<uint16_t>(0x8000));
        TXOR(key, bits, mask);
        TSHRS(bin, key, static_cast<uint16_t>(8));
        TSTORE(gu, bin);

        if (collect) {
            U32Tile bin32;
            TCVT(bin32, bin);
            I32Tile data;
            {   // in_idx = base + lane, from the TLOAD'd lane sequence so
                // the tile carries well-formed <32,1> tileInfo for MSCATTER
                I32Tile lane;
                global_tensor<int32_t, RowMajor<kLane, 1>> gl(sc.lane);
                TLOAD(lane, gl);
                TADDS(data, lane, base);
            }
            sc.num[0] += compact_candidates(sc.cand[0], bin32, data,
                                            static_cast<uint32_t>(thr), vc,
                                            sc.num[0], sc);
        }

        bool any = false;
        for (int32_t lane = 0; lane < kLane; ++lane) {
            sc.msk[lane] = 0;
            if (lane >= vc) continue;
            const int32_t b = sc.bin16[lane];
            if (!collect) {
                sc.hist[b]++;
                continue;
            }
            if (b > thr) {
                sc.off[lane] = static_cast<uint32_t>(sc.hist[b + 1]++) * 4u;
                sc.idx[lane] = base + lane;
                sc.msk[lane] = 1;
                any = true;
            }
        }
        if (collect && any) scatter_masked(out, sc);
    }
}

// Stage 2 round: byte (24-8*round) of the FP32 sortable key of each candidate.
// key32 = bits ^ (sign ? 0xFFFFFFFF : 0x80000000); sign = bits >> 31.
// collect=false: histogram. collect=true: bin>thr -> out slot prefix+hist[bin+1]++
// (scalar, needs the atomic RMW); bin==thr -> next-round candidate for rounds
// 0..2 (compacted by compact_candidates) or, on round 3, direct output while
// pos < kTopK (scalar, also a hist RMW).
inline void round_scan(const float *row, int round, bool collect, int32_t thr,
                       int32_t prefix, int32_t *out, Scratch &sc) {
    const int r = round & 1;
    const int nr = r ^ 1;
    const uint32_t shift = static_cast<uint32_t>(24 - 8 * round);
    const int32_t count = sc.num[r];
    for (int32_t t = 0; t * kLane < count; ++t) {
        const int32_t vc = min_i32(kLane, count - t * kLane);

        global_tensor<uint32_t, RowMajor<kLane, 1>> gcand(
            reinterpret_cast<const uint32_t *>(&sc.cand[r][t * kLane]));
        U32Tile cand, off, bits, sign, scaled, mask, key, shf, byte;
        TLOAD(cand, gcand);
        TMULS(off, cand, 4u);  // byte displacement per PR #313
        mgather_u32_m32(bits, row, off);
        TSHRS(sign, bits, 31u);
        TMULS(scaled, sign, 0x7FFFFFFFu);
        TADDS(mask, scaled, 0x80000000u);
        TXOR(key, bits, mask);
        TSHRS(shf, key, shift);
        TANDS(byte, shf, 0xFFu);
        global_tensor<uint32_t, RowMajor<kLane, 1>> gw(sc.word);
        TSTORE(gw, byte);

        if (collect && round < 3) {
            global_tensor<int32_t, RowMajor<kLane, 1>> gcd(
                &sc.cand[r][t * kLane]);
            I32Tile data;
            TLOAD(data, gcd);  // candidate indices, S32 tag for MSCATTER
            sc.num[nr] += compact_candidates(sc.cand[nr], byte, data,
                                             static_cast<uint32_t>(thr), vc,
                                             sc.num[nr], sc);
        }

        bool any = false;
        for (int32_t lane = 0; lane < kLane; ++lane) {
            sc.msk[lane] = 0;
            if (lane >= vc) continue;
            const int32_t b = static_cast<int32_t>(sc.word[lane]);
            if (!collect) {
                sc.hist[b]++;
                continue;
            }
            if (b > thr) {
                const int32_t pos = prefix + sc.hist[b + 1]++;
                sc.off[lane] = static_cast<uint32_t>(pos) * 4u;
                sc.idx[lane] = sc.cand[r][t * kLane + lane];
                sc.msk[lane] = 1;
                any = true;
            } else if (b == thr && round == 3) {
                const int32_t pos = prefix + sc.hist[b + 1]++;
                if (pos < kTopK) {
                    sc.off[lane] = static_cast<uint32_t>(pos) * 4u;
                    sc.idx[lane] = sc.cand[r][t * kLane + lane];
                    sc.msk[lane] = 1;
                    any = true;
                }
            }
        }
        if (collect && any) scatter_masked(out, sc);
    }
}

inline void run(int32_t *output, int32_t *errors, const float *input,
                const int32_t *starts, const int32_t *ends, Scratch *scratches) {
    const uint32_t tid = get_thread_idx();
    if (tid >= 4) return;

    for (int bx = static_cast<int>(tid); bx < kBatch; bx += 4) {
        Scratch &sc = scratches[bx];
        const float *row = input + bx * kCols;
        int32_t *out = output + bx * kTopK;

        tile_zero_i32(sc.hist, 288);
        sc.num[0] = 0;
        sc.num[1] = 0;
        sc.error = 0;
        tile_zero_i32(sc.cand[0], kCandCap);
        tile_zero_i32(sc.cand[1], kCandCap);
        tile_zero_i32(reinterpret_cast<int32_t *>(sc.scan), 2 * kLane);
        for (int32_t l = 0; l < kLane; ++l) sc.lane[l] = l;

        const int32_t x2 = clamp_lo(starts[bx], 0);
        const int32_t x3 = clamp_hi(ends[bx], kCols);
        const int32_t n = clamp_lo(x3 - x2, 0);
        if (n < kTopK) {  // degenerate range; test inputs never take this path
            sc.error = 2;
            errors[bx] = sc.error;
            continue;
        }

        int32_t rem = kTopK;
        stage1_scan(row, x2, n, false, 0, out, sc);
        suffix_cumsum(sc.hist);
        int32_t thr = find_threshold(sc, rem);
        rem -= sc.hist[thr + 1];
        stage1_scan(row, x2, n, true, thr, out, sc);

        for (int round = 0; round < 4 && rem > 0; ++round) {
            const int nr = (round & 1) ^ 1;
            const int32_t prefix = kTopK - rem;
            tile_zero_i32(sc.hist, 288);
            sc.num[nr] = 0;
            round_scan(row, round, false, 0, 0, out, sc);
            suffix_cumsum(sc.hist);
            thr = find_threshold(sc, rem);
            rem -= sc.hist[thr + 1];
            round_scan(row, round, true, thr, prefix, out, sc);
        }
        errors[bx] = sc.error;
    }
}

}  // namespace topk_radix
