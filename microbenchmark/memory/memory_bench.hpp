#ifndef MEMORY_BENCH_HPP
#define MEMORY_BENCH_HPP

// TLSU data-movement micro-bench templates.
// Intrinsic naming follows DavinciOO/PTO (OPERATOR_REFERENCE.md §7):
//   TLOAD / TSTORE / TMOV / MGATHER / MSCATTER (+ .MASK variants).

#include <common/pto_tileop.hpp>
#include <cstdint>
#include "benchmark.h"
#include "bench_utils.hpp"
#include "timg2col_tileop.hpp"

using namespace pto;

template <typename D, int M, int N>
using gm_t = global_tensor<D, RowMajor<M, N>>;
template <typename D, int M, int N>
using tile_t = Tile<Location::Vec, D, M, N, BLayout::RowMajor>;
template <typename D, int M, int N>
using iter_t = global_iterator<gm_t<D, M, N>, tile_t<D, M, N>>;

// GM -> Tile -> GM
template <typename D, int M, int N>
void bench_load(D *c, D *a) {
    iter_t<D, M, N> gA(a), gC(c);
    auto gA0 = gA(0, 0), gC0 = gC(0, 0);
    tile_t<D, M, N> tA;
    TLOAD(tA, gA0);
    TSTORE(gC0, tA);
}

// GM -> Tile -> TMOV -> Tile -> GM
// NOTE: toolchain exposes Tile<->Tile move as TMOV.
template <typename D, int M, int N>
void bench_mov(D *c, D *a) {
    iter_t<D, M, N> gA(a), gC(c);
    auto gA0 = gA(0, 0), gC0 = gC(0, 0);
    tile_t<D, M, N> tA, tB;
    TLOAD(tA, gA0);
    TMOV(tB, tA);
    TSTORE(gC0, tB);
}

// MGATHER: GM[indices] -> Tile -> GM
template <typename D, int M, int N>
void bench_gather(D *c, D *a, int32_t *idx) {
    using gmIdx = global_tensor<int32_t, RowMajor<M, N>>;
    using tileIdx = Tile<Location::Vec, int32_t, M, N, BLayout::RowMajor>;
    using itIdx = global_iterator<gmIdx, tileIdx>;
    iter_t<D, M, N> gA(a), gC(c); itIdx gIdx(idx);
    auto gA0 = gA(0, 0), gC0 = gC(0, 0);
    auto gI0 = gIdx(0, 0);
    tileIdx tIdx; tile_t<D, M, N> tDst;
    TLOAD(tIdx, gI0);
    MGATHER(tDst, gA0, tIdx);
    TSTORE(gC0, tDst);
}

// MGATHER.MASK
template <typename D, int M, int N>
void bench_gather_mask(D *c, D *a, int32_t *idx, uint16_t *mask) {
    using gmIdx = global_tensor<int32_t, RowMajor<M, N>>;
    using tileIdx = Tile<Location::Vec, int32_t, M, N, BLayout::RowMajor>;
    using gmMask = global_tensor<uint16_t, RowMajor<M, N>>;
    using tileMask = Tile<Location::Vec, uint16_t, M, N, BLayout::RowMajor>;
    using itIdx = global_iterator<gmIdx, tileIdx>;
    using itMask = global_iterator<gmMask, tileMask>;
    iter_t<D, M, N> gA(a), gC(c); itIdx gIdx(idx); itMask gMask(mask);
    auto gA0 = gA(0, 0), gC0 = gC(0, 0);
    auto gI0 = gIdx(0, 0);
    auto gM0 = gMask(0, 0);
    tileIdx tIdx; tileMask tMask; tile_t<D, M, N> tDst;
    TLOAD(tIdx, gI0);
    TLOAD(tMask, gM0);
    MGATHER_MASK(tDst, gA0, tIdx, tMask);
    TSTORE(gC0, tDst);
}

// MSCATTER: Tile -> GM[indices]
template <typename D, int M, int N>
void bench_scatter(D *c, D *a, int32_t *idx) {
    using gmIdx = global_tensor<int32_t, RowMajor<M, N>>;
    using tileIdx = Tile<Location::Vec, int32_t, M, N, BLayout::RowMajor>;
    using itIdx = global_iterator<gmIdx, tileIdx>;
    iter_t<D, M, N> gA(a), gC(c); itIdx gIdx(idx);
    auto gA0 = gA(0, 0), gC0 = gC(0, 0);
    auto gI0 = gIdx(0, 0);
    tile_t<D, M, N> tSrc; tileIdx tIdx;
    TLOAD(tSrc, gA0);
    TLOAD(tIdx, gI0);
    MSCATTER(gC0, tSrc, tIdx);
}

// MSCATTER.MASK
template <typename D, int M, int N>
void bench_scatter_mask(D *c, D *a, int32_t *idx, uint16_t *mask) {
    using gmIdx = global_tensor<int32_t, RowMajor<M, N>>;
    using tileIdx = Tile<Location::Vec, int32_t, M, N, BLayout::RowMajor>;
    using gmMask = global_tensor<uint16_t, RowMajor<M, N>>;
    using tileMask = Tile<Location::Vec, uint16_t, M, N, BLayout::RowMajor>;
    using itIdx = global_iterator<gmIdx, tileIdx>;
    using itMask = global_iterator<gmMask, tileMask>;
    iter_t<D, M, N> gA(a), gC(c); itIdx gIdx(idx); itMask gMask(mask);
    auto gA0 = gA(0, 0), gC0 = gC(0, 0);
    auto gI0 = gIdx(0, 0);
    auto gM0 = gMask(0, 0);
    tile_t<D, M, N> tSrc; tileIdx tIdx; tileMask tMask;
    TLOAD(tSrc, gA0);
    TLOAD(tIdx, gI0);
    TLOAD(tMask, gM0);
    MSCATTER_MASK(gC0, tSrc, tIdx, tMask);
}

// Cache-line prefetch has no Tile destination.
template <typename D, int M, int N>
void bench_prefetch(D *, D *a) {
    gm_t<D, M, N> gA(a);
    TPREFETCH(gA, N, M);
}

// Atomic compare-and-swap gather. Byte displacements select the corresponding
// element in the supplied GM array; expected/replacement are explicit tiles.
template <typename D, int M, int N>
void bench_gather_cas(D *c, D *a, int32_t *idx) {
    using gmIdx = global_tensor<int32_t, RowMajor<M, N>>;
    using tileIdx = Tile<Location::Vec, int32_t, M, N, BLayout::RowMajor>;
    using itIdx = global_iterator<gmIdx, tileIdx>;
    iter_t<D, M, N> gA(a), gC(c); itIdx gIdx(idx);
    auto gA0 = gA(0, 0);
    auto gC0 = gC(0, 0);
    auto gI0 = gIdx(0, 0);
    tile_t<D, M, N> tObserved, tExpected, tReplacement;
    tileIdx tIdx;
    TLOAD(tIdx, gI0);
    TLOAD(tExpected, gA0);
    TLOAD(tReplacement, gA0);
    MGATHER_CAS(tObserved, reinterpret_cast<uint64_t>(a), tIdx,
                tExpected, tReplacement, N, M);
    TSTORE(gC0, tObserved);
}

// Peer-Tile move. The formal operation only accepts packed 4-bit element
// types; peer_tid=0 keeps this as a deterministic single-PE encoding test.
template <typename D, int M, int N>
void bench_gmov(D *c, D *a) {
    iter_t<D, M, N> gA(a), gC(c);
    auto gA0 = gA(0, 0), gC0 = gC(0, 0);
    tile_t<D, M, N> tA, tC;
    TLOAD(tA, gA0);
    GMOV(tC, 0, tA);
    TSTORE(gC0, tC);
}

// A 1x1, stride-1, no-padding IMG2COL is an identity reshape from NHWC
// [1,H,W,C0] to the standard Left matrix [H*W,C0]. Local-CUBE TIMG2COL is a
// four-PE collective: LB1 carries the group-total rows and each PE materializes
// one 16-row CUBE_M16 slice, which is stored to its non-overlapping GM range.
template <typename D, int H, int W>
void bench_img2col_1x1(D *c, D *a) {
    constexpr int C0 = 32 / sizeof(D);
    constexpr int GroupM = H * W;
    constexpr int LocalM = 16;
    static_assert(GroupM == 4 * LocalM,
                  "the cooperative M16 microbenchmark requires 16 rows per PE");
    using GMIn = global_tensor<D, RowMajor<GroupM, C0>>;
    using GMOut = global_tensor<D, RowMajor<LocalM, C0>>;
    using OutTile = CubeTileM16<D, LocalM, C0>;

    const uint32_t pe = get_thread_idx();
    GMIn gA(a);
    GMOut gC(c + pe * LocalM * C0);
    OutTile tC;
    constexpr uint64_t param0 = microbench::PackTIMG2COLParam0(
        H, W, C0, 1, 1);
    constexpr uint64_t param1 = microbench::PackTIMG2COLParam1(
        0, 0, 0, 0, 1, 1, 1, 1);
    constexpr uint64_t param2 = microbench::PackTIMG2COLParam2(0, 0);
    microbench::TIMG2COL_ASM<GroupM>(tC, gA, param0, param1, param2);
    TSTORE_CUBE(gC, tC);
}

#endif
