#ifndef MEMORY_BENCH_HPP
#define MEMORY_BENCH_HPP

// TLSU data-movement micro-bench templates.
// Intrinsic naming follows DavinciOO/PTO (OPERATOR_REFERENCE.md §7):
//   TLOAD / TSTORE / TMOV / MGATHER / MSCATTER (+ .MASK variants).

#include <common/pto_tileop.hpp>
#include <cstdint>
#include "benchmark.h"
#include "bench_utils.hpp"

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

// GM -> Tile -> GM with layout conversion (ND2NZ)
template <typename D, int M, int N>
void bench_load_layout(D *c, D *a) {
    iter_t<D, M, N> gA(a), gC(c);
    auto gA0 = gA(0, 0), gC0 = gC(0, 0);
    tile_t<D, M, N> tA;
    TLOAD(tA, gA0);  // B.DATR Layout.ND2NZ handled by tile layout attribute
    TSTORE(gC0, tA);
}

// GM -> Tile -> TMOV -> Tile -> GM
// NOTE: toolchain exposes Tile<->Tile move as TCVT(not TMOV yet); using TCTV.
template <typename D, int M, int N>
void bench_mov(D *c, D *a) {
    iter_t<D, M, N> gA(a), gC(c);
    auto gA0 = gA(0, 0), gC0 = gC(0, 0);
    tile_t<D, M, N> tA, tB;
    TLOAD(tA, gA0);
    TCVT(tB, tA);
    TSTORE(gC0, tB);
}

// MGATHER: GM[indices] -> Tile -> GM
template <typename D, int M, int N>
void bench_gather(D *c, D *a, D *idx) {
    iter_t<D, M, N> gA(a), gIdx(idx), gC(c);
    auto gA0 = gA(0, 0), gI0 = gIdx(0, 0), gC0 = gC(0, 0);
    tile_t<D, M, N> tIdx, tDst;
    TLOAD(tIdx, gI0);
    MGATHER(tDst, gA0, tIdx);
    TSTORE(gC0, tDst);
}

// MGATHER.MASK
template <typename D, int M, int N>
void bench_gather_mask(D *c, D *a, D *idx, D *mask) {
    iter_t<D, M, N> gA(a), gIdx(idx), gMask(mask), gC(c);
    auto gA0 = gA(0, 0), gI0 = gIdx(0, 0), gM0 = gMask(0, 0), gC0 = gC(0, 0);
    tile_t<D, M, N> tIdx, tMask, tDst;
    TLOAD(tIdx, gI0);
    TLOAD(tMask, gM0);
    MGATHER_MASK(tDst, gA0, tIdx, tMask);
    TSTORE(gC0, tDst);
}

// MSCATTER: Tile -> GM[indices]
template <typename D, int M, int N>
void bench_scatter(D *c, D *a, D *idx) {
    iter_t<D, M, N> gA(a), gIdx(idx), gC(c);
    auto gA0 = gA(0, 0), gI0 = gIdx(0, 0), gC0 = gC(0, 0);
    tile_t<D, M, N> tSrc, tIdx;
    TLOAD(tSrc, gA0);
    TLOAD(tIdx, gI0);
    MSCATTER(gC0, tSrc, tIdx);
}

// MSCATTER.MASK
template <typename D, int M, int N>
void bench_scatter_mask(D *c, D *a, D *idx, D *mask) {
    iter_t<D, M, N> gA(a), gIdx(idx), gMask(mask), gC(c);
    auto gA0 = gA(0, 0), gI0 = gIdx(0, 0), gM0 = gMask(0, 0), gC0 = gC(0, 0);
    tile_t<D, M, N> tSrc, tIdx, tMask;
    TLOAD(tSrc, gA0);
    TLOAD(tIdx, gI0);
    TLOAD(tMask, gM0);
    MSCATTER_MASK(gC0, tSrc, tIdx, tMask);
}

#endif
