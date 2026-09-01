#ifndef VECTOR_BENCH_HPP
#define VECTOR_BENCH_HPP

// TEPL tile compute micro-bench templates.
// Intrinsic naming follows DavinciOO/PTO (OPERATOR_REFERENCE.md):
//   TLOAD/TSTORE for GM<->Tile, all TEPL opcodes by canonical PTO names.
// pto_tileop.hpp is expected to align to these names; until then these sources
// are structural reference and may not compile.

#include <common/pto_tileop.hpp>
#include <cstdint>
#include "benchmark.h"
#include "bench_utils.hpp"

using namespace pto;

// ---- common tile/global aliases ----
template <typename D, int M, int N>
using gm_t = global_tensor<D, RowMajor<M, N>>;
template <typename D, int M, int N>
using tile_t = Tile<Location::Vec, D, M, N, BLayout::RowMajor>;
template <typename D, int M, int N>
using iter_t = global_iterator<gm_t<D, M, N>, tile_t<D, M, N>>;

// dst = op(src0, src1)
template <typename D, int M, int N>
void bench_binary(D *c, D *a, D *b, auto op) {
    iter_t<D, M, N> gA(a), gB(b), gC(c);
    auto gA0 = gA(0, 0);
    auto gB0 = gB(0, 0);
    auto gC0 = gC(0, 0);
    tile_t<D, M, N> tA, tB, tC;
    TLOAD(tA, gA0);
    TLOAD(tB, gB0);
    op(tC, tA, tB);
    TSTORE(gC0, tC);
}

// row-broadcast arith: src0/dst = M×N, src1 = M×K per-row 32B scalar strip
// (K = 32/sizeof(D): fp16->16, fp32->8), per tileop-usage "PTO Mode 2 每行 32B 数据条"
template <typename D, int M, int N>
void bench_expand_row(D *c, D *a, D *b, auto op) {
    using gmB = global_tensor<D, RowMajor<M, 1>>;
    using tileB = Tile<Location::Vec, D, M, 1, BLayout::RowMajor>;
    using itB = global_iterator<gmB, tileB>;
    iter_t<D, M, N> gA(a), gC(c); itB gB(b);
    auto gA0 = gA(0, 0);
    auto gC0 = gC(0, 0);
    auto gB0 = gB(0, 0);
    tile_t<D, M, N> tA, tC; tileB tB;
    TLOAD(tA, gA0);
    TLOAD(tB, gB0);
    op(tC, tA, tB);
    TSTORE(gC0, tC);
}

// col-broadcast arith: src0/dst = M×N, src1 = K×N per-col 32B scalar strip
template <typename D, int M, int N>
void bench_expand_col(D *c, D *a, D *b, auto op) {
    constexpr int K = 32 / sizeof(D);
    using gmB = global_tensor<D, RowMajor<1, N>>;
    using tileB = Tile<Location::Vec, D, K, N, BLayout::RowMajor, 1, N>;
    using itB = global_iterator<gmB, tileB>;
    iter_t<D, M, N> gA(a), gC(c); itB gB(b);
    auto gA0 = gA(0, 0);
    auto gC0 = gC(0, 0);
    auto gB0 = gB(0, 0);
    tile_t<D, M, N> tA, tC; tileB tB;
    TLOAD(tA, gA0);
    TLOAD(tB, gB0);
    op(tC, tA, tB);
    TSTORE(gC0, tC);
}

// concat: src0 = M×K, src1 = M×K, dst = M×(2K); K = 32/sizeof(D) for 32B align
template <typename D, int M>
void bench_concat(D *c, D *a, D *b, auto op) {
    constexpr int K = 32 / sizeof(D);
    using gmA = global_tensor<D, RowMajor<M, K>>;
    using gmB = global_tensor<D, RowMajor<M, K>>;
    using gmC = global_tensor<D, RowMajor<M, 2 * K>>;
    using tileA = Tile<Location::Vec, D, M, K, BLayout::RowMajor>;
    using tileB = Tile<Location::Vec, D, M, K, BLayout::RowMajor>;
    using tileC = Tile<Location::Vec, D, M, 2 * K, BLayout::RowMajor>;
    using itA = global_iterator<gmA, tileA>;
    using itB = global_iterator<gmB, tileB>;
    using itC = global_iterator<gmC, tileC>;
    itA gA(a); itB gB(b); itC gC(c);
    auto gA0 = gA(0, 0);
    auto gB0 = gB(0, 0);
    auto gC0 = gC(0, 0);
    tileA tA; tileB tB; tileC tC;
    TLOAD(tA, gA0);
    TLOAD(tB, gB0);
    op(tC, tA, tB);
    TSTORE(gC0, tC);
}

// dst = op(src0)
template <typename D, int M, int N>
void bench_unary(D *c, D *a, auto op) {
    iter_t<D, M, N> gA(a), gC(c);
    auto gA0 = gA(0, 0), gC0 = gC(0, 0);
    tile_t<D, M, N> tA, tC;
    TLOAD(tA, gA0);
    op(tC, tA);
    TSTORE(gC0, tC);
}

// Copy expansion has only the broadcast source, unlike the binary arithmetic
// expansion forms above. Keep its valid shape distinct from physical padding.
template <typename D, int M, int N>
void bench_expand_copy_row(D *c, D *a, auto op) {
    using gmA = global_tensor<D, RowMajor<M, 1>>;
    using tileA = Tile<Location::Vec, D, M, 1, BLayout::RowMajor>;
    using itA = global_iterator<gmA, tileA>;
    itA gA(a); iter_t<D, M, N> gC(c);
    auto gA0 = gA(0, 0); auto gC0 = gC(0, 0);
    tileA tA; tile_t<D, M, N> tC;
    TLOAD(tA, gA0); op(tC, tA); TSTORE(gC0, tC);
}

template <typename D, int M, int N>
void bench_expand_copy_col(D *c, D *a, auto op) {
    constexpr int K = 32 / sizeof(D);
    using gmA = global_tensor<D, RowMajor<1, N>>;
    using tileA = Tile<Location::Vec, D, K, N, BLayout::RowMajor, 1, N>;
    using itA = global_iterator<gmA, tileA>;
    itA gA(a); iter_t<D, M, N> gC(c);
    auto gA0 = gA(0, 0); auto gC0 = gC(0, 0);
    tileA tA; tile_t<D, M, N> tC;
    TLOAD(tA, gA0); op(tC, tA); TSTORE(gC0, tC);
}

// dst = cond ? src0 : src1. PTO v0.58 uses a uint16 condition tile.
template <typename D, int M, int N>
void bench_select(D *c, D *a, D *b, auto op) {
    using gmMask = global_tensor<uint16_t, RowMajor<M, N>>;
    using tileMask = Tile<Location::Vec, uint16_t, M, N, BLayout::RowMajor>;
    using itMask = global_iterator<gmMask, tileMask>;
    uint16_t cond[M * N];
    for (int i = 0; i < M * N; ++i) cond[i] = (uint16_t)(i & 1);
    iter_t<D, M, N> gA(a), gB(b), gC(c); itMask gCond(cond);
    auto gA0 = gA(0, 0), gB0 = gB(0, 0), gC0 = gC(0, 0);
    auto gM0 = gCond(0, 0);
    tile_t<D, M, N> tA, tB, tC; tileMask tCond;
    TLOAD(tA, gA0);
    TLOAD(tB, gB0);
    TLOAD(tCond, gM0);
    op(tC, tCond, tA, tB);
    TSTORE(gC0, tC);
}

// dst = reduce(src) ; row-reduce -> Mx1 output (ValidCol==1 required by toolchain)
template <typename D, int M, int N>
void bench_reduce(D *c, D *a, auto op) {
    using gmC = global_tensor<D, RowMajor<M, 1>>;
    using tileC = Tile<Location::Vec, D, M, 1, BLayout::RowMajor>;
    using itC = global_iterator<gmC, tileC>;
    iter_t<D, M, N> gA(a); itC gC(c);
    auto gA0 = gA(0, 0), gC0 = gC(0, 0);
    tile_t<D, M, N> tA;
    tileC tC;
    TLOAD(tA, gA0);
    op(tC, tA);
    TSTORE(gC0, tC);
}

// dst = op(src0, scalar)
template <typename D, int M, int N>
void bench_scalar(D *c, D *a, D s, auto op) {
    iter_t<D, M, N> gA(a), gC(c);
    auto gA0 = gA(0, 0), gC0 = gC(0, 0);
    tile_t<D, M, N> tA, tC;
    TLOAD(tA, gA0);
    op(tC, tA, s);
    TSTORE(gC0, tC);
}

// dst = op(src0, src1, scalar)
template <typename D, int M, int N>
void bench_scalar3(D *c, D *a, D *b, D s, auto op) {
    iter_t<D, M, N> gA(a), gB(b), gC(c);
    auto gA0 = gA(0, 0);
    auto gB0 = gB(0, 0);
    auto gC0 = gC(0, 0);
    tile_t<D, M, N> tA, tB, tC;
    TLOAD(tA, gA0);
    TLOAD(tB, gB0);
    op(tC, tA, tB, s);
    TSTORE(gC0, tC);
}

// dst = broadcast(scalar)
template <typename D, int M, int N>
void bench_scalar_bcast(D *c, D s, auto op) {
    iter_t<D, M, N> gC(c);
    auto gC0 = gC(0, 0);
    tile_t<D, M, N> tC;
    op(tC, s);
    TSTORE(gC0, tC);
}

// dst = gather(src, indices)  -- tile-local gather (TGATHERB)
template <typename D, int M, int N>
void bench_gather(D *c, D *a, D *idx, auto op) {
    iter_t<D, M, N> gA(a), gIdx(idx), gC(c);
    auto gA0 = gA(0, 0), gI0 = gIdx(0, 0), gC0 = gC(0, 0);
    tile_t<D, M, N> tA, tIdx, tC;
    TLOAD(tA, gA0);
    TLOAD(tIdx, gI0);
    op(tC, tA, tIdx);
    TSTORE(gC0, tC);
}

// dst = histogram(src, idx, byteId)
template <typename D, int M, int N>
void bench_hist(D *c, D *a, D *idx, int byteId, auto op) {
    iter_t<D, M, N> gA(a), gIdx(idx), gC(c);
    auto gA0 = gA(0, 0), gI0 = gIdx(0, 0), gC0 = gC(0, 0);
    tile_t<D, M, N> tA, tIdx, tC;
    TLOAD(tA, gA0);
    TLOAD(tIdx, gI0);
    op(tC, tA, tIdx, byteId);
    TSTORE(gC0, tC);
}

#endif
