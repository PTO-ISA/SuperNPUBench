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

// dst = op(src0)
template <typename D, int M, int N>
void bench_unary(D *c, D *a, auto op) {
    iter_t<D, M, N> gA(a), gC(c);
    auto gA0 = gA(0, 0);
    auto gC0 = gC(0, 0);
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

// dst = predicate ? src_true : prior_dst. TSEL consumes the packed predicate
// produced by TCMP; it is not a normal uint16 data tile.
template <typename D, int M, int N>
void bench_select(D *c, D *a, D *b, auto op) {
    iter_t<D, M, N> gA(a), gB(b), gC(c);
    auto gA0 = gA(0, 0), gB0 = gB(0, 0), gC0 = gC(0, 0);
    tile_t<D, M, N> tA, tB, tC, tPred;
    TLOAD(tA, gA0);
    TLOAD(tB, gB0);
    TLOAD(tC, gB0); // explicit false source / prior destination
    TCMP<CmpMode::GT>(tPred, tA, tB);
    op(tC, tPred, tA);
    TSTORE(gC0, tC);
}

// dst = predicate ? src_true : scalar_false.
template <typename D, int M, int N>
void bench_select_scalar(D *c, D *a, D *b, D scalar_false, auto op) {
    iter_t<D, M, N> gA(a), gB(b), gC(c);
    auto gA0 = gA(0, 0), gB0 = gB(0, 0), gC0 = gC(0, 0);
    tile_t<D, M, N> tA, tB, tC, tPred;
    TLOAD(tA, gA0);
    TLOAD(tB, gB0);
    TCMP<CmpMode::GT>(tPred, tA, tB);
    op(tC, tPred, scalar_false, tA);
    TSTORE(gC0, tC);
}

// dst = src0 * src1 + src2.
template <typename D, int M, int N>
void bench_fma(D *c, D *a, D *b, D *d) {
    iter_t<D, M, N> gA(a), gB(b), gD(d), gC(c);
    auto gA0 = gA(0, 0), gB0 = gB(0, 0), gD0 = gD(0, 0), gC0 = gC(0, 0);
    tile_t<D, M, N> tA, tB, tD, tC;
    TLOAD(tA, gA0); TLOAD(tB, gB0); TLOAD(tD, gD0);
    TFMA(tC, tA, tB, tD);
    TSTORE(gC0, tC);
}

// Row reduction: source MxN, destination Mx1.
template <typename D, int M, int N>
void bench_reduce_row(D *c, D *a, auto op) {
    using gmC = global_tensor<D, RowMajor<M, 1>>;
    using tileC = Tile<Location::Vec, D, M, 1, BLayout::RowMajor>;
    using itC = global_iterator<gmC, tileC>;
    iter_t<D, M, N> gA(a); itC gC(c);
    auto gA0 = gA(0, 0);
    auto gC0 = gC(0, 0);
    tile_t<D, M, N> tA;
    tileC tC;
    TLOAD(tA, gA0);
    op(tC, tA);
    TSTORE(gC0, tC);
}

// Column reduction: source MxN, destination 1xN.
template <typename D, int M, int N>
void bench_reduce_col(D *c, D *a, auto op) {
    using gmC = global_tensor<D, RowMajor<1, N>>;
    using tileC = Tile<Location::Vec, D, 1, N, BLayout::RowMajor>;
    using itC = global_iterator<gmC, tileC>;
    iter_t<D, M, N> gA(a); itC gC(c);
    auto gA0 = gA(0, 0);
    auto gC0 = gC(0, 0);
    tile_t<D, M, N> tA; tileC tC;
    TLOAD(tA, gA0); op(tC, tA); TSTORE(gC0, tC);
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

// Tile-local gather/scatter use uint16 element indices.
template <typename D, int M, int N>
void bench_tile_gather(D *c, D *a, auto op) {
    using gmIdx = global_tensor<uint16_t, RowMajor<M, N>>;
    using tileIdx = Tile<Location::Vec, uint16_t, M, N, BLayout::RowMajor>;
    uint16_t idx[M * N]; for (int i = 0; i < M * N; ++i) idx[i] = i;
    iter_t<D, M, N> gA(a), gC(c); global_iterator<gmIdx, tileIdx> gIdx(idx);
    auto gA0 = gA(0, 0);
    auto gI0 = gIdx(0, 0);
    auto gC0 = gC(0, 0);
    tile_t<D, M, N> tA, tC; tileIdx tIdx;
    TLOAD(tA, gA0); TLOAD(tIdx, gI0);
    op(tC, tA, tIdx);
    TSTORE(gC0, tC);
}

template <typename D, int M, int N>
void bench_tile_scatter(D *c, D *a, auto op) {
    using gmIdx = global_tensor<uint16_t, RowMajor<M, N>>;
    using tileIdx = Tile<Location::Vec, uint16_t, M, N, BLayout::RowMajor>;
    uint16_t idx[M * N]; for (int i = 0; i < M * N; ++i) idx[i] = i;
    iter_t<D, M, N> gA(a), gC(c); global_iterator<gmIdx, tileIdx> gIdx(idx);
    auto gA0 = gA(0, 0);
    auto gI0 = gIdx(0, 0);
    auto gC0 = gC(0, 0);
    tile_t<D, M, N> tA, tC; tileIdx tIdx;
    TLOAD(tA, gA0); TLOAD(tC, gC0); TLOAD(tIdx, gI0);
    op(tC, tA, tIdx);
    TSTORE(gC0, tC);
}

template <typename D, int M, int N>
void bench_tri(D *c) {
    iter_t<D, M, N> gC(c); auto gC0 = gC(0, 0);
    tile_t<D, M, N> tC; TTRI(tC); TSTORE(gC0, tC);
}

// PTO 0.58.5 CUBE-layout rearrangement instructions.
inline void bench_permute(float *c, float *a, float *b) {
    using Data = VecTileM16<float, 16, 32>;
    using Index = VecTileM16<uint8_t, 16, 128>;
    global_tensor<float, RowMajor<16, 32>> gA(a), gB(b), gC(c);
    static uint8_t index_data[16 * 128] = {};
    global_tensor<uint8_t, RowMajor<16, 128>> gI(index_data);
    Data tA, tB, tC; Index tI;
    TLOAD_CUBE(tA, gA); TLOAD_CUBE(tB, gB); TLOAD_CUBE(tI, gI);
    TPERMUTE(tC, tA, tB, tI); TSTORE_CUBE(gC, tC);
}

inline void bench_shuf(uint32_t *c, uint32_t *a, uint32_t *b) {
    using Words = VecTileM16<uint32_t, 16, 32>;
    global_tensor<uint32_t, RowMajor<16, 32>> gA(a), gB(b), gC(c);
    Words tA, tB, tC;
    TLOAD_CUBE(tA, gA); TLOAD_CUBE(tB, gB);
    TSHUF(tC, tA, tB, 0); TSTORE_CUBE(gC, tC);
}

inline void bench_pack(uint32_t *c, uint32_t *a, uint32_t *b) {
    using Words = VecTileM16<uint32_t, 16, 32>;
    global_tensor<uint32_t, RowMajor<16, 32>> gA(a), gB(b), gC(c);
    Words tA, tB, tC;
    TLOAD_CUBE(tA, gA); TLOAD_CUBE(tB, gB);
    TPACK(tC, tA, tB, 0x00000202); TSTORE_CUBE(gC, tC);
}

inline void bench_unpack(uint32_t *c, uint32_t *a) {
    using Words = VecTileM16<uint32_t, 16, 32>;
    global_tensor<uint32_t, RowMajor<16, 32>> gA(a), gC(c);
    Words tA, tC;
    TLOAD_CUBE(tA, gA); TUNPACK(tC, tA, 0x00000201); TSTORE_CUBE(gC, tC);
}

inline void bench_gpr2t(uint8_t *c) {
    using Bytes = VecTileM16<uint8_t, 16, 8>;
    global_tensor<uint8_t, RowMajor<16, 8>> gC(c);
    Bytes tC; TGPR2T(tC, 1, 2, 3, 4); TSTORE_CUBE(gC, tC);
}

#endif
