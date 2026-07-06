#ifndef LAYOUT_BENCH_HPP
#define LAYOUT_BENCH_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
#include "benchmark.h"

using namespace pto;

template <typename dtype, int tM, int tN>
void ttrans_bench(dtype* c, dtype* a) {
    using gm = global_tensor<dtype, RowMajor<tM, tN>>;
    using tile = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    using it = global_iterator<gm, tile>;
    it gA(a); it gC(c);
    auto gA0 = gA(0,0); auto gC0 = gC(0,0);
    tile tSrc, tDst;
    TCOPYIN(tSrc, gA0);
    TTRANS(tDst, tSrc);
    TCOPYOUT(gC0, tDst);
}

void trans_fp32_16x16(float* c, float* a) { ttrans_bench<float, 16, 16>(c, a); }
void trans_fp32_32x32(float* c, float* a) { ttrans_bench<float, 32, 32>(c, a); }
void trans_fp16_16x16(__half* c, __half* a) { ttrans_bench<__half, 16, 16>(c, a); }

#endif
