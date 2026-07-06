#ifndef MEMORY_BENCH_HPP
#define MEMORY_BENCH_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
#include "benchmark.h"

using namespace pto;

template <typename dtype, int tM, int tN>
void tload_bench(dtype* c, dtype* a) {
    using gm = global_tensor<dtype, RowMajor<tM, tN>>;
    using tile = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    using it = global_iterator<gm, tile>;
    it gA(a); it gC(c);
    auto gA0 = gA(0,0); auto gC0 = gC(0,0);
    tile tTile;
    TCOPYIN(tTile, gA0);
    TCOPYOUT(gC0, tTile);
}

void tload_fp32_16x16(float* c, float* a) { tload_bench<float, 16, 16>(c, a); }
void tload_fp32_32x32(float* c, float* a) { tload_bench<float, 32, 32>(c, a); }
void tload_fp16_16x16(__half* c, __half* a) { tload_bench<__half, 16, 16>(c, a); }
void tload_int32_16x16(int32_t* c, int32_t* a) { tload_bench<int32_t, 16, 16>(c, a); }

#endif
