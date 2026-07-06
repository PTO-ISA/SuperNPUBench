#ifndef SCALAR_BENCH_HPP
#define SCALAR_BENCH_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
#include "benchmark.h"

using namespace pto;

template <typename dtype, int tM, int tN>
void bench_scalar(dtype* c, dtype* a, dtype s, auto op) {
    using gm = global_tensor<dtype, RowMajor<tM, tN>>;
    using tile = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    using it = global_iterator<gm, tile>;
    it gA(a); it gC(c);
    auto gA0 = gA(0,0); auto gC0 = gC(0,0);
    tile tA, tC;
    TCOPYIN(tA, gA0);
    op(tC, tA, s);
    TCOPYOUT(gC0, tC);
}

void tadds_fp32_16x16(float* c, float* a, float s) {
    bench_scalar<float, 16, 16>(c, a, s, [](auto& dst, auto& src, auto scal) { TADDS(dst, src, scal); });
}
void tadds_fp16_16x16(__half* c, __half* a, __half s) {
    bench_scalar<__half, 16, 16>(c, a, s, [](auto& dst, auto& src, auto scal) { TADDS(dst, src, scal); });
}
void tadds_int32_16x16(int32_t* c, int32_t* a, int32_t s) {
    bench_scalar<int32_t, 16, 16>(c, a, s, [](auto& dst, auto& src, auto scal) { TADDS(dst, src, scal); });
}
void tadds_int16_16x16(int16_t* c, int16_t* a, int16_t s) {
    bench_scalar<int16_t, 16, 16>(c, a, s, [](auto& dst, auto& src, auto scal) { TADDS(dst, src, scal); });
}
void tsubs_fp32_16x16(float* c, float* a, float s) {
    bench_scalar<float, 16, 16>(c, a, s, [](auto& dst, auto& src, auto scal) { TSUBS(dst, src, scal); });
}
void tsubs_fp16_16x16(__half* c, __half* a, __half s) {
    bench_scalar<__half, 16, 16>(c, a, s, [](auto& dst, auto& src, auto scal) { TSUBS(dst, src, scal); });
}
void tsubs_int32_16x16(int32_t* c, int32_t* a, int32_t s) {
    bench_scalar<int32_t, 16, 16>(c, a, s, [](auto& dst, auto& src, auto scal) { TSUBS(dst, src, scal); });
}
void tsubs_int16_16x16(int16_t* c, int16_t* a, int16_t s) {
    bench_scalar<int16_t, 16, 16>(c, a, s, [](auto& dst, auto& src, auto scal) { TSUBS(dst, src, scal); });
}
void tmuls_fp32_16x16(float* c, float* a, float s) {
    bench_scalar<float, 16, 16>(c, a, s, [](auto& dst, auto& src, auto scal) { TMULS(dst, src, scal); });
}
void tmuls_fp16_16x16(__half* c, __half* a, __half s) {
    bench_scalar<__half, 16, 16>(c, a, s, [](auto& dst, auto& src, auto scal) { TMULS(dst, src, scal); });
}
void tmuls_int32_16x16(int32_t* c, int32_t* a, int32_t s) {
    bench_scalar<int32_t, 16, 16>(c, a, s, [](auto& dst, auto& src, auto scal) { TMULS(dst, src, scal); });
}
void tmuls_int16_16x16(int16_t* c, int16_t* a, int16_t s) {
    bench_scalar<int16_t, 16, 16>(c, a, s, [](auto& dst, auto& src, auto scal) { TMULS(dst, src, scal); });
}
void tdivs_fp32_16x16(float* c, float* a, float s) {
    bench_scalar<float, 16, 16>(c, a, s, [](auto& dst, auto& src, auto scal) { TDIVS(dst, src, scal); });
}
void tdivs_fp16_16x16(__half* c, __half* a, __half s) {
    bench_scalar<__half, 16, 16>(c, a, s, [](auto& dst, auto& src, auto scal) { TDIVS(dst, src, scal); });
}
void tdivs_int32_16x16(int32_t* c, int32_t* a, int32_t s) {
    bench_scalar<int32_t, 16, 16>(c, a, s, [](auto& dst, auto& src, auto scal) { TDIVS(dst, src, scal); });
}
void tdivs_int16_16x16(int16_t* c, int16_t* a, int16_t s) {
    bench_scalar<int16_t, 16, 16>(c, a, s, [](auto& dst, auto& src, auto scal) { TDIVS(dst, src, scal); });
}
void tmaxs_fp32_16x16(float* c, float* a, float s) {
    bench_scalar<float, 16, 16>(c, a, s, [](auto& dst, auto& src, auto scal) { TMAXS(dst, src, scal); });
}
void tmaxs_fp16_16x16(__half* c, __half* a, __half s) {
    bench_scalar<__half, 16, 16>(c, a, s, [](auto& dst, auto& src, auto scal) { TMAXS(dst, src, scal); });
}
void tmaxs_int32_16x16(int32_t* c, int32_t* a, int32_t s) {
    bench_scalar<int32_t, 16, 16>(c, a, s, [](auto& dst, auto& src, auto scal) { TMAXS(dst, src, scal); });
}
void tmaxs_int16_16x16(int16_t* c, int16_t* a, int16_t s) {
    bench_scalar<int16_t, 16, 16>(c, a, s, [](auto& dst, auto& src, auto scal) { TMAXS(dst, src, scal); });
}
void tmins_fp32_16x16(float* c, float* a, float s) {
    bench_scalar<float, 16, 16>(c, a, s, [](auto& dst, auto& src, auto scal) { TMINS(dst, src, scal); });
}
void tmins_fp16_16x16(__half* c, __half* a, __half s) {
    bench_scalar<__half, 16, 16>(c, a, s, [](auto& dst, auto& src, auto scal) { TMINS(dst, src, scal); });
}
void tmins_int32_16x16(int32_t* c, int32_t* a, int32_t s) {
    bench_scalar<int32_t, 16, 16>(c, a, s, [](auto& dst, auto& src, auto scal) { TMINS(dst, src, scal); });
}
void tmins_int16_16x16(int16_t* c, int16_t* a, int16_t s) {
    bench_scalar<int16_t, 16, 16>(c, a, s, [](auto& dst, auto& src, auto scal) { TMINS(dst, src, scal); });
}
#endif
