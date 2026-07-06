#ifndef ELEM_BENCH_HPP
#define ELEM_BENCH_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
#include "benchmark.h"

using namespace pto;

template <typename dtype, int tM, int tN>
void bench_binary(dtype* c, dtype* a, dtype* b, auto op) {
    using gm = global_tensor<dtype, RowMajor<tM, tN>>;
    using tile = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    using it = global_iterator<gm, tile>;
    it gA(a); it gB(b); it gC(c);
    auto gA0 = gA(0,0); auto gB0 = gB(0,0); auto gC0 = gC(0,0);
    tile tA, tB, tC;
    TCOPYIN(tA, gA0);
    TCOPYIN(tB, gB0);
    op(tC, tA, tB);
    TCOPYOUT(gC0, tC);
}

template <typename dtype, int tM, int tN>
void bench_unary(dtype* c, dtype* a, auto op) {
    using gm = global_tensor<dtype, RowMajor<tM, tN>>;
    using tile = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    using it = global_iterator<gm, tile>;
    it gA(a); it gC(c);
    auto gA0 = gA(0,0); auto gC0 = gC(0,0);
    tile tA, tC;
    TCOPYIN(tA, gA0);
    op(tC, tA);
    TCOPYOUT(gC0, tC);
}

void tadd_fp32_16x16(float* c, float* a, float* b) {
    bench_binary<float, 16, 16>(c, a, b, [](auto& dst, auto& s0, auto& s1) { TADD(dst, s0, s1); });
}
void tadd_fp16_16x16(__half* c, __half* a, __half* b) {
    bench_binary<__half, 16, 16>(c, a, b, [](auto& dst, auto& s0, auto& s1) { TADD(dst, s0, s1); });
}
void tadd_int32_16x16(int32_t* c, int32_t* a, int32_t* b) {
    bench_binary<int32_t, 16, 16>(c, a, b, [](auto& dst, auto& s0, auto& s1) { TADD(dst, s0, s1); });
}
void tadd_int16_16x16(int16_t* c, int16_t* a, int16_t* b) {
    bench_binary<int16_t, 16, 16>(c, a, b, [](auto& dst, auto& s0, auto& s1) { TADD(dst, s0, s1); });
}
void tsub_fp32_16x16(float* c, float* a, float* b) {
    bench_binary<float, 16, 16>(c, a, b, [](auto& dst, auto& s0, auto& s1) { TSUB(dst, s0, s1); });
}
void tsub_fp16_16x16(__half* c, __half* a, __half* b) {
    bench_binary<__half, 16, 16>(c, a, b, [](auto& dst, auto& s0, auto& s1) { TSUB(dst, s0, s1); });
}
void tsub_int32_16x16(int32_t* c, int32_t* a, int32_t* b) {
    bench_binary<int32_t, 16, 16>(c, a, b, [](auto& dst, auto& s0, auto& s1) { TSUB(dst, s0, s1); });
}
void tsub_int16_16x16(int16_t* c, int16_t* a, int16_t* b) {
    bench_binary<int16_t, 16, 16>(c, a, b, [](auto& dst, auto& s0, auto& s1) { TSUB(dst, s0, s1); });
}
void tmul_fp32_16x16(float* c, float* a, float* b) {
    bench_binary<float, 16, 16>(c, a, b, [](auto& dst, auto& s0, auto& s1) { TMUL(dst, s0, s1); });
}
void tmul_fp16_16x16(__half* c, __half* a, __half* b) {
    bench_binary<__half, 16, 16>(c, a, b, [](auto& dst, auto& s0, auto& s1) { TMUL(dst, s0, s1); });
}
void tmul_int32_16x16(int32_t* c, int32_t* a, int32_t* b) {
    bench_binary<int32_t, 16, 16>(c, a, b, [](auto& dst, auto& s0, auto& s1) { TMUL(dst, s0, s1); });
}
void tmul_int16_16x16(int16_t* c, int16_t* a, int16_t* b) {
    bench_binary<int16_t, 16, 16>(c, a, b, [](auto& dst, auto& s0, auto& s1) { TMUL(dst, s0, s1); });
}
void tdiv_fp32_16x16(float* c, float* a, float* b) {
    bench_binary<float, 16, 16>(c, a, b, [](auto& dst, auto& s0, auto& s1) { TDIV(dst, s0, s1); });
}
void tdiv_fp16_16x16(__half* c, __half* a, __half* b) {
    bench_binary<__half, 16, 16>(c, a, b, [](auto& dst, auto& s0, auto& s1) { TDIV(dst, s0, s1); });
}
void tdiv_int32_16x16(int32_t* c, int32_t* a, int32_t* b) {
    bench_binary<int32_t, 16, 16>(c, a, b, [](auto& dst, auto& s0, auto& s1) { TDIV(dst, s0, s1); });
}
void tdiv_int16_16x16(int16_t* c, int16_t* a, int16_t* b) {
    bench_binary<int16_t, 16, 16>(c, a, b, [](auto& dst, auto& s0, auto& s1) { TDIV(dst, s0, s1); });
}
void tmax_fp32_16x16(float* c, float* a, float* b) {
    bench_binary<float, 16, 16>(c, a, b, [](auto& dst, auto& s0, auto& s1) { TMAX(dst, s0, s1); });
}
void tmax_fp16_16x16(__half* c, __half* a, __half* b) {
    bench_binary<__half, 16, 16>(c, a, b, [](auto& dst, auto& s0, auto& s1) { TMAX(dst, s0, s1); });
}
void tmax_int32_16x16(int32_t* c, int32_t* a, int32_t* b) {
    bench_binary<int32_t, 16, 16>(c, a, b, [](auto& dst, auto& s0, auto& s1) { TMAX(dst, s0, s1); });
}
void tmax_int16_16x16(int16_t* c, int16_t* a, int16_t* b) {
    bench_binary<int16_t, 16, 16>(c, a, b, [](auto& dst, auto& s0, auto& s1) { TMAX(dst, s0, s1); });
}
void tmin_fp32_16x16(float* c, float* a, float* b) {
    bench_binary<float, 16, 16>(c, a, b, [](auto& dst, auto& s0, auto& s1) { TMIN(dst, s0, s1); });
}
void tmin_fp16_16x16(__half* c, __half* a, __half* b) {
    bench_binary<__half, 16, 16>(c, a, b, [](auto& dst, auto& s0, auto& s1) { TMIN(dst, s0, s1); });
}
void tmin_int32_16x16(int32_t* c, int32_t* a, int32_t* b) {
    bench_binary<int32_t, 16, 16>(c, a, b, [](auto& dst, auto& s0, auto& s1) { TMIN(dst, s0, s1); });
}
void tmin_int16_16x16(int16_t* c, int16_t* a, int16_t* b) {
    bench_binary<int16_t, 16, 16>(c, a, b, [](auto& dst, auto& s0, auto& s1) { TMIN(dst, s0, s1); });
}
void tand_int32_16x16(int32_t* c, int32_t* a, int32_t* b) {
    bench_binary<int32_t, 16, 16>(c, a, b, [](auto& dst, auto& s0, auto& s1) { TAND(dst, s0, s1); });
}
void tand_int16_16x16(int16_t* c, int16_t* a, int16_t* b) {
    bench_binary<int16_t, 16, 16>(c, a, b, [](auto& dst, auto& s0, auto& s1) { TAND(dst, s0, s1); });
}
void tor_int32_16x16(int32_t* c, int32_t* a, int32_t* b) {
    bench_binary<int32_t, 16, 16>(c, a, b, [](auto& dst, auto& s0, auto& s1) { TOR(dst, s0, s1); });
}
void tor_int16_16x16(int16_t* c, int16_t* a, int16_t* b) {
    bench_binary<int16_t, 16, 16>(c, a, b, [](auto& dst, auto& s0, auto& s1) { TOR(dst, s0, s1); });
}
void tabs_fp32_16x16(float* c, float* a) {
    bench_unary<float, 16, 16>(c, a, [](auto& dst, auto& s0) { TABS(dst, s0); });
}
void tabs_fp16_16x16(__half* c, __half* a) {
    bench_unary<__half, 16, 16>(c, a, [](auto& dst, auto& s0) { TABS(dst, s0); });
}
void texp_fp32_16x16(float* c, float* a) {
    bench_unary<float, 16, 16>(c, a, [](auto& dst, auto& s0) { TEXP(dst, s0); });
}
void texp_fp16_16x16(__half* c, __half* a) {
    bench_unary<__half, 16, 16>(c, a, [](auto& dst, auto& s0) { TEXP(dst, s0); });
}
void tsqrt_fp32_16x16(float* c, float* a) {
    bench_unary<float, 16, 16>(c, a, [](auto& dst, auto& s0) { TSQRT(dst, s0); });
}
void tsqrt_fp16_16x16(__half* c, __half* a) {
    bench_unary<__half, 16, 16>(c, a, [](auto& dst, auto& s0) { TSQRT(dst, s0); });
}
void trecip_fp32_16x16(float* c, float* a) {
    bench_unary<float, 16, 16>(c, a, [](auto& dst, auto& s0) { TRECIP(dst, s0); });
}
void trecip_fp16_16x16(__half* c, __half* a) {
    bench_unary<__half, 16, 16>(c, a, [](auto& dst, auto& s0) { TRECIP(dst, s0); });
}
#endif
