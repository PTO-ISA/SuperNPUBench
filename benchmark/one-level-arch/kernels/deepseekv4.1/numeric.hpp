#pragma once

#include "types.hpp"
#include "tile.hpp"
#include <array>
#include <math.h>

namespace deepseek_v41 {

enum class Midpoint { Lower, Even, Upper };

template <int N, class Scale> void quant8(const float *x, uint8_t *payload, Scale *scales) {
    static_assert(N % MxGroup == 0);
    constexpr int groups = N / MxGroup;
    tile::Block<groups, MxGroup> values, magnitude;
    tile::Block<groups, 1> scale;
    tile::Block<groups, 1, __fp8_e8m0> exponent;
    tile::Block<groups, MxGroup, __fp8_e4m3> encoded;
    tile::load(values, x);
    TABS(magnitude, values);
    TROWMAX(scale, magnitude);
    TMAXS(scale, scale, .0001f);
    TDIVS(scale, scale, 448.f);
    TCVT<LINX_RUP>(exponent, scale);
    TCVT(scale, exponent);
    if constexpr (sizeof(Scale) == 1) {
        tile::store(reinterpret_cast<__fp8_e8m0 *>(scales), exponent);
    } else {
        tile::Block<groups, 1, __bf16> half;
        TCVT(half, scale);
        tile::store(reinterpret_cast<__bf16 *>(scales), half);
    }
    TROWEXPANDDIV(values, values, scale);
    TMINS(values, values, 448.f);
    TMAXS(values, values, -448.f);
    TCVT<LINX_RNE>(encoded, values);
    // Restore the E4M3FN sign when RNE underflows a negative nonzero lane.
    auto raw = pto::reinterpret_tile<uint32_t>(values);
    auto sign = pto::reinterpret_tile<uint32_t>(magnitude);
    TSHRS(sign, raw, 24);
    TANDS(sign, sign, 0x80u);
    tile::Block<groups, MxGroup, __fp8_e4m3> sign_storage;
    auto sign_bytes = pto::reinterpret_tile<uint8_t>(sign_storage);
    auto codes = pto::reinterpret_tile<uint8_t>(encoded);
    TCVT(sign_bytes, sign);
    TOR(codes, codes, sign_bytes);
    tile::store(payload, codes);
}

// A packed FP4 conversion can canonicalize a signed zero. Recover both
// source signs directly, including negative values rounded to zero.
template <int N> inline void restore_fp4_signs(uint8_t *bytes, const float *input) {
    using Strided = tile::GM<uint32_t, N / 2, 1, 2>;
    Strided even(reinterpret_cast<const uint32_t *>(input));
    Strided odd(reinterpret_cast<const uint32_t *>(input + 1));
    tile::Block<N / 2, 1, uint32_t> lo, hi;
    TLOAD(lo, even);
    TLOAD(hi, odd);
    TSHRS(lo, lo, 28);
    TANDS(lo, lo, 0x08u);
    TSHRS(hi, hi, 24);
    TANDS(hi, hi, 0x80u);
    TOR(lo, lo, hi);
    tile::Block<N / 2, 1, uint8_t> sign, packed;
    TCVT(sign, lo);
    tile::load(packed, bytes);
    TOR(packed, packed, sign);
    tile::store(bytes, packed);
}

template <int Group, Midpoint Ties, class Scale, int N>
void quant4(const float (&x)[N], uint8_t *payload, Scale *scales) {
    static_assert(N % Group == 0);
    constexpr int groups = N / Group;
    tile::Block<groups, Group> input, mag;
    tile::Block<groups, 1> scale;
    tile::load(input, x);
    TABS(mag, input);
    TROWMAX(scale, mag);
    TDIVS(scale, scale, 6.f);
    if constexpr (sizeof(Scale) == 1) {
        tile::Block<groups, 1, __fp8_e8m0> exponent;
        TMAXS(scale, scale, 0x1p-126f);
        TCVT<LINX_RUP>(exponent, scale);
        tile::store(reinterpret_cast<__fp8_e8m0 *>(scales), exponent);
        TCVT(scale, exponent);
    } else {
        tile::Block<groups, 1, __bf16> half;
        TMAXS(scale, scale, 0x1p-9f);
        TCVT<LINX_RNE>(half, scale);
        tile::store(reinterpret_cast<__bf16 *>(scales), half);
        TCVT(scale, half);
    }
    TROWEXPANDDIV(input, input, scale);
    if constexpr (Ties == Midpoint::Lower) {
        // These are precisely the three ties at which RNE rounds magnitude up.
        // Move only an exact tie one FP32 ulp towards zero.
        TABS(mag, input);
        auto raw = pto::reinterpret_tile<uint32_t>(input);
        tile::Block<groups, Group> adjacent;
        auto lower = pto::reinterpret_tile<uint32_t>(adjacent);
        TADDS(lower, raw, 0xffffffffu);
        tile::Block<groups, Group, uint8_t> tie;
        for (float boundary : {.75f, 1.75f, 3.5f}) {
            TCMPS<pto::CmpMode::EQ>(tie, mag, boundary);
            TSEL(raw, tie, lower);
        }
    }

    float normalized[N];
    tile::store(normalized, input);
    pto::CubeTileM16<float, 1, N> value;
    pto::CubeTileM16<__fp4_e2m1x2, 1, N> packed;
    tile::GM<float, 1, N> source(normalized);
    tile::GM<__fp4_e2m1x2, 1, N> dest(reinterpret_cast<__fp4_e2m1x2 *>(payload));
    TLOAD_CUBE(value, source);
    TCVT<Ties == Midpoint::Upper ? LINX_RNA : LINX_RNE>(packed, value);
    TSTORE_CUBE(dest, packed);
    restore_fp4_signs<N>(payload, x);
}

template <int Out, int In> void linear(const float (&x)[In], const float (&weight)[Out][In], float (&y)[Out]) {
    tile::Block<Out, In> matrix;
    tile::Block<1, In> input;
    tile::Block<Out, 1> sums;
    tile::load(matrix, weight[0]);
    tile::load(input, x);
    TCOLEXPANDMUL(matrix, matrix, input);
    TROWSUM(sums, matrix);
    tile::store(y, sums);
}

template <bool RoundInput = false, int N> void rms(float (&x)[N], const float (&w)[N]) {
    tile::Block<1, N> value, squares, weight;
    tile::Block<1, 1> scale;
    tile::load(value, x);
    if constexpr (RoundInput)
        tile::round_bf16(value);
    TMUL(squares, value, value);
    TROWSUM(scale, squares);
    TMULS(scale, scale, 1.f / N);
    TADDS(scale, scale, Eps);
    TSQRT(scale, scale);
    TRECIP(scale, scale);
    TROWEXPANDMUL(value, value, scale);
    tile::load(weight, w);
    TMUL(value, value, weight);
    tile::round_bf16(value);
    tile::store(x, value);
}

template <int N> struct MxActivation {
    uint8_t data[N], scale[N / MxGroup];
};

// Each gather addresses the original output-major weights. For FP4 the two
// nibbles arriving together belong to adjacent K, whereas CUBE B packs adjacent N.
template <int In, int Bits>
[[gnu::always_inline]] inline void load_weight_block(const uint8_t *weights, uint8_t *kn) {
    constexpr int N = 32, step = 8 / Bits;
    constexpr int rows = In / step, cols = N / step;
    static constexpr auto k_offsets = [] {
        std::array<uint32_t, rows> offsets{};
        for (int k = 0; k < rows; ++k)
            offsets[k] = k;
        return offsets;
    }();
    tile::Block<rows, 1, uint32_t> k;
    tile::Block<1, cols, uint32_t> channel;
    tile::Block<rows, cols, uint32_t> index;
    tile::load(k, k_offsets.data());
    TCI(channel, uint32_t{0});
    TMULS(channel, channel, uint32_t(In));
    TROWEXPAND(index, k);
    TCOLEXPANDADD(index, index, channel);  // byte offset = k + channel * In
    tile::GM<uint8_t, N, In / step> src(weights);
    tile::Block<rows, cols, uint8_t> even;
    MGATHER(even, src, index);
    if constexpr (Bits == 8) {
        tile::store(kn, even);
    } else {
        tile::GM<uint8_t, N, In / step> next(weights + In / step);
        tile::Block<rows, cols, uint8_t> odd, lo, hi;
        MGATHER(odd, next, index);
        TANDS(lo, even, 15);
        TSHLS(hi, odd, 4);
        TOR(lo, lo, hi);
        tile::GM<uint8_t, rows, cols, 2 * cols> even_k(kn), odd_k(kn + cols);
        TSTORE(even_k, lo);
        TSHRS(lo, even, 4);
        TANDS(hi, odd, 0xf0);
        TOR(hi, hi, lo);
        TSTORE(odd_k, hi);
    }
}

template <bool Round, int In, int Bits>
[[gnu::noinline]] void project_block(const MxActivation<In> &x, const uint8_t *weights, const uint8_t *scales,
                                     float *y) {
    using namespace pto;
    using W = std::conditional_t<Bits == 8, __fp8_e4m3, __fp4_e2m1x2>;
    constexpr int N = 32, groups = In / MxGroup;
    uint8_t kn[In * N * Bits / 8];
    load_weight_block<In, Bits>(weights, kn);
    CubeTileM16<__fp8_e4m3, 1, In> a;
    CubeTileN8<W, In, N> b;
    CubeAccumulatorM16<float, 1, N> d;
    Tile<Location::Scaling, __fp8_e8m0, 32, 4, BLayout::CubeM32, 1, groups> sa;
    Tile<Location::Scaling, __fp8_e8m0, 32, 4, BLayout::CubeM32, N, groups> sb;
    tile::GM<__fp8_e4m3, 1, In> gx(reinterpret_cast<const __fp8_e4m3 *>(x.data));
    tile::GM<__fp8_e8m0, 1, groups> gsa(reinterpret_cast<const __fp8_e8m0 *>(x.scale));
    tile::GM<W, In, N> gb(reinterpret_cast<const W *>(kn));
    tile::GM<__fp8_e8m0, N, groups> gsb(reinterpret_cast<const __fp8_e8m0 *>(scales));
    tile::GM<float, 1, N> gy(y);
    TLOAD_CUBE(a, gx);
    TLOAD_CUBE(sa, gsa);
    TLOAD_CUBE(b, gb);
    TLOAD_CUBE(sb, gsb);
    TGEMV_MX(d, b, sb, a, sa);
    if constexpr (Round) {
        CubeTileM16<__bf16, 1, N> rounded;
        TCVT<LINX_RNE>(rounded, d);
        TCVT(d, rounded);
    }
    TSTORE_CUBE(gy, d);
}

template <bool Round = true, int Out, int In, int Bits, int Width>
void project_rows(const MxActivation<In> &x, const MxWeight<Out, In, Bits> &w, float (*y)[Width]) {
    static_assert(Width % 32 == 0 && Out % Width == 0);
    for (int n = 0; n < Out; n += 32)
        project_block<Round, In, Bits>(x, w.data[n], w.scale[n], &y[n / Width][n % Width]);
}

template <int Out, int In, int Bits>
void project(const MxActivation<In> &x, const MxWeight<Out, In, Bits> &w, float (&y)[Out]) {
    project_rows(x, w, &y);
}

template <int N> MxActivation<N> activation(const float (&x)[N]) {
    MxActivation<N> out;
    quant8<N>(x, out.data, out.scale);
    return out;
}

template <bool Inverse = false, bool Round = true, int N> void rotate(float (&x)[N], const Rotary &r) {
    constexpr int pairs = RopeDim / 2;
    tile::GM<float, pairs, 1, 2> even(x + N - RopeDim), odd(x + N - RopeDim + 1);
    tile::Block<pairs, 1> a, b, c, s, re, im, tmp;
    TLOAD(a, even);
    TLOAD(b, odd);
    tile::load(c, r.cosine);
    tile::load(s, r.sine);
    if constexpr (Inverse)
        TNEG(s, s);
    TMUL(re, a, c);
    TMUL(tmp, b, s);
    TSUB(re, re, tmp);
    TMUL(im, a, s);
    TMUL(tmp, b, c);
    TADD(im, im, tmp);
    if constexpr (Round) {
        tile::round_bf16(re);
        tile::round_bf16(im);
    }
    TSTORE(even, re);
    TSTORE(odd, im);
}

} // namespace deepseek_v41
