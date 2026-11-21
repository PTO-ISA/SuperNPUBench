#pragma once

#include "types.hpp"
#include <common/pto_tileop.hpp>
#include <math.h>
#include <string.h>

namespace deepseek_v41 {

inline uint32_t bits(float x) {
    return __builtin_bit_cast(uint32_t, x);
}
inline float from_bits(uint32_t x) {
    return __builtin_bit_cast(float, x);
}
inline uint16_t bf16_bits(float x) {
    uint32_t u = bits(x);
    return (u + 0x7fff + ((u >> 16) & 1)) >> 16;
}
inline float from_bf16(uint16_t x) {
    return from_bits(uint32_t(x) << 16);
}
inline float bf16(float x) {
    return from_bf16(bf16_bits(x));
}
inline float hf32(float x) {
    return from_bits((bits(x) + 4096) & ~8191u);
}
inline float sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}
inline float ue8m0(unsigned code) {
    if (code == 255) return NAN;
    return code == 0 ? 0x1p-127f : from_bits(code << 23);
}
inline unsigned upward_scale(float x) {
    // x is positive and normal for the finite benchmark inputs.
    return (bits(x) + 0x7fffff) >> 23;
}

inline float e4m3(unsigned code) {
    if ((code & 127) == 127) return NAN;
    unsigned e = (code & 127) >> 3, m = code & 7;
    float value = e ? from_bits(((e + 120) << 23) | (m << 20)) : m * 0x1p-9f;
    return code & 128 ? -value : value;
}
inline uint8_t encode_e4m3(float x) {
    float mag = fminf(fabsf(x), 448.f);
    unsigned exponent = bits(mag) >> 23;
    // Adding half an ulp minus one plus the low retained bit rounds ties to even.
    unsigned normalized = bits(mag);
    unsigned normal = ((normalized + 0x7ffff + ((normalized >> 20) & 1)) >> 20) - 120 * 8;
    unsigned code = exponent < 121 ? unsigned(__builtin_rintf(mag * 512.f)) : normal;
    return code | ((bits(x) >> 24) & 128);
}

enum class Midpoint { Lower, Even, Upper };
inline float e2m1(unsigned code) {
    constexpr float levels[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    float value = levels[code & 7];
    return code & 8 ? -value : value;
}
inline unsigned encode_e2m1(float x, Midpoint ties) {
    constexpr float cut[] = {.25f, .75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.f};
    unsigned i = 0;
    float a = fabsf(x);
    while (i < 7 &&
           (a > cut[i] || (a == cut[i] && (ties == Midpoint::Upper || (ties == Midpoint::Even && (i & 1))))))
        ++i;
    return i | ((bits(x) >> 28) & 8);
}
inline unsigned nibble(const uint8_t *packed, int i) {
    return (packed[i / 2] >> ((i % 2) * 4)) & 15;
}

template <int N> float absmax(const float *x) {
    float a = 0;
    for (int i = 0; i < N; ++i)
        a = fmaxf(a, fabsf(x[i]));
    return a;
}
template <int N> void quant8(const float (&x)[N], uint8_t *payload, uint8_t *scales) {
    static_assert(N % MxGroup == 0);
    for (int i = 0; i < N; i += MxGroup) {
        unsigned code = upward_scale(fmaxf(absmax<MxGroup>(x + i), .0001f) / 448.f);
        scales[i / MxGroup] = code;
        float inv = 1.f / ue8m0(code);
        for (int j = 0; j < MxGroup; ++j)
            payload[i + j] = encode_e4m3(x[i + j] * inv);
    }
}

template <int Group, class Scale, int N>
void quant4(const float (&x)[N], uint8_t *payload, Scale *scales, Midpoint ties) {
    static_assert(N % Group == 0);
    for (int i = 0; i < N; i += Group) {
        float scale = absmax<Group>(x + i) / 6.f;
        if constexpr (sizeof(Scale) == 1) {
            auto code = upward_scale(fmaxf(scale, 0x1p-126f));
            scales[i / Group] = code;
            scale = ue8m0(code);
        } else {
            auto code = bf16_bits(fmaxf(scale, 0x1p-9f));
            scales[i / Group] = code;
            scale = from_bf16(code);
        }
        for (int j = 0; j < Group; j += 2) {
            unsigned lo = encode_e2m1(x[i + j] / scale, ties);
            unsigned hi = encode_e2m1(x[i + j + 1] / scale, ties);
            payload[(i + j) / 2] = lo | (hi << 4);
        }
    }
}

template <int N> float dot(const float *x, const float *y) {
    float sum = 0.f;
    for (int k = 0; k < N; ++k)
        sum += x[k] * y[k];
    return sum;
}
template <int Out, int In> void linear(const float (&x)[In], const float (&weight)[Out][In], float (&y)[Out]) {
    for (int o = 0; o < Out; ++o)
        y[o] = dot<In>(x, weight[o]);
}
template <int N> void rms(float (&x)[N], const float (&w)[N]) {
    float inv = 1.f / sqrtf(dot<N>(x, x) / N + Eps);
    for (int i = 0; i < N; ++i)
        x[i] = bf16(x[i] * inv * w[i]);
}

constexpr int TileRows = 8;
// Each tile computes eight output channels and one MX scale group.
__attribute__((noinline)) inline void group_dot(float *out, float *w, float *x) {
    using namespace pto;
    using Matrix = Tile<Location::Vec, float, TileRows, MxGroup, BLayout::RowMajor>;
    using Row = Tile<Location::Vec, float, 1, MxGroup, BLayout::RowMajor>;
    using Sums = Tile<Location::Vec, float, TileRows, 1, BLayout::RowMajor>;
    Matrix weights, products;
    Row input;
    Sums sums;
    global_tensor<float, RowMajor<TileRows, MxGroup>> wg(w);
    global_tensor<float, RowMajor<1, MxGroup>> xg(x);
    global_tensor<float, RowMajor<TileRows, 1>> yg(out);
    TLOAD(weights, wg);
    TLOAD(input, xg);
    TCOLEXPANDMUL(products, weights, input);
    TROWSUM(sums, products);
    TSTORE(yg, sums);
}

template <int N> struct MxActivation {
    uint8_t data[N], scale[N / MxGroup];
};
template <int Out, int In, int Bits>
void project(const MxActivation<In> &x, const MxWeight<Out, In, Bits> &w, float (&y)[Out],
             bool round_output = true) {
    static_assert(Out % TileRows == 0);
    memset(y, 0, sizeof(y));
    for (int base = 0; base < In; base += MxGroup) {
        alignas(64) float input[MxGroup];
        int group = base / MxGroup;
        for (int k = 0; k < MxGroup; ++k)
            input[k] = e4m3(x.data[base + k]);
        for (int o = 0; o < Out; o += TileRows) {
            alignas(64) float block[TileRows][MxGroup], partial[TileRows];
            for (int r = 0; r < TileRows; ++r) {
                for (int k = 0; k < MxGroup; ++k) {
                    if constexpr (Bits == 4)
                        block[r][k] = e2m1(nibble(w.data[o + r], base + k));
                    else
                        block[r][k] = e4m3(w.data[o + r][base + k]);
                }
            }
            group_dot(partial, &block[0][0], input);
            for (int r = 0; r < TileRows; ++r)
                y[o + r] += partial[r] * ue8m0(x.scale[group]) * ue8m0(w.scale[o + r][group]);
        }
    }
    if (round_output)
        for (float &v : y)
            v = bf16(v);
}
template <int N> MxActivation<N> activation(const float (&x)[N]) {
    MxActivation<N> out;
    quant8(x, out.data, out.scale);
    return out;
}

template <int N> void rotate(float (&x)[N], const Rotary &r, bool inverse = false, bool round_output = true) {
    for (int i = 0; i < RopeDim / 2; ++i) {
        int k = N - RopeDim + 2 * i;
        float a = x[k], b = x[k + 1], c = r.cosine[i], s = inverse ? -r.sine[i] : r.sine[i];
        float re = a * c - b * s, im = a * s + b * c;
        x[k] = round_output ? bf16(re) : re;
        x[k + 1] = round_output ? bf16(im) : im;
    }
}

} // namespace deepseek_v41
