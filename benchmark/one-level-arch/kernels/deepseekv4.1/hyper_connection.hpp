#pragma once

#include "numeric.hpp"

namespace deepseek_v41 {

struct Mix {
    float post[Streams], residual[Streams][Streams];
};

template <class B> [[gnu::always_inline]] inline void round_hf32(B &value) {
    // mHC uses round-half-up on the retained ten mantissa bits.
    auto bits = pto::reinterpret_tile<uint32_t>(value);
    TADDS(bits, bits, 0x1000u);
    TANDS(bits, bits, 0xffffe000u);
}

template <int Rows>
inline void mix_logits(const float *x, const float *weights, float inv, float scale, const float *base,
                       float *m) {
    constexpr int width = Streams * Hidden;
    tile::Block<1, width> input;
    tile::Block<Rows, width> fn;
    tile::Block<Rows, 1> dots, bias;
    tile::load(input, x);
    tile::load(fn, weights);
    round_hf32(input);
    round_hf32(fn);
    TCOLEXPANDMUL(fn, fn, input);
    TROWSUM(dots, fn);
    TMULS(dots, dots, inv);
    TMULS(dots, dots, scale);
    tile::load(bias, base);
    TADD(dots, dots, bias);
    tile::store(m, dots);
}

inline void sinkhorn(const float *logits, Mix &mix) {
    tile::Block<Streams, Streams> matrix;
    tile::Block<Streams, 1> rows;
    tile::Block<1, Streams> cols;
    tile::load(matrix, logits);
    TROWMAX(rows, matrix);
    TROWEXPANDSUB(matrix, matrix, rows);
    TEXP(matrix, matrix);
    TROWSUM(rows, matrix);
    TROWEXPANDDIV(matrix, matrix, rows);
    TADDS(matrix, matrix, Eps);
    TCOLSUM(cols, matrix);
    TADDS(cols, cols, Eps);
    TCOLEXPANDDIV(matrix, matrix, cols);
    // The first row softmax adds epsilon after division. All subsequent
    // normalizations add it to the denominator.
    for (int iter = 1; iter < 20; ++iter) {
        TROWSUM(rows, matrix);
        TADDS(rows, rows, Eps);
        TROWEXPANDDIV(matrix, matrix, rows);
        TCOLSUM(cols, matrix);
        TADDS(cols, cols, Eps);
        TCOLEXPANDDIV(matrix, matrix, cols);
    }
    tile::store(mix.residual[0], matrix);
}

inline void hc_pre(const HiddenState &h, const HCWeights &w, float (&norm)[Hidden], float (&next_pre)[Streams],
                   Mix &mix) {
    tile::Block<Streams, Hidden> hidden, weighted;
    tile::Block<Streams, 1> pre;
    tile::Block<1, Hidden> combined;
    tile::load(hidden, h.x[0]);
    tile::load(pre, h.pre);
    TROWEXPANDMUL(weighted, hidden, pre);
    TCOLSUM(combined, weighted);
    tile::round_bf16(combined);
    tile::store(norm, combined);
    rms(norm, w.norm);

    tile::Block<1, Streams * Hidden> flat;
    tile::Block<1, 1> square;
    tile::load(flat, h.x[0]);
    TMUL(flat, flat, flat);
    TROWSUM(square, flat);
    TMULS(square, square, 1.f / (Streams * Hidden));
    TADDS(square, square, Eps);
    TSQRT(square, square);
    TRECIP(square, square);
    float inv, m[Mixes];
    tile::store(&inv, square);
    mix_logits<Streams>(h.x[0], w.fn[0], inv, w.scale[0], w.base, m);
    mix_logits<Streams>(h.x[0], w.fn[Streams], inv, w.scale[1], w.base + Streams, m + Streams);
    mix_logits<Streams * Streams>(h.x[0], w.fn[2 * Streams], inv, w.scale[2], w.base + 2 * Streams,
                                  m + 2 * Streams);

    tile::Block<1, Streams> gate;
    tile::load(gate, m);
    tile::sigmoid(gate, gate);
    TADDS(gate, gate, Eps);
    tile::store(next_pre, gate);
    tile::load(gate, m + Streams);
    tile::sigmoid(gate, gate);
    TMULS(gate, gate, 2.f);
    tile::store(mix.post, gate);
    sinkhorn(m + 2 * Streams, mix);
}

inline void hc_post(const HiddenState &h, const Mix &mix, const float (&value)[Hidden], HiddenState &out) {
    tile::Block<Streams, Hidden> result, stream, coefficient;
    tile::Block<Streams, 1> per_output;
    tile::Block<1, Hidden> row;
    tile::load(row, value);
    tile::load(per_output, mix.post);
    TCOLEXPAND(result, row);
    TROWEXPANDMUL(result, result, per_output);
    for (int s = 0; s < Streams; ++s) {
        tile::load(row, h.x[s]);
        tile::load(per_output, mix.residual[s]);
        TCOLEXPAND(stream, row);
        TROWEXPAND(coefficient, per_output);
        TFMA(result, stream, coefficient, result);
    }
    tile::store(out.x[0], result);
}

inline void inject_engram(const Weights &w, const float (&lookup)[Hidden], HiddenState &h) {
    float kv[(Streams + 1) * Hidden];
    project(activation(lookup), w.engram_kv, kv);
    tile::Block<Streams, Hidden> hidden, keys, product;
    tile::Block<Streams, 1> score, hnorm, knorm, gate;
    tile::Block<1, Hidden> value;
    tile::load(hidden, h.x[0]);
    tile::load(keys, kv);
    tile::load(product, w.engram_weight[0]);
    TMUL(product, hidden, product);
    TMUL(product, product, keys);
    TROWSUM(score, product);
    TMUL(product, hidden, hidden);
    TROWSUM(hnorm, product);
    TMUL(keys, keys, keys);
    TROWSUM(knorm, keys);
    TMULS(hnorm, hnorm, 1.f / Hidden);
    TMULS(knorm, knorm, 1.f / Hidden);
    TADDS(hnorm, hnorm, Eps);
    TADDS(knorm, knorm, Eps);
    TSQRT(hnorm, hnorm);
    TSQRT(knorm, knorm);
    TMUL(hnorm, hnorm, knorm);
    TMULS(hnorm, hnorm, sqrtf(float(Hidden)));
    TDIV(score, score, hnorm);
    TABS(gate, score);
    TMAXS(gate, gate, Eps);
    TSQRT(gate, gate);
    auto sign = pto::reinterpret_tile<uint32_t>(score);
    auto magnitude = pto::reinterpret_tile<uint32_t>(gate);
    TANDS(sign, sign, 0x80000000u);
    TOR(magnitude, magnitude, sign);
    tile::sigmoid(gate, gate);
    tile::load(value, kv + Streams * Hidden);
    TCOLEXPAND(product, value);
    TROWEXPANDMUL(product, product, gate);
    TADD(hidden, hidden, product);
    tile::round_bf16(hidden);
    tile::store(h.x[0], hidden);
}

} // namespace deepseek_v41
