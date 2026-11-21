#pragma once

#include "numeric.hpp"

namespace deepseek_v41 {

struct Mix {
    float post[Streams], residual[Streams][Streams];
};

inline void hc_pre(const HiddenState &h, const HCWeights &w, float (&norm)[Hidden], float (&next_pre)[Streams],
                   Mix &mix) {
    float flat[Streams * Hidden], square = 0.f;
    for (int d = 0; d < Hidden; ++d) {
        float sum = 0.f;
        for (int s = 0; s < Streams; ++s) {
            float v = h.x[s][d];
            sum += v * h.pre[s];
            square += v * v;
            flat[s * Hidden + d] = hf32(v);
        }
        norm[d] = bf16(sum);
    }
    rms(norm, w.norm);
    float inv = 1.f / sqrtf(square / (Streams * Hidden) + Eps);
    float m[Mixes];
    for (int i = 0; i < Mixes; ++i) {
        float v = 0;
        for (int k = 0; k < Streams * Hidden; ++k)
            v += flat[k] * hf32(w.fn[i][k]);
        int group = i < Streams ? 0 : (i < 2 * Streams ? 1 : 2);
        m[i] = v * inv * w.scale[group] + w.base[i];
    }
    for (int s = 0; s < Streams; ++s) {
        next_pre[s] = sigmoid(m[s]) + Eps;
        mix.post[s] = 2.f * sigmoid(m[Streams + s]);
        const float *logit = m + (2 + s) * Streams;
        float peak = logit[0], total = 0;
        for (int t = 1; t < Streams; ++t)
            peak = fmaxf(peak, logit[t]);
        for (int t = 0; t < Streams; ++t)
            total += (mix.residual[s][t] = expf(logit[t] - peak));
        for (float &v : mix.residual[s])
            v = v / total + Eps;
    }
    // A row softmax above starts the first of 20 row/column normalizations.
    for (int iter = 0; iter < 20; ++iter) {
        if (iter)
            for (auto &row : mix.residual) {
                float total = Eps;
                for (float v : row)
                    total += v;
                for (float &v : row)
                    v /= total;
            }
        for (int t = 0; t < Streams; ++t) {
            float total = Eps;
            for (auto &row : mix.residual)
                total += row[t];
            for (auto &row : mix.residual)
                row[t] /= total;
        }
    }
}

inline void hc_post(const HiddenState &h, const Mix &mix, const float (&value)[Hidden], HiddenState &out) {
    for (int t = 0; t < Streams; ++t)
        for (int d = 0; d < Hidden; ++d) {
            float v = mix.post[t] * value[d];
            for (int s = 0; s < Streams; ++s)
                v = fmaf(h.x[s][d], mix.residual[s][t], v);
            out.x[t][d] = v;
        }
}

inline void inject_engram(const Weights &w, const float (&lookup)[Hidden], HiddenState &h) {
    float kv[(Streams + 1) * Hidden];
    project(activation(lookup), w.engram_kv, kv);
    for (int s = 0; s < Streams; ++s) {
        const float *key = kv + s * Hidden;
        float dot_key = 0.f;
        for (int d = 0; d < Hidden; ++d)
            dot_key += h.x[s][d] * w.engram_weight[s][d] * key[d];
        float norm_h = sqrtf(dot<Hidden>(h.x[s], h.x[s]) / Hidden + Eps);
        float norm_k = sqrtf(dot<Hidden>(key, key) / Hidden + Eps);
        float score = dot_key / (norm_h * norm_k * sqrtf(float(Hidden)));
        float gate = sigmoid(copysignf(sqrtf(fmaxf(fabsf(score), Eps)), score));
        for (int d = 0; d < Hidden; ++d)
            h.x[s][d] = bf16(h.x[s][d] + gate * kv[Streams * Hidden + d]);
    }
}

} // namespace deepseek_v41
