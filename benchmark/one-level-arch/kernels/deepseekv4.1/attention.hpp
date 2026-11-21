#pragma once

#include "hyper_connection.hpp"

namespace deepseek_v41 {

inline void unpack(const WindowRow &r, float (&x)[HeadDim]) {
    for (int d = 0; d < HeadDim; ++d)
        x[d] = e4m3(r.data[d]) * from_bf16(r.scale[d / MxGroup]);
}
inline void unpack(const CompressedRow &r, float (&x)[HeadDim]) {
    for (int d = 0; d < HeadDim; ++d)
        x[d] = bf16(e2m1(nibble(r.data, d)) * from_bf16(r.scale[d / CompGroup]));
}
inline void unpack(const IndexRow &r, float (&x)[IndexDim]) {
    for (int d = 0; d < IndexDim; ++d)
        x[d] = e2m1(nibble(r.data, d)) * ue8m0(r.scale[d / MxGroup]);
}

inline void publish_window(const Weights &w, const MxActivation<Hidden> &x, const Token &t, WindowRow &row) {
    float kv[HeadDim];
    project(x, w.kv, kv);
    rms(kv, w.kv_norm);
    rotate(kv, t.rope);
    uint8_t scale[HeadDim / MxGroup];
    quant8(kv, row.data, scale);
    for (int i = 0; i < HeadDim / MxGroup; ++i)
        row.scale[i] = bf16_bits(ue8m0(scale[i]));
}

inline int physical(const LayerCache &cache, int logical) {
    return cache.pages[logical / Page] * Page + logical % Page;
}

template <class Memory>
void compress(const Schedule &s, const Weights &w, const Token &t, const float (&x)[Hidden], LayerCache &cache,
              Memory &mem) {
    float latent[HeadDim];
    linear(x, w.compressor, latent);
    int logical = t.position;
    if (s.kind == Kind::C2Full) {
        int slot = t.position % 2;
        float gate[HeadDim];
        linear(x, w.compressor_gate, gate);
        memcpy(cache.pair_kv[slot], latent, sizeof(latent));
        memcpy(cache.pair_gate[slot], gate, sizeof(gate));
        if (slot == 0)
            return;
        for (int d = 0; d < HeadDim; ++d) {
            float alpha = sigmoid(gate[d] - cache.pair_gate[0][d]);
            latent[d] = cache.pair_kv[0][d] * (1.f - alpha) + latent[d] * alpha;
        }
        logical /= 2;
    }
    for (float &v : latent)
        v = bf16(v);
    rms(latent, w.comp_norm);
    float key[IndexDim];
    linear(latent, w.ik, key); // index key uses the unrotated compressor latent
    for (float &v : key)
        v = bf16(v);
    rms(key, w.ik_norm);
    rotate(key, t.compressed_rope);
    rotate(latent, t.compressed_rope);
    int slot = physical(cache, logical);
    CompressedRow kv;
    IndexRow index;
    quant4<CompGroup>(latent, kv.data, kv.scale, s.kind == Kind::C1Full ? Midpoint::Even : Midpoint::Lower);
    quant4<MxGroup>(key, index.data, index.scale, Midpoint::Even);
    mem.put(cache.banks[s.kv_bank].kv[slot], kv);
    mem.put(cache.banks[s.kv_bank].index[slot], index);
}

// Preserve logical order among selected rows, with the lower ID winning a tie.
template <int N, int K> void select_rows(const float (&scores)[N], int count, int (&out)[K]) {
    bool chosen[N] = {};
    for (int k = 0; k < K; ++k) {
        float best = -INFINITY;
        int id = -1;
        for (int i = 0; i < count; ++i)
            if (!chosen[i] && scores[i] > best) {
                id = i;
                best = scores[i];
            }
        if (id < 0)
            break;
        chosen[id] = true;
    }
    int n = 0;
    for (int i = 0; i < count; ++i)
        if (chosen[i])
            out[n++] = i;
    while (n < K)
        out[n++] = -1;
}

template <class Memory>
bool index_history(const Schedule &s, const Weights &w, const Token &t, const float (&x)[Hidden],
                   const MxActivation<Rank> &qr, const LayerCache &cache, Published &published, Selection &sel,
                   Candidates &pool, Memory &mem) {
    int length = s.kind == Kind::C2Full ? (t.position + 1) / 2 : t.position + 1;
    float query[IndexHeads * IndexDim], weights[IndexHeads];
    project(qr, w.iq, query, false);
    linear(x, w.iw, weights);
    float heads[IndexHeads][IndexDim];
    for (int h = 0; h < IndexHeads; ++h) {
        memcpy(heads[h], query + h * IndexDim, sizeof(heads[h]));
        rotate(heads[h], t.rope, false, false);
        IndexRow packed;
        quant4<MxGroup>(heads[h], packed.data, packed.scale, Midpoint::Upper);
        unpack(packed, heads[h]);
        weights[h] = bf16(bf16(weights[h]) / sqrtf(float(IndexHeads * IndexDim)));
    }
    bool allowed[History / Page] = {};
    if (s.kind == Kind::C1Reindex) {
        mem.get(pool, published.candidates);
        for (int b : pool.blocks) {
            if (b < -1 || b >= (length + Page - 1) / Page)
                return false;
            if (b >= 0)
                allowed[b] = true;
        }
    }
    float scores[History], blocks[History / Page];
    for (float &v : blocks)
        v = -INFINITY;
    for (int i = 0; i < length; ++i) {
        scores[i] = -INFINITY;
        if (s.kind == Kind::C1Reindex && !allowed[i / Page])
            continue;
        IndexRow packed;
        mem.get(packed, cache.banks[s.kv_bank].index[physical(cache, i)]);
        float key[IndexDim], score = 0;
        unpack(packed, key);
        for (int h = 0; h < IndexHeads; ++h) {
            float v = bf16(dot<IndexDim>(heads[h], key));
            score = bf16(score + fmaxf(0, v) * weights[h]);
        }
        scores[i] = score;
        blocks[i / Page] = fmaxf(blocks[i / Page], score);
    }
    select_rows(scores, length, sel.rows);
    for (int &row : sel.rows)
        if (row >= 0)
            row = physical(cache, row);
    mem.put(published.indices[s.index_bank], sel);
    if (s.kind == Kind::C1Full) {
        blocks[(length - 1) / Page] = INFINITY; // always include the latest block
        select_rows(blocks, (length + Page - 1) / Page, pool.blocks);
        mem.put(published.candidates, pool);
    }
    return true;
}

struct AttentionSum {
    float maximum, denom, weighted[HeadDim];
};
template <int N>
void attend_rows(const float (&query)[HeadDim], const float (&rows)[N][HeadDim], int count, AttentionSum &sum) {
    if (!count)
        return;
    float scores[N], peak = sum.maximum;
    for (int i = 0; i < count; ++i) {
        scores[i] = dot<HeadDim>(query, rows[i]) / sqrtf(float(HeadDim));
        peak = fmaxf(peak, scores[i]);
    }
    float correction = expf(sum.maximum - peak);
    sum.denom *= correction;
    for (float &v : sum.weighted)
        v *= correction;
    for (int i = 0; i < count; ++i) {
        float p = expf(scores[i] - peak);
        sum.denom += p;
        for (int d = 0; d < HeadDim; ++d)
            sum.weighted[d] += bf16(p) * rows[i][d];
    }
    sum.maximum = peak;
}

template <class Memory>
bool attention(const Schedule &s, const Weights &w, const Token &t, const HiddenState &h, LayerCache &cache,
               Published &published, LayerOutput &out, Memory &mem) {
    Mix mix;
    float x[Hidden], qr[Rank], query[Heads * HeadDim];
    hc_pre(h, w.attn, x, out.attention.pre, mix);
    auto x8 = activation(x);
    project(x8, w.qa, qr);
    rms(qr, w.q_norm);
    auto qr8 = activation(qr);
    project(qr8, w.qb, query);
    publish_window(w, x8, t, cache.window[t.window_slot]);

    for (int &row : out.selected.rows)
        row = -1;
    for (int &b : out.candidates.blocks)
        b = -1;
    if (s.kind == Kind::C2Full || s.kind == Kind::C1Full)
        compress(s, w, t, x, cache, mem);
    if (s.kind == Kind::C2Reuse || s.kind == Kind::C1Reuse)
        mem.get(out.selected, published.indices[s.index_bank]);
    else if (s.kind != Kind::Swa &&
             !index_history(s, w, t, x, qr8, cache, published, out.selected, out.candidates, mem))
        return false;

    float window[Window][HeadDim], sparse[SparseK][HeadDim];
    int nw = 0, ns = 0;
    for (int row : t.window_indices)
        if (row >= 0)
            unpack(cache.window[row], window[nw++]);
    for (int row : out.selected.rows) {
        if (row < -1 || row >= History)
            return false;
        if (row >= 0) {
            CompressedRow packed;
            mem.get(packed, cache.banks[s.kv_bank].kv[row]);
            unpack(packed, sparse[ns++]);
        }
    }
    float context[Heads][HeadDim];
    for (int i = 0; i < Heads; ++i) {
        float q[HeadDim];
        memcpy(q, query + i * HeadDim, sizeof(q));
        rotate(q, t.rope);
        AttentionSum sum{};
        sum.maximum = -INFINITY;
        attend_rows(q, window, nw, sum);
        attend_rows(q, sparse, ns, sum);
        float peak = fmaxf(sum.maximum, w.sink[i]);
        float corr = expf(sum.maximum - peak);
        float norm = corr / (sum.denom * corr + expf(w.sink[i] - peak));
        for (int d = 0; d < HeadDim; ++d)
            context[i][d] = bf16(sum.weighted[d] * norm);
        rotate(context[i], t.rope, true);
    }
    float latent[Groups * OutRank], value[Hidden];
    for (int g = 0; g < Groups; ++g) {
        float group[Heads * HeadDim / Groups], low[OutRank];
        memcpy(group, &context[g * (Heads / Groups)], sizeof(group));
        project(activation(group), w.oa[g], low);
        memcpy(latent + g * OutRank, low, sizeof(low));
    }
    project(activation(latent), w.ob, value);
    hc_post(h, mix, value, out.attention);
    return true;
}

} // namespace deepseek_v41
