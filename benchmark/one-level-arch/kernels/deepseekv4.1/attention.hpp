#pragma once

#include "hyper_connection.hpp"

namespace deepseek_v41 {

inline void unpack(const WindowRow &r, float (&x)[HeadDim]) {
    constexpr int groups = HeadDim / MxGroup;
    tile::Block<groups, MxGroup, __fp8_e4m3> packed;
    tile::Block<groups, MxGroup> values;
    tile::Block<groups, 1, __bf16> half;
    tile::Block<groups, 1> scale;
    tile::load(packed, reinterpret_cast<const __fp8_e4m3 *>(r.data));
    tile::load(half, reinterpret_cast<const __bf16 *>(r.scale));
    TCVT(values, packed);
    TCVT(scale, half);
    TROWEXPANDMUL(values, values, scale);
    tile::store(x, values);
}

template <int Group, class Scale, int N>
inline void unpack4(const uint8_t *data, const Scale *scales, float (&x)[N]) {
    constexpr int groups = N / Group;
    tile::GM<__fp4_e2m1x2, 1, N> source(reinterpret_cast<const __fp4_e2m1x2 *>(data));
    pto::CubeTileM16<__fp4_e2m1x2, 1, N> packed;
    pto::CubeTileM16<float, 1, N> decoded;
    tile::GM<float, 1, N> dest(x);
    TLOAD_CUBE(packed, source);
    TCVT(decoded, packed);
    TSTORE_CUBE(dest, decoded);

    using S = std::conditional_t<sizeof(Scale) == 1, __fp8_e8m0, __bf16>;
    tile::Block<groups, 1, S> encoded_scale;
    tile::Block<groups, 1> scale;
    tile::Block<groups, Group> values;
    tile::load(encoded_scale, reinterpret_cast<const S *>(scales));
    TCVT(scale, encoded_scale);
    tile::load(values, x);
    TROWEXPANDMUL(values, values, scale);
    if constexpr (sizeof(Scale) == 2)
        tile::round_bf16(values);
    tile::store(x, values);
}

inline void unpack(const CompressedRow &r, float (&x)[HeadDim]) { unpack4<CompGroup>(r.data, r.scale, x); }
inline void unpack(const IndexRow &r, float (&x)[IndexDim]) { unpack4<MxGroup>(r.data, r.scale, x); }

inline void publish_window(const Weights &w, const MxActivation<Hidden> &x, const Token &t, WindowRow &row) {
    float kv[HeadDim];
    project(x, w.kv, kv);
    rms(kv, w.kv_norm);
    rotate(kv, t.rope);
    quant8<HeadDim>(kv, row.data, row.scale);
}

inline int physical(const LayerCache &cache, int logical) {
    return cache.pages[logical / Page] * Page + logical % Page;
}

// Padding can repeat. Every real slot must be in range and still available.
template <int Count, int Capacity>
inline bool mark_slots(const int (&ids)[Count], bool (&occupied)[Capacity], int limit = Capacity) {
    if (limit < 0 || limit > Capacity)
        return false;
    for (int id : ids) {
        if (id == -1)
            continue;
        if (id < 0 || id >= limit || occupied[id])
            return false;
        occupied[id] = true;
    }
    return true;
}

template <class Memory>
void compress(const Schedule &s, const Weights &w, const Token &t, const float (&x)[Hidden], LayerCache &cache,
              Memory &mem) {
    float latent[HeadDim];
    int logical = t.position;
    if (s.kind == Kind::C2Full) {
        int slot = t.position % 2;
        linear(x, w.compressor, cache.pair_kv[slot]);
        linear(x, w.compressor_gate, cache.pair_gate[slot]);
        if (slot == 0)
            return;
        tile::Block<1, HeadDim> alpha, previous, current, complement;
        tile::load(alpha, cache.pair_gate[1]);
        tile::load(previous, cache.pair_gate[0]);
        TSUB(alpha, alpha, previous);
        tile::sigmoid(alpha, alpha);
        TNEG(complement, alpha);
        TADDS(complement, complement, 1.f);
        tile::load(previous, cache.pair_kv[0]);
        tile::load(current, cache.pair_kv[1]);
        TMUL(previous, previous, complement);
        TMUL(current, current, alpha);
        TADD(current, previous, current);
        tile::store(latent, current);
        logical /= 2;
    } else {
        linear(x, w.compressor, latent);
    }
    rms<true>(latent, w.comp_norm);
    float key[IndexDim];
    linear(latent, w.ik, key); // index key uses the unrotated compressor latent
    rms<true>(key, w.ik_norm);
    rotate(key, t.compressed_rope);
    rotate(latent, t.compressed_rope);
    int slot = physical(cache, logical);
    CompressedRow kv;
    IndexRow index;
    if (s.kind == Kind::C1Full)
        quant4<CompGroup, Midpoint::Even>(latent, kv.data, kv.scale);
    else
        quant4<CompGroup, Midpoint::Lower>(latent, kv.data, kv.scale);
    quant4<MxGroup, Midpoint::Even>(key, index.data, index.scale);
    mem.put(cache.banks[s.kv_bank].kv[slot], kv);
    mem.put(cache.banks[s.kv_bank].index[slot], index);
}

// IDs stay in rank order. TROWARGMAX chooses the lowest column on ties.
// When fewer than K scores are eligible the remaining all-negative-infinity row yields
// column zero; select_rows only publishes eligible, distinct logical IDs.
template <int N, int K>
inline void ranked_ids(const float (&scores)[N], int count, uint32_t (&ids)[K]) {
    tile::Block<1, N> values, missing;
    tile::Block<1, N, uint32_t> columns, selected;
    tile::Block<1, N, uint8_t> mask;
    tile::Block<1, 1, uint32_t> winner;
    tile::load(values, scores);
    TCI(columns, uint32_t{0});
    TEXPANDS(missing, -INFINITY);
    TCMPS<pto::CmpMode::GE>(mask, columns, uint32_t(count));
    TSEL(values, mask, missing);
    for (int k = 0; k < K; ++k) {
        TROWARGMAX(winner, values);
        tile::store(ids + k, winner);
        TROWEXPAND(selected, winner);
        TCMP<pto::CmpMode::EQ>(mask, columns, selected);
        TSEL(values, mask, missing);
    }
}

template <int N, int K> void select_rows(const float (&scores)[N], int count, int (&out)[K]) {
    uint32_t ids[K];
    ranked_ids(scores, count, ids);
    bool chosen[N] = {};
    for (uint32_t id : ids)
        if (id < uint32_t(count) && scores[id] > -INFINITY)
            chosen[id] = true;
    int n = 0;
    for (int i = 0; i < count; ++i)
        if (chosen[i])
            out[n++] = i;
    while (n < K)
        out[n++] = -1;
}

inline void score_page(const float (&keys)[Page][IndexDim], int count, const float (&heads)[IndexHeads][IndexDim],
                       const float (&weights)[IndexHeads], float *scores, float &block_max) {
    tile::ActiveRows<Page, IndexDim> keys_tile(count), products(count);
    tile::ActiveRows<Page, 1> sum(count), dots(count);
    tile::Block<1, IndexDim> query;
    tile::load(keys_tile, keys[0]);
    TEXPANDS(sum, 0.f);
    for (int h = 0; h < IndexHeads; ++h) {
        tile::load(query, heads[h]);
        TCOLEXPANDMUL(products, keys_tile, query);
        TROWSUM(dots, products);
        tile::round_bf16(dots);
        TMAXS(dots, dots, 0.f);
        TMULS(dots, dots, weights[h]);
        TADD(sum, sum, dots);
        tile::round_bf16(sum);
    }
    tile::store(scores, sum);
    tile::Block<1, 1> peak;
    TCOLMAX(peak, sum);
    tile::store(&block_max, peak);
}

template <class Memory>
bool index_history(const Schedule &s, const Weights &w, const Token &t, const float (&x)[Hidden],
                   const MxActivation<Rank> &qr, const LayerCache &cache, Published &published, Selection &sel,
                   Candidates &pool, Memory &mem) {
    int length = s.kind == Kind::C2Full ? (t.position + 1) / 2 : t.position + 1;
    float heads[IndexHeads][IndexDim], weights[IndexHeads];
    project_rows<false>(qr, w.iq, heads);
    linear(x, w.iw, weights);
    for (int h = 0; h < IndexHeads; ++h) {
        rotate<false, false>(heads[h], t.rope);
        IndexRow packed;
        quant4<MxGroup, Midpoint::Upper>(heads[h], packed.data, packed.scale);
        unpack(packed, heads[h]);
    }
    tile::Block<1, IndexHeads> iw;
    tile::load(iw, weights);
    tile::round_bf16(iw);
    TDIVS(iw, iw, sqrtf(float(IndexHeads * IndexDim)));
    tile::round_bf16(iw);
    tile::store(weights, iw);
    bool allowed[History / Page] = {};
    if (s.kind == Kind::C1Reindex) {
        mem.get(pool, published.candidates);
        if (!mark_slots(pool.blocks, allowed, (length + Page - 1) / Page))
            return false;
    }
    float scores[History], blocks[History / Page];
    tile::Block<1, History> invalid_rows;
    TEXPANDS(invalid_rows, -INFINITY);
    tile::store(scores, invalid_rows);
    tile::Block<1, History / Page> invalid_blocks;
    TEXPANDS(invalid_blocks, -INFINITY);
    tile::store(blocks, invalid_blocks);
    for (int first = 0; first < length; first += Page) {
        int page = first / Page;
        if (s.kind == Kind::C1Reindex && !allowed[page])
            continue;
        int count = length - first < Page ? length - first : Page;
        float keys[Page][IndexDim];
        for (int i = 0; i < count; ++i) {
            IndexRow packed;
            mem.get(packed, cache.banks[s.kv_bank].index[physical(cache, first + i)]);
            unpack(packed, keys[i]);
        }
        score_page(keys, count, heads, weights, scores + first, blocks[page]);
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
    tile::ActiveRows<N, HeadDim> kv(count), products(count);
    tile::ActiveRows<N, 1> p(count);
    tile::Block<1, HeadDim> q, weighted, addition;
    tile::Block<1, 1> peak, correction, denom, mass;
    tile::load(kv, rows[0]);
    tile::load(q, query);
    TCOLEXPANDMUL(products, kv, q);
    TROWSUM(p, products);
    TDIVS(p, p, sqrtf(float(HeadDim)));
    TCOLMAX(peak, p);
    tile::load(correction, &sum.maximum);
    TMAX(peak, peak, correction);
    tile::store(&sum.maximum, peak);
    TSUB(correction, correction, peak);
    TEXP(correction, correction);
    TCOLEXPANDSUB(p, p, peak);
    TEXP(p, p);
    TCOLSUM(mass, p);
    tile::load(denom, &sum.denom);
    TFMA(denom, denom, correction, mass);
    tile::store(&sum.denom, denom);

    // Only the numerator rounds probabilities to BF16 before multiplying KV.
    tile::round_bf16(p);
    TROWEXPANDMUL(products, kv, p);
    TCOLSUM(addition, products);
    tile::load(weighted, sum.weighted);
    TROWEXPANDMUL(weighted, weighted, correction);
    TADD(weighted, weighted, addition);
    tile::store(sum.weighted, weighted);
}

template <class Memory>
bool attention(const Schedule &s, const Weights &w, const Token &t, const HiddenState &h, LayerCache &cache,
               Published &published, LayerOutput &out, Memory &mem) {
    for (int &row : out.selected.rows)
        row = -1;
    const bool reuse = s.kind == Kind::C2Reuse || s.kind == Kind::C1Reuse;
    if (reuse) {
        mem.get(out.selected, published.indices[s.index_bank]);
        // Only the physical image of this query's logical prefix is available.
        bool occupied[History];
        for (bool &slot : occupied)
            slot = true;
        const int length = (t.position + 1) / (s.kind == Kind::C2Reuse ? 2 : 1);
        for (int logical = 0; logical < length; ++logical)
            occupied[physical(cache, logical)] = false;
        if (!mark_slots(out.selected.rows, occupied))
            return false;
    }

    Mix mix;
    float x[Hidden], qr[Rank], query[Heads][HeadDim];
    hc_pre(h, w.attn, x, out.attention.pre, mix);
    auto x8 = activation(x);
    project(x8, w.qa, qr);
    rms(qr, w.q_norm);
    auto qr8 = activation(qr);
    project_rows(qr8, w.qb, query);

    for (int &b : out.candidates.blocks)
        b = -1;
    if (s.kind == Kind::C2Full || s.kind == Kind::C1Full)
        compress(s, w, t, x, cache, mem);
    if (!reuse && s.kind != Kind::Swa &&
        !index_history(s, w, t, x, qr8, cache, published, out.selected, out.candidates, mem))
        return false;

    publish_window(w, x8, t, cache.window[t.window_slot]);
    float window[Window][HeadDim], sparse[SparseK][HeadDim];
    int nw = 0, ns = 0;
    for (int row : t.window_indices)
        if (row >= 0)
            unpack(cache.window[row], window[nw++]);
    for (int row : out.selected.rows) {
        if (row >= 0) {
            CompressedRow packed;
            mem.get(packed, cache.banks[s.kv_bank].kv[row]);
            unpack(packed, sparse[ns++]);
        }
    }
    float context[Heads][HeadDim];
    for (int i = 0; i < Heads; ++i) {
        rotate(query[i], t.rope);
        AttentionSum sum{};
        sum.maximum = -INFINITY;
        attend_rows(query[i], window, nw, sum);
        attend_rows(query[i], sparse, ns, sum);
        tile::Block<1, 1> peak, sink, corr, denom;
        tile::load(corr, &sum.maximum);
        tile::load(sink, &w.sink[i]);
        TMAX(peak, corr, sink);
        TSUB(corr, corr, peak);
        TEXP(corr, corr);
        TSUB(sink, sink, peak);
        TEXP(sink, sink);
        tile::load(denom, &sum.denom);
        TFMA(denom, denom, corr, sink);
        TDIV(corr, corr, denom);
        tile::Block<1, HeadDim> result;
        tile::load(result, sum.weighted);
        TROWEXPANDMUL(result, result, corr);
        tile::round_bf16(result);
        tile::store(context[i], result);
        rotate<true>(context[i], t.rope);
    }
    float latent[Groups][OutRank], value[Hidden];
    for (int g = 0; g < Groups; ++g) {
        constexpr int width = Heads * HeadDim / Groups;
        MxActivation<width> group;
        quant8<width>(context[g * (Heads / Groups)], group.data, group.scale);
        project(group, w.oa[g], latent[g]);
    }
    MxActivation<Groups * OutRank> joined;
    quant8<Groups * OutRank>(latent[0], joined.data, joined.scale);
    project(joined, w.ob, value);
    hc_post(h, mix, value, out.attention);
    return true;
}

} // namespace deepseek_v41
