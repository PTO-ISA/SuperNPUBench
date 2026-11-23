#pragma once

#include "attention.hpp"

namespace deepseek_v41 {

inline void route(const float (&x)[Hidden], const Weights &w, Route (&routes)[TopK]) {
    float logits[Experts], scores[Experts], biased[Experts];
    linear(x, w.router, logits);
    tile::Block<1, Experts> z, u, one_plus, correction, log_term, bias;
    tile::load(z, logits);
    TABS(u, z);
    TNEG(u, u);
    TEXP(u, u);
    TADDS(one_plus, u, 1.f);
    // Compensate the rounding of 1 + u so very negative logits retain log1p(u).
    TSUBS(correction, one_plus, 1.f);
    TSUB(correction, u, correction);
    TDIV(correction, correction, one_plus);
    TLOG(log_term, one_plus);
    TADD(log_term, log_term, correction);
    TMAXS(z, z, 0.f);
    TADD(z, z, log_term);
    TSQRT(z, z);
    tile::store(scores, z);
    tile::load(bias, w.bias);
    TADD(bias, bias, z);
    tile::store(biased, bias);

    uint32_t ids[TopK];
    ranked_ids(biased, Experts, ids);
    for (int k = 0; k < TopK; ++k) {
        routes[k].expert = ids[k];
        routes[k].slot = k;
    }
    tile::Block<TopK, 1> weights;
    tile::Block<1, 1> total;
    tile::Block<TopK, 1, uint32_t> offsets;
    tile::load(offsets, ids);
    TMULS(offsets, offsets, uint32_t(sizeof(float)));
    tile::GM<float, 1, Experts> unbiased(scores);
    MGATHER(weights, unbiased, offsets);
    TCOLSUM(total, weights);
    TADDS(total, total, 1.e-20f);
    TCOLEXPANDDIV(weights, weights, total);
    TMULS(weights, weights, 1.5f);
    tile::GM<float, TopK, 1, sizeof(Route) / sizeof(float)> dest(&routes[0].weight);
    TSTORE(dest, weights);
}

template <int Bits>
void run_expert(const MxActivation<Hidden> &x, const Expert<Bits> &w, float weight, Contribution &out) {
    float gate[Intermediate], up[Intermediate];
    project(x, w.gate, gate);
    project(x, w.up, up);
    tile::Block<1, Intermediate> g, u, sigmoid;
    tile::load(g, gate);
    tile::load(u, up);
    TMINS(g, g, 10.f);
    tile::sigmoid(sigmoid, g);
    TMUL(g, g, sigmoid);
    TMAXS(u, u, -10.f);
    TMINS(u, u, 10.f);
    TMUL(u, g, u);
    TMULS(u, u, weight);
    tile::store(up, u);
    project(activation(up), w.down, out.value);
}

struct Dispatch {
    Route route;
    MxActivation<Hidden> x;
    uint8_t reserved[3]{}; // fill the last word of the PGAS packet
};
static_assert(sizeof(Dispatch) == 48);

struct alignas(64) ExpertInbox {
    Dispatch packets[TopK];
    int count;
};

struct GlobalMemory {
    template <class T> void put(T &destination, const T &source) { destination = source; }
    template <class T> void get(T &destination, const T &source) { destination = source; }
    bool handoff(HiddenState &moe_input, const HiddenState &attention_output) {
        moe_input = attention_output;
        return true;
    }
    const Expert<4> &expert(const Weights &w, int id) const { return w.routed[id]; }
    ExpertInbox &inbox(int owner) { return inboxes[owner]; }
    ExpertInbox inboxes[2];
};

template <class Memory>
void moe(const Weights &w, const HiddenState &h, int shards, LayerOutput &out, Memory &mem) {
    Mix mix;
    float norm[Hidden];
    hc_pre(h, w.ffn, norm, out.next.pre, mix);
    route(norm, w, out.routes);
    auto x = activation(norm);
    run_expert(x, w.shared, 1.f, out.shared);

    // Each expert shard has its own receive window. Return addresses are the
    // original TopK slots, even when the execution order differs from routing order.
    int sent[2] = {};
    int per_shard = Experts / shards;
    for (const auto &r : out.routes) {
        int owner = r.expert / per_shard;
        Dispatch packet{r, x};
        mem.put(mem.inbox(owner).packets[sent[owner]++], packet);
    }
    for (int owner = 0; owner < shards; ++owner)
        mem.put(mem.inbox(owner).count, sent[owner]);
    for (int owner = 0; owner < shards; ++owner) {
        const auto &inbox = mem.inbox(owner);
        for (int i = 0; i < inbox.count; ++i) {
            const auto &p = inbox.packets[i];
            Contribution y;
            run_expert(p.x, mem.expert(w, p.route.expert), p.route.weight, y);
            mem.put(out.routed[p.route.slot], y);
        }
    }
    float sum[Hidden];
    tile::Block<1, Hidden> total, contribution;
    tile::load(total, out.shared.value);
    for (auto &r : out.routed) {
        tile::load(contribution, r.value);
        TADD(total, total, contribution);
    }
    tile::round_bf16(total);
    tile::store(sum, total);
    hc_post(h, mix, sum, out.next);
}

// One token at an arbitrary backbone layer. The caller owns this layer's window,
// C2A pair state, and the KV/index publications shared with subsequent layers.
template <class Memory>
bool decode_layer(int layer_idx, const Token &t, const Weights &w, LayerCache &cache, Published &published,
                  int shards, LayerOutput &out, Memory &mem) {
    if (layer_idx < 0 || layer_idx >= 40 || (shards != 1 && shards != 2) || t.position < 0 ||
        t.window_slot < 0 || t.window_slot >= Window)
        return false;
    bool window_rows[Window]{};
    if (!mark_slots(t.window_indices, window_rows))
        return false;
    Schedule s = Schedule::for_layer(layer_idx);
    if (s.kind != Kind::Swa) {
        int ratio = layer_idx < 20 ? 2 : 1;
        if (t.position / ratio >= History || !cache.valid_pages())
            return false;
    }

    HiddenState h = t.hidden;
    if (s.engram)
        inject_engram(w, t.lookup, h);
    if (!attention(s, w, t, h, cache, published, out, mem))
        return false;
    HiddenState moe_input;
    if (!mem.handoff(moe_input, out.attention))
        return false;
    moe(w, moe_input, shards, out, mem);
    return true;
}

} // namespace deepseek_v41
