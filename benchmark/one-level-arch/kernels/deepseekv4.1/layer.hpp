#pragma once

#include "attention.hpp"

namespace deepseek_v41 {

inline void route(const float (&x)[Hidden], const Weights &w, Route (&routes)[TopK]) {
    float logits[Experts], scores[Experts], biased[Experts];
    linear(x, w.router, logits);
    for (int e = 0; e < Experts; ++e) {
        float z = logits[e];
        scores[e] = sqrtf(fmaxf(z, 0.f) + log1pf(expf(-fabsf(z))));
        biased[e] = scores[e] + w.bias[e];
    }
    float total = 0;
    for (int k = 0; k < TopK; ++k) {
        int best = 0;
        for (int e = 1; e < Experts; ++e)
            if (biased[e] > biased[best])
                best = e;
        routes[k] = {best, k, scores[best]};
        total += scores[best];
        biased[best] = -INFINITY;
    }
    for (auto &r : routes)
        r.weight = 1.5f * (r.weight / (total + 1.e-20f));
}

template <int Bits>
void run_expert(const MxActivation<Hidden> &x, const Expert<Bits> &w, float weight, Contribution &out) {
    float gate[Intermediate], up[Intermediate];
    project(x, w.gate, gate);
    project(x, w.up, up);
    for (int i = 0; i < Intermediate; ++i) {
        float g = fminf(gate[i], 10.f);
        up[i] = (g * sigmoid(g)) * fminf(fmaxf(up[i], -10.f), 10.f) * weight;
    }
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
    for (int d = 0; d < Hidden; ++d) {
        float v = out.shared.value[d];
        for (auto &r : out.routed)
            v += r.value[d];
        sum[d] = bf16(v);
    }
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
    for (int i : t.window_indices)
        if (i < -1 || i >= Window)
            return false;
    Schedule s = Schedule::for_layer(layer_idx);
    if (s.kind != Kind::Swa) {
        int ratio = layer_idx < 20 ? 2 : 1;
        if (t.position / ratio >= History)
            return false;
        for (int p : cache.pages)
            if (p < 0 || p >= History / Page)
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
