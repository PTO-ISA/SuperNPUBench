#pragma once

#include "deepseekv4.1/types.hpp"

namespace deepseek_v41::test {

constexpr int Steps = 2;

struct Step {
    Token token;
    Published published;
    // In a stand-alone decoder call the source may publish a new physical row
    // for this query between invocations of the layer under test.
    int upstream_slot;
    CompressedRow upstream_kv;
    IndexRow upstream_key;
};

struct Input {
    int layer_idx, shards;
    Weights weights;
    LayerCache initial;
    Step steps[Steps];
};

struct Result {
    LayerOutput steps[Steps];
    Published published[Steps];
    LayerCache final_cache;
    unsigned dynamic_puts, dynamic_gets, sdma_transfers, sdma_bytes;
};

template <class Memory> int run(const Input &input, Result &out, Memory &mem) {
    if (input.layer_idx < 0 || input.layer_idx >= 40)
        return 1;
    auto schedule = Schedule::for_layer(input.layer_idx);
    out.final_cache = input.initial;
    for (int i = 0; i < Steps; ++i) {
        const auto &step = input.steps[i];
        if (step.upstream_slot >= 0) {
            if (schedule.kv_bank < 0 || step.upstream_slot >= History)
                return 2;
            auto &bank = out.final_cache.banks[schedule.kv_bank];
            bank.kv[step.upstream_slot] = step.upstream_kv;
            bank.index[step.upstream_slot] = step.upstream_key;
        }
        out.published[i] = step.published;
        if (!decode_layer(input.layer_idx, step.token, input.weights, out.final_cache, out.published[i],
                          input.shards, out.steps[i], mem))
            return 3;
    }
    return 0;
}

} // namespace deepseek_v41::test
