#pragma once

#include "../deepseekv4.1/layer.hpp"
#include "sdma.hpp"

namespace deepseek_v41::wse {

// Copy exactly the valid bytes. Each endpoint address is a PGAS GM address.
// A wider physical tile can carry a short row without touching adjacent slots.
template <class T> __attribute__((noinline)) void pgas_copy(T &dst, const T &src) {
    using namespace pto;
    static_assert(__is_trivially_copyable(T) && sizeof(T) <= 128);
    constexpr int bytes = sizeof(T);
    using Packet = Tile<Location::Vec, uint8_t, 1, 128, BLayout::RowMajor, 1, bytes>;
    using View = global_tensor<uint8_t, RowMajor<1, bytes>>;
    View source(reinterpret_cast<const uint8_t *>(&src));
    View destination(reinterpret_cast<uint8_t *>(&dst));
    Packet tile;
    TLOAD(tile, source);
    TSTORE(destination, tile);
}

template <class Sdma> class Memory {
  public:
    Memory(Sram &sram, Sdma &sdma, ExpertInbox &shard0, ExpertInbox &shard1)
        : sram_(sram), sdma_(sdma), inboxes_{&shard0, &shard1} {}

    ExpertInbox &inbox(int owner) const { return *inboxes_[owner]; }

    // Plan and stage each resident expert weight shard once for the decode sequence.
    bool stage_weights(int shards) {
        if (shards != 1 && shards != 2)
            return false;
        uint32_t bytes = sizeof(sram_.expert_weights) / shards;
        for (int shard = 0; shard < shards; ++shard) {
            auto destination = shard == 0 ? WindowId::Expert0 : WindowId::Expert1;
            if (!transfer(WindowId::WeightImage, destination, shard * bytes, bytes))
                return false;
        }
        return true;
    }
    const Expert<4> &expert(const Weights &, int id) const { return sram_.expert_weights[id]; }

    template <class T> void put(T &dst, const T &src) {
        pgas_copy(dst, src);
        ++puts;
    }
    template <class T> void get(T &dst, const T &src) {
        pgas_copy(dst, src);
        ++gets;
    }

    bool handoff(HiddenState &input, const HiddenState &output) {
        sram_.attention_output = output;
        if (!transfer(WindowId::AttentionOutput, WindowId::MoeInput, 0, sizeof(HiddenState)))
            return false;
        input = sram_.moe_input;
        return true;
    }
    uint32_t puts = 0, gets = 0, completed = 0, copied_bytes = 0;

  private:
    bool transfer(WindowId source, WindowId destination, uint32_t offset, uint32_t bytes) {
        CopyRequest request{++next_ticket_, source, destination, offset, 0, bytes};
        CopyDone done{};
        if (!sdma_.exchange(request, done) || !done.matches(request))
            return false;
        ++completed;
        copied_bytes += done.bytes;
        return true;
    }

    Sram &sram_;
    Sdma &sdma_;
    ExpertInbox *inboxes_[2];
    uint32_t next_ticket_ = 0;
};

} // namespace deepseek_v41::wse
