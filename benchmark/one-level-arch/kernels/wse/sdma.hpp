#pragma once

#include "../deepseekv4.1/types.hpp"
#include <stddef.h>
#include <string.h>

namespace deepseek_v41::wse {

// Application protocol for a SoC SDMA service. Offsets are bytes within negotiated L1 windows.
// Only the descriptor and completion travel through the scalar message buffer.
enum class WindowId : uint32_t { WeightImage, Expert0, Expert1, AttentionOutput, MoeInput };
struct CopyRequest {
    uint32_t ticket;
    WindowId source, destination;
    uint32_t source_offset, destination_offset, bytes;
};
struct CopyDone {
    uint32_t ticket, bytes;
    int32_t status;
    bool matches(const CopyRequest &request) const {
        return ticket == request.ticket && status == 0 && bytes == request.bytes;
    }
};
static_assert(sizeof(CopyRequest) == 24 && sizeof(CopyDone) == 12);

struct Sram {
    alignas(64) Expert<4> expert_weights[Experts];
    alignas(64) HiddenState attention_output, moe_input;
};

// Functional substitute for the SoC engine. The host image and the two expert
// shards occupy separate L1 windows in the guest's address space.
class SingleCoreSdma {
  public:
    SingleCoreSdma(const Weights &weights, Sram &sram, int shards)
        : weights_(weights), sram_(sram), shards_(shards) {}

    bool exchange(const CopyRequest &req, CopyDone &done) const {
        const uint8_t *src;
        uint8_t *dst;
        size_t src_size, dst_size;
        if (req.source == WindowId::AttentionOutput && req.destination == WindowId::MoeInput) {
            src = reinterpret_cast<const uint8_t *>(&sram_.attention_output);
            dst = reinterpret_cast<uint8_t *>(&sram_.moe_input);
            src_size = dst_size = sizeof(HiddenState);
        } else if ((shards_ == 1 || shards_ == 2) && req.source == WindowId::WeightImage &&
                   (req.destination == WindowId::Expert0 ||
                    (req.destination == WindowId::Expert1 && shards_ == 2))) {
            unsigned first = req.destination == WindowId::Expert0 ? 0 : Experts / 2;
            src = reinterpret_cast<const uint8_t *>(weights_.routed);
            dst = reinterpret_cast<uint8_t *>(sram_.expert_weights + first);
            src_size = sizeof(weights_.routed);
            dst_size = src_size / shards_;
        } else {
            done = {req.ticket, 0, -1};
            return true;
        }
        if (!req.bytes || req.source_offset > src_size || req.bytes > src_size - req.source_offset ||
            req.destination_offset > dst_size || req.bytes > dst_size - req.destination_offset) {
            done = {req.ticket, 0, -1};
            return true;
        }
        memcpy(dst + req.destination_offset, src + req.source_offset, req.bytes);
        done = {req.ticket, req.bytes, 0};
        return true;
    }

  private:
    const Weights &weights_;
    Sram &sram_;
    int shards_;
};

} // namespace deepseek_v41::wse
