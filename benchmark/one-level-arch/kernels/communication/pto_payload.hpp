#pragma once

#include <common/pto_tileop.hpp>
#include "communication/mapped_window.hpp"

namespace supernpu::comm {

// FP32, full 32-element tiles. Counts must be multiples of kWidth; pad a tail
// explicitly before publishing it. This adapter issues tile work, not a remote
// completion fence. A Sync implementation used with it must drain tile work.
struct PtoPayload {
    static constexpr std::size_t kWidth = 32;
    using Tile = pto::VecTileM32<float, 32, 1>;

    static inline __attribute__((always_inline)) void load(Tile& tile, const float* address) {
        pto::global_tensor<float, pto::RowMajor<32, 1>> global(address);
        ::TLOAD(tile, global);
    }
    static inline __attribute__((always_inline)) void store(float* address, Tile& tile) {
        pto::global_tensor<float, pto::RowMajor<32, 1>> global(address);
        ::TSTORE(global, tile);
    }
    static inline __attribute__((always_inline)) void zero(Tile& tile) {
        ::TEXPANDS(tile, 0.0f);
    }
    static inline __attribute__((always_inline)) void scale(Tile& dst, Tile& src, float weight) {
        ::TMULS(dst, src, weight);
    }
    static inline __attribute__((always_inline)) void add(Tile& dst, Tile& lhs, Tile& rhs) {
        ::TADD(dst, lhs, rhs);
    }
};

}  // namespace supernpu::comm
