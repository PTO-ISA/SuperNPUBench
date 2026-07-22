// =============================================================================
// per_channel_cast_kernel_pto.hpp -- standalone per_channel_cast source port
// =============================================================================
//
// Source: TileKernels/tile_kernels/quant/per_channel_cast_kernel.py @ 36d9e45.
// Upstream dispatches to per_channel_cast_fused for the supported e4m3/128-token
// case. This header keeps a standalone module-level port while preserving the
// existing per_channel_cast implementation.
// =============================================================================
#ifndef SUPERNPU_PER_CHANNEL_CAST_KERNEL_PTO_HPP
#define SUPERNPU_PER_CHANNEL_CAST_KERNEL_PTO_HPP

#include "per_token_cast_pto.hpp"

namespace supernpu::tile_isa {

template <int NumTokens, int Hidden, int TileK = 128>
void per_channel_cast_kernel(__bf16 *x, __bf16 *out, float *out_sf,
                             float max_val, float clamp_min) {
    static_assert(NumTokens % 128 == 0, "source requires token count multiple of 128");
    static_assert(Hidden % 64 == 0, "source requires hidden multiple of 64");
    static_assert(NumTokens * TileK * static_cast<int>(sizeof(__bf16)) >= 128,
                  "thread-local fragment must be at least 128 bytes");
    per_channel_cast<NumTokens, Hidden, TileK>(x, out, out_sf, max_val,
                                               clamp_min);
}

} // namespace supernpu::tile_isa

#endif
