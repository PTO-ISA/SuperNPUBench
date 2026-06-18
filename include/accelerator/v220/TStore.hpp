#ifndef TSTORE_HPP
#define TSTORE_HPP

#include "common/pto_tile.hpp"
#include "type.hpp"

using namespace pto;

namespace PTO
{
    template <is_tile_data tile_shape>
    __aicore__ void TSTORE(__gm__ tile_shape::DType *dst, tile_shape &src, unsigned gms)
    {
        constexpr uint16_t nBurst = src::kRows;
        constexpr uint32_t lenBurst = src::kCols * sizeof(tile_shape::DType);
        constexpr uint32_t gmGap = (GMS - src::kCols) * sizeof(tile_shape::DType);
        constexpr uint32_t blockSize = BLOCK_BYTE_SIZE / sizeof(tile_shape::DType);
        constexpr uint32_t ubGap = (src::kRowStride - src::kCols) / blockSize;

        static_assert(nBurst < ((1ULL << 12) - 1ULL));
        static_assert(lenBurst < ((1ULL << 21) - 1ULL));
        static_assert(gmGap < ((1ULL << 32) - 1ULL));

        if constexpr (sizeof(tile_shape::DType) == 1)
        {
            copy_ubuf_to_gm_align_b8(
                    dst, src.getAddr(), 0 /*sid*/, nBurst, lenBurst, 0 /*left padding count*/, 0 /*right padding count*/, ubGap, gmGap);
        }
        else if (sizeof(tile_shape::DType) == 2)
        {
            copy_ubuf_to_gm_align_b16(
                    dst, src.getAddr(), 0 /*sid*/, nBurst, lenBurst, 0 /*left padding count*/, 0 /*right padding count*/, ubGap, gmGap);
        }
        else
        {
            copy_ubuf_to_gm_align_b32(
                    dst, src.getAddr(), 0 /*sid*/, nBurst, lenBurst, 0 /*left padding count*/, 0 /*right padding count*/, ubGap, gmGap);
        }
    }
}
#endif