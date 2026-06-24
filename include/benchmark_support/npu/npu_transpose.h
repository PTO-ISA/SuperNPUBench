#include <common/pto_tileop.hpp>
#include "template_asm.h"
#include "multi_tile.hpp"

#ifndef __vbuf__
#define __vbuf__
#endif

template<typename TileU16, const int tD2, const int tD1, const int tD0>
void __vec__ transpose_src_idx_2d_102(
                                    typename TileU16::TileDType __out__ load_idx
                                )
{
    size_t i = blkv_get_index_x();
    size_t j = blkv_get_index_y();
    size_t k = blkv_get_index_z();
    size_t idx = k * (tD1 * tD0) + j * tD0 + i;
    // size_t load_offset = k * (tD1 * tD0) + j * tD0 + i;
    // size_t load_offset = j * (tD2 * tD0) + k * tD0 + i;
    size_t load_offset = idx;
    __vbuf__ typename TileU16::DType *load_idx_ptr = blkv_get_tile_ptr(load_idx);
    load_idx_ptr[idx] = static_cast<typename TileU16::DType>(load_offset);
}

template<typename TileU16, const int tD2, const int tD1, const int tD0>
void __vec__ transpose_dst_idx_2d_102(
                                    typename TileU16::TileDType __out__ store_idx
                                )
{
    size_t i = blkv_get_index_x();
    size_t j = blkv_get_index_y();
    size_t k = blkv_get_index_z();
    size_t idx = k * (tD1 * tD0) + j * tD0 + i;
    // size_t store_offset = j * (tD2 * tD0) + k * tD0 + i;
    // size_t idx = 1;
    // size_t store_offset = k * (tD1 * tD0) + j * tD0 + i;
    size_t store_offset = idx;
    __vbuf__ typename TileU16::DType *store_idx_ptr = blkv_get_tile_ptr(store_idx);
    store_idx_ptr[store_offset] = static_cast<typename TileU16::DType>(idx);
}
