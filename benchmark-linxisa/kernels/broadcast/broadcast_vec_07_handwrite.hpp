#include <common/pto_tileop.hpp>

#include "template_asm.h"
#include <cstdint>
#include <cstdio>


// 针对 (N,1) -> (N,129) 的特化优化 (N=1042 等)

#define DUMP_TILE(label, TileVar, DumpBuf, Rows, Cols) \
    do { \
        GlobalTensor<typename decltype(TileVar)::DType, \
                     Shape<1,1,1,Rows,Cols>, \
                     Stride<1,1,1,Cols,1>> _g(DumpBuf); \
        TCOPYOUT(_g, TileVar); \
        printf("[DUMP] %s (shape=%dx%d):\n", label, Rows, Cols); \
        for (int ri = 0; ri < Rows; ri++) { \
            printf("  row%2d: ", ri); \
            for (int ci = 0; ci < Cols; ci++) \
                printf("%12lld ", (long long)DumpBuf[ri * Cols + ci]); \
            printf("\n"); \
        } \
        fflush(stdout); \
    } while (0)

template<typename dtype, typename tile_shapeIn, typename tile_shapeOut>
void __vec__ vec_broadcast(
    typename tile_shapeInOffset::TileDType __in__ in,
    typename tile_shapeOffset::TileDType __out__ out,
    size_t t_in_offset,
    size_t t_out_offset
) {
    size_t index = blkv_get_index_x();
    size_t out_addr = index + t_out_offset;
    dtype data = blkv_get_tile_ptr(in)[t_in_offset];
    blkv_get_tile_ptr(out)[out_addr] = data;
}



// (N,1)->(N,129)
template<typename dtype, size_t MAX_DIM, size_t IN_DIM0, size_t IN_DIM1, size_t OUT_DIM0, size_t OUT_DIM1, size_t tI1, size_t tO0, size_t tO1>
void broadcast(
    dtype *in_ptr,
    dtype *out_ptr,
    const size_t *in_shape,
    const size_t *out_shape
    ) {
    const size_t M = IN_DIM1 / tI1;    // 外层循环
    const size_t rmd_M = IN_DIM1 % tI1;

    if (tI1 * 129 % 128 == 0) {
        const size_t tO0 = (tI1 * 129 / 128) * 128;    // 输出一行元素数量
    } else {
        const size_t tO0 = (tI1 * 129 / 128 + 1) * 128;
    }
    const size_t N = tI1;    // 内层循环，被广播的元素效数量
    const size_t rmd_N = rmd_M;
    const size_t vld_N = N * 129;    // 实际一次写回数量
    const size_t rmd_vld_N = rmd_M * 129;    // 尾块，实际一次写回数量
    

    Assert(tO0 > 129);
    Assert(tO0 % 128 == 0);
    using gm_shapeIn = global_tensor<dtype, RowMajor<IN_DIM1, IN_DIM0>>;
    using gm_shapeOut = global_tensor<dtype, RowMajor<OUT_DIM1, OUT_DIM0>>;
    using tile_shapeIn = Tile<Location::Vec, dtype, 1, tI1, BLayout::RowMajor>;
    using tile_shapeIn_rmd = Tile<Location::Vec, dtype, 1, tI1, BLayout::RowMajor, 1, rmd_M>;
    using tile_shapeOut = Tile<Location::Vec, dtype, 1, tO0, BLayout::RowMajor, 1, vld_N>;
    using tile_shapeOut_rmd = Tile<Location::Vec, dtype, 1, tO0, BLayout::RowMajor, 1, rmd_vld_N>;

    gm_shapeIn inGm(in_ptr);
    gm_shapeOut outGm(out_ptr);
    tile_shapeIn inTile;
    tile_shapeIn_rmd inTile_rmd;
    tile_shapeOut outTile;
    tile_shapeOut_rmd outTile_rmd;
    size_t t_in_offset = 0;
    size_t t_out_offset = 0;

    using itIn = global_iterator<gm_shapeIn, tile_shapeIn>;

    itIn gIIter(in_ptr);

    for (int i = 0; i < M; ++i) {
        auto gI = gIIter(0, i);
        t_in_offset = 0;
        t_out_offset = 0;
        TCOPYIN(inTile, gI);
        for (int j = 0; j < N; ++i) {
            vec_broadcast<dtype, tile_shapeIn, tile_shapeOut><<<129, 1, 1>>>(inTile, outTile, t_in_offset, t_out_offset);
            t_in_offset += 1;
            t_out_offset += 129;
        }
        TCOPYOUT(outTile, out_ptr);
        out_ptr += sizeof(dtype) * vld_N;
    }
    if constexpr (rmd_M) {
        auto gI = gIIter(0, Mb);
        t_in_offset = 0;
        t_out_offset = 0;
        TCOPYIN(inTile, gI);
        for (int j = 0; j < rmd_N; ++i) {
            vec_broadcast<dtype, tile_shapeIn_rmd, tile_shapeOut_rmd><<<129, 1, 1>>>(inTile_rmd, outTile_rmd, t_in_offset, t_out_offset);
            t_in_offset += 1;
            t_out_offset += 129;
        }
        TCOPYOUT(outTile, out_ptr);
        out_ptr += sizeof(dtype) * rmd_vld_N;
    }
}



