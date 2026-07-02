#include <common/pto_tileop.hpp>

#include "template_asm.h"
#include <cstdint>
#include <cstdio>


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
                printf("%d ", DumpBuf[ri * Cols + ci]); \
            printf("\n"); \
        } \
        fflush(stdout); \
    } while (0)


#define DUMP_TILE_FLOAT(label, TileVar, DumpBuf, Rows, Cols) \
    do { \
        GlobalTensor<typename decltype(TileVar)::DType, \
                     Shape<1,1,1,Rows,Cols>, \
                     Stride<1,1,1,Cols,1>> _g(DumpBuf); \
        TCOPYOUT(_g, TileVar); \
        printf("[DUMP] %s (shape=%dx%d):\n", label, Rows, Cols); \
        for (int ri = 0; ri < Rows; ri++) { \
            printf("  row%2d: ", ri); \
            for (int ci = 0; ci < Cols; ci++) \
                printf("%f ", DumpBuf[ri * Cols + ci]); \
            printf("\n"); \
        } \
        fflush(stdout); \
    } while (0)
template<typename tile_shapeInOffset, typename tile_shapeOffset, typename dtype, int gN>
void __vec__ gen_offset(
    typename tile_shapeInOffset::TileDType __in__ in,   // inOffset
    typename tile_shapeOffset::TileDType __out__ out,   // 
    const size_t n_base
) {
    size_t data_width = sizeof(dtype);
    size_t index_x = blkv_get_index_x();
    size_t index_y = blkv_get_index_y();
    size_t index = index_y * tile_shapeOffset::RowStride + index_x;

    size_t n_offset = (index_x + n_base);
    size_t index_m = blkv_get_tile_ptr(in)[index_y] * gN;    // 这里不需要m_base，因为传入的inOffset已经被切分为tile，起始元素就是m_base
    size_t out_offset = index_m + n_offset;
    out_offset *= data_width;

    blkv_get_tile_ptr(out)[index] = out_offset;
}

template<typename tile_shapeInOffset, typename tile_shapeOffset, typename dtype, int gN>
void gen_offset_impl(
    tile_shapeInOffset &in_offset,
    tile_shapeOffset &out_offset,
    const size_t n_base
    ) {
    static_assert(tile_shapeOffset::ValidRow != -1 && tile_shapeOffset::ValidCol != -1,
                  "Only static shape supported");
                  

    gen_offset<tile_shapeInOffset, tile_shapeOffset, dtype, gN><<<tile_shapeOffset::ValidCol, tile_shapeOffset::ValidRow, 1>>>(
        in_offset.data(),
        out_offset.data(),
        n_base);
}

template<typename dtype = float, typename otype = uint32_t, size_t gK, size_t gM, size_t gN, size_t tM, size_t tN>
void gather(
    dtype *in_data_ptr,
    otype *in_offset_ptr,
    dtype *out_ptr
    ) {
    const size_t Mb = gM / tM;
    const size_t Nb = gN / tN;
    const size_t rmd_M = gM % tM;
    const size_t rmd_N = gN % tN;

    using gm_shapeInOffset = global_tensor<otype, RowMajor<1, gM>>;
    using gm_shapeIn = global_tensor<dtype, RowMajor<gK, gN>>;
    using gm_shapeOut = global_tensor<dtype, RowMajor<gM, gN>>;
    
    using tile_shapeInData = Tile<Location::Vec, dtype, 1, tM, BLayout::RowMajor>; 
    using itIn = global_iterator<gm_shapeIn, tile_shapeInData>;
    tile_shapeInData inTile;
    itIn gInIter(in_data_ptr);

    using tile_shapeInOffset = Tile<Location::Vec, otype, 1, tM, BLayout::RowMajor>; 
    using tile_shapeData = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>; 
    using tile_shapeOffset = Tile<Location::Vec, uint32_t, tM, tN, BLayout::RowMajor>;

    using tile_shapeInOffset_rmd_n  = Tile<Location::Vec, otype, 1, tM, BLayout::RowMajor>; 
    using tile_shapeData_rmd_n      = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, tM, rmd_N>;
    using tile_shapeOffset_rmd_n    = Tile<Location::Vec, uint32_t, tM, tN, BLayout::RowMajor, tM, rmd_N>;

    using tile_shapeInOffset_rmd_mn = Tile<Location::Vec, otype, 1, tM, BLayout::RowMajor, 1, rmd_M>; 
    using tile_shapeData_rmd_mn     = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, rmd_N>;
    using tile_shapeOffset_rmd_mn   = Tile<Location::Vec, uint32_t, tM, tN, BLayout::RowMajor, rmd_M, rmd_N>;

    using tile_shapeInOffset_rmd_m  = Tile<Location::Vec, otype, 1, tM, BLayout::RowMajor, 1, rmd_M>; 
    using tile_shapeData_rmd_m      = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, tN>;
    using tile_shapeOffset_rmd_m    = Tile<Location::Vec, uint32_t, tM, tN, BLayout::RowMajor, rmd_M, tN>;

    gm_shapeIn inGm(in_data_ptr);

    tile_shapeInOffset inOffsetTile;
    tile_shapeData outTile;
    tile_shapeOffset offsetTile;

    tile_shapeInOffset_rmd_n inOffsetTile_rmd_n;
    tile_shapeData_rmd_n outTile_rmd_n;
    tile_shapeOffset_rmd_n offsetTile_rmd_n;
    
    tile_shapeInOffset_rmd_mn inOffsetTile_rmd_mn;
    tile_shapeData_rmd_mn outTile_rmd_mn;
    tile_shapeOffset_rmd_mn offsetTile_rmd_mn;
    
    tile_shapeInOffset_rmd_m inOffsetTile_rmd_m;
    tile_shapeData_rmd_m outTile_rmd_m;
    tile_shapeOffset_rmd_m offsetTile_rmd_m;

    using itInOffset = global_iterator<gm_shapeInOffset, tile_shapeInOffset>;
    using itOut = global_iterator<gm_shapeOut, tile_shapeData>;

    itInOffset gInOffsetIter(in_offset_ptr);
    itOut gOIter(out_ptr);
    // for test ///////////////////////////////////////
    // alignas(256) static uint32_t  g_dump[tM * rmd_N];
    alignas(256) static uint32_t  g_dump[tM * rmd_N];
    // // alignas(256) static uint32_t  g_dump_inoffset[rmd_N];
    // alignas(256) static uint32_t  g_dump_inoffset[rmd_N];
    // alignas(256) static dtype  g_dump_outdata[tM * rmd_N];
    // ///////////////////////////////////////

    size_t n_base = 0;
    
    // #pragma clang loop unroll(full)
    for (int j = 0; j < Mb; ++j) {
    printf("j = %d\n", j);
        for (int i = 0; i < Nb; ++i) {
            auto gInOffset = gInOffsetIter(0, j);
            auto gO = gOIter(j, i);
            TCOPYIN(inOffsetTile, gInOffset);
            // test
            // auto gIn = gInIter(j, i);
            // TCOPYIN(inTile, gIn);
            n_base = i * tN;
            // printf("j = %d\n", j);
            // printf("i = %d\n", i);
            // printf("base = %d\n", base);
            // printf("in_shape[0] = %d\n", in_shape[0]);
            gen_offset_impl<tile_shapeInOffset, tile_shapeOffset, dtype, gN>(inOffsetTile, offsetTile, n_base);
        
            MGATHER(outTile, inGm, offsetTile);

            // printf("inGm = %d\n", inGm);
            // DUMP_TILE("inOffsetTile", inOffsetTile, g_dump_inoffset, 1, tN);
            // DUMP_TILE("offsetTile", offsetTile, g_dump, 1, tN);
            // DUMP_TILE_FLOAT("inTile", inTile, g_dump_outdata, 1, tN);
            // DUMP_TILE_FLOAT("outTile", outTile, g_dump_outdata, 1, tN);
            TCOPYOUT(gO, outTile);
        }
        if constexpr (rmd_N) {
            auto gInOffset = gInOffsetIter(0, j);
            auto gO = gOIter(j, Nb);
            n_base = Nb * tN;
            TCOPYIN(inOffsetTile_rmd_n, gInOffset);
            gen_offset_impl<tile_shapeInOffset_rmd_n, tile_shapeOffset_rmd_n, dtype, gN>(inOffsetTile_rmd_n, offsetTile_rmd_n, n_base);
            MGATHER(outTile_rmd_n, inGm, offsetTile_rmd_n);
            // DUMP_TILE("inOffsetTile_rmd_n", inOffsetTile_rmd_n, g_dump_inoffset, 1, rmd_N);
            // DUMP_TILE_FLOAT("outTile_rmd_n", outTile_rmd_n, g_dump_outdata, 20, rmd_N);
            // DUMP_TILE("offsetTile_rmd_n", offsetTile_rmd_n, g_dump, 20, rmd_N);
            TCOPYOUT(gO, outTile_rmd_n);
        }
    }
    if constexpr (rmd_M) {
        for (int i = 0; i < Nb; ++i) {
            auto gInOffset = gInOffsetIter(0, Mb);
            auto gO = gOIter(Mb, i);
            n_base = i * tN;
            TCOPYIN(inOffsetTile_rmd_m, gInOffset);
            gen_offset_impl<tile_shapeInOffset_rmd_m, tile_shapeOffset_rmd_m, dtype, gN>(inOffsetTile_rmd_m, offsetTile_rmd_m, n_base);
            MGATHER(outTile_rmd_m, inGm, offsetTile_rmd_m);
            TCOPYOUT(gO, outTile_rmd_m);
        }
        if constexpr (rmd_N) {
            auto gInOffset = gInOffsetIter(0, Mb);
            auto gO = gOIter(Mb, Nb);
            n_base = Nb * tN;
            TCOPYIN(inOffsetTile_rmd_mn, gInOffset);
            gen_offset_impl<tile_shapeInOffset_rmd_mn, tile_shapeOffset_rmd_mn, dtype, gN>(inOffsetTile_rmd_mn, offsetTile_rmd_mn, n_base);
            MGATHER(outTile_rmd_mn, inGm, offsetTile_rmd_mn);
            TCOPYOUT(gO, outTile_rmd_mn);
        }
    }

}



