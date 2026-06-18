#ifndef REDUCESUMCOLVEC_KERNEL_HPP
#define REDUCESUMCOLVEC_KERNEL_HPP


#include <common/pto_tileop.hpp>
#include "template_asm.h"


using namespace pto;

#pragma once
#include <cstdint>
#include <cstdio>


// ==============================================
// ==============================================
template<typename tileSrc, typename timeProd>
void __vec__ reduceprod_col_kernel(
    typename timeProd::TileDType __out__ new_prod,
    const typename tileSrc::TileDType __in__ src,
    const typename timeProd::TileDType __in__ old_prod    
)
{
    size_t i = blkv_get_index_x();  

    __vbuf__ typename timeProd::DType *new_prod_ptr = blkv_get_tile_ptr(new_prod);
    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename timeProd::DType *old_prod_ptr = blkv_get_tile_ptr(old_prod);   


    typename timeProd::DType upd_prod = old_prod_ptr[i];

    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileSrc::ValidRow;j++){
        size_t src_idx =  i * tileSrc::ColStride + j * tileSrc::RowStride;
        upd_prod = upd_prod * src_ptr[src_idx];          
    }    
    new_prod_ptr[i] = upd_prod;    
}



template<typename dtype, int gIM, int gIN, int tM, int tN>
void reduceprod_col_rand(
    dtype *in_ptr,
//    dtype *inzero_ptr,    
    dtype *out_ptr
) 
{

    const int Mb = gIM / tM;
    const int Nb = gIN / tN;    

    const int rmd_M = gIM % tM;
    const int rmd_N = gIN % tN;
//    const int rmd_M = gOM % tM; // todo 尾块怎么处理？

    using gm_shapeIn = global_tensor<dtype, RowMajor<gIM, gIN>>;     // 
//    using gm_shapeSum = global_tensor<dtype, RowMajor<gIM, gIN>>;    
    using gm_shapeOut = global_tensor<dtype, RowMajor<1, gIN>>;
    using tile_shapeData = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>; //
    using tile_shapeData_col = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, tN>; //     
    using tile_shapeProd = Tile<Location::Vec, dtype, 1, tN, BLayout::RowMajor>; // 



    using tile_shapeData_row = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, tM, rmd_N>; // 
    using tile_shapeData_cor = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, rmd_N>; //     
    using tile_shapeProd_row = Tile<Location::Vec, dtype, 1, tN, BLayout::RowMajor, 1, rmd_N>; // 
    //need tM = 1;


    gm_shapeIn inGm(in_ptr);   
//    gm_shapeOut ZeroGm(inzero_ptr); 
    gm_shapeOut outGm(out_ptr);
//    gm_shapeSum olcSumGm(old_sum_ptr);    

    tile_shapeData dataTile;
    tile_shapeData_col dataTile_col;    
    tile_shapeProd ProdTile;
    tile_shapeProd oldProdTile;

    tile_shapeData_row dataTile_row;
    tile_shapeData_cor dataTile_cor;    
    tile_shapeProd_row ProdTile_row;
    tile_shapeProd_row oldProdTile_row;    

//    int base = 0;// todo 生成一个标量
//    int all_num = gOM; // 总元素数量

    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;      
    using itIn_row = global_iterator<gm_shapeIn, tile_shapeProd>;
    using itOut = global_iterator<gm_shapeOut, tile_shapeProd>;

    itIn  gIIter(in_ptr);
    itIn_row  gIIter_rmd_row(in_ptr);
//    itZero  gZeroIter(inzero_ptr);    
    itOut gOIter(out_ptr);

//    dtype zero = 0;

    for (int j = 0; j < Nb; ++j) {
//        auto gZero = gZeroIter(0, j);
        auto gO = gOIter(0, j);
        TEXPANDSCALAR(oldProdTile, 0);//初始化为0
//        TCOPYIN(oldSumTile, gZero);//初始化为0
        //初始化old_sum的tile      
        //need 
        for (int i = 0; i < Mb; ++i) {
            auto gI = gIIter(i, j);
            TCOPYIN(dataTile, gI);
            reduceprod_col_kernel<tile_shapeData, tile_shapeProd><<<tile_shapeProd::ValidCol, tile_shapeProd::ValidRow, 1>>>(ProdTile.data(), dataTile.data(), oldProdTile.data());
            oldProdTile = ProdTile;
        }
        if constexpr (rmd_M > 0){   
            auto gI = gIIter(Mb, j);
            TCOPYIN(dataTile_col, gI);
            reduceprod_col_kernel<tile_shapeData_col,tile_shapeProd><<<tile_shapeProd::ValidCol, tile_shapeProd::ValidRow, 1>>>(ProdTile.data(), dataTile_col.data(), oldProdTile.data());
            oldProdTile = ProdTile;
        }
        TCOPYOUT(gO, ProdTile);
    }
    if constexpr (rmd_N > 0){
//        auto gZero = gZeroIter(0, Nb);         
        auto gO = gOIter(0, Nb);
        TEXPANDSCALAR(oldProdTile_row, 0);//初始化为0        
//        TCOPYIN(oldSumTile_row, gZero);//初始化为0
        for (int i = 0; i < Mb; ++i) {   
            auto gI = gIIter(i, Nb);
            TCOPYIN(dataTile_row, gI);
            reduceprod_col_kernel<tile_shapeData_row,tile_shapeProd_row><<<tile_shapeProd_row::ValidCol, tile_shapeProd_row::ValidRow, 1>>>(ProdTile_row.data(), dataTile_row.data(), oldProdTile_row.data());
            oldProdTile_row = ProdTile_row;
        }
        if constexpr (rmd_M > 0){   
            auto gI = gIIter(Mb, Nb);
            TCOPYIN(dataTile_cor, gI);
            reduceprod_col_kernel<tile_shapeData_cor,tile_shapeProd_row><<<tile_shapeProd_row::ValidCol, tile_shapeProd_row::ValidRow, 1>>>(ProdTile_row.data(), dataTile_cor.data(), oldProdTile_row.data());
            oldProdTile_row = ProdTile_row;
        }
        TCOPYOUT(gO, ProdTile_row);
    }
}

#endif
