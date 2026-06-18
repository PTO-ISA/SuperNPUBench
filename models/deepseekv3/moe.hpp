#ifndef MOE_HPP
#define MOE_HPP

#include <common/pto_tileop.hpp>
#include "tensor.h"
#include "ds_config.h"
#include "matadd.hpp"
#include "matmul.hpp"
#include "softmax.hpp"
#include "mla.hpp"

#ifdef __linx
template <typename tile_shape>
void __vec__ BitonicSortStepDescend_RowMajor_Imp(
    typename tile_shape::TileDType __out__ dst,
    const typename tile_shape::TileDType __in__ src,
    const uint16_t __in__ stage, const uint16_t __in__ step) {
    // uint16_t j = blkv_get_index_x();
    // uint16_t i = blkv_get_index_y();
    // __vbuf__ typename tile_shape::DType *dst_ptr = blkv_get_tile_ptr(dst);
    // __vbuf__ typename tile_shape::DType *src_ptr = blkv_get_tile_ptr(src);

    // uint16_t tid = j;
    // uint16_t partner = tid ^ step;
    // note:: [dst_tid] == [src_tid]
    // if (tid < partner) {
    //     if ((partner & stage) == 0) {
    //         if (src_ptr[i * tile_shape::Cols + tid] <
    //             src_ptr[i * tile_shape::Cols + partner]) {
    //             dst_ptr[i * tile_shape::Cols + tid] =
    //                 src_ptr[i * tile_shape::Cols + partner];
    //             dst_ptr[i * tile_shape::Cols + partner] =
    //                 src_ptr[i * tile_shape::Cols + tid];
    //         }
    //         else{
    //             dst_ptr[i * tile_shape::Cols + tid] =
    //                 src_ptr[i * tile_shape::Cols + tid];
    //             dst_ptr[i * tile_shape::Cols + partner] =
    //                 src_ptr[i * tile_shape::Cols + partner];                
    //         }
    //     } else {
    //         if (src_ptr[i * tile_shape::Cols + tid] >
    //             src_ptr[i * tile_shape::Cols + partner]) {
    //             dst_ptr[i * tile_shape::Cols + tid] =
    //                 src_ptr[i * tile_shape::Cols + partner];
    //             dst_ptr[i * tile_shape::Cols + partner] =
    //                 src_ptr[i * tile_shape::Cols + tid];
    //         }else {
    //             dst_ptr[i * tile_shape::Cols + tid] =
    //                 src_ptr[i * tile_shape::Cols + tid];
    //             dst_ptr[i * tile_shape::Cols + partner] =
    //                 src_ptr[i * tile_shape::Cols + partner];
    //         }
    //     }
    // }

    // typename tile_shape::DType cur_value = src_ptr[i * tile_shape::Cols + tid];
    // typename tile_shape::DType partner_value = src_ptr[i * tile_shape::Cols + partner];
    // uint16_t sort_descend = (tid < partner) && ((partner & stage) == 0) && ( cur_value < partner_value );
    // uint16_t sort_ascend = (tid < partner) && ((partner & stage) != 0) && ( cur_value > partner_value );
    // if(sort_descend || sort_ascend){
    //     dst_ptr[i * tile_shape::Cols + tid] = partner_value;
    //     dst_ptr[i * tile_shape::Cols + partner] = cur_value;
    // }else{
    //     dst_ptr[i * tile_shape::Cols + tid] = cur_value;
    //     dst_ptr[i * tile_shape::Cols + partner] = partner_value;
    // }

    // ri0 stage, ri1 step
    // __asm__ __volatile__(
    //     "addi r0, %0, ->u\n"                         // ValidCol c.movi %0, ->u xxxx
    //     "l.xor  lc0.uh, ri1.uh, ->vu.h\n"            // partner
    //     "l.madd lc1.uh, u#1.uh, lc0.uh,  ->vm.h\n"   // index = i * col + j
    //     "l.madd lc1.uh, u#1.uh, vu#1.uh, ->vm.h\n"   // index_part = i * col + partner
    //     "l.lw   [ta, vm#2.uh<<2],     ->vt.w\n"      // src[index]
    //     "l.lw   [ta, vm#1.uh<<2],     ->vt.w\n"      // src[index_part]
    //     "l.sw  vt#2.sw, [to, vm#2.uh<<2]\n"           // dst[tid] = src[tid]
    //     "l.sw  vt#1.sw, [to, vm#1.uh<<2]\n"           // dst[partner] = src[partner]
    //     ""
    //     "l.cmp.lt lc0.uh, vu#1.uh, ->vn.b\n"         // tid < partner
    //     "l.addi p, 0, ->t.d\n"                       // save p for 1st branch
    //     "l.cmp.eqi vn#1.ub, 1, ->p\n"                // set p: if(tid < partner)
    //     "l.madd lc1.uh, u#1.uh, lc0.uh,  ->vm.h\n"   // index = i * col + j
    //     "l.madd lc1.uh, u#1.uh, vu#1.uh, ->vm.h\n"   // index_part = i * col + partner
    //     "l.lw   [ta, vm#2.uh<<2],     ->vt.w\n"      // src[index]
    //     "l.lw   [ta, vm#1.uh<<2],     ->vt.w\n"      // src[index_part]
    //     "l.and  vu#1.uh, ri0.uh, ->vn.h\n"           // partner & stage
    //     "l.addi p, 0, ->t.d\n"                       // save p for 2nd branch
    //     "l.cmp.eqi vn#1.uh, 0 ->p\n"                 // set p: if ((partner & stage) == 0)
    //     "l.cmp.lt vt#2.sw, vt#1.sw, -> vn.b\n"       // src[tid] < src[partner]
    //     "l.addi p, 0 ->t.d\n"                        // save p for 3rd branch
    //     "l.cmp.eqi vn#1.ub, 1,->p\n"                 // set p if(src[tid] < src[partner])
    //     "l.sw  vt#1.sw, [to, vm#2.uh<<2]\n"           // dst[tid] = src[partner]
    //     "l.sw  vt#2.sw, [to, vm#1.uh<<2]\n"           // dst[partner] = src[tid]
    //     "l.addi t#1.ud, 0, ->p\n"                     //resave p from 3rd branch
    //     "l.xori p, -1, ->p\n"                      
    //     "l.and p, t#2.ud, ->p\n"                      //go else for 2nd branch
    //     "l.cmp.lt vt#1.sw, vt#2.sw, -> vn.b\n"        // src[partner] < src[tid] 
    //     "l.addi p, 0 ->t.d\n"                         // save p for 4th branch
    //     "l.cmp.eqi vn#1.ub, 1,->p\n"                  // set p if(src[tid] < src[partner])
    //     "l.sw  vt#1.sw, [to, vm#2.uh<<2]\n"           //dst[tid] = src[partner]
    //     "l.sw  vt#2.sw, [to, vm#1.uh<<2]\n"           //dst[partner] = src[tid]
    //     "l.addi t#1.ud, 0, ->p\n"                     //resave p from 4rd branch
    //     ""                                            //merge 2nd branch two result 
    //     "l.addi t#3.ud, 0, ->p\n"                     //resave p from 2nd branch
    //     "l.addi t#4.ud, 0, ->p\n"                     //resave p from 1st branch
    //     "c.bstop\n"
    //     :
    //     :"i"(tile_shape::ValidCol)
    //     :"memory"
    // );

    // ri0 stage, ri1 step
    __asm__ __volatile__(
        "addi r0, %0, ->u\n"                          // ValidCol
        "v.xor  lc0.uh, ri1.uh, ->vu.h\n"             // partner = tid ^ step
        "v.madd lc1.uh, u#1.uh, lc0.uh,  ->vm.h\n"    // index = i * col + j
        "v.madd lc1.uh, u#1.uh, vu#1.reuse.uh, ->vm.h\n"    // index_part = i * col + partner
        "v.add vm#2.reuse.uh, lb0.uh,  ->vn.h\n"            // index = i * col + j + col/2
        "v.add vm#1.reuse.uh, lb0.uh,  ->vn.h\n"            // index_part = i * col + partner + col/2
        "v.lw   [ta, vn#2.reuse.uh<<2],     ->vt.w\n"       // src[index+col/2] = cur_idx
        "v.lw   [ta, vn#1.reuse.uh<<2],     ->vt.w\n"       // src[index_part+col/2] = partner_idx
        "v.lw   [ta, vm#2.reuse.uh<<2],     ->vt.w\n"       // src[index] = cur_value
        "v.lw   [ta, vm#1.reuse.uh<<2],     ->vt.w\n"       // src[index_part] = partner_value
        "v.sw  vt#2.reuse.sw, [to, vm#2.reuse.uh<<2]\n"           // dst[tid] = src[tid]   // copy first 
        "v.sw  vt#1.reuse.sw, [to, vm#1.reuse.uh<<2]\n"           // dst[partner] = src[partner] // copy first
        "v.sw  vt#4.reuse.sw, [to, vn#2.reuse.uh<<2]\n"           // dst[tid+col/2] = src[tid+col/2]   // copy first 
        "v.sw  vt#3.reuse.sw, [to, vn#1.reuse.uh<<2]\n"           // dst[partner+col/2] = src[partner+col/2] // copy first
        "v.cmp.lt lc0.uh, vu#1.reuse.uh, ->vn.b\n"          // tid < partner
        "v.and  vu#1.reuse.uh, ri0.uh, ->vn.h\n"            // partner & stage
        "v.cmp.eqi vn#1.reuse.uh, 0, ->vn.b\n"              // partner & stage == 0
        "v.cmp.lt vt#2.reuse.sw, vt#1.reuse.sw, ->vn.b\n"         // cur_value < partner_value
        "v.and vn#4.reuse.ub, vn#2.reuse.ub, ->vu.b\n"            // (tid < partner) & (partner & stage) == 0
        "v.and vu#1.reuse.ub, vn#1.reuse.ub ->vu.b\n"             // (tid < partner) & ((partner & stage) == 0) & (cur_value < partner_value) 
        "v.cmp.eqi vu#1.ub, 1, ->vm.b\n"              // sort_descend
        ""
        "v.cmp.eqi vn#3.uh, 1, ->vn.b\n"                // partner & stage == 1
        "v.cmp.ge vt#2.reuse.sw, vt#1.reuse.sw, ->vn.b\n"           // cur_value >= partner_value
        "v.xor  lc0.uh, ri1.uh, ->vu.h\n"               // partner = tid ^ step
        "v.cmp.lt lc0.uh, vu#1.uh, ->vn.b\n"            // tid < partner
        "v.and vn#1.ub, vn#3.ub, -> vu.b\n"             // (tid < partner) & (partner & stage) == 1
        "v.and vu#1.ub, vn#2.ub -> vu.b\n"              // (tid < partner) & ((partner & stage) == 1) & (cur_value >= partner_value)
        "v.cmp.eqi vu#1.ub, 1, -> vm.b\n"              // sort_ascend
        ""
        "v.or vm#1.ub, vm#2.ub, ->vu.b\n"             // sort_descend || sort_ascend
        "v.add vm#4.reuse.uh, lb0.uh,  ->vn.h\n"            // index = i * col + j + col/2
        "v.add vm#3.reuse.uh, lb0.uh,  ->vn.h\n"            // index_part = i * col + partner + col/2
        "l.addi p, 0, ->t.d\n"                        // save p
        "v.cmp.eqi vu#1.ub, 1, ->p\n"                 // sort_descend || sort_ascend == 1
        "v.sw  vt#1.sw, [to, vm#4.uh<<2]\n"           // dst[tid] = src[partner]
        "v.sw  vt#2.sw, [to, vm#3.uh<<2]\n"           // dst[partner] = src[tid]
        "v.sw  vt#3.sw, [to, vn#2.uh<<2]\n"           // dst[tid+col/2] = src[partner]
        "v.sw  vt#4.sw, [to, vn#1.uh<<2]\n"           // dst[partner+col/2] = src[tid]
        "l.addi t#1.ud, 0, ->p\n"                     // resave p from 1st branch
        ""                                            // merge 2nd branch two result 
        "c.bstop\n"
        :
        :"i"(tile_shape::ValidCol)
        :"memory"
    );
}

template <typename tile_shape>
void __vec__
TRANGE_RowMajor(typename tile_shape::TileDType __out__ dst) {
    unsigned i = blkv_get_index_x();
    unsigned j = blkv_get_index_y();

    typename tile_shape::DType index = i;
    blkv_get_tile_ptr(dst)[j * tile_shape::RowStride + i] = index;
}

template <is_tile_data_v tile_shape, bool ascending = true>
__attribute__((always_inline)) 
void TSORTROW(tile_shape &weight, tile_shape &indices, tile_shape &src) {
    static constexpr uint16_t row = tile_shape::ValidRow;
    static constexpr uint16_t col = tile_shape::ValidCol;

    typename tile_shape::DType gtmp[row*4*col];
    global_tensor<dtype, RowMajor<row, 4*col>> gIn(gtmp);

    using tile_shape_sort = Tile<Location::Vec, dtype, tile_shape::Rows, 2*tile_shape::Cols, BLayout::RowMajor>;
    tile_shape_sort dst_sort;
    tile_shape_sort src_sort; 

    TRANGE_RowMajor<tile_shape><<<col, row>>>(indices.data());
    tile_shape_sort padding(-1);
    Tile<Location::Vec, dtype, tile_shape::Rows, 4*tile_shape::Cols, BLayout::RowMajor> tmp;

    TASSEMBLE(tmp, src, indices, padding);
    TEXTRACT(src_sort, tmp, 0, 0);
    // TCOPY(dst_sort, src_sort);
    if constexpr (tile_shape::isRowMajor) {
        for (uint16_t stage = 2; stage <= tile_shape::Cols; stage <<= 1) {
            for (uint16_t step = stage >> 1; step > 0; step >>= 1) {
                BitonicSortStepDescend_RowMajor_Imp<tile_shape_sort><<<col, row>>>(dst_sort.data(), src_sort.data(), stage, step);
                TCOPY(src_sort, dst_sort);

                // TCOPYOUT(gIn, dst);
                // printf("stage:%d step:%d\n", stage, step);
                // for (int j=0;j<col;j++) {
                //     printf("%.0f ", tmp[j]);
                // }
                // printf("\n");
            }
        }
        TEXTRACT(weight, dst_sort, 0, 0);
        TEXTRACT(indices, dst_sort, 0, tile_shape::Cols);
    } else {
        static_assert(tile_shape::isRowMajor,
                    "Storage layout type not supported");
    }
}
#else
template <typename tile_shape>
void BitonicSortStepDescend_RowMajor_Imp(
    typename tile_shape::TileDType dst,
    const typename tile_shape::TileDType src,
    const uint16_t stage, const uint16_t step) {

    static constexpr uint16_t row = tile_shape::ValidRow;
    static constexpr uint16_t col = tile_shape::ValidCol/2;

    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            dst[i * tile_shape::Cols + j] = src[i * tile_shape::Cols + j];
            dst[i * tile_shape::Cols + col + j] = src[i * tile_shape::Cols + col + j];
        }
    }

    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){

            uint16_t tid = j;
            uint16_t partner = tid ^ step;
            if (tid < partner) {
                if ((partner & stage) == 0) {
                    if (src[i * tile_shape::Cols + tid] <
                        src[i * tile_shape::Cols + partner]) {
                        dst[i * tile_shape::Cols + tid] =
                            src[i * tile_shape::Cols + partner];
                        dst[i * tile_shape::Cols + partner] =
                            src[i * tile_shape::Cols + tid];

                        dst[i * tile_shape::Cols + col + tid ] =
                            src[i * tile_shape::Cols + col + partner];
                        dst[i * tile_shape::Cols + col + partner] =
                            src[i * tile_shape::Cols + col + tid];
                        
                    }
                } else {
                    if (src[i * tile_shape::Cols + tid] >
                        src[i * tile_shape::Cols + partner]) {
                        dst[i * tile_shape::Cols + tid] =
                            src[i * tile_shape::Cols + partner];
                        dst[i * tile_shape::Cols + partner] =
                            src[i * tile_shape::Cols + tid];

                        dst[i * tile_shape::Cols + col + tid] =
                            src[i * tile_shape::Cols + col + partner];
                        dst[i * tile_shape::Cols + col + partner] =
                            src[i * tile_shape::Cols + col + tid];
                    }
                }
            }
        }
    }
}

template <typename tile_shape>
void TRANGE_RowMajor(typename tile_shape::TileDType dst) {
  for (uint16_t i = 0; i < tile_shape::ValidRow; ++i)
    for (uint16_t j = 0; j < tile_shape::ValidCol; ++j) {
      dst[i * tile_shape::RowStride + j] = static_cast<typename tile_shape::DType>(j);
    }
}

template <is_tile_data_v tile_shape, bool ascending = true>
void TSORTROW(tile_shape &weight, tile_shape &indices, tile_shape &src) {
    using tile_shape_sort = Tile<Location::Vec, dtype, tile_shape::Rows, 2*tile_shape::Cols, BLayout::RowMajor>;
    tile_shape_sort dst_sort;
    tile_shape_sort src_sort; 

    TRANGE_RowMajor<tile_shape>(indices.data());
    tile_shape_sort padding(0);
    Tile<Location::Vec, dtype, tile_shape::Rows, 4*tile_shape::Cols, BLayout::RowMajor> tmp;
    TASSEMBLE(tmp, src, indices, padding);
    TEXTRACT(src_sort, tmp, 0, 0);

    // printf("src_sort: \n");
    // for(int i=0;i<tile_shape_sort::Rows;i++){
    //     for(int j=0;j<tile_shape_sort::Cols;j++){
    //         printf("%3.0f ", src_sort.data()[i*tile_shape_sort::Cols+j]);
    //     }
    //     printf("\n");
    // }

    if constexpr (tile_shape::isRowMajor) {
        for (uint16_t stage = 2; stage <= tile_shape::Cols; stage <<= 1) {
            for (uint16_t step = stage >> 1; step > 0; step >>= 1) {
                BitonicSortStepDescend_RowMajor_Imp<tile_shape_sort>(dst_sort.data(), src_sort.data(), stage, step);
                TCOPY(src_sort, dst_sort);
            }
        }
        TEXTRACT(weight, dst_sort, 0, 0);
        TEXTRACT(indices, dst_sort, 0, tile_shape::Cols);

    } else {
        static_assert(tile_shape::isRowMajor,
                    "Storage layout type not supported");
    }
}
#endif

#ifdef __linx
template <typename tile_shape_dst, typename tile_shape_srci>
void __vec__ TScatterRow_Vec_RowMajor(
  typename tile_shape_dst::TileDType __out__ dst,
  const typename tile_shape_dst::TileDType __in__ src,
  const typename tile_shape_srci::TileDType __in__ srci,
  const typename tile_shape_dst::DType __in__ s) {
    uint16_t i = blkv_get_index_x();
    uint16_t j = blkv_get_index_y();

    __vbuf__ typename tile_shape_dst::DType *dst_ptr = blkv_get_tile_ptr(dst);
    __vbuf__ typename tile_shape_dst::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tile_shape_srci::DType *si_ptr = blkv_get_tile_ptr(srci);
    dst_ptr[j*tile_shape_dst::RowStride + i] = src_ptr[j*tile_shape_dst::RowStride + i]; 
    for(uint16_t k=0;k<tile_shape_srci::ValidCol;k++){
        uint16_t index = j * tile_shape_srci::RowStride + k;
        uint16_t idx = si_ptr[index];
        dst_ptr[j*tile_shape_dst::RowStride + idx] = s;
    }
}
#else
template <typename tile_shape_dst, typename tile_shape_srci>
void TScatterRow_Vec_RowMajor(
    typename tile_shape_dst::TileDType dst,
    const typename tile_shape_dst::TileDType  src,
    const typename tile_shape_srci::TileDType srci,
    const typename tile_shape_dst::DType s) {
    
    for (uint16_t i = 0; i < tile_shape_dst::ValidRow; ++i){
        for (uint16_t j = 0; j < tile_shape_dst::ValidCol; ++j) {
            dst[i*tile_shape_dst::RowStride+j] = src[i*tile_shape_dst::RowStride+j];
        }
    }

    for (uint16_t i = 0; i < tile_shape_srci::ValidRow; ++i){
        for (uint16_t j = 0; j < tile_shape_srci::ValidCol; ++j) {
            uint16_t idx = srci[i * tile_shape_srci::RowStride + j];
            dst[i*tile_shape_dst::RowStride+idx] = s;
        }
    }
}
#endif

// [tokens, scores] -> sort score [tokens, num] indices [tokens, num]
template<typename dtype, const int tokens, const int scores, const int num>
void topk(dtype *weight, dtype* indices, dtype *x){
    const int tS = 16;
    using gmIn = global_tensor<dtype, RowMajor<tokens, scores>>;
    using gmOut = global_tensor<dtype, RowMajor<tokens, num>>;
    using tileIn = Tile<Location::Vec, dtype, tS, scores, BLayout::RowMajor>;
    using tileOut = Tile<Location::Vec, dtype, tS, 32, BLayout::RowMajor, tS, num>; // num < 32
   
    #ifdef __cpu_sim__
    //writeTensorToFile<dtype, tokens, scores>(x, "moe_topk_in_cpp.txt");
    #endif

    const int block = tokens/tS;
    for(int i=0;i<block;i++){
        gmIn gIn(x+i*tS*scores);
        tileIn tIn;
        tileIn tWeight;
        tileIn tIndice;
        TCOPYIN(tIn, gIn);
        TSORTROW(tWeight, tIndice, tIn);
        tileOut tWeightOut;
        TEXTRACT(tWeightOut, tWeight, 0, 0);

        tileOut tIndiceOut;
        TEXTRACT(tIndiceOut, tIndice, 0, 0);

        gmOut gWeight(weight+i*tS*num);
        TCOPYOUT(gWeight, tWeightOut);

        gmOut gIndice(indices+i*tS*num);
        TCOPYOUT(gIndice, tIndiceOut);
    }

    #ifdef __cpu_sim__
    //writeTensorToFile<dtype, tokens, 2>(weight, "moe_topk_weight_cpp.txt");
    #endif
}

template<typename dtype, int M, int N, int tM, int tN>
void sigmoid(dtype *out, dtype* in){
    using gm_shape = global_tensor<dtype, RowMajor<M, N>>;
    using tile_shape = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;

    using gm_iterator = global_iterator<gm_shape, tile_shape>;

    const int Mb = M / tM;
    const int Nb = N / tN;

    gm_iterator gIterIn(in);
    gm_iterator gIterOut(out);
    for(int i=0;i<Mb;i++){
        for(int j=0;j<Nb;j++){
            tile_shape tmp;
            auto gIn = gIterIn(i,j);
            TCOPYIN(tmp, gIn);
            TEXP(tmp,tmp); //e^x
            TRECIP(tmp,tmp); // e^-x
            TADDS(tmp,tmp,static_cast<dtype>(1)); // 1+ e^(-x)
            TRECIP(tmp,tmp); // 1/ (1 + e^-x)
            auto gOut = gIterOut(i,j);
            TCOPYOUT(gOut, tmp);
        }
    }
}

//select idx array to index every row in "in", then mask with value and create last dim to ext_dim 
//out[idx] -> [tokens, in_dim]
template<typename dtype, int tokens, int in_dim, int idx_dim, int ext_dim>
void scatter_expand(dtype *out, dtype*idx, dtype *in, dtype value){

    const int tS = 64;

    using gmIn = global_tensor<dtype, RowMajor<tokens, in_dim>>;
    using gmIdx = global_tensor<dtype, RowMajor<tokens, idx_dim>>;
    using gmOut = global_tensor<dtype, RowMajor<tokens*in_dim, ext_dim>>;

    using tileIn = Tile<Location::Vec, dtype, tS, in_dim, BLayout::RowMajor>;
    using tileIdx = Tile<Location::Vec, dtype, tS, idx_dim, BLayout::RowMajor>;
    using tileOut = Tile<Location::Vec, dtype, tS*in_dim, ext_dim, BLayout::RowMajor>;

    const int blocks = tokens/tS;

    for(int i=0;i<blocks;i++){
        gmIn gIn(in + i*tS*in_dim);
        tileIn tIn;
        TCOPYIN(tIn, gIn);

        gmIdx gIdx(idx + i*tS*idx_dim);
        tileIdx tIdx;
        TCOPYIN(tIdx, gIdx);

        #ifdef __linx
        static constexpr uint16_t row = tS;
        static constexpr uint16_t col_idx = idx_dim;
        static constexpr uint16_t col_in = in_dim;
        TScatterRow_Vec_RowMajor<tileIn, tileIdx><<<col_in, row>>>(tIn.data(), tIn.data(), tIdx.data(), value);
        #else
        TScatterRow_Vec_RowMajor<tileIn, tileIdx>(tIn.data(), tIn.data(), tIdx.data(), value);
        #endif
        
        Tile<Location::Vec, dtype, tS*in_dim, 1, BLayout::RowMajor> tRe;
        TRESHAPE(tRe, tIn);

        tileOut tOut;
        TEXPANDCOL(tOut, tRe);

        gmOut gOut(out + i*tS*in_dim*ext_dim);
        TCOPYOUT(gOut, tOut);
    }
}

template<typename dtype, int S, int dim>
void mask_fill(dtype *data, dtype *mask, dtype mask_value){
    const int tS = 64;
    using gm_shape = global_tensor<dtype, RowMajor<tS, dim>>;
    using tile_shape = Tile<Location::Vec, dtype, tS, dim, BLayout::RowMajor>;

    const int blocks = S/tS;

    for(int i=0;i<blocks;i++){
        gm_shape gdata(data+i*tS*dim);
        tile_shape tdata;

        gm_shape gmask(mask+i*tS*dim);
        tile_shape tmask;
        TCOPYIN(tdata, gdata);
        TCOPYIN(tmask, gmask);

        tile_shape tmaskval(mask_value);

        TSELECT(tdata, tmask, tmaskval, tdata);

        TCOPYOUT(gdata, tdata);
    }
}

template<typename dtype, const int bsz, const int seq_len, typename args>
void Gate(dtype *weights, 
          dtype *indices,
          dtype *x,
          dtype *bias=nullptr){

    const int tokens = bsz*seq_len;
    // linear layer to get each weight/score for each routed experts -> [b*s, n_routed_experts]
    Tensor<dtype, args::dim, args::n_routed_experts> routed_weight;

    //#ifdef __cpu_sim__
    for(int i=0;i<args::dim;i++){
        for(int j=0;j<args::n_routed_experts;j++){
            routed_weight(i,j) = (j+1)*0.1/args::dim;
        }
    }
    //#endif

    #ifdef __cpu_sim__
    writeTensorToFile<dtype, args::dim, args::n_routed_experts>(routed_weight.data(), "moe_routed_weight_cpp.txt");
    #endif

    // printf("gate input:\n");
    // for(int i=0;i<2;i++){
    //     for(int j=0;j<args::dim;j++){
    //         printf("%.1f ", x[i*args::dim+j]);
    //     }
    //     printf("\n");
    // }

    // printf("routed_weight:\n");
    // for(int i=0;i<2;i++){
    //     for(int j=0;j<args::n_routed_experts;j++){
    //         printf("%.8f ", routed_weight.data()[i*args::n_routed_experts+j]);
    //     }
    //     printf("\n");
    // }

    Tensor<dtype, tokens, args::n_routed_experts> scores;
    matmul_frac<tokens, args::n_routed_experts, args::dim, 32, 32, 32>(scores.data(), x, routed_weight.data());

    // printf("scores:\n");
    // for(int i=0;i<2;i++){
    //     for(int j=0;j<args::n_routed_experts;j++){
    //         printf("%.8f ", scores.data()[i*args::n_routed_experts+j]);
    //     }
    //     printf("\n");
    // }

    #ifdef __cpu_sim__
    writeTensorToFile<dtype, tokens, args::n_routed_experts>(scores.data(), "moe_gate_score_cpp.txt");
    #endif

    if constexpr(args::score_func == ScoreFunc::SOFTMAX){
        softmax<dtype, tokens, args::n_routed_experts,32,32>(scores.data(), scores.data());
    }else {
        sigmoid<dtype, tokens, args::n_routed_experts,32,32>(scores.data(), scores.data());
    }

    // printf("scores_after_sigmoid:\n");
    // for(int i=0;i<2;i++){
    //     for(int j=0;j<args::n_routed_experts;j++){
    //         printf("%.8f ", scores.data()[i*args::n_routed_experts+j]);
    //     }
    //     printf("\n");
    // }

    #ifdef __cpu_sim__
    writeTensorToFile<dtype, tokens, args::n_routed_experts>(scores.data(), "moe_gate_score_after_sigmoid_cpp.txt");
    #endif

    if (bias != nullptr){
        matadd<tokens, args::n_routed_experts, 32, 32>(scores.data(), scores.data(), bias);
    }

    Tensor<dtype, tokens, args::n_routed_experts> tmp_scores;

    // printf("scores_after_sigmoid:\n");
    // for(int i=0;i<tokens;i++){
    //     for(int j=0;j<args::n_routed_experts;j++){
    //         printf("%.4f ", scores.data()[i*args::n_routed_experts+j]);
    //     }
    //     printf("\n");
    // }

    //group expert and selecet topk limited groups then flatten
    if constexpr(args::n_expert_groups > 1){
        //reshape(scores, scores); // [b*s, n_routed_experts] ->[b*s, n_experts_groups, n_routed_experts/n_experts_group]

        Tensor<dtype,  tokens, args::n_expert_groups, args::n_routed_experts/args::n_expert_groups> group_scores;
        Tensor<dtype,  tokens, args::n_expert_groups, 2> group_weight; //[b*s, n_expert_groups, 2]
        Tensor<dtype,  tokens, args::n_expert_groups, 2> group_indices; //[b*s, n_expert_groups, 2]
        if (bias != nullptr){ // FIXME: bias == nullptr using amax
            //amax<dtype, tokens, args::n_routed_experts>(group_scores.data(), scores.data());
        }else {
            topk<dtype, tokens*args::n_expert_groups, args::n_routed_experts/args::n_expert_groups, 2>(group_weight.data(), group_indices.data(), scores.data()); //get 2 biggest weights for every group
        }

        #ifdef __cpu_sim__
        writeTensorToFile<dtype, tokens, args::n_expert_groups*2>(group_weight.data(), "moe_gate_group_weight_cpp.txt");
        writeTensorToFile<dtype, tokens, args::n_expert_groups*2>(group_indices.data(), "moe_gate_group_indices_cpp.txt");
        #endif
        // printf("group_weight:\n");
        // for(int i=0;i<tokens;i++){
        //     for(int j=0;j<args::n_expert_groups*2;j++){
        //         printf("%.8f ", group_weight.data()[i*args::n_expert_groups*2+j]);
        //     }
        //     printf("\n");
        // }

        // printf("group_indces:\n");
        // for(int i=0;i<tokens;i++){
        //     for(int j=0;j<args::n_expert_groups*2;j++){
        //         printf("%.2f ", group_indices.data()[i*args::n_expert_groups*2+j]);
        //     }
        //     printf("\n");
        // }

        Tensor<dtype, tokens, args::n_expert_groups> group_weight_sum; //[b*s, n_expert_groups]
        {//rowsum(group_weight_sum, group_weight);  //[b*s, n_experts_groups]
            const int tS = 64;
            using gm_shape = global_tensor<dtype, RowMajor<tokens*args::n_expert_groups, 2>>;
            using tile_shape = Tile<Location::Vec, dtype, tS, 32, BLayout::RowMajor, tS, 2>; //[tS, 2]

            using gm_shape_out = global_tensor<dtype, RowMajor<tokens*args::n_expert_groups, 1>>;
            using tile_shape_out = Tile<Location::Vec, dtype, tS, 32, BLayout::RowMajor, tS, 1>; //[tS, 1]
            const int Mb = tokens*args::n_expert_groups / tS;
            for(int i=0;i<Mb;i++){
                uint64_t offset = i * tS * 2;
                gm_shape gIn(group_weight.data()+offset);
                tile_shape tmp;
                TCOPYIN(tmp, gIn);

                tile_shape_out rowsum;
                TROWSUM(rowsum,tmp);

                offset = i * tS * 1;
                gm_shape_out gOut(group_weight_sum.data()+offset);
                TCOPYOUT(gOut, rowsum);
            }
        }
        #ifdef __cpu_sim__
        writeTensorToFile<dtype, tokens, args::n_expert_groups>(group_weight_sum.data(), "moe_gate_group_weight_sum_cpp.txt");
        #endif
        // printf("group_weight_sum:\n");
        // for(int i=0;i<tokens;i++){
        //     for(int j=0;j<args::n_expert_groups;j++){
        //         printf("%.8f ", group_weight_sum.data()[i*args::n_expert_groups+j]);
        //     }
        //     printf("\n");
        // }

        //from expert_groups to select top limited_groups -> [b*s, n_limited_groups]
        Tensor<dtype, tokens, args::n_limited_groups> limit_group_weight;
        Tensor<dtype, tokens, args::n_limited_groups> limit_group_indices;
        topk<dtype, tokens, args::n_expert_groups, args::n_limited_groups>(limit_group_weight.data(), limit_group_indices.data(), group_weight_sum.data());

        #ifdef __cpu_sim__
        writeTensorToFile<dtype, tokens, args::n_limited_groups>(limit_group_weight.data(), "moe_gate_limit_group_weight_cpp.txt");
        writeTensorToFile<dtype, tokens, args::n_limited_groups>(limit_group_indices.data(), "moe_gate_limit_indices_weight_cpp.txt");
        #endif
        // printf("limit_group_weight:\n");
        // for(int i=0;i<tokens;i++){
        //     for(int j=0;j<args::n_limited_groups;j++){
        //         printf("%.8f ", limit_group_weight.data()[i*args::n_limited_groups+j]);
        //     }
        //     printf("\n");
        // }

        // printf("group_indces:\n");
        // for(int i=0;i<tokens;i++){
        //     for(int j=0;j<args::n_limited_groups;j++){
        //         printf("%.2f ", limit_group_indices.data()[i*args::n_limited_groups+j]);
        //     }
        //     printf("\n");
        // }

        //[b*s, n_expert_groups] all-1 matrix to index limit_groups_indices, and set to zero
        Tensor<dtype, tokens, args::n_expert_groups> mask(1);
        Tensor<dtype, tokens*args::n_expert_groups, args::n_routed_experts/args::n_expert_groups> mask_expand;  
        scatter_expand<dtype, tokens, args::n_expert_groups, args::n_limited_groups, args::n_routed_experts/args::n_expert_groups>(mask_expand.data(), limit_group_indices.data(), mask.data(), 0); //TSCATTER(重新定义每行indices选对应行的某些列)
        //scores [b*s, n_expert_groups, n_routed_experts/n_expert_groups] mask [b*s, n_expert_groups]
        // to mask irelevant groups with "-inf" except selcted limited groups
        //TEXPANDCOL;mask_fill using TSELECT

        #ifdef __cpu_sim__
        writeTensorToFile<dtype, tokens*args::n_expert_groups, args::n_routed_experts/args::n_expert_groups>(mask_expand.data(), "moe_gate_mask_expand_cpp.txt");
        #endif
        // printf("mask_expand:\n");
        // for(int i=0;i<20;i++){
        //     for(int j=0;j<(args::n_routed_experts/args::n_expert_groups);j++){
        //         printf("%.2f ", mask_expand.data()[i*(args::n_routed_experts/args::n_expert_groups)+j]);
        //     }
        //     printf("\n");
        // }

        mask_fill<dtype, tokens*args::n_expert_groups, args::n_routed_experts/args::n_expert_groups>(scores.data(), mask_expand.data(), static_cast<dtype>(-10000.0f));
        //flatten(tmp_scores, scores); // [b*s, n_expert_groups, n_routed_experts/n_expert_groups] -> [b*s, n_routed_experts]
        #ifdef __cpu_sim__
        writeTensorToFile<dtype, tokens, args::n_routed_experts>(scores.data(), "moe_gate_score_masked_cpp.txt");
        #endif
        // printf("score_masked:\n");
        // for(int i=0;i<tokens;i++){
        //     for(int j=0;j<args::n_routed_experts;j++){
        //         printf("%.8f ", scores.data()[i*args::n_routed_experts+j]);
        //     }
        //     printf("\n");
        // }

    }

    topk<dtype, tokens, args::n_routed_experts, args::n_activated_experts>(weights,indices,scores.data());

    #ifdef __cpu_sim__
    writeTensorToFile<dtype, tokens, args::n_activated_experts>(weights, "moe_gate_weight_masked_cpp.txt");
    writeTensorToFile<dtype, tokens, args::n_activated_experts>(indices, "moe_gate_indices_masked_cpp.txt");
    #endif

    //weights = original_scores.gather(1, indices) should be same as above? 
    //gather is extract corresponding score on dim=1, weight([b*s, n_activated_experts])
    if constexpr(args::score_func == ScoreFunc::SIGMOID){
        // weights /= weights.sum weight sum normalization since we have extract some weight so the extract sum is not 1
        const int tS = 64;
        using gm_shape = global_tensor<dtype, RowMajor<tokens, args::n_activated_experts>>;
        using tile_shape = Tile<Location::Vec, dtype, tS, 16, BLayout::RowMajor, tS, args::n_activated_experts>;
        const int Mb = tokens / tS;
        for(int i=0;i<Mb;i++){
            uint64_t offset = i * tS * args::n_activated_experts;
            gm_shape gIn(weights+offset);
            tile_shape tmp;
            TCOPYIN(tmp, gIn);

            tile_shape rowsum;
            TROWSUMEXPAND(rowsum,tmp);
            TDIV(tmp, tmp, rowsum);
            gm_shape gOut(weights+offset);
            TCOPYOUT(gOut, tmp);
        }
    }
}

// #ifdef __linx
// template <typename tile_shape_dst, typename tile_shape_src>
// void __vec__ TSum_Vec_RowMajor(
//   typename tile_shape_dst::TileDType __out__ dst,
//   typename tile_shape_src::TileDType __in__ src) {
//     uint16_t i = blkv_get_index_x();
//     uint16_t j = blkv_get_index_y();

//     __vbuf__ typename tile_shape_src::DType *src_ptr = blkv_get_tile_ptr(src);
//     __vbuf__ typename tile_shape_dst::DType *dst_ptr = blkv_get_tile_ptr(dst);

//     const int total_size = tile_shape_dst::Rows*tile_shape_dst::Cols;
//     typename tile_shape_dst::DType sum[total_size] = {0};

//     uint16_t index = j * tile_shape::RowStride + i;
//     uint16_t idx = src_ptr[index];
//     sum[idx] = sum[idx] + 1;
     
//     dst_ptr[idx] = dst_ptr[idx] + 1;

// }
// #else
// template <typename tile_shape_dst, typename tile_shape_src>
// void TSum_Vec_RowMajor(
//     typename tile_shape_dst::TileDType dst,
//     typename tile_shape_src::TileDType src) {

//     const int total_size = tile_shape_dst:Rows*tile_shape_dst::Cols;
//     typename tile_shape_dst::DType sum[total_size] = {0};
//     for (uint16_t i = 0; i < tile_shape_src::ValidRow; ++i){
//         for (uint16_t j = 0; j < tile_shape_src::ValidCol; ++j) {
//             uint16_t idx = src[i * tile_shape_src::RowStride + j];
//             sum[idx] = sum[idx] + 1;            
//         }
//     }

//     for(uint16_t k=0;k<total_size;k++){
//         dst[k] = sum[k];
//     }
// }
// #endif

// i==xxx->rowcondset-> rowsum -> ttrans -> rowsum expand
// template<typename dtype, int tokens, int topk, int expert_cnt>
void bincount(size_t *counts, dtype *indices, size_t size){
    // const int tS = 64;
    // using gm_idx = global_tensor<dtype, RowMajor<tokens, topk>>;
    // using gm_cnt = global_tensor<size_t, RowMajor<expert_cnt, 1>>;

    // using tile_idx = TileLeft<dtype, tS, topk>;
    // using tile_cnt = TileLeft<size_t, expert_cnt, 1>;

    // const int block_tokens = tokens/tS;

    // gm_cnt gcnt(counts);
    // tile_cnt tcnt(0);
    // for (int i=0;i<block_tokens;i++) {
    //     gm_idx gidx(indices + i*tS*topk);
    //     tile_idx tidx;
    //     TCOPYIN(tidx, gidx);

    //     tile_cnt blk_cnt;
    //     TSUM(blk_cnt, gidx);
    //     TADD(tcnt, tcnt, blk_cnt);
    // }
    // TCOPYOUT(gcnt, tcnt);

    //naive scalar implementaton
    for(int i=0;i<size;i++){
        counts[static_cast<int>(indices[i])]++;
    }

    // for(int i=0;i<expert_cnt;i++){
    //     for (int j=0;j<block_tokens;j++) {
    //         register uint64_t s;
    //         TCOPYIN(tile_idx, gidx)
    //         rowcondset(tile_idx, tile_idx, j);
    //         TSUM(s, tile_idx);
    //         counts[i] += s;
    //     }
    // }
}

//return row_index of indices as tokens_idx, col_index of indices as topk_idx
void where(dtype *tokens_idx, dtype *topk_idx, dtype *indices, int idx){

}

// w2(silu(w1(in)) * w3(in))
//silu : x / (1 + e^-x) 
template <typename dtype, const int S, const int dim, const int inter_dim>
void MLP(dtype *out, dtype *in, dtype *w1, dtype *w2, dtype *w3){
    const int tS = 64;
    const int tInt = 64;
    const int tDim = 64;

    using gmIO = global_tensor<dtype, RowMajor<S, dim>>;
    using gmW13 = global_tensor<dtype, RowMajor<dim, inter_dim>>;
    using gmW2 = global_tensor<dtype, RowMajor<inter_dim, dim>>;

    using tileIO = TileLeft<dtype, tS, tDim>;
    using tileW13 = TileRight<dtype, tDim, tInt>;
    using tileACC = TileAcc<dtype, tS, tInt>;
    using tileP = TileLeft<dtype, tS, tInt>;

    using tileW2 = TileRight<dtype, tInt, tDim>;

    using tileACC2 = TileAcc<dtype, tS, tDim>;
    using tileACC2_CVT = Tile<Location::Vec, dtype, tS, tDim>;

    using gm_iteratorIO = global_iterator<gmIO, tileIO>;
    using gm_iteratorW13 = global_iterator<gmW13, tileW13>;
    using gm_iteratorW2 = global_iterator<gmW2, tileW2>;

    gm_iteratorIO  gIterIn(in);
    gm_iteratorW13 gIterW1(w1);
    gm_iteratorW13 gIterW3(w3);
    gm_iteratorW2  gIterW2(w2);
    gm_iteratorIO  gIterOut(out);

    const int Sb = S / tS;
    const int Intb = inter_dim / tInt;
    const int Dimb = dim / tDim;

    // printf("S:%d dim:%d inter_dim:%d Sb:%d Dimb:%d Intb:%d\n", S, dim, inter_dim, Sb, Intb, Dimb);
    // M*K @ K*N -> M*N @ N

    for(int i=0;i<Sb;i++){
        for(int j=0;j<Dimb;j++){
            tileACC2 tACC2_W2;
            tileACC2_CVT tACC2_W2_OUT(0);
            //#pragma clang loop unroll(full)
            for(int k=0;k<Intb;k++){
                tileACC tACC_W1;
                if constexpr(Dimb>0){
                    auto gIn = gIterIn(i,0);
                    auto gW1 = gIterW1(0,k);
                    tileIO  tIn;
                    tileW13 tW1;
                    TCOPYIN(tIn, gIn);
                    TCOPYIN(tW1, gW1);
                    MATMUL(tACC_W1, tIn, tW1);
                }

                //#pragma clang loop unroll(full)
                for(int d=1;d<Dimb;d++){
                    auto gIn = gIterIn(i,d);
                    auto gW1 = gIterW1(d,k);
                    tileIO  tIn;
                    tileW13 tW1;
                    TCOPYIN(tIn, gIn);
                    TCOPYIN(tW1, gW1);
                    MATMACC(tACC_W1, tIn, tW1);
                }

                tileP tACC_W1_L;
                TCVT(tACC_W1_L, tACC_W1);

                tileACC tACC_W3;
                if constexpr(Dimb>0){
                    auto gIn = gIterIn(i,0);
                    auto gW3 = gIterW3(0,k);
                    tileIO  tIn;
                    tileW13 tW3;
                    TCOPYIN(tIn, gIn);
                    TCOPYIN(tW3, gW3);
                    MATMUL(tACC_W3, tIn, tW3);
                }

                //#pragma clang loop unroll(full)
                for(int d=1;d<Dimb;d++){
                    auto gIn = gIterIn(i,d);
                    auto gW3 = gIterW3(d,k);
                    tileIO  tIn;
                    tileW13 tW3;
                    TCOPYIN(tIn, gIn);
                    TCOPYIN(tW3, gW3);
                    MATMACC(tACC_W3, tIn, tW3);
                }

                tileP tACC_W3_L;
                TCVT(tACC_W3_L, tACC_W3);

                //silu(w1)
                tileP tACC_W1_SiLU;
                TEXP(tACC_W1_SiLU,tACC_W1_L); //e^x
                TRECIP(tACC_W1_SiLU,tACC_W1_SiLU); // e^-x
                TADDS(tACC_W1_SiLU, tACC_W1_SiLU, static_cast<dtype>(1)); // 1+ e^(-x)
                TRECIP(tACC_W1_SiLU,tACC_W1_SiLU); // 1/ (1 + e^-x)
                TMUL(tACC_W1_SiLU, tACC_W1_L, tACC_W1_SiLU); // x / (1 + e^-x)

                // silu(w1(x)) * w3(x)
                tileP tACC_W13;
                TMUL(tACC_W13, tACC_W1_SiLU, tACC_W3_L);

                auto gW2 = gIterW2(k,j);
                tileW2 tW2;
                TCOPYIN(tW2, gW2);
                MATMUL(tACC2_W2, tACC_W13, tW2);

                tileACC2_CVT tACC2_W2_CVT;
                TCVT(tACC2_W2_CVT, tACC2_W2);
                TADD(tACC2_W2_OUT, tACC2_W2_CVT,tACC2_W2_OUT);
            }

            auto gOut = gIterOut(i,j);
            TCOPYOUT(gOut, tACC2_W2_OUT);
        }
    }
}

#ifdef __linx
template <typename tile_shape>
void __vec__ TRowCondSet_Vec_RowMajor(
  typename tile_shape::TileDType __out__ dst,
  typename tile_shape::TileDType __in__ src,
  const uint16_t __in__ cond) {
    uint16_t i = blkv_get_index_x();
    uint16_t j = blkv_get_index_y();

    __vbuf__ typename tile_shape::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tile_shape::DType *dst_ptr = blkv_get_tile_ptr(dst);
    uint16_t index = j * tile_shape::RowStride + i;
    uint16_t idx = src_ptr[index];
    if(idx == cond){
        typename tile_shape::DType one = 1;
        dst_ptr[j*tile_shape::RowStride + i] = one;
    }else{
        typename tile_shape::DType zero = 0;
        dst_ptr[j*tile_shape::RowStride + i] = zero;
    }
}
#else
template <typename tile_shape>
void TRowCondSet_Vec_RowMajor(
    typename tile_shape::TileDType dst,
    typename tile_shape::TileDType src,
    const uint16_t cond) {
  for (uint16_t i = 0; i < tile_shape::ValidRow; ++i){
    for (uint16_t j = 0; j < tile_shape::ValidCol; ++j) {
        uint16_t idx = src[i * tile_shape::RowStride + j];
        if(idx == cond){
            typename tile_shape::DType one = 1;
            dst[i*tile_shape::RowStride+j] = one;
        }else{
            typename tile_shape::DType zero = 0;
            dst[i*tile_shape::RowStride+j] = zero;            
        }
    }
  }
}
#endif

template<typename dtype, const int bsz, const int seq_len, typename args>
void MoE(Tensor<dtype, bsz, seq_len, args::dim> & out, 
        Tensor<dtype, bsz, seq_len, args::dim> &x){
    const int tokens = bsz*seq_len;
    //view(x, x); //[bsz, seq_len, dim] -> [b*s, dim]

    Tensor<dtype, tokens, args::n_activated_experts> weights, indices;
    Gate<dtype,bsz, seq_len, args>(weights.data(), indices.data(), x.data()); //weight [b*s, topk_expert_weight] indices [b*s, topk_expert_idx]

    #ifdef __cpu_sim__
    writeTensorToFile<dtype, tokens, args::n_activated_experts>(weights.data(), "moe_gate_weight_cpp.txt");
    writeTensorToFile<dtype, tokens, args::n_activated_experts>(indices.data(), "moe_gate_indices_cpp.txt");
    #endif
    // printf("gate_weights:\n");
    // for(int i=0;i<tokens;i++){
    //     for(int j=0;j<args::n_activated_experts;j++){
    //         printf("%.8f ", weights.data()[i*args::n_activated_experts+j]);
    //     }
    //     printf("\n");
    // }

    // printf("gate_indices:\n");
    // for(int i=0;i<tokens;i++){
    //     for(int j=0;j<args::n_activated_experts;j++){
    //         printf("%.4f ", indices.data()[i*args::n_activated_experts+j]);
    //     }
    //     printf("\n");
    // }


    size_t counts[args::n_routed_experts] = {0};
    bincount(counts, indices.data(), indices.total_size);

    #ifdef __cpu_sim__
    writeTensorToFile<size_t, 1, args::n_routed_experts>(counts, "moe_counts.txt");
    #endif

    Tensor<dtype, args::dim, args::moe_inter_dim> experts_w1[args::n_routed_experts];
    Tensor<dtype, args::moe_inter_dim, args::dim> experts_w2[args::n_routed_experts];
    Tensor<dtype, args::dim, args::moe_inter_dim> experts_w3[args::n_routed_experts];
    
    #ifdef __cpu_sim__
    writeTensorToFile<dtype, args::dim, args::moe_inter_dim>(experts_w1[3].data(), "moe_tmp.txt");
    #endif

    //routed_experts out 需要特定优化的片段
    Tensor<dtype, bsz*seq_len, args::dim> y(0);
    for(int idx=0;idx<args::n_routed_experts;idx++){
        if(counts[idx] == 0)
            continue;
        // printf("current idx is %d\n", idx);
        Tensor<dtype, args::dim, args::moe_inter_dim> expert_w1 =  experts_w1[idx];
        Tensor<dtype, args::moe_inter_dim, args::dim> expert_w2 =  experts_w2[idx];
        Tensor<dtype, args::dim, args::moe_inter_dim> expert_w3 =  experts_w3[idx]; 
        
        Tensor<dtype, tokens, args::dim> x_mask_w_wt;
        //Tensor<dtype, tokens, args::dim> weight_expand;
        //generate condition matrix for indices that indices == i -> 1, indices !=i -> 0
        //[tokens, n_activated_experts] with corresponding tokens all zeros or all ones
        //finally get x_mask with unselect tokens row set to 0 and multiply with weight in advance;
        {            
            const int tS = 64;
            const int tdim = 64;
            using gmIn = global_tensor<dtype, RowMajor<tokens, args::n_activated_experts>>;
            using gmOut = global_tensor<dtype, RowMajor<tokens, args::dim>>;
            using tileIn = Tile<Location::Vec, dtype, tS, 16, BLayout::RowMajor, tS, args::n_activated_experts>;//16 >= args::n_activated_experts
            using tileOut = Tile<Location::Vec, dtype, tS, tdim, BLayout::RowMajor>;

            const int block_tokens = tokens/tS;
            const int block_dim = args::dim/tdim;
            for(int i=0;i<block_tokens;i++){
                gmIn gidx(indices.data()+i*tS*args::n_activated_experts);
                tileIn  tidx;
                TCOPYIN(tidx, gidx);

                //judege ifeq "i"  to set 1 or 0
                #ifdef __linx
                static constexpr uint16_t row = tS;
                static constexpr uint16_t col = args::n_activated_experts;
                TRowCondSet_Vec_RowMajor<tileIn><<<col, row, 1>>>(tidx.data(), tidx.data(), static_cast<uint16_t>(idx));
                #else
                static constexpr uint16_t row = tS;
                static constexpr uint16_t col = args::n_activated_experts;
                TRowCondSet_Vec_RowMajor<tileIn>(tidx.data(), tidx.data(), idx);
                #endif

                Tile<Location::Vec, dtype, tS, 16, BLayout::RowMajor, tS, 1> tidx_sum;
                TROWSUM(tidx_sum, tidx);
                tileOut tcond;
                TEXPANDCOL(tcond, tidx_sum);

                gmIn gweight(weights.data()+i*tS*args::n_activated_experts);
                tileIn tweight;
                TCOPYIN(tweight, gweight);
                TMUL(tweight, tweight, tidx);
                Tile<Location::Vec, dtype, tS, 16, BLayout::RowMajor, tS, 1> tweight_sum;
                TROWSUM(tweight_sum, tweight);
                tileOut tweight_expand;
                TEXPANDCOL(tweight_expand, tweight_sum);

                for(int j=0;j<block_dim;j++){
                    uint64_t offset = i*(tS*args::dim)+j*tdim;
                    gmOut gIn(x.data()+offset);
                    tileOut tOut;
                    TCOPYIN(tOut, gIn);
                    TMUL(tOut, tOut, tcond);
                    TMUL(tOut, tOut, tweight_expand);
        
                    gmOut gOut(x_mask_w_wt.data()+offset);
                    TCOPYOUT(gOut, tOut);
                }
            }
        }

        #ifdef __cpu_sim__
        writeTensorToFile<dtype, tokens, args::dim>(x_mask_w_wt.data(), fstring("moe_x_mask_w_wt_", idx, ".txt"));
        #endif
        // printf("moe_x_mask_w_wt_%d:\n", idx);
        // for(int i=0;i<2;i++){
        //     for(int j=0;j<3;j++){
        //         printf("%.8f ", x_mask_w_wt.data()[i*args::dim+j]);
        //     }
        //     printf("\n");
        // }


        Tensor<dtype, tokens, args::dim> expert_out;
        MLP<dtype, tokens, args::dim, args::moe_inter_dim>(expert_out.data(), x_mask_w_wt.data(), expert_w1.data(), expert_w2.data(), expert_w3.data());

        #ifdef __cpu_sim__
        writeTensorToFile<dtype, tokens, args::dim>(expert_out.data(), fstring("moe_expert_out_w_wt_", idx, ".txt"));
        #endif

        matadd<tokens, args::dim, 64, 64>(y.data(), y.data(), expert_out.data());
        /*
        dtype tokens_idx[];
        dtype topk_idx[];
        where(tokens_idx, topk_idx, indices.data(), i);  //idx, top = torch.where(indices == i) 返回index=i的expert的行索引(idx)即哪些token属于这个专家，列索引(top)这个专家属于topk的哪个,需要看下ascend c++做法
        
        //Gather 选出来的tokens，在Scatter到最终out token dim
        for(int i=0;i<tokens_idx.size();i++){
            //shared expert logics
            Tensor<dtype, 1, args::inter_dim> inter_res_w3;
            Tensor<dtype, 1, args::inter_dim> inter_res_w1;
            Tensor<dtype, 1, args::dim> expert_out;
            matmul_frac(inter_res_w3.data(), x.data(), experts_w3.data());
            matmul_frac(inter_res_w1.data(), x.data(), experts_w1.data());
            silu(inter_res_w1.data(), inter_res_w1.data());
            elem_mul(inter_res_w1.data(), inter_res_w1.data(), inter_res_w3.data());
            matmul_frac(expert_out, inter_res_w1.data(), experts_w2.data());

            // y += x[idx]*weight[idx]
            mataccs(y(tokens_idx[i]), expert_out(tokens_idx[i]), weight(tokens_idx[i], topk_idx[i]))
        }
        */
    }
    
    Tensor<dtype, args::dim, args::n_shared_experts*args::moe_inter_dim> shared_expert_w1(1);
    Tensor<dtype, args::n_shared_experts*args::moe_inter_dim, args::dim> shared_expert_w2(1);
    Tensor<dtype, args::dim, args::n_shared_experts*args::moe_inter_dim> shared_expert_w3(1);
    Tensor<dtype, bsz*seq_len, args::dim> shared_expert_out;
    MLP<dtype, bsz*seq_len, args::dim, args::n_shared_experts*args::moe_inter_dim>(shared_expert_out.data(), x.data(), shared_expert_w1.data(), shared_expert_w2.data(), shared_expert_w3.data());
    
    matadd<bsz*seq_len, args::dim, 64, 64>(out.data(), y.data(), shared_expert_out.data());
    //reshape (bsz*seq_len, dim) -> (bsz, seq_len, dim)
}

#endif