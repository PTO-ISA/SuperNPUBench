#include <common/pto_tileop.hpp>
#include <string> 
#include "benchmark.h"

#ifndef GM
#define GM 160
#endif

#ifndef GN
#define GN 160
#endif

#ifndef GK
#define GK 160
#endif

#ifndef TM
#define TM 32
#endif

#ifndef TN
#define TN 32
#endif

#ifndef TK
#define TK 32
#endif

#define RM_TILE 0
#define FRAC_TILE 1
#define RM 2
#define FRAC 3

#ifndef MODE
#define MODE RM_TILE
#endif



template <const int M, const int N, const int K>
void matmul_rm_tile(float *c_ptr, float *a_ptr, float *b_ptr) {
  using tile_shapeA = Tile<Location::Vec, float, M, K, BLayout::RowMajor>;
  using tile_shapeB = Tile<Location::Vec, float, K, N, BLayout::RowMajor>;
  using tile_shapeC = Tile<Location::Vec, float, M, N, BLayout::RowMajor>;
 
  for (int i = 0; i < 20; ++i){
        tile_shapeA tA(i);
        tile_shapeB tB(i);
        tile_shapeC tC(i); 
        MATMUL(tC, tA, tB);
  }
}

template <const int M, const int N, const int K>
void matmul_frac_tile(float *c_ptr, float *a_ptr, float *b_ptr) {
    using tile_shapeA = TileLeft<float, M, K>;
    using tile_shapeB = TileRight<float, K, N>;
    using tile_shapeC = TileAcc<float, M, N>;
 
    tile_shapeA tA(0);
    tile_shapeB tB(0);
    tile_shapeC tC; 
    for (int i = 0; i < 20; ++i){
        MATMUL(tC, tA, tB);
    }
}

template<const int gM, const int gN, const int gK, const int tM, const int tN, const int tK>
void matmul_rm(float* dst, float* src0, float* src1){
    using gm_shapeA = global_tensor<float, RowMajor<gM, gK>>;
    using gm_shapeB = global_tensor<float, RowMajor<gK, gN>>;
    using gm_shapeC = global_tensor<float, RowMajor<gM, gN>>;
    using tile_shapeA = Tile<Location::Vec, float, tM, tK, BLayout::RowMajor>;
    using tile_shapeB = Tile<Location::Vec, float, tK, tN, BLayout::RowMajor>;
    using tile_shapeACC = Tile<Location::Vec, float, tM, tN, BLayout::RowMajor>;
    using gm_iteratorA = global_iterator<gm_shapeA, tile_shapeA>;
    using gm_iteratorB = global_iterator<gm_shapeB, tile_shapeB>;
    using gm_iteratorC = global_iterator<gm_shapeC, tile_shapeACC>;

    gm_iteratorA gAIter(src0);
    gm_iteratorB gBIter(src1);
    gm_iteratorC gCIter(dst);

    const int Mb = gM/tM;
    const int Nb = gN/tN;
    const int Kb = gK/tK;

    for(int i=0;i<Mb;i++){

        for(int j=0;j<Nb;j++){
            auto gC = gCIter(i, j);

            tile_shapeACC tACC(0);

            for(int k=0;k<Kb;k++){
                auto gA = gAIter(i,k);
                auto gB = gBIter(k,j);
                tile_shapeA tA(i+j+k);
                tile_shapeB tB(i+j+k);
                MATMACC(tACC, tA, tB);
            }
        }
    }
}

template<const int gM, const int gN, const int gK, const int tM, const int tN, const int tK>
void matmul_frac(float* dst, float* src0, float* src1){
    using gm_shapeA = global_tensor<float, RowMajor<gM, gK>>;
    using gm_shapeB = global_tensor<float, ColMajor<gK, gN>>;
    using gm_shapeC = global_tensor<float, RowMajor<gM, gN>>;
    using tile_shapeA = TileLeft<float, tM, tK>;
    using tile_shapeB = TileRight<float, tK, tN>;
    using tile_shapeACC = TileAcc<float, tM, tN>;
    using gm_iteratorA = global_iterator<gm_shapeA, tile_shapeA>;
    using gm_iteratorB = global_iterator<gm_shapeB, tile_shapeB>;
    using gm_iteratorC = global_iterator<gm_shapeC, tile_shapeACC>;

    gm_iteratorA gAIter(src0);
    gm_iteratorB gBIter(src1);
    gm_iteratorC gCIter(dst);

    const int Mb = gM / tM;
    const int Nb = gN / tN;
    const int Kb = gK / tK;

    tile_shapeACC tACC;
    tile_shapeA tA(0);
    tile_shapeB tB(0);
    #pragma clang loop unroll(full)
    for(int i=0;i<Mb;i++){
        #pragma clang loop unroll(full)
        for(int j=0;j<Nb;j++){
            auto gC = gCIter(i, j);
            MATMUL(tACC, tA, tB);
            #pragma clang loop unroll(full)
            for(int k=0;k<Kb;k++){
                auto gA = gAIter(i,k);
                auto gB = gBIter(k,j);
                MATMACC(tACC, tA, tB);
            }
        }
    }
}

int main() {
    float src0[GM*GK];
    float src1[GK*GN];
    float dst[GM*GN];

    BENCHSTART;
    
    #if MODE==RM_TILE
        matmul_rm_tile<GM, GN, GK>(dst, src0, src1);
    #elif MODE==FRAC_TILE
        matmul_frac_tile<GM, GN, GK>(dst, src0, src1);
    #elif MODE==RM
        matmul_rm<GM, GN, GK, TM, TN, TK>(dst, src0, src1);
    #elif MODE==FRAC
        matmul_frac<GM, GN, GK, TM, TN, TK>(dst, src0, src1);
    #else 
        matmul_frac<GM, GN, GK, TM, TN, TK>(dst, src0, src1);
    #endif

    BENCHEND;
    return 0;
}