#include <common/pto_tileop.hpp>
#include <cstring> 

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

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

#ifndef MODE
#define MODE "RM_TILE"
#endif

template<const int gM, const int gN, const int gK, const int tM, const int tN, const int tK>
void CubeVecTrans(float* dst, float* src0, float* src1){
    using gm_shapeA = global_tensor<float, RowMajor<gM, gK>>;
    using gm_shapeB = global_tensor<float, ColMajor<gK, gN>>;
    using gm_shapeC = global_tensor<float, RowMajor<gM, gN>>;
    using tile_shapeA = TileLeft<float, tM, tK>;
    using tile_shapeB = TileRight<float, tK, tN>;
    using tile_shapeACC = TileAcc<float, tM, tN>;
    using gm_iteratorA = global_iterator<gm_shapeA, tile_shapeA>;
    using gm_iteratorB = global_iterator<gm_shapeB, tile_shapeB>;
    using gm_iteratorC = global_iterator<gm_shapeC, tile_shapeACC>;

    using tile_shapeD  = Tile<Location::Vec, float, tM, tN, BLayout::RowMajor>;
    using tile_shapeD_left = TileLeft<float, tM, tN>;
    using tile_shapeD_right = TileRight<float, tN, tK>;

    using tile_shapeD_out = TileAcc<float, tM, tK>;

    gm_iteratorA gAIter(src0);
    gm_iteratorB gBIter(src1);
    gm_iteratorC gCIter(dst);

    const int Mb = gM/tM;
    const int Nb = gN/tN;
    const int Kb = gK/tK;
    for(int i=0;i<Mb;i++){
        for(int j=0;j<Nb;j++){
            tile_shapeACC tACC;
            for(int k=0;k<Kb;k++){
                tile_shapeA tA(i+j+k);
                tile_shapeB tB(i+j+k);
                // TCOPYIN(tA, gA);
                // TCOPYIN(tB, gB);
                MATMUL(tACC, tA, tB);
                tile_shapeD tVecIn;
                TCVT(tVecIn, tACC);
                TRECIP(tVecIn, tVecIn);
                tile_shapeD_left tCubeIn_left;
                TCVT(tCubeIn_left, tVecIn);
                tile_shapeD_right tCubeIn_right(i+j+k);
                tile_shapeD_out tCubeOut;
                MATMUL(tCubeOut, tCubeIn_left, tCubeIn_right);
            }
            //TCOPYOUT(gC, tACC);
        }
    }
}

int main() {
    float src0[GM*GK];
    float src1[GK*GN];
    float dst[GM*GN];
    
    #ifdef LINX_PMC
    PMC_START();
    #endif

    CubeVecTrans<GM, GN, GK, TM, TN, TK>(dst, src0, src1);

    #ifdef LINX_PMC
    PMC_END();
    #endif

    return 0;
}