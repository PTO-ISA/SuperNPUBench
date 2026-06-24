#include <common/pto_tileop.hpp>
#include "matmul.hpp"

using namespace pto;

template <typename DataType, const int gM, const int gN, const int gK, const int tM, const int tN, const int tK, const bool Relu, const bool Bias>
void gemm(DataType *dst, DataType *src0, DataType *src1, DataType *src2)
{
    using gm_shapeA = global_tensor<DataType, RowMajor<gM, gK>>;
    using gm_shapeB = global_tensor<DataType, RowMajor<gK, gN>>;
    using gm_shapeC = global_tensor<DataType, RowMajor<gM, gN>>;
    using gm_shapeBias = global_tensor<DataType, RowMajor<1, gN>>;

    using tile_shapeA = TileLeft<DataType, tM, tK>;
    using tile_shapeB = TileRight<DataType, tK, tN>;
    using tile_shapeACC = TileAcc<DataType, tM, tN>;
    using tile_shapeACC_RM = Tile<Location::Vec, DataType, tM, tN, BLayout::RowMajor>;
    using tile_shapeBias = Tile<Location::Vec, DataType, tM, tN, BLayout::RowMajor, 1, tN>;

    using gm_iteratorA = global_iterator<gm_shapeA, tile_shapeA>;
    using gm_iteratorB = global_iterator<gm_shapeB, tile_shapeB>;
    using gm_iteratorC = global_iterator<gm_shapeC, tile_shapeACC>;
    using gm_iteratorBias = global_iterator<gm_shapeBias, tile_shapeBias>;

    gm_iteratorA gAIter(src0);
    gm_iteratorB gBIter(src1);
    gm_iteratorC gCIter(dst);
    gm_iteratorBias gBiasIter(src2);

    const int Mb = gM / tM;
    const int Nb = gN / tN;
    const int Kb = gK / tK;

    for(int i = 0; i < Mb; i++)
    {
        for(int j = 0; j < Nb; j++)
        {
            auto gC = gCIter(i, j);
            tile_shapeA tA(0);
            tile_shapeB tB(0);
            tile_shapeACC tACC;
            MATMUL(tACC, tA, tB);

            #pragma clang loop unroll(full)
            for(int k = 0; k < Kb; k++)
            {
                auto gA = gAIter(i, k);
                auto gB = gBIter(k, j);
                tile_shapeA tA;
                tile_shapeB tB;
                TLOAD(tA, gA);
                TLOAD(tB, gB);
                MATMACC(tACC, tA, tB);
            }

            tile_shapeACC_RM tACC_RM;
            TCVT(tACC_RM, tACC);

            if constexpr (Bias) {
                tile_shapeBias tBias;
                tile_shapeACC_RM tExpandBias;
                auto gBias = gBiasIter(0,j);
                TLOAD(tBias, gBias);
                TEXPANDROW(tExpandBias, tBias);
                TADD(tACC_RM, tACC_RM, tExpandBias);
            }

            if constexpr (Relu) {
                TMAXS(tACC_RM, tACC_RM, 0);
            }

            TSTORE(gC, tACC_RM);
        }
    }
}