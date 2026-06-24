#include <common/pto_tileop.hpp>
#include "gemm.hpp"

using namespace pto;

//AI/IA = A, placeholder
template<typename dtype, const int kM, const int kN, const int kTM, const int kTN>
void Identity(dtype *dst, dtype *src){
    using gm_shape = global_tensor<dtype, RowMajor<kM, kN>>;
    using tile_shape = Tile<Location::Vec, dtype, kTM, kTN, BLayout::RowMajor>;
    using gIter = global_iterator<gm_shape, tile_shape>;

    gIter git_src(src);
    gIter git_dst(dst);

    const int Mb = kM / kTM;
    const int Nb = kN / kTN;

    for(int i=0;i<Mb;i++){
        for(int j=0;j<Nb;j++){
            tile_shape tsrc;
            TLOAD(tsrc, git_src(i,j));
            auto gdst = git_dst(i,j);
            TSTORE(gdst, tsrc);
        }
    }
}

// output = input * A^T + bias ;;; input = [batch_size, in_features] , A = [out_features, in_features]
// A should be row major
template<typename dtype, const int BatSize, const int InFeat, const int OutFeat>
void Linear(dtype *out, dtype *in, dtype *weight, dtype *bias){
    gemm<BatSize, OutFeat, InFeat, 64, 64, 64, false, true>(out, in, weight, bias);
}

// y = x1^T.A.x2 + b, where x1=1xd1, x2=1xd2, A=doxd1xd2
// in1[DimIn1, 1] in2 [DimIn2, 1] bias[DimOut, 1] weight [DimOut, DimIn1, DimIn2]
template<typename dtype, const int DimIn1, const int DimIn2, const int DimOut, const bool Bias>
void BiLinear(dtype *out, dtype *weight,
             dtype *in1, dtype *in2, dtype *bias){
    using gm_shapeA = global_tensor<dtype, RowMajor<1, DimIn1>>;
    using gm_shapeB = global_tensor<dtype, RowMajor<DimIn2, 1>>;
    using gm_shapeW = global_tensor<dtype, RowMajor<DimIn1, DimIn2>>;
    using gm_shapeO = global_tensor<dtype, RowMajor<DimOut, 1>>;

    using tile_shapeA  = Tile<Location::Vec, dtype, 4, DimIn1, BLayout::RowMajor, 1, DimIn1>;
    using tile_shapeB  = Tile<Location::Vec, dtype, DimIn2, 4, BLayout::RowMajor, DimIn2, 1>;
    using tile_shapeBT = Tile<Location::Vec, dtype, 4, DimIn2, BLayout::RowMajor, 1, DimIn2>;
    using tile_shapeW  = Tile<Location::Vec, dtype, DimIn1, DimIn2, BLayout::RowMajor>;
    using tile_shapeO  = Tile<Location::Vec, dtype, 16, 16, BLayout::RowMajor, 1, 1>;

    gm_shapeA gin1(in1);
    gm_shapeB gin2(in2);

    tile_shapeA  tin1;
    tile_shapeB  tin2;
    tile_shapeW  tW;
    tile_shapeBT tmp;
    tile_shapeO  tO;
    TLOAD(tin1, gin1);
    TLOAD(tin2, gin2);
    for (int i=0;i<DimOut;i++){
        gm_shapeW gW((weight + i * DimIn1 * DimIn2)); //weight[i, 0, 0]
        TLOAD(tW, gW);
        MATMUL(tmp, tin1, tW);
        MATMUL(tO, tmp, tin2);
        gm_shapeO gO(out+i);
        TSTORE(gO, tO);
    }
    if constexpr (Bias){
        Tile<Location::Vec, dtype, DimOut, 1, BLayout::RowMajor> tout;
        Tile<Location::Vec, dtype, DimOut, 1, BLayout::RowMajor> tbias;
        gm_shapeO gO(out);
        gm_shapeO gbias(bias);
        TLOAD(tout, gO);
        TLOAD(tbias, gbias);
        TADD(tout, tout, tbias);
        TSTORE(gO, tout);
    }
}

//
void LazyLinear(){

}