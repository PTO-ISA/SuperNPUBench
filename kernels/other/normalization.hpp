#include <common/pto_tileop.hpp>

using namespace pto;

// x / (Ex^2) ^ .5
template<typename dtype, const int kM, const int kN, const int kTM, const int kTN>
void rmsnorm(dtype *dst, dtype *src){
    using gm_shape = global_tensor<dtype, RowMajor<kM, kN>>;
    using tile_shape = Tile<Location::Vec, dtype, kTM, kTN, BLayout::RowMajor>;
 
    using tSum = Tile<Location::Vec, dtype, kTM, 1, BLayout::RowMajor>;
 
    using gIter = global_iterator<gm_shape, tile_shape>;

    gIter giter_src(src);
    gIter giter_dst(dst);

    const int Mb = kM / kTM;
    const int Nb = kN / kTN;
    // printf("in rmsnorm, M:%d, N:%d\n", kM, kN);
    for(int i=0;i<Mb;i++)
    {
        tSum tAccSquareSum(0);  //tiling square sum

        for(int j=0;j<Nb;j++)
        {
            auto gsrc = giter_src(i, j);
            tile_shape tsrc;
 
            TCOPYIN(tsrc, gsrc);

            tSum tLocalSum;
            TMUL(tsrc, tsrc, tsrc);
            TROWSUM(tLocalSum, tsrc);
            TADD(tAccSquareSum, tAccSquareSum, tLocalSum);
        }
 
        tSum gSqureMean;
        TDIVS(gSqureMean, tAccSquareSum, kN);
        TSQRT(gSqureMean, gSqureMean);
 
        tile_shape gSqureMean_i;
        TEXPANDCOL(gSqureMean_i, gSqureMean);

        for(int j=0;j<Nb;j++)
        {
            auto  gsrc = giter_src(i,j);
            tile_shape tsrc;
            TCOPYIN(tsrc, gsrc);
 
            TDIV(tsrc, tsrc, gSqureMean_i);
 
            auto gdst = giter_dst(i,j);
            TCOPYOUT(gdst, tsrc);
        }
    }
}

// (x - Ex) / (Ex^2 - (Ex)^2)^.5
template<typename dtype, const int kM, const int kN, const int kTM, const int kTN>
void layernorm(dtype *dst, dtype *src)
{
    using gm_shape = global_tensor<dtype, RowMajor<kM, kN>>;
    using tile_shape = Tile<Location::Vec, dtype, kTM, kTN, BLayout::RowMajor>;
 
    using tSum = Tile<Location::Vec, dtype, kTM, 1, BLayout::RowMajor>;
 
    using gIter = global_iterator<gm_shape, tile_shape>;

    gIter giter_src(src);
    gIter giter_dst(dst);

    const int Mb = kM / kTM;
    const int Nb = kN / kTN;

    for(int i=0;i<Mb;i++)
    {
        tSum tAccSum(0);        // tiling sum
        tSum tAccSquareSum(0);  // tiling square sum
  
        for(int j=0;j<Nb;j++)
        {
            auto gsrc = giter_src(i, j);
            tile_shape tsrc;
 
            TCOPYIN(tsrc, gsrc);
 
            tSum tLocalSum;
            TROWSUM(tLocalSum, tsrc);
            TADD(tAccSum, tAccSum, tLocalSum);
 
            TMUL(tsrc, tsrc, tsrc);
            TROWSUM(tLocalSum, tsrc);
            TADD(tAccSquareSum, tAccSquareSum, tLocalSum);
        }
 
        tSum gMean;        // Ex
        tSum gMeanSquare;  // (Ex)^2
        tSum gStdDev;      // Ex^2
        TDIVS(gMean, tAccSum, kN);
        TMUL(gMeanSquare, gMean, gMean);
        TDIVS(gStdDev, tAccSquareSum, kN);
        TSUB(gStdDev, gStdDev, gMeanSquare);
        TSQRT(gStdDev, gStdDev);
 
        tile_shape gMean_i;
        tile_shape gStdDev_i;
        TEXPANDCOL(gMean_i, gMean);
        TEXPANDCOL(gStdDev_i, gStdDev);

        for(int j=0;j<Nb;j++)
        {
            auto  gsrc = giter_src(i,j);
            tile_shape tsrc;
            TCOPYIN(tsrc, gsrc);
 
            TSUB(tsrc, tsrc, gMean_i);    // (x - Ex)
            TDIV(tsrc, tsrc, gStdDev_i);  // (x - Ex) / (Ex^2 - (Ex)^2)^.5
 
            auto gdst = giter_dst(i,j);
            TCOPYOUT(gdst, tsrc);
        }
    }
}