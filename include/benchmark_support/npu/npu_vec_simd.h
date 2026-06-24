#include <common/pto_tileop.hpp>

template <typename dtype, const int kM, const int kN, const int kTM, const int kTN>
void matadd(dtype *c_ptr, dtype *a_ptr, dtype *b_ptr) {
  using gm_shape    = global_tensor<dtype, RowMajor<kM, kN>>;
  using tile_shape  = Tile<Location::Vec, dtype, kTM, kTN, BLayout::RowMajor>;
  using gm_iterator = global_iterator<gm_shape, tile_shape>;

  gm_iterator gAIter(a_ptr);
  gm_iterator gBIter(b_ptr);
  gm_iterator gCIter(c_ptr);
  const int Mb = (kM + kTM - 1) / kTM;
  const int Nb = (kN + kTN - 1) / kTN;

  for (int i = 0; i < Mb; ++i) {
    for (int j = 0; j < Nb; ++j) {
      auto gA = gAIter(i, j);
      auto gB = gBIter(i, j);
      auto gC = gCIter(i, j);

      tile_shape tA, tB, tC;
      TCOPYIN(tA, gA);
      TCOPYIN(tB, gB);
      TADD(tC, tA, tB);
      TCOPYOUT(gC, tC);
    }
  }
}

template<typename tile_shape>
void __vec__ rmsnorm_kernel(
                        typename tile_shape::TileDType __out__ out,
                        const typename tile_shape::TileDType __in__ in,
                        const int col,
                        const int iter
                    ){
    size_t j = blkv_get_index_x();
    size_t i = blkv_get_index_y();

    __vbuf__ typename tile_shape::DType *in_ptr  = blkv_get_tile_ptr(in);
    __vbuf__ typename tile_shape::DType *out_ptr = blkv_get_tile_ptr(out);

    __half sum;
    
    __half data = static_cast<__half>(in_ptr[j]);
    asm volatile("l.rdfadd %1.fh, ->%0.h" 
            :"=r"(sum)
            :"vr"(data)
            );
    
    #pragma clang loop unroll(full)
    for(int i=1;i<iter;i++){
        size_t idx = i * col + j;
        data = static_cast<__half>(in_ptr[idx]) * static_cast<__half>(in_ptr[idx]);
        typename tile_shape::DType local_sum;
        asm volatile("l.rdfadd %1.fh, ->%0.h" 
                    :"=r"(local_sum)
                    :"vr"(data)
                    );
        sum += local_sum;
    }

    sum = blkv_fsqrt( (sum / static_cast<__half>(tile_shape::ValidCol)) );
      
    #pragma clang loop unroll(full)
    for(int i=0;i<iter;i++){
        size_t idx = i * col + j;
        data = static_cast<__half>(in_ptr[idx]) / sum;
        out_ptr[idx] = static_cast<typename tile_shape::DType>(data);
    }
}

template<typename dtype, const int COL>
void rmsnorm_oneline(dtype *dst, dtype *src){
    using gm_shape = global_tensor<dtype, RowMajor<1, COL>>;
    using tile_shape = Tile<Location::Vec, dtype, 1, COL, BLayout::RowMajor>;

    gm_shape gsrc(src);
    tile_shape tsrc;
    TCOPYIN(tsrc, gsrc);

    const int iter = tile_shape::ValidCol / LaneNum;
    tile_shape tdst;
    rmsnorm_kernel<tile_shape><<<LaneNum, 1, 1>>>(tdst.data(), tsrc.data(), LaneNum, iter);

    gm_shape gdst(dst);
    TCOPYOUT(gdst, tdst);
}

// x / (Ex^2) ^ .5
template<typename dtype, const int kM, const int kN, const int kTM, const int kTN>
void rmsnorm(dtype *dst, dtype *src){
    using gm_shape = global_tensor<dtype, RowMajor<kM, kN>>;
    using tile_shape = Tile<Location::Vec, dtype, kTM, kTN, BLayout::RowMajor>;
 
    using tSum = Tile<Location::Vec, dtype, kTM, 256, BLayout::RowMajor, kTM, 1>;
 
    using gIter = global_iterator<gm_shape, tile_shape>;

    gIter giter_src(src);
    gIter giter_dst(dst);

    const int Mb = kM / kTM;
    const int Nb = kN / kTN;

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



template<typename tile_shape>
void __vec__ layernorm_kernel(
                        typename tile_shape::TileDType __out__ out,
                        const typename tile_shape::TileDType __in__ in,
                        const int col,
                        const int iter
                    ){
    size_t j = blkv_get_index_x();
    size_t i = blkv_get_index_y();

    __vbuf__ typename tile_shape::DType *in_ptr  = blkv_get_tile_ptr(in);
    __vbuf__ typename tile_shape::DType *out_ptr = blkv_get_tile_ptr(out);

    __half sum;
    __half square_sum;

    __half data = static_cast<__half>(in_ptr[j]);
    __half data_square = data * data;
    asm volatile("l.rdfadd %1.fh, ->%0.h" 
            :"=r"(sum)
            :"vr"(data)
            );
    asm volatile("l.rdfadd %1.fh, ->%0.h" 
        :"=r"(square_sum)
        :"vr"(data_square)
        );
    
    #pragma clang loop unroll(full)
    for(int i=1;i<iter;i++){
        size_t idx = i * col + j;
        data = static_cast<__half>(in_ptr[idx]) * static_cast<__half>(in_ptr[idx]);
        data_square = data * data;
        typename tile_shape::DType local_sum;
        typename tile_shape::DType local_square_sum;
        asm volatile("l.rdfadd %1.fh, ->%0.h" 
                    :"=r"(local_sum)
                    :"vr"(data)
                    );
        asm volatile("l.rdfadd %1.fh, ->%0.h" 
                    :"=r"(local_square_sum)
                    :"vr"(data)
                    );
        sum += local_sum;
        square_sum += local_square_sum;
    }

    sum = (sum / static_cast<__half>(tile_shape::ValidCol));
    square_sum = (square_sum / static_cast<__half>(tile_shape::ValidCol));
      
    #pragma clang loop unroll(full)
    for(int i=0;i<iter;i++){
        size_t idx = i * col + j;
        data = (static_cast<__half>(in_ptr[idx]) - sum) / blkv_fsqrt(square_sum - sum * sum);
        out_ptr[idx] = static_cast<typename tile_shape::DType>(data);
    }
}

template<typename dtype, const int COL>
void layernorm_oneline(dtype *dst, dtype *src){
    using gm_shape = global_tensor<dtype, RowMajor<1, COL>>;
    using tile_shape = Tile<Location::Vec, dtype, 1, COL, BLayout::RowMajor>;

    gm_shape gsrc(src);
    tile_shape tsrc;
    TCOPYIN(tsrc, gsrc);

    const int iter = tile_shape::ValidCol / LaneNum;
    tile_shape tdst;
    layernorm_kernel<tile_shape><<<LaneNum, 1, 1>>>(tdst.data(), tsrc.data(), LaneNum, iter);

    gm_shape gdst(dst);
    TCOPYOUT(gdst, tdst);
}


// (x - Ex) / (E(x-Ex)^2)^.5 -> (x - Ex) / (Ex^2 - (Ex)^2)^.5
template<typename dtype, const int kM, const int kN, const int kTM, const int kTN>
void layernorm(dtype *dst, dtype *src, float *gamma, float *beta)
{
    using gm_shape = global_tensor<dtype, RowMajor<kM, kN>>;

    using tile_shape = Tile<Location::Vec, dtype, kTM, kTN, BLayout::RowMajor>;
 
    using tSum = Tile<Location::Vec, dtype, kTM, kTN, BLayout::RowMajor, kTM, 1>;
 
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

            TMULS(tsrc, tsrc, static_cast<dtype>(*gamma));
            TADDS(tsrc, tsrc, static_cast<dtype>(*beta));
 
            auto gdst = giter_dst(i,j);
            TCOPYOUT(gdst, tsrc);
        }
    }
}

template<const int kM, const int kN, const int kTM, const int kTN>
void layernorm_bf16(__bf16 *dst, __bf16 *src, float *gamma, float *beta)
{
    using gm_shape = global_tensor<__bf16, RowMajor<kM, kN>>;

    using tile_shape = Tile<Location::Vec, __bf16, kTM, kTN, BLayout::RowMajor>;

    using tile_shape_cast = Tile<Location::Vec, __half, kTM, kTN, BLayout::RowMajor>;
 
    using tSum = Tile<Location::Vec, __half, kTM, kTN, BLayout::RowMajor, kTM, 1>;
 
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
            tile_shape tsrc_ori;
            tile_shape_cast tsrc;
            TCOPYIN(tsrc_ori, gsrc);
            TCAST(tsrc, tsrc_ori);
 
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
 
        tile_shape_cast gMean_i;
        tile_shape_cast gStdDev_i;
        TEXPANDCOL(gMean_i, gMean);
        TEXPANDCOL(gStdDev_i, gStdDev);

        for(int j=0;j<Nb;j++)
        {
            auto  gsrc = giter_src(i,j);
            tile_shape tsrc_ori;
            tile_shape_cast tsrc;
            TCOPYIN(tsrc_ori, gsrc);
            TCAST(tsrc, tsrc_ori);
 
            TSUB(tsrc, tsrc, gMean_i);    // (x - Ex)
            TDIV(tsrc, tsrc, gStdDev_i);  // (x - Ex) / (Ex^2 - (Ex)^2)^.5

            TMULS(tsrc, tsrc, static_cast<__half>(*gamma));
            TADDS(tsrc, tsrc, static_cast<__half>(*beta));
 
            auto gdst = giter_dst(i,j);
            tile_shape_cast tdst;
            TCAST(tdst, tsrc);
            TCOPYOUT(gdst, tdst);
        }
    }
}

// for n in range(N):
//     for c in range(C):
//         for x1 in range(math.floor(H_out)):
//             for y in range(math.floor(W_out)):
//                 out[n, c, x1, y] = np.amax(x[n, c, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W])

//input: (N,C,H,W)
//output:(N,C,H',W')

// template<const int PH, const int PW, const int STRID>
// struct pool_param{
//     const int ph = PH;
//     const int pw = PW;
//     const int stride = STRID;
// };

// pic [N, C, H, W] -> out [N, C, H', W']
// template<typename dtype, const int N, const int C, const int H, const int W, typename pool_pm>
// void max_pool(dtype *out, dtype *pic, const pool_pm pool){
//     const int H_out = 1 + (H-pool.ph) / pool.stride;
//     const int W_out = 1 + (H-pool.pw) / pool.stride;

//     using  gm_pic = global_tensor<dtype, RowMajor<H,W>>;
//     using  gm_out = global_tensor<dtype, RowMajor<H_out, W_out>>;

//     using  tile_filt = Tile<Location::Vec, dtype, pool.ph, pool.pw, BLayout::RowMajor>;
//     using  tile_mask = Tile<Location::Vec, dtype, HH, WW, BLayout::RowMajor, 1, 1>;
//     for(int n=0;n<N;n++){
//         for(int c=0;c<C;c++){
//             for(int h=0;h<H_out;h++){
//                 for(int w=0;w<W_out;w++){
//                     tile_mask tmp;
//                     gm_pic gpic(pic+ n*C*H*W + c*H*W + h*pool.stride*W + w*pool.stride); //pic[n, c, h*pool.stride, w*pool.stride]

//                     tile_filt tpic;
//                     TCOPYIN(tpic, gpic);
//                     TROWMAXEXPAND(tpic, tpic);
//                     TCOLMAXEXPAND(tpic, tpic);
//                     TCOPY(tmp, tpic);

//                     int offset = n*C*H_out*W_out + c*H_out*W_out + h*W_out + w;
//                     gm_out gO(out+offset);
//                     TCOPYOUT(gO, tpic);
//                 }
//             }
//         }
//     }
// }

template<typename tile_shape>
void __vec__ softmax_kernel(
                        typename tile_shape::TileDType __out__ out,
                        const typename tile_shape::TileDType __in__ in,
                        const int col,
                        const int iter
                    ){
    size_t j = blkv_get_index_x();
    size_t i = blkv_get_index_y();

    __vbuf__ typename tile_shape::DType *in_ptr  = blkv_get_tile_ptr(in);
    __vbuf__ typename tile_shape::DType *out_ptr = blkv_get_tile_ptr(out);

    __half max;
    __half sum;
    
    __half data = static_cast<__half>(in_ptr[j]);
    asm volatile("l.rdfmax %1.fh, ->%0.h" 
            :"=r"(max)
            :"vr"(data)
            );

    asm volatile("l.rdfadd %1.fh, ->%0.h" 
            :"=r"(sum)
            :"vr"(data)
            );
    
    #pragma clang loop unroll(full)
    for(int i=1;i<iter;i++){
        size_t idx = i * col + j;
        data = static_cast<__half>(in_ptr[idx]);
        typename tile_shape::DType local_max;
        asm volatile("l.rdfmax %1.fh, ->%0.h" 
                    :"=r"(local_max)
                    :"vr"(data)
                    );
        max = blkv_max(max, local_max);

        typename tile_shape::DType local_sum;
        asm volatile("l.rdfadd %1.fh, ->%0.h" 
                    :"=r"(local_sum)
                    :"vr"(data)
                    );
        sum += local_sum;       
    }

    #pragma clang loop unroll(full)
    for(int i=0;i<iter;i++){
        size_t idx = i * col + j;
        data = blkv_fexp(static_cast<__half>(in_ptr[idx]) - max) / sum;
        out_ptr[idx] = static_cast<typename tile_shape::DType>(data);
    }
}

template<typename dtype, const int COL>
void softmax_oneline(dtype *dst, dtype *src){
    using gm_shape = global_tensor<dtype, RowMajor<1, COL>>;
    using tile_shape = Tile<Location::Vec, dtype, 1, COL, BLayout::RowMajor>;

    gm_shape gsrc(src);
    tile_shape tsrc;
    TCOPYIN(tsrc, gsrc);

    const int iter = tile_shape::ValidCol/ LaneNum;
    tile_shape tdst;
    softmax_kernel<tile_shape><<<LaneNum, 1, 1>>>(tdst.data(), tsrc.data(), LaneNum, iter);

    gm_shape gdst(dst);
    TCOPYOUT(gdst, tdst);
}

template<const int kM, const int kN, const int kTM, const int kTN>
void softmax_bf16(__bf16* dst, __bf16* src){
    using gm_shape = global_tensor<__bf16, RowMajor<kM, kN>>;
    using tile_shape_ori = Tile<Location::Vec, __bf16, kTM, kTN, BLayout::RowMajor>;

    using tile_shape = Tile<Location::Vec, __half, kTM, kTN, BLayout::RowMajor>;
    using tMax = Tile<Location::Vec, __half, kTM, kTN, BLayout::RowMajor, kTM, 1>;
    using tSum = Tile<Location::Vec, __half, kTM, kTN, BLayout::RowMajor, kTM, 1>;

    int Mb = kM/kTM;
    int Nb = kN/kTN;

    for(int i=0;i<Mb;i++){
        tMax tAccMax(-10000);
        tSum tAccSum(0);
        for(int j=0;j<Nb;j++){
            uint32_t offset = i*kTM*kN+j*kTN;
            gm_shape gsrc(src+offset);
            tile_shape_ori tsrc_ori;
            tile_shape tsrc;
            TCOPYIN(tsrc_ori, gsrc);
            TCAST(tsrc, tsrc_ori);

            tMax tLocalMax;
            TROWMAX(tLocalMax, tsrc);
            //new local max
            tMax tAccMaxNew;
            TMAX(tAccMaxNew, tAccMax, tLocalMax);

            //new local max expand --- e^(x-max_new(x))
            tile_shape tAccMaxNewExpand;
            TEXPANDCOL(tAccMaxNewExpand, tAccMaxNew);
            TSUB(tsrc, tsrc, tAccMaxNewExpand);
            TEXP(tsrc, tsrc);

            tSum tLocalSum;
            TROWSUM(tLocalSum, tsrc);

            //new local sum --- sum_new = sum_old*e^(old_max-new_max) + Σe ^ (x_local - new_max)
            tSum tAccSum_Scale;
            tMax tAccMax_Scale;
            TSUB(tAccMax_Scale, tAccMax, tAccMaxNew);
            TEXP(tAccMax_Scale, tAccMax_Scale);
            TMUL(tAccSum_Scale, tAccMax_Scale, tAccSum);
            TADD(tAccSum, tAccSum_Scale, tLocalSum);

            TCOPY(tAccMax, tAccMaxNew);
            //tAccMax = tAccMaxNew;
        }

        for(int j=0;j<Nb;j++){
            uint32_t offset = i*kTM*kN+j*kTN;
            gm_shape gsrc(src+offset);
            tile_shape_ori tsrc_ori;
            tile_shape tsrc;
            TCOPYIN(tsrc_ori, gsrc);
            TCAST(tsrc, tsrc_ori);

            tile_shape gMax;
            tile_shape gSum;
            TEXPANDCOL(gMax, tAccMax);
            TEXPANDCOL(gSum, tAccSum);

            TSUB(tsrc, tsrc, gMax);
            TEXP(tsrc, tsrc);
            TDIV(tsrc, tsrc, gSum);

            gm_shape gdst(dst+offset);

            tile_shape_ori tsrc_out;
            TCAST(tsrc_out, tsrc);
            TCOPYOUT(gdst, tsrc_out);
        }
    }
}

// e^(x-x_max)/Σ(e^(x-x_max))
template<typename dtype, const int kM, const int kN, const int kTM, const int kTN>
void softmax(dtype* dst, dtype* src){
    using gm_shape = global_tensor<dtype, RowMajor<kM, kN>>;
    using tile_shape = Tile<Location::Vec, dtype, kTM, kTN, BLayout::RowMajor>;

    using tMax = Tile<Location::Vec, dtype, kTM, kTN, BLayout::RowMajor, kTM, 1>;
    using tSum = Tile<Location::Vec, dtype, kTM, kTN, BLayout::RowMajor, kTM, 1>;

    int Mb = kM/kTM;
    int Nb = kN/kTN;

    for(int i=0;i<Mb;i++){
        tMax tAccMax(-10000);
        tSum tAccSum(0);
        for(int j=0;j<Nb;j++){
            uint32_t offset = i*kTM*kN+j*kTN;
            gm_shape gsrc(src+offset);
            tile_shape tsrc;
            TCOPYIN(tsrc, gsrc);

            tMax tLocalMax;
            TROWMAX(tLocalMax, tsrc);

            //new local max
            tMax tAccMaxNew;
            TMAX(tAccMaxNew, tAccMax, tLocalMax);

            //new local max expand --- e^(x-max_new(x))
            tile_shape tAccMaxNewExpand;
            TEXPANDCOL(tAccMaxNewExpand, tAccMaxNew);
            TSUB(tsrc, tsrc, tAccMaxNewExpand);
            TEXP(tsrc, tsrc);

            tSum tLocalSum;
            TROWSUM(tLocalSum, tsrc);

            //new local sum --- sum_new = sum_old*e^(old_max-new_max) + Σe ^ (x_local - new_max)
            tSum tAccSum_Scale;
            tMax tAccMax_Scale;
            TSUB(tAccMax_Scale, tAccMax, tAccMaxNew);
            TEXP(tAccMax_Scale, tAccMax_Scale);
            TMUL(tAccSum_Scale, tAccMax_Scale, tAccSum);
            TADD(tAccSum, tAccSum_Scale, tLocalSum);

            TCOPY(tAccMax, tAccMaxNew);
            //tAccMax = tAccMaxNew;
        }

        for(int j=0;j<Nb;j++){
            uint32_t offset = i*kTM*kN+j*kTN;
            gm_shape gsrc(src+offset);
            tile_shape tsrc;
            TCOPYIN(tsrc, gsrc);

            tile_shape gMax;
            tile_shape gSum;
            TEXPANDCOL(gMax, tAccMax);
            TEXPANDCOL(gSum, tAccSum);

            TSUB(tsrc, tsrc, gMax);
            TEXP(tsrc, tsrc);
            TDIV(tsrc, tsrc, gSum);

            gm_shape gdst(dst+offset);
            TCOPYOUT(gdst, tsrc);
        }
    }
}

template <const int kM, const int kN, const int kK, const int kTM, const int kTN, const int kTK, const bool Relu>
void gemm(float *c, float *a, float *b, float alpha, float beta)
{ 
    using gm_shapeA = global_tensor<float, RowMajor<kM, kK>>;
    using gm_shapeB = global_tensor<float, RowMajor<kK, kN>>;
    using gm_shapeC = global_tensor<float, RowMajor<kM, kN>>;
    using tile_shapeA = Tile<Location::Vec, float, kTM, kTK, BLayout::RowMajor>;
    using tile_shapeB = Tile<Location::Vec, float, kTK, kTN, BLayout::RowMajor>;
    using tile_shapeACC = Tile<Location::Vec, float, kTM, kTN, BLayout::RowMajor>;
    using gm_iteratorA = global_iterator<gm_shapeA, tile_shapeA>;
    using gm_iteratorB = global_iterator<gm_shapeB, tile_shapeB>;
    using gm_iteratorC = global_iterator<gm_shapeC, tile_shapeACC>;

    gm_iteratorA gAIter(a);
    gm_iteratorB gBIter(b);
    gm_iteratorC gCIter(c);

    const int Mb = kM / kTM;
    const int Nb = kN / kTN;
    const int Kb = kK / kTK;

    const int rmd_M = kM % kTM;
    const int rmd_N = kN % kTN;
    const int rmd_K = kK % kTK;

    using tile_shapeA_trows = Tile<Location::Vec, float, kTM, kTK, BLayout::RowMajor, kTM, rmd_K>;
    using tile_shapeA_tcols = Tile<Location::Vec, float, kTM, kTK, BLayout::RowMajor, rmd_M, kTK>;
    using tile_shapeA_tcorner = Tile<Location::Vec, float, kTM, kTK, BLayout::RowMajor, rmd_M, rmd_K>;

    using tile_shapeB_trows = Tile<Location::Vec, float, kTK, kTN, BLayout::RowMajor, kTK, rmd_N>;
    using tile_shapeB_tcols = Tile<Location::Vec, float, kTK, kTN, BLayout::RowMajor, rmd_K, kTN>;
    using tile_shapeB_tcorner = Tile<Location::Vec, float, kTK, kTN, BLayout::RowMajor, rmd_K, rmd_N>;

    using tile_shapeC_trows = Tile<Location::Vec, float, kTM, kTN, BLayout::RowMajor, kTM, rmd_N>;
    using tile_shapeC_tcols = Tile<Location::Vec, float, kTM, kTN, BLayout::RowMajor, rmd_M, kTN>;
    using tile_shapeC_tcorner = Tile<Location::Vec, float, kTM, kTN, BLayout::RowMajor, rmd_M, rmd_N>;

    for (int i = 0; i < Mb; ++i) {
        for (int j = 0; j < Nb; ++j) {
        auto gC = gCIter(i, j);

        tile_shapeACC tACC(0);
        #pragma clang loop unroll(full)
        for (int k = 0; k < Kb; ++k) {
            auto gA = gAIter(i, k);
            auto gB = gBIter(k, j);

            tile_shapeA tA;
            tile_shapeB tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB, gB);
            MATMACC(tACC, tA, tB);
        }

        if constexpr (rmd_K) {
            auto gA = gAIter(i, Kb);
            auto gB = gBIter(Kb, j);

            tile_shapeA_trows tA;
            tile_shapeB_tcols tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB, gB);
            MATMACC(tACC, tA, tB);
        }

        tile_shapeACC oldC;
        TCOPYIN(oldC, gC);
        TMULS(tACC, tACC, alpha);
        TMULS(oldC, oldC, beta);
        TADD(tACC, tACC, oldC);
        if constexpr(Relu){
            TMAXS(tACC, tACC, 0);
        }
        TCOPYOUT(gC, tACC);
        }
        if constexpr (rmd_N) {
        auto gC = gCIter(i, Nb);

        tile_shapeC_trows tACC(0);
        #pragma clang loop unroll(full)
        for (int k = 0; k < Kb; ++k) {
            auto gA = gAIter(i, k);
            auto gB = gBIter(k, Nb);

            tile_shapeA tA;
            tile_shapeB_trows tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB, gB);
            MATMACC(tACC, tA, tB);
        }
        if constexpr (rmd_K) {
            auto gA = gAIter(i, Kb);
            auto gB = gBIter(Kb, Nb);

            tile_shapeA_trows tA;
            tile_shapeB_tcorner tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB, gB);
            MATMACC(tACC, tA, tB);
        }

        tile_shapeC_trows oldC;
        TCOPYIN(oldC, gC);
        TMULS(tACC, tACC, alpha);
        TMULS(oldC, oldC, beta);
        TADD(tACC, tACC, oldC);
        if constexpr(Relu){
            TMAXS(tACC, tACC, 0);
        }
        TCOPYOUT(gC, tACC);
        }
    }
    if constexpr (rmd_M) {
        for (int j = 0; j < Nb; ++j) {
        auto gC = gCIter(Mb, j);

        tile_shapeC_tcols tACC(0);
        #pragma clang loop unroll(full)
        for (int k = 0; k < Kb; ++k) {
            auto gA = gAIter(Mb, k);
            auto gB = gBIter(k, j);

            tile_shapeA_tcols tA;
            tile_shapeB tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB, gB);
            MATMACC(tACC, tA, tB);
        }
        if constexpr (rmd_K) {
            auto gA = gAIter(Mb, Kb);
            auto gB = gBIter(Kb, j);

            tile_shapeA_tcorner tA;
            tile_shapeB_tcols tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB, gB);
            MATMACC(tACC, tA, tB);
        }

        tile_shapeC_tcols oldC;
        TCOPYIN(oldC, gC);
        TMULS(tACC, tACC, alpha);
        TMULS(oldC, oldC, beta);
        TADD(tACC, tACC, oldC);
        if constexpr(Relu){
            TMAXS(tACC, tACC, 0);
        }
        TCOPYOUT(gC, tACC);
        }
        if constexpr (rmd_N) {
        auto gC = gCIter(Mb, Nb);

        tile_shapeC_tcorner tACC(0);
        #pragma clang loop unroll(full)
        for (int k = 0; k < Kb; ++k) {
            auto gA = gAIter(Mb, k);
            auto gB = gBIter(k, Nb);

            tile_shapeA_tcols tA;
            tile_shapeB_trows tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB, gB);
            MATMACC(tACC, tA, tB);
        }
        if constexpr (rmd_K) {
            auto gA = gAIter(Mb, Kb);
            auto gB = gBIter(Kb, Nb);

            tile_shapeA_tcorner tA;
            tile_shapeB_tcorner tB;
            TCOPYIN(tA, gA);
            TCOPYIN(tB, gB);
            MATMACC(tACC, tA, tB);
        }

        tile_shapeC_tcorner oldC;
        TCOPYIN(oldC, gC);
        TMULS(tACC, tACC, alpha);
        TMULS(oldC, oldC, beta);
        TADD(tACC, tACC, oldC);
        if constexpr(Relu){
            TMAXS(tACC, tACC, 0);
        }
        TCOPYOUT(gC, tACC);
        }
    }    
}

template<typename dtype, int M, int N, int tM, int tN>
void gelu(dtype *out, dtype* in){

}

// w1(in) * silu(w2(in))
//silu : x / (1 + e^-x) 
template <typename dtype, const int S, const int InDim, const int OutDim, const int tS, const int tInDim, const int tOutDim>
void swiglu(dtype *out, dtype *in, dtype *w1, dtype *w2){
    using gmIn = global_tensor<dtype, RowMajor<S, InDim>>;
    using gmW = global_tensor<dtype, RowMajor<InDim, OutDim>>;
    using gmOut = global_tensor<dtype, RowMajor<S, OutDim>>;

    using tileIn = Tile<Location::Vec, dtype, tS, tInDim, BLayout::RowMajor>;
    using tileW = Tile<Location::Vec, dtype, tInDim, tOutDim, BLayout::RowMajor>;
    using tileACC = Tile<Location::Vec, dtype, tS, tOutDim, BLayout::RowMajor>;

    using gm_iteratorIn = global_iterator<gmIn, tileIn>;
    using gm_iteratorW = global_iterator<gmW, tileW>;
    using gm_iteratorOut = global_iterator<gmOut, tileACC>;

    gm_iteratorIn  gIterIn(in);
    gm_iteratorW   gIterW1(w1);
    gm_iteratorW   gIterW2(w2);
    gm_iteratorOut  gIterOut(out);

    const int Sb = S / tS;
    const int Inb = InDim / tInDim;
    const int Outb = OutDim / tOutDim;
    
    for(int i=0;i<Sb;i++){
        for(int j=0;j<Outb;j++){

            tileACC tACC_W1(0);
            tileACC tACC_W2(0);

            for(int k=0;k<Inb;k++){
                auto gIn = gIterIn(i,k);
                auto gW1 = gIterW1(k,j);
                auto gW2 = gIterW2(k,j);
                tileIn tIn;
                tileW tW1;
                tileW tW2;
                TCOPYIN(tIn, gIn);
                TCOPYIN(tW1, gW1);
                TCOPYIN(tW2, gW2);

                MATMACC(tACC_W1, tIn, tW1);
                MATMACC(tACC_W2, tIn, tW2);
            }
            tileACC tACC_tmp;
            TEXP(tACC_tmp, tACC_W2);
            TRECIP(tACC_tmp, tACC_tmp);
            TADDS(tACC_tmp, tACC_tmp, static_cast<dtype>(1));
            TRECIP(tACC_tmp, tACC_tmp);
            TMUL(tACC_W2, tACC_W2, tACC_tmp);

            tileACC tOut;
            TMUL(tOut, tACC_W1, tACC_W2);
            auto gOut = gIterOut(i,j);
            TCOPYOUT(gOut, tOut);
        }
    }
}

template<const int row, const int rope_dim, const int trow, const int tcol>
void rope(__bf16 *out, __bf16 *x, __bf16 *freqs_cis){
    const uint32_t rope_trow = trow*tcol/2;
    using gm_shape = global_tensor<__bf16, RowMajor<row, rope_dim>>;
    using tile_shape = Tile<Location::Vec, __bf16, trow, tcol, BLayout::RowMajor>;
    using tile_shape_cast = Tile<Location::Vec, __half, trow, tcol, BLayout::RowMajor>;

    using tile_shape_rope = Tile<Location::Vec, __half, rope_trow, 2, BLayout::RowMajor>;

    using tile_shape_half = Tile<Location::Vec, __half, rope_trow, 1, BLayout::RowMajor>;

    uint32_t n_row = row/trow;
    uint32_t n_col = rope_dim/tcol;
    for(int i=0;i<n_row;i++){
        for(int j=0;j<n_col;j++){
            uint32_t offset = i * (trow * rope_dim) + j * tcol;
            gm_shape input(x+offset);
            tile_shape tin_ori;
            tile_shape_rope resh_tin;
            TCOPYIN(tin_ori, input);   // 64*32
            tile_shape_cast tin;
            TCAST(tin, tin_ori);
            TRESHAPE(resh_tin, tin); // 64*32 -> 1024*2

            tile_shape_half tin_real;
            tile_shape_half tin_imag; 
            TEXTRACT(tin_real, resh_tin, 0, 0);        // real 1024*1
            TEXTRACT(tin_imag, resh_tin, 0, 1);        // image 1024*1

            offset = i * (trow * rope_dim) + j * tcol;
            gm_shape freqs(freqs_cis+offset);
            tile_shape tfreqs_ori;
            tile_shape_rope tfreqs_resh;
            TCOPYIN(tfreqs_ori, freqs);
            tile_shape_cast tfreqs;
            TCAST(tfreqs, tfreqs_ori);
            TRESHAPE(tfreqs_resh, tfreqs);

            tile_shape_half tfreqs_real;
            tile_shape_half tfreqs_imag;
            TEXTRACT(tfreqs_real, tfreqs_resh, 0, 0);        // real 1024*1
            TEXTRACT(tfreqs_imag, tfreqs_resh, 0, 1);        // image 1024*1

            tile_shape_half tac;
            tile_shape_half tbd;
            tile_shape_half tout_real;
            TMUL(tac, tin_real, tfreqs_real); // a*c
            TMUL(tbd, tin_imag, tfreqs_imag); // b*d
            TSUB(tout_real, tac, tbd);    // a*c - b*d

            tile_shape_half tbc;
            tile_shape_half tad;
            tile_shape_half tout_imag;
            TMUL(tbc, tin_imag, tfreqs_real); // b*c
            TMUL(tad, tin_real, tfreqs_imag); // a*d
            TADD(tout_imag, tbc, tad);    // b*c + a*d

            tile_shape_rope tout;
            {
            // TASSEMBLE2(tout, tout_real, tout_imag);
            tile_shape_rope padding(0);
            Tile<Location::Vec, __half, rope_trow, 4, BLayout::RowMajor> tout_tmp; // last col dim not used
            TASSEMBLE(tout_tmp, tout_real, tout_imag, padding);
            TEXTRACT(tout, tout_tmp, 0, 0);
            }
            tile_shape_cast tout_resh;
            TRESHAPE(tout_resh, tout);

            offset = i * (trow * rope_dim) + j * tcol;
            gm_shape output(out+offset);

            tile_shape_cast tout_resh_cast;
            TCAST(tout_resh_cast, tout_resh);
            TCOPYOUT(output, tout_resh_cast);
        }
    }
}

template<typename dtype, int M, int N, int tM, int tN>
void argmax(dtype *out, dtype* in){

}

template <typename tile_shape>
void __vec__ BitonicSortStepDescend_RowMajor_Imp(
    typename tile_shape::TileDType __out__ dst,
    const typename tile_shape::TileDType __in__ src,
    const uint16_t __in__ stage, const uint16_t __in__ step) {
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

template <typename dtype, is_tile_data_v tile_shape, bool ascending = true>
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
    if constexpr (tile_shape::isRowMajor) {
        for (uint16_t stage = 2; stage <= tile_shape::Cols; stage <<= 1) {
            for (uint16_t step = stage >> 1; step > 0; step >>= 1) {
                BitonicSortStepDescend_RowMajor_Imp<tile_shape_sort><<<col, row>>>(dst_sort.data(), src_sort.data(), stage, step);
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

template<typename dtype, const int tokens, const int scores, const int tS, const int tK>
void topk(dtype *weight, dtype* indices, dtype *x){
    using gmIn = global_tensor<dtype, RowMajor<tokens, scores>>;
    using gmOut = global_tensor<dtype, RowMajor<tokens, tK>>;
    using tileIn = Tile<Location::Vec, dtype, tS, scores, BLayout::RowMajor>;
    using tileOut = Tile<Location::Vec, dtype, tS, 32, BLayout::RowMajor, tS, tK>; // topk < 32
   
    const int block = tokens/tS;
    for(int i=0;i<block;i++){
        gmIn gIn(x+i*tS*scores);
        tileIn tIn;
        tileIn tWeight;
        tileIn tIndice;
        TCOPYIN(tIn, gIn);
        TSORTROW<dtype>(tWeight, tIndice, tIn);
        tileOut tWeightOut;
        TEXTRACT(tWeightOut, tWeight, 0, 0);

        tileOut tIndiceOut;
        TEXTRACT(tIndiceOut, tIndice, 0, 0);

        gmOut gWeight(weight+i*tS*tK);
        TCOPYOUT(gWeight, tWeightOut);

        gmOut gIndice(indices+i*tS*tK);
        TCOPYOUT(gIndice, tIndiceOut);
    }
}