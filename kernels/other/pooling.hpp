#include <common/pto_tileop.hpp>

using namespace pto;

// for n in range(N):
//     for c in range(C):
//         for x1 in range(math.floor(H_out)):
//             for y in range(math.floor(W_out)):
//                 out[n, c, x1, y] = np.amax(x[n, c, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W])

//input: (N,C,H,W)
//output:(N,C,H',W')

template<const int PH, const int PW, const int STRID>
struct pool_param{
    const int ph = PH;
    const int pw = PW;
    const int stride = STRID;
};

// pic [N, C, H, W] -> out [N, C, H', W']
template<typename dtype, const int N, const int C, const int H, const int W, typename pool_pm>
void max_pool_forward(dtype *out, dtype *pic, const pool_pm pool){
    const int H_out = 1 + (H-pool.ph) / stride;
    const int W_out = 1 + (H-pool.pw) / stride;

    using  gm_pic = global_tensor<dtype, RowMajor<H,W>>;
    using  gm_out = global_tensor<dtype, RowMajor<H_out, W_out>>;

    using  tile_filt = Tile<Location::Vec, dtype, pool.ph, pool.pw, BLayout::RowMajor>;
    using  tile_mask = Tile<Location::Vec, dtype, HH, WW, BLayout::RowMajor, 1, 1>;
    for(int n=0;n<N;n++){
        for(int c=0;c<C;c++){
            for(int h=0;h<H_out;h++){
                for(int w=0;w<W_out;w++){
                    tile_mask tmp;
                    gm_pic gpic(pic+ n*C*H*W + c*H*W + h*pool.stride*W + w*pool.stride); //pic[n, c, h*pool.stride, w*pool.stride]

                    tile_filt tpic;
                    TLOAD(tpic, gpic);
                    TROWMAXEXPAND(tpic, tpic);
                    TCOLMAXEXPAND(tpic, tpic);
                    TCOPY(tmp, tpic);

                    int offset = n*C*H_out*W_out + c*H_out*W_out + h*W_out + w;
                    gm_out gO(out+offset);
                    TSTORE(gO, tpic);
                }
            }
        }
    }
}

void max_pool_backward(){

}

// pic [N, C, H, W] -> out [N, C, H', W']
template<typename dtype, const int N, const int C, const int H, const int W, typename pool_pm>
void avg_pool_forward(dtype *out, dtype *pic, const pool_pm pool){
    const int H_out = 1 + (H-pool.ph) / stride;
    const int W_out = 1 + (H-pool.pw) / stride;

    using  gm_pic = global_tensor<dtype, RowMajor<H,W>>;
    using  gm_out = global_tensor<dtype, RowMajor<H_out, W_out>>;

    using  tile_filt = Tile<Location::Vec, dtype, pool.ph, pool.pw, BLayout::RowMajor>;
    using  tile_mask = Tile<Location::Vec, dtype, HH, WW, BLayout::RowMajor, 1, 1>;

    for(int n=0;n<N;n++){
        for(int c=0;c<C;c++){
            for(int h=0;h<H_out;h++){
                for(int w=0;w<W_out;w++){
                    tile_mask tmp;
                    gm_pic gpic(pic+ n*C*H*W + c*H*W + h*pool.stride*W + w*pool.stride); //pic[n, c, h*pool.stride, w*pool.stride]

                    tile_filt tpic;
                    TLOAD(tpic, gpic);
                    TROWSUMEXPAND(tpic, tpic);
                    TCOLSUMEXPAND(tpic, tpic);
                    TDIVS(tpic, tpic, HH*WW);
                    TCOPY(tmp, tpic);

                    int offset = n*C*H_out*W_out + c*H_out*W_out + h*W_out + w;
                    gm_out gO(out+offset);
                    TSTORE(gO, tpic);
                }
            }
        }
    }
}

void avg_pool_backward(){

}

