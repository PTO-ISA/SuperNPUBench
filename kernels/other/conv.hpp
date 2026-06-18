#include <common/pto_tileop.hpp>

using namespace pto;

//in: Input data of shape (N, C, H, W) ->        -> N, 1, C, H, W   ->  N, F, C, H, W
//filter: Filter weights of shape (F, C, HH, WW) -> 1 ,F, C, HH, WW - > N ,F, C, HH, WW
//out: Output data, of shape (N, F, H', W')      
// for j in range(0, H_prime):
//     for i in range(0, W_prime):
//        tmp_w = w
//        tmp_w = tmp_w[np.newaxis,:]
//        tmp_w = np.repeat(tmp_w, N, axis=0)
//        tmp_x = x_pad[:, :, j * stride:j * stride + HH, i * stride:i * stride + WW] 
//        tmp_x = tmp_x[:,np.newaxis]
//        tmp_x = np.repeat(tmp_x, F, axis=1)
//        out[:,:,j,i] = np.sum(np.sum(np.sum(tmp_x*tmp_w, axis=-1), axis=-1), axis=-1) \

//  Naive Loops
// for n in range(N):
//     for f in range(F):
//         for j in range(0, H_prime):
//             for i in range(0, W_prime):
//                 out[n, f, j, i] = (x_pad[n, :, j * stride:j * stride + HH, i * stride:i * stride + WW] * w[f, :, :, :]).sum()
// for f in range(F):
//     for j in range(0, H_prime):
//         for i in range(0, W_prime):
//             tmp_w = w[f, :, :, :]
//             tmp_w = tmp_w[np.newaxis,:]
//             tmp_w = np.repeat(tmp_w, N, axis=0) 
//             out[:, f, j, i] = np.sum(np.sum(np.sum(x_pad[:, :, j * stride:j * stride + HH, i * stride:i * stride + WW] * tmp_w, axis=-3), axis=-2), axis=-1)

//pic [N, C, H, W], filter [F, C, HH, WW] -> out [N, F, H', W']
template<typename dtype, const int N, const int C, const int H, const int W, 
        const int F, const int HH, const int WW>
void conv_forward(dtype *out, dtype *pic, dtype *filter){
    const int stride = 1;
    const int pad = 0;
    const int H_prime = 1+(H+2*pad-HH)/stride;
    const int W_prime = 1+(W+2*pad-WW)/stride;

    assert(((H + 2 * pad - HH) % stride == 0));
    assert(((W + 2 * pad - WW) % stride == 0));

    using gm_pic = global_tensor<dtype, RowMajor<H,W>>;
    using gm_filt = global_tensor<dtype, RowMajor<HH,WW>>;
    using gm_out  = global_tensor<dtype, RowMajor<H_prime, W_prime>>;

    using tile_filt  = Tile<Location::Vec, dtype, HH, WW, BLayout::RowMajor>;
    using tile_mask  = Tile<Location::Vec, dtype, HH, WW, BLayout::RowMajor, 1, 1>;

    for(int n=0;n<N;n++){
        for(int f=0;f<F;f++){
            for(int h=0;h<H_prime;h++){
                for(int w=0;w<W_prime;w++){
                    tile_mask tmp(0);
                    for(int c=0;c<C;c++){
                        gm_filt gfilt(filter + f *C*HH*WW + c*HH*WW); // filter[f, c, 0, 0]
                        gm_pic gpic(pic + n*C*H*W + c*H*W + h*stride*W + w*stride); // pic[n, c, h*stride, w*stride]
                        tile_filt tfilt;
                        tile_filt tpic;

                        TCOPYIN(tfilt, gfilt);
                        TCOPYIN(tpic, gpic);
                        TMUL(tpic, tpic, tfilt);
                        TROWSUMEXPAND(tpic, tpic, tpic);
                        TCOLSUMEXPAND(tpic, tpic, tpic); // sum all element
                        TADD(tmp, tmp, tpic);
                    }
                    int offset = n*F*H_prime*W_prime + f*H_prime*W_prime + h*W_prime + w;
                    gm_out gO(out+offset);
                    TCOPYOUT(gO, tmp);
                }
            }
        }
    }
}

//x [N, C, H, W] -> x_pad[N, C, H+2*pad, W+2*pad]
template<typename dtype, const int N, const int C, const int H, const int W, const int pad>
void pad(dtype *x_pad, dtype *x){

}