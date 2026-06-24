#ifndef MLA_HPP
#define MLA_HPP

#include <common/pto_tileop.hpp>
#include <stdio.h>
#include "tensor.h"
#include "ds_config.h"
#include "flash_attention.hpp"
#include "matmul.hpp"
#include "moe.hpp"

using namespace pto;

// template<typename dtype, const int tS>
// void precompute_freqs_cis(dtype *freqs_cis_real, dtype *freqs_cis_imag){
//     const int dim = ModelArgs::qk_rope_head_dim;
//     const int seq_len = ModelArgs::max_seq_len;

//     using freq_gshape = global_tensor<dtype, RowMajor<seq_len, dim/2>>;
//     using freq_tshape = Tile<Location::Vec, dtype, tS, dim/2, BLayout::RowMajor>;

//     using itFreq = global_iterator<freq_gshape, freq_tshape>;

//     itFreq gFreq_real(freqs_cis_real);
//     itFreq gFreq_imag(freqs_cis_imag);

//     const int Sb = seq_len / tS;
//     for(int i=0;i<Sb;i++){
//         Tile<Location::Vec, dtype, dim/2, 1, BLayout::RowMajor> freqs;
//         Tile<Location::Vec, dtype, tS, 1, BLayout::RowMajor> t;
//         TARANGE(freqs, 0, dim, 2);
//         TARANGE(t, 0, end, 1);
//         freq_tshape tfreq_cis;
//         freq_tshape tfreq_cis_real;
//         freq_tshape tfreq_cis_imag;
//         TOUTDOT(tfreq_cis, freqs, t); //outer product
//         TSIN(tfreq_cis_real, tfreq_cis); //
//         TCOS(tfreq_cis_imag, tfreq_cis);
//         auto gO_real = gFreq_real(i,0);
//         auto gO_imag = gFreq_imag(i,0);
//         TSTORE(gO_real, tfreq_cis_real);
//         TSTORE(gO_imag, tfreq_cis_imag);
//     }
// }

template<typename dype, const int bsz, const int seq_len, const int dim_in,const int dim_out>
void projection(Tensor<dtype, bsz, seq_len , dim_out> &out,
               Tensor<dtype, bsz, seq_len, dim_in> &x,
               Tensor<dtype, dim_in, dim_out> &proj){

    for(int i=0;i<bsz;i++){
        //M N K
        matmul_frac<seq_len, dim_out, dim_in, 64, 64, 64>(out.data(i), x.data(i), proj.data());
    }
}

// x / (Ex^2) ^ .5
template<typename dtype, const int kM, const int kN, const int kTM, const int kTN>
void rmsnorm(dtype *dst, dtype *src){
    using gm_shape = global_tensor<dtype, RowMajor<kM, kN>>;
    using tile_shape = Tile<Location::Vec, dtype, kTM, kTN, BLayout::RowMajor>;

    using tSum = Tile<Location::Vec, dtype, kTM, 16, BLayout::RowMajor, kTM, 1>;

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

            TLOAD(tsrc, gsrc);

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
            TLOAD(tsrc, gsrc);

            TDIV(tsrc, tsrc, gSqureMean_i);

            auto gdst = giter_dst(i,j);
            TSTORE(gdst, tsrc);
        }
    }
}

// template <typename tile_shape_in0, typename tile_shape_in1, typename tile_shape_out>
// void TAssemble2_RowMajor(typename tile_shape_out::TileDType dst,
//                             const typename tile_shape_in0::TileDType src0,
//                             const typename tile_shape_in1::TileDType src1) {
//   for (uint16_t i = 0; i < tile_shape_out::ValidRow; ++i) {
//     for (uint16_t j = 0; j < tile_shape_in0::ValidCol; ++j) {
//       dst[i * tile_shape_out::RowStride + j] =
//           src0[i * tile_shape_in0::RowStride + j];
//     }
//     for (uint16_t j = 0; j < tile_shape_in1::ValidCol; ++j) {
//       dst[tile_shape_in0::ValidCol + i * tile_shape_out::RowStride + j] =
//           src1[i * tile_shape_in1::RowStride + j];
//     }
//   }
// }

// template <is_tile_data_v tile_shape_in0, is_tile_data_v tile_shape_in1,
//           is_tile_data_v tile_shape_out>
// void TASSEMBLE2(tile_shape_out &dst, tile_shape_in0 &src0, tile_shape_in1 &src1) {
//   static constexpr uint16_t dst_col = tile_shape_out::ValidCol;
//     TAssemble2_RowMajor<tile_shape_in0, tile_shape_in1,
//                            tile_shape_out>(dst.data(), src0.data(), src1.data());
// }


// [b, s, n_head, qk_rope_head_dim] -> [b, s, n_head, qk_rope_head_dim/2, 2]
// only input freq_cis [s, qk_rope_head_dim]  freqs_cis [s, qk_rope_head_dim]
// freqs = [freqs_cis_real, freqs_cis_imag]
// do complex multipy in last dim(a+bi * c+di = ac-bd + (bc+ad)i -> )
// then reshape to [b, s, n_head, qk_rope_head_dim]
template<typename dtype, const int row, const int rope_dim>
void apply_rotary_emb(dtype *x, dtype *freqs_cis){
    const uint32_t trow = 64;
    const uint32_t tcol = 32;
    const uint32_t rope_trow = trow*tcol/2;
    using gm_shape = global_tensor<dtype, RowMajor<row, rope_dim>>;
    using tile_shape = Tile<Location::Vec, dtype, trow, tcol, BLayout::RowMajor>;
    using tile_shape_rope = Tile<Location::Vec, dtype, rope_trow, 2, BLayout::RowMajor>;

    using tile_shape_half = Tile<Location::Vec, dtype, rope_trow, 1, BLayout::RowMajor>;

    uint32_t n_row = row/trow;
    uint32_t n_col = rope_dim/tcol;
    for(int i=0;i<n_row;i++){
        for(int j=0;j<n_col;j++){
            uint32_t offset = i * (trow * rope_dim) + j * tcol;
            gm_shape input(x+offset);
            tile_shape tin;
            tile_shape_rope resh_tin;
            TLOAD(tin, input);   // 64*32
            TRESHAPE(resh_tin, tin); // 64*32 -> 1024*2
            //TTRANS();           // 128*2 -> 2*128

            tile_shape_half tin_real;
            tile_shape_half tin_imag;
            TEXTRACT(tin_real, resh_tin, 0, 0);        // real 1024*1
            TEXTRACT(tin_imag, resh_tin, 0, 1);        // image 1024*1

            offset = i * (trow * rope_dim) + j * tcol;
            gm_shape freqs(freqs_cis+offset);
            tile_shape tfreqs;
            tile_shape_rope tfreqs_resh;
            TLOAD(tfreqs, freqs);
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
            Tile<Location::Vec, dtype, rope_trow, 4, BLayout::RowMajor> tout_tmp; // last col dim not used
            TASSEMBLE(tout_tmp, tout_real, tout_imag, padding);
            TEXTRACT(tout, tout_tmp, 0, 0);
            }
            tile_shape tout_resh;
            TRESHAPE(tout_resh, tout);

            TSTORE(input, tout_resh);
        }
    }
}

template<typename dtype, const int row, const int dim1, const int dim2>
void split(dtype *out1, dtype *out2, dtype *in){
    using gm_in = global_tensor<dtype, RowMajor<row, dim1+dim2>>;
    using gm_out1 = global_tensor<dtype, RowMajor<row, dim1>>;
    using gm_out2 = global_tensor<dtype, RowMajor<row, dim2>>;

    const uint32_t trow = 32;
    const uint32_t tcol = 32;

    using tile_shape = Tile<Location::Vec, dtype, trow, tcol, BLayout::RowMajor>;

    uint32_t n_row =  row/trow;
    uint32_t n_col1 = dim1/tcol;
    uint32_t n_col2 = dim2/tcol;

    for(int i=0;i<n_row;i++){
        for(int j=0;j<n_col1;j++){
            uint32_t offset = i * (trow * (dim1+dim2)) + j * tcol;
            gm_in input(in+offset);
            tile_shape tmp;
            TLOAD(tmp, input);

            offset = i * (trow * dim1) + j * tcol;
            gm_out1 output1(out1+offset);
            TSTORE(output1,tmp);
        }
    }

    for(int i=0;i<n_row;i++){
        for(int j=0;j<n_col2;j++){
            uint32_t offset = i * (trow * (dim1+dim2)) + j * tcol + dim1;
            gm_in input(in+offset);
            tile_shape tmp;
            TLOAD(tmp, input);

            offset = i * (trow * dim2) + j * tcol;
            gm_out2 output2(out2+offset);
            TSTORE(output2,tmp);
        }
    }
}

template<typename dtype, const int row, const int dim1, const int dim2>
void concat(dtype *out, dtype *in1, dtype *in2){
    using gm_in1 = global_tensor<dtype, RowMajor<row, dim1>>;
    using gm_in2 = global_tensor<dtype, RowMajor<row, dim2>>;

    using gm_out = global_tensor<dtype, RowMajor<row, dim1+dim2>>;

    const uint32_t trow = 64;
    const uint32_t tcol = 64;

    using tile_shape = Tile<Location::Vec, dtype, trow, tcol, BLayout::RowMajor>;

    uint32_t n_row = row/trow;
    uint32_t n_col1 = dim1/tcol;
    uint32_t n_col2 = dim2/tcol;

    for(int i=0;i<n_row;i++){
        for(int j=0;j<n_col1;j++){
            uint32_t offset = i * (trow * dim1) + j * tcol;
            gm_in1 input1(in1 + offset);
            tile_shape tmp;
            TLOAD(tmp, input1);

            offset = i * (trow * (dim1+dim2)) + j * tcol;
            gm_out output(out+offset);
            TSTORE(output, tmp);
        }
    }

    for(int i=0;i<n_row;i++){
        for(int j=0;j<n_col2;j++){
            uint32_t offset = i * (trow * dim2) + j * tcol;

            gm_in2 input2(in2 + offset);
            tile_shape tmp;
            TLOAD(tmp, input2);

            offset = dim1 + i * (trow * (dim1+dim2)) + j * tcol;
            gm_out output(out+offset);
            TSTORE(output, tmp);
        }
    }
}

//[dim0, dim1, dim2, dim3] -> [dim0, dim2, dim1, dim3]
template<typename dtype, const int dim0, const int dim1, const int dim2, const int dim3>
void permute(dtype *out, dtype *in){
    using gm_shape_in  = global_tensor<dtype, RowMajor<1, dim3>>;
    using gm_shape_out = global_tensor<dtype, RowMajor<1, dim3>>;

    using tile_shape = Tile<Location::Vec, dtype, 1, dim3, BLayout::RowMajor>;

    //naive c impl
    // out[i,k,j,m] = in[i,j,k,m];
    for(int i=0;i<dim0;i++){
        for(int j=0;j<dim1;j++){
            for(int k=0;k<dim2;k++){
                    uint32_t src_offset = i*dim1*dim2*dim3+j*dim2*dim3+k*dim3;
                    uint32_t dst_offset = i*dim2*dim1*dim3+k*dim1*dim3+j*dim3;

                    tile_shape tmp;
                    gm_shape_in input(in+src_offset);
                    gm_shape_out ouput(out+dst_offset);
                    TLOAD(tmp, input);
                    TSTORE(ouput, tmp);
            }
        }
    }
}

//[row, dim] -> [row, ext_dim, dim]
template<typename dtype, const int row, const int dim, const int ext_dim>
void expand(dtype *out, dtype *in){
    using gm_in = global_tensor<dtype, RowMajor<row, dim>>;
    using gm_out = global_tensor<dtype, RowMajor<row*ext_dim, dim>>;

    using tile_shape = Tile<Location::Vec, dtype, 1, dim, BLayout::RowMajor>;

    //[i,j,k] = [i,j,*,k]
    for(int i=0;i<row;i++){
        uint32_t offset = i * dim;
        gm_in input(in + offset);
        tile_shape tmp;
        TLOAD(tmp, input);
        for(int j=0;j<ext_dim;j++){
            offset = i * ext_dim * dim + j * dim;
            gm_out output(out+offset);
            TSTORE(output, tmp);
        }
    }
}

template<typename dtype, const int bsz, const int seq_len, typename args>
void MLA(Tensor<dtype, bsz, seq_len, args::dim> & out,
        Tensor<dtype, bsz, seq_len, args::dim> &x,
        Tensor<dtype, seq_len, args::qk_rope_head_dim> & freqs_cis,
        Tensor<dtype, bsz, args::n_heads, seq_len>* atten_mask=nullptr){
    // do down projection to q_down then do up projection to q_up
    // weight matrix should be initiialized to memory, here just hack for temp variable;

    //writeTensorToFile<dtype, bsz*seq_len, args::dim>(mla_input.data(), "mla_input_cpp.txt");
    //writeTensorToFile<dtype, seq_len, args::qk_rope_head_dim>(freqs_cis.data(), "freqs_cis_cpp.txt");

    Tensor<dtype, bsz, seq_len, args::n_heads*args::qk_head_dim> q_up;
    {
        if constexpr(ModelArgs::q_lora_rank == 0){ //no lora_rank: [b,s,dim] -> [b, s, n_heads*qk_head_dim]
            Tensor<dtype, args::dim, args::n_heads*args::qk_head_dim> Wq_up(1);
            projection<dtype, bsz, seq_len, args::dim, args::n_heads*args::qk_head_dim>(q_up, x, Wq_up);
        }else{ //for lora rank [b,s,dim] -> [b,s,q_lora_rank] -> [b,s,n_head*qk_head_dim]
            Tensor<dtype, args::dim, args::q_lora_rank> Wq_down(1);
            Tensor<dtype, bsz, seq_len, args::q_lora_rank> q_down;
            projection<dtype, bsz, seq_len, args::dim, args::q_lora_rank>(q_down, x, Wq_down);
            for(int i=0;i<bsz;i++){
                rmsnorm<dtype,q_down.shape[1],q_down.shape[2],32,32>(q_down.data(i), q_down.data(i));
            }
            Tensor<dtype, args::q_lora_rank, args::n_heads*args::qk_head_dim> Wq_up(1);
            projection<dtype, bsz, seq_len, args::q_lora_rank, args::n_heads*args::qk_head_dim>(q_up, q_down, Wq_up);
        }
    }

    //split q to q_nope([b,s,n_heads,qk_nope_head_dim]), q_rope([b,s,n_heads, qk_rope_head_dim])
    Tensor<dtype, bsz, seq_len, args::n_heads*args::qk_nope_head_dim> q_nope;
    Tensor<dtype, bsz, seq_len, args::n_heads*args::qk_rope_head_dim> q_pe;
    split<dtype, bsz*seq_len*args::n_heads, args::qk_nope_head_dim, args::qk_rope_head_dim>(q_nope.data(), q_pe.data(), q_up.data());

    //for q_pe doing rotary embedding
    Tensor<dtype, bsz, args::n_heads, seq_len, args::qk_rope_head_dim> q_perm;
    permute<dtype, bsz, seq_len, args::n_heads, args::qk_rope_head_dim>(q_perm.data(), q_pe.data());
    for(int i=0;i<bsz;i++){
        for(int j=0;j<args::n_heads;j++){
            apply_rotary_emb<dtype, seq_len, args::qk_rope_head_dim>(q_perm.data(i,j), freqs_cis.data());
        }
    }
    permute<dtype, bsz, args::n_heads, seq_len, args::qk_rope_head_dim>(q_pe.data(), q_perm.data());

    //writeTensorToFile<dtype, bsz*seq_len, args::n_heads*args::qk_rope_head_dim>(q_pe.data(), "q_pe_cpp.txt");

    Tensor<dtype, bsz, seq_len, args::n_heads*args::qk_head_dim> q_attn;
    //concat q_nope and q_pe to Q
    concat<dtype, bsz*seq_len*args::n_heads, args::qk_nope_head_dim, args::qk_rope_head_dim>(q_attn.data(), q_nope.data(), q_pe.data());

    // do down projection to k_rope+k_lora_rank, and split to k_rope, k_lora_rank
    Tensor<dtype, bsz, seq_len, args::kv_lora_rank> kv;
    Tensor<dtype, bsz, seq_len, args::qk_rope_head_dim> k_pe;
    {
        Tensor<dtype, args::dim, args::kv_lora_rank+args::qk_rope_head_dim> Wkv_down(1);
        Tensor<dtype, bsz, seq_len, args::kv_lora_rank+args::qk_rope_head_dim> kv_down;
        projection<dtype, bsz, seq_len, args::dim, args::kv_lora_rank+args::qk_rope_head_dim>(kv_down, x, Wkv_down);
        split<dtype, bsz*seq_len, args::kv_lora_rank, args::qk_rope_head_dim>(kv.data(), k_pe.data(), kv_down.data());
    }

    // then for k_lora_rank do up projection to k_nope + v
    // for k_pe do rotary emb
    Tensor<dtype, bsz, seq_len, args::n_heads*(args::qk_nope_head_dim+args::v_head_dim)> kv_up;
    {
        for(int i=0;i<bsz;i++){
            rmsnorm<dtype,kv.shape[1],kv.shape[2],32,32>(kv.data(i), kv.data(i));
        }
        Tensor<dtype, args::kv_lora_rank, args::n_heads*(args::qk_nope_head_dim+args::v_head_dim)> Wkv_up(1);
        projection<dtype, bsz, seq_len, args::kv_lora_rank, args::n_heads*(args::qk_nope_head_dim+args::v_head_dim)>(kv_up, kv, Wkv_up);
        for(int i=0;i<bsz;i++){
            apply_rotary_emb<dtype, seq_len, args::qk_rope_head_dim>(k_pe.data(i), freqs_cis.data());
        }
    }

    Tensor<dtype, bsz, seq_len, args::n_heads*args::qk_nope_head_dim> k_nope;
    Tensor<dtype, bsz, seq_len, args::n_heads*args::v_head_dim> v_attn;
    split<dtype, bsz*seq_len*args::n_heads, args::qk_nope_head_dim, args::v_head_dim>(k_nope.data(), v_attn.data(), kv_up.data());

    Tensor<dtype, bsz, seq_len, args::n_heads*args::qk_head_dim> k_attn;
    Tensor<dtype, bsz, seq_len, args::n_heads, args::qk_rope_head_dim> k_pe_expand;
    expand<dtype, bsz*seq_len, args::qk_rope_head_dim, args::n_heads>(k_pe_expand.data(), k_pe.data());
    concat<dtype, bsz*seq_len*args::n_heads, args::qk_nope_head_dim, args::qk_rope_head_dim>(k_attn.data(), k_nope.data(), k_pe_expand.data());//FIXME

    Tensor<dtype, bsz, args::n_heads, seq_len, args::qk_head_dim> q_attn_pm;
    Tensor<dtype, bsz, args::n_heads, seq_len, args::qk_head_dim> k_attn_pm;
    Tensor<dtype, bsz, args::n_heads, seq_len, args::v_head_dim>  v_attn_pm;

    //writeTensorToFile<dtype, bsz*seq_len*args::n_heads, args::qk_head_dim>(q_attn.data(), "q_attn_cpp.txt");
    //writeTensorToFile<dtype, bsz*seq_len*args::n_heads, args::qk_head_dim>(k_attn.data(), "k_attn_cpp.txt");
    //writeTensorToFile<dtype, bsz*seq_len*args::n_heads, args::v_head_dim>(v_attn.data(), "v_attn_cpp.txt");

    permute<dtype, bsz, seq_len, args::n_heads, args::qk_head_dim>(q_attn_pm.data(), q_attn.data()); //[b,s,n_heads,qk_head_dim] permute to [b,n_heads,s,qk_head_dim]
    permute<dtype, bsz, seq_len, args::n_heads, args::qk_head_dim>(k_attn_pm.data(), k_attn.data()); //[b,s,n_heads,qk_head_dim] permute to [b,n_heads,s,qk_head_dim]
    permute<dtype, bsz, seq_len, args::n_heads, args::v_head_dim>(v_attn_pm.data(), v_attn.data()); //[b,s,n_heads,v_head_dim]  permute to [b,n_heads,s,v_head_dim]

    Tensor<dtype, bsz, seq_len, args::n_heads*args::v_head_dim>  attn_out;
    Tensor<dtype, bsz, args::n_heads, seq_len, args::v_head_dim> attn_tmp;
    for(int i=0;i<bsz;i++){
        for(int j=0;j<args::n_heads;j++){
            // NOTE: v_head_dim == qk_head_dim since flash_attention impl
            // FIXME: consider attn_mask in flash_attention before softmax
            // attn_mask: lower triangle matrix with upper fill with -inf
            flash_attention<seq_len,args::qk_head_dim,args::v_head_dim,32,32>(attn_tmp.data(i,j), q_attn_pm.data(i,j), k_attn_pm.data(i,j), v_attn_pm.data(i,j));
        }
    }

    //[b,n,s,v_head_dim] -> [b,s,n,v_head_dim]
    permute<dtype, bsz, args::n_heads, seq_len, args::v_head_dim>(attn_out.data(), attn_tmp.data());

    //writeTensorToFile<dtype, bsz*seq_len*args::n_heads, args::v_head_dim>(attn_out.data(), "attn_out_cpp.txt");

    //final output projection
    {
        Tensor<dtype, args::n_heads*args::v_head_dim, args::dim> Wout(1);
        projection<dtype, bsz, seq_len, args::n_heads*args::v_head_dim, args::dim>(out, attn_out, Wout);
    }
    //writeTensorToFile<dtype, bsz*seq_len, args::dim>(out.data(), "mla_out_cpp.txt");
    return;
}

#endif