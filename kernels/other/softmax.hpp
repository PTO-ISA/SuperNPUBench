#include <common/pto_tileop.hpp>

using namespace pto;

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
            TLOAD(tsrc, gsrc);

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
            TLOAD(tsrc, gsrc);

            tile_shape gMax;
            tile_shape gSum;
            TEXPANDCOL(gMax, tAccMax);
            TEXPANDCOL(gSum, tAccSum);

            TSUB(tsrc, tsrc, gMax);
            TEXP(tsrc, tsrc);
            TDIV(tsrc, tsrc, gSum);

            gm_shape gdst(dst+offset);
            TSTORE(gdst, tsrc);
        }
    }
}

template <typename tile_shape_size1, typename tile_shape_size2,
          typename tile_shape_size3>
void __vec__ OnlineSoftMax_RowMajor(
    typename tile_shape_size1::TileDType __out__ dst_out,
    const typename tile_shape_size2::TileDType __in__ src_S,
    const typename tile_shape_size3::TileDType __in__ src_max_old,
    const typename tile_shape_size3::TileDType __in__ src_sum_old_scale) {
  unsigned i = blkv_get_index_x(); // row

  __vbuf__ typename tile_shape_size1::DType *dst_P_ptr = blkv_get_tile_ptr(dst_out);
  __vbuf__ typename tile_shape_size2::DType *src_S_ptr = blkv_get_tile_ptr(src_S);
  typename tile_shape_size3::DType *src_max_old_ptr =
      blkv_get_tile_ptr(src_max_old);
  typename tile_shape_size3::DType *src_sum_old_scale_ptr =
      blkv_get_tile_ptr(src_sum_old_scale);

  // TROWMAX AND TMAX
  typename tile_shape_size3::DType max_val = src_max_old_ptr[i];
  typename tile_shape_size3::DType data_sum = 0;
#pragma clang loop unroll(full)
  for (size_t j = 0; j < tile_shape_size2::ValidCol; ++j) {
    size_t now_idx =
        i * tile_shape_size2::RowStride + j * tile_shape_size2::ColStride;
    typename tile_shape_size2::DType now_val = src_S_ptr[now_idx];
    max_val = blkv_max(max_val, now_val);
    data_sum = data_sum + blkv_fexp(now_val - max_val);
  }
  data_sum = data_sum +
             blkv_fexp(src_max_old_ptr[i] - max_val) * src_sum_old_scale_ptr[i];

#pragma clang loop unroll(full)
  for (size_t j = 0; j < tile_shape_size2::ValidCol; ++j) {
    size_t now_idx =
        i * tile_shape_size2::RowStride + j * tile_shape_size2::ColStride;
    size_t now_idx1 = i * tile_shape_size1::RowStride +
                      j * tile_shape_size1::ColStride +
                      tile_shape_size2::RowStride;
    dst_P_ptr[now_idx1] = blkv_fexp(src_S_ptr[now_idx] - max_val);
  }
  dst_P_ptr[i * tile_shape_size1::ValidCol] = max_val;
  dst_P_ptr[1 + i * tile_shape_size1::ValidCol] = data_sum;
}

// tile_shape_size1=64*66
// tile_shape_size2=64*1
template <typename tile_shape_size1, typename tile_shape_size2,
          typename tile_shape_size3>
void OnlineSoftMax(tile_shape_size1 &dst_p, tile_shape_size2 &src_S,
                   tile_shape_size3 &src_max_old,
                   tile_shape_size3 &src_sum_old_scale) {
  static constexpr size_t row = tile_shape_size1::ValidRow;
  static constexpr size_t col = tile_shape_size1::ValidCol;
  OnlineSoftMax_RowMajor<tile_shape_size1, tile_shape_size2, tile_shape_size3>
      <<<row, 1, 1>>>(dst_p.data(), src_S.data(), src_max_old.data(),
                      src_sum_old_scale.data());
}