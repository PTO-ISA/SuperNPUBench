#ifndef FLASH_ATTENTION_HPP
#define FLASH_ATTENTION_HPP

// Flash attention v2
#include <common/pto_tileop.hpp>

template <int S, int qD, int vD, int kTm, int kTk>
void flash_attention(float* out_ptr, float* q_ptr, float* k_ptr, float* v_ptr) {
    // 全局张量形状
    using gmQ = global_tensor<float, RowMajor<S, qD>>;  // Q: [S×qD]
    using gmK = global_tensor<float, ColMajor<qD, S>>;  // K: [qD×S]
    using gmV = global_tensor<float, RowMajor<S, vD>>;  // V: [S×vD]
    using gmO = global_tensor<float, RowMajor<S, vD>>;  // O: [SxvD]
    // tile 寄存器形状
    using tileQ      = TileLeft<float, kTm, qD>;       // [kTm×qD]
    using tileK      = TileRight<float, qD, kTk>;      // [vD×kTk]
    using tileW_out  = TileAcc<float, kTm, kTk>;      // [kTm×kTk]
    using tileW      = Tile<Location::Vec, float, kTm, kTk, BLayout::RowMajor>;
    using tileW_left = TileLeft<float, kTm, kTk>;

    using tileO_out  = TileAcc<float, kTm, vD>;
    using tileO      = Tile<Location::Vec, float, kTm, vD, BLayout::RowMajor>; // [kTm×vD]

    using tileV      = TileRight<float, kTk, vD>;     // [kTk×vD]
    using tileMax    = Tile<Location::Vec, float, kTm, 8, BLayout::RowMajor, kTm, 1>; // [kTm×1]
    using tileSum    = Tile<Location::Vec, float, kTm, 8, BLayout::RowMajor, kTm, 1>; // [kTm×1]
    using tileScale  = Tile<Location::Vec, float, kTm, 8, BLayout::RowMajor, kTm, 1>; // [kTm×1]

    // 全局迭代器
    using itQ = global_iterator<gmQ, tileQ>;
    using itK = global_iterator<gmK, tileK>;
    using itV = global_iterator<gmV, tileV>;
    using itO = global_iterator<gmO, tileO>;

    itQ gIterQ(q_ptr);
    itK gIterK(k_ptr);
    itV gIterV(v_ptr);
    itO gIterO(out_ptr);

    const float scale = 1.0f / sqrt((float)qD);
    const int Qb = (S + kTm - 1) / kTm;
    const int Kb = (S + kTk - 1) / kTk;

    // 对每个 Q-block (i)
    for (int i = 0; i < Qb; ++i) {
        // 加载当前Q块 (仅一次)
        tileQ tQ;
        auto gQ = gIterQ(i,0);
        TLOAD(tQ, gQ);

        // 初始化状态: 最大值/指数和/输出累加
        tileMax tMax;
        TEXPANDSCALAR(tMax, -1e30f); // 初始化为极小值
        tileSum tSum(0);              // 指数和归零
        tileO_out tO_out;                 // 输出累加归零
        tileO tO(0), tRescaleO(0);

        // 单次遍历所有K/V块
        #pragma clang loop unroll(full)
        for (int j = 0; j < Kb; ++j) {
        // 加载K_j和V_j
        auto gK = gIterK(0,j);
        auto gV = gIterV(j,0);
        tileK tK; TLOAD(tK, gK);
        tileV tV; TLOAD(tV, gV);

        // 计算注意力分数块
        tileW_out tW_out;
        MATMUL(tW_out, tQ, tK);

        // Nz -> RowMajor
        tileW tW;
        TCVT(tW, tW_out);
        TMULS(tW, tW, scale);

        // 计算当前块的行最大值
        tileMax tLocalMax;
        TROWMAX(tLocalMax, tW);

        // 更新全局行最大值
        tileMax tNewMax;
        TMAX(tNewMax, tMax, tLocalMax);

        // 计算旧状态的缩放因子: exp(old_max - new_max)
        tileScale tScaleOld;
        TSUB(tScaleOld, tMax, tNewMax);
        TEXP(tScaleOld, tScaleOld);

        // 调整历史指数和: sum = sum * scale_old
        tileSum tScaledSum;
        TMUL(tScaledSum, tSum, tScaleOld);

        // 计算当前块的指数值 (使用新最大值)
        tileW tNewMaxExpanded;
        TEXPANDCOL(tNewMaxExpanded, tNewMax);
        TSUB(tW, tW, tNewMaxExpanded);
        TEXP(tW, tW);

        // 计算当前块的指数和
        tileSum tLocalSum;
        TROWSUM(tLocalSum, tW);

        // 更新全局指数和
        TADD(tSum, tScaledSum, tLocalSum);

        // 调整历史累加值: O = O * scale_old
        tileO tScaleOldExpanded;
        TEXPANDCOL(tScaleOldExpanded, tScaleOld);

        TADD(tO, tO, tRescaleO);
        TMUL(tRescaleO, tO, tScaleOldExpanded);

        // RowMajor -> Nz
        tileW_left tW_left;
        TCVT(tW_left, tW);
        // 计算当前块的加权输出: O_j = W * V
        MATMUL(tO_out, tW_left, tV);
        TCVT(tO, tO_out);
        // 更新最大值状态
        tMax = tNewMax;
        }

        TADD(tO, tO, tRescaleO);
        // 归一化输出: O = O / sum
        tileSum tInvSum;
        TRECIP(tInvSum, tSum);
        tileO tInvSumExpanded;
        TEXPANDCOL(tInvSumExpanded, tInvSum);
        TMUL(tO, tO, tInvSumExpanded);

        Tile<Location::Vec, float, kTm, vD, BLayout::RowMajor> tO_cast;
        TCAST(tO_cast, tO);
        // 写回全局内存
        auto dstO = gIterO(i, 0);
        TSTORE(dstO, tO_cast);
    }
}

template<typename tileSrc, typename tileMax>
void __vec__ flashsoftmax_new_max(
                        typename tileMax::TileDType __out__ new_max,
                        const typename tileSrc::TileDType __in__ src,
                        const typename tileMax::TileDType __in__ old_max,
                        const typename tileSrc::DType src_scale
                    ){
    uint16_t i = blkv_get_index_x();

    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileMax::DType *new_max_ptr = blkv_get_tile_ptr(new_max);
    __vbuf__ typename tileMax::DType *old_max_ptr = blkv_get_tile_ptr(old_max);

    size_t max_idx = i*tileMax::RowStride;

    typename tileMax::DType old_max_val = old_max_ptr[max_idx];
    typename tileMax::DType upd_max = old_max_val;

    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileSrc::ValidCol;j++){
        size_t src_idx =  i * tileSrc::RowStride + j * tileSrc::ColStride;
        upd_max = blkv_max(upd_max, src_ptr[src_idx] * src_scale);
    }

    new_max_ptr[max_idx] = upd_max;
}

template<typename tileMax, typename tileScale>
void __vec__ flashsoftmax_scale(
                        typename tileScale::TileDType __out__ scale,
                        const typename tileMax::TileDType __in__ old_max,
                        const typename tileMax::TileDType __in__ new_max
                    ){
    size_t i = blkv_get_index_x();

    size_t idx = i * tileMax::RowStride;
    blkv_get_tile_ptr(scale)[idx] = blkv_fexp(blkv_get_tile_ptr(old_max)[idx] - blkv_get_tile_ptr(new_max)[idx]);
}

template<typename tileSrc, typename tileMax>
void __vec__ flashsoftmax_src_exp_nd(
                        typename tileSrc::TileDType __out__ src_exp,
                        const typename tileSrc::TileDType __in__ src,
                        const typename tileMax::TileDType __in__ new_max,
                        const typename tileSrc::DType src_scale
                    ){
    size_t i = blkv_get_index_x();
    size_t j = blkv_get_index_y();

    size_t idx = j * tileSrc::RowStride + i;
    size_t idx_max = j * tileMax::RowStride;
    blkv_get_tile_ptr(src_exp)[idx] = blkv_fexp(blkv_get_tile_ptr(src)[idx]*src_scale - blkv_get_tile_ptr(new_max)[idx_max]);
}

template<typename tileSrc, typename tileMax>
void __vec__ flashsoftmax_src_exp_dn(
                        typename tileSrc::TileDType __out__ src_exp,
                        const typename tileSrc::TileDType __in__ src,
                        const typename tileMax::TileDType __in__ new_max,
                        const typename tileSrc::DType src_scale
                    ){
    size_t i = blkv_get_index_x();
    size_t j = blkv_get_index_y();

    size_t idx = j * tileSrc::ColStride + i;
    size_t idx_max = i * tileMax::RowStride;
    blkv_get_tile_ptr(src_exp)[idx] = blkv_fexp(blkv_get_tile_ptr(src)[idx]*src_scale - blkv_get_tile_ptr(new_max)[idx_max]);
}

template<typename tileSrc, typename tileSum, typename tileScale>
void __vec__ flashsoftmax_new_sum(
                        typename tileSum::TileDType __out__ new_sum,
                        const typename tileSrc::TileDType __in__ src,
                        const typename tileSum::TileDType __in__ old_sum,
                        const typename tileScale::TileDType __in__ scale
                    ){
    uint16_t i = blkv_get_index_x();

    __vbuf__ typename tileSrc::DType   *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileSum::DType   *old_sum_ptr = blkv_get_tile_ptr(old_sum);
    __vbuf__ typename tileScale::DType *scale_ptr = blkv_get_tile_ptr(scale);

    size_t sum_idx = i*tileSum::RowStride;

    typename tileSum::DType upd_sum = old_sum_ptr[sum_idx] * scale_ptr[sum_idx];

    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileSrc::ValidCol;j++){
        size_t src_idx =  i * tileSrc::RowStride + j * tileSrc::ColStride;
        upd_sum += src_ptr[src_idx];
    }

    blkv_get_tile_ptr(new_sum)[sum_idx] = upd_sum;
}

template<typename tileOut, typename tileScale>
void __vec__ flashsoftmax_new_out(
                        typename tileOut::TileDType __out__ new_out,
                        const typename tileOut::TileDType __in__ old_out,
                        const typename tileScale::TileDType __in__ scale,
                        const typename tileOut::TileDType __in__ rescale_out
                    ){
    size_t i = blkv_get_index_x();
    size_t j = blkv_get_index_y();

    size_t idx = j * tileOut::RowStride + i;
    size_t idx_scale = j * tileScale::RowStride;

    blkv_get_tile_ptr(new_out)[idx] = (blkv_get_tile_ptr(rescale_out)[idx] + blkv_get_tile_ptr(old_out)[idx]) * blkv_get_tile_ptr(scale)[idx_scale];
}

template <typename tileSrc, typename tileOut, typename tileMax, typename tileSum, typename tileScale>
void flashsoftmax_kernel(
                        tileOut &rescale_out,
                        tileMax &new_max,
                        tileSum &new_sum,
                        tileSrc &src_exp,
                        tileSrc &src,
                        tileOut &old_out,
                        tileMax &old_max,
                        tileSum &old_sum,
                        const typename tileSrc::DType &src_scale
                        ){
    tileScale scale;
    flashsoftmax_new_max<tileSrc, tileMax><<<tileMax::ValidRow, 1, 1>>>(new_max.data(), src.data(), old_max.data(), src_scale);
    flashsoftmax_scale<tileMax, tileScale><<<tileMax::ValidRow, 1, 1>>>(scale.data(), old_max.data(), new_max.data());
    if constexpr(tileSrc::isRowMajor){
        flashsoftmax_src_exp_nd<tileSrc, tileMax><<<tileSrc::ValidCol, tileSrc::ValidRow, 1>>>(src_exp.data(), src.data(), new_max.data(), src_scale);
    }else{
        flashsoftmax_src_exp_dn<tileSrc, tileMax><<<tileSrc::ValidRow, tileSrc::ValidCol, 1>>>(src_exp.data(), src.data(), new_max.data(), src_scale);
    }
    flashsoftmax_new_sum<tileSrc, tileSum, tileScale><<<tileSum::ValidRow, 1, 1>>>(new_sum.data(), src_exp.data(), old_sum.data(), scale.data());
    flashsoftmax_new_out<tileOut, tileScale><<<tileOut::ValidCol, tileOut::ValidRow, 1>>>(rescale_out.data(), old_out.data(), scale.data(), rescale_out.data());
}

template <int S, int qD, int vD, int kTm, int kTk>
void flash_attention_opt(float* out_ptr, float* q_ptr, float* k_ptr, float* v_ptr) {
    // 全局张量形状
    using gmQ = global_tensor<float, RowMajor<S, qD>>;  // Q: [S×qD]
    using gmK = global_tensor<float, ColMajor<qD, S>>;  // K: [qD×S]
    using gmV = global_tensor<float, RowMajor<S, vD>>;  // V: [S×vD]
    using gmO = global_tensor<float, RowMajor<S, vD>>;  // O: [SxvD]
    // tile 寄存器形状
    using tileQ      = TileLeft<float, kTm, qD>;       // [kTm×qD]
    using tileK      = TileRight<float, qD, kTk>;      // [vD×kTk]
    using tileW_out  = TileAcc<float, kTm, kTk>;      // [kTm×kTk]
    using tileW      = Tile<Location::Vec, float, kTm, kTk, BLayout::RowMajor>;
    using tileW_left = TileLeft<float, kTm, kTk>;

    using tileO_out  = TileAcc<float, kTm, vD>;
    using tileO      = Tile<Location::Vec, float, kTm, vD, BLayout::RowMajor>; // [kTm×vD]

    using tileV      = TileRight<float, kTk, vD>;     // [kTk×vD]
    using tileMax    = Tile<Location::Vec, float, kTm, 8, BLayout::RowMajor, kTm, 1>; // [kTm×1]
    using tileSum    = Tile<Location::Vec, float, kTm, 8, BLayout::RowMajor, kTm, 1>; // [kTm×1]
    using tileScale  = Tile<Location::Vec, float, kTm, 8, BLayout::RowMajor, kTm, 1>; // [kTm×1]

    // 全局迭代器
    using itQ = global_iterator<gmQ, tileQ>;
    using itK = global_iterator<gmK, tileK>;
    using itV = global_iterator<gmV, tileV>;
    using itO = global_iterator<gmO, tileO>;

    itQ gIterQ(q_ptr);
    itK gIterK(k_ptr);
    itV gIterV(v_ptr);
    itO gIterO(out_ptr);

    const float scale = 1.0f / sqrt((float)qD);
    const int Qb = (S + kTm - 1) / kTm;
    const int Kb = (S + kTk - 1) / kTk;

    // 对每个 Q-block (i)
    for (int i = 0; i < Qb; ++i) {
        // 加载当前Q块 (仅一次)
        tileQ tQ;
        auto gQ = gIterQ(i, 0);
        TLOAD(tQ, gQ);

        // 初始化状态: 最大值/指数和/输出累加
        tileMax tMax;
        TEXPANDSCALAR(tMax, -1e30f); // 初始化为极小值
        tileSum tSum(0);              // 指数和归零
        tileO_out tO_out;                 // 输出累加归零
        tileO tO(0), tRescaleO(0);

        // 单次遍历所有K/V块
        #pragma clang loop unroll(full)
        for (int j = 0; j < Kb; ++j) {
        // 加载K_j和V_j
        auto gK = gIterK(0, j);
        auto gV = gIterV(j, 0);
        tileK tK; TLOAD(tK, gK);
        tileV tV; TLOAD(tV, gV);

        // 计算注意力分数块
        tileW_out tW_out;
        MATMUL(tW_out, tQ, tK);

        // Nz -> RowMajor
        tileW tW;
        TCVT(tW, tW_out);

        tileMax tNewMax;
        tileSum tNewSum;
        tileW tExpW;
        flashsoftmax_kernel<tileW, tileO, tileMax, tileSum, tileScale>(tRescaleO, tNewMax, tNewSum, tExpW, tW, tO, tMax, tSum, scale);

        // RowMajor -> Nz
        tileW_left tW_left;
        TCVT(tW_left, tExpW);
        // 计算当前块的加权输出: O_j = W * V
        MATMUL(tO_out, tW_left, tV);
        TCVT(tO, tO_out);
        // 更新最大值状态
        tMax = tNewMax;
        tSum = tNewSum;
        }

        TADD(tO, tO, tRescaleO);
        // 归一化输出: O = O / sum
        tileSum tInvSum;
        TRECIP(tInvSum, tSum);
        tileO tInvSumExpanded;
        TEXPANDCOL(tInvSumExpanded, tInvSum);
        TMUL(tO, tO, tInvSumExpanded);

        Tile<Location::Vec, float, kTm, vD, BLayout::RowMajor> tO_cast;
        TCAST(tO_cast, tO);
        // 写回全局内存
        auto dstO = gIterO(i, 0);
        TSTORE(dstO, tO_cast);
    }
}

template <int S, int qD, int vD, int kTm, int kTk>
void flash_attention_opt2(float* out_ptr, float* q_ptr, float* k_ptr, float* v_ptr) {
    // 全局张量形状
    using gmQ = global_tensor<float, RowMajor<S, qD>>;  // Q: [S×qD]
    using gmK = global_tensor<float, ColMajor<qD, S>>;  // K: [qD×S]
    using gmV = global_tensor<float, RowMajor<S, vD>>;  // V: [S×vD]
    using gmO = global_tensor<float, RowMajor<S, vD>>;  // O: [SxvD]
    // tile 寄存器形状
    using tileQ      = TileLeft<float, kTm, qD>;       // [kTm×qD]
    using tileK      = TileRight<float, qD, kTk>;      // [vD×kTk]
    using tileW_out  = TileAcc<float, kTm, kTk>;      // [kTm×kTk]
    using tileW      = Tile<Location::Vec, float, kTm, kTk, BLayout::ColMajor>;
    using tileW_left = TileLeft<float, kTm, kTk>;

    using tileO_out  = TileAcc<float, kTm, vD>;
    using tileO      = Tile<Location::Vec, float, kTm, vD, BLayout::RowMajor>; // [kTm×vD]

    using tileV      = TileRight<float, kTk, vD>;     // [kTk×vD]
    using tileMax    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>; // [kTm×1]
    using tileSum    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>; // [kTm×1]
    using tileScale  = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>; // [kTm×1]

    // 全局迭代器
    using itQ = global_iterator<gmQ, tileQ>;
    using itK = global_iterator<gmK, tileK>;
    using itV = global_iterator<gmV, tileV>;
    using itO = global_iterator<gmO, tileO>;

    itQ gIterQ(q_ptr);
    itK gIterK(k_ptr);
    itV gIterV(v_ptr);
    itO gIterO(out_ptr);

    const float scale = 1.0f / sqrt((float)qD);
    const int Qb = (S + kTm - 1) / kTm;
    const int Kb = (S + kTk - 1) / kTk;

    // 对每个 Q-block (i)
    for (int i = 0; i < Qb; ++i) {
        // 加载当前Q块 (仅一次)
        tileQ tQ;
        auto gQ = gIterQ(i, 0);
        TLOAD(tQ, gQ);

        // 初始化状态: 最大值/指数和/输出累加
        tileMax tMax;
        TEXPANDSCALAR(tMax, -1e30f); // 初始化为极小值
        tileSum tSum(0);              // 指数和归零
        tileO_out tO_out;                 // 输出累加归零
        tileO tO(0), tRescaleO(0);

        // 单次遍历所有K/V块
        #pragma clang loop unroll(full)
        for (int j = 0; j < Kb; ++j) {
        // 加载K_j和V_j
        auto gK = gIterK(0, j);
        auto gV = gIterV(j, 0);
        tileK tK; TLOAD(tK, gK);
        tileV tV; TLOAD(tV, gV);

        // 计算注意力分数块
        tileW_out tW_out;
        MATMUL(tW_out, tQ, tK);

        // Nz -> RowMajor
        tileW tW;
        TCVT(tW, tW_out);

        tileMax tNewMax;
        tileSum tNewSum;
        tileW tExpW;
        flashsoftmax_kernel<tileW, tileO, tileMax, tileSum, tileScale>(tRescaleO, tNewMax, tNewSum, tExpW, tW, tO, tMax, tSum, scale);

        // RowMajor -> Nz
        tileW_left tW_left;
        TCVT(tW_left, tExpW);
        // 计算当前块的加权输出: O_j = W * V
        MATMUL(tO_out, tW_left, tV);
        TCVT(tO, tO_out);
        // 更新最大值状态
        tMax = tNewMax;
        tSum = tNewSum;
        }

        TADD(tO, tO, tRescaleO);
        // 归一化输出: O = O / sum
        tileSum tInvSum;
        TRECIP(tInvSum, tSum);
        tileO tInvSumExpanded;
        TEXPANDCOL(tInvSumExpanded, tInvSum);
        TMUL(tO, tO, tInvSumExpanded);

        Tile<Location::Vec, float, kTm, vD, BLayout::RowMajor> tO_cast;
        TCAST(tO_cast, tO);
        // 写回全局内存
        auto dstO = gIterO(i, 0);
        TSTORE(dstO, tO_cast);
    }
}

/* Dont' USE!!! because PTO not allow fractal vec computation*/
template <int S, int qD, int vD, int kTm, int kTk>
void flash_attention_frac(float *out_ptr,
                     float *q_ptr,
                     float *k_ptr,
                     float *v_ptr) {
  // 全局张量形状
  using gmQ = global_tensor<float, RowMajor<S, qD>>;  // Q: [S×qD]
  using gmK = global_tensor<float, ColMajor<qD, S>>;  // K: [qDxS]
  using gmV = global_tensor<float, RowMajor<S, vD>>;  // V: [S×vD]
  using gmO = global_tensor<float, RowMajor<S, vD>>;  // O: [S×vD]

  // tile 寄存器形状
  using tileQ      = TileLeft<float, kTm, qD>;    // [kTm×qD]
  using tileK      = TileRight<float, qD, kTk>;    // [qD×kTk]
  using tileS      = TileAcc<float, kTm, kTk>;  // [kTm×kTk]
  using tileP      = TileLeft<float, kTm, kTk>;  // [kTm×kTk]
  using tileO      = TileAcc<float, kTm, vD>;     // [kTm×vD]
  using tileV      = TileRight<float, kTk, vD>;    // [kTk×vD]
  using tileMax    = Tile<Location::Vec, float, kTm, 8, BLayout::RowMajor, kTm, 1>; // [kTm×1]
  using tileSum    = Tile<Location::Vec, float, kTm, 8, BLayout::RowMajor, kTm, 1>; // [kTm×1]
  using tileScale  = Tile<Location::Vec, float, kTm, 8, BLayout::RowMajor, kTm, 1>; // [kTm×1]

  // 全局迭代器
  using itQ = global_iterator<gmQ, tileQ>;
  using itK = global_iterator<gmK, tileK>;
  using itV = global_iterator<gmV, tileV>;
  using itO = global_iterator<gmO, tileO>;

  itQ gQ(q_ptr);
  itK gK(k_ptr);
  itV gV(v_ptr);
  itO gO(out_ptr);

  const float scale = 1.0f / sqrt((float)qD);
  const int Qb = (S + kTm - 1) / kTm;
  const int Kb = (S + kTk - 1) / kTk;

  // 对每个 Q-block (i)
  for (int i = 0; i < Qb; ++i) {
    // 加载当前Q块 (仅一次)
    tileQ tQ;
    TLOAD(tQ, gQ(i, 0));

    // 初始化状态: 最大值/指数和/输出累加
    tileMax tMax;
    TEXPANDSCALAR(tMax, -1e30f); // 初始化为极小值
    tileSum tSum(0);              // 指数和归零
    tileO tO(0);                 // 输出累加归零

    // 单次遍历所有K/V块
    #pragma clang loop unroll(full)
    for (int j = 0; j < Kb; ++j) {
      // 加载K_j和V_j
      tileK tK; TLOAD(tK, gK(0, j));
      tileV tV; TLOAD(tV, gV(j, 0));

      // 计算注意力分数块
      tileS tS;
      tileP tP;
      MATMUL(tS, tQ, tK);
      TCVT(tP, tS);
      TMULS(tP, tP, scale);

      // 计算当前块的行最大值
      tileMax tLocalMax;
      TROWMAX(tLocalMax, tP);

      // 更新全局行最大值
      tileMax tNewMax;
      TMAX(tNewMax, tMax, tLocalMax);

      // 计算旧状态的缩放因子: exp(old_max - new_max)
      tileScale tScaleOld;
      TSUB(tScaleOld, tMax, tNewMax);
      TEXP(tScaleOld, tScaleOld);

      // 调整历史累加值: O = O * scale_old
      tileO tScaleOldExpanded;
      TEXPANDCOL(tScaleOldExpanded, tScaleOld);
      TMUL(tO, tO, tScaleOldExpanded);

      // 调整历史指数和: sum = sum * scale_old
      tileSum tScaledSum;
      TMUL(tScaledSum, tSum, tScaleOld);

      // 计算当前块的指数值 (使用新最大值)
      tileP tNewMaxExpanded;
      TEXPANDCOL(tNewMaxExpanded, tNewMax);
      TSUB(tP, tP, tNewMaxExpanded);
      TEXP(tP, tP);

      // 计算当前块的指数和
      tileSum tLocalSum;
      TROWSUM(tLocalSum, tP);

      // 更新全局指数和
      TADD(tSum, tScaledSum, tLocalSum);

      // 计算当前块的加权输出: O_j = P * V

      MATMACC(tO, tP, tV);
      TCVT(tO, tO); // FIXME: tmp add for loop unroll

      // 更新最大值状态
      tMax = tNewMax;
    }

    // 归一化输出: O = O / sum
    tileSum tInvSum;
    TRECIP(tInvSum, tSum);
    tileO tInvSumExpanded;
    TEXPANDCOL(tInvSumExpanded, tInvSum);
    TMUL(tO, tO, tInvSumExpanded);

    // 写回全局内存
    auto dstO = gO(i, 0);
    TSTORE(dstO, tO);
  }
}


template <int S, int qD, int vD, int kTm, int kTk>
void flash_attention_rm(float* out_ptr, float* q_ptr, float* k_ptr, float* v_ptr) {

    // 全局张量形状
    using gmQ = global_tensor<float, RowMajor<S, qD>>;  // Q: [S×qD]
    using gmK = global_tensor<float, RowMajor<qD, S>>;  // K: [qD×S]
    using gmV = global_tensor<float, RowMajor<S, vD>>;  // V: [S×vD]
    using gmO = global_tensor<float, RowMajor<S, vD>>;  // O: [S×vD]

    // tile 寄存器形状
    using tileQ      = Tile<Location::Vec, float, kTm, qD, BLayout::RowMajor>;    // [kTm×qD]
    using tileK      = Tile<Location::Vec, float, qD, kTk, BLayout::RowMajor>;    // [D×kTk]
    using tileW      = Tile<Location::Vec, float, kTm, kTk, BLayout::RowMajor>;   // [kTm×kTk]
    using tileO      = Tile<Location::Vec, float, kTm, vD, BLayout::RowMajor>;    // [kTm×vD]
    using tileV      = Tile<Location::Vec, float, kTk, vD, BLayout::RowMajor>;    // [kTk×vD]
    using tileMax    = Tile<Location::Vec, float, kTm, 8, BLayout::RowMajor, kTm, 1>;     // [kTm×1]
    using tileSum    = Tile<Location::Vec, float, kTm, 8, BLayout::RowMajor, kTm, 1>;     // [kTm×1]
    using tileScale  = Tile<Location::Vec, float, kTm, 8, BLayout::RowMajor, kTm, 1>;     // [kTm×1]

    // 全局迭代器
    using itQ = global_iterator<gmQ, tileQ>;
    using itK = global_iterator<gmK, tileK>;
    using itV = global_iterator<gmV, tileV>;
    using itO = global_iterator<gmO, tileO>;

    itQ gIterQ(q_ptr);
    itK gIterK(k_ptr);
    itV gIterV(v_ptr);
    itO gIterO(out_ptr);

    const float scale = 1.0f / sqrt((float)qD);
    const int Qb = (S + kTm - 1) / kTm;
    const int Kb = (S + kTk - 1) / kTk;
    // printf("Qb is %d;Kb is %d\n", Qb, Kb);

    // 对每个 Q-block (i)
    for (int i = 0; i < Qb; ++i) {
        // 加载当前Q块 (仅一次)
        tileQ tQ;
        auto gQ = gIterQ(i, 0);
        TLOAD(tQ, gQ);

        // 初始化状态: 最大值/指数和/输出累加
        tileMax tMax;
        TEXPANDSCALAR(tMax, -1e30f); // 初始化为极小值
        tileSum tSum(0);              // 指数和归零
        tileO tO(0);                 // 输出累加归零

        // 单次遍历所有K/V块
        for (int j = 0; j < Kb; ++j) {
        // 加载K_j和V_j
        auto gK = gIterK(0, j);
        auto gV = gIterV(j, 0);
        tileK tK; TLOAD(tK, gK);
        tileV tV; TLOAD(tV, gV);

        // 计算注意力分数块
        tileW tW;
        MATMUL(tW, tQ, tK);
        TMULS(tW, tW, scale);

        // 计算当前块的行最大值
        tileMax tLocalMax;
        TROWMAX(tLocalMax, tW);

        // 更新全局行最大值
        tileMax tNewMax;
        TMAX(tNewMax, tMax, tLocalMax);

        // 计算旧状态的缩放因子: exp(old_max - new_max)
        tileScale tScaleOld;
        TSUB(tScaleOld, tMax, tNewMax);
        TEXP(tScaleOld, tScaleOld);

        // 调整历史累加值: O = O * scale_old
        tileO tScaleOldExpanded;
        TEXPANDCOL(tScaleOldExpanded, tScaleOld);
        TMUL(tO, tO, tScaleOldExpanded);

        // 调整历史指数和: sum = sum * scale_old
        tileSum tScaledSum;
        TMUL(tScaledSum, tSum, tScaleOld);

        // 计算当前块的指数值 (使用新最大值)
        tileW tNewMaxExpanded;
        TEXPANDCOL(tNewMaxExpanded, tNewMax);
        TSUB(tW, tW, tNewMaxExpanded);
        TEXP(tW, tW);

        // 计算当前块的指数和
        tileSum tLocalSum;
        TROWSUM(tLocalSum, tW);

        // 更新全局指数和
        TADD(tSum, tScaledSum, tLocalSum);

        // 计算当前块的加权输出: O_j = W * V

        MATMACC(tO, tW, tV);

        // 更新最大值状态
        tMax = tNewMax;
        }

        // 归一化输出: O = O / sum
        tileSum tInvSum;
        TRECIP(tInvSum, tSum);
        tileO tInvSumExpanded;
        TEXPANDCOL(tInvSumExpanded, tInvSum);
        TMUL(tO, tO, tInvSumExpanded);

        // 写回全局内存
        auto dstO = gIterO(i, 0);
        TSTORE(dstO, tO);
    }
}

template <int S, int qD, int vD, int kTm, int kTk>
void flash_attention_opt2_unroll2_aligned(float* out_ptr, float* q_ptr, float* k_ptr, float* v_ptr) {
    // --- 1. 形状参数检查 ---
    // 计算 Key 的分块数量 (Compile-time constant)
    constexpr int Kb = (S + kTk - 1) / kTk;
    constexpr int Qb = (S + kTm - 1) / kTm;

    // 【新增】静态断言：强制要求 Kb 必须是偶数，否则编译报错
    // 这里的逻辑是：如果 Kb % 2 != 0，则触发错误信息
    static_assert(Kb % 2 == 0, "Total Key blocks (Kb) must be even for 2x unrolling optimization!");

    // --- 2. 类型定义 ---
    using gmQ = global_tensor<float, RowMajor<S, qD>>;
    using gmK = global_tensor<float, ColMajor<qD, S>>;
    using gmV = global_tensor<float, RowMajor<S, vD>>;
    using gmO = global_tensor<float, RowMajor<S, vD>>;

    using tileQ      = TileLeft<float, kTm, qD>;
    using tileK      = TileRight<float, qD, kTk>;
    using tileV      = TileRight<float, kTk, vD>;

    using tileW_out  = TileAcc<float, kTm, kTk>;
    using tileW      = Tile<Location::Vec, float, kTm, kTk, BLayout::ColMajor>;
    using tileW_left = TileLeft<float, kTm, kTk>;

    using tileO_out  = TileAcc<float, kTm, vD>;
    using tileO      = Tile<Location::Vec, float, kTm, vD, BLayout::RowMajor>;

    using tileMax    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>;
    using tileSum    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>;
    using tileScale  = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>;

    using itQ = global_iterator<gmQ, tileQ>;
    using itK = global_iterator<gmK, tileK>;
    using itV = global_iterator<gmV, tileV>;
    using itO = global_iterator<gmO, tileO>;

    itQ gIterQ(q_ptr);
    itK gIterK(k_ptr);
    itV gIterV(v_ptr);
    itO gIterO(out_ptr);

    const float scale = 1.0f / sqrt((float)qD);

    // --- 外层循环: 遍历 Q ---
    for (int i = 0; i < Qb; ++i) {
        tileQ tQ;
        auto gQ = gIterQ(i, 0);
        TLOAD(tQ, gQ);

        // 初始化状态
        tileMax tMax; TEXPANDSCALAR(tMax, -1e30f);
        tileSum tSum(0);
        tileO tO(0); // 最终累加器

        // --- 定义双缓冲寄存器 ---
        tileK tK_0, tK_1;
        tileV tV_0, tV_1;

        tileW_out tW_out_0, tW_out_1;
        tileW tW_0, tW_1;

        tileW tExpW_0, tExpW_1;
        tileW_left tW_left_0, tW_left_1;

        tileO_out tO_out_0, tO_out_1;
        tileO tO_tmp_0, tO_tmp_1;
        tileO tRescaleO_0, tRescaleO_1;

        // 状态更新的临时变量
        tileMax tNewMax_0, tNewMax_1;
        tileSum tNewSum_0, tNewSum_1;

        // --- 内层循环: 2倍步长 ---
        // 因为有 static_assert 保证 Kb 是偶数，所以 j < Kb 是安全的，不需要处理余数
        #pragma clang loop unroll(full)
        for (int j = 0; j < Kb; j += 2) {
            // 1. Load K, V (Double Buffering)
            auto gK_0 = gIterK(0, j);
            auto gV_0 = gIterV(j, 0);
            auto gK_1 = gIterK(0, j + 1);
            auto gV_1 = gIterV(j + 1, 0);

            TLOAD(tK_0, gK_0);
            TLOAD(tV_0, gV_0);
            TLOAD(tK_1, gK_1);
            TLOAD(tV_1, gV_1);

            // 2. MatMul QK Grouping
            // Doubled buffer
            MATMUL(tW_out_0, tQ, tK_0);
            MATMUL(tW_out_1, tQ, tK_1);

            // 3. Convert
            TCVT(tW_0, tW_out_0);
            TCVT(tW_1, tW_out_1);

            // --- Pipeline Stage 0 ---
            flashsoftmax_kernel<tileW, tileO, tileMax, tileSum, tileScale>(
                tRescaleO_0, tNewMax_0, tNewSum_0, tExpW_0, tW_0, tO, tMax, tSum, scale
            );

            TMUL(tO, tO, tRescaleO_0); // Rescale O_old

            TCVT(tW_left_0, tExpW_0);
            MATMUL(tO_out_0, tW_left_0, tV_0); // Compute P0 * V0
            TCVT(tO_tmp_0, tO_out_0);

            TADD(tO, tO, tO_tmp_0); // Accumulate O_new

            // Update State 0 -> 1
            tMax = tNewMax_0;
            tSum = tNewSum_0;


            // --- Pipeline Stage 1 ---
            flashsoftmax_kernel<tileW, tileO, tileMax, tileSum, tileScale>(
                tRescaleO_1, tNewMax_1, tNewSum_1, tExpW_1, tW_1, tO, tMax, tSum, scale
            );

            TMUL(tO, tO, tRescaleO_1); // Rescale O_old (updated by stage 0)

            TCVT(tW_left_1, tExpW_1);
            MATMUL(tO_out_1, tW_left_1, tV_1); // Compute P1 * V1
            TCVT(tO_tmp_1, tO_out_1);

            TADD(tO, tO, tO_tmp_1); // Accumulate O_new

            // Update State 1 -> Next Loop
            tMax = tNewMax_1;
            tSum = tNewSum_1;
        }
        // 这里的 Remainder Loop 已经完全移除

        // --- 最终归一化 ---
        tileSum tInvSum;
        TRECIP(tInvSum, tSum);
        tileO tInvSumExpanded;
        TEXPANDCOL(tInvSumExpanded, tInvSum);
        TMUL(tO, tO, tInvSumExpanded);

        // 写回全局内存
        Tile<Location::Vec, float, kTm, vD, BLayout::RowMajor> tO_cast;
        TCAST(tO_cast, tO);
        auto dstO = gIterO(i, 0);
        TSTORE(dstO, tO_cast);
    }
}

template <typename dtype, int qD, int vD, int kTm, int kTk>
__attribute__((noinline))
void flash_attention_dynamic(dtype* out_ptr, dtype* q_ptr, dtype* k_ptr, dtype* v_ptr, int Sq, int Skv) {

    using gmQ = global_tensor<float, RowMajor<-1, qD>>;  // Q: [S×qD]
    using gmK = global_tensor<float, ColMajor<qD, -1>>;  // K: [qD×S]
    using gmV = global_tensor<float, RowMajor<-1, vD>>;  // V: [S×vD]
    using gmO = global_tensor<float, RowMajor<-1, vD>>;  // O: [SxvD]

    using tileQ      = TileLeft<dtype, kTm, qD, -1, qD>;         // [kTm×qD]
    using tileK      = TileRight<dtype, qD, kTk, qD, -1>;        // [vD×kTk]
    using tileW_out  = TileAcc<float, kTm, kTk, -1, -1>;         // [kTm×kTk]
    using tileW      = Tile<Location::Vec, float, kTm, kTk, BLayout::RowMajor, -1, -1>;
    using tileW_left = TileLeft<dtype, kTm, kTk, -1, -1>;

    using tileO_out  = TileAcc<float, kTm, vD, -1, vD>;
    using tileO      = Tile<Location::Vec, float, kTm, vD, BLayout::RowMajor, -1, vD>; // [kTm×vD]
    using tileO_cast = Tile<Location::Vec, dtype, kTm, vD, BLayout::RowMajor, -1, vD>;

    using tileV      = TileRight<dtype, kTk, vD, -1, vD>;     // [kTk×vD]
    using tileMax    = Tile<Location::Vec, float, kTm, 8, BLayout::RowMajor, -1, 1>; // [kTm×1]
    using tileSum    = Tile<Location::Vec, float, kTm, 8, BLayout::RowMajor, -1, 1>; // [kTm×1]
    using tileScale  = Tile<Location::Vec, float, kTm, 8, BLayout::RowMajor, -1, 1>; // [kTm×1]

    const float scale = 1.0f / sqrt((float)qD);
    const int Qb = (Sq + kTm - 1) / kTm;
    const int Kb = (Skv + kTk - 1) / kTk;
    const int rQ = Sq % kTm;
    const int rK = Skv % kTk;

    for (int i = 0; i < Qb; ++i) {

        int dyn_m = (i+1) * kTm > Sq ? rQ:kTm;

        size_t offset_Q = i * tileQ::Rows * qD;
        gmQ gQ(q_ptr+offset_Q, Sq);

        tileQ tQ(dyn_m);
        TLOAD(tQ, gQ);

        tileMax tMax(-1e30f, dyn_m);
        tileSum tSum(0, dyn_m);
        tileO_out tO_out(dyn_m);
        tileO tO(0, dyn_m), tRescaleO(0, dyn_m);

        #pragma clang loop unroll(full)
        for (int j = 0; j < Kb; ++j) {

            int dyn_k = (j+1) * kTk > Skv ? rK:kTk;

            size_t offset_K = j * tileK::Cols * qD;
            size_t offset_V = j * tileV::Rows * vD;
            gmK gK(k_ptr+offset_K, Skv);
            gmV gV(v_ptr+offset_V, Skv);

            tileK tK(dyn_k); TLOAD(tK, gK);
            tileV tV(dyn_k); TLOAD(tV, gV);

            tileW_out tW_out(dyn_m, dyn_k);
            MATMUL(tW_out, tQ, tK);

            // Nz -> RowMajor
            tileW tW(dyn_m, dyn_k);
            TCVT(tW, tW_out);
            TMULS(tW, tW, scale);

            // 计算当前块的行最大值
            tileMax tLocalMax(dyn_m);
            TROWMAX(tLocalMax, tW);

            // 更新全局行最大值
            tileMax tNewMax(dyn_m);
            TMAX(tNewMax, tMax, tLocalMax);

            // 计算旧状态的缩放因子: exp(old_max - new_max)
            tileScale tScaleOld(dyn_m);
            TSUB(tScaleOld, tMax, tNewMax);
            TEXP(tScaleOld, tScaleOld);

            // 调整历史指数和: sum = sum * scale_old
            tileSum tScaledSum(dyn_m);
            TMUL(tScaledSum, tSum, tScaleOld);

            // 计算当前块的指数值 (使用新最大值)
            tileW tNewMaxExpanded(dyn_m, dyn_k);
            TEXPANDCOL(tNewMaxExpanded, tNewMax);
            TSUB(tW, tW, tNewMaxExpanded);
            TEXP(tW, tW);

            // 计算当前块的指数和
            tileSum tLocalSum(dyn_m);
            TROWSUM(tLocalSum, tW);

            // 更新全局指数和
            TADD(tSum, tScaledSum, tLocalSum);

            // 调整历史累加值: O = O * scale_old
            tileO tScaleOldExpanded(dyn_m);
            TEXPANDCOL(tScaleOldExpanded, tScaleOld);

            TADD(tO, tO, tRescaleO);
            TMUL(tRescaleO, tO, tScaleOldExpanded);

            // RowMajor -> Nz
            tileW_left tW_left(dyn_m, dyn_k);
            TCVT(tW_left, tW);
            // 计算当前块的加权输出: O_j = W * V
            MATMUL(tO_out, tW_left, tV);
            TCVT(tO, tO_out);
            // 更新最大值状态
            tMax = tNewMax;
        }

        TADD(tO, tO, tRescaleO);
        // 归一化输出: O = O / sum
        tileSum tInvSum(dyn_m);
        TRECIP(tInvSum, tSum);
        tileO tInvSumExpanded(dyn_m);
        TEXPANDCOL(tInvSumExpanded, tInvSum);
        TMUL(tO, tO, tInvSumExpanded);

        tileO_cast tO_cast(dyn_m);
        TCAST(tO_cast, tO);

        size_t offset_O = i * tileO_cast::Rows * vD;
        gmO dstO(out_ptr+offset_O, Sq);
        TSTORE(dstO, tO_cast);
    }
}

#endif

