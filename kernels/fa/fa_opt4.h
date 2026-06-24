template<typename tileSrc, typename tileMax, typename tileSum, typename tileScale>
void __vec__ flashsoftmax_opt4_with_scale(
                        typename tileScale::TileDType __out__ rescale,
                        typename tileMax::TileDType __out__ new_max,
                        typename tileSum::TileDType __out__ new_sum,
                        typename tileSrc::TileDType __out__ src_exp,
                        const typename tileSrc::TileDType __in__ src,
                        const typename tileMax::TileDType __in__ old_max,
                        const typename tileSum::TileDType __in__ old_sum,
                        const typename tileSrc::DType src_scale
                                ){
    size_t i = blkv_get_index_x();

    __vbuf__ typename tileMax::DType *new_max_ptr = blkv_get_tile_ptr(new_max);
    __vbuf__ typename tileSum::DType *new_sum_ptr = blkv_get_tile_ptr(new_sum);
    __vbuf__ typename tileSrc::DType *src_exp_ptr = blkv_get_tile_ptr(src_exp);
    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileMax::DType *old_max_ptr = blkv_get_tile_ptr(old_max);
    __vbuf__ typename tileSum::DType *old_sum_ptr = blkv_get_tile_ptr(old_sum);
    __vbuf__ typename tileScale::DType *rescale_ptr = blkv_get_tile_ptr(rescale);

    size_t max_idx = i*tileMax::RowStride;
    size_t sum_idx = i*tileSum::RowStride;
    typename tileMax::DType old_max_val = old_max_ptr[max_idx];
    typename tileSum::DType old_sum_val = old_sum_ptr[sum_idx];
    typename tileSum::DType upd_sum = old_sum_val;
    typename tileMax::DType upd_max = old_max_val;
    
    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileSrc::ValidCol;j+=4){
        size_t src_idx_0 =  i * tileSrc::RowStride + j * tileSrc::ColStride;
        size_t src_idx_1 =  i * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        size_t src_idx_2 =  i * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        size_t src_idx_3 =  i * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
        typename tileSrc::DType src_0 = src_ptr[src_idx_0] * src_scale;
        typename tileSrc::DType src_1 = src_ptr[src_idx_1] * src_scale;
        typename tileSrc::DType src_2 = src_ptr[src_idx_2] * src_scale;
        typename tileSrc::DType src_3 = src_ptr[src_idx_3] * src_scale;
        typename tileMax::DType max_01 = blkv_max(src_0, src_1);
        typename tileMax::DType max_23 = blkv_max(src_2, src_3);
        typename tileMax::DType max_0123 = blkv_max(max_01, max_23);
        upd_max = blkv_max(upd_max, max_0123);

        typename tileSrc::DType exp_src_0 = blkv_fexp(src_0 - old_max_val);
        typename tileSrc::DType exp_src_1 = blkv_fexp(src_1 - old_max_val);
        typename tileSrc::DType exp_src_2 = blkv_fexp(src_2 - old_max_val);
        typename tileSrc::DType exp_src_3 = blkv_fexp(src_3 - old_max_val);
        typename tileSrc::DType exp_src_01 = exp_src_0 + exp_src_1;
        typename tileSrc::DType exp_src_23 = exp_src_2 + exp_src_3;
        typename tileSrc::DType exp_src_0123 = exp_src_01 + exp_src_23;
        upd_sum += exp_src_0123;
        src_exp_ptr[src_idx_0] = exp_src_0;
        src_exp_ptr[src_idx_1] = exp_src_1;
        src_exp_ptr[src_idx_2] = exp_src_2;
        src_exp_ptr[src_idx_3] = exp_src_3;
    }
    new_max_ptr[max_idx] = upd_max;

    typename tileScale::DType scale = blkv_fexp(old_max_ptr[max_idx] - upd_max);
    new_sum_ptr[sum_idx] = upd_sum * scale;

    size_t scale_idx = i*tileScale::RowStride;
    rescale_ptr[scale_idx] = scale;
}

template<typename tileSrc, typename tileMax, typename tileSum, typename tileScale>
void __vec__ flashsoftmax_opt4(
                        typename tileScale::TileDType __out__ rescale,
                        typename tileMax::TileDType __out__ new_max,
                        typename tileSum::TileDType __out__ new_sum,
                        typename tileSrc::TileDType __out__ src_exp,
                        const typename tileSrc::TileDType __in__ src,
                        const typename tileMax::TileDType __in__ old_max,
                        const typename tileSum::TileDType __in__ old_sum
                                ){
    size_t i = blkv_get_index_x();

    __vbuf__ typename tileMax::DType *new_max_ptr = blkv_get_tile_ptr(new_max);
    __vbuf__ typename tileSum::DType *new_sum_ptr = blkv_get_tile_ptr(new_sum);
    __vbuf__ typename tileSrc::DType *src_exp_ptr = blkv_get_tile_ptr(src_exp);
    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileMax::DType *old_max_ptr = blkv_get_tile_ptr(old_max);
    __vbuf__ typename tileSum::DType *old_sum_ptr = blkv_get_tile_ptr(old_sum);
    __vbuf__ typename tileScale::DType *rescale_ptr = blkv_get_tile_ptr(rescale);

    size_t max_idx = i*tileMax::RowStride;
    size_t sum_idx = i*tileSum::RowStride;
    typename tileMax::DType old_max_val = old_max_ptr[max_idx];
    typename tileSum::DType old_sum_val = old_sum_ptr[sum_idx];
    typename tileSum::DType upd_sum = old_sum_val;
    typename tileMax::DType upd_max = old_max_val;
    
    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileSrc::ValidCol;j+=4){
        size_t src_idx_0 =  i * tileSrc::RowStride + j * tileSrc::ColStride;
        size_t src_idx_1 =  i * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        size_t src_idx_2 =  i * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        size_t src_idx_3 =  i * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
        typename tileSrc::DType src_0 = src_ptr[src_idx_0];
        typename tileSrc::DType src_1 = src_ptr[src_idx_1];
        typename tileSrc::DType src_2 = src_ptr[src_idx_2];
        typename tileSrc::DType src_3 = src_ptr[src_idx_3];
        typename tileMax::DType max_01 = blkv_max(src_0, src_1);
        typename tileMax::DType max_23 = blkv_max(src_2, src_3);
        typename tileMax::DType max_0123 = blkv_max(max_01, max_23);
        upd_max = blkv_max(upd_max, max_0123);

        typename tileSrc::DType exp_src_0 = blkv_fexp(src_0 - old_max_val);
        typename tileSrc::DType exp_src_1 = blkv_fexp(src_1 - old_max_val);
        typename tileSrc::DType exp_src_2 = blkv_fexp(src_2 - old_max_val);
        typename tileSrc::DType exp_src_3 = blkv_fexp(src_3 - old_max_val);
        typename tileSrc::DType exp_src_01 = exp_src_0 + exp_src_1;
        typename tileSrc::DType exp_src_23 = exp_src_2 + exp_src_3;
        typename tileSrc::DType exp_src_0123 = exp_src_01 + exp_src_23;
        upd_sum += exp_src_0123;
        src_exp_ptr[src_idx_0] = exp_src_0;
        src_exp_ptr[src_idx_1] = exp_src_1;
        src_exp_ptr[src_idx_2] = exp_src_2;
        src_exp_ptr[src_idx_3] = exp_src_3;
    }
    new_max_ptr[max_idx] = upd_max;

    typename tileScale::DType scale = blkv_fexp(old_max_ptr[max_idx] - upd_max);
    new_sum_ptr[sum_idx] = upd_sum * scale;

    size_t scale_idx = i*tileScale::RowStride;
    rescale_ptr[scale_idx] = scale;
}

template<typename tileO, typename tileScale>
void __vec__ update_output_opt4(
                        typename tileO::TileDType __out__ rescale_out,
                        const typename tileO::TileDType __in__ old_out,
                        const typename tileO::TileDType __in__ pv,
                        const typename tileScale::TileDType __in__ rescale
                    ){
    size_t j = blkv_get_index_x();
    size_t i = blkv_get_index_y();

    size_t idx = i * tileO::ColStride + j;
    size_t idx_scale = j * tileScale::RowStride;

    blkv_get_tile_ptr(rescale_out)[idx] = (blkv_get_tile_ptr(old_out)[idx] + blkv_get_tile_ptr(pv)[idx]) * blkv_get_tile_ptr(rescale)[idx_scale] ;
}

template<typename tileO_cast, typename tileO, typename tileSum>
void __vec__ normalize_opt4(
                        typename tileO_cast::TileDType __out__ out_cast,
                        const typename tileO::TileDType __in__ out,
                        const typename tileSum::TileDType __in__ sum
                    ){
    size_t j = blkv_get_index_x();
    size_t i = blkv_get_index_y();

    size_t idx = i * tileO::ColStride + j;
    size_t idx_sum = j * tileSum::RowStride;

    blkv_get_tile_ptr(out_cast)[idx] = static_cast<typename tileO_cast::DType>( blkv_get_tile_ptr(out)[idx] /  blkv_get_tile_ptr(sum)[idx_sum] );
}

template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk>
void flash_attention_opt4(dtype* out_ptr, dtype* q_ptr, dtype* k_ptr, dtype* v_ptr) {
    // 全局张量形状
    using gmQ = global_tensor<dtype, RowMajor<Sq, qD>>;  // Q: [S×qD]
    using gmK = global_tensor<dtype, ColMajor<qD, Skv>>;  // K: [qD×S]
    using gmV = global_tensor<dtype, RowMajor<Skv, vD>>;  // V: [S×vD]
    using gmO = global_tensor<dtype, ColMajor<Sq, vD>>;  // O: [SxvD]
    // tile 寄存器形状
    using tileQ      = TileLeft<dtype, kTm, (qD==192? 256:qD), kTm, qD>;       // [kTm×qD]
    using tileK      = TileRight<dtype, (qD==192? 256:qD), kTk, qD, kTk>;      // [vD×kTk]
    using tileW_out  = TileAcc<float, kTm, kTk>;      // [kTm×kTk]
    using tileW      = Tile<Location::Vec, float, kTm, kTk, BLayout::ColMajor>;
    using tileW_left = TileLeft<dtype, kTm, kTk>; 

    using tileO_out  = TileAcc<float, kTm, vD>;
    using tileO      = Tile<Location::Vec, float, kTm, vD, BLayout::ColMajor>; // [kTm×vD]
    using tileO_cast = Tile<Location::Vec, dtype, kTm, vD, BLayout::ColMajor>;

    using tileV      = TileRight<dtype, kTk, vD>;     // [kTk×vD]
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
    const int Qb = (Sq + kTm - 1) / kTm;
    const int Kb = (Skv + kTk - 1) / kTk;
    
    // 对每个 Q-block (i)
    for (int i = 0; i < Qb; ++i) {
        // 加载当前Q块 (仅一次)
        tileQ tQ;
        auto gQ = gIterQ(i,0);
        TCOPYIN(tQ, gQ);

        // 初始化状态: 最大值/指数和/输出累加
        tileMax tMax;
        TEXPANDSCALAR(tMax, -1e30f);  // 初始化为极小值
        tileSum tSum(0);              // 指数和归零
        tileO_out tPV_out;            
        tileO tPV;
        tileO tO(0);                  // 输出累加归零

        // 单次遍历所有K/V块
        #pragma clang loop unroll(full)
        for (int j = 0; j < Kb; ++j) {
        // 加载K_j和V_j
        auto gK = gIterK(0, j);
        auto gV = gIterV(j, 0);
        tileK tK; TCOPYIN(tK, gK);
        tileV tV; TCOPYIN(tV, gV);

        // 计算注意力分数块
        tileW_out tW_out;
        MATMUL(tW_out, tQ, tK);

        // Nz -> ColMajor
        tileW tW;
        #ifdef TEMPLATE 
        ACCSCALE_NZ2DN(tW, tW_out, scale);

        tileMax tNewMax;
        tileSum tNewSum;
        tileW tExpW;
        tileScale tScale;
        flashsoftmax_opt4<tileW, tileMax, tileSum, tileScale><<<tileW::ValidRow, 1, 1>>>(tScale.data(), tNewMax.data(), tNewSum.data(), tExpW.data(), tW.data(), tMax.data(), tSum.data());
        #else
        TCVT(tW, tW_out);

        tileMax tNewMax;
        tileSum tNewSum;
        tileW tExpW;
        tileScale tScale;
        flashsoftmax_opt4_with_scale<tileW, tileMax, tileSum, tileScale><<<tileW::ValidRow, 1, 1>>>(tScale.data(), tNewMax.data(), tNewSum.data(), tExpW.data(), tW.data(), tMax.data(), tSum.data(), scale);
        #endif
        // ColMajor -> Nz
        tileW_left tW_left;
        #ifdef TEMPLATE
        TCVT_DN2NZ(tW_left, tExpW);
        #else
        TCVT(tW_left, tExpW);
        #endif
        // 计算当前块的加权输出: O_j = W * V
        MATMUL(tPV_out, tW_left, tV);
        TCVT(tPV, tPV_out);
        // 更新最大值状态

        update_output_opt4<tileO, tileScale><<<tileO::ValidRow, tileO::ValidCol, 1>>>(tO.data(), tO.data(), tPV.data(), tScale.data());
        tMax = tNewMax;
        tSum = tNewSum;
        }

        tileO_cast tO_cast;
        // // 归一化输出: O = O / sum
        // tileSum tInvSum;
        // TRECIP(tInvSum, tSum);
        // tileO tInvSumExpanded;
        // TEXPANDCOL(tInvSumExpanded, tInvSum);
        // TMUL(tO, tO, tInvSumExpanded);
        // TCAST(tO_cast, tO);
        normalize_opt4<tileO_cast, tileO, tileSum><<<tileO::ValidRow, tileO::ValidCol, 1>>>(tO_cast.data(), tO.data(), tSum.data());
        auto dstO = gIterO(i, 0);
        TCOPYOUT(dstO, tO_cast);
    }
}