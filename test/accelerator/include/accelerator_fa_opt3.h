template<typename tileSrc, typename tileOut, typename tileMax, typename tileSum, typename tileScale>
void __vec__ flashsoftmax_dn_mout_no_scale_kernel(
                        typename tileOut::TileDType __out__ rescale_out,
                        typename tileMax::TileDType __out__ new_max,
                        typename tileSum::TileDType __out__ new_sum,
                        typename tileSrc::TileDType __out__ src_exp,
                        const typename tileSrc::TileDType __in__ src,
                        const typename tileOut::TileDType __in__ old_out,
                        const typename tileMax::TileDType __in__ old_max,
                        const typename tileSum::TileDType __in__ old_sum,
                        const typename tileOut::TileDType __in__ rescale_old_out
                                ){
    size_t i = blkv_get_index_x();

    __vbuf__ typename tileOut::DType *rescale_out_ptr = blkv_get_tile_ptr(rescale_out);
    __vbuf__ typename tileMax::DType *new_max_ptr = blkv_get_tile_ptr(new_max);
    __vbuf__ typename tileSum::DType *new_sum_ptr = blkv_get_tile_ptr(new_sum);
    __vbuf__ typename tileSrc::DType *src_exp_ptr = blkv_get_tile_ptr(src_exp);
    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileMax::DType *old_max_ptr = blkv_get_tile_ptr(old_max);
    __vbuf__ typename tileSum::DType *old_sum_ptr = blkv_get_tile_ptr(old_sum);
    __vbuf__ typename tileOut::DType *rescale_old_out_ptr = blkv_get_tile_ptr(rescale_old_out);
    __vbuf__ typename tileOut::DType *old_out_ptr = blkv_get_tile_ptr(old_out);

    size_t max_idx = i*tileMax::RowStride;

    typename tileMax::DType upd_max = old_max_ptr[max_idx];

    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileSrc::ValidCol;j+=4){
        size_t src_idx_0 =  i * tileSrc::RowStride + j * tileSrc::ColStride;
        size_t src_idx_1 =  i * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        size_t src_idx_2 =  i * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        size_t src_idx_3 =  i * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
        typename tileMax::DType max_01 = blkv_max(src_ptr[src_idx_0], src_ptr[src_idx_1]);
        typename tileMax::DType max_23 = blkv_max(src_ptr[src_idx_2], src_ptr[src_idx_3]);
        typename tileMax::DType max_0123 = blkv_max(max_01, max_23);
        upd_max = blkv_max(upd_max, max_0123);
    }
    new_max_ptr[max_idx] = upd_max;

    typename tileMax::DType scale = blkv_fexp(old_max_ptr[max_idx] - upd_max);

    size_t sum_idx = i*tileSum::RowStride;
    typename tileSum::DType upd_sum = old_sum_ptr[sum_idx] * scale;

    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileSrc::ValidCol;j+=4){
        size_t src_idx_0 =  i * tileSrc::RowStride + j * tileSrc::ColStride;
        size_t src_idx_1 =  i * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        size_t src_idx_2 =  i * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        size_t src_idx_3 =  i * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
        typename tileSrc::DType exp_src_0 = blkv_fexp(src_ptr[src_idx_0] - upd_max);
        typename tileSrc::DType exp_src_1 = blkv_fexp(src_ptr[src_idx_1] - upd_max);
        typename tileSrc::DType exp_src_2 = blkv_fexp(src_ptr[src_idx_2] - upd_max);
        typename tileSrc::DType exp_src_3 = blkv_fexp(src_ptr[src_idx_3] - upd_max);
        typename tileSrc::DType exp_src_01 = exp_src_0 + exp_src_1;
        typename tileSrc::DType exp_src_23 = exp_src_2 + exp_src_3;
        typename tileSrc::DType exp_src_0123 = exp_src_01 + exp_src_23;
        upd_sum += exp_src_0123;
        src_exp_ptr[src_idx_0] = exp_src_0;
        src_exp_ptr[src_idx_1] = exp_src_1;
        src_exp_ptr[src_idx_2] = exp_src_2;
        src_exp_ptr[src_idx_3] = exp_src_3;
    }
    new_sum_ptr[sum_idx] = upd_sum;

    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileOut::ValidCol;j++){
        size_t out_idx = i * tileOut::RowStride + j * tileOut::ColStride;
        rescale_out_ptr[out_idx] = (rescale_old_out_ptr[out_idx] + old_out_ptr[out_idx]) * scale;
    }
}

template <typename tileSrc, typename tileOut, typename tileMax, typename tileSum, typename tileScale>
void flashsoftmax_dn_mout_no_scale(
                        tileOut &rescale_out,
                        tileMax &new_max,
                        tileSum &new_sum,
                        tileSrc &src_exp,
                        tileSrc &src,
                        tileOut &old_out,
                        tileMax &old_max,
                        tileSum &old_sum
                        ){
    flashsoftmax_dn_mout_no_scale_kernel<tileSrc, tileOut, tileMax, tileSum, tileScale><<<tileSrc::ValidRow, 1, 1>>>(rescale_out.data(), new_max.data(), new_sum.data(), src_exp.data(), src.data(), old_out.data(), old_max.data(), old_sum.data(), rescale_out.data());
}

template <is_multi_tile tileSrc, is_multi_tile tileOut, is_multi_tile tileMax, is_multi_tile tileSum, is_multi_tile tileScale>
void flashsoftmax_dn_mout_multi_no_scale(
                        tileOut &rescale_out,
                        tileMax &new_max,
                        tileSum &new_sum,
                        tileSrc &src_exp,
                        tileSrc &src,
                        tileOut &old_out,
                        tileMax &old_max,
                        tileSum &old_sum
                        ){
    #pragma clang loop unroll(full)
    for (int i = 0; i < tileOut::NumTiles; ++i) {
    flashsoftmax_dn_mout_no_scale_kernel<typename tileSrc::TileType, typename tileOut::TileType, typename tileMax::TileType, typename tileSum::TileType, typename tileScale::TileType>
        <<<tileSrc::TileType::ValidRow, 1, 1>>>(rescale_out.Tiles[i].data(),
                                                new_max.Tiles[i].data(),
                                                new_sum.Tiles[i].data(),
                                                src_exp.Tiles[i].data(),
                                                src.Tiles[i].data(),
                                                old_out.Tiles[i].data(),
                                                old_max.Tiles[i].data(),
                                                old_sum.Tiles[i].data(),
                                                rescale_out.Tiles[i].data()
                                                );
    }
}

template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk>
void flash_attention_opt3(dtype* out_ptr, dtype* q_ptr, dtype* k_ptr, dtype* v_ptr) {
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
        flashsoftmax_dn_mout_no_scale<tileW, tileO, tileMax, tileSum, tileScale>(tRescaleO, tNewMax, tNewSum, tExpW, tW, tO, tMax, tSum);

        #else
        TCVT(tW, tW_out);

        tileMax tNewMax;
        tileSum tNewSum;
        tileW tExpW;
        flashsoftmax_dn_mout<tileW, tileO, tileMax, tileSum, tileScale>(tRescaleO, tNewMax, tNewSum, tExpW, tW, tO, tMax, tSum, scale);
        #endif

        // ColMajor -> Nz
        tileW_left tW_left;
        #ifdef TEMPLATE
        TCVT_DN2NZ(tW_left, tExpW);
        #else
        TCVT(tW_left, tExpW);
        #endif

        // 计算当前块的加权输出: O_j = W * V
        MATMUL(tO_out, tW_left, tV);
        TCVT(tO, tO_out);
        // 更新最大值状态
        tMax = tNewMax;
        tSum = tNewSum;
        }

        tileO_cast tO_cast;
        // TADD(tO, tO, tRescaleO);
        // // 归一化输出: O = O / sum
        // tileSum tInvSum;
        // TRECIP(tInvSum, tSum);
        // tileO tInvSumExpanded;
        // TEXPANDCOL(tInvSumExpanded, tInvSum);
        // TMUL(tO, tO, tInvSumExpanded);
        // TCAST(tO_cast, tO);
        normalize<tileO_cast, tileO, tileSum><<<tileO::ValidRow, tileO::ValidCol, 1>>>(tO_cast.data(), tO.data(), tRescaleO.data(), tSum.data());
        // 写回全局内存
        auto dstO = gIterO(i, 0);
        TCOPYOUT(dstO, tO_cast);
    }
}

template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk, int MULTI>
void flash_attention_multitile_opt3(dtype* out_ptr, dtype* q_ptr, dtype* k_ptr, dtype* v_ptr) {
    // 全局张量形状
    using gmQ = global_tensor<dtype, RowMajor<Sq, qD>>;  // Q: [S×qD]
    using gmK = global_tensor<dtype, ColMajor<qD, Skv>>;  // K: [qD×S]
    using gmV = global_tensor<dtype, RowMajor<Skv, vD>>;  // V: [S×vD]
    using gmO = global_tensor<dtype, ColMajor<Sq, vD>>;  // O: [SxvD]
    // tile 寄存器形状
    using tileQ      = MultiTile<MULTI, TileLeft<dtype, kTm, (qD==192? 256:qD), kTm, qD>>;       // [kTm×qD]
    using tileK      = MultiTile<MULTI, TileRight<dtype, (qD==192? 256:qD), kTk, qD, kTk>>;      // [vD×kTk]
    // using tileW_out  = TileAcc<float, kTm, kTk>;      // [kTm×kTk]
    using tileW      = MultiTile<MULTI, Tile<Location::Vec, float, kTm, kTk, BLayout::ColMajor>>; 
    using tileW_left = MultiTile<MULTI, TileLeft<dtype, kTm, kTk>>;

    // using tileO_out  = TileAcc<float, kTm, vD>;
    using tileO      = MultiTile<MULTI, Tile<Location::Vec, float, kTm, vD, BLayout::ColMajor>>; // [kTm×vD]
    using tileO_cast = MultiTile<MULTI, Tile<Location::Vec, dtype, kTm, vD, BLayout::ColMajor>>;

    using tileV      = MultiTile<MULTI, TileRight<dtype, kTk, vD>>;     // [kTk×vD]
    using tileMax    = MultiTile<MULTI, Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>>; // [kTm×1]
    using tileSum    = MultiTile<MULTI, Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>>; // [kTm×1]
    using tileScale  = MultiTile<MULTI, Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>>; // [kTm×1]

    // 全局迭代器
    using itQ = global_iterator<gmQ, typename tileQ::TileType>;
    using itK = global_iterator<gmK, typename tileK::TileType>;
    using itV = global_iterator<gmV, typename tileV::TileType>;
    using itO = global_iterator<gmO, typename tileO::TileType>;

    itQ gIterQ(q_ptr);
    itK gIterK(k_ptr);
    itV gIterV(v_ptr);
    itO gIterO(out_ptr);

    const float scale = 1.0f / sqrt((float)qD);
    const int Qb = (Sq + kTm - 1) / kTm;
    const int Kb = (Skv + kTk - 1) / kTk;
    
    // 对每个 Q-block (i)
    for (int i = 0; i < Qb; i += MULTI) {
        // 加载当前Q块 (仅一次)
        tileQ tQ;

        #ifdef MULTI_LDST
        auto gQ = gIterQ(i,0);
        TLOAD2_ND2NZ(tQ.Tiles[1], tQ.Tiles[0], gQ);
        #else
        TCOPYIN(tQ, [&](int t) { return gIterQ(i + t, 0); });
        #endif

        // 初始化状态: 最大值/指数和/输出累加
        tileMax tMax;
        TEXPANDSCALAR(tMax, -1e30f);
        tileSum tSum;              // 指数和归零
        TEXPANDSCALAR(tSum, 0);
        tileO tO, tRescaleO;
        // TEXPANDSCALAR(tO, 0);
        // TEXPANDSCALAR(tRescaleO, 0);

        // 单次遍历所有K/V块
        #pragma clang loop unroll(full)
        for (int j = 0; j < Kb; ++j) {
        // 加载K_j和V_j
        auto gK = gIterK(0, j);
        tileK tK;
        TCOPYIN(tK, gK);

        // 计算注意力分数块
        tileW tW;
        #ifdef MMSCALE
        MATMUL_SCALE(tW, tQ, tK, scale);
        tileMax tNewMax;
        tileSum tNewSum;
        tileW tExpW;
        if (j == 0) {
            TEXPANDSCALAR(tRescaleO, 0);
            TEXPANDSCALAR(tO, 0);
        }
        flashsoftmax_dn_mout_multi_no_scale<tileW, tileO, tileMax, tileSum, tileScale>(tRescaleO, tNewMax, tNewSum, tExpW, tW, tO, tMax, tSum);
        #else
        MATMUL(tW, tQ, tK);
        tileMax tNewMax;
        tileSum tNewSum;
        tileW tExpW;
        if (j == 0) {
            flashsoftmax_dn_mout_norescale_multi<tileW, tileMax, tileSum, tileScale>(tNewMax, tNewSum, tExpW, tW, tMax, tSum, scale);
            TEXPANDSCALAR(tRescaleO, 0);
        } else
            flashsoftmax_dn_mout_multi<tileW, tileO, tileMax, tileSum, tileScale>(tRescaleO, tNewMax, tNewSum, tExpW, tW, tO, tMax, tSum, scale);
        #endif

        // ColMajor -> Nz
        tileW_left tW_left;
        #ifdef TEMPLATE
        TCVT_DN2NZ(tW_left, tExpW);
        #else
        TCVT(tW_left, tExpW);
        #endif
        // 计算当前块的加权输出: O_j = W * V
        auto gV = gIterV(j, 0);
        tileV tV;
        TCOPYIN(tV, gV);
        MATMUL(tO, tW_left, tV);
        // 更新最大值状态
        tMax = tNewMax;
        tSum = tNewSum;
        }

        tileO_cast tO_cast;
        // TADD(tO, tO, tRescaleO);
        // // 归一化输出: O = O / sum
        // tileSum tInvSum;
        // TRECIP(tInvSum, tSum);
        // tileO tInvSumExpanded;
        // TEXPANDCOL(tInvSumExpanded, tInvSum);
        // TMUL(tO, tO, tInvSumExpanded);
        // TCAST(tO_cast, tO);

        #pragma clang loop unroll(full)
        for (int i = 0; i < tileO::NumTiles; ++i) {
            normalize<typename tileO_cast::TileType, typename tileO::TileType, typename tileSum::TileType><<<tileO::TileType::ValidRow, tileO::TileType::ValidCol, 1>>>(tO_cast.Tiles[i].data(), tO.Tiles[i].data(), tRescaleO.Tiles[i].data(), tSum.Tiles[i].data());
        }
        // 写回全局内存
        #ifdef MULTI_LDST
        auto gO = gIterO(i, 0);
        TSTORE2_DN2DN(gO, tO_cast.Tiles[1], tO_cast.Tiles[0]);
        #else
        TCOPYOUT([&](int t) { return gIterO(i + t, 0); }, tO_cast);
        #endif
    }
}