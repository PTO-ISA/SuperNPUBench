
template<typename tileSrc, typename tileSrc_cast, typename tileMax, typename tileSum, typename tileScale>
void __vec__ flashsoftmax_dn_mout_cast_kernel(
                        typename tileScale::TileDType __out__ rescale,
                        typename tileMax::TileDType __out__ new_max,
                        typename tileSum::TileDType __out__ new_sum,
                        typename tileSrc_cast::TileDType __out__ src_exp,
                        const typename tileSrc::TileDType __in__ src,
                        const typename tileMax::TileDType __in__ old_max,
                        const typename tileSum::TileDType __in__ old_sum,
                        const typename tileSrc::DType src_scale
                                ){
    size_t i = blkv_get_index_x();

    __vbuf__ typename tileScale::DType *scale_ptr = blkv_get_tile_ptr(rescale);
    __vbuf__ typename tileMax::DType *new_max_ptr = blkv_get_tile_ptr(new_max);
    __vbuf__ typename tileSum::DType *new_sum_ptr = blkv_get_tile_ptr(new_sum);
    __vbuf__ typename tileSrc_cast::DType *src_exp_ptr = blkv_get_tile_ptr(src_exp);
    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileMax::DType *old_max_ptr = blkv_get_tile_ptr(old_max);
    __vbuf__ typename tileSum::DType *old_sum_ptr = blkv_get_tile_ptr(old_sum);

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
    upd_max = upd_max * src_scale;
    new_max_ptr[max_idx] = upd_max;

    typename tileScale::DType scale = blkv_fexp(old_max_ptr[max_idx] - upd_max);

    size_t scale_idx = i*tileScale::RowStride;
    scale_ptr[scale_idx] = scale;

    size_t sum_idx = i*tileSum::RowStride;
    typename tileSum::DType upd_sum = old_sum_ptr[sum_idx] * scale;

    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileSrc::ValidCol;j+=4){
        size_t src_idx_0 =  i * tileSrc::RowStride + j * tileSrc::ColStride;
        size_t src_idx_1 =  i * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        size_t src_idx_2 =  i * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        size_t src_idx_3 =  i * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
        typename tileSrc::DType exp_src_0 = blkv_fexp(src_ptr[src_idx_0] * src_scale - upd_max);
        typename tileSrc::DType exp_src_1 = blkv_fexp(src_ptr[src_idx_1] * src_scale - upd_max);
        typename tileSrc::DType exp_src_2 = blkv_fexp(src_ptr[src_idx_2] * src_scale - upd_max);
        typename tileSrc::DType exp_src_3 = blkv_fexp(src_ptr[src_idx_3] * src_scale - upd_max);
        typename tileSrc::DType exp_src_01 = exp_src_0 + exp_src_1;
        typename tileSrc::DType exp_src_23 = exp_src_2 + exp_src_3;
        typename tileSrc::DType exp_src_0123 = exp_src_01 + exp_src_23;
        upd_sum += exp_src_0123;
        src_exp_ptr[src_idx_0] = static_cast<typename tileSrc_cast::DType>(exp_src_0);
        src_exp_ptr[src_idx_1] = static_cast<typename tileSrc_cast::DType>(exp_src_1);
        src_exp_ptr[src_idx_2] = static_cast<typename tileSrc_cast::DType>(exp_src_2);
        src_exp_ptr[src_idx_3] = static_cast<typename tileSrc_cast::DType>(exp_src_3);
    }
    new_sum_ptr[sum_idx] = upd_sum;
}

template<typename tileO, typename tileScale>
void __vec__ global_update(
                        typename tileO::TileDType __out__ update_out,
                        const typename tileO::TileDType __in__ old_out,
                        const typename tileO::TileDType __in__ pv,
                        const typename tileScale::TileDType __in__ scale
                    ){
    size_t i = blkv_get_index_x();

    size_t idx_scale = i * tileScale::RowStride;

    typename tileScale::DType local_scale = blkv_get_tile_ptr(scale)[idx_scale];

    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileO::ValidCol;j++){
        size_t idx = j * tileO::ColStride + i * tileO::RowStride;
        blkv_get_tile_ptr(update_out)[idx] = blkv_get_tile_ptr(old_out)[idx] * local_scale + blkv_get_tile_ptr(pv)[idx] ;
    }
}

template<typename tileO_cast, typename tileO, typename tileSum, typename tileScale>
void __vec__ normalize_with_last_update(
                        typename tileO_cast::TileDType __out__ out_cast,
                        const typename tileO::TileDType __in__ old_out,
                        const typename tileO::TileDType __in__ last_pv,
                        const typename tileScale::TileDType __in__ last_scale,
                        const typename tileSum::TileDType __in__ sum
                    ){
    size_t j = blkv_get_index_x();
    size_t i = blkv_get_index_y();

    size_t idx = i * tileO::ColStride + j;
    size_t idx_sum = j * tileSum::RowStride;
    size_t idx_scale = j * tileScale::RowStride;

    BLKC_ASSIGN_CAST(
        out_cast,
        idx,
        (blkv_get_tile_ptr(last_pv)[idx] +
         blkv_get_tile_ptr(old_out)[idx] * blkv_get_tile_ptr(last_scale)[idx_scale]) /
        blkv_get_tile_ptr(sum)[idx_sum]
    );

}

template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk, int MULTI = 2>
void flash_attention_dcore(dtype* out_ptr, dtype* q_ptr, dtype* k_ptr, dtype* v_ptr) {
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
    using tileW_cast = MultiTile<MULTI, Tile<Location::Vec, dtype, kTm, kTk, BLayout::ColMajor>>;
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
        TEXPANDSCALAR(tMax, -1e30f); // 初始化为极小值
        tileSum tSum;              // 指数和归零
        TEXPANDSCALAR(tSum, 0);
        tileO tO, tPV;
        tileScale tScale;

        // 单次遍历所有K/V块
        #pragma clang loop unroll(full)
        for (int j = 0; j < Kb; ++j) {
            // 加载K_j和V_j
            auto gK = gIterK(0, j);
            tileK tK;
            TCOPYIN(tK, gK);

            // 计算注意力分数块
            tileW tW;
            MATMUL(tW, tQ, tK);

            tileMax tNewMax;
            tileSum tNewSum;
            tileW_cast tExpW;
            #pragma clang loop unroll(full)
            for (int i = 0; i < tileW::NumTiles; ++i) {
            flashsoftmax_dn_mout_cast_kernel<typename tileW::TileType, typename tileW_cast::TileType, typename tileMax::TileType, typename tileSum::TileType, typename tileScale::TileType><<<tileW::TileType::ValidRow, 1, 1>>>(
                                                        tScale.Tiles[i].data(),
                                                        tNewMax.Tiles[i].data(),
                                                        tNewSum.Tiles[i].data(),
                                                        tExpW.Tiles[i].data(),
                                                        tW.Tiles[i].data(),
                                                        tMax.Tiles[i].data(),
                                                        tSum.Tiles[i].data(),
                                                        scale);
            }

            // ColMajor -> Nz
            tileW_left tW_left;
            TCVT_DN2NZ(tW_left, tExpW);
            //TCVT(tW_left, tExpW);
            // 计算当前块的加权输出: O_j = W * V
            auto gV = gIterV(j, 0);
            tileV tV;
            TCOPYIN(tV, gV);
            MATMUL(tPV, tW_left, tV);

            if(j==0){
                tO = tPV;
            }else if(j!=(Kb-1)){
                #pragma clang loop unroll(full)
                for (int i = 0; i < tileO::NumTiles; ++i) {
                    global_update<typename tileO::TileType, typename tileScale::TileType><<<tileO::TileType::ValidRow, 1, 1>>>(tO.Tiles[i].data(), tO.Tiles[i].data(), tPV.Tiles[i].data(), tScale.Tiles[i].data());
                }
            }
            // 更新最大值状态
            tMax = tNewMax;
            tSum = tNewSum;
        }

        tileO_cast tO_cast;
        #pragma clang loop unroll(full)
        for (int i = 0; i < tileO::NumTiles; ++i) {
            normalize_with_last_update<typename tileO_cast::TileType, typename tileO::TileType, typename tileSum::TileType, typename tileScale::TileType><<<tileO::TileType::ValidRow, tileO::TileType::ValidCol, 1>>>(tO_cast.Tiles[i].data(), tO.Tiles[i].data(), tPV.Tiles[i].data(), tScale.Tiles[i].data(), tSum.Tiles[i].data());
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
