
template<typename tileO_cast, typename tileO, typename tileSum>
void __vec__ normalize_no_last_update(
                        typename tileO_cast::TileDType __out__ out_cast,
                        const typename tileO::TileDType __in__ out,
                        const typename tileSum::TileDType __in__ sum
                    ){
    size_t j = blkv_get_index_x();
    size_t i = blkv_get_index_y();

    size_t idx = i * tileO::ColStride + j;
    size_t idx_sum = j * tileSum::RowStride;
    blkv_get_tile_ptr(out_cast)[idx] = static_cast<typename tileO_cast::DType>(blkv_get_tile_ptr(out)[idx] /  blkv_get_tile_ptr(sum)[idx_sum]);
}

template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk>
void flash_attention_unalign_2d_unroll(dtype* out_ptr, dtype* q_ptr, dtype* k_ptr, dtype* v_ptr) {

    const float scale = 1.0f / sqrt((float)qD);
    const int Qb = Sq / kTm;
    const int Kb = Skv / kTk;
    const int rQ = Sq % kTm;
    const int rK = Skv % kTk;

    #ifdef _UNALIGN_2D_UNROLL
    static_assert(Qb%Xdim==0, "Qb needs to be a multiple of Xdim");
    static_assert(Kb%Ydim==0, "Kb needs to be a multiple of Ydim");
    static_assert(rQ!=0, "rQ needs not to be a zero in unalign func");
    static_assert(rK!=0, "rK needs not to be a zero in unalign func");
    #endif

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
    using tileW_cast = Tile<Location::Vec, dtype, kTm, kTk, BLayout::ColMajor>;
    using tileW_left = TileLeft<dtype, kTm, kTk>; 

    using tileO_out  = TileAcc<float, kTm, vD>;
    using tileO      = Tile<Location::Vec, float, kTm, vD, BLayout::ColMajor>; // [kTm×vD]
    using tileO_cast = Tile<Location::Vec, dtype, kTm, vD, BLayout::ColMajor>;

    using tileV      = TileRight<dtype, kTk, vD>;     // [kTk×vD]
    using tileMax    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>; // [kTm×1]
    using tileSum    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>; // [kTm×1]
    using tileScale  = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>; // [kTm×1]

    //remainder block
    using tileQ_trows = TileLeft<dtype, kTm, qD, rQ, qD>;
    using tileK_tcols = TileRight<dtype, qD, kTk, qD, rK>;

    using tileW_out_tcols = TileAcc<float, kTm, kTk, kTm, rK>;
    using tileW_out_trows = TileAcc<float, kTm, kTk, rQ,kTk>;
    using tileW_out_tcorner = TileAcc<float, kTm, kTk, rQ, rK>;

    using tileW_tcols = Tile<Location::Vec, float, kTm, kTk, BLayout::ColMajor, kTm, rK>;
    using tileW_trows = Tile<Location::Vec, float, kTm, kTk, BLayout::ColMajor, rQ, kTk>;
    using tileW_tcorner = Tile<Location::Vec, float, kTm, kTk, BLayout::ColMajor, rQ, rK>;

    using tileW_cast_tcols = Tile<Location::Vec, dtype, kTm, kTk, BLayout::ColMajor, kTm, rK>;
    using tileW_cast_trows = Tile<Location::Vec, dtype, kTm, kTk, BLayout::ColMajor, rQ, kTk>;
    using tileW_cast_tcorner = Tile<Location::Vec, dtype, kTm, kTk, BLayout::ColMajor, rQ, rK>;

    using tileW_left_tcols = TileLeft<dtype, kTm, kTk, kTm, rK>;
    using tileW_left_trows = TileLeft<dtype, kTm, kTk, rQ, kTk>;
    using tileW_left_tcorner = TileLeft<dtype, kTm, kTk, rQ, rK>;

    using tileO_out_trows = TileAcc<float, kTm, vD, rQ, vD>;
    using tileO_trows = Tile<Location::Vec, float, kTm, vD, BLayout::ColMajor, rQ, vD>;
    using tileO_cast_trows = Tile<Location::Vec, dtype, kTm, vD, BLayout::ColMajor, rQ, vD>;

    using tileMax_trows = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, rQ, 1>;
    using tileSum_trows = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, rQ, 1>;
    using tileScale_trows = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, rQ, 1>;

    using tileV_trows = TileRight<dtype, kTk, vD, rK, vD>;

    // 全局迭代器
    using itQ = global_iterator<gmQ, tileQ>;
    using itK = global_iterator<gmK, tileK>;
    using itV = global_iterator<gmV, tileV>;
    using itO = global_iterator<gmO, tileO>;

    itQ gIterQ(q_ptr);
    itK gIterK(k_ptr);
    itV gIterV(v_ptr);
    itO gIterO(out_ptr);

    // 对每个 Q-block (i)
    for (int i = 0; i < Qb; i+=Xdim) {

        tileQ tQ[Xdim];
        #pragma clang loop unroll(full)
        for(int x=0;x<Xdim;x++){
            auto gQ = gIterQ(i+x,0);
            TCOPYIN(tQ[x], gQ);
        }

        tileMax tMax[Xdim];
        #pragma clang loop unroll(full)
        for(int x=0;x<Xdim;x++){
            TEXPANDSCALAR(tMax[x], -1e30f);
        }

        tileSum tSum[Xdim];
        #pragma clang loop unroll(full)
        for(int x=0;x<Xdim;x++){
            TEXPANDSCALAR(tSum[x], 0);
        }

        tileO_out tPV_out;
        tileO tO[Xdim], tPV[Xdim];
        tileScale tScale[Xdim];

        #pragma clang loop unroll(full)
        for (int j = 0; j < Kb; j+=Ydim) {

            tileK tK[Ydim];

            #pragma clang loop unroll(full)
            for(int y=0;y<Ydim;y++){
                auto gK = gIterK(0, j+y);
                TCOPYIN(tK[y], gK);
            }

            tileW tW[Xdim][Ydim];
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                #pragma clang loop unroll(full)
                for(int y=0;y<Ydim;y++){
                    tileW_out tW_out;
                    MATMUL(tW_out, tQ[x], tK[y]);
                    // Nz -> ColMajor
                    TCVT(tW[x][y], tW_out);
                }
            }

            tileMax tNewMax[Xdim];
            tileSum tNewSum[Xdim];

            #if Ydim == 1
            tileW_cast tExpW[Xdim][Ydim];
            #else
            tileW tExpW[Xdim][Ydim];
            #endif

            #if Ydim == 1
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    flashsoftmax_dn_mout_cast_kernel<tileW, tileW_cast, tileMax, tileSum, tileScale><<<tileW::ValidRow, 1, 1>>>(
                                                    tScale[x].data(),
                                                    tNewMax[x].data(),
                                                    tNewSum[x].data(),
                                                    tExpW[x][0].data(),
                                                    tW[x][0].data(),
                                                    tMax[x].data(),
                                                    tSum[x].data(),
                                                    scale);
                }
            #elif Ydim == 2
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    new_max_2src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(
                                                                tScale[x].data(),
                                                                tNewMax[x].data(),
                                                                tW[x][0].data(), tW[x][1].data(),
                                                                tMax[x].data(),
                                                                scale);
                    src_exp_2src<tileW, tileMax><<<tileW::ValidRow, tileW::ValidCol, 1>>>(
                                                                tExpW[x][0].data(), tExpW[x][1].data(),
                                                                tW[x][0].data(), tW[x][1].data(),
                                                                tNewMax[x].data(),
                                                                scale);
                    new_sum_2src<tileW, tileSum, tileScale><<<tileSum::ValidRow, 1, 1>>>(
                                                                tNewSum[x].data(),
                                                                tExpW[x][0].data(), tExpW[x][1].data(),
                                                                tSum[x].data(),
                                                                tScale[x].data()
                                                                );
                }
            #elif Ydim == 4
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    new_max_4src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(
                                                                tScale[x].data(), 
                                                                tNewMax[x].data(), 
                                                                tW[x][0].data(), tW[x][1].data(), tW[x][2].data(), tW[x][3].data(),
                                                                tMax[x].data(),
                                                                scale);
                    src_exp_4src<tileW, tileMax><<<tileW::ValidRow, tileW::ValidCol, 1>>>(
                                                                tExpW[x][0].data(), tExpW[x][1].data(), tExpW[x][2].data(), tExpW[x][3].data(),
                                                                tW[x][0].data(), tW[x][1].data(), tW[x][2].data(), tW[x][3].data(),
                                                                tNewMax[x].data(),
                                                                scale);
                    new_sum_4src<tileW, tileSum, tileScale><<<tileSum::ValidRow, 1, 1>>>(
                                                                tNewSum[x].data(),
                                                                tExpW[x][0].data(), tExpW[x][1].data(), tExpW[x][2].data(), tExpW[x][3].data(),
                                                                tSum[x].data(),
                                                                tScale[x].data()
                                                                );
                }
            #else
                #ifdef _UNALIGN_2D_UNROLL
                static_assert(Ydim==1 || Ydim == 2 || Ydim==4, "Not Supprot Ydim != 1 or 2 or 4.");
                #endif
            #endif

            tileV tV[Ydim];
            #pragma clang loop unroll(full)
            for(int y=0;y<Ydim;y++){
                auto gV = gIterV(j+y, 0);
                TCOPYIN(tV[y], gV);
            }

            // ColMajor -> Nz
            // 计算当前块的加权输出: O_j = W * V
            tileW_left tW_left[Xdim][Ydim];
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                #pragma clang loop unroll(full)
                for(int y=0;y<Ydim;y++){
                    TCVT_DN2NZ(tW_left[x][y], tExpW[x][y]);
                    //TCVT(tW_left[x][y], tExpW[x][y]);
                    if(y==0){
                        MATMUL(tPV_out, tW_left[x][y], tV[y]);
                    }else{
                        MATMACC(tPV_out, tW_left[x][y], tV[y]);
                    }
                }
                TCVT(tPV[x], tPV_out);
            }

            // update
            if(j==0){
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    tO[x] = tPV[x];
                }
            }else{
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    global_update<tileO, tileScale><<<tileO::ValidRow, 1, 1>>>(tO[x].data(), tO[x].data(), tPV[x].data(), tScale[x].data());
                }
            }
            // 更新最大值状态
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                tMax[x] = tNewMax[x];
                tSum[x] = tNewSum[x];
            }
        }

        if constexpr(rK) {

            tileK_tcols tK;
            auto gK = gIterK(0, Kb);
            TCOPYIN(tK, gK);

            tileW_tcols tW[Xdim];
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                tileW_out_tcols tW_out;
                MATMUL(tW_out, tQ[x], tK);
                TCVT(tW[x], tW_out);
            }

            tileMax tNewMax[Xdim];
            tileSum tNewSum[Xdim];

            tileW_cast_tcols tExpW[Xdim];

            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                flashsoftmax_dn_mout_cast_kernel<tileW_tcols, tileW_cast_tcols, tileMax, tileSum, tileScale><<<tileW_tcols::ValidRow, 1, 1>>>(
                                                tScale[x].data(),
                                                tNewMax[x].data(),
                                                tNewSum[x].data(),
                                                tExpW[x].data(),
                                                tW[x].data(),
                                                tMax[x].data(),
                                                tSum[x].data(),
                                                scale);
            }

            tileV_trows tV;
            auto gV = gIterV(Kb, 0);
            TCOPYIN(tV, gV);

            // 计算当前块的加权输出: O_j = W * V
            tileW_left_tcols tW_left[Xdim];
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                TCVT_DN2NZ(tW_left[x], tExpW[x]);
                //TCVT(tW_left[x], tExpW[x]);
                MATMUL(tPV_out, tW_left[x], tV);
                TCVT(tPV[x], tPV_out);
            }

            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                global_update<tileO, tileScale><<<tileO::ValidRow, 1, 1>>>(tO[x].data(), tO[x].data(), tPV[x].data(), tScale[x].data());
            }

            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                tMax[x] = tNewMax[x];
                tSum[x] = tNewSum[x];
            }
        }

        tileO_cast tO_cast[Xdim];
        #pragma clang loop unroll(full)
        for (int x = 0; x < Xdim; ++x) {
            normalize_no_last_update<tileO_cast, tileO, tileSum><<<tileO::ValidRow, tileO::ValidCol, 1>>>(tO_cast[x].data(), tO[x].data(), tSum[x].data());
        }
        // 写回全局内存

        #pragma clang loop unroll(full)
        for (int x = 0; x < Xdim; ++x) {
            auto dstO = gIterO(i+x, 0);
            TCOPYOUT(dstO, tO_cast[x]);
        }
    }

    if constexpr(rQ){

        tileQ_trows tQ;

        auto gQ = gIterQ(Qb,0);
        TCOPYIN(tQ, gQ);

        tileMax_trows tMax;
        TEXPANDSCALAR(tMax, -1e30f);

        tileSum_trows tSum;
        TEXPANDSCALAR(tSum, 0);

        tileO_out_trows tPV_out;
        tileO_trows tO, tPV;
        tileScale_trows tScale;

        #pragma clang loop unroll(full)
        for (int j = 0; j < Kb; j+=Ydim) {

            tileK tK[Ydim];

            #pragma clang loop unroll(full)
            for(int y=0;y<Ydim;y++){
                auto gK = gIterK(0, j+y);
                TCOPYIN(tK[y], gK);
            }

            tileW_trows tW[Ydim];

            #pragma clang loop unroll(full)
            for(int y=0;y<Ydim;y++){
                tileW_out_trows tW_out;
                MATMUL(tW_out, tQ, tK[y]);
                TCVT(tW[y], tW_out);
            }

            tileMax_trows tNewMax;
            tileSum_trows tNewSum;

            #if Ydim == 1
            tileW_cast_trows tExpW[Ydim];
            #else
            tileW_trows tExpW[Ydim];
            #endif

            #if Ydim == 1
                flashsoftmax_dn_mout_cast_kernel<tileW_trows, tileW_cast_trows, tileMax_trows, tileSum_trows, tileScale_trows><<<tileW_trows::ValidRow, 1, 1>>>(
                                                tScale.data(),
                                                tNewMax.data(),
                                                tNewSum.data(),
                                                tExpW[0].data(),
                                                tW[0].data(),
                                                tMax.data(),
                                                tSum.data(),
                                                scale);
            #elif Ydim == 2
                new_max_2src<tileW_trows, tileMax_trows><<<tileMax_trows::ValidRow, 1, 1>>>(
                                                            tScale.data(),
                                                            tNewMax.data(),
                                                            tW[0].data(), tW[1].data(),
                                                            tMax.data(),
                                                            scale);
                src_exp_2src<tileW_trows, tileMax_trows><<<tileW_trows::ValidRow, tileW_trows::ValidCol, 1>>>(
                                                            tExpW[0].data(), tExpW[1].data(),
                                                            tW[0].data(), tW[1].data(),
                                                            tNewMax.data(),
                                                            scale);
                new_sum_2src<tileW_trows, tileSum_trows, tileScale_trows><<<tileSum_trows::ValidRow, 1, 1>>>(
                                                            tNewSum.data(),
                                                            tExpW[0].data(), tExpW[1].data(),
                                                            tSum.data(),
                                                            tScale.data()
                                                            );
            #elif Ydim == 4
                new_max_4src<tileW_trows, tileMax_trows><<<tileMax_trows::ValidRow, 1, 1>>>(
                                                            tScale.data(), 
                                                            tNewMax.data(), 
                                                            tW[0].data(), tW[1].data(), tW[2].data(), tW[3].data(),
                                                            tMax.data(),
                                                            scale);
                src_exp_4src<tileW_trows, tileMax_trows><<<tileW_trows::ValidRow, tileW_trows::ValidCol, 1>>>(
                                                            tExpW[0].data(), tExpW[1].data(), tExpW[2].data(), tExpW[3].data(),
                                                            tW[0].data(), tW[1].data(), tW[2].data(), tW[3].data(),
                                                            tNewMax.data(),
                                                            scale);
                new_sum_4src<tileW_trows, tileSum_trows, tileScale_trows><<<tileSum_trows::ValidRow, 1, 1>>>(
                                                            tNewSum.data(),
                                                            tExpW[0].data(), tExpW[1].data(), tExpW[2].data(), tExpW[3].data(),
                                                            tSum.data(),
                                                            tScale.data()
                                                            );
            #else
                #ifdef _UNALIGN_2D_UNROLL
                static_assert(Ydim==1 || Ydim == 2 || Ydim==4, "Not Supprot Ydim != 1 or 2 or 4.");
                #endif
            #endif

            tileV tV[Ydim];

            #pragma clang loop unroll(full)
            for(int y=0;y<Ydim;y++){
                auto gV = gIterV(j+y, 0);
                TCOPYIN(tV[y], gV);
            }

            tileW_left_trows tW_left[Ydim];
            #pragma clang loop unroll(full)
            for(int y=0;y<Ydim;y++){
                TCVT_DN2NZ(tW_left[y], tExpW[y]);
                //TCVT(tW_left[y], tExpW[y]);
                if(y==0){
                    MATMUL(tPV_out, tW_left[y], tV[y]);
                }else{
                    MATMACC(tPV_out, tW_left[y], tV[y]);
                }
            }
            TCVT(tPV, tPV_out);

            if(j==0){
                tO = tPV;
            }else{
                global_update<tileO_trows, tileScale_trows><<<tileO_trows::ValidRow, 1, 1>>>(tO.data(), tO.data(), tPV.data(), tScale.data());
            }

            tMax = tNewMax;
            tSum = tNewSum;
        }

        if constexpr(rK) {

            tileK_tcols tK;
            auto gK = gIterK(0, Kb);
            TCOPYIN(tK, gK);

            tileW_tcorner tW;
            tileW_out_tcorner tW_out;
            MATMUL(tW_out, tQ, tK);
            TCVT(tW, tW_out);

            tileMax_trows tNewMax;
            tileSum_trows tNewSum;

            tileW_cast_tcorner tExpW;

            flashsoftmax_dn_mout_cast_kernel<tileW_tcorner, tileW_cast_tcorner, tileMax_trows, tileSum_trows, tileScale_trows><<<tileW_tcorner::ValidRow, 1, 1>>>(
                                            tScale.data(),
                                            tNewMax.data(),
                                            tNewSum.data(),
                                            tExpW.data(),
                                            tW.data(),
                                            tMax.data(),
                                            tSum.data(),
                                            scale);

            tileV_trows tV;
            auto gV = gIterV(Kb, 0);
            TCOPYIN(tV, gV);

            tileW_left_tcorner tW_left;
            TCVT_DN2NZ(tW_left, tExpW);
            //TCVT(tW_left, tExpW);
            MATMUL(tPV_out, tW_left, tV);
            TCVT(tPV, tPV_out);

            global_update<tileO_trows, tileScale_trows><<<tileO_trows::ValidRow, 1, 1>>>(tO.data(), tO.data(), tPV.data(), tScale.data());

            tMax = tNewMax;
            tSum = tNewSum;
        }

        tileO_cast_trows tO_cast;
        normalize_no_last_update<tileO_cast_trows, tileO_trows, tileSum_trows><<<tileO_trows::ValidRow, tileO_trows::ValidCol, 1>>>(tO_cast.data(), tO.data(), tSum.data());

        auto dstO = gIterO(Qb, 0);
        TCOPYOUT(dstO, tO_cast);
    }
}