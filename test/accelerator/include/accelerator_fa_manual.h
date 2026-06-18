template<typename tileSrc, typename tileMax>
void __vec__ new_max_manual(
                        typename tileMax::TileDType __out__ scale,
                        typename tileMax::TileDType __out__ new_max,
                        const typename tileSrc::TileDType __in__ src,
                        const typename tileMax::TileDType __in__ old_max,
                        const typename tileSrc::DType src_scale
                    ){
    uint16_t i = blkv_get_index_x();

    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileMax::DType *new_max_ptr = blkv_get_tile_ptr(new_max);
    __vbuf__ typename tileMax::DType *old_max_ptr = blkv_get_tile_ptr(old_max);
    __vbuf__ typename tileMax::DType *scale_ptr = blkv_get_tile_ptr(scale);

    size_t max_idx = i*tileMax::RowStride;

    typename tileMax::DType old_max_val = old_max_ptr[max_idx];
    typename tileMax::DType upd_max = old_max_val;

    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileSrc::ValidCol;j+=4){
        size_t src_idx_0 =  i * tileSrc::RowStride + j * tileSrc::ColStride;
        size_t src_idx_1 =  i * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        size_t src_idx_2 =  i * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        size_t src_idx_3 =  i * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;
        #ifndef RES_CHECK
        typename tileMax::DType max_01 = blkv_max(src_ptr[src_idx_0], src_ptr[src_idx_1]);
        typename tileMax::DType max_23 = blkv_max(src_ptr[src_idx_2], src_ptr[src_idx_3]);
        #else
        typename tileMax::DType max_01 = blkv_max(src_ptr[src_idx_0] * src_scale, src_ptr[src_idx_1] * src_scale);
        typename tileMax::DType max_23 = blkv_max(src_ptr[src_idx_2] * src_scale, src_ptr[src_idx_3] * src_scale);
        #endif
        typename tileMax::DType max_0123 = blkv_max(max_01, max_23);
        upd_max = blkv_max(upd_max, max_0123);
    }
    #ifndef RES_CHECK
    upd_max = upd_max * src_scale;
    #endif
    new_max_ptr[max_idx] = upd_max; 

    scale_ptr[max_idx] =  blkv_fexp(old_max_val - upd_max); 
}

template<typename tileSrc, typename tileSrc_cast, typename tileMax>
void __vec__ src_exp_manual(
                        typename tileSrc_cast::TileDType __out__ src_exp,
                        const typename tileSrc::TileDType __in__ src,
                        const typename tileMax::TileDType __in__ new_max,
                        const typename tileSrc::DType src_scale
                    ){
    size_t i = blkv_get_index_x();
    size_t j = blkv_get_index_y();

    size_t idx = j * tileSrc::ColStride + i;
    size_t idx_max = i * tileMax::RowStride;

    blkv_get_tile_ptr(src_exp)[idx] = static_cast<typename tileSrc_cast::DType>(blkv_fexp(blkv_get_tile_ptr(src)[idx]*src_scale - blkv_get_tile_ptr(new_max)[idx_max]));
}

template<typename tileSrc, typename tileSum, typename tileScale>
void __vec__ new_sum_manual(
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
    for(size_t j=0;j<tileSrc::ValidCol;j+=4){
        size_t src_idx_0 =  i * tileSrc::RowStride + j * tileSrc::ColStride;
        size_t src_idx_1 =  i * tileSrc::RowStride + (j + 1) * tileSrc::ColStride;
        size_t src_idx_2 =  i * tileSrc::RowStride + (j + 2) * tileSrc::ColStride;
        size_t src_idx_3 =  i * tileSrc::RowStride + (j + 3) * tileSrc::ColStride;

        typename tileSrc::DType exp_src_0 = src_ptr[src_idx_0];
        typename tileSrc::DType exp_src_1 = src_ptr[src_idx_1];
        typename tileSrc::DType exp_src_2 = src_ptr[src_idx_2];
        typename tileSrc::DType exp_src_3 = src_ptr[src_idx_3];
        typename tileSrc::DType exp_src_01 = exp_src_0 + exp_src_1;
        typename tileSrc::DType exp_src_23 = exp_src_2 + exp_src_3;
        typename tileSrc::DType exp_src_0123 = exp_src_01 + exp_src_23;        
        upd_sum += exp_src_0123;
    }
    blkv_get_tile_ptr(new_sum)[sum_idx] = upd_sum;
}


template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk>
void flash_attention_manual(dtype* out_ptr, dtype* q_ptr, dtype* k_ptr, dtype* v_ptr) {
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

    #ifdef MANUAL
    static_assert(Qb%Xdim==0, "Qb needs to be a multiple of Xdim");
    static_assert(Kb%Ydim==0, "Kb needs to be a multiple of Ydim");
    static_assert(Ydim==4, "Kb needs to be a multiple of Ydim");
    #endif

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


            const int j = 0;
            tileK tK[Ydim];
            tileW tW[Xdim][Ydim];
            tileMax tNewMax[Xdim];
            tileSum tNewSum[Xdim];

            tileW_cast tExpW[Xdim][Ydim];

            auto gK = gIterK(0, j+0);
            TCOPYIN(tK[0], gK);
            gK = gIterK(0, j+1);
            TCOPYIN(tK[1], gK);

            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                tileW_out tW_out;

                //Q0 * K0 = S0
                MATMUL(tW_out, tQ[x], tK[0]);
                TCVT(tW[x][0], tW_out);

                //Q0 * K1 = S1
                MATMUL(tW_out, tQ[x], tK[1]);
                TCVT(tW[x][1], tW_out);
            }

            //softmax0
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                // flashsoftmax_dn_mout_cast_kernel<tileW, tileW_cast, tileMax, tileSum, tileScale><<<tileW::ValidRow, 1, 1>>>(
                //                                 tScale[x].data(),
                //                                 tNewMax[x].data(),
                //                                 tNewSum[x].data(),
                //                                 tExpW[x][0].data(),
                //                                 tW[x][0].data(),
                //                                 tMax[x].data(),
                //                                 tSum[x].data(),
                //                                 scale);
                new_max_manual<tileW, tileMax><<<tileW::ValidRow, 1, 1>>>(
                    tScale[x].data(),
                    tNewMax[x].data(),
                    tW[x][0].data(),
                    tMax[x].data(),
                    scale
                );
                src_exp_manual<tileW, tileW_cast, tileMax><<<tileW::ValidRow, tileW::ValidCol, 1>>>(tExpW[x][0].data(), tW[x][0].data(), tNewMax[x].data(), scale);
                new_sum_manual<tileW_cast, tileSum, tileScale><<<tileSum::ValidRow, 1, 1>>>(tNewSum[x].data(), tExpW[x][0].data(), tSum[x].data(), tScale[x].data());
            }

            //Q0 * K2 = S2
            gK = gIterK(0, j+2);
            TCOPYIN(tK[2], gK);
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                tileW_out tW_out;
                MATMUL(tW_out, tQ[x], tK[2]);
                TCVT(tW[x][2], tW_out);
            }

            // P0 * V0 = PV0
            tileV tV[Ydim];
            auto gV = gIterV(j+0, 0);
            TCOPYIN(tV[0], gV);

            tileW_left tW_left[Xdim][Ydim];
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                TCVT_DN2NZ(tW_left[x][0], tExpW[x][0]);
                MATMUL(tPV_out, tW_left[x][0], tV[0]);
                TCVT(tPV[x], tPV_out);
            }

            // update0
            if(j==0){
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    tO[x] = tPV[x];
                }
            } else{
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    global_update<tileO, tileScale><<<tileO::ValidRow, 1, 1>>>(tO[x].data(), tO[x].data(), tPV[x].data(), tScale[x].data());
                }
            }


            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                tMax[x] = tNewMax[x];
                tSum[x] = tNewSum[x];
            }

            //softmax 1
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                // flashsoftmax_dn_mout_cast_kernel<tileW, tileW_cast, tileMax, tileSum, tileScale><<<tileW::ValidRow, 1, 1>>>(
                //                                 tScale[x].data(),
                //                                 tNewMax[x].data(),
                //                                 tNewSum[x].data(),
                //                                 tExpW[x][1].data(),
                //                                 tW[x][1].data(),
                //                                 tMax[x].data(),
                //                                 tSum[x].data(),
                //                                 scale);
                new_max_manual<tileW, tileMax><<<tileW::ValidRow, 1, 1>>>(
                    tScale[x].data(),
                    tNewMax[x].data(),
                    tW[x][1].data(),
                    tMax[x].data(),
                    scale
                );
                src_exp_manual<tileW, tileW_cast, tileMax><<<tileW::ValidRow, tileW::ValidCol, 1>>>(tExpW[x][1].data(), tW[x][1].data(), tNewMax[x].data(), scale);
                new_sum_manual<tileW_cast, tileSum, tileScale><<<tileSum::ValidRow, 1, 1>>>(tNewSum[x].data(), tExpW[x][1].data(), tSum[x].data(), tScale[x].data());
            }

            //Q0 * K3 = S3
            gK = gIterK(0, j+3);
            TCOPYIN(tK[3], gK);
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                tileW_out tW_out;
                MATMUL(tW_out, tQ[x], tK[3]);
                TCVT(tW[x][3], tW_out);
            }

            //P1 * V1 = PV1
            gV = gIterV(j+1, 0);
            TCOPYIN(tV[1], gV);

            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                TCVT_DN2NZ(tW_left[x][1], tExpW[x][1]);
                MATMUL(tPV_out, tW_left[x][1], tV[1]);
                TCVT(tPV[x], tPV_out);
            }

            // update 1
            for(int x=0;x<Xdim;x++){
                global_update<tileO, tileScale><<<tileO::ValidRow, 1, 1>>>(tO[x].data(), tO[x].data(), tPV[x].data(), tScale[x].data());
            }


            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                tMax[x] = tNewMax[x];
                tSum[x] = tNewSum[x];
            }

            //softmax 2
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                // flashsoftmax_dn_mout_cast_kernel<tileW, tileW_cast, tileMax, tileSum, tileScale><<<tileW::ValidRow, 1, 1>>>(
                //                                 tScale[x].data(),
                //                                 tNewMax[x].data(),
                //                                 tNewSum[x].data(),
                //                                 tExpW[x][2].data(),
                //                                 tW[x][2].data(),
                //                                 tMax[x].data(),
                //                                 tSum[x].data(),
                //                                 scale);
                new_max_manual<tileW, tileMax><<<tileW::ValidRow, 1, 1>>>(
                    tScale[x].data(),
                    tNewMax[x].data(),
                    tW[x][2].data(),
                    tMax[x].data(),
                    scale
                );
                src_exp_manual<tileW, tileW_cast, tileMax><<<tileW::ValidRow, tileW::ValidCol, 1>>>(tExpW[x][2].data(), tW[x][2].data(), tNewMax[x].data(), scale);
                new_sum_manual<tileW_cast, tileSum, tileScale><<<tileSum::ValidRow, 1, 1>>>(tNewSum[x].data(), tExpW[x][2].data(), tSum[x].data(), tScale[x].data());
            }

            //P2 * V2 = PV2
            gV = gIterV(j+2, 0);
            TCOPYIN(tV[2], gV);

            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                TCVT_DN2NZ(tW_left[x][2], tExpW[x][2]);
                MATMUL(tPV_out, tW_left[x][2], tV[2]);
                TCVT(tPV[x], tPV_out);
            }

            // update 2
            for(int x=0;x<Xdim;x++){
                global_update<tileO, tileScale><<<tileO::ValidRow, 1, 1>>>(tO[x].data(), tO[x].data(), tPV[x].data(), tScale[x].data());
            }


            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                tMax[x] = tNewMax[x];
                tSum[x] = tNewSum[x];
            }

            //softmax 3
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                // flashsoftmax_dn_mout_cast_kernel<tileW, tileW_cast, tileMax, tileSum, tileScale><<<tileW::ValidRow, 1, 1>>>(
                //                                 tScale[x].data(),
                //                                 tNewMax[x].data(),
                //                                 tNewSum[x].data(),
                //                                 tExpW[x][3].data(),
                //                                 tW[x][3].data(),
                //                                 tMax[x].data(),
                //                                 tSum[x].data(),
                //                                 scale);
                new_max_manual<tileW, tileMax><<<tileW::ValidRow, 1, 1>>>(
                    tScale[x].data(),
                    tNewMax[x].data(),
                    tW[x][3].data(),
                    tMax[x].data(),
                    scale
                );
                src_exp_manual<tileW, tileW_cast, tileMax><<<tileW::ValidRow, tileW::ValidCol, 1>>>(tExpW[x][3].data(), tW[x][3].data(), tNewMax[x].data(), scale);
                new_sum_manual<tileW_cast, tileSum, tileScale><<<tileSum::ValidRow, 1, 1>>>(tNewSum[x].data(), tExpW[x][3].data(), tSum[x].data(), tScale[x].data());
            }

            //P3 * V3 = PV3
            gV = gIterV(j+3, 0);
            TCOPYIN(tV[3], gV);

            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                TCVT_DN2NZ(tW_left[x][3], tExpW[x][3]);
                MATMUL(tPV_out, tW_left[x][3], tV[3]);
                TCVT(tPV[x], tPV_out);
            }

            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                tMax[x] = tNewMax[x];
                tSum[x] = tNewSum[x];
            }        

        tileO_cast tO_cast[Xdim];
        #pragma clang loop unroll(full)
        for (int x = 0; x < Xdim; ++x) {
            normalize_with_last_update<tileO_cast, tileO, tileSum, tileScale><<<tileO::ValidRow, tileO::ValidCol, 1>>>(tO_cast[x].data(), tO[x].data(), tPV[x].data(), tScale[x].data(), tSum[x].data());
        }
        // 写回全局内存

        #pragma clang loop unroll(full)
        for (int x = 0; x < Xdim; ++x) {
            auto dstO = gIterO(i+x, 0);
            TCOPYOUT(dstO, tO_cast[x]);
        }

    }
}