template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk, int scaleD = qD>
void flash_attention_2d_unroll_pto(dtype* out_ptr, dtype* q_ptr, dtype* k_ptr, dtype* v_ptr) {
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
    using tileW_cast = Tile<Location::Vec, typename tileW_type<dtype>::DType, kTm, kTk, BLayout::ColMajor>;
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

    const float scale = 1.0f / sqrt((float)scaleD);
    const int Qb = (Sq + kTm - 1) / kTm;
    const int Kb = (Skv + kTk - 1) / kTk;

    #ifdef _2D_UNROLL_PTO
    static_assert(Qb%Xdim==0, "Qb needs to be a multiple of Xdim");
    static_assert(Kb%Ydim==0, "Kb needs to be a multiple of Ydim");
    static_assert(type_traits<dtype>::bits == 4 || type_traits<typename tileW_type<dtype>::DType>::bits == type_traits<dtype>::bits, "when dtype=fp8 or fp16 or fp32, tileW_cast dtype must the same");
    #endif

    for (int i = 0; i < Qb; i+=Xdim) {

        tileQ tQ[Xdim];

        #pragma clang loop unroll(full)
        for(int x=0;x<Xdim;x++){
            auto gQ = gIterQ(i+x,0);
            TLOAD(tQ[x], gQ);
        }

        tileMax tMax[Xdim];
        #pragma clang loop unroll(full)
        for(int x=0;x<Xdim;x++){
            TEXPANDS(tMax[x], -1e30f);
        }

        tileSum tSum[Xdim];
        #pragma clang loop unroll(full)
        for(int x=0;x<Xdim;x++){
            TEXPANDS(tSum[x], 0);
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
                TLOAD(tK[y], gK);
            }

            tileW tW[Xdim][Ydim];
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                #pragma clang loop unroll(full)
                for(int y=0;y<Ydim;y++){
                    tileW_out tW_out;
                    TMATMUL(tW_out, tQ[x], tK[y]);
                    // Nz -> ColMajor
                    TCVT(tW[x][y], tW_out);
                    TMULS(tW[x][y], tW[x][y], scale);
                }
            }

            tileMax tNewMax[Xdim];
            tileSum tNewSum[Xdim];

            tileW_cast tExpW[Xdim][Ydim];
            
            tileMax tLocalMax[Xdim][Ydim];
            tileSum tLocalSum[Xdim][Ydim];
            tileSum tScaledOldSum[Xdim];

            //softmax
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                //new max
                #pragma clang loop unroll(full)
                for(int y=0;y<Ydim;y++){
                    TCOLMAX(tLocalMax[x][y], tW[x][y]);
                }
                
                #if Ydim == 1
                    TMAX(tNewMax[x], tMax[x], tLocalMax[x][0]);
                #elif Ydim == 2
                    tileMax tLocalMax01;
                    TMAX(tLocalMax01, tLocalMax[x][0], tLocalMax[x][1]);
                    TMAX(tNewMax[x], tMax[x], tLocalMax01);
                #elif Ydim == 4
                    tileMax tLocalMax01;
                    tileMax tLocalMax23;
                    tileMax tLocalMax0123;
                    TMAX(tLocalMax01, tLocalMax[x][0], tLocalMax[x][1]);
                    TMAX(tLocalMax23, tLocalMax[x][2], tLocalMax[x][3]);
                    TMAX(tLocalMax0123, tLocalMax01, tLocalMax23);
                    TMAX(tNewMax[x], tMax[x], tLocalMax0123);
                #else
                    #error "Not Supprot Ydim != 1 or 2 or 4.";
                #endif

                //rescaling factor
                TSUB(tScale[x], tMax[x], tNewMax[x]);
                TEXP(tScale[x], tScale[x]);

                //new sum
                TMUL(tScaledOldSum[x], tSum[x], tScale[x]);

                #pragma clang loop unroll(full)
                for(int y=0;y<Ydim;y++){
                    TCOLEXPANDSUB(tW[x][y], tW[x][y], tNewMax[x]);
                    TEXP(tW[x][y], tW[x][y]);
                    TCOLSUM(tLocalSum[x][y], tW[x][y]);
                }

                #if Ydim == 1
                    TADD(tNewSum[x], tScaledOldSum[x], tLocalSum[x][0]);
                #elif Ydim == 2
                    tileSum tLocalSum01;
                    TADD(tLocalSum01, tLocalSum[x][0], tLocalSum[x][1]);
                    TADD(tNewSum[x], tScaledOldSum[x], tLocalSum01);
                #elif Ydim == 4
                    tileSum tLocalSum01;
                    tileSum tLocalSum23;
                    tileSum tLocalSum0123;
                    TADD(tLocalSum01, tLocalSum[x][0], tLocalSum[x][1]);
                    TADD(tLocalSum23, tLocalSum[x][2], tLocalSum[x][3]);
                    TADD(tLocalSum0123, tLocalSum01, tLocalSum23);
                    TADD(tNewSum[x], tScaledOldSum[x], tLocalSum0123);
                #else
                    #error "Not Supprot Ydim != 1 or 2 or 4.";
                #endif

                //src exp
                #pragma clang loop unroll(full)
                for(int y=0;y<Ydim;y++){
                    TCVT(tExpW[x][y], tW[x][y]);
                }
            }

            tileV tV[Ydim];

            #pragma clang loop unroll(full)
            for(int y=0;y<Ydim;y++){
                auto gV = gIterV(j+y, 0);
                TLOAD(tV[y], gV);
            }

            // ColMajor -> Nz
            // 计算当前块的加权输出: O_j = W * V
            tileW_left tW_left[Xdim][Ydim];
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                #pragma clang loop unroll(full)
                for(int y=0;y<Ydim;y++){
                    TMOV(tW_left[x][y], tExpW[x][y]);
                    if(y==0){
                        TMATMUL(tPV_out, tW_left[x][y], tV[y]);
                    }else{
                        TMATMUL_ACC(tPV_out, tW_left[x][y], tV[y]);
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
                    TCOLEXPANDMUL(tO[x], tO[x], tScale[x]);
                    TADD(tO[x], tO[x], tPV[x]);
                }
            }

            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                tMax[x] = tNewMax[x];
                tSum[x] = tNewSum[x];
            }
        }

        tileSum tInvSum[Xdim];
        tileO_cast tO_cast[Xdim];

        #pragma clang loop unroll(full)
        for (int x = 0; x < Xdim; ++x) {
            TRECIP(tInvSum[x], tSum[x]);
            TCOLEXPANDMUL(tO[x], tO[x], tInvSum[x]);
            TCVT(tO_cast[x], tO[x]);
            auto dstO = gIterO(i+x, 0);
            TSTORE(dstO, tO_cast[x]);
        }
    }
}
