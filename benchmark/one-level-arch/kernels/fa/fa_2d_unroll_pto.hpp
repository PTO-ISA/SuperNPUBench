template <class dtype>
struct tileW_type {
    using DType = dtype;
};

template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk, int scaleD = qD>
void flash_attention_2d_unroll_pto(dtype* out_ptr, dtype* q_ptr, dtype* k_ptr, dtype* v_ptr) {
    // 计算语义:
    //   O = softmax((Q * K^T) / sqrt(scaleD)) * V
    //   Q: [Sq, qD], K: [Skv, qD], V: [Skv, vD], O: [Sq, vD]
    //
    // 该实现按 Q 维度切成 Qb 个 kTm 行块，按 K/V 的 sequence 维度切成 Kb 个 kTk 列块。
    // Xdim 表示一次 unroll 处理多少个 Q block，Ydim 表示一次 unroll 处理多少个 K/V block。
    // 对每个 Q block 使用 online softmax: 逐个 K/V block 更新 row max、row sum 和 partial output。

    // 全局张量形状和内存布局。
    // gmK/gmO 都使用 RowMajor，和测试侧线性 buffer 的 [Skv,qD]/[Sq,vD] 布局一致。
    // K load 成 [kTk,qD] 的 Right tile，作为 TMATMUL 右操作数时等价参与 Q * K^T。
    using gmQ = global_tensor<dtype, RowMajor<Sq, qD>>;   // Q global: [Sq, qD]
    using gmK = global_tensor<dtype, RowMajor<Skv, qD>>;  // K global: [Skv, qD]
    using gmV = global_tensor<dtype, RowMajor<Skv, vD>>;  // V global: [Skv, vD]
    using gmO = global_tensor<dtype, RowMajor<Sq, vD>>;   // O global: [Sq, vD]

    // tile 寄存器形状。
    //
    // Q/K:
    //   tileQ: L0A/Left tile, logical shape [kTm, qD]。
    //   tileK: L0B/Right tile, logical loaded shape [kTk, qD] from row-major K.
    //          It is consumed as the transposed right operand for Q * K^T.
    //   qD==192 时物理 qD 维 pad 到 256，逻辑有效列仍然是 qD。
    //
    // QK score:
    //   tileW_out: Acc tile, TMATMUL 输出，logical shape [kTm, kTk]，dtype=float。
    //   tileW: Vec tile, ACCCVT 后的 score tile，ColMajor，logical shape [kTm, kTk]。
    //   tileW_cast: softmax 后 score cast 成 TMATMUL 左操作数需要的 dtype。
    //   tileW_left: L0A/Left tile，用于后续 softmax(score) * V。
    //
    // Output:
    //   tileO_out: Acc tile, PV 矩阵乘输出，logical shape [kTm, vD]。
    //   tileO: Vec tile，online softmax 的 float partial output，logical shape [kTm, vD]。
    //   tileO_cast: 写回前从 float cast 到输出 dtype 的 tile，logical shape [kTm, vD]。
    //
    // V:
    //   tileV: L0B/Right tile，logical shape [kTk, vD]。
    //
    // Softmax row state:
    //   tileMax/tileSum/tileScale 的逻辑 shape 都是 [kTm, 1]。
    //   这里第二维物理写成 8，valid col 为 1，用于满足 Vec tile 对齐/active size 要求。
    using tileQ      = TileLeft<dtype, kTm, (qD==192? 256:qD), kTm, qD>;
    using tileK      = TileRight<dtype, kTk, (qD==192? 256:qD), kTk, qD>;
    using tileW_out  = TileAcc<float, kTm, kTk>;
    using tileW      = Tile<Location::Vec, float, kTm, kTk, BLayout::ColMajor>;
    using tileW_cast = Tile<Location::Vec, typename tileW_type<dtype>::DType, kTm, kTk, BLayout::ColMajor>;
    using tileW_left = TileLeft<dtype, kTm, kTk>; 

    using tileO_out  = TileAcc<float, kTm, vD>;
    using tileO      = Tile<Location::Vec, float, kTm, vD, BLayout::ColMajor>;
    using tileO_cast = Tile<Location::Vec, dtype, kTm, vD, BLayout::ColMajor>;

    using tileV      = TileRight<dtype, kTk, vD>;
    using tileMax    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>;
    using tileSum    = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>;
    using tileScale  = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor, kTm, 1>;

    // 全局迭代器。iterator 的 block 坐标与对应 tile 的 logical shape 匹配:
    //   gIterQ(q_block, 0) -> Q[q_block*kTm : (q_block+1)*kTm, 0:qD]
    //   gIterK(k_block, 0) -> K[k_block*kTk : (k_block+1)*kTk, 0:qD]
    //   gIterV(k_block, 0) -> V[k_block*kTk : (k_block+1)*kTk, 0:vD]
    //   gIterO(q_block, 0) -> O[q_block*kTm : (q_block+1)*kTm, 0:vD]
    using itQ = global_iterator<gmQ, tileQ>;
    using itK = global_iterator<gmK, tileK>;
    using itV = global_iterator<gmV, tileV>;
    using itO = global_iterator<gmO, tileO>;

    itQ gIterQ(q_ptr);
    itK gIterK(k_ptr);
    itV gIterV(v_ptr);
    itO gIterO(out_ptr);

    // FlashAttention score scale。默认 scaleD=qD，与标准 attention 的 1/sqrt(qD) 一致。
    const float scale = 1.0f / sqrt((float)scaleD);
    const int Qb = (Sq + kTm - 1) / kTm;
    const int Kb = (Skv + kTk - 1) / kTk;

    #ifdef _2D_UNROLL_PTO
    // 当前实现没有为尾块做 mask/pad 分支，因此要求 block 数能被 unroll 因子整除。
    static_assert(Qb%Xdim==0, "Qb needs to be a multiple of Xdim");
    static_assert(Kb%Ydim==0, "Kb needs to be a multiple of Ydim");
    static_assert(type_traits<dtype>::bits == 4 || type_traits<typename tileW_type<dtype>::DType>::bits == type_traits<dtype>::bits, "when dtype=fp8 or fp16 or fp32, tileW_cast dtype must the same");
    #endif

    // 外层遍历 Q block。每轮并行/展开处理 Xdim 个 Q tile。
    for (int i = 0; i < Qb; i+=Xdim) {

        tileQ tQ[Xdim];

        // TLOAD:
        //   global Q block -> tileQ
        //   before: gQ points to Q block [kTm, qD]
        //   after : tQ[x] is Left tile, logical shape [kTm, qD]
        #pragma clang loop unroll(full)
        for(int x=0;x<Xdim;x++){
            auto gQ = gIterQ(i+x,0);
            TLOAD(tQ[x], gQ);
        }

        tileMax tMax[Xdim];
        // 初始化 online softmax 的历史行最大值 m_old。
        // tMax[x]: [kTm, 1]，每行初值为 -inf 的近似值。
        #pragma clang loop unroll(full)
        for(int x=0;x<Xdim;x++){
            TEXPANDS(tMax[x], -1e30f);
        }

        tileSum tSum[Xdim];
        // 初始化 online softmax 的历史行归一化因子 l_old。
        // tSum[x]: [kTm, 1]，每行初值为 0。
        #pragma clang loop unroll(full)
        for(int x=0;x<Xdim;x++){
            TEXPANDS(tSum[x], 0);
        }

        // tPV_out 是每个 Q block 计算 PV 时复用的 Acc tile。
        // tO[x] 保存截至当前 K/V block 的未最终归一化输出:
        //   tO = sum_j exp(score_j - m_current) * V_j
        // tPV[x] 保存当前 K/V unroll 组贡献:
        //   tPV = exp(score_current - m_current) * V_current
        // tScale[x] = exp(m_old - m_new)，用于把历史 tO/tSum 缩放到新 max 基准。
        tileO tO[Xdim], tPV[Xdim];
        tileScale tScale[Xdim];

        // 内层遍历 K/V block。每轮展开处理 Ydim 个 K/V tile。
        #pragma clang loop unroll(full)
        for (int j = 0; j < Kb; j+=Ydim) {

            tileK tK[Ydim];

            // TLOAD:
            //   global K block -> tileK
            //   before: gK points to row-major K block [kTk, qD]
            //   after : tK[y] is Right tile, logical loaded shape [kTk, qD]
            #pragma clang loop unroll(full)
            for(int y=0;y<Ydim;y++){
                auto gK = gIterK(j+y, 0);
                TLOAD(tK[y], gK);
            }

            tileW tW[Xdim][Ydim];
            // QK score:
            //   TMATMUL: tileQ [kTm, qD] x tileK^T [qD, kTk] -> tileW_out [kTm, kTk] Acc(float)
            //   ACCCVT : Acc/NZ-like layout -> Vec ColMajor tileW [kTm, kTk]
            //   TMULS  : tileW *= 1/sqrt(scaleD)
            //
            // After this block:
            //   tW[x][y] holds scaled attention logits for Q block (i+x) and K block (j+y).
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                #pragma clang loop unroll(full)
                for(int y=0;y<Ydim;y++){
                    tileW_out tW_out;
                    TMATMUL(tW_out, tQ[x], tK[y]);
                    // Acc/NZ -> Vec ColMajor
                    ACCCVT(tW[x][y], tW_out);
                    TMULS(tW[x][y], tW[x][y], scale);
                }
            }

            tileMax tNewMax[Xdim];
            tileSum tNewSum[Xdim];

            tileW_cast tExpW[Xdim][Ydim];
            
            tileMax tLocalMax[Xdim][Ydim];
            tileSum tLocalSum[Xdim][Ydim];
            tileSum tScaledOldSum[Xdim];

            // Online softmax for current Ydim K blocks.
            // 数学形式按行计算:
            //   m_new = max(m_old, rowmax(score_block_0), ..., rowmax(score_block_Y-1))
            //   scale_old = exp(m_old - m_new)
            //   l_new = l_old * scale_old + sum_y rowsum(exp(score_y - m_new))
            //   p_y = exp(score_y - m_new)
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                // TROWMAX:
                //   input : tW[x][y] [kTm, kTk]
                //   output: tLocalMax[x][y] [kTm, 1], each row is max over kTk columns.
                #pragma clang loop unroll(full)
                for(int y=0;y<Ydim;y++){
                    TROWMAX(tLocalMax[x][y], tW[x][y]);
                }
                
                // Reduce Ydim local max tiles together, then compare with historical tMax.
                // tNewMax[x] remains [kTm, 1].
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

                // Rescale factor for old softmax state:
                //   tScale = exp(tMax - tNewMax), shape [kTm, 1].
                // 该因子会同时作用到旧 tSum 和旧 tO。
                TSUB(tScale[x], tMax[x], tNewMax[x]);
                TEXP(tScale[x], tScale[x]);

                // Scale historical denominator:
                //   tScaledOldSum = tSum * exp(m_old - m_new), shape [kTm, 1].
                TMUL(tScaledOldSum[x], tSum[x], tScale[x]);

                // Convert logits to unnormalized probabilities under the new max:
                //   TROWEXPANDSUB broadcasts tNewMax [kTm,1] along columns and computes
                //     tW = tW - tNewMax
                //   TEXP computes exp(tW) elementwise, so tW now stores p_y.
                //   TROWSUM reduces p_y over columns to [kTm,1].
                #pragma clang loop unroll(full)
                for(int y=0;y<Ydim;y++){
                    TROWEXPANDSUB(tW[x][y], tW[x][y], tNewMax[x]);
                    TEXP(tW[x][y], tW[x][y]);
                    TROWSUM(tLocalSum[x][y], tW[x][y]);
                }

                // Combine the new Ydim local sums with scaled historical sum.
                // tNewSum[x] is the updated denominator l_new, shape [kTm,1].
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

                // Cast exp(score - m_new) to the dtype required by the next TMATMUL.
                // For fp16/fp8 this usually narrows from float tileW to tileW_type<dtype>::DType.
                // tExpW[x][y] still has logical shape [kTm, kTk].
                #pragma clang loop unroll(full)
                for(int y=0;y<Ydim;y++){
                    TCVT(tExpW[x][y], tW[x][y]);
                }
            }

            tileV tV[Ydim];

            // TLOAD:
            //   global V block -> tileV
            //   before: gV points to V block [kTk, vD]
            //   after : tV[y] is Right tile, logical shape [kTk, vD]
            #pragma clang loop unroll(full)
            for(int y=0;y<Ydim;y++){
                auto gV = gIterV(j+y, 0);
                TLOAD(tV[y], gV);
            }

            // Compute current weighted value:
            //   TCVT      : tExpW Vec ColMajor [kTm,kTk] -> tileW_left Left [kTm,kTk]
            //   TMATMUL   : first y,  tPV_out = p_y * V_y
            //   TMATMUL_ACC: next y,  tPV_out += p_y * V_y
            //   ACCCVT    : Acc [kTm,vD] -> Vec float tPV[x] [kTm,vD]
            //
            // After this block:
            //   tPV[x] = sum_y exp(score_y - m_new) * V_y, for current Ydim K/V blocks.
            tileW_left tW_left[Xdim][Ydim];
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                tileO_out tPV_out;
                #pragma clang loop unroll(full)
                for(int y=0;y<Ydim;y++){
                    TCVT(tW_left[x][y], tExpW[x][y]);
                    if(y==0){
                        TMATMUL(tPV_out, tW_left[x][y], tV[y]);
                    }else{
                        TMATMUL_ACC(tPV_out, tW_left[x][y], tV[y]);
                    }
                }
                ACCCVT(tPV[x], tPV_out);
            }

            // Update online output numerator.
            //
            // First K/V block group:
            //   tO = tPV
            // Later K/V block groups:
            //   tO = tO * exp(m_old - m_new) + tPV
            // TROWEXPANDMUL broadcasts tScale [kTm,1] over vD columns.
            if(j==0){
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    tO[x] = tPV[x];
                }
            }else{
                #pragma clang loop unroll(full)
                for(int x=0;x<Xdim;x++){
                    TROWEXPANDMUL(tO[x], tO[x], tScale[x]);
                    TADD(tO[x], tO[x], tPV[x]);
                }
            }

            // Commit current online softmax state for the next K/V block group.
            #pragma clang loop unroll(full)
            for(int x=0;x<Xdim;x++){
                tMax[x] = tNewMax[x];
                tSum[x] = tNewSum[x];
            }
        }

        tileSum tInvSum[Xdim];
        tileO_cast tO_cast[Xdim];

        // Final normalization and store:
        //   tInvSum = 1 / l_final, shape [kTm,1]
        //   tO      = numerator * tInvSum, broadcasting tInvSum over vD columns
        //   TCVT    = float output tile -> dtype output tile
        //   TSTORE  = store O block [kTm, vD] back to global output.
        #pragma clang loop unroll(full)
        for (int x = 0; x < Xdim; ++x) {
            TRECIP(tInvSum[x], tSum[x]);
            TROWEXPANDMUL(tO[x], tO[x], tInvSum[x]);
            TCVT(tO_cast[x], tO[x]);
            auto dstO = gIterO(i+x, 0);
            TSTORE(dstO, tO_cast[x]);
        }
    }
}
