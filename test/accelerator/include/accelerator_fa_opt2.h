template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk>
void flash_attention_opt2(dtype* out_ptr, dtype* q_ptr, dtype* k_ptr, dtype* v_ptr) {
    // 全局张量形状
    using gmQ = global_tensor<dtype, RowMajor<Sq, qD>>;  // Q: [S×qD]
    using gmK = global_tensor<dtype, ColMajor<qD, Skv>>;  // K: [qD×S]
    using gmV = global_tensor<dtype, RowMajor<Skv, vD>>;  // V: [S×vD]
    using gmO = global_tensor<dtype, RowMajor<Sq, vD>>;  // O: [SxvD]
    // tile 寄存器形状
    using tileQ      = TileLeft<dtype, kTm, (qD==192? 256:qD), kTm, qD>;       // [kTm×qD]
    using tileK      = TileRight<dtype, (qD==192? 256:qD), kTk, qD, kTk>;      // [vD×kTk]
    using tileW_out  = TileAcc<float, kTm, kTk>;      // [kTm×kTk]
    using tileW      = Tile<Location::Vec, float, kTm, kTk, BLayout::ColMajor>; 
    using tileW_left = TileLeft<dtype, kTm, kTk>; 

    using tileO_out  = TileAcc<float, kTm, vD>;
    using tileO      = Tile<Location::Vec, float, kTm, vD, BLayout::RowMajor>; // [kTm×vD]

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
        TCVT(tW, tW_out);

        tileMax tNewMax;
        tileSum tNewSum;
        tileW tExpW;
        flashsoftmax_dn_sout<tileW, tileO, tileMax, tileSum, tileScale>(tRescaleO, tNewMax, tNewSum, tExpW, tW, tO, tMax, tSum, scale);

        // ColMajor -> Nz
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

        Tile<Location::Vec, dtype, kTm, vD, BLayout::RowMajor> tO_cast;
        TCAST(tO_cast, tO);
        // 写回全局内存
        auto dstO = gIterO(i, 0);
        TCOPYOUT(dstO, tO_cast);
    }
}