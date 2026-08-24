#ifndef QLI_PTO_HPP
#define QLI_PTO_HPP

// =============================================================================
// qli_pto.hpp — Quant Lightning Indexer (PTO tile-op variant)
// =============================================================================
//
// 【算子功能】
//   QuantLightningIndexer (QLI) 是 SparseFlashAttention (SFA) 的前处理算子。
//   从全量 K/V 序列中选出最关键的 token 索引，供后续稀疏注意力使用。
//
// 【计算公式】（与参考实现 quant_lightning_indexer_v2 一致）
//   score[s1, s2] = scale_k[s2] * Σ_g ( W[s1, g] * scale_q[s1, g] * ReLU(QK[g, s1, s2]) )
//
//   展开为 7 个步骤:
//     Step 1: S = Q @ K^T                       → [g, Skv]   量化矩阵乘
//     Step 2: S = ReLU(S)                        → [g, Skv]   激活过滤负信号
//     Step 3: load scale_q, fuse W *= scale_q    → [g, 1]    反量化 scale 折入权重
//     Step 4: S *= W*scale_q (broadcast)         → [g, Skv]   权重加权
//     Step 5: out = [1]_g @ S (ReduceG)          → [1, Skv]   沿 head 维度求和
//     Step 6: out *= scale_k                     → [1, Skv]   反量化 scale_k 后乘
//     Step 7: indices = TopK(out)                → [topK]     选取关键索引
//
// 【与参考实现的一致性】
//   参考实现 (quant_lightning_indexer_v2) 的 golden 公式:
//     score[s1,s2] = k_scale[s2] * Σ_g ( w[s1,g] * q_scale[s1,g] * ReLU(QK[g,s1,s2]) )
//   - scale_q[s1, g] 折入 weights，在 g-reduction 之内、ReLU 之后
//   - scale_k[s2] 在 g-reduction 之外作为后乘因子
//   本实现严格遵循此两阶段应用顺序。
//
// 【数据布局 — BSND】
//   Q:        [B, S, N, D] → 展平为 [S*N, D]，同一 token 的 g 个 head 连续存放
//   K:        [Skv, D]      → Key 不分组，所有 head 共享
//   W:        [S, N]        → 与 query 前 2 维同形，float
//   scale_q:  [S, N]        → 与 W 同形，per-token per-head，float
//   scale_k:  [Skv]         → per-token（N2=1），float
//   scores:   [S, Skv]      → ReduceG 后的输出
//   indices:  [S, topK]     → TopK 索引输出
//
// 【数据类型】
//   dtype: Q/K 的数据类型，支持 __half / int8_t
//   W / scale_q / scale_k: 统一使用 float（与参考实现 arch35 一致）
//
// 【与 CANN 原版的对应关系】
//   CANN 代码位置: op_kernel/arch22/quant_lightning_indexer_v2_*.h
//
//   Step 1 (Q@K^T):  CANN ComputeQk (Mmad)              → QLI TMATMUL(tSacc, tQ, tK)
//   Step 2 (ReLU):   CANN FixpSToL1 reluPre=1 (Fixpipe) → QLI TMAX(tS, tS, tZero)
//   Step 3 (W*Sq):   CANN ProcessVec0 Mul(W, QScale)    → QLI TMUL(tWf, tWf, tSq)
//   Step 4 (W广播):  CANN Brcb(W*QScale)                → QLI TCOLEXPANDMUL(tS, tS, tWf)
//   Step 5 (ReduceG):CANN ComputeWs Mmad([1,g]×[g,Skv]) → QLI TCOLSUM(tPartial, tRed)
//   Step 6 (ScK):    CANN ProcessVec1 Mul(mmIn, kScale) → QLI TMUL(tPartial, tPartial, tSk)
//   Step 7 (TopK):   CANN SortAll+MergeSort+LD           → QLI qli_topk_npu (NPU tile op)
//
//   CANN ProcessVec0 中先计算 W *= Scale_Q（Mul(inWeights, inQScale)），再 Brcb 广播；
//   QLI 在 Step 3 用 TMUL 融合 W *= scale_q，在 Step 4 用 TCOLEXPANDMUL 广播。
//   CANN ProcessVec1 中 Mul(mmIn, kScale) 在 g-reduction 之后应用 scale_k；
//   QLI 在 Step 6 用 TMUL 在 TCOLSUM 之后应用 scale_k。
//
// 【与 FA 的区别】
//   FA:  O = softmax(Q@K^T / √d) @ V    — online softmax, P@V 矩阵乘, 输出 [S, vD]
//   QLI: out = TopK(scale_k * ReduceG(W*scale_q * ReLU(Q@K^T)))  — 无 softmax, 无 V
//
// 【工具链约束】
//   - TCOLEXPANDMUL 使用 TileOP-API 标准算子（src0/src1/dst 均为 float，
//     满足 dtype 断言）。早期用于绕过 dtype 断言 bug 的 _TEPL 内联汇编变体
//     （test/common/template_asm.h）已随 v5 迁移弃用。
//   - TCOLSUM 参考 reduction/reducesum_colvec_pto.hpp 的归约模式。
//   - TCVT 用于布局转换（ColMajor → RowMajor）以满足 TCOLSUM 的输入要求。
//
// 【tile 尺寸约束】
//   每个 tile 活跃尺寸须在 128B..8KB（DavinciOO active PE-local profile）。
//   当前配置: dtype=__half/int8_t, D=128, kTm=16, kTk=32
//     tileQ:   [16, 128] × 2B = 4KB (fp16) / 2KB (int8)  ✅
//     tileK:   [32, 128] × 2B = 8KB (fp16) / 4KB (int8)  ✅
//     tileS:   [16, 32]  × 4B = 2KB (float)               ✅
//     tileWf:  [16, 8]   × 4B = 512B (float) valid[16,1]  ✅
//     tileSq:  [16, 8]   × 4B = 512B (float) valid[16,1]  ✅
//     tileSk:  [1,  32]  × 4B = 128B (float)              ✅ (同 tileSum)
//     tileSum: [1,  32]  × 4B = 128B (float)              ✅ (刚好达到下限)
//
// 【已知简化（相对 CANN 原版）】
//   - 不支持 Causal Mask (CANN sparse_mode=3, rightDownCausal)
//   - 不支持 actual_seq_lengths (变长序列)
//   - 不支持 PageAttention (block_table 间接寻址)
//   - TopK 在 NPU tile op 实现（TROWMAX+TCMP+TSEL argmax 模拟）
//   - g=kTm（未分块循环），g>16 时需扩展为 g/kTm 内层循环
// =============================================================================

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <limits>

using namespace pto;

// -----------------------------------------------------------------------------
// qli_pto — QLI 核心计算（Step 1-6，NPU tile op 实现）
// -----------------------------------------------------------------------------
//
// 模板参数:
//   dtype   : Q/K 的数据类型（__half / int8_t）
//   Sq      : Query 序列长度 S
//   Skv     : Key 序列长度（通常 = Sq）
//   D       : Head 维度（固定 128）
//   g       : Head 数 N，kTm 必须等于 g
//   kTm     : tile M 维度（= g）
//   kTk     : tile K/N 维度（K block 大小，如 32）
//
// 输入:
//   q_ptr       : Q [Sq*g, D]       BSND 布局，dtype
//   k_ptr       : K [Skv, D]        所有 head 共享，dtype
//   w_ptr       : W [Sq, g]         BSND weight，float
//   scale_q_ptr : scale_q [Sq, g]   per-token per-head 反量化 scale，float
//   scale_k_ptr : scale_k [Skv]     per-token 反量化 scale，float
//
// 输出:
//   scores_ptr  : scores [Sq, Skv]  ReduceG + scale_k 后的 score 矩阵，float
//
// 外部循环结构:
//   for i in 0..Sq:           // 逐 token（外层 S 维度）
//     for j in 0..Kb:         // 逐 K-block（外层 Skv 维度，步长 kTk）
//       load K_j [kTk, D]     // K 只加载一次，所有 gi 共享
//       load scale_k_j        // scale_k 只加载一次
//       tSum = 0              // 片上 G 维累加器
//       for gi in 0..Gb:      // 逐 G-block（内层 g 维度，步长 kTm）
//         load Q_gi [kTm, D]  // 加载该 token 第 gi 组 head
//         load W_gi [kTm, 1]
//         load Sq_gi [kTm, 1]
//         W *= scale_q        // 融合：TMUL(tWf, tWf, tSq)
//         Step 1: Q@K^T       // TMATMUL → [kTm, kTk]
//         Step 2: ReLU        // TMAX
//         Step 3: skipped (W*scale_q already fused)
//         Step 4: *= W*scale_q // TCOLEXPANDMUL (broadcast)
//         Step 5: ReduceG     // TCOLSUM → [1, kTk]
//         if gi==0: tSum = tPartial
//         else:      tSum += tPartial  // TADD 累加跨 G-block
//       Step 6: tSum *= scale_k // TMUL
//       store scores[i, j]    // TSTORE [1, kTk]
// -----------------------------------------------------------------------------
template <typename dtype, int Sq, int Skv, int D, int g, int kTm, int kTk>
void qli_pto(float* scores_ptr,
             dtype* q_ptr, dtype* k_ptr,
             float* w_ptr,
             float* scale_q_ptr,
             float* scale_k_ptr)
{
    constexpr int Qb = Sq;              // S 维度 block 数（每 token 一个 block）
    constexpr int Kb = Skv / kTk;       // K 维度 block 数
    constexpr int Gb = g / kTm;         // G 维度 block 数（g 须为 kTm 的倍数）
    static_assert(g % kTm == 0, "g must be multiple of kTm for G-blocking");

    // ---- 全局张量形状和内存布局 ----
    using gmQ   = global_tensor<dtype,  RowMajor<Sq * g, D>>;
    using gmK   = global_tensor<dtype,  ColMajor<D, Skv>>;
    using gmOut = global_tensor<float,  RowMajor<Sq, Skv>>;

    // ---- tile 寄存器形状 ----
    using tileQ    = TileLeft<dtype, kTm, D>;
    using tileK    = TileRight<dtype, D, kTk>;
    using tileS    = Tile<Location::Vec, float, kTm, kTk, BLayout::RowMajor>;

    // W / scale_q: [kTm, 8] RowMajor valid=[kTm, 1], float
    using tileWf   = Tile<Location::Vec, float, kTm, 8, BLayout::RowMajor, kTm, 1>;
    using tileSq   = Tile<Location::Vec, float, kTm, 8, BLayout::RowMajor, kTm, 1>;

    // ReduceG / scale_k: [1, kTk] RowMajor, float
    using tileRed  = Tile<Location::Vec, float, kTm, kTk, BLayout::RowMajor>;
    using tileSum  = Tile<Location::Vec, float, 1, kTk, BLayout::RowMajor>;
    using tileSk   = Tile<Location::Vec, float, 1, kTk, BLayout::RowMajor>;

    // ---- 全局迭代器 ----
    using itQ   = global_iterator<gmQ,   tileQ>;
    using itK   = global_iterator<gmK,   tileK>;
    using itOut = global_iterator<gmOut, tileSum>;

    itQ   gIterQ(q_ptr);
    itK   gIterK(k_ptr);
    itOut gIterOut(scores_ptr);

    // ================================================================
    //  外层循环：逐 token 处理（S 维度）
    // ================================================================
    for (int i = 0; i < Qb; i++) {

        // ================================================================
        //  中层循环：逐 K-block（Skv 维度，步长 kTk）
        //    K 每个 (i,j) 只加载一次，所有 gi 共享
        // ================================================================
        for (int j = 0; j < Kb; j++) {

            // ---- TLOAD: 加载 K block [kTk, D] ----
            tileK tK;
            auto gK = gIterK(0, j);
            TLOAD(tK, gK);

            // ---- TLOAD: 加载 scale_k [1, kTk] ----
            tileSk tSk;
            {
                using gmSkLocal = global_tensor<float, RowMajor<1, Skv>>;
                using itSkLocal = global_iterator<gmSkLocal, tileSk>;
                itSkLocal gIterSk(scale_k_ptr);
                auto gSk = gIterSk(0, j);
                TLOAD(tSk, gSk);
            }

            // ---- 片上 G 维累加器 ----
            tileSum tSum;
            tileSum tZeroSum;
            TEXPANDS(tZeroSum, 0.0f);

            // ================================================================
            //  内层循环：逐 G-block（g 维度，步长 kTm）
            //    Q/W/scale_q 按 gi 分块加载
            //    必须 unroll(full)：不展开时 Linx TReg 分配 pass 崩溃
            //    已知限制：Gb>1 时 gfrun 输出非零元素减少（TReg 重用问题）
            // ================================================================
            #pragma clang loop unroll(full)
            for (int gi = 0; gi < Gb; gi++) {

                // ---- TLOAD: 加载 Q block [kTm, D] ----
                tileQ tQ;
                auto gQ = gIterQ(i * Gb + gi, 0);
                TLOAD(tQ, gQ);

                // ---- TLOAD: 加载 W [kTm, 1] (float) ----
                tileWf tWf;
                {
                    using gmWLocal = global_tensor<float, RowMajor<kTm, 1>>;
                    using itWLocal = global_iterator<gmWLocal, tileWf>;
                    itWLocal gIterW(w_ptr + (uint64_t)i * g + gi * kTm);
                    auto gW = gIterW(0, 0);
                    TLOAD(tWf, gW);
                }

                // ---- TLOAD: 加载 scale_q [kTm, 1] (float) ----
                tileSq tSq;
                {
                    using gmSqLocal = global_tensor<float, RowMajor<kTm, 1>>;
                    using itSqLocal = global_iterator<gmSqLocal, tileSq>;
                    itSqLocal gIterSq(scale_q_ptr + (uint64_t)i * g + gi * kTm);
                    auto gSq = gIterSq(0, 0);
                    TLOAD(tSq, gSq);
                }

                // ---- 融合: W *= scale_q ----
                TMUL(tWf, tWf, tSq);

                // ================================================================
                //  Step 1: S = Q @ K^T   (Cube: TMATMUL)
                // ================================================================
                tileS tS;
                TMATMUL(tS, tQ, tK);

                // ================================================================
                //  Step 2: S = ReLU(S)   (Vector: TEXPANDS + TMAX)
                // ================================================================
                tileS tZero;
                TEXPANDS(tZero, 0.0f);
                TMAX(tS, tS, tZero);

                // ================================================================
                //  Step 3+4: S *= (W * scale_q) (broadcast)   (Vector: TROWEXPANDMUL)
                // ================================================================
                TROWEXPANDMUL(tS, tS, tWf);

                // ================================================================
                //  Step 5: out = [1]_g @ S (ReduceG)   (Vector: TCVT + TCOLSUM)
                // ================================================================
                tileRed tRed;
                TCVT(tRed, tS);

                tileSum tPartial;
                TCOLSUM(tPartial, tRed);

                // ---- 累加跨 G-block 的 partial sum ----
                if (gi == 0) {
                    TADD(tSum, tZeroSum, tPartial);
                } else {
                    TADD(tSum, tSum, tPartial);
                }
            }

            // ================================================================
            //  Step 6: out *= scale_k   (Vector: TMUL)
            // ================================================================
            TMUL(tSum, tSum, tSk);

            // ---- TSTORE: 写回 score [1, kTk] 到全局内存 ----
            auto gOut = gIterOut(i, j);
            TSTORE(gOut, tSum);
        }
    }
}


// -----------------------------------------------------------------------------
// qli_topk_npu — TopK 索引选择（Step 7，TROWARGMAX tile op 实现）
// -----------------------------------------------------------------------------
//
// 【TROWARGMAX 版 — 使用 tile op 做 argmax，标量做消零】
//   TROWARGMAX 直接返回每行最大值的列索引（uint32），支持 FP32
//   B.IOT 同时绑定 src（输入 tile）和 dst（输出索引 tile），无 TSEL 编码问题
//   每轮：TLOAD scores → TROWARGMAX → TSTORE idx → 标量消零 → 重新 TLOAD
//
// 【算法（每 token，k 轮）】
//   1) TLOAD [1, Skv] scores 到 tile register
//   2) TROWARGMAX(idx_tile, S) → idx_tile 含 argmax 索引（uint32）
//   3) TSTORE idx 到 GM
//   4) 标量读 GM 获取索引值
//   5) 标量写 GM 把 scores[best] 设为 -1e30f（volatile uint32 写入，规避 -inf bug）
//   6) 重新 TLOAD（因为标量修改了 GM，tile register 数据已过期）
//
// 【性能】
//   ~35 blocks/轮 vs 标量 C++ ~259 blocks/轮 → ~7x 加速
//
// 【约束】
//   - Skv 须 8 的倍数
//   - TROWARGMAX 支持 FP32 dtype
//   - 标量消零用 volatile uint32_t 写入 0xF149F2CAu（-1e30f 位模式）
// -----------------------------------------------------------------------------
template <int Sq, int Skv, int topK, int TileN = 8>
void qli_topk_npu(float* scores_gm, int32_t* indices_gm) {
    static_assert(Skv % 8 == 0, "Skv must be multiple of 8");
    static_assert(topK <= Skv, "topK must be <= Skv");

    constexpr int MaxTileCol = 2048;

    if constexpr (Skv <= MaxTileCol) {
        using tile_s   = Tile<Location::Vec, float,    1, Skv, BLayout::RowMajor>;
        using tile_idx = Tile<Location::Vec, uint32_t, 1, 32,  BLayout::RowMajor, 1, 1>;

        using gm_s   = global_tensor<float,    RowMajor<Sq, Skv>>;
        using gm_idx = global_tensor<uint32_t, RowMajor<1, 1>>;
        using it_s   = global_iterator<gm_s, tile_s>;

        it_s s_iter(scores_gm);

        for (int i = 0; i < Sq; i++) {
            auto gin = s_iter(i, 0);
            tile_s S;
            TLOAD(S, gin);

            for (int k = 0; k < topK; k++) {
                tile_idx idx;
                TROWARGMAX(idx, S);

                uint32_t idx_val = 0;
                {
                    gm_idx idxGm(&idx_val);
                    TSTORE(idxGm, idx);
                }

                volatile int32_t* out = reinterpret_cast<volatile int32_t*>(indices_gm) + (uint64_t)i * topK + k;
                *out = static_cast<int32_t>(idx_val);

                volatile uint32_t* row_u32 = reinterpret_cast<volatile uint32_t*>(scores_gm + (uint64_t)i * Skv);
                row_u32[idx_val] = 0xF149F2CAu;

                TLOAD(S, gin);
            }
        }
    } else {
        constexpr int NumChunks = (Skv + MaxTileCol - 1) / MaxTileCol;

        using tile_chunk = Tile<Location::Vec, float,    1, MaxTileCol, BLayout::RowMajor>;
        using tile_max   = Tile<Location::Vec, float,    1, 32,         BLayout::RowMajor, 1, 1>;
        using tile_idx   = Tile<Location::Vec, uint32_t, 1, 32,         BLayout::RowMajor, 1, 1>;

        using gm_chunk  = global_tensor<float,    RowMajor<1, MaxTileCol>>;
        using gm_idx    = global_tensor<uint32_t, RowMajor<1, 1>>;
        using it_chunk  = global_iterator<gm_chunk, tile_chunk>;

        for (int i = 0; i < Sq; i++) {
            for (int k = 0; k < topK; k++) {
                float best_val = -1e30f;
                int best_chunk = 0;

                for (int c = 0; c < NumChunks; c++) {
                    constexpr int chunkSize = MaxTileCol;
                    int offset = c * chunkSize;
                    int validCol = (offset + chunkSize <= Skv) ? chunkSize : (Skv - offset);
                    if (validCol <= 0) break;

                    float* chunk_ptr = scores_gm + (uint64_t)i * Skv + offset;

                    it_chunk cIter(chunk_ptr);
                    auto gin = cIter(0, 0);
                    tile_chunk S_chunk;
                    TLOAD(S_chunk, gin);

                    tile_max mx;
                    TROWMAX(mx, S_chunk);

                    float chunk_max = 0;
                    {
                        gm_idx mxGm(reinterpret_cast<uint32_t*>(&chunk_max));
                        TSTORE(mxGm, mx);
                    }

                    if (chunk_max > best_val) {
                        best_val = chunk_max;
                        best_chunk = c;
                    }
                }

                int best_offset = best_chunk * MaxTileCol;
                int best_validCol = (best_offset + MaxTileCol <= Skv) ? MaxTileCol : (Skv - best_offset);
                float* best_ptr = scores_gm + (uint64_t)i * Skv + best_offset;

                it_chunk bestIter(best_ptr);
                auto gin = bestIter(0, 0);
                tile_chunk S_best;
                TLOAD(S_best, gin);

                tile_idx idx;
                TROWARGMAX(idx, S_best);

                uint32_t idx_val = 0;
                {
                    gm_idx idxGm(&idx_val);
                    TSTORE(idxGm, idx);
                }

                int32_t global_idx = static_cast<int32_t>(best_offset + idx_val);

                volatile int32_t* out = reinterpret_cast<volatile int32_t*>(indices_gm) + (uint64_t)i * topK + k;
                *out = global_idx;

                volatile uint32_t* row_u32 = reinterpret_cast<volatile uint32_t*>(scores_gm + (uint64_t)i * Skv);
                row_u32[global_idx] = 0xF149F2CAu;
            }
        }
    }
}

#endif
