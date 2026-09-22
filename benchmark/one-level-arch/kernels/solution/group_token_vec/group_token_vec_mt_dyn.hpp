#ifndef GROUP_TOKEN_VEC_MT_DYN_HPP
#define GROUP_TOKEN_VEC_MT_DYN_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>

// ============================================================================
// MoE Token Grouping — Multi-thread Vector (Tile) variant, runtime DYNAMIC
// SHAPE version
//
// 基于 group_token_vec_mt.hpp 的动态 shape 版: 三阶段算子语义与分片模型
// 完全一致, 唯一区别是 bs/topK/expertPerRank/expertPerPod/superPodNum
// 编译期不可知, 运行期由 tiling 指针传入, 全部切分参数运行期计算。
//
//   tiling = {bs, topK, expertPerRank, expertPerPod, superPodNum}   (int64)
//   expertNum = expertPerPod * superPodNum;  topkEleNum = bs * topK
//
// 设计对照 quant/dynamic_mx_quant_*_dyn (动态入口范式):
//   · physical tile 形状仍编译期锁定 (Phase1/3a 每 PE 4×kTileN 行块,
//     TROWMIN 目的 tile 物理 4×1 契约不变); 行有效域 ValidRow=-1 运行期
//     传入 (尾块 vr), 列维保持编译期 —— TEPL B.DIM 立即数约束。
//   · global_iterator 依赖编译期 RowStride, 全部改为运行时构造的
//     global_tensor<RowMajor<-1,-1>> (ctor 传运行时 rows/cols), 基址按
//     运行时维度手动计算。
//   · tile 路径 (TLOAD/TREMS/TROWMIN) 仅在 topK == kTileN 时启用
//     (tile 列宽与 topK 绑定的硬件约束); 其余 topK 走纯标量兜底
//     (直方图/散射本就是标量正确性路径, tile 只是 DMA 预取/归约加速)。
//   · 编译期定长栈数组 (dstPodLocal / counts / writePos) 改为 GM scratch
//     入参: podScratch 需 4 * superPodNum 个 uint32 (每 PE tid 切片),
//     sortScratch 需 2 * expertPerRank 个 uint32 (counts + writePos)。
//   · PE 分片/私有段尺寸全部运行时 ceil 计算:
//       kBsPerPE = (bs + 3) / 4;  每 token 循环按 tid stride 遍历 (原样);
//     专家域归约 (Phase1 reduce) 用运行时 ceil 分片 (前 rem 个 PE 多 1)。
//
// 其余与静态版相同的约定 (详见 group_token_vec_mt.hpp):
//   · 每个 TLOAD 结果经 tile op 消费, 结果经 TSTORE 出 tile 域, 标量
//     读回只打 GM (extract_vector_elt 后端崩溃规避);
//   · 每 PE tile 不相交: PE tid 拥有 16 行块内 [4*tid, 4*tid+4) 行;
//   · 跨 PE 交接由 mtBarrier 保护 (相位 1/2/3 与静态版一致)。
// ============================================================================

constexpr uint32_t kThreadsPerBlock = 4;
constexpr uint32_t kTileM = 16;   // rows per TMA block (4 rows per PE)
constexpr uint32_t kTileN = 16;   // tile 列宽; tile 路径仅 topK == kTileN

// ============================================================================
// Multi-PE barrier: volatile per-PE phase flags + compiler memory barrier,
// same convention as multi_thread/matmul RES_CHECK leader_ready spin.
// ============================================================================
static volatile uint32_t sPhaseDone[kThreadsPerBlock];

static inline void mtCompilerBarrier()
{
    __asm__ volatile("" : : : "memory");
}

static inline void mtBarrier(uint32_t phase)
{
    mtCompilerBarrier();
    sPhaseDone[get_thread_idx()] = phase;
    mtCompilerBarrier();
    for (uint32_t t = 0; t < kThreadsPerBlock; ++t) {
        while (sPhaseDone[t] < phase) {
        }
    }
    mtCompilerBarrier();
}

// ============================================================================
// Phase 1 (Tile + multi-thread, runtime shape): 每 PE 不相交行 TLOAD +
// 标量直方图。topK == kTileN 时走 tile 预取路径 (每 PE 4 行块 + 尾块 vr),
// 其余 topK 走纯标量 stride 兜底。专家域归约用运行时 ceil 分片。
// ============================================================================
static inline void calTokenPerExpertCnt_mt_tile_dyn(
    uint32_t *topkIndex,
    uint32_t *tokenPerExpertCnt,
    uint32_t *cntLocal,
    uint32_t expertNum,
    uint32_t bs,
    uint32_t topK)
{
    const uint32_t tid = get_thread_idx();

    uint32_t *myCnt = cntLocal + tid * expertNum;
    for (uint32_t i = 0; i < expertNum; i++) {
        myCnt[i] = 0;
    }

    if (topK == kTileN) {
        using TilePerPE = Tile<Location::Vec, uint32_t, 4, kTileN,
                               BLayout::RowMajor, -1, kTileN>;
        using GmPerPE = global_tensor<uint32_t, RowMajor<-1, -1>>;

        const uint32_t fullBlocks = bs / kTileM;
        for (uint32_t blk = 0; blk < fullBlocks; ++blk) {
            // rows [16*blk + 4*tid, +4): 完整块内每 PE 4 行恒有效
            const uint32_t r0 = blk * kTileM + tid * 4U;
            TilePerPE dataTile(static_cast<size_t>(4));
            GmPerPE gIn(topkIndex + r0 * topK, static_cast<int>(bs),
                        static_cast<int>(topK));
            TLOAD(dataTile, gIn);

            // scalar histogram over the same (disjoint) rows, reading GM
            for (uint32_t row = 0; row < 4; ++row) {
                uint32_t tokenId = r0 + row;
                uint32_t base = tokenId * topK;
                for (uint32_t col = 0; col < topK; ++col) {
                    uint32_t expertId = topkIndex[base + col];
                    if (expertId < expertNum) {
                        myCnt[expertId]++;
                    }
                }
            }
        }
        // 尾块: 行域 [fullBlocks*16, bs), PE tid 拥有 [4*tid, +4) 切片
        const uint32_t tailRows = bs - fullBlocks * kTileM;
        if (tailRows > 0) {
            const uint32_t r0 = fullBlocks * kTileM + tid * 4U;
            const uint32_t vr = (r0 < bs) ? ((bs - r0 < 4U) ? (bs - r0) : 4U) : 0U;
            if (vr > 0) {
                TilePerPE dataTile(static_cast<size_t>(vr));
                GmPerPE gIn(topkIndex + r0 * topK, static_cast<int>(bs),
                            static_cast<int>(topK));
                TLOAD(dataTile, gIn);

                for (uint32_t row = 0; row < vr; ++row) {
                    uint32_t tokenId = r0 + row;
                    uint32_t base = tokenId * topK;
                    for (uint32_t col = 0; col < topK; ++col) {
                        uint32_t expertId = topkIndex[base + col];
                        if (expertId < expertNum) {
                            myCnt[expertId]++;
                        }
                    }
                }
            }
        }
    } else {
        // topK != kTileN: 纯标量 stride 兜底 (tile 列宽与 topK 绑定)
        for (uint32_t i = tid; i < bs; i += kThreadsPerBlock) {
            uint32_t base = i * topK;
            for (uint32_t col = 0; col < topK; ++col) {
                uint32_t expertId = topkIndex[base + col];
                if (expertId < expertNum) {
                    myCnt[expertId]++;
                }
            }
        }
    }

    // Reduce: each PE writes its assigned expert range (运行时 ceil 分片)
    const uint32_t eseg = expertNum / kThreadsPerBlock;
    const uint32_t erem = expertNum % kThreadsPerBlock;
    const uint32_t ebegin = tid * eseg + (tid < erem ? tid : erem);
    const uint32_t elen = eseg + (tid < erem ? 1U : 0U);
    for (uint32_t i = 0; i < elen; i++) {
        uint32_t globalExpert = ebegin + i;
        uint32_t sum = 0;
        for (uint32_t t = 0; t < kThreadsPerBlock; t++) {
            sum += cntLocal[t * expertNum + globalExpert];
        }
        tokenPerExpertCnt[globalExpert] = sum;
    }
}

// ============================================================================
// Phase 2 (multi-thread, runtime shape): 标量散射到每 PE 私有段 (stride 模式)
// 与静态版逐行一致, 仅 kBsPerPE / per-PE 段 offset 改为运行时计算,
// dstPodLocal 固定栈数组改为 podScratch 的 per-PE 切片。
// ============================================================================
static inline void groupToken_mt_tile_dyn(
    uint32_t *topkIndex,
    uint32_t *perPegroupedIds,
    uint32_t *perPeSectionCnt,
    uint32_t *perPePodInfo,
    uint32_t *podScratch,
    uint32_t batchSize,
    uint32_t topk,
    uint32_t expertPerRank,
    uint32_t expertPerPod,
    uint32_t superPodNum)
{
    const uint32_t tid = get_thread_idx();
    const uint32_t kBsPerPE = (batchSize + kThreadsPerBlock - 1U) / kThreadsPerBlock;

    uint32_t *mySectionCnt = perPeSectionCnt + tid * expertPerRank;
    for (uint32_t i = 0; i < expertPerRank; i++) {
        mySectionCnt[i] = 0;
    }
    uint32_t *dstPodLocal = podScratch + tid * superPodNum;

    for (uint32_t i = tid; i < batchSize; i += kThreadsPerBlock) {
        uint32_t minLocalExpId = expertPerRank;
        for (uint32_t s = 0; s < superPodNum; s++) dstPodLocal[s] = 0;

        uint32_t base = i * topk;
        for (uint32_t col = 0; col < topk; ++col) {
            uint32_t expertId = topkIndex[base + col];
            uint32_t curLocalExpId = expertId % expertPerRank;
            if (curLocalExpId < minLocalExpId) {
                minLocalExpId = curLocalExpId;
            }
            uint32_t curDstPod = expertId / expertPerPod;
            if (curDstPod < superPodNum) {
                dstPodLocal[curDstPod] = 1;
            }
        }

        uint32_t idxInSection = mySectionCnt[minLocalExpId]++;
        uint32_t peOffset = minLocalExpId * kThreadsPerBlock * kBsPerPE
                          + tid * kBsPerPE + idxInSection;
        perPegroupedIds[peOffset] = i;

        uint32_t podPeOffset = minLocalExpId * kThreadsPerBlock * kBsPerPE * superPodNum
                             + tid * kBsPerPE * superPodNum
                             + idxInSection * superPodNum;
        for (uint32_t s = 0; s < superPodNum; s++) {
            perPePodInfo[podPeOffset + s] = dstPodLocal[s];
        }
    }
}

// ============================================================================
// Host-side merge (scalar, single-PE, runtime shape)
// ============================================================================
static inline void mergeGroupTokenResults_dyn(
    const uint32_t *perPegroupedIds,
    const uint32_t *perPeSectionCnt,
    const uint32_t *perPePodInfo,
    uint32_t *groupedTokenIds,
    uint32_t *tokenSuperPodInfo,
    uint32_t *expertSectionTokenCnt,
    uint32_t expertPerRank,
    uint32_t superPodNum,
    uint32_t batchSize)
{
    const uint32_t kBsPerPE = (batchSize + kThreadsPerBlock - 1U) / kThreadsPerBlock;

    for (uint32_t s = 0; s < expertPerRank; s++) {
        uint32_t globalIdx = 0;
        for (uint32_t t = 0; t < kThreadsPerBlock; t++) {
            uint32_t peCnt = perPeSectionCnt[t * expertPerRank + s];
            for (uint32_t i = 0; i < peCnt; i++) {
                uint32_t peOffset = s * kThreadsPerBlock * kBsPerPE
                                  + t * kBsPerPE + i;
                groupedTokenIds[s * batchSize + globalIdx] = perPegroupedIds[peOffset];

                uint32_t podPeOffset = s * kThreadsPerBlock * kBsPerPE * superPodNum
                                     + t * kBsPerPE * superPodNum
                                     + i * superPodNum;
                for (uint32_t j = 0; j < superPodNum; j++) {
                    tokenSuperPodInfo[s * batchSize * superPodNum + globalIdx * superPodNum + j]
                        = perPePodInfo[podPeOffset + j];
                }
                globalIdx++;
            }
        }
        expertSectionTokenCnt[s] = globalIdx;
    }
}

// ============================================================================
// Phase 3 (runtime shape): TROWMIN FloorFunc + PE0 counting sort
//
// Phase 3a: topK == kTileN 时走 tile 流水 (每 PE 4×kTileN 不相交块,
//   尾块 ValidRow 运行时 vr), 否则纯标量 stride 兜底。标量 GM 读回仅在
//   TSTORE 之后 (extract_vector_elt 契约)。Phase 3b 保持 PE0 标量,
//   counts/writePos 改 sortScratch GM (counts = sortScratch[0..],
//   writePos = sortScratch + expertPerRank)。
// ============================================================================
static inline void sortKernel_mt_tile_dyn(
    uint32_t *topkIndex,
    uint32_t *minLocalExpIds,
    uint32_t *sortedTokenIds,
    uint32_t *sectionStarts,
    uint32_t *sortScratch,
    uint32_t batchSize,
    uint32_t topk,
    uint32_t expertPerRank)
{
    const uint32_t tid = get_thread_idx();

    // Phase 3a: FloorFunc via tile ops on this PE's disjoint rows
    if (topk == kTileN) {
        using TilePerPE = Tile<Location::Vec, uint32_t, 4, kTileN,
                               BLayout::RowMajor, -1, kTileN>;
        // TROWMIN 目的 tile 必须物理单列 (PTO 契约: ValidCol==1 && Cols==1)
        using TileMinPE = Tile<Location::Vec, uint32_t, 4, 1, BLayout::RowMajor>;
        using GmPerPE = global_tensor<uint32_t, RowMajor<-1, -1>>;
        using GmMin = global_tensor<uint32_t, RowMajor<-1, -1>>;

        const uint32_t fullBlocks = batchSize / kTileM;
        for (uint32_t blk = 0; blk < fullBlocks; ++blk) {
            const uint32_t r0 = blk * kTileM + tid * 4U;
            TilePerPE tIn(static_cast<size_t>(4));
            TilePerPE tRem(static_cast<size_t>(4));
            TileMinPE tMin;
            GmPerPE gIn(topkIndex + r0 * topk, static_cast<int>(batchSize),
                        static_cast<int>(topk));
            GmMin gMin(minLocalExpIds + r0, static_cast<int>(batchSize), 1);
            TLOAD(tIn, gIn);
            TREMS(tRem, tIn, expertPerRank);   // local expert ids
            TROWMIN(tMin, tRem);               // per-token min
            TSTORE(gMin, tMin);                // -> minLocalExpIds[r0..]
        }
        const uint32_t tailRows = batchSize - fullBlocks * kTileM;
        if (tailRows > 0) {
            const uint32_t r0 = fullBlocks * kTileM + tid * 4U;
            const uint32_t vr = (r0 < batchSize)
                              ? ((batchSize - r0 < 4U) ? (batchSize - r0) : 4U) : 0U;
            if (vr > 0) {
                TilePerPE tIn(static_cast<size_t>(vr));
                TilePerPE tRem(static_cast<size_t>(vr));
                TileMinPE tMin;
                GmPerPE gIn(topkIndex + r0 * topk, static_cast<int>(batchSize),
                            static_cast<int>(topk));
                GmMin gMin(minLocalExpIds + r0, static_cast<int>(batchSize), 1);
                TLOAD(tIn, gIn);
                TREMS(tRem, tIn, expertPerRank);
                TROWMIN(tMin, tRem);
                TSTORE(gMin, tMin);
            }
        }
    } else {
        for (uint32_t i = tid; i < batchSize; i += kThreadsPerBlock) {
            uint32_t minLocal = expertPerRank;
            uint32_t base = i * topk;
            for (uint32_t col = 0; col < topk; ++col) {
                uint32_t local = topkIndex[base + col] % expertPerRank;
                if (local < minLocal) minLocal = local;
            }
            minLocalExpIds[i] = minLocal;
        }
    }

    // Phase 3b: Counting sort — only PE 0 (needs global coordination)
    if (tid == 0) {
        uint32_t *counts = sortScratch;
        uint32_t *writePos = sortScratch + expertPerRank;
        for (uint32_t i = 0; i < expertPerRank; i++) {
            counts[i] = 0;
        }
        for (uint32_t i = 0; i < batchSize; i++) {
            counts[minLocalExpIds[i]]++;
        }
        sectionStarts[0] = 0;
        for (uint32_t i = 0; i < expertPerRank; i++) {
            sectionStarts[i + 1] = sectionStarts[i] + counts[i];
        }
        for (uint32_t i = 0; i < expertPerRank; i++) {
            writePos[i] = sectionStarts[i];
        }
        for (uint32_t i = 0; i < batchSize; i++) {
            uint32_t section = minLocalExpIds[i];
            sortedTokenIds[writePos[section]++] = i;
        }
    }
}

// ============================================================================
// Entry point (multi-PE SPMD; called by every PE, runtime tiling)
//
//   tiling = {bs, topK, expertPerRank, expertPerPod, superPodNum}
//
// Cross-PE data hand-offs are separated by mtBarrier (与静态版一致):
//   Phase1 reduce  reads all PEs' cntLocal      -> barrier(1)
//   merge          reads all PEs' scatter state -> barrier(2); PE0 独占
//   Phase3b sort   reads all PEs' minLocalExpIds -> barrier(3) 末端汇合
// 注: tiling 违例由各 PE 同值判定并提前返回, 不触达任何栅栏, 无死锁。
// ============================================================================
static inline void runGroupTokenVecMTDyn(
    uint32_t *topkIndex,
    uint32_t *tokenPerExpertCnt,
    uint32_t *groupedTokenIds,
    uint32_t *tokenSuperPodInfo,
    uint32_t *expertSectionTokenCnt,
    uint32_t *sortedTokenIds,
    uint32_t *sectionStarts,
    uint32_t *cntLocal,
    uint32_t *perPegroupedIds,
    uint32_t *perPeSectionCnt,
    uint32_t *perPePodInfo,
    uint32_t *minLocalExpIds,
    uint32_t *podScratch,
    uint32_t *sortScratch,
    const int64_t *tiling)
{
    const uint32_t tid = get_thread_idx();

    const uint32_t bs = static_cast<uint32_t>(tiling[0]);
    const uint32_t topK = static_cast<uint32_t>(tiling[1]);
    const uint32_t expertPerRank = static_cast<uint32_t>(tiling[2]);
    const uint32_t expertPerPod = static_cast<uint32_t>(tiling[3]);
    const uint32_t superPodNum = static_cast<uint32_t>(tiling[4]);
    const uint32_t expertNum = expertPerPod * superPodNum;

    // 运行时契约: 全部 PE 读同一 tiling → 同值判定 → 同进同出, 无死锁
    if (tid >= kThreadsPerBlock) return;
    if (bs == 0 || topK == 0 || expertPerRank == 0 ||
        expertPerPod == 0 || superPodNum == 0) return;

    calTokenPerExpertCnt_mt_tile_dyn(topkIndex, tokenPerExpertCnt, cntLocal,
                                     expertNum, bs, topK);
    mtBarrier(1);   // all PEs' histograms complete before reduce consumers

    groupToken_mt_tile_dyn(topkIndex, perPegroupedIds, perPeSectionCnt, perPePodInfo,
                           podScratch, bs, topK, expertPerRank, expertPerPod, superPodNum);
    mtBarrier(2);   // all PEs' scatter sections complete before merge reads

    if (tid == 0) {
        mergeGroupTokenResults_dyn(perPegroupedIds, perPeSectionCnt, perPePodInfo,
                                   groupedTokenIds, tokenSuperPodInfo, expertSectionTokenCnt,
                                   expertPerRank, superPodNum, bs);
    }

    sortKernel_mt_tile_dyn(topkIndex, minLocalExpIds, sortedTokenIds, sectionStarts,
                           sortScratch, bs, topK, expertPerRank);
    mtBarrier(3);   // FloorFunc writes visible before any PE reads results
}

#endif // GROUP_TOKEN_VEC_MT_DYN_HPP
