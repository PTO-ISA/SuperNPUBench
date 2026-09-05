#ifndef GROUP_TOKEN_VEC_MT_HPP
#define GROUP_TOKEN_VEC_MT_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>

// ============================================================================
// MoE Token Grouping — Multi-thread Vector (Tile) variant
//
// 4-PE SPMD. Each PE uses get_thread_idx() for tile/stride parallelism.
//
// Tile usage rules (established against this toolchain, see notes below):
//   1. Every TLOAD result is consumed by tile ops (TREMS/TREM/TMIN/TROWMIN/
//      TROWSUM) and results leave the tile domain via TSTORE to GM; scalar
//      reads then hit GM, not tile registers. (Scalar tile-register reads
//      hit a backend "Cannot select: extract_vector_elt" crash.)
//   2. Per-PE tiles are disjoint: PE tid owns rows [4*tid, 4*tid+3] of every
//      16-row block (trowsum/tadd convention). No duplicated TLOAD traffic.
//   3. TCMP/TCMPS on u32 tiles are rejected by the assembler
//      ("Match Instruction Error"), so value-equality histograms are not
//      expressible as tile ops here; Phase 1 counts per expert via scalar
//      GM reads (after a TSTORE round-trip would be pure overhead).
//   4. Cross-PE hand-offs are guarded by mtBarrier.
//
// Phase 1: TLOAD per-PE tiles + scalar histogram on GM data (disjoint rows)
// Phase 2: scalar scatter into per-PE private sections (stride mode)
// Phase 3a: TLOAD per-PE tiles + TREMS + TROWMIN -> TSTORE minLocalExpIds
// Phase 3b: scalar counting sort (PE 0 only)
// ============================================================================

constexpr uint32_t kBS            = 512;
constexpr uint32_t kTopK          = 16;
constexpr uint32_t kExpertPerRank = 4;
constexpr uint32_t kRankPerPod    = 16;
constexpr uint32_t kSuperPodNum   = 2;
constexpr uint32_t kExpertPerPod  = kExpertPerRank * kRankPerPod;
constexpr uint32_t kExpertNum     = kExpertPerPod * kSuperPodNum;
constexpr uint32_t kTopKEleNum    = kBS * kTopK;

constexpr uint32_t kThreadsPerBlock = 4;
constexpr uint32_t kTileM = 16;   // rows per TMA block (4 rows per PE)
constexpr uint32_t kTileN = 16;   // = kTopK

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
// Phase 1 (Tile + multi-thread): disjoint per-PE TLOAD + scalar histogram
//
// PE tid owns rows [4*tid, 4*tid+3] of each 16-row block. Tiles are loaded
// once per PE without overlap (rule 2). Per-expert counting is inherently
// an indexed reduction with no tile-op equivalent (rule 3), so the count
// walks GM directly. The TLOAD still serves as the prefetch/ DMA path for
// this PE's rows and keeps the "vec variant" data path consistent.
// ============================================================================
static inline void calTokenPerExpertCnt_mt_tile(
    uint32_t *topkIndex,
    uint32_t *tokenPerExpertCnt,
    uint32_t *cntLocal,
    uint32_t expertNum,
    uint32_t topkEleNum)
{
    const uint32_t tid = get_thread_idx();

    uint32_t *myCnt = cntLocal + tid * expertNum;
    for (uint32_t i = 0; i < expertNum; i++) {
        myCnt[i] = 0;
    }

    using TilePerPE = Tile<Location::Vec, uint32_t, 4, kTileN, BLayout::RowMajor>;
    using GmPerPE = global_tensor<uint32_t, RowMajor<kBS, kTopK>>;
    using itPerPE = global_iterator<GmPerPE, TilePerPE>;
    itPerPE gIter(topkIndex);   // full tensor; PE tid prefetches its own slice

    TilePerPE dataTile;
    for (uint32_t blk = 0; blk < kBS / kTileM; ++blk) {
        // rows [16*blk + 4*tid, +4): row-tile index 4*blk + tid on the full
        // tensor. (A base pointer offset of tid*4 rows would only reach
        // tokens [4*tid, 4*tid + 128) across the 32 blocks.)
        auto src = gIter(4 * blk + tid, 0);
        TLOAD(dataTile, src);   // DMA in this PE's 4 rows

        // scalar histogram over the same (disjoint) rows, reading GM
        for (uint32_t row = 0; row < 4; ++row) {
            uint32_t tokenId = blk * kTileM + tid * 4 + row;
            uint32_t base = tokenId * kTopK;
            for (uint32_t col = 0; col < kTopK; ++col) {
                uint32_t expertId = topkIndex[base + col];
                if (expertId < expertNum) {
                    myCnt[expertId]++;
                }
            }
        }
    }

    // Reduce: each PE writes its assigned expert range
    uint32_t expertsPerPE = expertNum / kThreadsPerBlock;
    for (uint32_t e = 0; e < expertsPerPE; e++) {
        uint32_t globalExpert = tid * expertsPerPE + e;
        uint32_t sum = 0;
        for (uint32_t t = 0; t < kThreadsPerBlock; t++) {
            sum += cntLocal[t * expertNum + globalExpert];
        }
        tokenPerExpertCnt[globalExpert] = sum;
    }
}

// ============================================================================
// Phase 2 (Tile + multi-thread): scalar scatter into per-PE private sections
//
// No tile usage here: the scatter is a token-granularity random write and
// the min/pod computation reads each token's row once. A TLOAD whose data
// is never consumed by a tile op is dead traffic (and was previously
// duplicated 4x across PEs), so tiles are intentionally not used.
// ============================================================================
static inline void groupToken_mt_tile(
    uint32_t *topkIndex,
    uint32_t *perPegroupedIds,
    uint32_t *perPeSectionCnt,
    uint32_t *perPePodInfo,
    uint32_t batchSize,
    uint32_t topk,
    uint32_t expertPerRank,
    uint32_t expertPerPod,
    uint32_t superPodNum)
{
    const uint32_t tid = get_thread_idx();
    constexpr uint32_t kBsPerPE = kBS / kThreadsPerBlock;

    uint32_t *mySectionCnt = perPeSectionCnt + tid * expertPerRank;
    for (uint32_t i = 0; i < expertPerRank; i++) {
        mySectionCnt[i] = 0;
    }
    uint32_t dstPodLocal[kSuperPodNum];

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
// Host-side merge (scalar, single-PE)
// ============================================================================
static inline void mergeGroupTokenResults(
    const uint32_t *perPegroupedIds,
    const uint32_t *perPeSectionCnt,
    const uint32_t *perPePodInfo,
    uint32_t *groupedTokenIds,
    uint32_t *tokenSuperPodInfo,
    uint32_t *expertSectionTokenCnt,
    uint32_t expertPerRank,
    uint32_t superPodNum)
{
    constexpr uint32_t kBsPerPE = kBS / kThreadsPerBlock;

    for (uint32_t s = 0; s < expertPerRank; s++) {
        uint32_t globalIdx = 0;
        for (uint32_t t = 0; t < kThreadsPerBlock; t++) {
            uint32_t peCnt = perPeSectionCnt[t * expertPerRank + s];
            for (uint32_t i = 0; i < peCnt; i++) {
                uint32_t peOffset = s * kThreadsPerBlock * kBsPerPE
                                  + t * kBsPerPE + i;
                groupedTokenIds[s * kBS + globalIdx] = perPegroupedIds[peOffset];

                uint32_t podPeOffset = s * kThreadsPerBlock * kBsPerPE * superPodNum
                                     + t * kBsPerPE * superPodNum
                                     + i * superPodNum;
                for (uint32_t j = 0; j < superPodNum; j++) {
                    tokenSuperPodInfo[s * kBS * superPodNum + globalIdx * superPodNum + j]
                        = perPePodInfo[podPeOffset + j];
                }
                globalIdx++;
            }
        }
        expertSectionTokenCnt[s] = globalIdx;
    }
}

// ============================================================================
// Phase 3 (Tile + multi-thread): TROWMIN FloorFunc + PE0 counting sort
//
// Phase 3a is a true tile pipeline on disjoint per-PE tiles:
//   TLOAD(4x16 rows of this PE) -> TREMS(%, kExpertPerRank)
//   -> TROWMIN (per-row min) -> TSTORE(minLocalExpIds[token])
// The scalar GM read-back happens only after TSTORE, on a 1-column GM tensor
// (rule 1). Phase 3b stays scalar on PE 0 (global coordination).
// ============================================================================
static inline void sortKernel_mt_tile(
    uint32_t *topkIndex,
    uint32_t *minLocalExpIds,
    uint32_t *sortedTokenIds,
    uint32_t *sectionStarts,
    uint32_t batchSize,
    uint32_t topk,
    uint32_t expertPerRank)
{
    const uint32_t tid = get_thread_idx();

    // Phase 3a: FloorFunc via tile ops on this PE's disjoint rows
    using TilePerPE = Tile<Location::Vec, uint32_t, 4, kTileN, BLayout::RowMajor>;
    using TileMinPE = Tile<Location::Vec, uint32_t, 4, 8, BLayout::RowMajor, 4, 1>;
    using GmPerPE = global_tensor<uint32_t, RowMajor<kBS, kTopK>>;
    using GmMin = global_tensor<uint32_t, RowMajor<kBS, 1>>;
    using itPerPE = global_iterator<GmPerPE, TilePerPE>;
    using itMin = global_iterator<GmMin, TileMinPE>;

    // Full-tensor iterators; PE tid addresses its 4-row slice [16*blk + 4*tid,
    // +4) via row-tile index 4*blk + tid. (The previous base-pointer offset
    // tid*4 combined with per-block stride 4 only covered tokens [0, 140);
    // tokens 140..511 were never written and stayed zero — root cause of the
    // R2=4 section-bounds verification failure.)
    itPerPE gIter(topkIndex);
    itMin oIter(minLocalExpIds);

    TilePerPE tIn;
    TilePerPE tRem;
    TileMinPE tMin;
    for (uint32_t blk = 0; blk < kBS / kTileM; ++blk) {
        auto src = gIter(4 * blk + tid, 0);
        auto dst = oIter(4 * blk + tid, 0);
        TLOAD(tIn, src);
        TREMS(tRem, tIn, expertPerRank);   // local expert ids
        TROWMIN(tMin, tRem);               // per-token min
        TSTORE(dst, tMin);                 // -> minLocalExpIds[token]
    }

    // Phase 3b: Counting sort — only PE 0 (needs global coordination)
    if (tid == 0) {
        uint32_t counts[kExpertPerRank];
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
        uint32_t writePos[kExpertPerRank];
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
// Entry point (multi-PE SPMD; called by every PE)
//
// Cross-PE data hand-offs are separated by mtBarrier:
//   Phase1 reduce  reads all PEs' cntLocal      -> barrier after Phase1
//   merge          reads all PEs' scatter state -> barrier after Phase2;
//                  executed by PE0 only (single merge, no duplicate work)
//   Phase3b sort   reads all PEs' minLocalExpIds -> barrier after Phase3a
// ============================================================================
static inline void runGroupTokenVecMT(
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
    uint32_t *minLocalExpIds)
{
    const uint32_t tid = get_thread_idx();

    calTokenPerExpertCnt_mt_tile(topkIndex, tokenPerExpertCnt, cntLocal,
                                   kExpertNum, kTopKEleNum);
    mtBarrier(1);   // all PEs' histograms complete before reduce consumers

    groupToken_mt_tile(topkIndex, perPegroupedIds, perPeSectionCnt, perPePodInfo,
                         kBS, kTopK, kExpertPerRank, kExpertPerPod, kSuperPodNum);
    mtBarrier(2);   // all PEs' scatter sections complete before merge reads

    if (tid == 0) {
        mergeGroupTokenResults(perPegroupedIds, perPeSectionCnt, perPePodInfo,
                                groupedTokenIds, tokenSuperPodInfo, expertSectionTokenCnt,
                                kExpertPerRank, kSuperPodNum);
    }

    sortKernel_mt_tile(topkIndex, minLocalExpIds, sortedTokenIds, sectionStarts,
                         kBS, kTopK, kExpertPerRank);
    mtBarrier(3);   // FloorFunc writes visible before any PE reads results
}

#endif // GROUP_TOKEN_VEC_MT_HPP
