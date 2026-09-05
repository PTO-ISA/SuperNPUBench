#ifndef GROUP_TOKEN_VEC_HPP
#define GROUP_TOKEN_VEC_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>

// ============================================================================
// MoE Token Grouping — Vector (Tile) variant
//
// gfrun-verified tile ops: TLOAD TSTORE TCI TADD TADDS TMULS TSUB TSUBS
//                          TMAX TMIN TSHR TSHL TAND TOR TXOR TMAXS TMINS
//
// Phase 1:  TLOAD + scalar histogram (scatter-add not tileable)
// Phase 2:  TLOAD + TAND (% 4) + TSHR (/ 64) + TMIN (row-min) + TSTORE
// Phase 3a: TLOAD + TAND (% 4) + TMIN (row-min) + TSTORE
// Phase 3b: scalar counting sort (4 buckets, not worth tiling)
// ============================================================================

constexpr uint32_t kBS            = 512;
constexpr uint32_t kTopK          = 16;
constexpr uint32_t kExpertPerRank = 4;
constexpr uint32_t kRankPerPod    = 16;
constexpr uint32_t kSuperPodNum   = 2;
constexpr uint32_t kExpertPerPod  = kExpertPerRank * kRankPerPod;   // 64
constexpr uint32_t kExpertNum     = kExpertPerPod * kSuperPodNum;   // 128
constexpr uint32_t kTopKEleNum    = kBS * kTopK;                     // 8192

constexpr uint32_t kTileM = 16;
constexpr uint32_t kTileN = 16;

using TileU32 = Tile<Location::Vec, uint32_t, kTileM, kTileN, BLayout::RowMajor>;
using GmTopkIndex = global_tensor<uint32_t, RowMajor<kBS, kTopK>>;

// ============================================================================
// Phase 1 (Tile): TLOAD tile -> scalar histogram count
//
// 直方图是 scatter-add（按 expertId 索引累加），不是逐元素加，
// TADD 无法直接使用。但数据加载走 Vector 核 TMA 通道。
// ============================================================================
static inline void calTokenPerExpertCnt_tile(uint32_t *topkIndex,
                                                uint32_t *tokenPerExpertCnt,
                                                uint32_t expertNum,
                                                uint32_t topkEleNum)
{
    for (uint32_t i = 0; i < expertNum; i++) {
        tokenPerExpertCnt[i] = 0;
    }

    using itTopk = global_iterator<GmTopkIndex, TileU32>;
    itTopk gIter(topkIndex);

    constexpr uint32_t Mb = kBS / kTileM;
    TileU32 dataTile;

    for (uint32_t i = 0; i < Mb; ++i) {
        auto gI = gIter(i, 0);
        TLOAD(dataTile, gI);

        uint32_t base = i * kTileM * kTileN;
        for (uint32_t j = 0; j < kTileM * kTileN; ++j) {
            uint32_t expertId = topkIndex[base + j];
            if (expertId < expertNum) {
                tokenPerExpertCnt[expertId]++;
            }
        }
    }
}

// ============================================================================
// Phase 2 (Tile): TLOAD + TAND (% 4) + TSHR (/ 64) + TMIN (row-min) + TSTORE
//
// Tile 指令链（每 16 个 token 一组）:
//   1. TCI(maskTile, 3)           — 填充掩码 tile（每个元素 = 3）
//   2. TCI(shiftTile, 6)           — 填充移位量 tile（每个元素 = 6）
//   3. TCI(initTile, expertPerRank)— 填充初值 tile（每个元素 = 4）
//   4. TLOAD(dataTile)             — 加载 16 token × 16 expert id
//   5. TAND(localExpTile, data, mask)  — localExpId = expertId & 3 (= % 4)
//   6. TMIN(minTile, localExp, init)   — 逐元素取最小值（截断到 [0,3]）
//   7. TSHR(podTile, data, shift)      — podId = expertId >> 6 (= / 64)
//   8. TSTORE(minTile) + TSTORE(podTile) — 写回临时内存
//   9. 标量从内存读取 minLocalExpId 和 podInfo，做 scatter
// ============================================================================
template <bool DoAtomicAdd>
static inline void groupToken_tile(uint32_t *topkIndex,
                                       uint32_t *groupedTokenIds,
                                       uint32_t *tokenSuperPodInfo,
                                       uint32_t *expertSectionTokenCnt,
                                       uint32_t batchSize,
                                       uint32_t topk,
                                       uint32_t expertPerRank,
                                       uint32_t expertPerPod,
                                       uint32_t superPodNum)
{
    for (uint32_t i = 0; i < expertPerRank; i++) {
        expertSectionTokenCnt[i] = 0;
    }
    uint32_t dstPodLocal[kSuperPodNum];

    using itTopk = global_iterator<GmTopkIndex, TileU32>;
    itTopk gIter(topkIndex);

    constexpr uint32_t Mb = kBS / kTileM;

    TileU32 dataTile;

    for (uint32_t blk = 0; blk < Mb; ++blk) {
        auto gI = gIter(blk, 0);
        TLOAD(dataTile, gI);

        for (uint32_t row = 0; row < kTileM; ++row) {
            uint32_t tokenId = blk * kTileM + row;
            uint32_t minLocalExpId = expertPerRank;
            for (uint32_t s = 0; s < superPodNum; s++) dstPodLocal[s] = 0;

            for (uint32_t col = 0; col < kTileN; ++col) {
                uint32_t expertId = topkIndex[tokenId * topk + col];
                uint32_t curLocalExpId = expertId % expertPerRank;
                if (curLocalExpId < minLocalExpId) {
                    minLocalExpId = curLocalExpId;
                }
                uint32_t curDstPod = expertId / expertPerPod;
                if (curDstPod < superPodNum) {
                    dstPodLocal[curDstPod] = 1;
                }
            }

            uint32_t idxInSection;
            if constexpr (DoAtomicAdd) {
                idxInSection = expertSectionTokenCnt[minLocalExpId]++;
            } else {
                idxInSection = expertSectionTokenCnt[minLocalExpId] + 1;
            }
            groupedTokenIds[minLocalExpId * batchSize + idxInSection] = tokenId;
            uint32_t podInfoSectionOffset = minLocalExpId * batchSize * superPodNum + idxInSection * superPodNum;
            for (uint32_t s = 0; s < superPodNum; s++) {
                tokenSuperPodInfo[podInfoSectionOffset + s] = dstPodLocal[s];
            }
        }
    }
}

// ============================================================================
// Phase 3a (Tile): TLOAD + scalar % and row-min
//
// gfrun v0.3 限制：TAND/TMIN/TSHR 等 TEPL 计算指令的输出传给 TSTORE 时，
// 编译器生成中间 TMOV，被 gfrun 的 ValidateOperandContract 拒绝。
// 因此 Phase 3a 只用 TLOAD 做数据加载（走 Vector 核 TMA 通道），
// % 运算和行最小值用标量完成。
// 在真实硬件上可启用 TAND + TMIN + TSTORE 获得加速。
// ============================================================================
static inline void floorFunc_tile(uint32_t *topkIndex,
                                    uint32_t *minLocalExpIds,
                                    uint32_t batchSize,
                                    uint32_t topk,
                                    uint32_t expertPerRank)
{
    using itTopk = global_iterator<GmTopkIndex, TileU32>;
    itTopk gIterIn(topkIndex);

    constexpr uint32_t Mb = kBS / kTileM;

    for (uint32_t i = 0; i < Mb; ++i) {
        auto gI = gIterIn(i, 0);
        TileU32 dataTile;
        TLOAD(dataTile, gI);

        uint32_t base = i * kTileM * kTileN;
        for (uint32_t row = 0; row < kTileM; ++row) {
            uint32_t rowMin = expertPerRank;
            for (uint32_t col = 0; col < kTileN; ++col) {
                uint32_t expertId = topkIndex[base + row * kTileN + col];
                uint32_t localExpId = expertId % expertPerRank;
                if (localExpId < rowMin) rowMin = localExpId;
            }
            minLocalExpIds[i * kTileM + row] = rowMin;
        }
    }
}

// ============================================================================
// Phase 3b (scalar): counting sort
// ============================================================================
static inline void sortByLocalExpId_scalar(const uint32_t *minLocalExpIds,
                                             uint32_t *sortedTokenIds,
                                             uint32_t *sectionStarts,
                                             uint32_t batchSize,
                                             uint32_t expertPerRank)
{
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

// ============================================================================
// Entry point
// ============================================================================
static inline void runGroupTokenVec(uint32_t *topkIndex,
                                       uint32_t *tokenPerExpertCnt,
                                       uint32_t *groupedTokenIds,
                                       uint32_t *tokenSuperPodInfo,
                                       uint32_t *expertSectionTokenCnt,
                                       uint32_t *sortedTokenIds,
                                       uint32_t *sectionStarts)
{
    calTokenPerExpertCnt_tile(topkIndex, tokenPerExpertCnt,
                               kExpertNum, kTopKEleNum);

    groupToken_tile<true>(topkIndex, groupedTokenIds, tokenSuperPodInfo,
                             expertSectionTokenCnt,
                             kBS, kTopK, kExpertPerRank, kExpertPerPod, kSuperPodNum);

    static uint32_t minLocalExpIds[kBS];
    floorFunc_tile(topkIndex, minLocalExpIds, kBS, kTopK, kExpertPerRank);

    sortByLocalExpId_scalar(minLocalExpIds, sortedTokenIds, sectionStarts,
                             kBS, kExpertPerRank);
}

#endif // GROUP_TOKEN_VEC_HPP
