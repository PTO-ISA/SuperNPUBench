/**
 * Group Token Vec — 4-PE SPMD 动态 shape test driver (group_token_vec_mt_dyn)
 *
 * 基于 group_token_vec_mt.cpp 的动态 shape 版驱动:
 *   - 单次运行内顺序执行 2 组运行时 shape:
 *       cfgA: {bs=512, topK=16, epr=4, epp=64, spn=2}  — 与静态版同值, 等价回归
 *       cfgB: {bs=517, topK=16, epr=8, epp=128, spn=2} — bs=517%16=5 触发
 *             尾块 ValidRow 路径 (PE0 vr=4/PE1 vr=1/PE2..3 空转),
 *             expertPerRank=8 覆盖专家域运行时路径
 *   - GM 缓冲按两组 shape 的最大值定长分配, 运行时只使用有效段;
 *   - 两次 kernel 调用之间把 header 内 sPhaseDone barrier 数组清零
 *     (静态相位编号在第二次调用会因陈旧 flag 立即通过而失去同步);
 *   - 验证 PE0 独占, 逐组比对 (5 项检查与静态版一致, 边界运行时化):
 *     cfgA 失败返回静态版同款诊断码 1..5, cfgB 失败返回 +10 偏移码。
 *
 * 运行 (gfrun 功能仿真, 必须四线程):
 *   gfrun -t 1 -s softcore.multiThreadNum=4 -f <elf>
 */

#include <common/pto_tileop.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "benchmark.h"
#include "solution/group_token_vec/group_token_vec_mt_dyn.hpp"

// ---- max-shape 缓冲尺寸 (cfgA/cfgB 的逐维最大值) ----
constexpr uint32_t kBSMax = 517;
constexpr uint32_t kTopKMax = 16;
constexpr uint32_t kExpertPerRankMax = 8;
constexpr uint32_t kSuperPodNumMax = 2;
constexpr uint32_t kExpertPodMax = kExpertPerRankMax * 16U;         // 128
constexpr uint32_t kExpertNumMax = kExpertPodMax * kSuperPodNumMax; // 256
constexpr uint32_t kTopKEleNumMax = kBSMax * kTopKMax;              // 8272
constexpr uint32_t kBsPerPEMax = (kBSMax + kThreadsPerBlock - 1U) / kThreadsPerBlock;

static void genTopkIndex(uint32_t *topkIndex, uint32_t bs, uint32_t k,
                          uint32_t expertNum)
{
    uint32_t seed = 0x1234ABCDu;
    for (uint32_t i = 0; i < bs; i++) {
        for (uint32_t j = 0; j < k; j++) {
            seed = seed * 1103515245u + 12345u;
            topkIndex[i * k + j] = (seed >> 16) % expertNum;
        }
    }
}

static void refCalTokenPerExpertCnt(const uint32_t *topkIndex,
                                     uint32_t *refExpertCnt,
                                     uint32_t expertNum, uint32_t topkEleNum)
{
    for (uint32_t i = 0; i < expertNum; i++) refExpertCnt[i] = 0;
    for (uint32_t i = 0; i < topkEleNum; i++) {
        if (topkIndex[i] < expertNum) refExpertCnt[topkIndex[i]]++;
    }
}

static void refGroupToken(const uint32_t *topkIndex,
                           uint32_t *refGroupedIds,
                           uint32_t *refSectionCnt,
                           uint32_t bs, uint32_t topk, uint32_t expertPerRank)
{
    for (uint32_t i = 0; i < expertPerRank; i++) refSectionCnt[i] = 0;
    for (uint32_t i = 0; i < bs; i++) {
        uint32_t minLocal = expertPerRank;
        for (uint32_t j = 0; j < topk; j++) {
            uint32_t local = topkIndex[i * topk + j] % expertPerRank;
            if (local < minLocal) minLocal = local;
        }
        uint32_t idx = refSectionCnt[minLocal]++;
        refGroupedIds[minLocal * bs + idx] = i;
    }
}

static void refSortByLocalExpId(const uint32_t *topkIndex,
                                 uint32_t *refSortedIds,
                                 uint32_t *refSectionStarts,
                                 uint32_t bs, uint32_t topk,
                                 uint32_t expertPerRank)
{
    static uint32_t minLocalExpIds[kBSMax];
    static uint32_t counts[kExpertPerRankMax];
    static uint32_t writePos[kExpertPerRankMax];
    for (uint32_t i = 0; i < bs; i++) {
        uint32_t minLocal = expertPerRank;
        for (uint32_t j = 0; j < topk; j++) {
            uint32_t local = topkIndex[i * topk + j] % expertPerRank;
            if (local < minLocal) minLocal = local;
        }
        minLocalExpIds[i] = minLocal;
    }
    for (uint32_t i = 0; i < expertPerRank; i++) counts[i] = 0;
    for (uint32_t i = 0; i < bs; i++) counts[minLocalExpIds[i]]++;
    refSectionStarts[0] = 0;
    for (uint32_t i = 0; i < expertPerRank; i++)
        refSectionStarts[i + 1] = refSectionStarts[i] + counts[i];
    for (uint32_t i = 0; i < expertPerRank; i++)
        writePos[i] = refSectionStarts[i];
    for (uint32_t i = 0; i < bs; i++) {
        uint32_t section = minLocalExpIds[i];
        refSortedIds[writePos[section]++] = i;
    }
}

// GM 缓冲 (verify/main 共同引用, 声明置于使用之前)
static uint32_t topkIndex[kTopKEleNumMax + 2 * 4096];
static uint32_t *topkIndexAligned = (uint32_t *)(((uint64_t)topkIndex & ~0xFFFu) + 0x1000);

static uint32_t tokenPerExpertCnt[kExpertNumMax];
static uint32_t groupedTokenIds[kExpertPerRankMax * kBSMax];
static uint32_t tokenSuperPodInfo[kExpertPerRankMax * kBSMax * kSuperPodNumMax];
static uint32_t expertSectionTokenCnt[kExpertPerRankMax];
static uint32_t sortedTokenIds[kBSMax];
static uint32_t sectionStarts[kExpertPerRankMax + 1];

static uint32_t cntLocal[kThreadsPerBlock * kExpertNumMax];
static uint32_t perPegroupedIds[kExpertPerRankMax * kThreadsPerBlock * kBsPerPEMax];
static uint32_t perPeSectionCnt[kExpertPerRankMax * kThreadsPerBlock];
static uint32_t perPePodInfo[kExpertPerRankMax * kThreadsPerBlock * kBsPerPEMax * kSuperPodNumMax];
static uint32_t minLocalExpIds[kBSMax];
static uint32_t podScratch[kThreadsPerBlock * kSuperPodNumMax];
static uint32_t sortScratch[2 * kExpertPerRankMax];

// 验证 (静态版 group_token_vec_mt.cpp 5 项检查同逻辑, 边界运行时化)。
// 返回 0 = PASS; 1..5 = 静态版同款诊断码。
// 注: 函数体引用文件作用域 GM 缓冲 (topkIndexAligned/tokenPerExpertCnt 等)。
static int verify(const int64_t *tiling)
{
    const uint32_t bs = static_cast<uint32_t>(tiling[0]);
    const uint32_t topK = static_cast<uint32_t>(tiling[1]);
    const uint32_t expertPerRank = static_cast<uint32_t>(tiling[2]);
    const uint32_t expertPerPod = static_cast<uint32_t>(tiling[3]);
    const uint32_t superPodNum = static_cast<uint32_t>(tiling[4]);
    const uint32_t expertNum = expertPerPod * superPodNum;
    const uint32_t topkEleNum = bs * topK;

    static uint32_t refExpertCnt[kExpertNumMax];
    static uint32_t refGroupedIds[kExpertPerRankMax * kBSMax];
    static uint32_t refSectionCnt[kExpertPerRankMax];
    static uint32_t refSortedIds[kBSMax];
    static uint32_t refSectionStarts[kExpertPerRankMax + 1];
    // PE0-private verification scratch
    static uint32_t verBuf[kBSMax];
    static uint32_t verRef[kBSMax];

    for (uint32_t i = 0; i < expertPerRank * bs; i++) refGroupedIds[i] = 0;
    refCalTokenPerExpertCnt(topkIndexAligned, refExpertCnt, expertNum, topkEleNum);
    refGroupToken(topkIndexAligned, refGroupedIds, refSectionCnt, bs, topK, expertPerRank);
    refSortByLocalExpId(topkIndexAligned, refSortedIds, refSectionStarts, bs, topK, expertPerRank);

    int cntMatch = 0;
    for (uint32_t i = 0; i < expertNum; i++) {
        if (tokenPerExpertCnt[i] == refExpertCnt[i]) cntMatch++;
    }

    int secMatch = 0;
    for (uint32_t i = 0; i < expertPerRank; i++) {
        if (expertSectionTokenCnt[i] == refSectionCnt[i]) secMatch++;
    }

    int idMatch = 0;
    int idTotal = 0;
    for (uint32_t s = 0; s < expertPerRank; s++) {
        uint32_t n = expertSectionTokenCnt[s];
        idTotal += n;
        for (uint32_t i = 0; i < n; i++) {
            verBuf[i] = groupedTokenIds[s * bs + i];
            verRef[i] = refGroupedIds[s * bs + i];
        }
        for (uint32_t i = 0; i < n; i++) {
            for (uint32_t j = i + 1; j < n; j++) {
                if (verBuf[i] > verBuf[j]) { uint32_t t = verBuf[i]; verBuf[i] = verBuf[j]; verBuf[j] = t; }
                if (verRef[i] > verRef[j]) { uint32_t t = verRef[i]; verRef[i] = verRef[j]; verRef[j] = t; }
            }
        }
        for (uint32_t i = 0; i < n; i++) {
            if (verBuf[i] == verRef[i]) idMatch++;
        }
    }

    int boundMatch = 0;
    for (uint32_t i = 0; i <= expertPerRank; i++) {
        if (sectionStarts[i] == refSectionStarts[i]) boundMatch++;
    }

    int sortMatch = 0;
    for (uint32_t i = 0; i < bs; i++) {
        if (sortedTokenIds[i] == refSortedIds[i]) sortMatch++;
    }

    if (cntMatch != (int)expertNum) return 1;
    if (secMatch != (int)expertPerRank) return 2;
    if (idMatch != idTotal) return 3;
    if (boundMatch != (int)(expertPerRank + 1)) return 4;
    if (sortMatch != (int)bs) return 5;
    return 0;
}

int main()
{
    const uint32_t tid = get_thread_idx();

    const int64_t cfgA[5] = {512, 16, 4, 4 * 16, 2};      // 与静态版同值 (等价回归)
    const int64_t cfgB[5] = {517, 16, 8, 8 * 16, 2};      // 尾块 + 专家域运行时覆盖
    const int64_t *cfgs[2] = {cfgA, cfgB};

    int failA = 0;
    for (int c = 0; c < 2; ++c) {
        // Barrier reset: 静态相位编号跨次调用会因陈旧 flag 立即通过而
        // 失去同步, 每次调用前清零 (各 PE 冗余同值写, 无需栅栏)。
        for (uint32_t t = 0; t < kThreadsPerBlock; ++t) sPhaseDone[t] = 0;

        // Input init runs redundantly on every PE (deterministic, identical
        // writes — same convention as group_token_vec_mt's genTopkIndex).
        genTopkIndex(topkIndexAligned,
                     static_cast<uint32_t>(cfgs[c][0]),
                     static_cast<uint32_t>(cfgs[c][1]),
                     static_cast<uint32_t>(cfgs[c][3]) *
                         static_cast<uint32_t>(cfgs[c][4]));

        BENCHSTART;

        runGroupTokenVecMTDyn(topkIndexAligned, tokenPerExpertCnt,
                              groupedTokenIds, tokenSuperPodInfo, expertSectionTokenCnt,
                              sortedTokenIds, sectionStarts,
                              cntLocal, perPegroupedIds, perPeSectionCnt, perPePodInfo,
                              minLocalExpIds, podScratch, sortScratch, cfgs[c]);

        BENCHEND;

        // cfgA 验证须在其输入 (topkIndex) 被 cfgB 数据生成覆盖之前完成:
        // PE0 立即验证, 其余 PE 在 mtBarrier(4) 汇合等待 (kernel 内部相位 1/2/3)。
        if (c == 0) {
            if (tid == 0) {
                failA = verify(cfgs[0]);
            }
            mtBarrier(4);
        }
    }

    // --- verification & reference: PE0 only (after final barrier inside
    // runGroupTokenVecMTDyn, all outputs are fully written and visible) ---
    if (tid != 0) {
        return 0;
    }

    if (failA != 0) return failA;          // cfgA: 静态版同款诊断码
    int rcB = verify(cfgs[1]);
    return rcB == 0 ? 0 : 10 + rcB;        // cfgB: +10 偏移诊断码
}