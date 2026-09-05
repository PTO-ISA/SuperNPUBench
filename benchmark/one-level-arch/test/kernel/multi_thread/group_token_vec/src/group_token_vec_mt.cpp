#include <common/pto_tileop.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "benchmark.h"
#include "group_token_vec_mt/group_token_vec_mt.hpp"

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
    static uint32_t minLocalExpIds[kBS];
    for (uint32_t i = 0; i < bs; i++) {
        uint32_t minLocal = expertPerRank;
        for (uint32_t j = 0; j < topk; j++) {
            uint32_t local = topkIndex[i * topk + j] % expertPerRank;
            if (local < minLocal) minLocal = local;
        }
        minLocalExpIds[i] = minLocal;
    }
    uint32_t counts[kExpertPerRank];
    for (uint32_t i = 0; i < expertPerRank; i++) counts[i] = 0;
    for (uint32_t i = 0; i < bs; i++) counts[minLocalExpIds[i]]++;
    refSectionStarts[0] = 0;
    for (uint32_t i = 0; i < expertPerRank; i++)
        refSectionStarts[i + 1] = refSectionStarts[i] + counts[i];
    uint32_t writePos[kExpertPerRank];
    for (uint32_t i = 0; i < expertPerRank; i++)
        writePos[i] = refSectionStarts[i];
    for (uint32_t i = 0; i < bs; i++) {
        uint32_t section = minLocalExpIds[i];
        refSortedIds[writePos[section]++] = i;
    }
}

int main()
{
    const uint32_t tid = get_thread_idx();

#ifndef __linx
    if (tid == 0) {
        printf("=== Multi-Thread Group Token Vec Test (4-PE, Tile) ===\n");
        printf("BS=%u  TopK=%u  ExpertPerRank=%u  ExpertNum=%u  ThreadsPerBlock=%u\n",
               kBS, kTopK, kExpertPerRank, kExpertNum, kThreadsPerBlock);
        fflush(stdout);
    }
#endif

    static uint32_t topkIndex[kTopKEleNum + 2 * 4096];
    uint32_t *topkIndexAligned = (uint32_t *)(((uint64_t)topkIndex & ~0xFFFu) + 0x1000);

    static uint32_t tokenPerExpertCnt[kExpertNum];
    static uint32_t groupedTokenIds[kExpertPerRank * kBS];
    static uint32_t tokenSuperPodInfo[kExpertPerRank * kBS * kSuperPodNum];
    static uint32_t expertSectionTokenCnt[kExpertPerRank];
    static uint32_t sortedTokenIds[kBS];
    static uint32_t sectionStarts[kExpertPerRank + 1];

    constexpr uint32_t kBsPerPE = kBS / kThreadsPerBlock;
    static uint32_t cntLocal[kThreadsPerBlock * kExpertNum];
    static uint32_t perPegroupedIds[kExpertPerRank * kThreadsPerBlock * kBsPerPE];
    static uint32_t perPeSectionCnt[kExpertPerRank * kThreadsPerBlock];
    static uint32_t perPePodInfo[kExpertPerRank * kThreadsPerBlock * kBsPerPE * kSuperPodNum];
    static uint32_t minLocalExpIds[kBS];

    genTopkIndex(topkIndexAligned, kBS, kTopK, kExpertNum);

    BENCHSTART;

    runGroupTokenVecMT(topkIndexAligned, tokenPerExpertCnt,
                        groupedTokenIds, tokenSuperPodInfo, expertSectionTokenCnt,
                        sortedTokenIds, sectionStarts,
                        cntLocal, perPegroupedIds, perPeSectionCnt, perPePodInfo,
                        minLocalExpIds);

    BENCHEND;

    // --- verification & reference: PE0 only (after final barrier inside
    // runGroupTokenVecMT, all outputs are fully written and visible) ---
    int ret = 0;
    if (tid == 0) {
        static uint32_t refExpertCnt[kExpertNum];
        static uint32_t refGroupedIds[kExpertPerRank * kBS];
        static uint32_t refSectionCnt[kExpertPerRank];
        static uint32_t refSortedIds[kBS];
        static uint32_t refSectionStarts[kExpertPerRank + 1];
        // PE0-private verification scratch
        static uint32_t verBuf[kBS];
        static uint32_t verRef[kBS];

        for (uint32_t i = 0; i < kExpertPerRank * kBS; i++) refGroupedIds[i] = 0;
        refCalTokenPerExpertCnt(topkIndexAligned, refExpertCnt, kExpertNum, kTopKEleNum);
        refGroupToken(topkIndexAligned, refGroupedIds, refSectionCnt,
                      kBS, kTopK, kExpertPerRank);
        refSortByLocalExpId(topkIndexAligned, refSortedIds, refSectionStarts,
                             kBS, kTopK, kExpertPerRank);

        int cntMatch = 0;
        for (uint32_t i = 0; i < kExpertNum; i++) {
            if (tokenPerExpertCnt[i] == refExpertCnt[i]) cntMatch++;
        }

        int secMatch = 0;
        for (uint32_t i = 0; i < kExpertPerRank; i++) {
            if (expertSectionTokenCnt[i] == refSectionCnt[i]) secMatch++;
        }

        int idMatch = 0;
        int idTotal = 0;
        for (uint32_t s = 0; s < kExpertPerRank; s++) {
            uint32_t n = expertSectionTokenCnt[s];
            idTotal += n;
            for (uint32_t i = 0; i < n; i++) {
                verBuf[i] = groupedTokenIds[s * kBS + i];
                verRef[i] = refGroupedIds[s * kBS + i];
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
        for (uint32_t i = 0; i <= kExpertPerRank; i++) {
            if (sectionStarts[i] == refSectionStarts[i]) boundMatch++;
        }

        int sortMatch = 0;
        for (uint32_t i = 0; i < kBS; i++) {
            if (sortedTokenIds[i] == refSortedIds[i]) sortMatch++;
        }

        if (cntMatch != (int)kExpertNum) ret = 1;
        else if (secMatch != (int)kExpertPerRank) ret = 2;
        else if (idMatch != idTotal) ret = 3;
        else if (boundMatch != (int)(kExpertPerRank + 1)) ret = 4;
        else if (sortMatch != (int)kBS) ret = 5;

#ifndef __linx
        printf("\n=== Verification (vs scalar reference) ===\n");
        printf("Phase 1 (expert counts):   %d/%u match\n", cntMatch, kExpertNum);
        printf("Phase 2 (section counts):  %d/%u match\n", secMatch, kExpertPerRank);
        printf("Phase 2 (grouped ids):     %d/%d match\n", idMatch, idTotal);
        printf("Phase 3 (section bounds):  %d/%u match\n", boundMatch, kExpertPerRank + 1);
        printf("Phase 3 (sorted ids):      %d/%u match\n", sortMatch, kBS);
        printf("%s\n", ret ? "FAIL" : "PASS");
        fflush(stdout);
#endif
    }

    return ret;
}
