#include <common/pto_tileop.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "benchmark.h"
#include "group_token_old/group_token_old.hpp"

// PTO-ISA multi-thread profile: 4 threads per block.
static constexpr uint32_t kThreadsPerBlock = 4;

// ============================================================================
// Multi-PE barrier (software, sense-reversal-free phase counter).
//
// Follows the synchronization convention of multi_thread/matmul RES_CHECK:
// volatile flag + compiler memory barrier. Each PE publishes its own phase
// progress; a PE leaves the barrier only after all PEs reached the phase.
// ============================================================================
static volatile uint32_t sPhaseDone[kThreadsPerBlock];

static inline void mtCompilerBarrier()
{
    __asm__ volatile("" : : : "memory");
}

static inline void mtBarrier(uint32_t phase)
{
    mtCompilerBarrier();
    sPhaseDone[get_thread_idx()] = phase;      // publish my arrival
    mtCompilerBarrier();
    for (uint32_t t = 0; t < kThreadsPerBlock; ++t) {
        while (sPhaseDone[t] < phase) {
        }                                      // spin until all PEs arrived
    }
    mtCompilerBarrier();
}

// ============================================================================
// Data generation (identical to single-thread version)
// ============================================================================
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

// ============================================================================
// Multi-thread kernel: Phase 1 — CalTokenPerExpertCnt
//
// Following the .asc pattern: stride-mode parallelism.
// Each PE processes a stride of the topkIndex array and accumulates into
// its private cntLocal. After the parallel loop, each PE writes its local
// counts for its assigned expert range into the global tokenPerExpertCnt.
//
// Expert partition: 128 experts / 4 PEs = 32 experts per PE.
// Stride over topkIndex: PE tid processes elements [tid, tid+4, tid+8, ...].
// ============================================================================
static void calTokenPerExpertCnt_multithread(
    const uint32_t *topkIndex,
    uint32_t *tokenPerExpertCnt,
    uint32_t *cntLocal,            // [kThreadsPerBlock * kExpertNum]
    uint32_t expertNum,
    uint32_t topkEleNum)
{
    const uint32_t tid = get_thread_idx();

    // Each PE initializes its private cntLocal to zero
    uint32_t *myCnt = cntLocal + tid * expertNum;
    for (uint32_t i = 0; i < expertNum; i++) {
        myCnt[i] = 0;
    }

    // Stride-mode histogram: each PE counts every 4th element
    for (uint32_t i = tid; i < topkEleNum; i += kThreadsPerBlock) {
        uint32_t expertId = topkIndex[i];
        if (expertId < expertNum) {
            myCnt[expertId]++;
        }
    }

    // Reduce: each PE writes its contribution for its assigned expert range
    // Expert range: [tid * expertsPerPE, (tid+1) * expertsPerPE)
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
// Multi-thread kernel: Phase 2 — GroupToken (scatter)
//
// Following the .asc pattern: stride-mode parallelism with per-PE write
// pointers. Each PE processes a stride of tokens (bs/4 ≈ 128 tokens),
// computes minLocalExpId and pod info, then scatters into its private
// section of groupedTokenIds. After the parallel loop, the host merges
// per-PE sections into the global groupedTokenIds.
//
// Memory layout for multi-thread scatter:
//   groupedTokenIds:   [expertPerRank][kThreadsPerBlock][bs/kThreadsPerBlock]
//   tokenSuperPodInfo: [expertPerRank][kThreadsPerBlock][bs/kThreadsPerBlock][superPodNum]
//   expertSectionTokenCnt: [expertPerRank][kThreadsPerBlock]  (per-PE counts)
// ============================================================================
static void groupToken_multithread(
    const uint32_t *topkIndex,
    uint32_t *groupedTokenIds,        // [kExpertPerRank * kBS] (merged)
    uint32_t *tokenSuperPodInfo,      // [kExpertPerRank * kBS * kSuperPodNum] (merged)
    uint32_t *expertSectionTokenCnt,  // [kExpertPerRank] (merged)
    uint32_t *perPegroupedIds,        // [kExpertPerRank * kThreadsPerBlock * kBS_per_PE]
    uint32_t *perPeSectionCnt,        // [kExpertPerRank * kThreadsPerBlock]
    uint32_t *perPePodInfo,           // [kExpertPerRank * kThreadsPerBlock * kBS_per_PE * kSuperPodNum]
    uint32_t batchSize,
    uint32_t topk,
    uint32_t expertPerRank,
    uint32_t expertPerPod,
    uint32_t superPodNum)
{
    const uint32_t tid = get_thread_idx();
    constexpr uint32_t kBsPerPE = kBS / kThreadsPerBlock;  // 128

    // Per-PE section counts
    uint32_t *mySectionCnt = perPeSectionCnt + tid * expertPerRank;
    for (uint32_t i = 0; i < expertPerRank; i++) {
        mySectionCnt[i] = 0;
    }

    // Per-PE dstPodLocal
    uint32_t dstPodLocal[kSuperPodNum];
    for (uint32_t i = 0; i < superPodNum; i++) {
        dstPodLocal[i] = 0;
    }

    // Stride-mode scatter: each PE processes tokens [tid, tid+4, tid+8, ...]
    for (uint32_t i = tid; i < batchSize; i += kThreadsPerBlock) {
        uint32_t minLocalExpId = expertPerRank;
        uint32_t stop = (i + 1) * topk;
        for (uint32_t j = i * topk; j < stop; j++) {
            uint32_t curLocalExpId = topkIndex[j] % expertPerRank;
            if (curLocalExpId < minLocalExpId) {
                minLocalExpId = curLocalExpId;
            }
            uint32_t curDstPod = topkIndex[j] / expertPerPod;
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
        for (uint32_t j = 0; j < superPodNum; j++) {
            perPePodInfo[podPeOffset + j] = dstPodLocal[j];
            dstPodLocal[j] = 0;
        }
    }
}

// ============================================================================
// Host-side merge: combine per-PE scatter results into global arrays
// ============================================================================
static void mergeGroupTokenResults(
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
// Multi-thread kernel: Phase 3 — FloorFunc + Counting Sort
//
// Following the .asc pattern: FloorFunc is parallelized across PEs (stride
// mode), counting sort is done by a single PE (tid==0) because it requires
// global coordination.
// ============================================================================
static void sortKernel_multithread(
    const uint32_t *topkIndex,
    uint32_t *minLocalExpIds,     // [kBS] shared
    uint32_t *sortedTokenIds,
    uint32_t *sectionStarts,
    uint32_t batchSize,
    uint32_t topk,
    uint32_t expertPerRank)
{
    const uint32_t tid = get_thread_idx();

    // Phase 3a: FloorFunc — stride-mode parallel
    for (uint32_t i = tid; i < batchSize; i += kThreadsPerBlock) {
        uint32_t minLocalExpId = expertPerRank;
        uint32_t stop = (i + 1) * topk;
        for (uint32_t j = i * topk; j < stop; j++) {
            uint32_t curLocalExpId = topkIndex[j] % expertPerRank;
            if (curLocalExpId < minLocalExpId) {
                minLocalExpId = curLocalExpId;
            }
        }
        minLocalExpIds[i] = minLocalExpId;
    }

    // Phase 3b: Counting sort — only PE 0 does the global sort
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
// Scalar reference implementations (for verification)
// ============================================================================
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

// ============================================================================
// Main
// ============================================================================
int main()
{
    const uint32_t tid = get_thread_idx();

#ifndef __linx
    if (tid == 0) {
        printf("=== Multi-Thread Group Token Old Test (4-PE, 3-phase) ===\n");
        printf("BS=%u  TopK=%u  ExpertPerRank=%u  ExpertNum=%u  ThreadsPerBlock=%u\n",
               kBS, kTopK, kExpertPerRank, kExpertNum, kThreadsPerBlock);
        fflush(stdout);
    }
#endif

    // Global input/output arrays (in .bss via static)
    static uint32_t topkIndex[kTopKEleNum];
    static uint32_t tokenPerExpertCnt[kExpertNum];
    static uint32_t groupedTokenIds[kExpertPerRank * kBS];
    static uint32_t tokenSuperPodInfo[kExpertPerRank * kBS * kSuperPodNum];
    static uint32_t expertSectionTokenCnt[kExpertPerRank];
    static uint32_t sortedTokenIds[kBS];
    static uint32_t sectionStarts[kExpertPerRank + 1];

    // Per-PE scratch buffers for multi-thread scatter
    constexpr uint32_t kBsPerPE = kBS / kThreadsPerBlock;  // 128
    static uint32_t cntLocal[kThreadsPerBlock * kExpertNum];
    static uint32_t perPegroupedIds[kExpertPerRank * kThreadsPerBlock * kBsPerPE];
    static uint32_t perPeSectionCnt[kExpertPerRank * kThreadsPerBlock];
    static uint32_t perPePodInfo[kExpertPerRank * kThreadsPerBlock * kBsPerPE * kSuperPodNum];

    // Phase 3 shared
    static uint32_t minLocalExpIds[kBS];

    // Generate input data (deterministic per PE; identical on every PE)
    genTopkIndex(topkIndex, kBS, kTopK, kExpertNum);

    BENCHSTART;

    // Phase 1: Multi-thread histogram (4 PEs, stride mode)
    calTokenPerExpertCnt_multithread(
        topkIndex, tokenPerExpertCnt, cntLocal,
        kExpertNum, kTopKEleNum);
    mtBarrier(1);   // all PEs finished histogram before reduce consumers run

    // Phase 2: Multi-thread scatter (4 PEs, stride mode)
    groupToken_multithread(
        topkIndex, groupedTokenIds, tokenSuperPodInfo, expertSectionTokenCnt,
        perPegroupedIds, perPeSectionCnt, perPePodInfo,
        kBS, kTopK, kExpertPerRank, kExpertPerPod, kSuperPodNum);
    mtBarrier(2);   // all PEs finished scatter before merge reads their sections

    // Phase 2 merge: single-PE merge of per-PE results (avoids duplicated work
    // and guarantees the merge reads fully-written per-PE sections)
    if (tid == 0) {
        mergeGroupTokenResults(
            perPegroupedIds, perPeSectionCnt, perPePodInfo,
            groupedTokenIds, tokenSuperPodInfo, expertSectionTokenCnt,
            kExpertPerRank, kSuperPodNum);
    }

    // Phase 3: Multi-thread FloorFunc + single-PE counting sort
    sortKernel_multithread(
        topkIndex, minLocalExpIds, sortedTokenIds, sectionStarts,
        kBS, kTopK, kExpertPerRank);
    mtBarrier(3);   // FloorFunc stride writes visible before PE0 sort & verify

    BENCHEND;

    // --- verification & reference: PE0 only (single writer/reader domain) ---
    int ret = 0;
    if (tid == 0) {
        // --- compute reference results ---
        static uint32_t refExpertCnt[kExpertNum];
        static uint32_t refGroupedIds[kExpertPerRank * kBS];
        static uint32_t refSectionCnt[kExpertPerRank];
        static uint32_t refSortedIds[kBS];
        static uint32_t refSectionStarts[kExpertPerRank + 1];
        // scratch buffers used only by PE0 (plain locals on its stack)
        static uint32_t verBuf[kBS];
        static uint32_t verRef[kBS];

        for (uint32_t i = 0; i < kExpertPerRank * kBS; i++) refGroupedIds[i] = 0;
        refCalTokenPerExpertCnt(topkIndex, refExpertCnt, kExpertNum, kTopKEleNum);
        refGroupToken(topkIndex, refGroupedIds, refSectionCnt,
                      kBS, kTopK, kExpertPerRank);
        refSortByLocalExpId(topkIndex, refSortedIds, refSectionStarts,
                            kBS, kTopK, kExpertPerRank);

        // --- verify Phase 1: expert counts ---
        int cntMatch = 0;
        for (uint32_t i = 0; i < kExpertNum; i++) {
            if (tokenPerExpertCnt[i] == refExpertCnt[i]) cntMatch++;
        }

        // --- verify Phase 2: section counts ---
        int secMatch = 0;
        for (uint32_t i = 0; i < kExpertPerRank; i++) {
            if (expertSectionTokenCnt[i] == refSectionCnt[i]) secMatch++;
        }

        // --- verify Phase 2: grouped token ids (as sorted sets per section) ---
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

        // --- verify Phase 3: section boundaries ---
        int boundMatch = 0;
        for (uint32_t i = 0; i <= kExpertPerRank; i++) {
            if (sectionStarts[i] == refSectionStarts[i]) boundMatch++;
        }

        // --- verify Phase 3: sorted token ids (exact match, order matters) ---
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
        printf("\nSection counts: ");
        for (uint32_t i = 0; i < kExpertPerRank; i++)
            printf("%u ", expertSectionTokenCnt[i]);
        printf("\nSection starts: ");
        for (uint32_t i = 0; i <= kExpertPerRank; i++)
            printf("%u ", sectionStarts[i]);
        printf("\n");
        fflush(stdout);
#endif
    }

#ifndef __linx
    if (tid == 0) {
        printf("%s\n", ret ? "FAIL" : "PASS");
        fflush(stdout);
    }
#endif
    return ret;
}
