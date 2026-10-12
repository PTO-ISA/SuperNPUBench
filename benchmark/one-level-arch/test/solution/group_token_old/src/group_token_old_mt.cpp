#include <common/pto_tileop.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "benchmark.h"
#include "solution/group_token_old/group_token_old.hpp"

// PTO-ISA multi-thread profile: 4 threads per block.
static constexpr uint32_t kThreadsPerBlock = 4;

// ============================================================================
// Multi-PE barrier (software, sense-reversal-free phase counter).
//
// [2026-10-04 修改] 保留本算子原有的 volatile flag 裸自旋屏障，不移植
// group_token_vec_mt 的驱逐屏障：实测（--dump-memory 数据取证，见
// kernels/solution/group_token_old/group_token_old_mt_gfsim_fix_report.md
// §4.4）本算子映射驱逐屏障后 Phase 3 计数排序结果被污染（sortMatch
// 494/512），裸自旋屏障下同负载确定性全对（512/512）；且本屏障的 flag
// 轮询为 volatile（lwi.u）语义，长安静窗口（末端验证汇合，worker 自旋
// ~4×10^5 cycle）下多次实测均可靠放行，未复现 vec 家族报告的陈旧 flag
// 活锁（该场景与其屏障的缓存式轮询实现相关）。
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
// [2026-10-04 修复] gfsim `--conf fourpe` 数据完整性修复（--dump-memory 取证，
// 见 kernels/solution/group_token_old/group_token_old_mt_gfsim_fix_report.md）：
//   1. 输入分解由 stride-4（4 PE 并发读同一 cacheline）改为每 PE 连续 1/4 段
//      （直方图对输入划分可交换，结果等价）——并发同行填充竞态是直方图计数
//      出垃圾值（~12/128 bin）的第二大来源；
//   2. 计数累加由 .bss 缓存式 RMW（myCnt[e]++）改为寄存器/栈私有累加 +
//      volatile（非缓存）一次性导出——group_token_vec_mt.hpp 同款约定
//      （"acc → myCnt volatile 标量逐元素一次性"）；
//   3. reduce 读取改为 volatile——非缓存直读 L2，规避读侧 L1D 陈旧行。
// ============================================================================
static void calTokenPerExpertCnt_multithread(
    const uint32_t *topkIndex,
    uint32_t *tokenPerExpertCnt,
    uint32_t *cntLocal,            // [kThreadsPerBlock * kExpertNum]
    uint32_t expertNum,
    uint32_t topkEleNum)
{
    const uint32_t tid = get_thread_idx();

    // [2026-10-04 修改] 寄存器/栈私有累加（不再对 .bss 做缓存式 RMW）
    uint32_t acc[256];
    for (uint32_t i = 0; i < expertNum; i++) {
        acc[i] = 0;
    }

    // [2026-10-04 修改] 每 PE 连续 1/4 输入段（替代 stride-4 共享行并发读）
    uint32_t perPE = topkEleNum / kThreadsPerBlock;
    const uint32_t *myIn = topkIndex + tid * perPE;
    for (uint32_t i = 0; i < perPE; i++) {
        uint32_t expertId = myIn[i];
        if (expertId < expertNum) {
            acc[expertId]++;
        }
    }

    // [2026-10-04 修改] volatile（非缓存）一次性导出到私有 cntLocal 切片
    volatile uint32_t *myCntV = cntLocal + tid * expertNum;
    for (uint32_t i = 0; i < expertNum; i++) {
        myCntV[i] = acc[i];
    }

    // [2026-10-04 修复] 跨 PE 顺序竞争：reduce 需要读取所有 PE 的 cntLocal
    // 切片，原实现把 mtBarrier 放在函数返回后的 main 里（reduce 之后），
    // gfsim 真实 PE 偏斜下 reduce 会读到其它 PE 未完成的计数（gfrun 靠确定性
    // lockstep 侥幸通过）。屏障移入函数内部：直方图导出后、cross-PE reduce
    // 读取前（与 group_token_vec_mt.hpp calTokenPerExpertCnt_mt_tile 内屏障
    // 同款布局；相位编号 1=数据生成发布, 2=本屏障, 3=scatter→merge,
    // 4=FloorFunc→sort, 5=验证汇合）。
    mtBarrier(2);   // all PEs' private histograms complete before reduce

    // Reduce: each PE writes its contribution for its assigned expert range
    // Expert range: [tid * expertsPerPE, (tid+1) * expertsPerPE)
    // [2026-10-04 修改] reduce 读取 volatile——非缓存直读 L2
    uint32_t expertsPerPE = expertNum / kThreadsPerBlock;
    const volatile uint32_t *cl = cntLocal;
    for (uint32_t e = 0; e < expertsPerPE; e++) {
        uint32_t globalExpert = tid * expertsPerPE + e;
        uint32_t sum = 0;
        for (uint32_t t = 0; t < kThreadsPerBlock; t++) {
            sum += cl[t * expertNum + globalExpert];
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
    // [2026-10-04 修改] per-PE 分区计数器改为 volatile（非缓存）RMW：
    // 该计数器是跨 PE 交接数据（PE0 的 merge 在 mtBarrier(3) 后读取），
    // volatile 访问直连 L2，规避写侧 L1D 脏行滞留（group_token_vec_mt.hpp
    // 同款约定）。
    volatile uint32_t *mySectionCnt = perPeSectionCnt + tid * expertPerRank;
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

        uint32_t idxInSection = mySectionCnt[minLocalExpId];
        mySectionCnt[minLocalExpId] = idxInSection + 1;
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

    // [2026-10-04 修复] 跨 PE 顺序竞争：Phase 3b 的 counting sort（PE0 独占）
    // 读取所有 PE 在 3a 写入的 minLocalExpIds，原实现把 mtBarrier 放在
    // 函数返回后的 main 里（3b 之后），gfsim 真实 PE 偏斜下 PE0 会读到其它
    // PE 未完成的 minLocalExpIds → counts/writePos 垃圾（gfrun 靠确定性
    // lockstep 侥幸）。屏障移入函数内部：FloorFunc 完成后、PE0 counting
    // sort 前（与 group_token_vec_mt.hpp sortKernel_mt_tile 内屏障同款布局；
    // 相位 4 = FloorFunc→sort）。
    mtBarrier(4);   // all PEs' FloorFunc writes visible before PE0's sort

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

    // [2026-10-04 修复] 数据生成改为 PE0 独占 + 发布屏障（相位 1）：原实现
    // 4 个 PE 冗余执行 genTopkIndex 同时写同一批 topkIndex 行（gfrun 功能
    // 模型确定性同值写无害；gfsim 时序模型跨 PE 无 snoop，同行并发写在真实
    // PE 偏斜下产生丢更新/行损坏）——--dump-memory 取证实测这是 Phase 1
    // 直方图计数垃圾值（~12/128 bin，实测 cntMatch=116/128）的最大来源，
    // 改为 PE0 独占生成 + mtBarrier(1) 发布后全部计数正确（cntMatch=128/128）。
    if (tid == 0) {
        genTopkIndex(topkIndex, kBS, kTopK, kExpertNum);
    }
    mtBarrier(1);   // PE0's input generation visible to all PEs

    BENCHSTART;

    // Phase 1: Multi-thread histogram (4 PEs, per-PE contiguous quarter)
    // [2026-10-04 修改] mtBarrier(2) 在函数内部（直方图导出后、cross-PE
    // reduce 读取前），原调用点屏障已随竞争修复移除，见函数内注。
    calTokenPerExpertCnt_multithread(
        topkIndex, tokenPerExpertCnt, cntLocal,
        kExpertNum, kTopKEleNum);

    // Phase 2: Multi-thread scatter (4 PEs, stride mode)
    groupToken_multithread(
        topkIndex, groupedTokenIds, tokenSuperPodInfo, expertSectionTokenCnt,
        perPegroupedIds, perPeSectionCnt, perPePodInfo,
        kBS, kTopK, kExpertPerRank, kExpertPerPod, kSuperPodNum);
    mtBarrier(3);   // all PEs finished scatter before merge reads their sections

    // Phase 2 merge: single-PE merge of per-PE results (avoids duplicated work
    // and guarantees the merge reads fully-written per-PE sections)
    if (tid == 0) {
        mergeGroupTokenResults(
            perPegroupedIds, perPeSectionCnt, perPePodInfo,
            groupedTokenIds, tokenSuperPodInfo, expertSectionTokenCnt,
            kExpertPerRank, kSuperPodNum);
    }

    // Phase 3: Multi-thread FloorFunc + single-PE counting sort
    // [2026-10-04 修改] mtBarrier(4) 在函数内部（FloorFunc 完成后、PE0
    // counting sort 读取 minLocalExpIds 前），原调用点屏障已随竞争修复
    // 移除，见函数内注。
    sortKernel_multithread(
        topkIndex, minLocalExpIds, sortedTokenIds, sectionStarts,
        kBS, kTopK, kExpertPerRank);

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
        // [2026-10-04 优化] 原实现为分区 O(n^2) 双数组冒泡排序比对（本数据
        // 分布下 n_0≈506，~12.8 万对比较）。gfsim fourpe 下该标量冒泡的
        // 代码生成/时序模型组合效率极低（实测拖慢至 ~4.6M cycle / 仿真
        // 40+ 分钟），且与退出 lockstep 修复无关。改为 O(n) 计数式多重集
        // 比对（语义等价：两组分区前缀的多重集相等 ⇔ 排序后逐元素相等），
        // gfsim 全流程 ~0.66M cycle，判定结果与冒泡版一致（gfrun R2 同判）。
        int idMatch = 0;
        int idTotal = 0;
        // PE0-private verification scratch ([C9] 同款一次性 .bss scratch)
        static uint32_t cntA[kBS];
        static uint32_t cntB[kBS];
        for (uint32_t s = 0; s < kExpertPerRank; s++) {
            uint32_t n = expertSectionTokenCnt[s];
            idTotal += n;
            for (uint32_t v = 0; v < kBS; v++) {
                cntA[v] = 0;
                cntB[v] = 0;
            }
            for (uint32_t i = 0; i < n; i++) {
                uint32_t a = groupedTokenIds[s * kBS + i];
                uint32_t b = refGroupedIds[s * kBS + i];
                if (a < kBS) cntA[a]++;
                if (b < kBS) cntB[b]++;
            }
            uint32_t diff = 0;
            for (uint32_t v = 0; v < kBS; v++) {
                diff += (cntA[v] > cntB[v]) ? (cntA[v] - cntB[v])
                                            : (cntB[v] - cntA[v]);
            }
            if (diff == 0) idMatch += n;   // 分区多重集完全一致
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

    // [2026-10-04 修复] 退出 lockstep 搁浅（本次 gfsim `--conf fourpe` 确定性
    // 崩溃的根因，实测 cycle=242,720 断言复现）：原实现 worker (tid!=0) 在
    // kernel 末端屏障放行后直接 `return ret`，先于 PE0 到达 _end 退出 ecall
    // 并 park 到退出 lockstep AND 组（gfsim 实测 ready=3/4 joined=3/4，
    // bpc=0x112b0）；PE0 独占的参考计算 + 比对（~10^5 cycle 标量长尾）使
    // 退出 AND 组停在 ready=3/4 超过 T_deadlock=9999 cycle →
    // SyscallBarrier.cpp:1889 LockstepWatchdog 断言中止仿真（单 PE 版本无此
    // 问题：退出 lockstep 组仅 1 个参与者，AND 到达即完成）。
    // 修复镜像 group_token_vec_mt 的已证方案（gfsim fourpe 实测 PASS，见
    // group_token_vec_mt_gfsim_fix_report.md）：新增验证汇合屏障（相位 5，
    // kernel 内部用 1..4）——验证在 worker 存活期间执行（worker 在汇合屏障
    // 自旋而非 park 到退出 lockstep），PE0 验证完成后全 PE 一起放行、同时
    // 到达 _end（vec 同款修复实测 4 线程到达窗口 <100 cycle << 9999）。
    mtBarrier(5);

    return ret;
}
