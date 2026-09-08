#ifndef GROUP_TOKEN_OLD_HPP
#define GROUP_TOKEN_OLD_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
// ============================================================================
// MoE Token Grouping operator — "old" variant with SIMD sort (PTO one-level-arch)
//
// Implements the full 3-phase MoE dispatch from cann-samples
// group_token_old_main.asc:
//
//   Phase 1 — CalTokenPerExpertCnt
//     topkIndex [bs * k]  ->  tokenPerExpertCnt [expertNum]
//     Histogram: count how many tokens select each expert.
//
//   Phase 2 — GroupToken (SIMT scatter)
//     topkIndex [bs * k]  ->  groupedTokenIds [expertPerRank * bs]
//                              expertSectionTokenCnt [expertPerRank]
//     For each token, find the minimum "local expert id" (expertId %
//     expertPerRank) among its top-k experts, then scatter the token id
//     into the corresponding expert section via atomic add.
//
//   Phase 3 — SortKernel (SIMD sort)
//     topkIndex [bs * k]  ->  sortedGroupedTokenIds [bs]
//     Compute per-token minLocalExpId via FloorFunc, then sort tokens
//     by minLocalExpId to produce a contiguous, partitioned token id list.
//     This replaces the scattered output of Phase 2 with a compact,
//     sorted-by-section array.
//
// The scalar implementations below are unguarded and compile under every
// backend (__linx / __cpu_sim__ / __ARM_FEATURE_SME), guaranteeing correct
// results on gfrun/gfsim.  Optional SIMT-accelerated paths are provided
// under #if !defined(__linx) && !defined(__cpu_sim__), following the same
// convention as sort/topk.hpp.
// ============================================================================

// --- MoE topology constants (matching gen_expert_ids.py defaults) ---
constexpr uint32_t kBS            = 512;
constexpr uint32_t kTopK          = 16;
constexpr uint32_t kExpertPerRank = 4;
constexpr uint32_t kRankPerPod    = 16;
constexpr uint32_t kSuperPodNum   = 2;
constexpr uint32_t kExpertPerPod  = kExpertPerRank * kRankPerPod;   // 64
constexpr uint32_t kExpertNum     = kExpertPerPod * kSuperPodNum;   // 128
constexpr uint32_t kTopKEleNum    = kBS * kTopK;                     // 8192

// --- Tile type: 16×16 uint32 = 256 elements ---
using TileU32 = Tile<Location::Vec, uint32_t, 16, 16, BLayout::RowMajor>;

// ============================================================================
// Phase 1 (scalar): CalTokenPerExpertCnt
//
// 功能：遍历 topkIndex 数组，统计每个 expert 被多少个 token 选中（直方图）。
//       对应 .asc 中 __simt_vf__ CalTokenPerExpertCnt 的标量保底实现。
//
// 输入：topkIndex      — [expertNum] 每个 token 选中的 top-k expert id 列表，共 bs*k 个
//       expertNum      — 全局 expert 总数（128）
//       topkEleNum     — topkIndex 元素总数（bs*k = 8192）
// 输出：tokenPerExpertCnt — [expertNum] 每个 expert 收到的 token 计数
//
// 执行方式：单线程标量 for 循环，顺序遍历 topkIndex 逐个计数。
//           对应 .asc 版本用 128 线程 stride 并行 + asc_atomic_add 原子累加。
// ============================================================================
static inline void calTokenPerExpertCnt_scalar(const uint32_t *topkIndex,
                                                  uint32_t *tokenPerExpertCnt,
                                                  uint32_t expertNum,
                                                  uint32_t topkEleNum)
{
    for (uint32_t i = 0; i < expertNum; i++) {
        tokenPerExpertCnt[i] = 0;
    }
    for (uint32_t i = 0; i < topkEleNum; i++) {
        uint32_t expertId = topkIndex[i];
        if (expertId < expertNum) {
            tokenPerExpertCnt[expertId]++;
        }
    }
}

#if !defined(__linx) && !defined(__cpu_sim__)

// ============================================================================
// Phase 1 (SIMT): CalTokenPerExpertCnt_Vec_Impl
//
// 功能：SIMT vector kernel，每个 lane 以 stride 模式遍历 topkIndex，
//       通过原子操作累加到共享的 cntLocal 局部直方图，再将 cntLocal
//       原子累加到全局 tokenPerExpertCnt。
//       对应 .asc 中 __simt_vf__ CalTokenPerExpertCnt 的 PTO-ISA 实现。
//
// 输入：src              — topkIndex 数组指针
//       topkEleNum      — topkIndex 元素总数
//       expertNum       — expert 总数
// 输入/输出：cntLocal    — [expertNum] 局部直方图缓冲区（线程块内共享）
// 输出：dst             — tile 寄存器，第 expertId 个元素存储该 expert 的计数
//
// 执行方式：128 个 lane stride 模式并行遍历 topkIndex，asc_atomic_add 累加到
//           cntLocal，同步后再将 cntLocal 原子累加到全局 tokenPerExpertCnt。
//           对应 .asc 中的三步：init → histogram → reduce。
//
// 注意：此路径在 __linx / __cpu_sim__ 下不编译，v300 仿真器不支持。
// ============================================================================
template <typename tile_shape_out>
void __vec__ CalTokenPerExpertCnt_Vec_Impl(
    typename tile_shape_out::TileDType __out__ dst,
    const uint32_t* __in__ src,
    uint32_t topkEleNum,
    uint32_t expertNum,
    uint32_t* __out__ cntLocal)
{
    uint32_t tid = blkv_get_index_y();
    uint32_t stride = blockDim.x;

    // Step 1: 初始化 cntLocal（对应 .asc 第 129-132 行）
    for (uint32_t i = tid; i < expertNum; i += stride) {
        cntLocal[i] = 0;
    }
    asc_syncthreads();

    // Step 2: stride 模式统计直方图，原子累加到 cntLocal（对应 .asc 第 135-138 行）
    for (uint32_t i = tid; i < topkEleNum; i += stride) {
        uint32_t expertId = src[i];
        if (expertId < expertNum) {
            asc_atomic_add(cntLocal + expertId, 1);
        }
    }
    asc_syncthreads();

    // Step 3: 将 cntLocal 原子累加到全局 tokenPerExpertCnt（对应 .asc 第 141-145 行）
    for (uint32_t i = tid; i < expertNum; i += stride) {
        blkv_get_tile_ptr(dst)[i] = cntLocal[i];
    }
    asc_syncthreads();
}

// ============================================================================
// Phase 1 (SIMT): CalTokenPerExpertCnt_Impl
//
// 功能：Phase 1 SIMT 启动函数，启动 expertNum 个 lane 执行 CalTokenPerExpertCnt_Vec_Impl。
//       对应 .asc 中 asc_vf_call<CalTokenPerExpertCnt>(dim3(THREAD_COUNT), ...) 的调用。
//
// 输入：src              — topkIndex 数组指针
//       topkEleNum      — topkIndex 元素总数
//       expertNum       — expert 总数，决定启动的 lane 数
//       cntLocal        — [expertNum] 局部直方图缓冲区（线程块内共享）
// 输出：dst              — tile 寄存器，存储每个 expert 的 token 计数
//
// 执行方式：<<<1, expertNum, 1>>> 启动 1 个 block、expertNum 个 lane。
//
// 注意：此路径在 __linx / __cpu_sim__ 下不编译，v300 仿真器不支持。
// ============================================================================
template <typename tile_shape_out>
void CalTokenPerExpertCnt_Impl(tile_shape_out& dst, const uint32_t* src,
                               uint32_t topkEleNum, uint32_t expertNum,
                               uint32_t* cntLocal)
{
    CalTokenPerExpertCnt_Vec_Impl<tile_shape_out>
        <<<1, expertNum, 1>>>(dst.data(), src, topkEleNum, expertNum, cntLocal);
}

// ============================================================================
// Phase 2 (SIMT): GroupToken_Vec_Impl
//
// 功能：SIMT vector kernel，每个 lane 以 stride 模式处理一部分 token，
//       先将 topkIndex 拷贝到局部缓冲区 topkIndexLocal，再计算 minLocalExpId
//       和 pod 信息，通过原子自增写指针 scatter 到对应分区。
//       对应 .asc 中 __simt_vf__ GroupToken 的 PTO-ISA 实现。
//
// 输入：topkIndex           — [bs*k] 每个 token 选中的 top-k expert id 列表
//       topkIndexLocal     — [bs*k] 局部缓冲区，拷贝 topkIndex 用于 UB 内访问
//       dstPodLocal        — [blockDim.x * superPodNum] 各线程的 pod 信息缓冲区
//       batchSize          — token 总数
//       topk               — 每个 token 选中的 expert 数
//       expertPerRank      — 每个 rank 的本地 expert 数
//       expertPerPod       — 每个 pod 的 expert 数
//       superPodNum        — super pod 数量
// 输出：groupedTokenIds     — [expertPerRank * bs] scatter 分组后的 token id
//       tokenSuperPodInfo   — [expertPerRank * bs * superPodNum] token 的目标 pod 信息
//       expertSectionTokenCnt — [expertPerRank] 各分区已写入的 token 数（原子自增）
//
// 执行方式：<<<1, THREAD_COUNT, 1>>> 启动 128 个 lane。
//   Step 1: 初始化 expertSectionTokenCnt 为 0（对应 .asc 第 223-226 行）
//   Step 2: 拷贝 topkIndex 到 topkIndexLocal（对应 .asc 第 230-236 行）
//   Step 3: 初始化 dstPodLocal 为 0（对应 .asc 第 239-242 行）
//   Step 4: stride 模式遍历 token，计算 minLocalExpId + pod info，scatter 写入（对应 .asc 第 246-269 行）
//
// 分支说明（对应 .asc 的 #if 0 / #else 结构）：
//   #if 0  版本：功能正确版，有 return 提前退出，只做初始化不执行 scatter（用于隔离测试）
//   #else  版本：完整版，去掉了 return，完整执行初始化 + scatter
//   当前启用 #else 版本（完整版），与 .asc 保持一致。
//
// 注意：此路径在 __linx / __cpu_sim__ 下不编译，v300 仿真器不支持。
// ============================================================================
#if 0
// ---------------------------------------------------------------------------
// 版本 A（#if 0）：功能正确版，DoAtomicAdd 要使能。
// 有 return 提前退出，只做 Step 1-2（初始化 + 拷贝），不执行 Step 3-4。
// 纯粹为了隔离测试初始化逻辑，不产出 scatter 结果。
// 对应 .asc 第 148-210 行。
// ---------------------------------------------------------------------------
template <bool DoAtomicAdd>
void __vec__ GroupToken_Vec_Impl(
    const uint32_t* __in__ topkIndex,
    uint32_t* __out__ groupedTokenIds,
    uint32_t* __out__ tokenSuperPodInfo,
    uint32_t* __out__ expertSectionTokenCnt,
    uint32_t* __out__ topkIndexLocal,
    uint32_t* __out__ dstPodLocal,
    uint32_t batchSize, uint32_t topk,
    uint32_t expertPerRank, uint32_t expertPerPod, uint32_t superPodNum)
{
    uint32_t tid = blkv_get_index_y();
    uint32_t stride = blockDim.x;
    uint32_t begin = tid;

    // Step 1: 初始化各本地专家分区的已写入 token_id 数为 0（对应 .asc 第 161-164 行）
    for (uint32_t i = begin; i < expertPerRank; i += stride) {
        expertSectionTokenCnt[i] = 0;
    }
    asc_syncthreads();

    // Step 2: 初始化本线程块负责处理的 token 对应的 topkidx（对应 .asc 第 166-174 行）
    uint32_t topkIndexNum = batchSize * topk;
    for (uint32_t i = begin; i < topkIndexNum; i += stride) {
        topkIndexLocal[i] = topkIndex[i];
    }
    asc_syncthreads();

    return;

    // 以下代码在 #if 0 版本中不可达（return 之后），仅保留用于对照。
    // 初始化各线程所处理 token 的目的超节点信息
    for (uint32_t i = tid; i < blockDim.x * superPodNum; i += stride) {
        dstPodLocal[i] = 0;
    }
    asc_syncthreads();

    // 各线程将 token id 及 token 目标超节点信息写入相应输出张量的本地专家分区
    uint32_t localPodInfoOffset = tid * superPodNum;
    for (uint32_t i = begin; i < batchSize; i += stride) {
        uint32_t minLocalExpId = expertPerRank;
        uint32_t stop = (i + 1) * topk;
        for (uint32_t j = i * topk; j < stop; j++) {
            uint32_t curLocalExpId = topkIndexLocal[j] % expertPerRank;
            minLocalExpId = llmin(minLocalExpId, curLocalExpId);
            uint32_t curDstPod = topkIndexLocal[j] / expertPerPod;
            if (curDstPod < superPodNum) {
                dstPodLocal[localPodInfoOffset + curDstPod] = 1;
            }
        }

        uint32_t idxInSection;
        if constexpr (DoAtomicAdd) {
            idxInSection = asc_atomic_add(expertSectionTokenCnt + minLocalExpId, 1);
        } else {
            idxInSection = expertSectionTokenCnt[minLocalExpId] + 1;
        }
        groupedTokenIds[minLocalExpId * batchSize + idxInSection] = i;

        uint32_t podInfoSectionOffset = minLocalExpId * batchSize * superPodNum
                                     + idxInSection * superPodNum;
        for (uint32_t j = 0; j < superPodNum; j++) {
            tokenSuperPodInfo[podInfoSectionOffset + j] = dstPodLocal[localPodInfoOffset + j];
            dstPodLocal[localPodInfoOffset + j] = 0;
        }
    }
    asc_syncthreads();
}
#else
// ---------------------------------------------------------------------------
// 版本 B（#else）：完整版，去掉了 return，完整执行 Step 1-4。
// DoAtomicAdd=true 时用 asc_atomic_add 原子写指针（功能正确）。
// DoAtomicAdd=false 时用普通自增（纯粹为了测性能，结果不正确）。
// 对应 .asc 第 212-270 行。当前启用此版本。
// ---------------------------------------------------------------------------
template <bool DoAtomicAdd>
void __vec__ GroupToken_Vec_Impl(
    const uint32_t* __in__ topkIndex,
    uint32_t* __out__ groupedTokenIds,
    uint32_t* __out__ tokenSuperPodInfo,
    uint32_t* __out__ expertSectionTokenCnt,
    uint32_t* __out__ topkIndexLocal,
    uint32_t* __out__ dstPodLocal,
    uint32_t batchSize, uint32_t topk,
    uint32_t expertPerRank, uint32_t expertPerPod, uint32_t superPodNum)
{
    uint32_t tid = blkv_get_index_y();
    uint32_t stride = blockDim.x;
    uint32_t begin = tid;

    // Step 1: 初始化各本地专家分区的已写入 token_id 数为 0（对应 .asc 第 223-226 行）
    for (uint32_t i = begin; i < expertPerRank; i += stride) {
        expertSectionTokenCnt[i] = 0;
    }
    asc_syncthreads();

    // Step 2: 拷贝 topkIndex 到局部缓冲区 topkIndexLocal（对应 .asc 第 230-236 行）
    uint32_t topkIndexNum = batchSize * topk;
    for (uint32_t i = begin; i < topkIndexNum; i += stride) {
        topkIndexLocal[i] = topkIndex[i];
    }
    asc_syncthreads();

    // Step 3: 初始化各线程的 pod 信息缓冲区为 0（对应 .asc 第 239-242 行）
    uint32_t localPodInfoOffset = tid * superPodNum;
    for (uint32_t i = tid; i < blockDim.x * superPodNum; i += stride) {
        dstPodLocal[i] = 0;
    }
    asc_syncthreads();

    // Step 4: 各线程将 token id 及 token 目标超节点信息写入相应输出张量的本地专家分区
    //        （对应 .asc 第 246-269 行）
    for (uint32_t i = begin; i < batchSize; i += stride) {
        uint32_t minLocalExpId = expertPerRank;
        uint32_t stop = (i + 1) * topk;
        for (uint32_t j = i * topk; j < stop; j++) {
            uint32_t curLocalExpId = topkIndexLocal[j] % expertPerRank;
            minLocalExpId = llmin(minLocalExpId, curLocalExpId);
            uint32_t curDstPod = topkIndexLocal[j] / expertPerPod;
            if (curDstPod < superPodNum) {
                dstPodLocal[localPodInfoOffset + curDstPod] = 1;
            }
        }

        uint32_t idxInSection;
        if constexpr (DoAtomicAdd) {
            idxInSection = asc_atomic_add(expertSectionTokenCnt + minLocalExpId, 1);
        } else {
            idxInSection = expertSectionTokenCnt[minLocalExpId] + 1;
        }
        groupedTokenIds[minLocalExpId * batchSize + idxInSection] = i;

        uint32_t podInfoSectionOffset = minLocalExpId * batchSize * superPodNum
                                     + idxInSection * superPodNum;
        for (uint32_t j = 0; j < superPodNum; j++) {
            tokenSuperPodInfo[podInfoSectionOffset + j] = dstPodLocal[localPodInfoOffset + j];
            dstPodLocal[localPodInfoOffset + j] = 0;
        }
    }
    asc_syncthreads();
}
#endif

// ============================================================================
// Phase 2 (SIMT): GroupToken_Impl
//
// 功能：Phase 2 SIMT 启动函数，启动 128 个 lane 执行 GroupToken_Vec_Impl。
//       对应 .asc 中 asc_vf_call<GroupToken<DoAtomicAdd>>(dim3(THREAD_COUNT), ...) 的调用。
//
// 输入：topkIndex           — topkIndex 数组指针
//       topkIndexLocal     — [bs*k] 局部缓冲区，拷贝 topkIndex
//       dstPodLocal        — [blockDim.x * superPodNum] pod 信息缓冲区
//       batchSize, topk, expertPerRank, expertPerPod, superPodNum — MoE 拓扑参数
// 输出：groupedTokenIds, tokenSuperPodInfo, expertSectionTokenCnt
//
// 注意：此路径在 __linx / __cpu_sim__ 下不编译，v300 仿真器不支持。
// ============================================================================
template <bool DoAtomicAdd>
void GroupToken_Impl(const uint32_t* topkIndex,
                     uint32_t* groupedTokenIds,
                     uint32_t* tokenSuperPodInfo,
                     uint32_t* expertSectionTokenCnt,
                     uint32_t* topkIndexLocal,
                     uint32_t* dstPodLocal,
                     uint32_t batchSize, uint32_t topk,
                     uint32_t expertPerRank, uint32_t expertPerPod, uint32_t superPodNum)
{
    GroupToken_Vec_Impl<DoAtomicAdd>
        <<<1, 128, 1>>>(topkIndex, groupedTokenIds, tokenSuperPodInfo,
                        expertSectionTokenCnt, topkIndexLocal, dstPodLocal,
                        batchSize, topk,
                        expertPerRank, expertPerPod, superPodNum);
}

// ============================================================================
// Phase 3b (SIMT): SortByLocalExpId_Vec_Impl
//
// 功能：SIMT vector kernel，每个 lane 负责一个分区（section），将属于该分区
//       的 token id 写入 sortedTokenIds 的对应连续区间。
//       对应 .asc 中 GroupTokenKernel 内 Phase 3 的 counting sort 的 scatter 步骤。
//
// 输入：minLocalExpIds  — [bs] 每个 token 的最小本地 expert id
//       sectionStarts   — [expertPerRank+1] 各分区起始边界（已由 host 计算）
//       batchSize       — token 总数
//       expertPerRank   — 每个 rank 的本地 expert 数（= 分区数 = lane 数）
// 输出：sortedTokenIds  — [bs] 按分区连续排列的 token id 列表
//
// 执行方式：<<<1, expertPerRank, 1>>> 启动 expertPerRank 个 lane，每个 lane
//           负责一个分区，扫描全部 token 将属于自己分区的写入对应位置。
//
// 注意：此路径在 __linx / __cpu_sim__ 下不编译，v300 仿真器不支持。
// ============================================================================
template <typename tile_shape_out>
void __vec__ SortByLocalExpId_Vec_Impl(
    uint32_t* __out__ sortedTokenIds,
    const uint32_t* __in__ minLocalExpIds,
    const uint32_t* __in__ sectionStarts,
    uint32_t batchSize, uint32_t expertPerRank)
{
    uint32_t section = blkv_get_index_y();
    uint32_t writePos = sectionStarts[section];
    for (uint32_t i = 0; i < batchSize; i++) {
        if (minLocalExpIds[i] == section) {
            sortedTokenIds[writePos++] = i;
        }
    }
}

// ============================================================================
// Phase 3b (SIMT): SortByLocalExpId_Impl
//
// 功能：Phase 3b SIMT 启动函数。先由 host 标量计算 counts 和 sectionStarts，
//       再启动 expertPerRank 个 lane 并行 scatter token id 到各分区。
//
// 注意：此路径在 __linx / __cpu_sim__ 下不编译，v300 仿真器不支持。
// ============================================================================
template <typename tile_shape_out>
void SortByLocalExpId_Impl(uint32_t* sortedTokenIds,
                           const uint32_t* minLocalExpIds,
                           uint32_t* sectionStarts,
                           uint32_t batchSize, uint32_t expertPerRank)
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

    SortByLocalExpId_Vec_Impl<tile_shape_out>
        <<<1, expertPerRank, 1>>>(sortedTokenIds, minLocalExpIds,
                                   sectionStarts, batchSize, expertPerRank);
}

#endif // !__linx && !__cpu_sim__

// ============================================================================
// Phase 2 (scalar): GroupToken — scatter token id into expert sections
//
// 功能：对每个 token，计算其 top-k expert 中最小的本地 expert id
//      （expertId % expertPerRank），然后将 token id scatter 到对应的
//       expert 分区，同时记录该 token 需要发往哪些 super pod。
//       对应 .asc 中 __simt_vf__ GroupToken 的标量保底实现。
//
// 输入：topkIndex           — [bs*k] 每个 token 选中的 top-k expert id 列表
//       batchSize           — token 总数（512）
//       topk                — 每个 token 选中的 expert 数（16）
//       expertPerRank       — 每个 rank 的本地 expert 数（4）
//       expertPerPod        — 每个 pod 的 expert 数（64）
//       superPodNum         — super pod 数量（2）
// 输出：groupedTokenIds     — [expertPerRank * bs] scatter 分组后的 token id
//       tokenSuperPodInfo   — [expertPerRank * bs * superPodNum] 每个 token 的目标 pod 信息
//       expertSectionTokenCnt — [expertPerRank] 各分区已写入的 token 数
//
// 执行方式：单线程标量 for 循环。DoAtomicAdd=true 时用自增写指针模拟
//           .asc 中的 asc_atomic_add 原子操作。
// ============================================================================
template <bool DoAtomicAdd>
static inline void groupToken_scalar(const uint32_t *topkIndex,
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
    for (uint32_t i = 0; i < superPodNum; i++) {
        dstPodLocal[i] = 0;
    }

    for (uint32_t i = 0; i < batchSize; i++) {
        uint32_t minLocalExpId = expertPerRank;
        uint32_t stop = (i + 1) * topk;
        for (uint32_t j = i * topk; j < stop; j++) {
            uint32_t curLocalExpId = topkIndex[j] % expertPerRank;
            if (curLocalExpId < minLocalExpId) {
                minLocalExpId = curLocalExpId;
            }
            uint32_t curDstPod = topkIndex[j] >> 6;
            dstPodLocal[curDstPod] = 1;
        }
        uint32_t idxInSection;
        if constexpr (DoAtomicAdd) {
            idxInSection = expertSectionTokenCnt[minLocalExpId]++;
        } else {
            idxInSection = expertSectionTokenCnt[minLocalExpId] + 1;
        }
        groupedTokenIds[minLocalExpId * batchSize + idxInSection] = i;
        uint32_t podInfoSectionOffset = minLocalExpId * batchSize * superPodNum + idxInSection * superPodNum;
        for (uint32_t j = 0; j < superPodNum; j++) {
            tokenSuperPodInfo[podInfoSectionOffset + j] = dstPodLocal[j];
            dstPodLocal[j] = 0;
        }
    }
}

// ============================================================================
// Phase 3 (scalar): FloorFunc — compute per-token minLocalExpId
//
// 功能：计算每个 token 的 top-k expert 中最小的本地 expert id
//      （minLocalExpId = min(topkIndex[i*k..] % expertPerRank)）。
//       这是 counting sort 之前的预处理步骤，用于确定每个 token 属于哪个分区。
//       对应 .asc 中 GroupTokenKernel 内 Phase 3 的 FloorFunc 部分。
//
// 输入：topkIndex      — [bs*k] 每个 token 选中的 top-k expert id 列表
//       batchSize      — token 总数（512）
//       topk           — 每个 token 选中的 expert 数（16）
//       expertPerRank  — 每个 rank 的本地 expert 数（4）
// 输出：minLocalExpIds — [bs] 每个 token 的最小本地 expert id
//
// 执行方式：单线程标量 for 循环，顺序遍历每个 token 的 top-k expert。
// ============================================================================
static inline void floorFunc_scalar(const uint32_t *topkIndex,
                                     uint32_t *minLocalExpIds,
                                     uint32_t batchSize,
                                     uint32_t topk,
                                     uint32_t expertPerRank)
{
    for (uint32_t i = 0; i < batchSize; i++) {
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
}

// ============================================================================
// Phase 3 (scalar): SortByLocalExpId — stable sort token ids by minLocalExpId
//
// 功能：按 minLocalExpId 对 token id 做稳定排序（counting sort），生成连续
//       分区的 token id 列表。将 Phase 2 中散布的分组输出替换为紧凑的、
//       按分区排列的数组。
//       对应 .asc 中 GroupTokenKernel 内 Phase 3 的 counting sort 部分。
//
// 输入：minLocalExpIds  — [bs] 每个 token 的最小本地 expert id（FloorFunc 的输出）
//       batchSize       — token 总数（512）
//       expertPerRank   — 每个 rank 的本地 expert 数（4），即排序桶数
// 输出：sortedTokenIds  — [bs] 按 minLocalExpId 排序后的 token id 列表（连续分区）
//       sectionStarts   — [expertPerRank+1] 各分区的起始偏移边界
//
// 执行方式：单线程标量 counting sort（O(n)），桶数 = expertPerRank（4）。
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

#if !defined(__linx) && !defined(__cpu_sim__)

// ============================================================================
// Phase 3 (SIMT): FloorFunc_Vec_Impl
//
// 功能：SIMT vector kernel，每个 lane 以 stride 模式处理一部分 token，
//       计算每个 token 的 minLocalExpId，结果写入 tile 的对应位置。
//       对应 .asc 中 FloorFunc 的 PTO-ISA SIMT 实现。
//
// 输入：topkIndex      — [bs*k] 每个 token 选中的 top-k expert id 列表
//       batchSize      — token 总数
//       topk           — 每个 token 选中的 expert 数
//       expertPerRank  — 每个 rank 的本地 expert 数
// 输出：dst            — tile 寄存器，第 i 个元素存储 token i 的 minLocalExpId
//
// 执行方式：<<<1, 128, 1>>> 启动 128 个 lane，stride 模式跨步遍历 token。
//
// 注意：此路径在 __linx / __cpu_sim__ 下不编译，v300 仿真器不支持。
// ============================================================================
template <typename tile_shape_out>
void __vec__ FloorFunc_Vec_Impl(
    typename tile_shape_out::TileDType __out__ dst,
    const uint32_t* __in__ topkIndex,
    uint32_t batchSize, uint32_t topk, uint32_t expertPerRank)
{
    uint32_t tid = blkv_get_index_y();
    uint32_t stride = blockDim.x;
    for (uint32_t i = tid; i < batchSize; i += stride) {
        uint32_t minLocal = expertPerRank;
        for (uint32_t j = 0; j < topk; j++) {
            uint32_t cur = topkIndex[i * topk + j] % expertPerRank;
            if (cur < minLocal) minLocal = cur;
        }
        blkv_get_tile_ptr(dst)[i] = minLocal;
    }
}

// ============================================================================
// Phase 3 (SIMT): FloorFunc_Impl
//
// 功能：Phase 3a SIMT 启动函数，启动 128 个 lane 执行 FloorFunc_Vec_Impl。
//       对应 .asc 中 asc_vf_call<FloorFunc>(dim3(128), ...) 的调用。
//
// 输入：topkIndex      — topkIndex 数组指针
//       batchSize      — token 总数
//       topk           — 每个 token 选中的 expert 数
//       expertPerRank  — 每个 rank 的本地 expert 数
// 输出：dst            — tile 寄存器，存储每个 token 的 minLocalExpId
//
// 执行方式：<<<1, 128, 1>>> 启动 1 个 block、128 个 lane。
//
// 注意：此路径在 __linx / __cpu_sim__ 下不编译，v300 仿真器不支持。
// ============================================================================
template <typename tile_shape_out>
void FloorFunc_Impl(tile_shape_out& dst, const uint32_t* topkIndex,
                     uint32_t batchSize, uint32_t topk, uint32_t expertPerRank)
{
    FloorFunc_Vec_Impl<tile_shape_out>
        <<<1, 128, 1>>>(dst.data(), topkIndex, batchSize, topk, expertPerRank);
}

#endif // !__linx && !__cpu_sim__

// ============================================================================
// 便捷入口函数：依次执行三个 Phase 的完整 MoE Dispatch 流程。
//
// 功能：按顺序调用 Phase 1（直方图）、Phase 2（scatter 分组）、
//       Phase 3（FloorFunc + counting sort），完成 token 分组和排序。
//       对应 .asc 中 GroupTokenKernel 的完整执行流程。
//
// 输入：topkIndex — [bs*k] 每个 token 选中的 top-k expert id 列表
// 输出：tokenPerExpertCnt     — [expertNum] 每个 expert 的 token 计数（Phase 1）
//       groupedTokenIds      — [expertPerRank*bs] scatter 分组后的 token id（Phase 2）
//       tokenSuperPodInfo    — [expertPerRank*bs*superPodNum] token 的目标 pod 信息（Phase 2）
//       expertSectionTokenCnt — [expertPerRank] 各分区 token 数（Phase 2）
//       sortedTokenIds       — [bs] 排序后的连续分区 token id 列表（Phase 3）
//       sectionStarts        — [expertPerRank+1] 各分区起始边界（Phase 3）
//
// 执行方式：
//   - __linx / __cpu_sim__ 环境（v300 仿真器）：调用标量保底路径
//   - 真实硬件环境：调用 SIMT 加速路径（128 lane 并行）
// ============================================================================
static inline void runGroupTokenOld(const uint32_t *topkIndex,
                                     uint32_t *tokenPerExpertCnt,
                                     uint32_t *groupedTokenIds,
                                     uint32_t *tokenSuperPodInfo,
                                     uint32_t *expertSectionTokenCnt,
                                     uint32_t *sortedTokenIds,
                                     uint32_t *sectionStarts)
{
#if defined(__linx) || defined(__cpu_sim__)
    // ========================================================================
    // 标量路径（v300 仿真器 / CPU 模拟器）
    // 单线程顺序执行，保证功能正确性验证。
    // ========================================================================

    // Phase 1: histogram
    calTokenPerExpertCnt_scalar(topkIndex, tokenPerExpertCnt,
                                 kExpertNum, kTopKEleNum);
    // Phase 2: scatter (atomic-add write pointer)
    groupToken_scalar<true>(topkIndex, groupedTokenIds, tokenSuperPodInfo,
                             expertSectionTokenCnt,
                             kBS, kTopK, kExpertPerRank, kExpertPerPod, kSuperPodNum);
    // Phase 3: FloorFunc + Sort (counting sort)
    static uint32_t minLocalExpIds[kBS];
    floorFunc_scalar(topkIndex, minLocalExpIds, kBS, kTopK, kExpertPerRank);
    sortByLocalExpId_scalar(minLocalExpIds, sortedTokenIds, sectionStarts,
                             kBS, kExpertPerRank);

#else
    // ========================================================================
    // SIMT 加速路径（真实 PTO-ISA 硬件）
    // 128 lane 并行执行，对应 .asc 中 asc_vf_call 的调用方式。
    // ========================================================================

    // Phase 1: histogram (128 lane, stride + atomic_add, cntLocal 局部归约)
    static uint32_t cntLocal[kExpertNum];
    CalTokenPerExpertCnt_Impl<TileU32>(reinterpret_cast<uint32_t*>(tokenPerExpertCnt),
                                        topkIndex, kTopKEleNum, kExpertNum, cntLocal);
    // Phase 2: scatter (128 lane, stride, topkIndexLocal 拷贝 + atomic_add 写指针)
    static uint32_t topkIndexLocal[kTopKEleNum];
    static uint32_t dstPodLocal[128 * kSuperPodNum];
    GroupToken_Impl<true>(topkIndex, groupedTokenIds, tokenSuperPodInfo,
                          expertSectionTokenCnt, topkIndexLocal, dstPodLocal,
                          kBS, kTopK, kExpertPerRank, kExpertPerPod, kSuperPodNum);
    // Phase 3a: FloorFunc (128 lane, stride 模式计算 minLocalExpId)
    static uint32_t minLocalExpIds[kBS];
    FloorFunc_Impl<TileU32>(reinterpret_cast<uint32_t*>(minLocalExpIds),
                            topkIndex, kBS, kTopK, kExpertPerRank);
    // Phase 3b: SortByLocalExpId (expertPerRank lane, 每 lane 负责一个分区)
    SortByLocalExpId_Impl<TileU32>(sortedTokenIds, minLocalExpIds,
                                   sectionStarts, kBS, kExpertPerRank);

#endif
}

#endif // GROUP_TOKEN_OLD_HPP
