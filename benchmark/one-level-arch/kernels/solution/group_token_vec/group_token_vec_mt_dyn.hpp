#ifndef GROUP_TOKEN_VEC_MT_DYN_HPP
#define GROUP_TOKEN_VEC_MT_DYN_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>

#include "solution/group_token_vec/gt_tile_common.hpp"

// ============================================================================
// MoE Token Grouping — Multi-thread Vector (Tile) variant, runtime DYNAMIC
// SHAPE version
//
// 动态 shape 版 (语义与分片模型同 group_token_vec_mt.hpp): bs/topK/
// expertPerRank/expertPerPod/superPodNum 运行期由 tiling 传入; tile 物理
// 形状仍编译期锁定 (契约 [C4]/[C6]/[C8]/[C12] 见 gt_tile_common.hpp)。
// 直方图 = 计数链 [C10]; rank = 成对比较 [C11] + 平 MGATHER/MSCATTER;
// 原子族等价链替换 (TimingSim TLSU 无 GM 原子完成路径)。tile 路径仅
// topK == 16 / epr ≤ 8 / spn ≤ 2 时启用, 违例与尾块走标量兜底;
// 跨 PE 交接由 mtBarrier 单调相位保护 (per-PE 调用计数, 免跨 PE 复位)。
// ============================================================================

constexpr uint32_t kThreadsPerBlock = 4;
constexpr uint32_t kTileM = 16;   // rows per TMA block (4 rows per PE)
constexpr uint32_t kTileN = 16;   // tile 列宽; tile 路径仅 topK == kTileN
// tile 路径的运行时参数容量上限 (静态 scratch 尺寸; 违例走标量兜底):
constexpr uint32_t kEprCap = 8;   // expertPerRank ≤ 8 (driver kExpertPerRankMax)
constexpr uint32_t kSpnCap = 2;   // superPodNum ≤ 2 (driver kSuperPodNumMax)
constexpr uint32_t kExpertNumCap = 256;  // expertNum ≤ 256 (计数链 acc 栈数组)

// ============================================================================
// Multi-PE barrier —— 单行 flag + 寄存器延迟链 + 稀疏定向集驱逐
// (时序模型跨 PE 无 snoop: flag 行须主动驱逐使 L1D miss 才能看到新值;
// 驱逐流量须稀疏, 否则 prior 洪泛饿死 L2 tile 派发)。gfrun 无缓存, 语义同
// 原始屏障。
// ============================================================================
alignas(16384) static volatile uint32_t sEvictSpan[6 * 4096];   // 96KB 驱逐区
alignas(64) static volatile uint32_t sPhaseDone[kThreadsPerBlock];  // 4 flag 同一行

static inline void mtCompilerBarrier()
{
    __asm__ volatile("" : : : "memory");
}

static inline void mtBarrier(uint32_t phase)
{
    mtCompilerBarrier();
    const uint32_t tid = get_thread_idx();
    sPhaseDone[tid] = phase;
    mtCompilerBarrier();
    // 写方到达驱逐: 写方从不自旋, 其脏 flag 行不会经自旋被写回 —— 立即
    // 定向驱逐一次, 使 L2 尽早看到到达 store (等待方的稀疏驱逐 poll 才有
    // 新值可见; 原始设计靠写方自旋期驱逐, 最后到达者无自旋 → 缺这一步)
    const uint32_t wordOffW =
        (static_cast<uint32_t>(
             reinterpret_cast<uint64_t>(&sPhaseDone[0]) >> 2)) & 4095u;
    for (uint32_t k = 1; k <= 5; ++k) {
        (void)sEvictSpan[k * 4096u + wordOffW];
    }
    mtCompilerBarrier();
    // flag 行的 L1D set 内偏移 (字粒度): sEvictSpan 基址 16KB 对齐 →
    // sEvictSpan[k*4096 + wordOff] 与 sPhaseDone[0] 同 set (k 任意)
    const uint32_t wordOff =
        (static_cast<uint32_t>(
             reinterpret_cast<uint64_t>(&sPhaseDone[0]) >> 2)) & 4095u;
    // 定向集驱逐自旋 + 跳自槽: 纯 plain 热自旋在长偏斜下读陈旧 flag 活锁,
    // 须周期性驱逐 flag 行所在 set 迫使 miss 重取新值; 跳自槽消除同地址
    // store→load 对。
    for (uint32_t t = 0; t < kThreadsPerBlock; ++t) {
        if (t == tid) {
            continue;
        }
        uint32_t spins = 0x9E3779B9u ^ (phase * 2654435761u) ^ (t * 0x85EBCA6Bu);
        while (sPhaseDone[t] < phase) {
            // 寄存器驻留延迟链 + 稀疏驱逐: 驱逐读是 prior 流量, 过密会饿死
            // L2 tile 派发, 故延迟链全程寄存器 (零内存流量) + 驱逐降频
            // 1/64 轮; 可见性由写方到达驱逐 + 稀疏驱逐后的 poll miss 保证。
            spins = spins * 2654435761u + 0x2545F491u;
            spins = spins * 2654435761u + 0x2545F492u;
            spins = spins * 2654435761u + 0x2545F493u;
            spins = spins * 2654435761u + 0x2545F494u;
            spins = spins * 2654435761u + 0x2545F495u;
            spins = spins * 2654435761u + 0x2545F496u;
            spins = spins * 2654435761u + 0x2545F497u;
            spins = spins * 2654435761u + 0x2545F498u;
            spins = spins * 2654435761u + 0x2545F499u;
            spins = spins * 2654435761u + 0x2545F49Au;
            spins = spins * 2654435761u + 0x2545F49Bu;
            spins = spins * 2654435761u + 0x2545F49Cu;
            spins = spins * 2654435761u + 0x2545F49Du;
            spins = spins * 2654435761u + 0x2545F49Eu;
            spins = spins * 2654435761u + 0x2545F49Fu;
            spins = spins * 2654435761u + 0x2545F4A0u;
            __asm__ volatile("" : "+r"(spins));
            if ((spins & 63u) == 0u) {
                for (uint32_t k = 1; k <= 5; ++k) {
                    (void)sEvictSpan[k * 4096u + wordOff];   // 同 set 驱逐
                }
            }
        }
    }
    mtCompilerBarrier();
}

// ============================================================================
// Phase 1 (runtime shape): 满块标量计数 (hybrid) + 尾块标量兜底; 归约按
// 专家段分片 (≤64/PE, 标量读回)。全 tile 计数链保留给静态 mt 版 (dyn
// 形状下回边 TMOV 描述符断言/依赖 TLOAD 超时, 见静态版注释)。
// ============================================================================
static inline void calTokenPerExpertCnt_mt_tile_dyn(
    uint32_t *topkIndex,
    uint32_t *tokenPerExpertCnt,
    uint32_t *cntLocal,
    uint32_t expertNum,
    uint32_t bs,
    uint32_t topK,
    uint32_t histBarrierPhase)
{
    using namespace gt_tile;
    const uint32_t tid = get_thread_idx();

    uint32_t *myCnt = cntLocal + tid * expertNum;
    // 清零用 volatile 标量: 常量 TEXPANDS(0)+TSTORE 被后端折叠为零寄存器
    // 别名绑定, cfgB 重复调用时 TSTORE 永不完成 ([C9])。
    {
        volatile uint32_t *v = myCnt;
        for (uint32_t i = 0; i < expertNum; i++) v[i] = 0u;
    }

    if (topK == kTileN && expertNum <= kExpertNumCap) {
        using GmPerPE = global_tensor<uint32_t, RowMajor<-1, -1>>;

        const uint32_t fullBlocks = bs / kTileM;
        // dyn 满块直方图 = 标量计数 (hybrid; 全 tile 计数链在 dyn 形状下
        // 受模型 TMOV 断言/依赖 TLOAD 超时限制, 静态版保留全 tile 链)。
        // 满块 TLOAD 预取已删除 (无消费者, 语义不变, 块数下降)。
        for (uint32_t blk = 0; blk < fullBlocks; ++blk) {
            // rows [16*blk + 4*tid, +4): 标量计数 (hybrid)
            const uint32_t r0 = blk * kTileM + tid * 4U;
            for (uint32_t row = 0; row < 4U; ++row) {
                const uint32_t base = (r0 + row) * topK;
                for (uint32_t col = 0; col < topK; ++col) {
                    const uint32_t expertId = topkIndex[base + col];
                    if (expertId < expertNum) {
                        myCnt[expertId]++;
                    }
                }
            }
        }
        // 尾块: 行域 [fullBlocks*16, bs), PE tid 拥有 [4*tid, +4) 切片。
        // 动态 valid 计数链泄漏 ([C6]) → TLOAD 预取 + 标量直方图 (原实现)。
        const uint32_t tailRows = bs - fullBlocks * kTileM;
        if (tailRows > 0) {
            const uint32_t r0 = fullBlocks * kTileM + tid * 4U;
            const uint32_t vr = (r0 < bs) ? ((bs - r0 < 4U) ? (bs - r0) : 4U) : 0U;
            if (vr > 0) {
                // 原 dyn-valid TLOAD 预取已删除: 动态 ValidRow tile 的描述符
                // 会污染复用 tile 寄存器, 循环回边 TMOV bank 重排束触发模型
                // "Local TMOV requires matching descriptors" 断言 (gfsim);
                // 预取无消费价值 (数据由标量经 GM 读), 删除后语义不变。
                {
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

    // 直方图导出完成后才可读其它 PE 的 cntLocal (否则为真实跨 PE 竞争)。
    mtBarrier(histBarrierPhase);

    // Reduce: PE tid 负责专家段 [tid*epn, +epn) —— 标量读回 ([C14]:
    // 屏障后的 scalar store → tile load 即时读触发 TMA 管线停摆)。
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
// dyn noinline tile 分体: 每个函数完整包含 tile 链, 寄存器不跨函数边界,
// 函数间只经 GM 交接 (消除 raw tile spill 与跨形状回边 TMOV 断言)。
// ============================================================================

// P2-ab (合并分体): minLocalExpId + pod any-flag —— 单次 TLOAD [4×16] 供
// 给两条链 (min: TREMS→TROWMIN→TSTORE; pod: TDIVS + 每 p 守卫直线段
// TCMPS<EQ>→TSEL→TROWMAX→TSTORE)。合并后每 PE 每块 1 条 TLOAD (原
// p2_min+pod_flags 分体为 3 条同址读)。
__attribute__((noinline)) static inline void gt_dyn_p2_min_pods(
    const uint32_t *topkIndex, uint32_t r0, uint32_t topk, uint32_t batchSize,
    uint32_t expertPerRank, uint32_t expertPerPod, uint32_t superPodNum,
    uint32_t *minScratchOut, uint32_t *podFlagOut)
{
    using namespace gt_tile;
    global_tensor<uint32_t, RowMajor<-1, -1>> gIn(
        const_cast<uint32_t *>(topkIndex) + r0 * topk,
        static_cast<int>(batchSize), static_cast<int>(topk));
    T4x16 t;
    TLOAD(t, gIn);
    T4x16 rem;
    TREMS(rem, t, expertPerRank);
    TRed4 minCol;
    TROWMIN(minCol, rem);
    G4x1 gMinW(minScratchOut);
    TSTORE(gMinW, minCol);
    T4x16 pod;
    TDIVS(pod, t, expertPerPod);
    if (superPodNum > 0U) {
        T4x16 pred;
        TCMPS<CmpMode::EQ>(pred, pod, 0U);
        T4x16 onev;
        TEXPANDS(onev, 1u);
        T4x16 sel;
        TEXPANDS(sel, 0u);
        TSEL(sel, pred, onev);
        TRed4 flagCol;
        TROWMAX(flagCol, sel);
        G4x1 gFlagW(podFlagOut);
        TSTORE(gFlagW, flagCol);
    }
    if (superPodNum > 1U) {
        T4x16 pred;
        TCMPS<CmpMode::EQ>(pred, pod, 1U);
        T4x16 onev;
        TEXPANDS(onev, 1u);
        T4x16 sel;
        TEXPANDS(sel, 0u);
        TSEL(sel, pred, onev);
        TRed4 flagCol;
        TROWMAX(flagCol, sel);
        G4x1 gFlagW(podFlagOut + 4u);
        TSTORE(gFlagW, flagCol);
    }
}

// P2-c: 成对 rank + groupedIds 散射 + offE 导出 (GM 交接给 podinfo)
//   offE = min*sectStride + peBase + MGATHER(sectionCnt) + (rankIncl-1)
__attribute__((noinline)) static inline void gt_dyn_p2_rank_scatter(
    const uint32_t *minScratch, uint32_t *sectionCnt, uint32_t sectStride,
    uint32_t peBase, uint32_t tokBase, uint32_t *groupedIds, uint32_t idsLen,
    uint32_t *rankScratch, uint32_t *offEScratch)
{
    using namespace gt_tile;
    G4x1 gMinC(const_cast<uint32_t *>(minScratch));
    TRed4 minCol;
    TLOAD(minCol, gMinC);                 // [4×1] 列视图
    G1x4 gMinR(const_cast<uint32_t *>(minScratch));
    TCol4 minRow;
    TLOAD(minRow, gMinR);                 // [1×4] 行视图 (同 GM 数据)

    T4x4 Mc;
    TROWEXPAND(Mc, minCol);               // Mc[i][j] = min[i]
    T4x4 Mr;
    TCOLEXPAND(Mr, minRow);               // Mr[i][j] = min[j]
    T4x4 eq;
    TCMP<CmpMode::EQ>(eq, Mc, Mr);
    T4x4 tri;
    TTRI(tri);
    T4x4 mat;
    TEXPANDS(mat, 0u);
    TSEL(mat, eq, tri);
    TRed4 rankIncl;
    TROWSUM(rankIncl, mat);
    G4x1 gRankW(rankScratch);
    TSTORE(gRankW, rankIncl);
    TCol4 rankRow;
    G1x4 gRankR(rankScratch);
    TLOAD(rankRow, gRankR);
    TSUBS(rankRow, rankRow, 1u);

    TCol4 minIdx;
    TSHLS(minIdx, minRow, 2u);
    global_tensor<uint32_t, RowMajor<1, kEprCap>> gCnt(sectionCnt);
    TCol4 base;
    MGATHER(base, gCnt, minIdx);
    TCol4 minB;
    TMULS(minB, minRow, sectStride);
    TADDS(minB, minB, peBase);
    TCol4 offE;
    TADD(offE, minB, base);
    TADD(offE, offE, rankRow);
    G4x1 gOffW(offEScratch);
    TSTORE(gOffW, offE);                  // GM 交接给 podinfo 分体
    TCol4 offB;
    TSHLS(offB, offE, 2u);
    TCol4 tok;
    TCI(tok, tokBase);
    GFlat gIds(groupedIds, static_cast<int>(idsLen), 1);
    MSCATTER(gIds, tok, offB);
}

// P2-d: podInfo 散射 — poE = offE*superPodNum + p
__attribute__((noinline)) static inline void gt_dyn_p2_podinfo(
    const uint32_t *offEScratch, const uint32_t *podFlagScratch,
    uint32_t superPodNum, uint32_t *podInfo, uint32_t podInfoLen)
{
    using namespace gt_tile;
    GFlat gPodInfo(podInfo, static_cast<int>(podInfoLen), 1);
    for (uint32_t p = 0; p < superPodNum; ++p) {
        TCol4 flagRow;
        G1x4 gFlagR(const_cast<uint32_t *>(podFlagScratch) + p * 4u);
        TLOAD(flagRow, gFlagR);
        TCol4 offE2;
        G4x1 gOffR(const_cast<uint32_t *>(offEScratch));
        TLOAD(offE2, gOffR);
        TCol4 po;
        TMULS(po, offE2, superPodNum);
        TADDS(po, po, p);
        TCol4 poB;
        TSHLS(poB, po, 2u);
        MSCATTER(gPodInfo, flagRow, poB);
    }
}

// P2-e: 块末进位 — sectionCnt[s] += #{本块 min==s} (计数链 [C10])
__attribute__((noinline)) static inline void gt_dyn_p2_carry(
    const uint32_t *minScratch, uint32_t expertPerRank, uint32_t *sectionCnt,
    uint32_t *cntGmOut)
{
    using namespace gt_tile;
    for (uint32_t s = 0; s < expertPerRank; ++s) {
        TCol4 minRow2;
        G1x4 gMinR2(const_cast<uint32_t *>(minScratch));
        TLOAD(minRow2, gMinR2);
        TCol4 pred2;
        TCMPS<CmpMode::EQ>(pred2, minRow2, s);
        TCol4 one2;
        TEXPANDS(one2, 1u);
        TCol4 sel2;
        TEXPANDS(sel2, 0u);
        TSEL(sel2, pred2, one2);
        TSum1 cnt;
        TROWSUM(cnt, sel2);
        G1x1 gC(cntGmOut);
        TSTORE(gC, cnt);
        volatile uint32_t *vc = sectionCnt;
        vc[s] = vc[s] + *cntGmOut;
    }
}

// P3b-a: 单 tile per-bin 计数 → cntGmArr[e] (调用方标量累加)
__attribute__((noinline)) static inline void gt_dyn_p3b_counts1(
    const uint32_t *src32, uint32_t expertPerRank, uint32_t *cntGmArr)
{
    using namespace gt_tile;
    for (uint32_t e = 0; e < expertPerRank; ++e) {
        G1x32 gS(const_cast<uint32_t *>(src32));
        T1x32 s2;
        TLOAD(s2, gS);
        T1x32 pred;
        TCMPS<CmpMode::EQ>(pred, s2, e);
        T1x32 one;
        TEXPANDS(one, 1u);
        T1x32 sel;
        TEXPANDS(sel, 0u);
        TSEL(sel, pred, one);
        TSum1 c;
        TROWSUM(c, sel);
        G1x1 gC(cntGmArr + e);
        TSTORE(gC, c);
    }
}

// P3b-b: 单 tile [32×32] 成对 rank + 平 MGATHER(writePos) + TCI + MSCATTER
__attribute__((noinline)) static inline void gt_dyn_p3b_scatter1(
    const uint32_t *src32, uint32_t *writePos, uint32_t *sortedTokenIds,
    uint32_t bsLen, uint32_t tokBase, uint32_t *rankBuf)
{
    using namespace gt_tile;
    G1x32 gS(const_cast<uint32_t *>(src32));
    T1x32 sRow;
    TLOAD(sRow, gS);
    G32x1 gSC(const_cast<uint32_t *>(src32));
    TRed32 sCol;
    TLOAD(sCol, gSC);
    T32x32 Mc;
    TROWEXPAND(Mc, sCol);
    T32x32 Mr;
    TCOLEXPAND(Mr, sRow);
    T32x32 eq;
    TCMP<CmpMode::EQ>(eq, Mc, Mr);
    T32x32 tri;
    TTRI(tri);
    T32x32 mat;
    TEXPANDS(mat, 0u);
    TSEL(mat, eq, tri);
    TRed32 rankIncl;
    TROWSUM(rankIncl, mat);
    G32x1 gRankW(rankBuf);
    TSTORE(gRankW, rankIncl);
    T1x32 rankRow;
    G1x32 gRankR(rankBuf);
    TLOAD(rankRow, gRankR);
    TSUBS(rankRow, rankRow, 1u);
    T1x32 sIdx;
    TSHLS(sIdx, sRow, 2u);
    global_tensor<uint32_t, RowMajor<1, kEprCap>> gWP(writePos);
    T1x32 base;
    MGATHER(base, gWP, sIdx);
    T1x32 pos;
    TADD(pos, base, rankRow);
    T1x32 posB;
    TSHLS(posB, pos, 2u);
    T1x32 tok;
    TCI(tok, tokBase);
    GFlat gSorted(sortedTokenIds, static_cast<int>(bsLen), 1);
    MSCATTER(gSorted, tok, posB);
}

// P3b-c: 单 tile writePos 进位 (计数链 + volatile RMW)
__attribute__((noinline)) static inline void gt_dyn_p3b_carry1(
    const uint32_t *src32, uint32_t expertPerRank, uint32_t *writePos,
    uint32_t *cntGmOut)
{
    using namespace gt_tile;
    for (uint32_t e = 0; e < expertPerRank; ++e) {
        G1x32 gS(const_cast<uint32_t *>(src32));
        T1x32 sRow2;
        TLOAD(sRow2, gS);
        T1x32 pred;
        TCMPS<CmpMode::EQ>(pred, sRow2, e);
        T1x32 one;
        TEXPANDS(one, 1u);
        T1x32 sel;
        TEXPANDS(sel, 0u);
        TSEL(sel, pred, one);
        TSum1 c;
        TROWSUM(c, sel);
        G1x1 gC(cntGmOut);
        TSTORE(gC, c);
        volatile uint32_t *w = writePos;
        w[e] = w[e] + *cntGmOut;
    }
}

// ============================================================================
// Phase 2 (Tile + multi-thread, runtime shape): 全 tile 散射到每 PE 私有段
//
// 满块 (bs/16 个 16-row 块): 与静态 mt 版同款 tile 链, PE tid 处理每块行
// [4*tid, +4) —— TREMS+TROWMIN (min) / TDIVS+TCMPS+TSEL+TROWMAX (pod flag)
// / MGATHER_ADD (rank=原子写指针) / TMULS+TADD+TSHLS (偏移) / TCI (token
// ramp) / MSCATTER (groupedIds + podInfo)。[N×1] 归约输出经 per-PE GM
// scratch 往返转 [1×4] 行向量 ([C4])。
//
// 尾块 (bs%16 行): 动态 valid 散射链泄漏 lane ([C6]) → 标量兜底。tail 行 r
// 轮转归属 PE (r%4) —— 保证每 PE token 总数 ≤ ceil(bs/4) = kBsPerPE
// (driver 段容量契约; 若沿用 [4*tid,+4) 连续切片, PE0 尾块最多 +4 token
// 会超容量)。标量尾块接续 tile 段的 mySectionCnt 计数 (同 PE 程序序,
// TMA/标量混合访问与既有 "TSTORE 后标量读 GM" 约定同款)。
//
// 契约违例 (topK!=16 / epr>kEprCap / spn>kSpnCap) → 全量标量 stride 兜底
// (原实现, 容量 ceil(bs/4) 天然满足)。
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
    using namespace gt_tile;
    const uint32_t tid = get_thread_idx();
    const uint32_t kBsPerPE = (batchSize + kThreadsPerBlock - 1U) / kThreadsPerBlock;

    uint32_t *mySectionCnt = perPeSectionCnt + tid * expertPerRank;
    {
        volatile uint32_t *v = mySectionCnt;
        for (uint32_t i = 0; i < expertPerRank; i++) v[i] = 0u;
    }

    const bool tileOk = (topk == kTileN) && (expertPerRank <= kEprCap) &&
                        (superPodNum <= kSpnCap) && (expertPerPod != 0u);
    if (tileOk) {
        // GM 往返 scratch (per-PE 私有切片, [C4]; bss 静态, TMA 写可靠)。
        // tile 访问的 scratch 每 PE 补齐到 64B 行粒度 (原 [4][4] 布局使
        // 4 PE 的 16B tile 读写挤同一 cacheline —— 与 p2 合并修复同源的
        // 跨 PE 同行并发竞态规避; cntGmDyn 仅标量访问, 保持紧凑)。
        static uint32_t minScratch[kThreadsPerBlock][16];
        static uint32_t podFlagScratch[kThreadsPerBlock][16];
        static uint32_t rankScratch[kThreadsPerBlock][16];
        static uint32_t cntGmDyn[kThreadsPerBlock];
        static uint32_t offEScratch[kThreadsPerBlock][16];

        const uint32_t sectStride = kThreadsPerBlock * kBsPerPE;
        const uint32_t peBase = tid * kBsPerPE;

        const uint32_t fullBlocks = batchSize / kTileM;
        const uint32_t idsLen = expertPerRank * kThreadsPerBlock * kBsPerPE;
        const uint32_t podLen = idsLen * superPodNum;
        for (uint32_t blk = 0; blk < fullBlocks; ++blk) {
            const uint32_t r0 = blk * kTileM + tid * 4U;
            // 全链经 noinline 分体顺序执行 (调用点零活跃 tile, GM 交接;
            // 消除 raw tile spill 与跨形状回边 TMOV 两类模型断言)
            gt_dyn_p2_min_pods(topkIndex, r0, topk, batchSize, expertPerRank,
                               expertPerPod, superPodNum,
                               minScratch[tid], podFlagScratch[tid]);
            gt_dyn_p2_rank_scatter(minScratch[tid], mySectionCnt, sectStride,
                                   peBase, r0, perPegroupedIds, idsLen,
                                   rankScratch[tid], offEScratch[tid]);
            gt_dyn_p2_podinfo(offEScratch[tid], podFlagScratch[tid],
                              superPodNum, perPePodInfo, podLen);
            gt_dyn_p2_carry(minScratch[tid], expertPerRank, mySectionCnt,
                            cntGmDyn + tid);
        }

        // 尾块标量兜底: tail 行 r 归 PE (r%4), 保证每 PE ≤ kBsPerPE token
        const uint32_t tailBase = fullBlocks * kTileM;
        const uint32_t tailRows = batchSize - tailBase;
        uint32_t *dstPodLocal = podScratch + tid * superPodNum;
        for (uint32_t r = tid; r < tailRows; r += kThreadsPerBlock) {
            const uint32_t i = tailBase + r;
            uint32_t minLocalExpId = expertPerRank;
            for (uint32_t s = 0; s < superPodNum; s++) dstPodLocal[s] = 0;
            uint32_t base = i * topk;
            for (uint32_t col = 0; col < topk; ++col) {
                uint32_t expertId = topkIndex[base + col];
                uint32_t curLocalExpId = expertId % expertPerRank;
                if (curLocalExpId < minLocalExpId) minLocalExpId = curLocalExpId;
                uint32_t curDstPod = expertId / expertPerPod;
                if (curDstPod < superPodNum) dstPodLocal[curDstPod] = 1;
            }
            uint32_t idxInSection = mySectionCnt[minLocalExpId]++;
            uint32_t peOffset = minLocalExpId * sectStride + peBase + idxInSection;
            perPegroupedIds[peOffset] = i;
            uint32_t podPeOffset = peOffset * superPodNum;
            for (uint32_t s = 0; s < superPodNum; s++) {
                perPePodInfo[podPeOffset + s] = dstPodLocal[s];
            }
        }
        return;
    }

    // ---- 契约违例: 原标量 stride 兜底 (逐行与原实现一致) ----
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
// Host-side merge (scalar, single-PE, runtime shape) —— PE0 独占 (分布式
// 并发标量读 tile 刚写入的 perPe* 缓冲会在时序模型触发硬停摆)。
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
    uint32_t expertPerRank,
    uint32_t sortBarrierPhase)
{
    const uint32_t tid = get_thread_idx();

    // Phase 3a: FloorFunc via tile ops on this PE's disjoint rows
    // (满块 = 静态 valid T4x16 [C6]; 原 dyn-valid TilePerPE 删除 —— 动态
    //  ValidRow 描述符污染复用寄存器 → 回边 TMOV 描述符不匹配断言)
    if (topk == kTileN) {
        using namespace gt_tile;
        // TROWMIN 目的 tile 必须物理单列 (PTO 契约: ValidCol==1 && Cols==1)
        using TileMinPE = Tile<Location::Vec, uint32_t, 4, 1, BLayout::RowMajor>;
        using GmPerPE = global_tensor<uint32_t, RowMajor<-1, -1>>;
        using GmMin = global_tensor<uint32_t, RowMajor<-1, -1>>;

        const uint32_t fullBlocks = batchSize / kTileM;
        for (uint32_t blk = 0; blk < fullBlocks; ++blk) {
            const uint32_t r0 = blk * kTileM + tid * 4U;
            T4x16 tIn;
            T4x16 tRem;
            TileMinPE tMin;
            GmPerPE gIn(topkIndex + r0 * topk, static_cast<int>(batchSize),
                        static_cast<int>(topk));
            GmMin gMin(minLocalExpIds + r0, static_cast<int>(batchSize), 1);
            TLOAD(tIn, gIn);
            TREMS(tRem, tIn, expertPerRank);   // local expert ids
            TROWMIN(tMin, tRem);               // per-token min
            TSTORE(gMin, tMin);                // -> minLocalExpIds[r0..]
        }
        // 尾块 (≤15 token): 标量兜底 (dyn-valid tile 链已删除, 同上契约)
        const uint32_t tailRows = batchSize - fullBlocks * kTileM;
        if (tailRows > 0) {
            const uint32_t r0 = fullBlocks * kTileM + tid * 4U;
            const uint32_t vr = (r0 < batchSize)
                              ? ((batchSize - r0 < 4U) ? (batchSize - r0) : 4U) : 0U;
            for (uint32_t row = 0; row < vr; ++row) {
                const uint32_t token = r0 + row;
                uint32_t minLocal = expertPerRank;
                const uint32_t base = token * topk;
                for (uint32_t col = 0; col < topk; ++col) {
                    uint32_t local = topkIndex[base + col] % expertPerRank;
                    if (local < minLocal) minLocal = local;
                }
                minLocalExpIds[token] = minLocal;
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

    // Phase 3b 读全部 PE 的 minLocalExpIds (3a 各 PE 写自己的行切片) ——
    // 3a→3b 之间必须汇合, 否则 PE0 的 counting sort 读到未完成的
    // minLocalExpIds → counts/writePos 垃圾 → MSCATTER 越界 (跨 PE 竞争,
    // 与 reduce 同款; 原实现仅靠 lockstep 侥幸)。
    mtBarrier(sortBarrierPhase);

    // Phase 3b: Counting sort — only PE 0 (needs global coordination)
    // tile 化 (gfsim 兼容面板 [C12]; 替代 MSCATTER_ADD/MGATHER_ADD):
    //   counts   = per-bin 计数链 ([C10], 1×32 tile; 谓词循环内重物化)
    //   starts   = counts 前缀和 (≤kEprCap+1 元素 < 128B tile 下限, 标量)
    //   散射     = [32×32] 成对 rank ([C11]) + 平 MGATHER(writePos) + TCI +
    //              平 MSCATTER + per-tile 计数链进位 (稳定序 = 标量一致)
    //   尾部 (bs%32 元素) = 标量兜底 ([C6] 动态 valid 散射链泄漏)
    // counts = sortScratch[0..epr), writePos = sortScratch[epr..2epr) (driver
    // 既有 scratch 契约, 无需扩容)。
    if (tid == 0) {
        using namespace gt_tile;
        uint32_t *counts = sortScratch;
        uint32_t *writePos = sortScratch + expertPerRank;
        const bool tileOk = (expertPerRank <= kEprCap);
        if (tileOk) {
            static uint32_t cntGm0[1];
            static uint32_t rankBuf[32];
            const uint32_t nTiles = batchSize / 32u;

            // ---- counts: per-bin 计数链 (s2 循环内重物化) ----
            uint32_t acc[kEprCap];
            for (uint32_t e = 0; e < kEprCap; ++e) acc[e] = 0u;
            static uint32_t cntArr[kEprCap];
            for (uint32_t tb = 0; tb < nTiles; ++tb) {
                gt_dyn_p3b_counts1(minLocalExpIds + tb * 32u, expertPerRank,
                                   cntArr);
                for (uint32_t e = 0; e < expertPerRank; ++e) acc[e] += cntArr[e];
            }
            {
                volatile uint32_t *vc = counts;
                for (uint32_t e = 0; e < expertPerRank; ++e) vc[e] = acc[e];
            }
            // 尾部元素标量计数 (bs%32 < 32 个, counts 上接续 RMW)
            for (uint32_t i = nTiles * 32u; i < batchSize; i++) {
                uint32_t sec = minLocalExpIds[i];
                if (sec < expertPerRank) counts[sec]++;
            }

            sectionStarts[0] = 0;
            for (uint32_t i = 0; i < expertPerRank; i++) {
                sectionStarts[i + 1] = sectionStarts[i] + counts[i];
            }
            {
                volatile uint32_t *w = writePos;
                for (uint32_t i = 0; i < expertPerRank; i++) w[i] = sectionStarts[i];
            }

            // ---- 散射: 每 tile 经 noinline 分体 (成对 rank + MGATHER +
            //      TCI + MSCATTER + writePos 计数链进位; 调用点零活跃 tile) ----
            for (uint32_t tb = 0; tb < nTiles; ++tb) {
                gt_dyn_p3b_scatter1(minLocalExpIds + tb * 32u, writePos,
                                    sortedTokenIds, batchSize, tb * 32u,
                                    rankBuf);
                gt_dyn_p3b_carry1(minLocalExpIds + tb * 32u, expertPerRank,
                                  writePos, cntGm0);
            }
            // 尾部元素标量散射 (稳定序接续: tile 段已按 token 序消费 writePos)
            for (uint32_t i = nTiles * 32u; i < batchSize; i++) {
                uint32_t section = minLocalExpIds[i];
                sortedTokenIds[writePos[section]++] = i;
            }
        } else {
            // epr 超容量: 原标量实现
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
}

// ============================================================================
// Entry point (multi-PE SPMD; called by every PE, runtime tiling)
//
//   tiling = {bs, topK, expertPerRank, expertPerPod, superPodNum}
//
// Cross-PE data hand-offs are separated by mtBarrier (与静态版一致; 相位
// = inv*8 + k, 单调编号见 sInvCnt 注):
//   Phase1 reduce  reads all PEs' cntLocal       -> barrier(ph+1, 函数内部)
//   merge          reads all PEs' scatter state  -> barrier(ph+2); PE0 独占
//   Phase3b sort   reads all PEs' minLocalExpIds -> barrier(ph+3, 函数内部)
//   末端汇合 (全部输出写完才可离开 kernel)       -> barrier(ph+4)
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

    // 单调相位编号: per-PE 私用调用计数 sInvCnt, 相位 = inv*8 + k 跨调用
    // 单调递增, 无跨 PE 复位写 (清零晚于置位会抹掉 flag → 互等自旋)。
    // kernel 相位 = inv*8+1..5, driver 验证汇合 = 6, 第二轮 = 9..13。
    static uint32_t sInvCnt[kThreadsPerBlock];   // bss 零初始化 = 首轮 inv 0
    const uint32_t inv = sInvCnt[tid];
    sInvCnt[tid] = inv + 1u;
    const uint32_t ph = inv * 8u;

    // 相位分配 (步长 8): ph+1 直方图完成; ph+2 散射段完成;
    // ph+3 merge 完成; ph+4 floorFunc 完成; ph+5 全部输出写完
    //   ph+5 = 全部输出写完 (任何 PE 离开 kernel 前)
    calTokenPerExpertCnt_mt_tile_dyn(topkIndex, tokenPerExpertCnt, cntLocal,
                                     expertNum, bs, topK, ph + 1u);

    groupToken_mt_tile_dyn(topkIndex, perPegroupedIds, perPeSectionCnt, perPePodInfo,
                           podScratch, bs, topK, expertPerRank, expertPerPod, superPodNum);
    mtBarrier(ph + 2u);   // all PEs' scatter sections complete before merge reads

    if (tid == 0) {
        mergeGroupTokenResults_dyn(perPegroupedIds, perPeSectionCnt, perPePodInfo,
                                   groupedTokenIds, tokenSuperPodInfo, expertSectionTokenCnt,
                                   expertPerRank, superPodNum, bs);
    }
    // 汇合 (ph+3): 全 PE 等 PE0 merge 完成后才进 sortKernel (避免 sort
    // tile 流量与 merge 标量流量并发)
    mtBarrier(ph + 3u);

    sortKernel_mt_tile_dyn(topkIndex, minLocalExpIds, sortedTokenIds, sectionStarts,
                           sortScratch, bs, topK, expertPerRank, ph + 4u);
    mtBarrier(ph + 5u);   // all outputs fully written before any PE leaves
}

#endif // GROUP_TOKEN_VEC_MT_DYN_HPP
