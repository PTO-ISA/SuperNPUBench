#ifndef GROUP_TOKEN_VEC_MT_HPP
#define GROUP_TOKEN_VEC_MT_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>

#include "solution/group_token_vec/gt_tile_common.hpp"

// ============================================================================
// MoE Token Grouping — Multi-thread Vector (Tile) variant
//
// 4-PE SPMD. Each PE uses get_thread_idx() for tile/stride parallelism.
//
// 全 tile 指令链版 (契约依据 gt_tile_common.hpp [C1..C9], tile_probe 实证):
//   1. 每个 TLOAD 结果由 tile op 消费, 结果经 TSTORE/MSCATTER 出 tile 域;
//      标量读回只打 GM (extract_vector_elt 后端崩溃规避)。
//   2. Per-PE tiles are disjoint: PE tid owns rows [4*tid, 4*tid+3] of every
//      16-row block. No duplicated TLOAD traffic.
//   3. 直方图 = TCMPS<GE> 守卫 (非法 lane value 置 0 / index 钳 0) +
//      MSCATTER_ADD (数字编码 TLSU 21, 元素下标 index [PTO v0.58.6], tile 内重复下标
//      row-major 顺序 RMW) —— 旧注释 "TCMP/TCMPS u32 被汇编器拒绝" 已过时
//      ([C1], qli #177 与探针双重实证)。
//   4. Cross-PE hand-offs are guarded by mtBarrier.
//      [2026-10-01 修改] mtBarrier 由裸热自旋改为 mt_dyn 同款驱逐屏障
//      (写方定向驱逐 + 读方寄存器延迟链 + 1/64 稀疏驱逐): gfsim 跨 PE
//      无 L1D snoop, 裸自旋仅靠 busy 阶段容量驱逐侥幸放行, 末端低流量
//      屏障存在确定性活锁风险 (dyn 开发期实证, 见 barrier 实现处注);
//      Phase2 per-PE scratch 补齐 64B 行粒度 (跨 PE 同行 16B tile 写
//      丢更新竞态, dyn 同源已证修复)。gfsim fourpe 下本算子的确定性
//      失败根因 (退出 lockstep 搁浅) 在驱动层, 见
//      group_token_vec_mt_gfsim_fix_report.md。
//   5. [N×1] 归约输出只可 TSTORE; 行向量经 GM 往返转 [1×N] ([C4]);
//      原子族/标量 TEPL 一律 [1×N] 静态 valid 形状 ([C4]/[C6])。
//
// Phase 1: TLOAD [4×16] + MSCATTER_ADD 直方图 (每 PE 私有桶);
//          reduce = 4× TLOAD [1×32] + TADD 链 + TSTORE (每 PE 32 专家)
// Phase 2: TREMS+TROWMIN (min) + TDIVS+TCMPS+TSEL+TROWMAX (pod flag) +
//          MGATHER_ADD (rank=原子写指针) + TMULS/TADD/TSHLS (偏移) +
//          TCI (token ramp) + MSCATTER (groupedIds/podInfo 散射);
//          per-PE 4-token 连续块分解 (与 Phase1/3a 同款; 输出按分区集合
//          校验, 分解方式自由)
// Phase 3a: TLOAD + TREMS + TROWMIN + TSTORE minLocalExpIds (原有, 保留)
// Phase 3b: PE0 tile counting sort — MSCATTER_ADD counts + MGATHER_ADD
//           writePos + TCI + MSCATTER (稳定序 = 标量逐元素一致)
// merge:    PE0 标量 (数据依赖变长段拷贝 + 顺序 globalIdx 累加, 全局协调;
//           tile 尾块因 [C6] 动态 valid 泄漏不可用 — 保留标量)
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
// Multi-PE barrier —— 单行 flag + 寄存器延迟链 + 稀疏定向集驱逐
// (时序模型跨 PE 无 snoop: flag 行须主动驱逐使 L1D miss 才能看到新值;
//  驱逐流量须稀疏, 否则 prior 洪泛饿死 L2 tile 派发)。gfrun 无缓存, 语义同
// 原始屏障。
// [2026-10-01 修改] 原实现为裸热自旋 (plain hot spin): busy 阶段靠海量
//  内存流量的容量驱逐侥幸观察到 flag 更新; 屏障处于低流量窗口 (如末端
//  汇合、其它 PE 已安静) 时自旋 PE 持续命中私有 L1D 陈旧 flag 行 →
//  活锁风险 (dyn 开发期实证, 其注释: "纯 plain 热自旋在长偏斜下读陈旧
//  flag 活锁")。自 group_token_vec_mt_dyn.hpp 移植同款驱逐屏障 (该变体
//  gfsim fourpe 实测 PASS, Total Cycles 1,282,647)。
// ============================================================================
// [2026-10-01 修改] flag 行显式 64B 对齐; 新增 96KB 驱逐区 (16KB 对齐)
alignas(16384) static volatile uint32_t sEvictSpan[6 * 4096];   // 96KB 驱逐区
alignas(64) static volatile uint32_t sPhaseDone[kThreadsPerBlock];  // 4 flag 同一行

static inline void mtCompilerBarrier()
{
    __asm__ volatile("" : : : "memory");
}

// [2026-10-01 修改] 整体替换为 mt_dyn 同款驱逐屏障 (原裸自旋见上方修复注)
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
// Phase 1 (Tile + multi-thread): 每 PE 不相交 [4×16] TLOAD + MSCATTER_ADD
// 直方图 + tile 归约
//
// PE tid owns rows [4*tid, 4*tid+3] of each 16-row block (rule 2)。
// 直方图 (每块一条 MSCATTER_ADD):
//   守卫 (等价标量 if (expertId < expertNum)): TCMPS<GE> 生成 inv predicate,
//   index = inv ? 0 : expertId (钳到安全桶 0, 地址恒在界内), value = inv ? 0
//   : 1 (非法 lane 加 0, 值中性) —— cntLocal 每 PE 切片恰为 expertNum 个
//   u32, 无 scratch 桶空间, 故守卫走 value 置零而非 index 改指 ([C7] TSEL
//   假分支 = dst 旧值)。
// 归约 (每 PE 负责 32 个专家 = 恰一个 1×32 tile):
//   TLOAD ×4 (各 PE 切片的同一段) + TADD ×3 + TSTORE → tokenPerExpertCnt。
// ============================================================================
static inline void calTokenPerExpertCnt_mt_tile(
    uint32_t *topkIndex,
    uint32_t *tokenPerExpertCnt,
    uint32_t *cntLocal,
    uint32_t expertNum,
    uint32_t topkEleNum)
{
    using namespace gt_tile;
    const uint32_t tid = get_thread_idx();

    // myCnt 清零: expertNum(128) u32 = 16×8 tile 填充 ([C9] 规避标量合并)
    uint32_t *myCnt = cntLocal + tid * expertNum;
    {
        T16x8 z;
        TEXPANDS(z, 0u);
        G16x8 gz(myCnt);
        TSTORE(gz, z);
    }

    using GmPerPE = global_tensor<uint32_t, RowMajor<kBS, kTopK>>;
    using TilePerPE = Tile<Location::Vec, uint32_t, 4, kTileN, BLayout::RowMajor>;
    using itPerPE = global_iterator<GmPerPE, TilePerPE>;
    itPerPE gIter(topkIndex);   // full tensor; PE tid prefetches its own slice

    // 计数链直方图 ([C10], gfsim 兼容; 替代 MSCATTER_ADD —— TimingSim 无
    // TLSU 原子族 [C12]): 每块 TLOAD 一次, 每 bin e: TCMPS<EQ>+TSEL+
    // TCOLSUM([4×16]→[1×16])+TROWSUM(→[1×1])+TSTORE→标量寄存器累加。
    // bin 循环天然限定合法值域 (无需守卫)。谓词循环内 tile 全部重物化
    // (无 loop-carried tile —— TMOV/U8 谓词重载束规避)。
    static uint32_t sumGm[kThreadsPerBlock];
    uint32_t *mySum = sumGm + tid;      // per-PE 单值出口 (写不相交)
    uint32_t acc[kExpertNum];
    for (uint32_t e = 0; e < kExpertNum; ++e) acc[e] = 0u;
    for (uint32_t blk = 0; blk < kBS / kTileM; ++blk) {
        // rows [16*blk + 4*tid, +4): row-tile index 4*blk + tid on full tensor
        auto src = gIter(4 * blk + tid, 0);
        TilePerPE t;
        TLOAD(t, src);
        for (uint32_t e = 0; e < expertNum; ++e) {
            TilePerPE pred;
            TCMPS<CmpMode::EQ>(pred, t, e);
            TilePerPE one;
            TEXPANDS(one, 1u);
            TilePerPE sel;
            TEXPANDS(sel, 0u);
            TSEL(sel, pred, one);
            TCol16 cs;
            TCOLSUM(cs, sel);
            TSum1 s;
            TROWSUM(s, cs);
            G1x1 gS(mySum);
            TSTORE(gS, s);
            acc[e] += *mySum;
        }
    }
    // acc → myCnt (128 u32: volatile 标量逐元素 [C9], 一次性)
    {
        volatile uint32_t *vm = myCnt;
        for (uint32_t e = 0; e < expertNum; ++e) vm[e] = acc[e];
    }

    // 全部 PE 的直方图 (myCnt 导出) 完成后, reduce 才可读其它 PE 的
    // cntLocal 切片 —— 原实现把 reduce 放在调用方 barrier 之前, 是真实的
    // 跨 PE 顺序竞争 (ROOTCAUSE_gfsim §6.3 同款, "应修"); gfrun 靠确定性
    // lockstep 侥幸, gfsim 真实 PE 偏斜下读到未完成计数。
    mtBarrier(1);   // all PEs' histograms complete before reduce consumers

    // Reduce: PE tid 负责专家段 [tid*32, tid*32+32) = 一个 1×32 tile;
    // 4 个 PE 切片同段 TLOAD + TADD 链 + TSTORE
    uint32_t expertsPerPE = expertNum / kThreadsPerBlock;
    {
        G1x32 g0(cntLocal + 0 * expertNum + tid * expertsPerPE);
        G1x32 g1(cntLocal + 1 * expertNum + tid * expertsPerPE);
        G1x32 g2(cntLocal + 2 * expertNum + tid * expertsPerPE);
        G1x32 g3(cntLocal + 3 * expertNum + tid * expertsPerPE);
        T1x32 a, b, c, d;
        TLOAD(a, g0);
        TLOAD(b, g1);
        TLOAD(c, g2);
        TLOAD(d, g3);
        TADD(a, a, b);
        TADD(c, c, d);
        TADD(a, a, c);
        G1x32 gOut(tokenPerExpertCnt + tid * expertsPerPE);
        TSTORE(gOut, a);
    }
}

// ============================================================================
// Phase 2 (Tile + multi-thread): 全 tile 散射到每 PE 私有段
//
// 分解: PE tid 处理每个 16-row 块的行 [4*tid, +4) —— 与 Phase1/3a 同款
// 不相交连续 4-token 块 (原 stride 分解 i=tid+4m 的 token 集合不同, 但每
// token 恰归属一个 PE、输出按分区集合校验, 语义等价)。
//
// 每块 tile 链 (形状契约 [C4]/[C5]/[C8]):
//   TLOAD [4×16] → TREMS → TROWMIN [32×1,v4] → TSTORE minScratch[tid] →
//   TLOAD minRow [1×4] (GM 往返列转行)
//   TDIVS(pod, t, expertPerPod) → 每 p: TCMPS<EQ>+TSEL+TROWMAX [32×1,v4] →
//   TSTORE podScratch[tid][p] (后续 TLOAD 转 [1×4])
//   TSHLS(minRow<<2) → MGATHER_ADD(rank, mySectionCnt, ...) — rank = 原子
//   写指针 old 值 ([C3], 替代标量 mySectionCnt[min]++), 计数器终值 =
//   perPeSectionCnt 输出
//   offE = minRow*(4*kBsPerPE) + tid*kBsPerPE + rank (TMULS/TADDS/TADD) →
//   offE (元素下标) → MSCATTER(perPegroupedIds, TCI(16*blk+4*tid), offE)
//   podInfo: poE = offE*superPodNum + p → MSCATTER(perPePodInfo, flagRow_p)
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
    using namespace gt_tile;
    const uint32_t tid = get_thread_idx();
    constexpr uint32_t kBsPerPE = kBS / kThreadsPerBlock;

    // 计数器清零: expertPerRank(≤4) u32 < 128B tile 下限 → volatile 标量 ([C9])
    uint32_t *mySectionCnt = perPeSectionCnt + tid * expertPerRank;
    {
        volatile uint32_t *v = mySectionCnt;
        for (uint32_t i = 0; i < expertPerRank; i++) v[i] = 0u;
    }

    // GM 往返 scratch (per-PE 私有切片, [C4] 列→行转换; bss 静态, TMA 写可靠)。
    // [2026-10-01 修改] 原 [kThreadsPerBlock][4] 布局使 4 PE 的 16B tile 读写
    // 挤同一 cacheline —— gfsim 跨 PE 无 snoop, 同行并发的 16B tile 写会
    // 互相丢更新 (mt_dyn 同源竞态的已证修复); tile 访问的 scratch 每 PE
    // 补齐到 64B 行粒度 (pod flag p 位于 +p*4 字), cntGm 仅标量访问保持紧凑。
    static uint32_t minScratch[kThreadsPerBlock][16];
    static uint32_t podFlagScratch[kThreadsPerBlock][16];
    static uint32_t rankScratch[kThreadsPerBlock][16];
    static uint32_t cntGm[kThreadsPerBlock];

    using TilePerPE = Tile<Location::Vec, uint32_t, 4, kTileN, BLayout::RowMajor>;
    using GmPerPE = global_tensor<uint32_t, RowMajor<kBS, kTopK>>;
    using itPerPE = global_iterator<GmPerPE, TilePerPE>;
    itPerPE gIter(topkIndex);

    GFlat gIds(perPegroupedIds,
               static_cast<int>(kExpertPerRank * kThreadsPerBlock * kBsPerPE), 1);
    GFlat gPodInfo(perPePodInfo,
                   static_cast<int>(kExpertPerRank * kThreadsPerBlock * kBsPerPE
                                    * kSuperPodNum), 1);
    global_tensor<uint32_t, RowMajor<1, kExpertPerRank>> gCnt(mySectionCnt);
    // per-PE 段内布局: peOffset = min*sectStride + peBase + rank
    const uint32_t sectStride = kThreadsPerBlock * kBsPerPE;
    const uint32_t peBase = tid * kBsPerPE;

    for (uint32_t blk = 0; blk < kBS / kTileM; ++blk) {
        // 1. 本 PE 的 4 token × 16 expert id
        auto src = gIter(4 * blk + tid, 0);
        TilePerPE t;
        TLOAD(t, src);

        // 2. minLocalExpId: TREMS → TROWMIN [4×1] → GM 往返 → [1×4]
        TilePerPE rem;
        TREMS(rem, t, expertPerRank);
        TRed4 minCol;
        TROWMIN(minCol, rem);
        G4x1 gMinW(minScratch[tid]);
        TSTORE(gMinW, minCol);
        TCol4 minRow;
        G1x4 gMinR(minScratch[tid]);
        TLOAD(minRow, gMinR);

        // 3. pod any-flag (pod 每 p 迭代重物化 —— 谓词循环无 loop-carried
        //    tile 契约, 见单 PE 版注)
        for (uint32_t p = 0; p < superPodNum; ++p) {
            TilePerPE pod;
            TDIVS(pod, t, expertPerPod);
            TilePerPE pred;
            TCMPS<CmpMode::EQ>(pred, pod, p);
            TilePerPE onev;
            TEXPANDS(onev, 1u);
            TilePerPE sel;
            TEXPANDS(sel, 0u);
            TSEL(sel, pred, onev);
            TRed4 flagCol;
            TROWMAX(flagCol, sel);
            G4x1 gFlagW(podFlagScratch[tid] + p * 4u);   // [2026-10-01 修改] 64B 行内 +p*4 字
            TSTORE(gFlagW, flagCol);
        }

        // 4. 成对比较 rank ([C11], [4×4]): rankIncl[i] = 1+#{j<i:min_j==min_i}
        T4x4 Mc;
        TROWEXPAND(Mc, minCol);          // Mc[i][j] = min[i]
        T4x4 Mr;
        TCOLEXPAND(Mr, minRow);          // Mr[i][j] = min[j] (源物理 Cols=4)
        T4x4 eq;
        TCMP<CmpMode::EQ>(eq, Mc, Mr);
        T4x4 tri;
        TTRI(tri);
        T4x4 mat;
        TEXPANDS(mat, 0u);
        TSEL(mat, eq, tri);
        TRed4 rankIncl;
        TROWSUM(rankIncl, mat);
        G4x1 gRankW(rankScratch[tid]);
        TSTORE(gRankW, rankIncl);
        TCol4 rankRow;
        G1x4 gRankR(rankScratch[tid]);
        TLOAD(rankRow, gRankR);
        TSUBS(rankRow, rankRow, 1u);

        // 5. perPegroupedIds[min*sectStride + peBase + base + rank] = token
        TCol4 base;
        MGATHER(base, gCnt, minRow);     // 平 MGATHER 查 per-PE 写指针
        TCol4 minB;
        TMULS(minB, minRow, sectStride);
        TADDS(minB, minB, peBase);
        TCol4 offE;
        TADD(offE, minB, base);
        TADD(offE, offE, rankRow);
        TCol4 tok;
        TCI(tok, blk * kTileM + tid * 4u);
        MSCATTER(gIds, tok, offE);

        // 6. perPePodInfo[(...)*spn + p] = podFlag_p
        for (uint32_t p = 0; p < superPodNum; ++p) {
            TCol4 flagRow;
            G1x4 gFlagR(podFlagScratch[tid] + p * 4u);   // [2026-10-01 修改] 64B 行内 +p*4 字
            TLOAD(flagRow, gFlagR);
            TCol4 po;
            TMULS(po, offE, superPodNum);
            TADDS(po, po, p);
            MSCATTER(gPodInfo, flagRow, po);
        }

        // 7. 块末进位: mySectionCnt[s] += #{本块 min==s} (计数链 [C10],
        //    minRow 循环内重物化)
        for (uint32_t s = 0; s < expertPerRank; ++s) {
            TCol4 minRow2;
            G1x4 gMinR2(minScratch[tid]);
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
            G1x1 gC(cntGm + tid);
            TSTORE(gC, cnt);
            volatile uint32_t *vc = mySectionCnt;
            vc[s] = vc[s] + cntGm[tid];
        }
    }
}

// ============================================================================
// Host-side merge (scalar, single-PE)
//
// 保留标量的原因 (不可 tile): 每 (section, PE) 段的拷贝长度 peCnt 是数据
// 依赖的运行时值, tile 拷贝的 valid 区必须编译期静态 ([C6] 动态 ValidRow
// 链会按物理行泄漏 lane, 越界写坏相邻段); globalIdx 为跨段顺序累加。
// 段拷贝总量 ≤ 2048+4096 u32, PE0 独占执行, 非热点。
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
// Phase 3 (Tile + multi-thread): TROWMIN FloorFunc + PE0 tile counting sort
//
// Phase 3a is a true tile pipeline on disjoint per-PE tiles:
//   TLOAD(4x16 rows of this PE) -> TREMS(%, kExpertPerRank)
//   -> TROWMIN (per-row min) -> TSTORE(minLocalExpIds[token])
// Phase 3b (PE0, global coordination) 同为 tile 链: MSCATTER_ADD counts +
// MGATHER_ADD writePos (稳定序) + TCI + MSCATTER, 详见函数内注。
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
    // TROWMIN 目的 tile 必须是物理单列（PTO 契约: ValidCol==1 && Cols==1），
    // 此前声明 Tile<..., 4, 8, RowMajor, 4, 1>（物理 4x8/有效 4x1）违反契约，
    // 自 PR #74 合入起 vec_mt 一直无法编译。输出 GM 张量为 RowMajor<kBS, 1>，
    // 物理形状直接取 4x1。
    using TileMinPE = Tile<Location::Vec, uint32_t, 4, 1, BLayout::RowMajor>;
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

    // Phase 3b 读全部 PE 的 minLocalExpIds (3a 各 PE 写自己的行切片) ——
    // 3a→3b 之间必须汇合, 否则 PE0 的 counting sort 读到未完成的
    // minLocalExpIds → counts/writePos 垃圾 → MSCATTER 越界 (跨 PE 竞争,
    // 与 reduce 同款; 原实现仅靠 lockstep 侥幸)。
    mtBarrier(3);   // FloorFunc writes visible before PE0's sort reads

    // Phase 3b: Counting sort — only PE 0 (needs global coordination)
    // tile 化 (gfsim 兼容面板 [C12]; 替代 MSCATTER_ADD/MGATHER_ADD):
    //   counts = per-bin 计数链 ([C10], 1×32 tile)
    //   sectionStarts 前缀和 (≤5 元素 < 128B tile 下限, 标量, GM 读回)
    //   散射 = [32×32] 成对 rank ([C11]) + 平 MGATHER(writePos) + TCI +
    //          平 MSCATTER + per-tile 计数链进位
    if (tid == 0) {
        using namespace gt_tile;
        static uint32_t writePos[kExpertPerRank];
        static uint32_t cntGm0[1];
        static uint32_t rankBuf[32];

        const uint32_t nTiles = batchSize / 32u;

        // ---- counts: per-bin 计数链 ([C10], 替代 MSCATTER_ADD);
        //      谓词循环内 s2 重物化 (无 loop-carried tile) ----
        uint32_t acc[kExpertPerRank];
        for (uint32_t e = 0; e < kExpertPerRank; ++e) acc[e] = 0u;
        for (uint32_t tb = 0; tb < nTiles; ++tb) {
            for (uint32_t e = 0; e < expertPerRank; ++e) {
                G1x32 gS2(minLocalExpIds + tb * 32u);
                T1x32 s2;
                TLOAD(s2, gS2);
                T1x32 pred;
                TCMPS<CmpMode::EQ>(pred, s2, e);
                T1x32 one;
                TEXPANDS(one, 1u);
                T1x32 sel;
                TEXPANDS(sel, 0u);
                TSEL(sel, pred, one);
                TSum1 c;
                TROWSUM(c, sel);
                G1x1 gC(cntGm0);
                TSTORE(gC, c);
                acc[e] += cntGm0[0];
            }
        }

        sectionStarts[0] = 0;
        for (uint32_t i = 0; i < expertPerRank; i++) {
            sectionStarts[i + 1] = sectionStarts[i] + acc[i];
        }
        {
            volatile uint32_t *w = writePos;
            for (uint32_t i = 0; i < expertPerRank; i++) w[i] = sectionStarts[i];
        }

        // ---- 散射: [32×32] 成对 rank ([C11], 替代 MGATHER_ADD) +
        //      平 MGATHER(writePos) + TCI + 平 MSCATTER; 稳定序 ----
        GFlat gSorted(sortedTokenIds, static_cast<int>(batchSize), 1);
        global_tensor<uint32_t, RowMajor<1, kExpertPerRank>> gWP(writePos);
        for (uint32_t tb = 0; tb < nTiles; ++tb) {
            // 同一 GM 数据双视图: [1×32] 行 + [32×1] 列
            G1x32 gS(minLocalExpIds + tb * 32u);
            T1x32 sRow;
            TLOAD(sRow, gS);
            G32x1 gSC(minLocalExpIds + tb * 32u);
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

            T1x32 base;
            MGATHER(base, gWP, sRow);
            T1x32 pos;
            TADD(pos, base, rankRow);
            T1x32 tok;
            TCI(tok, tb * 32u);
            MSCATTER(gSorted, tok, pos);

            // writePos 进位 (计数链, sRow2 重物化)
            for (uint32_t e = 0; e < expertPerRank; ++e) {
                G1x32 gS3(minLocalExpIds + tb * 32u);
                T1x32 sRow2;
                TLOAD(sRow2, gS3);
                T1x32 pred;
                TCMPS<CmpMode::EQ>(pred, sRow2, e);
                T1x32 one;
                TEXPANDS(one, 1u);
                T1x32 sel;
                TEXPANDS(sel, 0u);
                TSEL(sel, pred, one);
                TSum1 c;
                TROWSUM(c, sel);
                G1x1 gC(cntGm0);
                TSTORE(gC, c);
                volatile uint32_t *w = writePos;
                w[e] = w[e] + cntGm0[0];
            }
        }
    }
}

// ============================================================================
// Entry point (multi-PE SPMD; called by every PE)
//
// Cross-PE data hand-offs are separated by mtBarrier:
//   Phase1 reduce  reads all PEs' cntLocal       -> barrier(1) 函数内部
//                  (直方图导出后、reduce 读取前)
//   merge          reads all PEs' scatter state  -> barrier(2) after Phase2;
//                  executed by PE0 only (single merge, no duplicate work)
//   Phase3b sort   reads all PEs' minLocalExpIds -> barrier(3) 函数内部
//                  (floorFunc 后、PE0 counting sort 前)
//   末端汇合 (全部输出写完才可离开 kernel)       -> barrier(4)
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

    // barrier(1) 在函数内部: 直方图导出后、cross-PE reduce 前
    calTokenPerExpertCnt_mt_tile(topkIndex, tokenPerExpertCnt, cntLocal,
                                   kExpertNum, kTopKEleNum);

    groupToken_mt_tile(topkIndex, perPegroupedIds, perPeSectionCnt, perPePodInfo,
                         kBS, kTopK, kExpertPerRank, kExpertPerPod, kSuperPodNum);
    mtBarrier(2);   // all PEs' scatter sections complete before merge reads

    if (tid == 0) {
        mergeGroupTokenResults(perPegroupedIds, perPeSectionCnt, perPePodInfo,
                                groupedTokenIds, tokenSuperPodInfo, expertSectionTokenCnt,
                                kExpertPerRank, kSuperPodNum);
    }

    // barrier(3) 在 sortKernel 内部: floorFunc 后、PE0 counting sort 前
    sortKernel_mt_tile(topkIndex, minLocalExpIds, sortedTokenIds, sectionStarts,
                         kBS, kTopK, kExpertPerRank);
    mtBarrier(4);   // all outputs fully written before any PE leaves the kernel
}


#endif // GROUP_TOKEN_VEC_MT_HPP
