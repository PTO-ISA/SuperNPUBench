#ifndef GROUP_TOKEN_VEC_HPP
#define GROUP_TOKEN_VEC_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>

#include "solution/group_token_vec/gt_tile_common.hpp"

// ============================================================================
// MoE Token Grouping — Vector (Tile) variant
//
// 全 tile 指令链版 (契约与形状依据见 gt_tile_common.hpp [C1..C9], 全部经
// tile_probe 探针在 gfrun 实证):
//
// 全链 gfsim 兼容 (指令面板 [C12]; TimingSim TLSU 无 GM 原子族完成路径,
// MSCATTER_ADD/MGATHER_ADD 仅 gfrun 可用 → 已全部替换为等价 tile 链):
// Phase 1:  计数链直方图 ([C10]): 每 16×16 块 TLOAD 一次, 每 bin e:
//           TCMPS<EQ>+TSEL+TCOLSUM+TROWSUM+TSTORE → 标量寄存器累加;
//           bin 循环天然限定合法值域 (无需越界守卫)。
// Phase 2:  TLOAD + TREMS + TROWMIN (minLocalExpId) + TDIVS + TCMPS<EQ> +
//           TSEL + TROWMAX (pod any-flag) + 成对比较 rank ([C11]:
//           TROWEXPAND/TCOLEXPAND/TCMP/TSEL(∧TTRI)/TROWSUM, 稳定序 =
//           标量 cnt[section]++) + 平 MGATHER (写指针查表) + TCI (token
//           ramp) + 平 MSCATTER (groupedTokenIds/tokenSuperPodInfo 散射) +
//           块末计数链进位。[N×1] 归约结果经 GM 往返转 [1×N] ([C4])。
// Phase 3a: TLOAD + TREMS + TROWMIN + TSTORE (与 _mt 版同款流水)。
// Phase 3b: counts = 计数链 ([C10], 1×32 tile); sectionStarts 前缀和
//           (≤5 元素, 低于 128B tile 下限, 标量); 散射 = [32×32] 成对
//           rank ([C11]) + 平 MGATHER(writePos) + TCI + 平 MSCATTER +
//           per-tile 计数链进位 —— 与标量 counting sort 逐元素一致。
//
// 保留标量的位置 (不可 tile, 原因):
//   - 计数器/前缀和等 ≤5 元素小数组操作: 低于 128B tile 粒度下限 ([C9]
//     初始化用 volatile 逐元素写规避 16B 合并 store 写丢失)。
//   - DoAtomicAdd=false 分支: 性能对照专用 (driver 从不实例化), 语义
//     故意非原子, 保持原标量实现。
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
// Phase 1 (Tile): 全 tile 计数链直方图 ([C10], gfsim 兼容)
//
// 每 16×16 块 TLOAD 一次 (tile 常驻), 对每个 bin e ∈ [0, expertNum):
//   TCMPS<EQ>(pred, t, e) → TEXPANDS(sel,0)+TSEL(sel,pred,one) 物化 0/1
//   → TCOLSUM([16×16]→[1×16]) → TROWSUM(→[32×1,v1×1] 全和) → TSTORE
//   → 标量读回累加 acc[e] (寄存器累加, 末端 volatile 落 GM [C9])。
// bin 循环只覆盖 [0, expertNum) → 非法 id 天然不计数 (等价标量守卫)。
// 原 MSCATTER_ADD 方案仅 gfrun 可用 (TimingSim TLSU 原子族缺失 [C12])。
// 成本 ([C13]): 128bin×32tile=4096 链 ≈ 382K gfsim cycles / 6s gfrun。
// ============================================================================
static inline void calTokenPerExpertCnt_tile(uint32_t *topkIndex,
                                                 uint32_t *tokenPerExpertCnt,
                                                 uint32_t expertNum,
                                                 uint32_t topkEleNum)
{
    using namespace gt_tile;

    static uint32_t sumGm[1];      // 计数链 TROWSUM 单值出口 (TSTORE→标量读回)
    uint32_t acc[kExpertNum];      // 寄存器累加 (栈), 末端一次性落 GM
    for (uint32_t e = 0; e < kExpertNum; ++e) acc[e] = 0u;

    using itTopk = global_iterator<GmTopkIndex, TileU32>;
    itTopk gIter(topkIndex);

    constexpr uint32_t Mb = kBS / kTileM;
    for (uint32_t blk = 0; blk < Mb; ++blk) {
        auto gI = gIter(blk, 0);
        T16x16 t;
        TLOAD(t, gI);
        T16x16 one;
        TEXPANDS(one, 1u);
        for (uint32_t e = 0; e < expertNum; ++e) {
            T16x16 pred;
            TCMPS<CmpMode::EQ>(pred, t, e);
            T16x16 sel;
            TEXPANDS(sel, 0u);
            TSEL(sel, pred, one);
            TCol16 cs;
            TCOLSUM(cs, sel);
            TSum1 s;
            TROWSUM(s, cs);
            G1x1 gS(sumGm);
            TSTORE(gS, s);
            acc[e] += sumGm[0];
        }
    }

    // acc → tokenPerExpertCnt (128 u32 = 512B: volatile 标量逐元素, [C9];
    // 一次性导出, 非热点)
    volatile uint32_t *v = tokenPerExpertCnt;
    for (uint32_t e = 0; e < expertNum; ++e) v[e] = acc[e];
}

// ============================================================================
// Phase 2 (Tile): 全 tile 散射流水 (每 16-token 块)
//
// Tile 指令链:
//   1.  TLOAD(dataTile [16×16])            — 16 token × 16 expert id
//   2.  TREMS(rem, data, expertPerRank)    — localExpId = expertId % epr
//   3.  TROWMIN(minCol, rem)               — 每 token 最小 localExpId [16×1]
//   4.  TSTORE(minCol→minScratch) + TLOAD(minRow [1×16]) — 列→行 GM 往返 ([C4])
//   5.  TDIVS(pod, data, expertPerPod)     — podId = expertId / epp (运行时除数)
//   6.  每 pod p: TCMPS<EQ>+TSEL+TROWMAX   — podFlag_p = any_j(podId==p) [16×1]
//       TSTORE→podScratch[p], 后续 TLOAD 转 [1×16] 行向量
//   7.  成对比较 rank ([C11], gfsim 兼容; 替代 MGATHER_ADD —— TimingSim
//       无 TLSU 原子族 [C12]): TROWEXPAND(Mc) + TCOLEXPAND(Mr) + TCMP<EQ>
//       + TSEL(∧TTRI 下三角) + TROWSUM → rankIncl = 1+#{j<i:min_j==min_i},
//       GM 往返转 [1×16] 后 TSUBS(-1) = rank (稳定序 = 标量 cnt++ 一致)
//   8.  base = MGATHER(expertSectionTokenCnt, min<<2) (平 MGATHER 查写指针)
//       offE = base + rank → TSHLS(<<2) 字节位移
//   9.  TCI(tok, blk*16)                   — token id 行 ramp ([C5])
//   10. MSCATTER(groupedTokenIds, tok, offB)
//   11. 每 pod p: TMULS(offE*superPodNum)+TADDS(p)+TSHLS(<<2) +
//       MSCATTER(tokenSuperPodInfo, podFlagRow_p, poB)
//   12. 块末进位: 每 section 计数链 ([C10]) + 标量 volatile RMW 更新
//       expertSectionTokenCnt (= 输出计数器本身, 终值 = 各区总数)
//
// 落地顺序 = token 升序 (块顺序 × 块内 row-major), 与标量版逐元素一致。
// DoAtomicAdd=false 为性能对照分支 (driver 从不实例化), 保留原标量实现。
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
    if constexpr (!DoAtomicAdd) {
        // ---- 性能对照分支 (非原子, 结果故意不正确; 保持原标量实现) ----
        for (uint32_t i = 0; i < expertPerRank; i++) {
            expertSectionTokenCnt[i] = 0;
        }
        uint32_t dstPodLocal[kSuperPodNum];
        for (uint32_t i = 0; i < batchSize; i++) {
            uint32_t minLocalExpId = expertPerRank;
            for (uint32_t s = 0; s < superPodNum; s++) dstPodLocal[s] = 0;
            uint32_t stop = (i + 1) * topk;
            for (uint32_t j = i * topk; j < stop; j++) {
                uint32_t curLocalExpId = topkIndex[j] % expertPerRank;
                if (curLocalExpId < minLocalExpId) minLocalExpId = curLocalExpId;
                uint32_t curDstPod = topkIndex[j] / expertPerPod;
                if (curDstPod < superPodNum) dstPodLocal[curDstPod] = 1;
            }
            uint32_t idxInSection = expertSectionTokenCnt[minLocalExpId] + 1;
            groupedTokenIds[minLocalExpId * batchSize + idxInSection] = i;
            uint32_t podInfoSectionOffset =
                minLocalExpId * batchSize * superPodNum + idxInSection * superPodNum;
            for (uint32_t s = 0; s < superPodNum; s++) {
                tokenSuperPodInfo[podInfoSectionOffset + s] = dstPodLocal[s];
            }
        }
        return;
    }

    using namespace gt_tile;

    // 计数器清零: expertPerRank(≤4) 个 u32 = 16B < 128B tile 下限 →
    // volatile 标量逐元素 ([C9])
    {
        volatile uint32_t *v = expertSectionTokenCnt;
        for (uint32_t i = 0; i < expertPerRank; i++) v[i] = 0u;
    }

    // GM 往返 scratch ([C4] 列→行转换 + rank 出口 + 计数链单值出口)
    static uint32_t minScratch[kTileM];
    static uint32_t podScratch[kSuperPodNum][kTileM];
    static uint32_t rankScratch[kTileM];
    static uint32_t cntGm[1];

    using itTopk = global_iterator<GmTopkIndex, TileU32>;
    itTopk gIter(topkIndex);

    constexpr uint32_t Mb = kBS / kTileM;
    GFlat gIds(groupedTokenIds,
               static_cast<int>(kExpertPerRank * kBS), 1);
    GFlat gPodInfo(tokenSuperPodInfo,
                   static_cast<int>(kExpertPerRank * kBS * kSuperPodNum), 1);
    global_tensor<uint32_t, RowMajor<1, kExpertPerRank>> gCnt(
        expertSectionTokenCnt);

    for (uint32_t blk = 0; blk < Mb; ++blk) {
        // 1. 载入 16 token × 16 expert id
        auto gI = gIter(blk, 0);
        T16x16 t;
        TLOAD(t, gI);

        // 2-4. minLocalExpId: TREMS → TROWMIN [16×1] → GM 往返 → [1×16]
        T16x16 rem;
        TREMS(rem, t, expertPerRank);
        TRed16 minCol;
        TROWMIN(minCol, rem);
        G16x1 gMinW(minScratch);
        TSTORE(gMinW, minCol);
        TCol16 minRow;
        G1x16 gMinR(minScratch);
        TLOAD(minRow, gMinR);

        // 5-6. pod any-flag: TDIVS → 每 p TCMPS<EQ>+TSEL+TROWMAX → scratch
        //      (常量 tile 迭代内重物化 —— 避免 loop-carried tile 触发编译器
        //       TMOV bank 重排束, gfrun 对其解码为 "TLOAD empty dstTile")
        for (uint32_t p = 0; p < superPodNum; ++p) {
            // pod 每迭代重物化 (谓词生成循环不得有 loop-carried tile ——
            // 回边 TMOV bank 重排束含 U8 谓词重载, gfrun 解码为
            // "TLOAD empty dstTile" 后状态损坏)
            T16x16 pod;
            TDIVS(pod, t, expertPerPod);
            T16x16 pred;
            TCMPS<CmpMode::EQ>(pred, pod, p);
            T16x16 onev;
            TEXPANDS(onev, 1u);
            T16x16 sel;
            TEXPANDS(sel, 0u);
            TSEL(sel, pred, onev);
            TRed16 flagCol;
            TROWMAX(flagCol, sel);
            G16x1 gFlagW(podScratch[p]);
            TSTORE(gFlagW, flagCol);
        }

        // 7. 成对比较 rank ([C11], 替代 MGATHER_ADD —— TimingSim 无 TLSU
        //    原子族 [C12]): rankIncl[i] = 1 + #{j<i : min_j == min_i}
        //    (t/pod/rem 已死, 峰值活跃 1KB tile = Mc/Mr/eq/tri/mat 五个)
        T16x16 Mc;
        TROWEXPAND(Mc, minCol);          // Mc[i][j] = min[i] (单列源右广播)
        T16x16 Mr;
        TCOLEXPAND(Mr, minRow);          // Mr[i][j] = min[j] (源物理 Cols
                                         // 须 == dst Cols [C11])
        T16x16 eq;
        TCMP<CmpMode::EQ>(eq, Mc, Mr);
        T16x16 tri;
        TTRI(tri);                       // 下三角含对角
        T16x16 mat;
        TEXPANDS(mat, 0u);
        TSEL(mat, eq, tri);              // (min_i==min_j) ∧ (j<=i)
        TRed16 rankIncl;
        TROWSUM(rankIncl, mat);
        G16x1 gRankW(rankScratch);
        TSTORE(gRankW, rankIncl);
        TCol16 rankRow;
        G1x16 gRankR(rankScratch);
        TLOAD(rankRow, gRankR);
        TSUBS(rankRow, rankRow, 1u);     // rank = rankIncl - 1 (去对角)

        // 8-10. groupedTokenIds[min*bs + base+rank] = tokenId
        //       base = 平 MGATHER 查当前写指针 (计数器 = 输出本身);
        //       平坦下标 = 段基址 min*batchSize + 段内位置 base+rank
        TCol16 minIdx;
        TSHLS(minIdx, minRow, 2u);
        TCol16 base;
        MGATHER(base, gCnt, minIdx);
        TCol16 minB;
        TMULS(minB, minRow, batchSize);
        TCol16 offE;
        TADD(offE, minB, base);
        TADD(offE, offE, rankRow);
        TCol16 offB;
        TSHLS(offB, offE, 2u);
        TCol16 tok;
        TCI(tok, blk * kTileM);
        MSCATTER(gIds, tok, offB);

        // 11. tokenSuperPodInfo[(min*bs+base+rank)*spn + p] = podFlag_p
        for (uint32_t p = 0; p < superPodNum; ++p) {
            TCol16 flagRow;
            G1x16 gFlagR(podScratch[p]);
            TLOAD(flagRow, gFlagR);
            TCol16 po;
            TMULS(po, offE, superPodNum);
            TADDS(po, po, p);
            TCol16 poB;
            TSHLS(poB, po, 2u);
            MSCATTER(gPodInfo, flagRow, poB);
        }

        // 12. 块末 base 进位: expertSectionTokenCnt[s] += #{本块 min==s}
        //     (计数链 [C10] 于 [1×16] + 标量 volatile RMW ≤4 元素)
        for (uint32_t s = 0; s < expertPerRank; ++s) {
            TCol16 minRow2;                 // 循环内重物化 (同上契约)
            G1x16 gMinR2(minScratch);
            TLOAD(minRow2, gMinR2);
            TCol16 pred2;
            TCMPS<CmpMode::EQ>(pred2, minRow2, s);
            TCol16 one2;
            TEXPANDS(one2, 1u);
            TCol16 sel2;
            TEXPANDS(sel2, 0u);
            TSEL(sel2, pred2, one2);
            TSum1 cnt;
            TROWSUM(cnt, sel2);
            G1x1 gC(cntGm);
            TSTORE(gC, cnt);
            volatile uint32_t *vc = expertSectionTokenCnt;
            vc[s] = vc[s] + cntGm[0];
        }
    }
}

// ============================================================================
// Phase 3a (Tile): TLOAD + TREMS + TROWMIN + TSTORE
//
// 与 _mt 版 Phase 3a 同款真 tile 流水 (旧版因 "TEPL 输出→TSTORE 中间 TMOV
// 被拒" 退回标量, 该 gfrun v0.3 限制已消失 —— _mt 版与本轮探针 P1 双重
// 实证 TREMS→TROWMIN→TSTORE 链可用)。
// ============================================================================
static inline void floorFunc_tile(uint32_t *topkIndex,
                                     uint32_t *minLocalExpIds,
                                     uint32_t batchSize,
                                     uint32_t topk,
                                     uint32_t expertPerRank)
{
    using namespace gt_tile;

    using itTopk = global_iterator<GmTopkIndex, TileU32>;
    itTopk gIterIn(topkIndex);

    constexpr uint32_t Mb = kBS / kTileM;
    for (uint32_t i = 0; i < Mb; ++i) {
        auto gI = gIterIn(i, 0);
        T16x16 t;
        TLOAD(t, gI);
        T16x16 rem;
        TREMS(rem, t, expertPerRank);   // localExpId = expertId % epr
        TRed16 m;
        TROWMIN(m, rem);                // 每 token 最小 localExpId [16×1]
        G16x1 gOut(minLocalExpIds + i * kTileM);
        TSTORE(gOut, m);                // -> minLocalExpIds[16i..16i+15]
    }
}

// ============================================================================
// Phase 3b (scalar): counting sort — 标量参照实现 (tile 版兜底路径,
// batchSize 非 32 整数倍等契约违例时使用)
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
// Phase 3b (Tile): counting sort — 全 tile 化实现 (gfsim 兼容面板 [C12])
//
//   counts        = 每 1×32 tile 每 bin: TCMPS<EQ>+TSEL+TROWSUM([32×1,v1×1])
//                   +TSTORE→标量寄存器累加 ([C10]; bin 循环天然限定合法值域,
//                   无需守卫; 替代 MSCATTER_ADD)
//   sectionStarts = counts 前缀和 (≤5 元素 < 128B tile 下限, 标量)
//   散射          = 每 1×32 tile 成对比较 rank ([C11], 替代 MGATHER_ADD):
//                   同一 GM 数据双视图 TLOAD ([1×32] 行 + [32×1] 列) →
//                   TROWEXPAND/TCOLEXPAND [32×32] → TCMP<EQ> → TSEL(∧TTRI)
//                   → TROWSUM → rankIncl-1; base = 平 MGATHER(writePos);
//                   pos = base + rank → TCI(token ramp) + MSCATTER
//   writePos 进位 = 每 tile 末 per-bin 计数链 + 标量 volatile RMW
//
// 稳定序保证: tile 按 token 序, 块内成对 rank = #{j<i} 精确前缀计数,
// writePos 跨块进位 → 与标量版逐元素一致 (driver 精确匹配校验)。
// 契约: batchSize % 32 == 0 且 expertPerRank ≤ kExpertPerRank, 违例回退标量。
// ============================================================================
static inline void sortByLocalExpId_tile(const uint32_t *minLocalExpIds,
                                           uint32_t *sortedTokenIds,
                                           uint32_t *sectionStarts,
                                           uint32_t batchSize,
                                           uint32_t expertPerRank)
{
    if ((batchSize % 32u) != 0u || expertPerRank > kExpertPerRank ||
        expertPerRank == 0u) {
        sortByLocalExpId_scalar(minLocalExpIds, sortedTokenIds, sectionStarts,
                                 batchSize, expertPerRank);
        return;
    }

    using namespace gt_tile;

    static uint32_t writePos[kExpertPerRank];
    static uint32_t cntGm[1];
    static uint32_t rankBuf[32];

    const uint32_t nTiles = batchSize / 32u;

    // ---- counts: per-bin 计数链 ([C10]; 外 tile 内 bin, TLOAD 每 tile 一次) ----
    uint32_t acc[kExpertPerRank];
    for (uint32_t e = 0; e < kExpertPerRank; ++e) acc[e] = 0u;
    {
        for (uint32_t tb = 0; tb < nTiles; ++tb) {
            G1x32 gS(const_cast<uint32_t *>(minLocalExpIds) + tb * 32u);
            T1x32 s;
            TLOAD(s, gS);
            for (uint32_t e = 0; e < expertPerRank; ++e) {
                T1x32 s2;                   // 循环内重物化 (谓词循环无
                G1x32 gS2(const_cast<uint32_t *>(minLocalExpIds) + tb * 32u);
                TLOAD(s2, gS2);             // loop-carried tile 契约)
                T1x32 pred;
                TCMPS<CmpMode::EQ>(pred, s2, e);
                T1x32 one;
                TEXPANDS(one, 1u);
                T1x32 sel;
                TEXPANDS(sel, 0u);
                TSEL(sel, pred, one);
                TSum1 c;
                TROWSUM(c, sel);
                G1x1 gC(cntGm);
                TSTORE(gC, c);
                acc[e] += cntGm[0];
            }
        }
    }

    // ---- sectionStarts 前缀和 + writePos 初始化 (≤5 元素标量, [C9]) ----
    sectionStarts[0] = 0;
    for (uint32_t e = 0; e < expertPerRank; e++) {
        sectionStarts[e + 1] = sectionStarts[e] + acc[e];
    }
    {
        volatile uint32_t *w = writePos;
        for (uint32_t e = 0; e < expertPerRank; e++) w[e] = sectionStarts[e];
    }

    // ---- 散射: sortedTokenIds[writePos[section]++] = token (稳定序) ----
    GFlat gSorted(sortedTokenIds, static_cast<int>(batchSize), 1);
    global_tensor<uint32_t, RowMajor<1, kExpertPerRank>> gWP(writePos);
    for (uint32_t tb = 0; tb < nTiles; ++tb) {
        // 同一 GM 数据双视图: [1×32] 行 + [32×1] 列 (免 GM 往返)
        G1x32 gS(const_cast<uint32_t *>(minLocalExpIds) + tb * 32u);
        T1x32 sRow;
        TLOAD(sRow, gS);
        G32x1 gSC(const_cast<uint32_t *>(minLocalExpIds) + tb * 32u);
        TRed32 sCol;
        TLOAD(sCol, gSC);

        // 成对比较 rank ([C11]): rankIncl[i] = 1 + #{j<i : sec_j == sec_i}
        T32x32 Mc;
        TROWEXPAND(Mc, sCol);            // Mc[i][j] = sec[i]
        T32x32 Mr;
        TCOLEXPAND(Mr, sRow);            // Mr[i][j] = sec[j]
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

        // pos = writePos[sec] + rank → MSCATTER(token ramp)
        T1x32 sIdx;
        TSHLS(sIdx, sRow, 2u);
        T1x32 base;
        MGATHER(base, gWP, sIdx);
        T1x32 pos;
        TADD(pos, base, rankRow);
        T1x32 posB;
        TSHLS(posB, pos, 2u);
        T1x32 tok;
        TCI(tok, tb * 32u);
        MSCATTER(gSorted, tok, posB);

        // writePos 进位: 每 bin 计数链 + 标量 volatile RMW
        for (uint32_t e = 0; e < expertPerRank; ++e) {
            T1x32 sRow2;                    // 循环内重物化 (同上契约)
            G1x32 gS2(const_cast<uint32_t *>(minLocalExpIds) + tb * 32u);
            TLOAD(sRow2, gS2);
            T1x32 pred;
            TCMPS<CmpMode::EQ>(pred, sRow2, e);
            T1x32 one;
            TEXPANDS(one, 1u);
            T1x32 sel;
            TEXPANDS(sel, 0u);
            TSEL(sel, pred, one);
            TSum1 c;
            TROWSUM(c, sel);
            G1x1 gC(cntGm);
            TSTORE(gC, c);
            volatile uint32_t *w = writePos;
            w[e] = w[e] + cntGm[0];
        }
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

    sortByLocalExpId_tile(minLocalExpIds, sortedTokenIds, sectionStarts,
                           kBS, kExpertPerRank);
}

#endif // GROUP_TOKEN_VEC_HPP
