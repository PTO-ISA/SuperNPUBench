#ifndef SUPERNPU_MOE_DISPATCH_MT_HPP
#define SUPERNPU_MOE_DISPATCH_MT_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "solution/moe_dispatch/dispatch_tile_common.hpp"

namespace supernpu::tile_isa {

// ============================================================================
// MoE Dispatch — Multi-thread 4-PE SPMD variant
//
// This is the four-PE version of kernels/solution/moe_dispatch/moe_dispatch_v2.hpp.
// The operator semantics are unchanged (pack x into 512B-stride window slots,
// per-expert cumsum, expert-major copy-out into continuous expandXOut); only
// the execution is partitioned across PEs with get_thread_idx():
//
//   Phase 1  dispatch_pack_mt   PE tid owns slots [tid*slotsPerPE,
//                               (tid+1)*slotsPerPE). x rows, window rows,
//                               flags and triples are all PE-disjoint.
//   Phase 2  cal_cumsum_mt      per-PE local histogram over own slots
//                               (计数链 [C10]: TLOAD [1×N] + 每 bin
//                               TCMPS<EQ>+TSEL+TROWSUM+TSTORE → 标量寄存器
//                               累加, gfsim 兼容; 契约见
//                               dispatch_tile_common.hpp), histogram 完成后
//                               mtBarrier(2) 再 cross-PE reduce (≤E×4 元素
//                               标量, 低于 tile 粒度); each PE writes its
//                               assigned expert range of sendCountsOut /
//                               expertTokenNumsOut / expertStarts.
//   Phase 3  dispatch_copyout_mt PE tid re-emits its own slots at exactly the
//                               expert-major position the single-PE version
//                               produces: dstPos = expertStarts[eid] + rank.
//                               rank 批量 tile 化 (成对比较 rank [C11],
//                               gfsim 兼容; 替代 MGATHER_ADD): per-PE 进位
//                               计数器预置 crossPePrefix[eid] = Σ_{t<tid}
//                               cntLocal[t][eid] (cntLocal 段为连续 [begin,i)
//                               前缀计数, 段间前缀即跨 PE 偏移) → 块内
//                               [32×32] TROWEXPAND/TCOLEXPAND/TCMP<EQ>/
//                               TSEL(∧TTRI)/TROWSUM = #{j<i} 精确前缀 +
//                               平 MGATHER(rankCnt carry / expertStarts)
//                               → TADD → TSTORE dstPosBuf; 块末 per-bin
//                               计数链进位; 输出循环标量 (行级 copy-out/
//                               flag 链原有 tile 保留)。
//
// Tile rules (established toolchain contract, 探针实证 [C1..C9]):
//   1. Every TLOAD result is consumed by tile ops and leaves the tile
//      domain via TSTORE/MSCATTER; scalar reads hit GM.
//   2. Per-PE tiles are disjoint — no duplicated TLOAD traffic.
//   3. 原子族/标量 TEPL 一律 [1×N] 行 tile (N≥2, lb0 必须存在 [C4]);
//      段尾 1 slot 走标量。
//   4. Cross-PE hand-offs are guarded by mtBarrier.
// ============================================================================

constexpr int kMtThreadsPerBlock = 4;

// Multi-PE barrier: volatile per-PE phase flags + compiler memory barrier,
// same convention as kernels/solution/group_token_vec/group_token_vec_mt.hpp.
static volatile uint32_t sMtPhaseDone[kMtThreadsPerBlock];

static inline void mtCompilerBarrier()
{
    __asm__ volatile("" : : : "memory");
}

static inline void mtBarrier(uint32_t phase)
{
    mtCompilerBarrier();
    sMtPhaseDone[get_thread_idx()] = phase;
    mtCompilerBarrier();
    for (int t = 0; t < kMtThreadsPerBlock; ++t) {
        while (sMtPhaseDone[t] < phase) {
        }
    }
    mtCompilerBarrier();
}

// ====== Phase 1: Pack (PE tid owns slots [tid*slotsPerPE, +slotsPerPE)) ======
template <typename DType, int BS, int H, int K, int MoeExpertNum, int TileW,
          int WindowStride>
void dispatch_pack_mt(DType* x, int32_t* expertIds, int32_t* windowTriple,
                      DType* windowData, float* windowFlag)
{
    constexpr int slotCount = BS * K;
    constexpr int hTiles = H / TileW;
    constexpr int slotsPerPE = slotCount / kMtThreadsPerBlock;
    const int tid = static_cast<int>(get_thread_idx());
    using namespace pto;

    using gm_x    = global_tensor<DType, RowMajor<BS, H>>;
    using gm_w    = global_tensor<DType, RowMajor<slotCount, WindowStride>>;
    using gm_flag = global_tensor<float,   RowMajor<slotCount, TileW>>;
    using tile_d  = Tile<Location::Vec, DType, 1, TileW, BLayout::RowMajor>;
    using tile_f  = Tile<Location::Vec, float,  1, TileW, BLayout::RowMajor>;
    using it_x    = global_iterator<gm_x,    tile_d>;
    using it_w    = global_iterator<gm_w,    tile_d>;
    using it_flag = global_iterator<gm_flag, tile_f>;

    // Full-tensor iterators; each PE addresses its own slot range so the
    // per-PE tiles are disjoint (rule 2).
    it_x    x_iter(x);
    it_w    w_iter(windowData);
    it_flag flag_iter(windowFlag);

    for (int tk = tid * slotsPerPE; tk < (tid + 1) * slotsPerPE; tk++) {
        int tokenId = tk / K;
        int topkId  = tk % K;

        for (int t = 0; t < hTiles; t++) {
            tile_d xq;
            auto gx = x_iter(tokenId, t);
            TLOAD(xq, gx);

            // Pipeline sync (SyncFunc<MTE2_V> aligned); TSUB as TMOV
            // stand-in (see moe_dispatch_v2.hpp note).
            tile_d sync_d;
            TSUB(sync_d, xq, xq);

            // 512B stride write: data at [tk*WindowStride + t*TileW]
            auto gw = w_iter(tk, t);
            TSTORE(gw, xq);
        }

        // 512B block packing: flag fill (TEXPANDS + TSTORE)
        tile_f flagTile;
        TEXPANDS(flagTile, 1.0f);

        // Pipeline sync (SyncFunc<V_MTE3> aligned); TSUB as TMOV stand-in.
        tile_f sync_f;
        TSUB(sync_f, flagTile, flagTile);

        auto gf = flag_iter(tk, 0);
        TSTORE(gf, flagTile);

        // FillTriple (scalar, 12B — too small for tile)
        windowTriple[tk * 3 + 0] = 0;
        windowTriple[tk * 3 + 1] = tokenId;
        windowTriple[tk * 3 + 2] = topkId;
    }
}

// ====== Phase 2: CalCumSum (per-PE tile histogram + cross-PE reduce) ======
// cntLocal layout: [pe * MoeExpertNum + expert]. After the histogram barrier
// every PE reduces its assigned expert range; expertStarts[e] holds the
// exclusive prefix (number of slots owned by experts < e), which Phase 3
// uses to reproduce the single-PE expert-major output order.
//
// 直方图 = 计数链 ([C10], gfsim 兼容; 替代 MSCATTER_ADD —— TimingSim 无
// TLSU 原子族 [C12]): 每 PE 自己的 slotsPerPE 段, 按 32 分块 + [1×rem]
// 尾块 (rem==1 时该 slot 标量); 每 bin e: TCMPS<EQ>+TSEL(物化0/1)+
// TROWSUM+TSTORE → 标量寄存器累加 → 末端 volatile 导出 myCnt ([C9])。
// bin 循环天然限定合法值域 (等价标量守卫)。
// reduce (sum/before 双前缀) ≤ E×4 元素, 低于 128B tile 粒度 → 标量。
// 注: reduce 读全部 PE 的 cntLocal 行 → 直方图导出后 mtBarrier(2)
// (原实现无此栅栏, 依赖确定性调度侥幸; gtv mt 同款结构化保证)。
template <int BS, int K, int MoeExpertNum>
void cal_cumsum_mt(int32_t* expertIds, int32_t* sendCountsOut,
                   int64_t* expertTokenNumsOut, int32_t* cntLocal,
                   int32_t* expertStarts)
{
    constexpr int slotCount = BS * K;
    constexpr int slotsPerPE = slotCount / kMtThreadsPerBlock;
    constexpr int expertsPerPE = MoeExpertNum / kMtThreadsPerBlock;
    const int tid = static_cast<int>(get_thread_idx());

    // Per-PE local histogram over this PE's own slots (计数链 [C10])
    int32_t* myCnt = cntLocal + tid * MoeExpertNum;
    int32_t acc[MoeExpertNum];
    for (int e = 0; e < MoeExpertNum; e++) acc[e] = 0;
    {
        using namespace dispatch_tile;
        static int32_t cntGm[kMtThreadsPerBlock];   // per-PE 单值出口 (写不相交)
        int32_t* mySum = cntGm + tid;
        const int segBegin = tid * slotsPerPE;
        constexpr int kNTiles = slotsPerPE / 32;
        constexpr int kRem = slotsPerPE % 32;
        for (int tb = 0; tb < kNTiles; tb++) {
            GI1x32 gIds(expertIds + segBegin + tb * 32);
            TI1x32 ids;
            TLOAD(ids, gIds);
            for (int e = 0; e < MoeExpertNum; e++) {
                TI1x32 pred;
                TCMPS<CmpMode::EQ>(pred, ids, static_cast<int32_t>(e));
                TI1x32 one;
                TEXPANDS(one, static_cast<int32_t>(1));
                TI1x32 sel;
                TEXPANDS(sel, static_cast<int32_t>(0));
                TSEL(sel, pred, one);
                TSum1I c;
                TROWSUM(c, sel);
                GI1x1 gC(mySum);
                TSTORE(gC, c);
                acc[e] += *mySum;
            }
        }
        if constexpr (kRem >= 2) {
            using TIRem = Tile<Location::Vec, int32_t, 1, 32,
                               BLayout::RowMajor, 1, kRem>;
            global_tensor<int32_t, RowMajor<1, kRem>> gIdsR(
                expertIds + segBegin + kNTiles * 32);
            TIRem ids;
            TLOAD(ids, gIdsR);
            for (int e = 0; e < MoeExpertNum; e++) {
                TIRem pred;
                TCMPS<CmpMode::EQ>(pred, ids, static_cast<int32_t>(e));
                TIRem one;
                TEXPANDS(one, static_cast<int32_t>(1));
                TIRem sel;
                TEXPANDS(sel, static_cast<int32_t>(0));
                TSEL(sel, pred, one);
                TSum1I c;
                TROWSUM(c, sel);
                GI1x1 gC(mySum);
                TSTORE(gC, c);
                acc[e] += *mySum;
            }
        } else if constexpr (kRem == 1) {
            const int i = segBegin + kNTiles * 32;
            const int32_t eid = expertIds[i];
            if (eid >= 0 && eid < MoeExpertNum) acc[eid]++;
        }
    }
    {
        volatile int32_t* vc = myCnt;
        for (int e = 0; e < MoeExpertNum; e++) vc[e] = acc[e];
    }
    // 全部 PE 的直方图导出完成后才可 cross-PE reduce (读 cntLocal 全行)
    mtBarrier(2);

    // Reduce: each PE writes its assigned expert range
    // (≤ E×4 元素双前缀, 低于 tile 粒度 → 标量)
    for (int e = 0; e < expertsPerPE; e++) {
        int ge = tid * expertsPerPE + e;
        int32_t sum = 0;
        int32_t before = 0;
        for (int e2 = 0; e2 <= ge; e2++) {
            for (int t = 0; t < kMtThreadsPerBlock; t++) {
                int32_t c = cntLocal[t * MoeExpertNum + e2];
                if (e2 == ge) sum += c; else before += c;
            }
        }
        expertTokenNumsOut[ge] = sum;
        sendCountsOut[ge] = before + sum;   // inclusive cumsum (v2 semantics)
        expertStarts[ge] = before;          // exclusive start for Phase 3
    }
}

// ====== CUMSUM flag write (PE0 only; TEXPANDS + TSTORE at windowState+4) ======
template <int TileW>
void write_cumsum_flag(uint32_t* windowState)
{
    using namespace pto;
    using tile_f = Tile<Location::Vec, float, 1, TileW, BLayout::RowMajor>;
    tile_f cumsumFlag;
    TEXPANDS(cumsumFlag, 1.0f);

    // TSUB as TMOV stand-in (see moe_dispatch_v2.hpp note).
    tile_f sync_cf;
    TSUB(sync_cf, cumsumFlag, cumsumFlag);

    using gm_st = global_tensor<float, RowMajor<1, TileW>>;
    auto gs = reinterpret_cast<gm_st*>(windowState + 4);
    TSTORE(*gs, cumsumFlag);
}

// ====== Flag check — 真 EQ 谓词 tile 链 (fp32 TCMPS<EQ> 0923 基线探针
// 实证可用, 旧 pass-through 退化删除; 详见 moe_dispatch_v2.hpp 注) ======
template <int BS, int K, int TileW>
void check_flag_mt(float* windowFlag, float* predBuf, int srcSlot)
{
    constexpr int slotCount = BS * K;
    using namespace pto;
    using gm_flag = global_tensor<float, RowMajor<slotCount, TileW>>;
    using gm_pred = global_tensor<float, RowMajor<slotCount, TileW>>;
    using tile_f  = Tile<Location::Vec, float,  1, TileW, BLayout::RowMajor>;
    using tile_i  = Tile<Location::Vec, int32_t, 1, TileW, BLayout::RowMajor>;
    using it_flag = global_iterator<gm_flag, tile_f>;
    using it_pred = global_iterator<gm_pred, tile_f>;

    it_flag flag_iter(windowFlag);
    it_pred pred_iter(predBuf);

    tile_f flagTile;
    auto gf = flag_iter(srcSlot, 0);
    TLOAD(flagTile, gf);

    tile_f pred;
    TCMPS<CmpMode::EQ>(pred, flagTile, 1.0f);
    tile_i oneI;
    TEXPANDS(oneI, static_cast<int32_t>(1));
    tile_i selI;
    TEXPANDS(selI, static_cast<int32_t>(0));
    TSEL(selI, pred, oneI);
    tile_f norm;
    TCVT(norm, selI);

    auto gp = pred_iter(srcSlot, 0);
    TSTORE(gp, norm);
}

// ====== CUMSUM flag check — 真 EQ 谓词 tile 链 (PE0 only) ======
template <int TileW>
void check_cumsum_flag_mt(uint32_t* windowState, float* predBuf)
{
    using namespace pto;
    using gm_st  = global_tensor<float, RowMajor<1, TileW>>;
    using gm_pred = global_tensor<float, RowMajor<1, TileW>>;
    using tile_f = Tile<Location::Vec, float,  1, TileW, BLayout::RowMajor>;
    using tile_i = Tile<Location::Vec, int32_t, 1, TileW, BLayout::RowMajor>;
    using it_pred = global_iterator<gm_pred, tile_f>;

    auto gs = reinterpret_cast<gm_st*>(windowState + 4);
    it_pred pred_iter(predBuf);

    tile_f readFlag;
    TLOAD(readFlag, *gs);

    tile_f pred;
    TCMPS<CmpMode::EQ>(pred, readFlag, 1.0f);
    tile_i oneI;
    TEXPANDS(oneI, static_cast<int32_t>(1));
    tile_i selI;
    TEXPANDS(selI, static_cast<int32_t>(0));
    TSEL(selI, pred, oneI);
    tile_f norm;
    TCVT(norm, selI);

    auto gp = pred_iter(0, 0);
    TSTORE(gp, norm);
}

// ====== Clear flag ======
template <int BS, int K, int TileW>
void clear_flag_mt(float* windowFlag, int srcSlot)
{
    constexpr int slotCount = BS * K;
    using namespace pto;
    using gm_flag = global_tensor<float, RowMajor<slotCount, TileW>>;
    using tile_f  = Tile<Location::Vec, float, 1, TileW, BLayout::RowMajor>;
    using it_flag = global_iterator<gm_flag, tile_f>;
    it_flag flag_iter(windowFlag);

    tile_f zeroFlag;
    TEXPANDS(zeroFlag, 0.0f);

    // TSUB as TMOV stand-in (see moe_dispatch_v2.hpp note).
    tile_f sync_zf;
    TSUB(sync_zf, zeroFlag, zeroFlag);

    auto gf = flag_iter(srcSlot, 0);
    TSTORE(gf, zeroFlag);
}

// ====== Phase 3: Read data from window → expandXOut (PE-disjoint rows) ======
template <typename DType, int BS, int H, int K, int TileW, int WindowStride>
void dispatch_copy_out_mt(DType* windowData, DType* expandXOut,
                          int32_t srcSlot, int dstPos)
{
    constexpr int hTiles = H / TileW;
    using namespace pto;

    using gm_w   = global_tensor<DType, RowMajor<BS * K, WindowStride>>;
    using gm_out = global_tensor<DType, RowMajor<BS * K, H>>;
    using tile_d = Tile<Location::Vec, DType, 1, TileW, BLayout::RowMajor>;
    using it_w   = global_iterator<gm_w,   tile_d>;
    using it_out = global_iterator<gm_out, tile_d>;

    it_w   w_iter(windowData);
    it_out out_iter(expandXOut);

    for (int t = 0; t < hTiles; t++) {
        tile_d xq;
        auto gw = w_iter(srcSlot, t);
        TLOAD(xq, gw);

        // Pipeline sync (SyncFunc<MTE2_V> aligned)
        tile_d sync_d;
        TSUB(sync_d, xq, xq);

        auto gout = out_iter(dstPos, t);
        TSTORE(gout, xq);
    }
}

// ====== Phase 3 helper: 批量 rank + dstPos tile 链 ======
// 对 PE 段 [segBegin, segBegin+SegLen) 的 slot 批量计算 (成对比较 rank
// [C11], gfsim 兼容; 替代 MGATHER_ADD —— TimingSim 无 TLSU 原子族 [C12]):
//   块内 rank = [32×32] TROWEXPAND/TCOLEXPAND/TCMP<EQ>/TSEL(∧TTRI)/TROWSUM
//     → rankIncl-1 = #{j<i : eid_j==eid_i} (稳定序 = 原子写指针 row-major
//     确定序 [C3] = 标量 "for j<i" 前缀计数)
//   全局 rank = 块内 rank + carry; carry = 平 MGATHER(rankCnt, idx<<2),
//     rankCnt 预置 crossPePrefix 且每块末 per-bin 计数链进位 (标量
//     volatile RMW) → carry = 段内前块计数 + 跨 PE 前缀 = 全局前缀
//   dstPos = 平 MGATHER(expertStarts, idx<<2) + 全局 rank → TSTORE dstPosBuf
// 形状: 32-slot 满 tile + [1×rem] 尾 tile (rem>=2); rem==1 该 slot 标量。
// 尾块列视图 [32×1] TLOAD 越段读取落在 driver 4096 对齐 bss 裕量内
// (mapped, 垃圾 lane 被 TTRI 掩出有效行前缀, rank 仅垃圾行消费——输出
// 循环按 expertIds 跳过); GM 面写回一律 [1×rem] 静态 valid, 不越段写。
// 非法 eid lane: index 钳 0 → carry/starts 为桶 0 查表值, rank 为垃圾,
// 输出循环按 expertIds 跳过, 不消费。
template <int SegLen, int MoeExpertNum>
static inline void rank_dstpos_tile(const int32_t* expertIds, int segBegin,
                                     int32_t* rankCnt,
                                     const int32_t* expertStarts,
                                     int32_t* dstPosBuf)
{
    using namespace dispatch_tile;
    constexpr int kNTiles = SegLen / 32;
    constexpr int kRem = SegLen % 32;
    const int tid = static_cast<int>(get_thread_idx());
    static int32_t rankBuf[kMtThreadsPerBlock][32];  // 成对 rank GM 往返 ([C4])
    static int32_t cntGm[kMtThreadsPerBlock];        // 进位计数链单值出口

    for (int tb = 0; tb < kNTiles; tb++) {
        GI1x32 gIds(const_cast<int32_t*>(expertIds) + segBegin + tb * 32);
        TI1x32 ids;
        TLOAD(ids, gIds);
        GI32x1 gIdsC(const_cast<int32_t*>(expertIds) + segBegin + tb * 32);
        TI32x1 idsCol;
        TLOAD(idsCol, gIdsC);

        // 成对比较 rank ([C11]): rankIncl[i] = 1 + #{j<=i : eid_j==eid_i}
        TI32x32 Mc;
        TROWEXPAND(Mc, idsCol);          // Mc[i][j] = eid[i]
        TI32x32 Mr;
        TCOLEXPAND(Mr, ids);             // Mr[i][j] = eid[j]
        TI32x32 eq;
        TCMP<CmpMode::EQ>(eq, Mc, Mr);
        TI32x32 tri;
        TTRI(tri);                       // 下三角含对角
        TI32x32 mat;
        TEXPANDS(mat, static_cast<int32_t>(0));
        TSEL(mat, eq, tri);
        TI32x1 rankIncl;
        TROWSUM(rankIncl, mat);
        GI32x1 gRankW(rankBuf[tid]);
        TSTORE(gRankW, rankIncl);
        TI1x32 rankRow;
        GI1x32 gRankR(rankBuf[tid]);
        TLOAD(rankRow, gRankR);
        TSUBS(rankRow, rankRow, static_cast<int32_t>(1));  // 去对角

        // 守卫: 非法 lane index 钳 0 (地址恒界内)
        TI1x32 neg;
        TCMPS<CmpMode::LT>(neg, ids, static_cast<int32_t>(0));
        TI1x32 oob;
        TCMPS<CmpMode::GE>(oob, ids, static_cast<int32_t>(MoeExpertNum));
        TI1x32 zero;
        TEXPANDS(zero, static_cast<int32_t>(0));
        TI1x32 idx;
        TADD(idx, ids, zero);
        TSEL(idx, oob, zero);
        TSEL(idx, neg, zero);
        TI1x32 carry;
        global_tensor<int32_t, RowMajor<1, MoeExpertNum>> gCnt(rankCnt);
        MGATHER(carry, gCnt, idx);
        TI1x32 starts;
        global_tensor<int32_t, RowMajor<1, MoeExpertNum>> gStarts(
            const_cast<int32_t*>(expertStarts));
        MGATHER(starts, gStarts, idx);
        TI1x32 pos;
        TADD(pos, starts, carry);
        TADD(pos, pos, rankRow);
        GI1x32 gPos(dstPosBuf + segBegin + tb * 32);
        TSTORE(gPos, pos);

        // 块末进位: rankCnt[e] += #{本块 eid==e} (计数链 [C10] + volatile RMW)
        for (int e = 0; e < MoeExpertNum; e++) {
            TI1x32 ids2;                    // 循环内重物化 (无 loop-carried)
            GI1x32 gIds2(const_cast<int32_t*>(expertIds) + segBegin + tb * 32);
            TLOAD(ids2, gIds2);
            TI1x32 pred;
            TCMPS<CmpMode::EQ>(pred, ids2, static_cast<int32_t>(e));
            TI1x32 one;
            TEXPANDS(one, static_cast<int32_t>(1));
            TI1x32 sel;
            TEXPANDS(sel, static_cast<int32_t>(0));
            TSEL(sel, pred, one);
            TSum1I c;
            TROWSUM(c, sel);
            GI1x1 gC(cntGm + tid);
            TSTORE(gC, c);
            volatile int32_t* vr = rankCnt;
            vr[e] = vr[e] + cntGm[tid];
        }
    }
    if constexpr (kRem >= 2) {
        using TIRem = Tile<Location::Vec, int32_t, 1, 32,
                           BLayout::RowMajor, 1, kRem>;
        // 尾块成对 rank 形状 = gtv mt T4x4 同款契约 (gfrun
        // ValidateReduceAndExpandTepl 实证): TCOLEXPAND 要求源 tile 注册的
        // 物理 col == dst lb2 且源 validCol == dst lb0 —— [1×32 phys,
        // valid 1×kRem] 源 (col=32) 配 [kRem×kRem] dst (lb2=kRem) 会被拒;
        // 广播源必须用满 [1×kRem] / [32×1 phys, valid kRem×1] (TCol4/TRed4
        // 同款), 矩阵为满 [kRem×kRem]。守卫/MGATHER/TSTORE 链保持
        // [1×32 phys, valid 1×kRem] (TLSU SizeCode ≥128B 契约)。
        using TI1xRem = Tile<Location::Vec, int32_t, 1, kRem,
                             BLayout::RowMajor>;
        using TIMatR = Tile<Location::Vec, int32_t, kRem, kRem,
                            BLayout::RowMajor>;
        using TIRedR = Tile<Location::Vec, int32_t, 32, 1,
                            BLayout::RowMajor, kRem, 1>;
        global_tensor<int32_t, RowMajor<1, kRem>> gIdsR(
            const_cast<int32_t*>(expertIds) + segBegin + kNTiles * 32);
        TIRem ids;
        TLOAD(ids, gIdsR);
        TI1xRem idsRowF;
        TLOAD(idsRowF, gIdsR);
        // 列视图 [32×1 phys, valid kRem×1]: TLOAD 只读 kRem 个有效行,
        // 无越段读取
        global_tensor<int32_t, RowMajor<kRem, 1>> gIdsC(
            const_cast<int32_t*>(expertIds) + segBegin + kNTiles * 32);
        TIRedR idsCol;
        TLOAD(idsCol, gIdsC);
        TIMatR Mc;
        TROWEXPAND(Mc, idsCol);          // Mc[i][j] = eid[i]
        TIMatR Mr;
        TCOLEXPAND(Mr, idsRowF);         // Mr[i][j] = eid[j]
        TIMatR eq;
        TCMP<CmpMode::EQ>(eq, Mc, Mr);
        TIMatR tri;
        TTRI(tri);
        TIMatR mat;
        TEXPANDS(mat, static_cast<int32_t>(0));
        TSEL(mat, eq, tri);
        TIRedR rankIncl;
        TROWSUM(rankIncl, mat);
        global_tensor<int32_t, RowMajor<kRem, 1>> gRankW(rankBuf[tid]);
        TSTORE(gRankW, rankIncl);
        TIRem rankRow;
        global_tensor<int32_t, RowMajor<1, kRem>> gRankR(rankBuf[tid]);
        TLOAD(rankRow, gRankR);
        TSUBS(rankRow, rankRow, static_cast<int32_t>(1));
        TIRem neg;
        TCMPS<CmpMode::LT>(neg, ids, static_cast<int32_t>(0));
        TIRem oob;
        TCMPS<CmpMode::GE>(oob, ids, static_cast<int32_t>(MoeExpertNum));
        TIRem zero;
        TEXPANDS(zero, static_cast<int32_t>(0));
        TIRem idx;
        TADD(idx, ids, zero);
        TSEL(idx, oob, zero);
        TSEL(idx, neg, zero);
        TIRem carry;
        global_tensor<int32_t, RowMajor<1, MoeExpertNum>> gCnt(rankCnt);
        MGATHER(carry, gCnt, idx);
        TIRem starts;
        global_tensor<int32_t, RowMajor<1, MoeExpertNum>> gStarts(
            const_cast<int32_t*>(expertStarts));
        MGATHER(starts, gStarts, idx);
        TIRem pos;
        TADD(pos, starts, carry);
        TADD(pos, pos, rankRow);
        global_tensor<int32_t, RowMajor<1, kRem>> gPos(
            dstPosBuf + segBegin + kNTiles * 32);
        TSTORE(gPos, pos);
        // 尾块进位 (计数链 [C10], [1×kRem] 静态 valid)
        for (int e = 0; e < MoeExpertNum; e++) {
            TIRem ids2;                   // 循环内重物化 (无 loop-carried)
            global_tensor<int32_t, RowMajor<1, kRem>> gIds2(
                const_cast<int32_t*>(expertIds) + segBegin + kNTiles * 32);
            TLOAD(ids2, gIds2);
            TIRem pred;
            TCMPS<CmpMode::EQ>(pred, ids2, static_cast<int32_t>(e));
            TIRem one;
            TEXPANDS(one, static_cast<int32_t>(1));
            TIRem sel;
            TEXPANDS(sel, static_cast<int32_t>(0));
            TSEL(sel, pred, one);
            TSum1I c;
            TROWSUM(c, sel);
            GI1x1 gC(cntGm + tid);
            TSTORE(gC, c);
            volatile int32_t* vr = rankCnt;
            vr[e] = vr[e] + cntGm[tid];
        }
    } else if constexpr (kRem == 1) {
        const int i = segBegin + kNTiles * 32;
        const int32_t eid = expertIds[i];
        if (eid >= 0 && eid < MoeExpertNum) {
            dstPosBuf[i] = expertStarts[eid] + rankCnt[eid]++;
        } else {
            dstPosBuf[i] = -1;
        }
    }
}

// ====== Main entry (multi-PE SPMD; called by every PE) ======
// Cross-PE hand-offs separated by mtBarrier:
//   barrier(1): all PEs' pack writes (windowData/flag/triple) visible before
//               the histogram reduce and the Phase-3 reads
//   barrier(2): all PEs' cntLocal histogram rows complete before the
//               cross-PE reduce reads them (inside cal_cumsum_mt)
//   barrier(3): expertStarts + PE0's cumsum flag visible before Phase 3
//   barrier(4): all outputs fully written before any PE leaves the kernel
// cntLocal needs kMtThreadsPerBlock * MoeExpertNum int32 words,
// expertStarts needs MoeExpertNum int32 words of scratch GM.
template <typename DType, int BS, int H, int K, int MoeExpertNum, int TileW = 128,
          int WindowStride = 256>
void moe_dispatch_mt(
    DType* x, int32_t* expertIds, float* expertScales,
    DType* expandXOut, int32_t* expandIdxOut, float* expandScalesOut,
    int32_t* sendCountsOut, int64_t* expertTokenNumsOut,
    DType* windowData, float* windowFlag, float* predBuf,
    int32_t* windowTriple, uint32_t* windowState, DType* outBuf,
    int32_t* cntLocal, int32_t* expertStarts)
{
    constexpr int slotCount = BS * K;
    constexpr int slotsPerPE = slotCount / kMtThreadsPerBlock;
    static_assert(slotCount % kMtThreadsPerBlock == 0,
                  "slotCount (BS*K) must be divisible by the 4 PEs");
    static_assert(MoeExpertNum % kMtThreadsPerBlock == 0,
                  "MoeExpertNum must be divisible by the 4 PEs");
    const int tid = static_cast<int>(get_thread_idx());

    // Window State Init (InitWinState aligned) — PE0 only
    if (tid == 0) {
        uint32_t dataState = windowState[0];
        windowState[0] = (dataState == 0) ? 1 : 0;
        windowState[1] = 1;
        windowState[2] = 0;
    }

    // ====== Phase 1: AllToAllDispatch (pack x → window + flag) ======
    dispatch_pack_mt<DType, BS, H, K, MoeExpertNum, TileW, WindowStride>(
        x, expertIds, windowTriple, windowData, windowFlag);
    mtBarrier(1);

    // ====== Phase 2: CalCumSum (count + cumsum) ======
    cal_cumsum_mt<BS, K, MoeExpertNum>(expertIds, sendCountsOut,
                                        expertTokenNumsOut, cntLocal,
                                        expertStarts);

    // CUMSUM soft sync + flag check — PE0 only (same PE writes then reads)
    if (tid == 0) {
        write_cumsum_flag<TileW>(windowState);
        check_cumsum_flag_mt<TileW>(windowState, predBuf);
    }
    mtBarrier(3);

    // ====== Phase 3: LocalWindowCopy (read window → continuous output) ======
    // Each PE re-emits its own slots at the expert-major position the
    // single-PE version produces: dstPos = expertStarts[eid] + rank of the
    // slot among the slots of the same expert.
    // rank 批量 tile 化 (rank_dstpos_tile, 成对比较 rank [C11]): per-PE
    // 进位计数器 rankCnt 预置 crossPePrefix[eid] = Σ_{t<tid} cntLocal[t][eid]
    // —— PE 段是连续区间, 段前所有 slot 的同 expert 计数恰为前序 PE 的
    // cntLocal 之和 → 全局 rank = 块内 #{j<i} + 平 MGATHER(rankCnt) carry,
    // 与原标量 "for j<i" 扫描逐 slot 一致。
    // expertIds read-only, cntLocal/expertStarts 由 barrier(1)/(3) 覆盖。
    static int32_t rankCnt[kMtThreadsPerBlock][MoeExpertNum];
    static int32_t dstPosBuf[slotCount];
    {
        volatile int32_t* vr = rankCnt[tid];
        for (int e = 0; e < MoeExpertNum; e++) {
            int32_t cross = 0;
            for (int t = 0; t < tid; t++) {
                cross += cntLocal[t * MoeExpertNum + e];
            }
            vr[e] = cross;
        }
    }
    rank_dstpos_tile<slotsPerPE, MoeExpertNum>(
        expertIds, tid * slotsPerPE, rankCnt[tid], expertStarts, dstPosBuf);

    for (int i = tid * slotsPerPE; i < (tid + 1) * slotsPerPE; i++) {
        int32_t eid = expertIds[i];
        if (eid < 0 || eid >= MoeExpertNum) continue;

        int32_t dstPos = dstPosBuf[i];

        // Flag check (tile pass-through; TCMP unavailable on 0828)
        check_flag_mt<BS, K, TileW>(windowFlag, predBuf, i);

        // Read data from window → expandXOut (tile copy, 512B stride)
        dispatch_copy_out_mt<DType, BS, H, K, TileW, WindowStride>(
            windowData, expandXOut, i, dstPos);

        // Read triple + scale (scalar, 12B/4B < 128B tile 粒度)
        expandIdxOut[dstPos * 3 + 0] = windowTriple[i * 3 + 0];
        expandIdxOut[dstPos * 3 + 1] = windowTriple[i * 3 + 1];
        expandIdxOut[dstPos * 3 + 2] = windowTriple[i * 3 + 2];
        expandScalesOut[dstPos] = expertScales[i];

        // Clear flag (TEXPANDS(0.0) + TSTORE)
        clear_flag_mt<BS, K, TileW>(windowFlag, i);
    }
    mtBarrier(4);

    // Window state writeback — PE0 only
    if (tid == 0) {
        windowState[4] = BS;
    }
}

} // namespace supernpu::tile_isa
#endif
