#ifndef SUPERNPU_MOE_DISPATCH_V2_HPP
#define SUPERNPU_MOE_DISPATCH_V2_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "solution/moe_dispatch/dispatch_tile_common.hpp"

namespace supernpu::tile_isa {

// bf16 H=128 → 256B data per slot. A5 hCommuSize_ = ceil(256,480)*512 = 512B.
// windowStride = 512B / sizeof(bf16) = 256 bf16 elements per slot.
// Data occupies [0, 128), [128, 256) is padding (same as A5 480B block with 32B flag).
//
// Tile 化说明 (契约见 dispatch_tile_common.hpp [C2..C12], tile_probe 实证):
//   全链 gfsim 兼容 (指令面板 [C12]; TimingSim TLSU 无 GM 原子族完成路径,
//   MSCATTER_ADD/MGATHER_ADD 仅 gfrun 可用 → 已替换为等价 tile 链):
//   Phase 2 直方图 = 计数链 ([C10]): 每 32-slot tile TLOAD 一次, 每 bin e:
//     TCMPS<EQ>+TSEL(物化0/1)+TROWSUM([32×1,v1×1])+TSTORE → 标量寄存器累加;
//     bin 循环天然限定合法值域 (等价标量 if (eid>=0 && eid<E), 无需守卫);
//     cumsum 前缀/输出写 (≤E 元素 < 128B tile 下限) 标量。
//   Phase 3 rank = 成对比较 rank ([C11], 替代 MGATHER_ADD): 同一 GM 数据
//     双视图 TLOAD → TROWEXPAND/TCOLEXPAND [32×32] → TCMP<EQ> → TSEL(∧TTRI)
//     → TROWSUM → rankIncl-1 = slot 在同 expert 组内的块内序号; 跨块进位
//     rankCnt = 每块末 per-bin 计数链 + 标量 volatile RMW; carry/starts =
//     平 MGATHER 查表 → TADD → dstPos tile → TSTORE dstPosBuf。稳定序 =
//     原 writePos 序 (块序 × 块内 row-major = 标量前缀计数, 逐 slot 一致)。
//     原 O(n^3) (e,s,i) 三重扫描 → 每 32-slot tile 一组矩阵链 + O(n) 输出。
//   保留标量 (不可 tile, 原因): FillTriple/triple+scale 拷贝 (12B/4B <
//     128B tile 粒度)、windowState (5 字)、cumsum 前缀 (≤E 元素)、
//     flag check/clear 的调用控制流、计数链单值出口读回。

// ====== Phase 1: Pack ======
template <typename DType, int BS, int H, int K, int MoeExpertNum, int TileW,
          int WindowStride>
void dispatch_pack(DType* x, int32_t* expertIds, int32_t* windowTriple,
                   DType* windowData, float* windowFlag)
{
    constexpr int slotCount = BS * K;
    constexpr int hTiles = H / TileW;
    using namespace pto;

    using gm_x    = global_tensor<DType, RowMajor<BS, H>>;
    using gm_w    = global_tensor<DType, RowMajor<slotCount, WindowStride>>;
    using gm_flag = global_tensor<float,   RowMajor<slotCount, TileW>>;
    using tile_d  = Tile<Location::Vec, DType, 1, TileW, BLayout::RowMajor>;
    using tile_f  = Tile<Location::Vec, float,  1, TileW, BLayout::RowMajor>;
    using it_x    = global_iterator<gm_x,    tile_d>;
    using it_w    = global_iterator<gm_w,    tile_d>;
    using it_flag = global_iterator<gm_flag, tile_f>;

    it_x    x_iter(x);
    it_w    w_iter(windowData);
    it_flag flag_iter(windowFlag);

    for (int tk = 0; tk < slotCount; tk++) {
        int tokenId = tk / K;
        int topkId  = tk % K;

        for (int t = 0; t < hTiles; t++) {
            tile_d xq;
            auto gx = x_iter(tokenId, t);
            TLOAD(xq, gx);

            // #3 Pipeline sync (SyncFunc<MTE2_V> aligned)
            // TSUB(dst, src, src) is a compile-time stand-in for TMOV(dst, src):
            // the 0828 toolchain's asm matcher rejects TMOV's 0.58.4 B.DATR
            // syntax ("NORM, DTYPE_NONE, Zero"); TSUB emits no B.DATR and keeps
            // the same src->dst dependency edge (dst = src - src = 0).
            tile_d sync_d;
            TSUB(sync_d, xq, xq);

            // #9 512B stride write: data at [tk*WindowStride + t*TileW]
            auto gw = w_iter(tk, t);
            TSTORE(gw, xq);
        }

        // #1 512B block packing: flag fill (TEXPANDS + TSTORE)
        tile_f flagTile;
        TEXPANDS(flagTile, 1.0f);

        // #3 Pipeline sync (SyncFunc<V_MTE3> aligned)
        // TSUB as TMOV stand-in (see dispatch_pack note).
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

// ====== #4 Flag check — 真 EQ 谓词 tile 链 (fp32 TCMPS<EQ> 0923 基线探针
// tile_probe P16 实证可用, 旧 "0828 matcher 拒绝 TCMP B.DATR" 的
// pass-through 退化删除; TSEL payload 须整数 → int32 物化 + TCVT fp32。
// predBuf is never read back — the chain keeps the original windowFlag read
// side-effect and structural alignment, 且谓词语义与源 A5 一致) ======
template <int BS, int K, int TileW>
void check_flag(float* windowFlag, float* predBuf, int srcSlot)
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

// ====== #7 CUMSUM flag check — 真 EQ 谓词 tile 链 (同 check_flag 注;
// 读 windowState+4 的 TileW-float 区间, 归一化 0/1 写 predBuf) ======
template <int TileW>
void check_cumsum_flag(uint32_t* windowState, float* predBuf)
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

// ====== #1 Clear flag ======
template <int BS, int K, int TileW>
void clear_flag(float* windowFlag, int srcSlot)
{
    constexpr int slotCount = BS * K;
    using namespace pto;
    using gm_flag = global_tensor<float, RowMajor<slotCount, TileW>>;
    using tile_f  = Tile<Location::Vec, float, 1, TileW, BLayout::RowMajor>;
    using it_flag = global_iterator<gm_flag, tile_f>;
    it_flag flag_iter(windowFlag);

    tile_f zeroFlag;
    TEXPANDS(zeroFlag, 0.0f);

    // TSUB as TMOV stand-in (see dispatch_pack note).
    tile_f sync_zf;
    TSUB(sync_zf, zeroFlag, zeroFlag);

    auto gf = flag_iter(srcSlot, 0);
    TSTORE(gf, zeroFlag);
}

// ====== Phase 3: Read data from window → expandXOut ======
template <typename DType, int BS, int H, int K, int TileW, int WindowStride>
void dispatch_copy_out(DType* windowData, DType* expandXOut,
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

        // #3 Pipeline sync (SyncFunc<MTE2_V> aligned)
        // TSUB as TMOV stand-in (see dispatch_pack note).
        tile_d sync_d;
        TSUB(sync_d, xq, xq);

        auto gout = out_iter(dstPos, t);
        TSTORE(gout, xq);
    }
}

// ====== Main entry ======
template <typename DType, int BS, int H, int K, int MoeExpertNum, int TileW = 128,
          int WindowStride = 256>
void moe_dispatch_v2(
    DType* x, int32_t* expertIds, float* expertScales,
    DType* expandXOut, int32_t* expandIdxOut, float* expandScalesOut,
    int32_t* sendCountsOut, int64_t* expertTokenNumsOut,
    DType* windowData, float* windowFlag, float* predBuf,
    int32_t* windowTriple, uint32_t* windowState, DType* outBuf)
{
    constexpr int slotCount = BS * K;

    // #5 Window State Init (InitWinState aligned)
    uint32_t dataState = windowState[0];
    windowState[0] = (dataState == 0) ? 1 : 0;
    windowState[1] = 1;
    windowState[2] = 0;

    // ====== Phase 1: AllToAllDispatch (pack x → window + flag) ======
    dispatch_pack<DType, BS, H, K, MoeExpertNum, TileW, WindowStride>(
        x, expertIds, windowTriple, windowData, windowFlag);

    // ====== Phase 2: CalCumSum (tile histogram + scalar cumsum) ======
    // expertCounts = 计数链直方图 ([C10], gfsim 兼容; 替代 MSCATTER_ADD ——
    // TimingSim 无 TLSU 原子族 [C12]): 每 32-slot tile TLOAD 一次, 每 bin e:
    //   TCMPS<EQ>+TSEL(物化0/1)+TROWSUM+TSTORE → 标量寄存器累加。
    //   bin 循环只覆盖 [0, E) → 非法 eid 天然不计数 (等价标量守卫)。
    //   expertCounts/cumSum 为 static bss (TMA 写可靠), 清零/导出 volatile
    //   标量 (≤4 元素 < 128B, [C9])。
    static int32_t expertCounts[MoeExpertNum];
    static int32_t cntGm[1];      // 计数链 TROWSUM 单值出口 (TSTORE→标量读回)
    {
        int32_t acc[MoeExpertNum];
        for (int e = 0; e < MoeExpertNum; e++) acc[e] = 0;
        {
            using namespace dispatch_tile;
            constexpr int kNTiles = slotCount / 32;
            for (int tb = 0; tb < kNTiles; tb++) {
                GI1x32 gIds(expertIds + tb * 32);
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
                    GI1x1 gC(cntGm);
                    TSTORE(gC, c);
                    acc[e] += cntGm[0];
                }
            }
            // 尾部 (<32 slot) 标量兜底 ([C6] 静态 valid 契约)
            for (int i = kNTiles * 32; i < slotCount; i++) {
                int32_t eid = expertIds[i];
                if (eid >= 0 && eid < MoeExpertNum) acc[eid]++;
            }
        }
        volatile int32_t *vc = expertCounts;
        for (int e = 0; e < MoeExpertNum; e++) vc[e] = acc[e];
    }
    int32_t cumSum[MoeExpertNum] = {0};
    int32_t acc = 0;
    for (int e = 0; e < MoeExpertNum; e++) {
        acc += expertCounts[e];
        cumSum[e] = acc;
        sendCountsOut[e] = cumSum[e];
        expertTokenNumsOut[e] = expertCounts[e];
    }

    // #9 CUMSUM soft sync: write cumsum flag (TEXPANDS + TSTORE)
    {
        using namespace pto;
        using tile_f = Tile<Location::Vec, float, 1, TileW, BLayout::RowMajor>;
        tile_f cumsumFlag;
        TEXPANDS(cumsumFlag, 1.0f);

        // TSUB as TMOV stand-in (see dispatch_pack note).
        tile_f sync_cf;
        TSUB(sync_cf, cumsumFlag, cumsumFlag);

        using gm_st = global_tensor<float, RowMajor<1, TileW>>;
        auto gs = reinterpret_cast<gm_st*>(windowState + 4);
        TSTORE(*gs, cumsumFlag);
    }

    // #7 CUMSUM flag check (tile pass-through; TCMP unavailable on 0828)
    check_cumsum_flag<TileW>(windowState, predBuf);

    // ====== Phase 3: LocalWindowCopy (batched rank tile 链 + slot 序输出) ======
    // expertStarts = exclusive prefix (≤E 元素 < 128B tile 下限 → 标量;
    // static bss + volatile 写 [C9])
    static int32_t expertStarts[MoeExpertNum];
    {
        volatile int32_t *vs = expertStarts;
        int32_t acc2 = 0;
        for (int e = 0; e < MoeExpertNum; e++) {
            vs[e] = acc2;
            acc2 += expertCounts[e];
        }
    }
    // batched rank: 每 32-slot tile 成对比较 rank ([C11], gfsim 兼容;
    // 替代 MGATHER_ADD —— TimingSim 无 TLSU 原子族 [C12]):
    //   同一 GM 数据双视图 TLOAD ([1×32] 行 + [32×1] 列) →
    //   TROWEXPAND/TCOLEXPAND [32×32] → TCMP<EQ> → TSEL(∧TTRI 下三角) →
    //   TROWSUM → GM 往返 → TSUBS(-1) = 块内 rank (#{j<i: eid_j==eid_i},
    //   稳定序 = 原子写指针 row-major 确定序 [C3])
    //   carry = 平 MGATHER(rankCnt, idx<<2) — 跨块进位计数器 (每块末
    //     per-bin 计数链 [C10] + 标量 volatile RMW 更新)
    //   starts = 平 MGATHER(expertStarts, idx<<2); pos = starts+carry+rank
    //   → TSTORE dstPosBuf
    //   非法 lane: index 钳 0 (双守卫 TCMPS+TSEL) → carry/starts 为桶 0
    //   查表值, rank 为垃圾 —— 输出循环按 expertIds 跳过非法 slot, 不消费。
    static int32_t rankCnt[MoeExpertNum];
    static int32_t dstPosBuf[slotCount];
    static int32_t rankBuf[32];   // 成对 rank GM 往返 scratch ([C4] 列→行)
    {
        volatile int32_t *vr = rankCnt;
        for (int e = 0; e < MoeExpertNum; e++) vr[e] = 0;
    }
    {
        using namespace dispatch_tile;
        constexpr int kNTiles = slotCount / 32;
        for (int tb = 0; tb < kNTiles; tb++) {
            GI1x32 gIds(expertIds + tb * 32);
            TI1x32 ids;
            TLOAD(ids, gIds);
            GI32x1 gIdsC(expertIds + tb * 32);
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
            TSEL(mat, eq, tri);              // (eid_i==eid_j) ∧ (j<=i)
            TI32x1 rankIncl;
            TROWSUM(rankIncl, mat);
            GI32x1 gRankW(rankBuf);
            TSTORE(gRankW, rankIncl);
            TI1x32 rankRow;
            GI1x32 gRankR(rankBuf);
            TLOAD(rankRow, gRankR);
            TSUBS(rankRow, rankRow, static_cast<int32_t>(1));  // 去对角

            // 守卫 (等价标量 if (eid >= 0 && eid < E)): 非法 lane index 钳 0
            TI1x32 neg;
            TCMPS<CmpMode::LT>(neg, ids, static_cast<int32_t>(0));
            TI1x32 oob;
            TCMPS<CmpMode::GE>(oob, ids, static_cast<int32_t>(MoeExpertNum));
            TI1x32 zero;
            TEXPANDS(zero, static_cast<int32_t>(0));
            TI1x32 idx;
            TADD(idx, ids, zero);      // 预填 dst = ids (TSEL 假分支, [C7])
            TSEL(idx, oob, zero);      // eid >= E → 0
            TSEL(idx, neg, zero);      // eid < 0  → 0

            TI1x32 carry;
            global_tensor<int32_t, RowMajor<1, MoeExpertNum>> gCnt(rankCnt);
            MGATHER(carry, gCnt, idx);
            TI1x32 starts;
            global_tensor<int32_t, RowMajor<1, MoeExpertNum>> gStarts(
                const_cast<int32_t *>(expertStarts));
            MGATHER(starts, gStarts, idx);
            TI1x32 pos;
            TADD(pos, starts, carry);
            TADD(pos, pos, rankRow);
            GI1x32 gPos(dstPosBuf + tb * 32);
            TSTORE(gPos, pos);

            // 块末进位: rankCnt[e] += #{本块 eid==e} (计数链 [C10] +
            // 标量 volatile RMW; ids 循环内重物化 —— 谓词循环无
            // loop-carried tile 契约)
            for (int e = 0; e < MoeExpertNum; e++) {
                TI1x32 ids2;
                GI1x32 gIds2(expertIds + tb * 32);
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
                GI1x1 gC(cntGm);
                TSTORE(gC, c);
                volatile int32_t *vr = rankCnt;
                vr[e] = vr[e] + cntGm[0];
            }
        }
        // 尾部 (<32 slot) 标量兜底 ([C6]); rankCnt 接续 (同 PE 程序序)
        for (int i = kNTiles * 32; i < slotCount; i++) {
            int32_t eid = expertIds[i];
            if (eid < 0 || eid >= MoeExpertNum) {
                dstPosBuf[i] = -1;
                continue;
            }
            dstPosBuf[i] = expertStarts[eid] + rankCnt[eid]++;
        }
    }

    // 输出循环 (slot 序): dstPos = expertStarts[eid] + rank 与原 (e,s) 序
    // writePos 逐 slot 一致 (mt 版已证等价); copy-out/flag 链原有 tile 保留
    for (int i = 0; i < slotCount; i++) {
        int32_t eid = expertIds[i];
        if (eid < 0 || eid >= MoeExpertNum) continue;
        int32_t dstPos = dstPosBuf[i];

        // #4 Flag check (tile pass-through; TCMP unavailable on 0828)
        // Scalar readback skipped — self-loopback data always ready
        check_flag<BS, K, TileW>(windowFlag, predBuf, i);

        // Read data from window → expandXOut (tile copy, 512B stride)
        dispatch_copy_out<DType, BS, H, K, TileW, WindowStride>(
            windowData, expandXOut, i, dstPos);

        // Read triple + scale (scalar, 12B/4B < 128B tile 粒度)
        expandIdxOut[dstPos * 3 + 0] = windowTriple[i * 3 + 0];
        expandIdxOut[dstPos * 3 + 1] = windowTriple[i * 3 + 1];
        expandIdxOut[dstPos * 3 + 2] = windowTriple[i * 3 + 2];
        expandScalesOut[dstPos] = expertScales[i];

        // #1 Clear flag (TEXPANDS(0.0) + TSTORE)
        clear_flag<BS, K, TileW>(windowFlag, i);
    }

    // Window state writeback
    windowState[4] = BS;
}

} // namespace supernpu::tile_isa
#endif
