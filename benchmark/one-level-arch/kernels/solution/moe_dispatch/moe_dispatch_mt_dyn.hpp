#ifndef SUPERNPU_MOE_DISPATCH_MT_DYN_HPP
#define SUPERNPU_MOE_DISPATCH_MT_DYN_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "solution/moe_dispatch/dispatch_tile_common.hpp"

namespace supernpu::tile_isa {

// ============================================================================
// MoE Dispatch — Multi-thread 4-PE SPMD, runtime DYNAMIC SHAPE variant
//
// 基于 moe_dispatch_mt.hpp 的动态 shape 版: 算子语义与阶段划分完全一致,
// 唯一区别是 BS/H/K/MoeExpertNum 编译期不可知, 运行期由 tiling 指针传入,
// 全部切分参数运行期计算。
//
//   tiling = {BS, H, K, MoeExpertNum}   (int64)
//
// 设计对照 quant/dynamic_mx_quant_*_dyn (动态入口范式):
//   · physical tile 形状 (1×TileW) 仍编译期锁定 —— 寄存器分配约束;
//     列维有效宽度保持编译期 (TEPL B.DIM 立即数要求), 故保留调用契约
//     H % TileW == 0 (违例由各 PE 同值提前返回, 不触达栅栏, 无死锁)。
//   · global_iterator 依赖编译期 RowStride, 全部改为运行时构造的
//     global_tensor<RowMajor<-1,-1>> (ctor 传运行时 rows/cols), 基址按
//     运行时维度手动计算。
//   · PE 分片统一运行时 ceil 公式 (静态版的 slotsPerPE/expertsPerPE):
//       seg = n / 4;  rem = n % 4;
//       begin = tid*seg + min(tid, rem);  len = seg + (tid < rem ? 1 : 0);
//     前 rem 个 PE 各多承担 1 项, slotCount/MoeExpertNum 非 4 整除天然覆盖。
//   · 静态版 static_assert(slotCount%4==0 / MoeExpertNum%4==0) 删除,
//     由上述 ceil 分片吸收; tid >= 4 的冗余 PE 直接返回。
//
// 其余与静态版相同的约定 (详见 moe_dispatch_mt.hpp):
//   · 每个 TLOAD 结果经 tile op 消费, 经 TSTORE/MSCATTER 出 tile 域;
//   · Phase2 直方图 / Phase3 rank = 32-slot 分块 tile 链 (hist_segment_dyn
//     / rank_dstpos_segment_dyn: 计数链 [C10] + 成对比较 rank [C11],
//     gfsim 兼容面板 [C12]; 替代 MSCATTER_ADD/MGATHER_ADD —— TimingSim
//     无 TLSU 原子族完成路径), 任意运行时段长; 契约违例
//     (slotCount > kSlotCapDyn / expertNum > kExpertCapDyn) 标量兜底;
//   · TCMP flag 谓词不可用 (0.58.4 B.DATR 语法被 0828 asm matcher 拒绝),
//     flag check 保持 tile pass-through;
//   · 跨 PE 交接由 mtBarrier 保护 (相位 1/2/3/4, 直方图后新增 barrier(2)
//     结构化保证 cross-PE reduce 读到完整 cntLocal);
//   · cntLocal 需 4 * MoeExpertNum 个 int32, expertStarts 需 MoeExpertNum
//     个 int32 的 GM scratch。
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

// ====== Phase 1: Pack (PE tid owns slots [begin, begin+len)) ======
template <typename DType, int TileW, int WindowStride>
void dispatch_pack_mt_dyn(DType* x, int32_t* expertIds, int32_t* windowTriple,
                          DType* windowData, float* windowFlag,
                          int64_t bs, int64_t h, int64_t k)
{
    const int tid = static_cast<int>(get_thread_idx());
    const int64_t slotCount = bs * k;
    const int64_t hTiles = h / TileW;   // H % TileW == 0 契约 (入口已校验)
    const int64_t seg = slotCount / kMtThreadsPerBlock;
    const int64_t rem = slotCount % kMtThreadsPerBlock;
    const int64_t begin = tid * seg + (tid < rem ? tid : rem);
    const int64_t len = seg + (tid < rem ? 1 : 0);
    using namespace pto;

    using gm_x    = global_tensor<DType, RowMajor<-1, -1>>;
    using gm_w    = global_tensor<DType, RowMajor<-1, -1>>;
    using gm_flag = global_tensor<float,   RowMajor<-1, -1>>;
    using tile_d  = Tile<Location::Vec, DType, 1, TileW, BLayout::RowMajor>;
    using tile_f  = Tile<Location::Vec, float,  1, TileW, BLayout::RowMajor>;

    for (int64_t tk = begin; tk < begin + len; tk++) {
        int64_t tokenId = tk / k;
        int64_t topkId  = tk % k;

        for (int64_t t = 0; t < hTiles; t++) {
            tile_d xq;
            gm_x gx(x + tokenId * h + t * TileW,
                    static_cast<int>(bs), static_cast<int>(h));
            TLOAD(xq, gx);

            // #3 Pipeline sync (SyncFunc<MTE2_V> aligned); TSUB as TMOV
            // stand-in (see moe_dispatch_v2.hpp note).
            tile_d sync_d;
            TSUB(sync_d, xq, xq);

            // 512B stride write: data at [tk*WindowStride + t*TileW]
            gm_w gw(windowData + tk * WindowStride + t * TileW,
                    static_cast<int>(slotCount), WindowStride);
            TSTORE(gw, xq);
        }

        // 512B block packing: flag fill (TEXPANDS + TSTORE)
        tile_f flagTile;
        TEXPANDS(flagTile, 1.0f);

        // Pipeline sync (SyncFunc<V_MTE3> aligned); TSUB as TMOV stand-in.
        tile_f sync_f;
        TSUB(sync_f, flagTile, flagTile);

        gm_flag gf(windowFlag + tk * TileW,
                   static_cast<int>(slotCount), TileW);
        TSTORE(gf, flagTile);

        // FillTriple (scalar, 12B — too small for tile)
        windowTriple[tk * 3 + 0] = 0;
        windowTriple[tk * 3 + 1] = static_cast<int32_t>(tokenId);
        windowTriple[tk * 3 + 2] = static_cast<int32_t>(topkId);
    }
}

// ====== dyn 分块 tile 链 helper (任意运行时段长, 32-slot 分块) ======
// 全链 gfsim 兼容 ([C12]; 替代 MSCATTER_ADD/MGATHER_ADD —— TimingSim 无
// TLSU 原子族完成路径):
//   直方图 = 计数链 [C10]: lane 守卫 (TCI ramp vs chunkLen) 一次性把越段
//     lane TSEL 到 -1 哨兵 (bin 循环 [0,E) 永不匹配 -1, 谓词 tile 不得
//     loop-carried → 哨兵物化在 bin 循环外), 每 bin: TCMPS<EQ>+TSEL+
//     TROWSUM+TSTORE → 标量 volatile RMW 累加 cnt[e]。
//   rank = 成对比较 [C11]: 同一 GM 数据双视图 TLOAD ([1×32] 行 + [32×1]
//     列) → TROWEXPAND/TCOLEXPAND [32×32] → TCMP<EQ> → TSEL(∧TTRI) →
//     TROWSUM → rankIncl-1 = 块内 #{j<i}; 全局 rank = 块内 rank +
//     平 MGATHER(rankCnt carry); 块末 per-bin 计数链进位 (volatile RMW)。
// 末块 TLOAD/TSTORE 按静态 valid 1×32 读写 (动态 valid 链泄漏 lane [C6]),
// 越段读取落在 driver max-shape 缓冲的对齐裕量内 (mapped, 值被守卫屏蔽);
// dstPosBuf 为 per-PE 私有并带 32 槽过写裕量, 越段写不触及其他 PE。
constexpr int64_t kSlotCapDyn = 128;    // dstPosBuf 容量契约 (driver max 32, 4× 裕量)
constexpr int64_t kExpertCapDyn = 16;   // rankCnt 容量契约 (driver max 5)

// 直方图段: cnt[eid] += 1, eid ∈ [begin, begin+len) (计数链 [C10])
static inline void hist_segment_dyn(const int32_t* expertIds, int64_t begin,
                                     int64_t len, int32_t* cnt,
                                     int32_t moeExpertNum)
{
    using namespace dispatch_tile;
    const int tid = static_cast<int>(get_thread_idx());
    static int32_t cntGm[kMtThreadsPerBlock];   // per-PE 单值出口 (写不相交)
    int32_t* mySum = cntGm + tid;
    for (int64_t c = 0; c < len; c += 32) {
        const int64_t chunkLen = (len - c < 32) ? (len - c) : 32;
        GI1x32 gIds(const_cast<int32_t*>(expertIds) + begin + c);
        TI1x32 ids;
        TLOAD(ids, gIds);
        // lane 守卫一次性合成哨兵: 越段 lane → -1 (bin 循环 [0,E) 不匹配)
        TI1x32 lane;
        TCI(lane, static_cast<int32_t>(0));
        TI1x32 over;
        TCMPS<CmpMode::GE>(over, lane, static_cast<int32_t>(chunkLen));
        TI1x32 negOne;
        TEXPANDS(negOne, static_cast<int32_t>(-1));
        TI1x32 zero;
        TEXPANDS(zero, static_cast<int32_t>(0));
        TI1x32 idsM;
        TADD(idsM, ids, zero);         // 预填 dst = ids (TSEL 假分支, [C7])
        TSEL(idsM, over, negOne);      // lane >= chunkLen → -1
        for (int32_t e = 0; e < moeExpertNum; e++) {
            TI1x32 pred;
            TCMPS<CmpMode::EQ>(pred, idsM, e);
            TI1x32 one;
            TEXPANDS(one, static_cast<int32_t>(1));
            TI1x32 sel;
            TEXPANDS(sel, static_cast<int32_t>(0));
            TSEL(sel, pred, one);
            TSum1I s;
            TROWSUM(s, sel);
            GI1x1 gC(mySum);
            TSTORE(gC, s);
            volatile int32_t* vc = cnt;
            vc[e] = vc[e] + cntGm[tid];
        }
    }
}

// rank+dstPos 段: dstPosBufLocal[slot] = expertStarts[eid] + 全局 rank
// (成对比较 rank [C11] + 平 MGATHER carry; rankCnt 预置 crossPePrefix,
// 每块末计数链进位 → carry = 跨 PE 前缀 + 段内前块计数 = 全局前缀)
static inline void rank_dstpos_segment_dyn(const int32_t* expertIds,
                                            int64_t begin, int64_t len,
                                            int32_t* rankCnt,
                                            const int32_t* expertStarts,
                                            int32_t* dstPosBufLocal,
                                            int32_t moeExpertNum)
{
    using namespace dispatch_tile;
    const int tid = static_cast<int>(get_thread_idx());
    static int32_t rankBuf[kMtThreadsPerBlock][32];  // 成对 rank GM 往返 ([C4])
    static int32_t cntGm[kMtThreadsPerBlock];        // 进位计数链单值出口
    for (int64_t c = 0; c < len; c += 32) {
        const int64_t chunkLen = (len - c < 32) ? (len - c) : 32;
        GI1x32 gIds(const_cast<int32_t*>(expertIds) + begin + c);
        TI1x32 ids;
        TLOAD(ids, gIds);
        GI32x1 gIdsC(const_cast<int32_t*>(expertIds) + begin + c);
        TI32x1 idsCol;
        TLOAD(idsCol, gIdsC);

        // 成对比较 rank ([C11]): rankIncl[i] = 1 + #{j<=i : eid_j==eid_i}
        // (越段 lane 的垃圾列被 TTRI 掩出有效行前缀; 垃圾行 rank 落入
        //  dstPosBufLocal 过写裕量, 输出循环不消费)
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

        // lane/eid 三重守卫 → index 钳 0 (MGATHER 地址恒界内)
        TI1x32 lane;
        TCI(lane, static_cast<int32_t>(0));
        TI1x32 over;
        TCMPS<CmpMode::GE>(over, lane, static_cast<int32_t>(chunkLen));
        TI1x32 neg;
        TCMPS<CmpMode::LT>(neg, ids, static_cast<int32_t>(0));
        TI1x32 oob;
        TCMPS<CmpMode::GE>(oob, ids, moeExpertNum);
        TI1x32 zero;
        TEXPANDS(zero, static_cast<int32_t>(0));
        TI1x32 idx;
        TADD(idx, ids, zero);
        TSEL(idx, neg, zero);
        TSEL(idx, oob, zero);
        TSEL(idx, over, zero);
        TI1x32 idxB;
        TSHLS(idxB, idx, 2u);

        TI1x32 carry;
        global_tensor<int32_t, RowMajor<-1, -1>> gCnt(
            rankCnt, 1, moeExpertNum);
        MGATHER(carry, gCnt, idxB);
        TI1x32 starts;
        global_tensor<int32_t, RowMajor<-1, -1>> gStarts(
            const_cast<int32_t*>(expertStarts), 1, moeExpertNum);
        MGATHER(starts, gStarts, idxB);
        TI1x32 pos;
        TADD(pos, starts, carry);
        TADD(pos, pos, rankRow);
        GI1x32 gPos(dstPosBufLocal + c);
        TSTORE(gPos, pos);

        // 块末进位: rankCnt[e] += #{本块 eid==e} (计数链 [C10] + 哨兵守卫)
        TI1x32 negOne;
        TEXPANDS(negOne, static_cast<int32_t>(-1));
        TI1x32 idsM;
        TADD(idsM, ids, zero);
        TSEL(idsM, over, negOne);        // 越段 lane → -1 (bin 循环不匹配)
        for (int32_t e = 0; e < moeExpertNum; e++) {
            TI1x32 pred;
            TCMPS<CmpMode::EQ>(pred, idsM, e);
            TI1x32 one;
            TEXPANDS(one, static_cast<int32_t>(1));
            TI1x32 sel;
            TEXPANDS(sel, static_cast<int32_t>(0));
            TSEL(sel, pred, one);
            TSum1I s;
            TROWSUM(s, sel);
            GI1x1 gC(cntGm + tid);
            TSTORE(gC, s);
            volatile int32_t* vr = rankCnt;
            vr[e] = vr[e] + cntGm[tid];
        }
    }
}

// ====== Phase 2: CalCumSum (per-PE histogram + cross-PE reduce) ======
// 语义与静态版 cal_cumsum_mt 一致; expert 区间同样用运行时 ceil 分片。
// histBarrierPhase = 直方图完成栅栏的单调相位 (调用方传入, 见主入口
// sInvCnt 注 —— 跨 cfg 调用陈旧 flag 天然失效)。
static inline void cal_cumsum_mt_dyn(int32_t* expertIds, int32_t* sendCountsOut,
                                     int64_t* expertTokenNumsOut,
                                     int32_t* cntLocal, int32_t* expertStarts,
                                     int64_t bs, int64_t k, int64_t moeExpertNum,
                                     uint32_t histBarrierPhase)
{
    const int64_t slotCount = bs * k;
    const int tid = static_cast<int>(get_thread_idx());
    const int64_t seg = slotCount / kMtThreadsPerBlock;
    const int64_t rem = slotCount % kMtThreadsPerBlock;
    const int64_t begin = tid * seg + (tid < rem ? tid : rem);
    const int64_t len = seg + (tid < rem ? 1 : 0);

    // Per-PE local histogram over this PE's own slots — 计数链 [C10]
    // (hist_segment_dyn: 32-slot 分块 + lane 哨兵守卫 + 每 bin
    // TCMPS<EQ>+TSEL+TROWSUM+TSTORE+volatile RMW);
    // 清零 ≤ kExpertCapDyn 元素 volatile 标量 ([C9])
    int32_t* myCnt = cntLocal + tid * moeExpertNum;
    {
        volatile int32_t* vc = myCnt;
        for (int64_t e = 0; e < moeExpertNum; e++) vc[e] = 0;
    }
    hist_segment_dyn(expertIds, begin, len, myCnt, moeExpertNum);
    // 全部 PE 的直方图完成后才可 cross-PE reduce (读 cntLocal 全行;
    // 原实现无此栅栏, 依赖确定性调度侥幸 —— 现结构化保证, gtv mt 同款)
    mtBarrier(histBarrierPhase);

    // Reduce: each PE writes its assigned expert range
    const int64_t eseg = moeExpertNum / kMtThreadsPerBlock;
    const int64_t erem = moeExpertNum % kMtThreadsPerBlock;
    const int64_t ebegin = tid * eseg + (tid < erem ? tid : erem);
    const int64_t elen = eseg + (tid < erem ? 1 : 0);
    for (int64_t e = ebegin; e < ebegin + elen; e++) {
        int32_t sum = 0;
        int32_t before = 0;
        for (int64_t e2 = 0; e2 <= e; e2++) {
            for (int t2 = 0; t2 < kMtThreadsPerBlock; t2++) {
                int32_t c = cntLocal[t2 * moeExpertNum + e2];
                if (e2 == e) sum += c; else before += c;
            }
        }
        expertTokenNumsOut[e] = sum;
        sendCountsOut[e] = before + sum;   // inclusive cumsum (v2 semantics)
        expertStarts[e] = before;          // exclusive start for Phase 3
    }
}

// ====== CUMSUM flag write (PE0 only; TEXPANDS + TSTORE at windowState+4) ======
template <int TileW>
void write_cumsum_flag_dyn(uint32_t* windowState)
{
    using namespace pto;
    using tile_f = Tile<Location::Vec, float, 1, TileW, BLayout::RowMajor>;
    tile_f cumsumFlag;
    TEXPANDS(cumsumFlag, 1.0f);

    // TSUB as TMOV stand-in (see moe_dispatch_v2.hpp note).
    tile_f sync_cf;
    TSUB(sync_cf, cumsumFlag, cumsumFlag);

    using gm_st = global_tensor<float, RowMajor<1, TileW>>;
    // 构造而非 reinterpret_cast：后者把零初始化的 windowState+4 当
    // global_tensor 对象解引用，取到 NULL 基址（TSTORE GMBase=0x0）。
    gm_st gs(reinterpret_cast<float*>(windowState + 4));
    TSTORE(gs, cumsumFlag);
}

// ====== Flag check — 真 EQ 谓词 tile 链 (fp32 TCMPS<EQ> 0923 基线探针
// 实证可用, 旧 pass-through 退化删除; 详见 moe_dispatch_v2.hpp 注) ======
template <int TileW>
void check_flag_mt_dyn(float* windowFlag, float* predBuf, int64_t srcSlot,
                       int64_t slotCount)
{
    using namespace pto;
    using gm_flag = global_tensor<float, RowMajor<-1, -1>>;
    using gm_pred = global_tensor<float, RowMajor<-1, -1>>;
    using tile_f  = Tile<Location::Vec, float,  1, TileW, BLayout::RowMajor>;
    using tile_i  = Tile<Location::Vec, int32_t, 1, TileW, BLayout::RowMajor>;

    tile_f flagTile;
    gm_flag gf(windowFlag + srcSlot * TileW,
               static_cast<int>(slotCount), TileW);
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

    gm_pred gp(predBuf + srcSlot * TileW,
               static_cast<int>(slotCount), TileW);
    TSTORE(gp, norm);
}

// ====== CUMSUM flag check — 真 EQ 谓词 tile 链 (PE0 only) ======
template <int TileW>
void check_cumsum_flag_mt_dyn(uint32_t* windowState, float* predBuf)
{
    using namespace pto;
    using gm_st  = global_tensor<float, RowMajor<1, TileW>>;
    using gm_pred = global_tensor<float, RowMajor<1, TileW>>;
    using tile_f = Tile<Location::Vec, float,  1, TileW, BLayout::RowMajor>;
    using tile_i = Tile<Location::Vec, int32_t, 1, TileW, BLayout::RowMajor>;
    using it_pred = global_iterator<gm_pred, tile_f>;

    // 同 write_cumsum_flag_dyn：正确构造，避免 reinterpret_cast 的 NULL 基址。
    gm_st gs(reinterpret_cast<float*>(windowState + 4));
    it_pred pred_iter(predBuf);

    tile_f readFlag;
    TLOAD(readFlag, gs);

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
template <int TileW>
void clear_flag_mt_dyn(float* windowFlag, int64_t srcSlot, int64_t slotCount)
{
    using namespace pto;
    using gm_flag = global_tensor<float, RowMajor<-1, -1>>;
    using tile_f  = Tile<Location::Vec, float, 1, TileW, BLayout::RowMajor>;

    tile_f zeroFlag;
    TEXPANDS(zeroFlag, 0.0f);

    // TSUB as TMOV stand-in (see moe_dispatch_v2.hpp note).
    tile_f sync_zf;
    TSUB(sync_zf, zeroFlag, zeroFlag);

    gm_flag gf(windowFlag + srcSlot * TileW,
               static_cast<int>(slotCount), TileW);
    TSTORE(gf, zeroFlag);
}

// ====== Phase 3: Read data from window → expandXOut (PE-disjoint rows) ======
template <typename DType, int TileW, int WindowStride>
void dispatch_copy_out_mt_dyn(DType* windowData, DType* expandXOut,
                              int64_t srcSlot, int64_t dstPos,
                              int64_t slotCount, int64_t h)
{
    const int64_t hTiles = h / TileW;   // H % TileW == 0 契约
    using namespace pto;

    using gm_w   = global_tensor<DType, RowMajor<-1, -1>>;
    using gm_out = global_tensor<DType, RowMajor<-1, -1>>;
    using tile_d = Tile<Location::Vec, DType, 1, TileW, BLayout::RowMajor>;

    for (int64_t t = 0; t < hTiles; t++) {
        tile_d xq;
        gm_w gw(windowData + srcSlot * WindowStride + t * TileW,
                static_cast<int>(slotCount), WindowStride);
        TLOAD(xq, gw);

        // Pipeline sync (SyncFunc<MTE2_V> aligned)
        tile_d sync_d;
        TSUB(sync_d, xq, xq);

        gm_out gout(expandXOut + dstPos * h + t * TileW,
                    static_cast<int>(slotCount), static_cast<int>(h));
        TSTORE(gout, xq);
    }
}

// ====== Main entry (multi-PE SPMD; called by every PE) ======
// 跨 PE 交接与静态版一致:
//   barrier(1): 所有 PE 的 pack 写 (windowData/flag/triple) 在直方图归约与
//               阶段 3 读取前可见
//   barrier(2): 所有 PE 的 cntLocal 直方图行写完后才可 cross-PE reduce
//               (cal_cumsum_mt_dyn 内部)
//   barrier(3): expertStarts + PE0 的 cumsum flag 在阶段 3 前可见
//   barrier(4): 全部输出写完后任何 PE 才离开 kernel
// tiling = {BS, H, K, MoeExpertNum}; 调用契约 H % TileW == 0。
// 注: tiling 违例由各 PE 同值判定并提前返回, 不触达任何栅栏, 无死锁。
template <typename DType, int TileW = 128, int WindowStride = 256>
void moe_dispatch_mt_dyn(
    DType* x, int32_t* expertIds, float* expertScales,
    DType* expandXOut, int32_t* expandIdxOut, float* expandScalesOut,
    int32_t* sendCountsOut, int64_t* expertTokenNumsOut,
    DType* windowData, float* windowFlag, float* predBuf,
    int32_t* windowTriple, uint32_t* windowState, DType* outBuf,
    int32_t* cntLocal, int32_t* expertStarts,
    const int64_t* tiling)
{
    const int64_t bs = tiling[0];
    const int64_t h = tiling[1];
    const int64_t k = tiling[2];
    const int64_t moeExpertNum = tiling[3];
    const int64_t slotCount = bs * k;
    const int tid = static_cast<int>(get_thread_idx());

    // 运行时契约: 全部 PE 读同一 tiling → 同值判定 → 同进同出, 无死锁
    if (tid >= kMtThreadsPerBlock) return;
    if (bs <= 0 || h <= 0 || k <= 0 || moeExpertNum <= 0) return;
    if (h % TileW != 0) return;   // 列维 tile 宽度 (列 valid 必须编译期)

    // 单调相位编号 (gtv mt_dyn 同款 gfsim 活锁修复): driver 曾在两次 cfg
    // 调用之间清零 sMtPhaseDone —— 那是无同步的跨 PE 写, 若某 PE 的清零
    // 晚于另一 PE 对下一轮 barrier 的置位 (PE0 独占验证段造成进度偏差),
    // flag 被抹掉 → 双方永久互等自旋。改为 per-PE 私用调用计数 sInvCnt
    // (每 PE 只读写自己的槽, 零跨 PE 写), 相位 = inv*8 + k 跨调用单调
    // 递增 → 陈旧 flag 天然失效, driver 复位彻底删除。契约同值判定保证
    // 各 PE 的 inv 序列一致。
    // 步长 8 而非 4: driver 在 kernel 前有输入发布屏障 (8c+1)、两次 cfg
    // 之间/末端有验证汇合屏障 (8c+6) —— 若 kernel 相位与 driver 相位重合
    // (gtv dyn 现状: driver barrier(4) == cfgB 首个 kernel 相位 4), 后者被
    // 陈旧 flag 立即通过而失去同步 (gfrun 靠 lockstep 侥幸, gfsim 时序漂移
    // 下成真竞争)。
    static uint32_t sInvCnt[kMtThreadsPerBlock];   // bss 零初始化 = 首轮 inv 0
    const uint32_t inv = sInvCnt[tid];
    sInvCnt[tid] = inv + 1u;
    const uint32_t ph = inv * 8u;

    // Window State Init (InitWinState aligned) — PE0 only
    if (tid == 0) {
        uint32_t dataState = windowState[0];
        windowState[0] = (dataState == 0) ? 1 : 0;
        windowState[1] = 1;
        windowState[2] = 0;
    }

    // ====== Phase 1: AllToAllDispatch (pack x → window + flag) ======
    dispatch_pack_mt_dyn<DType, TileW, WindowStride>(
        x, expertIds, windowTriple, windowData, windowFlag, bs, h, k);
    // [2026-10-09 优化] 相位整体 +1: driver 在 kernel 前新增输入发布屏障
    // (8c+1, genInputs 已按 PE 分片), kernel 内部相位 = inv*8 + 2..5。
    mtBarrier(ph + 2u);

    // ====== Phase 2: CalCumsum (count + cumsum) ======
    cal_cumsum_mt_dyn(expertIds, sendCountsOut, expertTokenNumsOut,
                      cntLocal, expertStarts, bs, k, moeExpertNum,
                      ph + 3u);

    // CUMSUM soft sync + flag check — PE0 only (same PE writes then reads)
    if (tid == 0) {
        write_cumsum_flag_dyn<TileW>(windowState);
        check_cumsum_flag_mt_dyn<TileW>(windowState, predBuf);
    }
    mtBarrier(ph + 4u);

    // ====== Phase 3: LocalWindowCopy (read window → continuous output) ======
    // 每 PE 把自己的 slot 段按 expert-major 位置重发 (与静态版语义一致):
    //   dstPos = expertStarts[eid] + slot 在同 expert 组内的全局序号
    // rank 批量 tile 化 (rank_dstpos_segment_dyn, 成对比较 rank [C11]):
    // per-PE rankCnt 预置 crossPePrefix[eid] = Σ_{t<tid} cntLocal[t][eid]
    // (PE 段为连续区间, 段前同 expert 计数恰为前序 PE 的 cntLocal 之和)
    // → 全局 rank = 块内 #{j<i} (TTRI 掩蔽成对比较) + 平 MGATHER(rankCnt)
    // carry, 块末计数链进位 (与原 "for j<i" 标量扫描逐 slot 一致)。
    // 契约违例 (slotCount/expertNum 超 kernel 容量) → 标量扫描兜底。
    const int64_t seg = slotCount / kMtThreadsPerBlock;
    const int64_t rem = slotCount % kMtThreadsPerBlock;
    const int64_t begin = tid * seg + (tid < rem ? tid : rem);
    const int64_t len = seg + (tid < rem ? 1 : 0);
    static int32_t rankCnt[kMtThreadsPerBlock][kExpertCapDyn];
    static int32_t dstPosBuf[kMtThreadsPerBlock][kSlotCapDyn + 32];
    const bool rankTileOk =
        (slotCount <= kSlotCapDyn) && (moeExpertNum <= kExpertCapDyn);
    if (rankTileOk) {
        volatile int32_t* vr = rankCnt[tid];
        for (int64_t e = 0; e < moeExpertNum; e++) {
            int32_t cross = 0;
            for (int t = 0; t < tid; t++) {
                cross += cntLocal[t * moeExpertNum + e];
            }
            vr[e] = cross;
        }
        rank_dstpos_segment_dyn(expertIds, begin, len, rankCnt[tid],
                                 expertStarts, dstPosBuf[tid], moeExpertNum);
    }
    for (int64_t i = begin; i < begin + len; i++) {
        int32_t eid = expertIds[i];
        if (eid < 0 || eid >= moeExpertNum) continue;

        int64_t dstPos;
        if (rankTileOk) {
            dstPos = dstPosBuf[tid][i - begin];
        } else {
            int32_t rank = 0;
            for (int64_t j = 0; j < i; j++) {
                if (expertIds[j] == eid) rank++;
            }
            dstPos = static_cast<int64_t>(expertStarts[eid]) + rank;
        }

        // Flag check (tile pass-through; TCMP unavailable on 0828)
        check_flag_mt_dyn<TileW>(windowFlag, predBuf, i, slotCount);

        // Read data from window → expandXOut (tile copy, 512B stride)
        dispatch_copy_out_mt_dyn<DType, TileW, WindowStride>(
            windowData, expandXOut, i, dstPos, slotCount, h);

        // Read triple + scale (scalar, 12B/4B < 128B tile 粒度)
        expandIdxOut[dstPos * 3 + 0] = windowTriple[i * 3 + 0];
        expandIdxOut[dstPos * 3 + 1] = windowTriple[i * 3 + 1];
        expandIdxOut[dstPos * 3 + 2] = windowTriple[i * 3 + 2];
        expandScalesOut[dstPos] = expertScales[i];

        // Clear flag (TEXPANDS(0.0) + TSTORE)
        clear_flag_mt_dyn<TileW>(windowFlag, i, slotCount);
    }
    mtBarrier(ph + 5u);

    // Window state writeback — PE0 only
    if (tid == 0) {
        windowState[4] = static_cast<uint32_t>(bs);
    }
}

} // namespace supernpu::tile_isa

// ============================================================================
// [2026-10-06 修复注] gfsim `--conf fourpe` 下本算子末端 exit lockstep 断言
// (@ cycle 45,448) 的根因与修复在驱动层 (test/solution/moe_dispatch/src/
// mega.../moe_dispatch_mt_dyn.cpp): 原实现 worker (tid!=0) 在 c-loop 末端
// 直接 `return 0` 先行 park 到退出 lockstep AND 组, PE0 独占 cfgB 验证后
// 迟到 —— gfsim 时序模型的退出 ecall 按 lockstep AND 组汇聚, 迟到者
// t0 永远未 join (st=0, 全簇无 retired 进展) → T_deadlock=9999 后
// SyscallBarrier.cpp:1889 断言截断。修复: cfgB 验证后补齐验证汇合屏障
// mtBarrier(14) (相位 14 = 8*1+6, 与本 kernel 的 sInvCnt 单调相位方案
// 10..13 严格衔接), 全 PE 一起放行到 _end。kernel 本身无需修改; 详见同目录
// moe_dispatch_mt_dyn_gfsim_fix_report.md。修复后 gfsim fourpe:
// 完整 PASS (Total Cycles = 36,798, 3 次运行确定性一致, exit_parks=4),
// gfrun 4 线程 R2=0。
// ============================================================================
#endif
