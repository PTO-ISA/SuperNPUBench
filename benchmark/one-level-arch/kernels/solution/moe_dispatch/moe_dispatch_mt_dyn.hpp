#ifndef SUPERNPU_MOE_DISPATCH_MT_DYN_HPP
#define SUPERNPU_MOE_DISPATCH_MT_DYN_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>

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
//   · 每个 TLOAD 结果经 tile op (TSUB sync stand-in) 消费, 经 TSTORE 出 tile 域;
//   · TCMP 不可用, flag check 保持 tile pass-through;
//   · 跨 PE 交接由 mtBarrier 保护 (相位 1/2/3 与静态版一致);
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

// ====== Phase 2: CalCumSum (per-PE histogram + cross-PE reduce) ======
// 语义与静态版 cal_cumsum_mt 一致; expert 区间同样用运行时 ceil 分片。
static inline void cal_cumsum_mt_dyn(int32_t* expertIds, int32_t* sendCountsOut,
                                     int64_t* expertTokenNumsOut,
                                     int32_t* cntLocal, int32_t* expertStarts,
                                     int64_t bs, int64_t k, int64_t moeExpertNum)
{
    const int64_t slotCount = bs * k;
    const int tid = static_cast<int>(get_thread_idx());
    const int64_t seg = slotCount / kMtThreadsPerBlock;
    const int64_t rem = slotCount % kMtThreadsPerBlock;
    const int64_t begin = tid * seg + (tid < rem ? tid : rem);
    const int64_t len = seg + (tid < rem ? 1 : 0);

    // Per-PE local histogram over this PE's own slots (scalar GM reads)
    int32_t* myCnt = cntLocal + tid * moeExpertNum;
    for (int64_t e = 0; e < moeExpertNum; e++) {
        myCnt[e] = 0;
    }
    for (int64_t i = begin; i < begin + len; i++) {
        int32_t eid = expertIds[i];
        if (eid >= 0 && eid < moeExpertNum) myCnt[eid]++;
    }

    // Reduce: each PE writes its assigned expert range (needs every PE's
    // cntLocal — caller guarantees the phase-1 barrier has completed)
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

// ====== Flag check (tile pass-through; TCMP unavailable on the 0828
// toolchain — see moe_dispatch_v2.hpp note. predBuf is never read back) ======
template <int TileW>
void check_flag_mt_dyn(float* windowFlag, float* predBuf, int64_t srcSlot,
                       int64_t slotCount)
{
    using namespace pto;
    using gm_flag = global_tensor<float, RowMajor<-1, -1>>;
    using gm_pred = global_tensor<float, RowMajor<-1, -1>>;
    using tile_f  = Tile<Location::Vec, float, 1, TileW, BLayout::RowMajor>;

    tile_f flagTile;
    gm_flag gf(windowFlag + srcSlot * TileW,
               static_cast<int>(slotCount), TileW);
    TLOAD(flagTile, gf);

    // Pipeline sync (SyncFunc<MTE2_V> aligned)
    tile_f sync_f1;
    TSUB(sync_f1, flagTile, flagTile);

    gm_pred gp(predBuf + srcSlot * TileW,
               static_cast<int>(slotCount), TileW);
    TSTORE(gp, flagTile);
}

// ====== CUMSUM flag check (tile pass-through, PE0 only) ======
template <int TileW>
void check_cumsum_flag_mt_dyn(uint32_t* windowState, float* predBuf)
{
    using namespace pto;
    using gm_st  = global_tensor<float, RowMajor<1, TileW>>;
    using gm_pred = global_tensor<float, RowMajor<1, TileW>>;
    using tile_f = Tile<Location::Vec, float, 1, TileW, BLayout::RowMajor>;
    using it_pred = global_iterator<gm_pred, tile_f>;

    // 同 write_cumsum_flag_dyn：正确构造，避免 reinterpret_cast 的 NULL 基址。
    gm_st gs(reinterpret_cast<float*>(windowState + 4));
    it_pred pred_iter(predBuf);

    tile_f readFlag;
    TLOAD(readFlag, gs);

    // Pipeline sync (SyncFunc<MTE2_V> aligned)
    tile_f sync_f1;
    TSUB(sync_f1, readFlag, readFlag);

    auto gp = pred_iter(0, 0);
    TSTORE(gp, readFlag);
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
//   barrier(2): expertStarts + PE0 的 cumsum flag 在阶段 3 前可见
//   barrier(3): 全部输出写完后任何 PE 才离开 kernel
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
    mtBarrier(1);

    // ====== Phase 2: CalCumSum (count + cumsum) ======
    cal_cumsum_mt_dyn(expertIds, sendCountsOut, expertTokenNumsOut,
                      cntLocal, expertStarts, bs, k, moeExpertNum);

    // CUMSUM soft sync + flag check — PE0 only (same PE writes then reads)
    if (tid == 0) {
        write_cumsum_flag_dyn<TileW>(windowState);
        check_cumsum_flag_mt_dyn<TileW>(windowState, predBuf);
    }
    mtBarrier(2);

    // ====== Phase 3: LocalWindowCopy (read window → continuous output) ======
    // 每 PE 把自己的 slot 段按 expert-major 位置重发 (与静态版语义一致):
    //   dstPos = expertStarts[eid] + slot 在同 expert 组内的序号
    const int64_t seg = slotCount / kMtThreadsPerBlock;
    const int64_t rem = slotCount % kMtThreadsPerBlock;
    const int64_t begin = tid * seg + (tid < rem ? tid : rem);
    const int64_t len = seg + (tid < rem ? 1 : 0);
    for (int64_t i = begin; i < begin + len; i++) {
        int32_t eid = expertIds[i];
        if (eid < 0 || eid >= moeExpertNum) continue;

        int32_t rank = 0;
        for (int64_t j = 0; j < i; j++) {
            if (expertIds[j] == eid) rank++;
        }
        int64_t dstPos = static_cast<int64_t>(expertStarts[eid]) + rank;

        // Flag check (tile pass-through; TCMP unavailable on 0828)
        check_flag_mt_dyn<TileW>(windowFlag, predBuf, i, slotCount);

        // Read data from window → expandXOut (tile copy, 512B stride)
        dispatch_copy_out_mt_dyn<DType, TileW, WindowStride>(
            windowData, expandXOut, i, dstPos, slotCount, h);

        // Read triple + scale (scalar)
        expandIdxOut[dstPos * 3 + 0] = windowTriple[i * 3 + 0];
        expandIdxOut[dstPos * 3 + 1] = windowTriple[i * 3 + 1];
        expandIdxOut[dstPos * 3 + 2] = windowTriple[i * 3 + 2];
        expandScalesOut[dstPos] = expertScales[i];

        // Clear flag (TEXPANDS(0.0) + TSTORE)
        clear_flag_mt_dyn<TileW>(windowFlag, i, slotCount);
    }
    mtBarrier(3);

    // Window state writeback — PE0 only
    if (tid == 0) {
        windowState[4] = static_cast<uint32_t>(bs);
    }
}

} // namespace supernpu::tile_isa
#endif
