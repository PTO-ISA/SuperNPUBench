#ifndef SUPERNPU_MOE_COMBINE_MT_DYN_HPP
#define SUPERNPU_MOE_COMBINE_MT_DYN_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace supernpu::tile_isa {

// ============================================================================
// MoE Combine — Multi-thread 4-PE SPMD, runtime DYNAMIC SHAPE variant
//
// 基于 moe_combine_mt.hpp 的动态 shape 版: 算子语义与阶段划分完全一致,
// 唯一区别是 BS/H/K/NumExpanded 编译期不可知, 运行期由 tiling 指针传入,
// 全部切分参数运行期计算。
//
//   tiling = {BS, H, K, NumExpanded}   (int64)
//
// 设计对照 quant/dynamic_mx_quant_*_dyn (动态入口范式):
//   · physical tile 形状 (1×TileW) 仍编译期锁定 —— 寄存器分配约束;
//     列维有效宽度保持编译期 (TEPL B.DIM 立即数要求), 故保留调用契约
//     H % TileW == 0 (违例由各 PE 同值提前返回, 不触达栅栏, 无死锁)。
//   · global_iterator 依赖编译期 RowStride, 全部改为运行时构造的
//     global_tensor<RowMajor<-1,-1>> (ctor 传运行时 rows/cols), 基址按
//     运行时维度手动计算。
//   · PE 分片统一运行时 ceil 公式 (静态版的 rowsPerPE/tokensPerPE):
//       seg = n / 4;  rem = n % 4;
//       begin = tid*seg + min(tid, rem);  len = seg + (tid < rem ? 1 : 0);
//     前 rem 个 PE 各多承担 1 行, NumExpanded/BS 非 4 整除天然覆盖。
//   · 静态版 static_assert(NumExpanded%4==0 / BS%4==0) 删除,
//     由上述 ceil 分片吸收; tid >= 4 的冗余 PE 直接返回。
//
// 其余与静态版相同的约定 (详见 moe_combine_mt.hpp):
//   · 每个 TLOAD 结果经 tile op (TSUB sync stand-in) 消费, 经 TSTORE 出 tile 域;
//   · TCMP 不可用, flag check 保持 tile pass-through;
//   · 跨 PE 交接由 combineMtBarrier 保护 (相位 1/2 与静态版一致)。
// ============================================================================

constexpr int kCombineMtThreads = 4;

// Multi-PE barrier: volatile per-PE phase flags + compiler memory barrier,
// same convention as kernels/solution/group_token_vec/group_token_vec_mt.hpp.
static volatile uint32_t sCombineMtPhaseDone[kCombineMtThreads];

static inline void combineMtCompilerBarrier()
{
    __asm__ volatile("" : : : "memory");
}

static inline void combineMtBarrier(uint32_t phase)
{
    combineMtCompilerBarrier();
    sCombineMtPhaseDone[get_thread_idx()] = phase;
    combineMtCompilerBarrier();
    for (int t = 0; t < kCombineMtThreads; ++t) {
        while (sCombineMtPhaseDone[t] < phase) {
        }
    }
    combineMtCompilerBarrier();
}

// ====== Phase 1: Pack (PE tid owns expanded rows [begin, begin+len)) ======
template <typename DType, int TileW>
void combine_pack_mt_dyn(DType* expandX, int32_t* expandIdx,
                         DType* windowData, float* windowFlag,
                         int64_t h, int64_t k, int64_t numExpanded)
{
    const int tid = static_cast<int>(get_thread_idx());
    const int64_t kTiles = h / TileW;   // H % TileW == 0 契约 (入口已校验)
    const int64_t seg = numExpanded / kCombineMtThreads;
    const int64_t rem = numExpanded % kCombineMtThreads;
    const int64_t begin = tid * seg + (tid < rem ? tid : rem);
    const int64_t len = seg + (tid < rem ? 1 : 0);
    using namespace pto;

    using gm_x    = global_tensor<DType, RowMajor<-1, -1>>;
    using gm_win  = global_tensor<DType, RowMajor<-1, -1>>;
    using gm_flag = global_tensor<float, RowMajor<-1, -1>>;
    using tile_d  = Tile<Location::Vec, DType, 1, TileW, BLayout::RowMajor>;
    using tile_f  = Tile<Location::Vec, float, 1, TileW, BLayout::RowMajor>;

    for (int64_t tk = begin; tk < begin + len; tk++) {
        int64_t tokenId = expandIdx[tk * 3 + 1];
        int64_t topkId  = expandIdx[tk * 3 + 2];
        int64_t slot    = tokenId * k + topkId;

        for (int64_t t = 0; t < kTiles; t++) {
            tile_d xq;
            gm_x gx(expandX + tk * h + t * TileW,
                    static_cast<int>(numExpanded), static_cast<int>(h));
            TLOAD(xq, gx);

            // #3 Pipeline sync (SyncFunc<MTE2_V> aligned)
            // TSUB(dst, src, src) is a compile-time stand-in for TMOV(dst, src):
            // the 0828 toolchain's asm matcher rejects TMOV's 0.58.4 B.DATR
            // syntax ("NORM, DTYPE_NONE, Zero"); TSUB emits no B.DATR and keeps
            // the same src->dst dependency edge (dst = src - src = 0).
            tile_d sync_d;
            TSUB(sync_d, xq, xq);

            gm_win gw(windowData + slot * h + t * TileW,
                      static_cast<int>(numExpanded), static_cast<int>(h));
            TSTORE(gw, xq);

            // #1 Flag fill: data ready marker (TEXPANDS + TSTORE)
            tile_f flagTile;
            TEXPANDS(flagTile, 1.0f);

            gm_flag gf(windowFlag + slot * TileW,
                       static_cast<int>(numExpanded), TileW);
            TSTORE(gf, flagTile);
        }
    }
}

// ====== #4 Flag check (tile pass-through; TCMP's 0.58.4 B.DATR syntax is
// rejected by the 0828 toolchain asm matcher, so the EQ predicate collapses
// to a flag->predBuf copy. predBuf consumers treat non-1.0f as "not ready";
// the flag is 1.0f after pack, so pass-through preserves the wait semantics.
// TSUB stands in for the removed TMOV sync, see combine_pack_mt_dyn note) ======
template <int TileW>
void combine_check_flag_mt_dyn(float* windowFlag, float* predBuf,
                               int64_t slot, int64_t t, int64_t numExpanded)
{
    using namespace pto;
    using gm_flag = global_tensor<float, RowMajor<-1, -1>>;
    using gm_pred = global_tensor<float, RowMajor<-1, -1>>;
    using tile_f  = Tile<Location::Vec, float, 1, TileW, BLayout::RowMajor>;

    tile_f flagTile;
    gm_flag gf(windowFlag + slot * TileW,
               static_cast<int>(numExpanded), TileW);
    TLOAD(flagTile, gf);

    // #3 Pipeline sync (SyncFunc<MTE2_V> aligned)
    tile_f sync_f1;
    TSUB(sync_f1, flagTile, flagTile);

    gm_pred gp(predBuf + slot * TileW,
               static_cast<int>(numExpanded), TileW);
    TSTORE(gp, flagTile);
}

// ====== #1 Clear flag ======
template <int TileW>
void combine_clear_flag_mt_dyn(float* windowFlag, int64_t slot,
                               int64_t numExpanded)
{
    using namespace pto;
    using gm_flag = global_tensor<float, RowMajor<-1, -1>>;
    using tile_f  = Tile<Location::Vec, float, 1, TileW, BLayout::RowMajor>;

    tile_f zeroFlag;
    TEXPANDS(zeroFlag, 0.0f);

    // TSUB as TMOV stand-in (see combine_pack_mt_dyn note).
    tile_f sync_zf;
    TSUB(sync_zf, zeroFlag, zeroFlag);

    gm_flag gf(windowFlag + slot * TileW,
               static_cast<int>(numExpanded), TileW);
    TSTORE(gf, zeroFlag);
}

// ====== Phase 2: Reduce (PE tid owns tokens [begin, begin+len)) ======
// 读本 PE token 的 K 个 window slot (由任意 PE 在阶段 1 写入 — 调用方保证
// 阶段 1 栅栏完成), fp32 累加 scale 加权行, 写 PE 不相交 out 行, 清 flag。
template <typename DTypeIn, typename DTypeOut, int TileW>
void combine_reduce_mt_dyn(float* expertScales, DTypeIn* windowData,
                           float* windowFlag, float* predBuf, DTypeOut* out,
                           int64_t bs, int64_t h, int64_t k, int64_t numExpanded)
{
    const int tid = static_cast<int>(get_thread_idx());
    const int64_t kTiles = h / TileW;   // H % TileW == 0 契约
    const int64_t seg = bs / kCombineMtThreads;
    const int64_t rem = bs % kCombineMtThreads;
    const int64_t begin = tid * seg + (tid < rem ? tid : rem);
    const int64_t len = seg + (tid < rem ? 1 : 0);
    using namespace pto;

    using gm_win  = global_tensor<DTypeIn,  RowMajor<-1, -1>>;
    using gm_out  = global_tensor<DTypeOut, RowMajor<-1, -1>>;
    using tile_d  = Tile<Location::Vec, DTypeIn,  1, TileW, BLayout::RowMajor>;
    using tile_f  = Tile<Location::Vec, float,    1, TileW, BLayout::RowMajor>;
    using tile_o  = Tile<Location::Vec, DTypeOut, 1, TileW, BLayout::RowMajor>;

    for (int64_t n = begin; n < begin + len; n++) {
        for (int64_t t = 0; t < kTiles; t++) {
            // #4 Flag check (tile pass-through; TCMP unavailable on 0828)
            // Scalar readback skipped — barrier-covered data always ready
            for (int64_t kk = 0; kk < k; kk++) {
                combine_check_flag_mt_dyn<TileW>(windowFlag, predBuf,
                                                 n * k + kk, t, numExpanded);
            }

            // Flag wait (scalar readback of predBuf)
            for (int64_t kk = 0; kk < k; kk++) {
                int64_t slot = n * k + kk;
                if (predBuf[slot * TileW] < 0.5f) break;
            }

            tile_f acc;
            TEXPANDS(acc, 0.0f);

            for (int64_t kk = 0; kk < k; kk++) {
                int64_t slot = n * k + kk;
                float scale = expertScales[n * k + kk];

                tile_d xq;
                gm_win gw(windowData + slot * h + t * TileW,
                          static_cast<int>(numExpanded), static_cast<int>(h));
                TLOAD(xq, gw);

                // #3 Pipeline sync (SyncFunc<MTE2_V> aligned)
                // TSUB as TMOV stand-in (see combine_pack_mt_dyn note).
                tile_d sync_d;
                TSUB(sync_d, xq, xq);

                tile_f xf;
                TCVT(xf, xq);
                TMULS(xf, xf, scale);
                TADD(acc, acc, xf);
            }

            tile_o oq;
            TCVT(oq, acc);
            gm_out gout(out + n * h + t * TileW,
                        static_cast<int>(bs), static_cast<int>(h));
            TSTORE(gout, oq);
        }

        // #1 Clear flag (TEXPANDS(0.0) + TSTORE)
        for (int64_t kk = 0; kk < k; kk++) {
            combine_clear_flag_mt_dyn<TileW>(windowFlag, n * k + kk,
                                             numExpanded);
        }
    }
}

// ====== Main entry (multi-PE SPMD; called by every PE) ======
// 跨 PE 交接与静态版一致:
//   barrier(1): 所有 PE 的 pack 写 (windowData/windowFlag) 在任意 PE 的
//               reduce 读取其他 PE slot 前可见
//   barrier(2): 所有 out 行写完且 flag 清空后任何 PE 才离开
// tiling = {BS, H, K, NumExpanded}; 调用契约 H % TileW == 0。
// 注: tiling 违例由各 PE 同值判定并提前返回, 不触达任何栅栏, 无死锁。
template <typename DTypeIn, typename DTypeOut, int TileW = 128>
void moe_combine_mt_dyn(DTypeIn* expandX, float* expertScales,
                        int32_t* expandIdx,
                        DTypeIn* windowData, float* windowFlag,
                        uint32_t* windowState, float* predBuf,
                        DTypeOut* out,
                        const int64_t* tiling)
{
    const int64_t bs = tiling[0];
    const int64_t h = tiling[1];
    const int64_t k = tiling[2];
    const int64_t numExpanded = tiling[3];
    const int tid = static_cast<int>(get_thread_idx());

    // 运行时契约: 全部 PE 读同一 tiling → 同值判定 → 同进同出, 无死锁
    if (tid >= kCombineMtThreads) return;
    if (bs <= 0 || h <= 0 || k <= 0 || numExpanded <= 0) return;
    if (h % TileW != 0) return;   // 列维 tile 宽度 (列 valid 必须编译期)

    // #5 Window State Init (InitWinState aligned) — PE0 only
    if (tid == 0) {
        uint32_t dataState = windowState[0];
        windowState[0] = (dataState == 0) ? 1 : 0;
        windowState[1] = 1;
        windowState[2] = 0;
    }

    // ====== Phase 1: Pack (expandX → window + flag) ======
    combine_pack_mt_dyn<DTypeIn, TileW>(
        expandX, expandIdx, windowData, windowFlag, h, k, numExpanded);
    combineMtBarrier(1);

    // ====== Phase 2: Reduce (window → weighted sum → out) ======
    combine_reduce_mt_dyn<DTypeIn, DTypeOut, TileW>(
        expertScales, windowData, windowFlag, predBuf, out,
        bs, h, k, numExpanded);
    combineMtBarrier(2);

    // Window state writeback — PE0 only
    if (tid == 0) {
        windowState[4] = static_cast<uint32_t>(bs);
    }
}

} // namespace supernpu::tile_isa
#endif
