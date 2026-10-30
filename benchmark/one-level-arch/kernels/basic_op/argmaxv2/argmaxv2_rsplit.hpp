#pragma once

// ============================================================================
// ArgMaxV2 切 R 模板（对标 CANN arc35 GROUP_REDUCE / tilingKey 30001）
//
// 适用（路由见 argmaxv2.hpp 尾轴分支）：「R 大 A 小」——行分区喂不满行
// tile（ceil(A/PeCount)*4 < tM）且 R 足够大（kAxisDim >= PeCount*tM）。
//
// 流程（4 PE，全 tile 域——TLSU 装载/存储方向与 tile 寄存器溢出载体
// 的形态约束决定了实现形态：(a) TLSU store → LSU 标量 load 方向存在
// 结构性陈旧（L1D 无失效机制）；(b) TLSU store 后立即 TLOAD 同行存在
// 竞态挂死（距离要够远）；(c) tile 寄存器压力超限触发后端溢出断言）：
//
//   1. R 切分：seg = align16_down(ceil(R/4))，PE0..2 各取 [i*seg, (i+1)*seg)，
//      PE3 取 [3*seg, R)。内部分界 16 对齐 ⇒ 内段 S%16==0，主视图无尾块。
//   2. 段内 reshape（核心增益：单行 1/tM 利用率 → ~98% 满载块）：每条
//      原始归约线的连续 S 元素重排为 [S/16, 16] 视图（宽恒 16）+
//      可选 [1, S%16] 尾档（仅 PE3）；argmaxv2_row_segmerge_dynamic 逐
//      [tM,16] 行块归约并就地折叠出行号 → winner (kmax, 全局 idx)[1,1]
//      落盘 stage 区。
//   3. mt_barrier（驱逐屏障）。
//   4. 屏障后低压区统一合并（全 PE 冗余，watchdog 友好；批量形态）：
//      W-big 以 [kOuter,kPad] 行 tile 整批装载 4 PE stage（valid
//      (kOuter,kMB)）→ PE 升序严格 GT 运行合并（同位平局保先 PE——
//      段连续 ⇒ 先 PE 的 idx 全体更小）+ winner 链全行并行 → key 平局
//      取最小 idx = 全局首现（min-idx 语义与位序无关）→ PE0 TCVT
//      (i64 出口)/直存 + TSTORE。
//   5. mt_barrier 末端汇合。屏障相位按每 PE 调用计数推进（多次调用
//      不回卷）。
//
// 正确性要点：stage 区 per-PE 64B 独占行（跨 PE 无 snoop 下同行并发
// tile 写丢更新）；未写满至 kMB 的尾块以
// (kMinKey, 0xFFFFFFFF) 哨兵补齐（哨兵 key 仅在全域最小值平局时出现，
// 此时真实 winner 的 idx < 0xFFFFFFFF 恒胜）。
//
// 语义（与静态版完全一致）：平局首个最大值索引；NaN 视为最大值（首个
// NaN）；±0.0 相等——全部由 sortable key 域继承。
// ============================================================================

#include "basic_op/argmaxv2/detail/argmaxv2_dynamic.hpp"
#include "basic_op/utils/mt_barrier.hpp"

#include <cstddef>
#include <cstdint>

namespace supernpu {
namespace multi_thread {

namespace rsplit_detail {
constexpr std::size_t pad64(std::size_t b) { return (b + 63u) / 64u * 64u; }
}  // namespace rsplit_detail

// ----------------------------------------------------------------------------
// argmaxv2_rsplit：切 R 模板入口（每 PE 调用；shape 为编译期常量）。
//   argmaxv2_rsplit<DType, IdType, kOuter, kAxisDim, PeCount>(input, output)
// 可选 kTileM/kTileN（>0 时覆盖 traits 缺省块形，=0 取缺省）：行块/列块
// 物理尺寸调参用——几何（W/尾档/kMaxBlocks/ramp）全部随之 constexpr 推
// 导，路由契约断言仍按 traits 缺省块形校验（路由决策与块形无关）。
//   约束（detail/argmaxv2_dynamic.hpp segmerge static_assert）：块
// tM*tN*sizeof(KeyT) <= 2048B；行态 tile >= 128B ⇒ fp32/i32 tM >= 32、
// fp16/bf16 tM >= 64。
// 路由前置（由 argmaxv2.hpp 保证）：尾轴行归约、kOuter <= PeCount*tM/4、
// kAxisDim >= PeCount*tM（蕴含内段非空且 3*seg < kAxisDim）。
// ----------------------------------------------------------------------------
template <typename DType, typename IdType, int kOuter, int kAxisDim,
          std::uint32_t PeCount = kDefaultPeCount, int kTileM = 0,
          int kTileN = 0>
void argmaxv2_rsplit(DType *input, IdType *output) {
    using Traits = argmaxv2_detail::ArgMaxTraits<DType>;
    using KeyT = typename Traits::KeyT;
    // 块形：kTileM/kTileN > 0 覆盖 traits 缺省（调参），=0 取缺省。
    constexpr int kTM = kTileM > 0 ? kTileM : Traits::kRowTileM;
    constexpr int kTN = kTileN > 0 ? kTileN : Traits::kRowTileN;
    constexpr int64_t kPC = static_cast<int64_t>(PeCount);

    static_assert(!std::is_same_v<DType, int64_t>,
                  "int64 input is not supported: the tile-op layer has no "
                  "64-bit reduce/compare instructions (TROWARGMAX/TCOLARGMAX "
                  "contracts and the model reduce/compare dtype whitelist "
                  "cap at 32-bit)");
    static_assert(std::is_same_v<IdType, int32_t> ||
                      std::is_same_v<IdType, int64_t>,
                  "output dtype must be int32_t or int64_t");
    static_assert(kOuter > 0 && kAxisDim > 0, "shape must be positive");
    static_assert(PeCount >= 2 && PeCount <= 4,
                  "rsplit barrier flag slots cap at 4 PEs");
    // 路由契约按 traits 缺省块形校验（argmaxv2.hpp 的路由决策依据）。
    static_assert((kOuter + static_cast<int>(PeCount) - 1) /
                          static_cast<int>(PeCount) * 4 <
                      Traits::kRowTileM,
                  "routing contract violated: rows-per-PE must underfill the "
                  "row tile by 4x");
    static_assert(kAxisDim >= static_cast<int>(PeCount) * Traits::kRowTileM,
                  "routing contract violated: axis too short to split");

    // ---- 编译期切分几何（W-大视图，见文件头流程 2）----------------------
    constexpr int64_t kSeg =
        ((kAxisDim + kPC - 1) / kPC) & ~static_cast<int64_t>(15);
    static_assert(kSeg >= 16, "interior segment must be 16-aligned nonzero");
    static_assert((kPC - 1) * kSeg < kAxisDim,
                  "interior segments must not cover the whole axis");
    constexpr int64_t kSLast = kAxisDim - (kPC - 1) * kSeg;
    constexpr int64_t kSMax = kSeg > kSLast ? kSeg : kSLast;
    constexpr int64_t kWMax =
        ((kSMax + kTM - 1) / kTM + 15) & ~static_cast<int64_t>(15);
    // 每行 stage 单元数上界（W-big）：主视图 1（nFull <= tM 恰一行块）+
    // 尾档 [r/16,16] 的行块数 (kWMax/16)/tM + 尾档 [1,r%16] 1 + 余量 1。
    constexpr int64_t kMaxBlocks =
        1 + (kWMax / 16 + kTM - 1) / kTM + 2;

    // ---- 主循环形态择优（块数上界 A/B 定界）------------------------------
    // A 大且段短（W-big 的 nFull 填不满行块）时改行并行直减：原始行作
    // tile 行（跨步 kAxisDim）、段作列块串，[kOuter,1] 寄存器运行态——
    // 无 reshape/winner 链/哨兵/ramp，stage 仅两个 [kOuter,1] 列存
    // （规避 W-big 的行定价小存储）。
    // 择优依据 = 两形态的物理块数上界（同 2048B 预算）：
    //   W-big: kOuter * ceil(kWMax/kTN)（逐行 reshape）
    //   rowpar: ceil(kSMax/16) * ceil(kOuter/traits-tM)（整段列块串）
    constexpr int kTMt = Traits::kRowTileM;   // rowpar 块形（traits 缺省）
    constexpr int64_t kWBigBlocks =
        kOuter * ((kWMax + kTN - 1) / kTN);
    constexpr int64_t kRowparBlocks =
        ((kSMax + 15) / 16) *
        ((kOuter + kTMt - 1) / kTMt);
    constexpr bool kUseRowpar =
        kOuter >= kTMt / 2 && kRowparBlocks <= kWBigBlocks;
    // 每 PE stage 单元数：rowpar 恒 1（块维寄存器内消耗）。
    constexpr int64_t kMB = kUseRowpar ? 1 : kMaxBlocks;

    // stage 行宽（cell 数）：>= kMB 且 [kOuter,kPad] 行 tile >= 128B
    //（key/idx 两种张量），4 对齐。
    constexpr int kMinCell =
        sizeof(KeyT) < 4 ? static_cast<int>(sizeof(KeyT)) : 4;
    constexpr int kPadLo =
        (128 + kOuter * kMinCell - 1) / (kOuter * kMinCell);
    constexpr int kPad =
        ((kMB > kPadLo ? static_cast<int64_t>(kMB) : kPadLo) + 3) & ~3;
    // 哨兵整行铺的静态行宽（tile 模板参须 int）：行宽即 kPad。
    constexpr int kPadC = kPad;
    static_assert(kOuter * kPad * static_cast<int>(sizeof(KeyT)) <= 2048 &&
                      kOuter * kPad * 4 <= 2048,
                  "stage row tile budget exceeded (axis too long)");

    // ---- BSS workspace --------------------------------------------------
    // stage 区布局 [PeCount][kOuter][kPad]（行距 kPad ⇒ 屏障后按 PE 整批
    // 取 [kOuter,kPad] 行 tile）。per-PE 行 64B 独占（TLSU 写丢更新防护）。
    constexpr std::size_t kStKeyRowBytes = rsplit_detail::pad64(
        static_cast<std::size_t>(kOuter) * kPad * sizeof(KeyT));
    constexpr std::size_t kStIdxRowBytes = rsplit_detail::pad64(
        static_cast<std::size_t>(kOuter) * kPad * sizeof(std::uint32_t));
    alignas(64) static unsigned char s_st_key[PeCount][kStKeyRowBytes];
    alignas(64) static unsigned char s_st_idx[PeCount][kStIdxRowBytes];
    // ramp 交叉暂存（[1,N] TSTORE → [N,1] TLOAD 往返）。仅当存在多行块
    // 视图（gIM > tM，长轴 tailA——idx 折叠需要 ramp[ib..ib+vr)）时加长
    // 到覆盖全行号；否则保持 [1,tM] 单块。
    constexpr bool kRampMulti = (kWMax / 16) > kTM;
    constexpr int kRampLen =
        kRampMulti
            ? (((static_cast<int>(kWMax) / 16 + kTM + 63) & ~63) + 64)
            : kTM;
    static std::uint32_t s_ramp[PeCount][kRampLen];
    // 屏障相位：每 PE 独立调用计数（< phase 比较，回卷会让先行 PE 提前
    // 通过；共享计数器跨调用交错有竞态，按 PE 分槽——SPMD 序列一致）。
    static std::uint32_t s_call_seq[PeCount];

    const std::uint32_t tid = get_thread_idx();
    if (tid >= PeCount) {
        return;
    }
    const int64_t tid64 = static_cast<int64_t>(tid);
    const std::uint32_t phase1 = 2u * s_call_seq[tid] + 1u;
    const std::uint32_t phase2 = phase1 + 1u;
    ++s_call_seq[tid];

    // ---- 本 PE 段几何（对 tid 常量折叠） --------------------------------
    const int64_t segStart = tid64 * kSeg;
    const int64_t S = (tid < PeCount - 1) ? kSeg : kSLast;

    KeyT *stK =
        reinterpret_cast<KeyT *>(s_st_key[tid]);
    std::uint32_t *stI =
        reinterpret_cast<std::uint32_t *>(s_st_idx[tid]);

    if constexpr (kUseRowpar) {
        // ---- 行并行直减：整段列块串 + [kOuter,1] 运行态 + 两列存 -----
        ::argmaxv2_rowpar_reduce<DType, kTMt, Traits::kRowTileN>(
            input, kOuter, kAxisDim, segStart, S, stK, stI, kPad);
    } else {
    // ---- W-big reshape（A 小：单行 reshape 填满行块）--------------------
    const int64_t W = ((S + kTM - 1) / kTM + 15) & ~static_cast<int64_t>(15);
    const int64_t nFull = S / W;
    const int64_t rTail = S - nFull * W;
    const int64_t tailRows = rTail >> 4;
    const int64_t tailB = rTail & 15;

    // 基础 ramp（0..tM-1）：入口一次构建（TCI [1,tM] → [C4] GM 往返），
    // 全部行/视图共用——与首次消费间隔整段行循环，最大化 TLSU
    // store→load 距离（gfsim tile-store 通路缺口规避）。
    if constexpr (kRampMulti) {
        // TCI 块形契约限制 [1,64]：分块生成（起始值递增）。
        using tileRampRow =
            Tile<Location::Vec, std::uint32_t, 1, 64, BLayout::RowMajor>;
        for (int rc = 0; rc < kRampLen; rc += 64) {
            tileRampRow rr;
            TCI(rr, static_cast<std::uint32_t>(rc));
            global_tensor<std::uint32_t, RowMajor<-1, -1>> gr(
                s_ramp[tid] + rc, 1, 64);
            TSTORE(gr, rr);
        }
    } else {
        using tileRampRow =
            Tile<Location::Vec, std::uint32_t, 1, kTM, BLayout::RowMajor>;
        tileRampRow rr;
        TCI(rr, 0u);
        global_tensor<std::uint32_t, RowMajor<-1, -1>> gr(s_ramp[tid], 1,
                                                          kTM);
        TSTORE(gr, rr);
    }

    for (int64_t a = 0; a < kOuter; ++a) {
         DType *base = input + a * kAxisDim + segStart;
         KeyT *kDest = stK + a * kPad;
         std::uint32_t *iDest = stI + a * kPad;
         int64_t cell = 0;

        // 哨兵整行一次预铺（store 攒批）：[1,kPad] 静态行 tile（ramp
        // 同款合法形态）单次 TEXPANDS+TSTORE 覆盖全行，视图 winner
        // [1,1] 随后覆写 cell 0..cellCnt-1；取代逐 cell 哨兵 [1,1]×
        // (kMB-cellCnt)×2，减少小存储事务与同 cacheline 写对（WAW）。
        // cell kMB..kPad-1 超出合并 valid 范围，写之无害。
        {
            global_tensor<KeyT, RowMajor<-1, -1>> gks(kDest, 1,
                                                     kPadC);
            Tile<Location::Vec, KeyT, 1, kPadC, BLayout::RowMajor> sks;
            TEXPANDS(sks, static_cast<KeyT>(Traits::kMinKey));
            TSTORE(gks, sks);
            global_tensor<std::uint32_t, RowMajor<-1, -1>> gis(
                iDest, 1, kPadC);
            Tile<Location::Vec, std::uint32_t, 1, kPadC, BLayout::RowMajor>
                sis;
            TEXPANDS(sis, 0xFFFFFFFFu);
            TSTORE(gis, sis);
        }

        // 主视图 [nFull, W]：nFull <= tM 恰一行块，行利用率 nFull/tM。
        if (nFull > 0) {
            ::argmaxv2_row_segmerge_dynamic<DType, kTM, kTN>(
                base, s_ramp[tid], nFull, W, kTM, kTN,
                static_cast<std::uint32_t>(segStart), kDest, iDest);
            cell = 1;
        }
        // 尾档 A [r/16, 16]：多行块时按块落盘（cells 1..）。
        if (tailRows > 0) {
            ::argmaxv2_row_segmerge_dynamic<DType, kTM, kTN>(
                base + nFull * W, s_ramp[tid], tailRows, 16, kTM, kTN,
                static_cast<std::uint32_t>(segStart + nFull * W),
                kDest + cell, iDest + cell);
            cell += (tailRows + kTM - 1) / kTM;
        }
        // 尾档 B [1, r%16]（仅末段 16 不整除时）。
        if (tailB > 0) {
            ::argmaxv2_row_segmerge_dynamic<DType, kTM, kTN>(
                base + nFull * W + tailRows * 16, s_ramp[tid], 1, tailB,
                kTM, kTN,
                static_cast<std::uint32_t>(segStart + nFull * W +
                                           tailRows * 16),
                kDest + cell, iDest + cell);
            ++cell;
        }
    }
    }  // W-big 分支结束

    mt_sync::mt_barrier(phase1, tid, PeCount);

    // ---- 屏障后统一合并（全 PE 冗余执行：消除停车歪斜，watchdog 友好）-
    // 批量形式：整批取 stage 替代逐行取——规避驱逐后 [1,kMB] 小装载的
    // L2 延迟在少行数下无摊销。两形态按 kMB 分岔：
    //   kMB > 1（W-big）：[kOuter,kPad] 行 tile（valid (kOuter,kMB)），
    //     2*PeCount 次装载 + 4-PE GT 运行合并 + winner 链全行并行
    //     （TROWMAX/TROWEXPAND/TROWMIN 的 [A,1]<-[A,N] 形态）。
    //   kMB == 1（rowpar）：[kTMt,1] 列 tile（物理列 1，segmerge 同款），
    //     张量行距 kPad 读 cell 0；PE 升序严格 GT 定胜，无块内链。
    // tile 形态约束（两参运行期构造的维度须与物理形状兼容）：
    //   (a) [kOuter,kPad] 的 validCol=1 形态触发 compare/select 维度
    //       断言（physicalCol < validCol 编码），rowpar 须用 [kTMt,1]
    //       列 tile 规避；
    //   (b) [tM,1] 窄列 tile 须单参构造——两参 (rows,1) 触发后端 [2,16]
    //       打包载体 → TCMP 源断言；
    //   (c) [kMaxBlocks,1] 列式合并链触发 L2 侧 TLOAD dst 丢失断言，
    //       合并须用 [1,N]/[A,N] 行式 TROWMAX 链。
    using tileKeyOneM =
        Tile<Location::Vec, KeyT, kTMt, 1, BLayout::RowMajor, -1, 1>;
    using tileIdxOneM =
        Tile<Location::Vec, std::uint32_t, kTMt, 1, BLayout::RowMajor, -1,
             1>;
    const size_t kAS = static_cast<size_t>(kOuter);

    if constexpr (kMB > 1) {
        using tileKeyRowM =
            Tile<Location::Vec, KeyT, kOuter, kPad, BLayout::RowMajor, -1,
                 -1>;
        using tileIdxRowM =
            Tile<Location::Vec, std::uint32_t, kOuter, kPad,
                 BLayout::RowMajor, -1, -1>;
        using tilePredRowM = tileKeyRowM;
        const size_t kMS = static_cast<size_t>(kMB);
        tileKeyRowM curK(kAS, kMS), newK(kAS, kMS);
        tileIdxRowM curI(kAS, kMS), newI(kAS, kMS);
        tilePredRowM pd(kAS, kMS);
        tileKeyOneM kmax(kAS);
        tileIdxOneM idxAt(kAS);

        for (std::uint32_t pe = 0; pe < PeCount; ++pe) {
            global_tensor<KeyT, RowMajor<-1, -1>> gk(
                reinterpret_cast<KeyT *>(s_st_key[pe]), kOuter, kPad);
            global_tensor<std::uint32_t, RowMajor<-1, -1>> gi(
                reinterpret_cast<std::uint32_t *>(s_st_idx[pe]), kOuter,
                kPad);
            if (pe == 0) {
                TLOAD(curK, gk);
                TLOAD(curI, gi);
            } else {
                TLOAD(newK, gk);
                TLOAD(newI, gi);
                TCMP<CmpMode::GT>(pd, newK, curK);
                TSEL(curK, pd, newK);
                TSEL(curI, pd, newI);
            }
        }

        // 全局 winner（行并行）：key 平局取最小 idx = 行内块序首现。
        TROWMAX(kmax, curK);
        TROWEXPAND(newK, kmax);
        TCMP<CmpMode::EQ>(pd, curK, newK);
        TEXPANDS(newI, 0xFFFFFFFFu);
        TSEL(newI, pd, curI);
        TROWMIN(idxAt, newI);

        if (tid == 0) {
            // 批量输出 [kOuter,1]（行距 sizeof(IdType)）。
            if constexpr (std::is_same_v<IdType, int32_t>) {
                global_tensor<std::uint32_t, RowMajor<-1, -1>> go(
                    reinterpret_cast<std::uint32_t *>(output), kOuter, 1);
                TSTORE(go, idxAt);
            } else {
                Tile<Location::Vec, int64_t, kTMt, 1, BLayout::RowMajor, -1,
                     1>
                    outI64(kAS);
                TCVT(outI64, idxAt);
                global_tensor<int64_t, RowMajor<-1, -1>> go(
                    reinterpret_cast<int64_t *>(output), kOuter, 1);
                TSTORE(go, outI64);
            }
        }
    } else {
        // rowpar 单 cell 合并：列 tile + kPad 行距张量。
        tileKeyOneM curK(kAS), newK(kAS), pd(kAS);
        tileIdxOneM curI(kAS), newI(kAS);
        for (std::uint32_t pe = 0; pe < PeCount; ++pe) {
            global_tensor<KeyT, RowMajor<-1, -1>> gk(
                reinterpret_cast<KeyT *>(s_st_key[pe]), kOuter, kPad);
            global_tensor<std::uint32_t, RowMajor<-1, -1>> gi(
                reinterpret_cast<std::uint32_t *>(s_st_idx[pe]), kOuter,
                kPad);
            if (pe == 0) {
                TLOAD(curK, gk);
                TLOAD(curI, gi);
            } else {
                TLOAD(newK, gk);
                TLOAD(newI, gi);
                TCMP<CmpMode::GT>(pd, newK, curK);
                TSEL(curK, pd, newK);
                TSEL(curI, pd, newI);
            }
        }

        // curI 即每行 winner idx（PE 升序严格 GT，段连续 ⇒ 先 PE idx 小）。
        if (tid == 0) {
            if constexpr (std::is_same_v<IdType, int32_t>) {
                global_tensor<std::uint32_t, RowMajor<-1, -1>> go(
                    reinterpret_cast<std::uint32_t *>(output), kOuter, 1);
                TSTORE(go, curI);
            } else {
                Tile<Location::Vec, int64_t, kTMt, 1, BLayout::RowMajor, -1,
                     1>
                    outI64(kAS);
                TCVT(outI64, curI);
                global_tensor<int64_t, RowMajor<-1, -1>> go(
                    reinterpret_cast<int64_t *>(output), kOuter, 1);
                TSTORE(go, outI64);
            }
        }
    }

    mt_sync::mt_barrier(phase2, tid, PeCount);
}

}  // namespace multi_thread
}  // namespace supernpu
