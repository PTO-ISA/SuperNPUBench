// ============================================================================
// ArgMaxV2 算子 — PTO 一层编程模型（单 PE 动态 shape 基础实现）
//
// 静态版（detail/argmaxv2.hpp）的运行期 tiling 对应实现，风格对齐
// solution/normalization 的 static/dynamic 拆分（rms_norm / group_norm_grad）：
//
//   - 静态版：shape / tile 全部为编译期模板参数，尾块用 4 套静态 Valid
//     载体（tileData / _rN / _rM / _rMN）+ global_iterator 静态寻址；
//   - 动态版：gIM/gIN/tile_m/tile_n 为运行期参数，Tile 模板 Valid = -1
//     （动态载体），构造时传入运行期 valid 形状；GM 寻址用
//     global_tensor<T, RowMajor<-1,-1>>(ptr, rows, cols) 逐块构造
//     （normalization 动态版同风格）；尾行/尾列/尾角统一收敛为
//     min(remaining, tile) 的运行期 valid，不再需要静态尾块载体。
//
// 归约算法与静态版逐指令一致（结果 bit-exact 对齐）：
//   [fp16/fp32/bf16] sortable key 位变换（NaN 归一化 + ±0 归一化 +
//     符号单调化）→ 整数域全序；[int32] key 即原值。
//   块内：TROWMAX/TROWARGMAX（或 TCOLMAX/TCOLARGMAX，平局取最小索引）；
//   跨块：TCMP<GT> 严格大于 + TSEL 双状态（key/idx）合并 —— 严格大于
//     保证跨块平局取先出现块（首个最大值索引）。
//
// 指令契约（__linx 后端已验证的动态 Valid 支持）：
//   - TROWMAX/TROWARGMAX：目的物理单列，目的 valid rows = 源 valid rows
//     （DYNAMIC 载体 + 运行期 GetValidRow()）；
//   - TCOLMAX/TCOLARGMAX：目的恰好一行（1×N），目的列几何随源；
//   - TSEL：dest/mask/true 三者模板 Valid 参数必须一致（全动态载体即可），
//     mask 为谓词载体不做位宽检查；
//   - TCMP/TCMPS/TADDS/TANDS/TSHRS/TMULS/TXOR/TEXPANDS：动态 valid 走
//     GetValidRow()/GetValidCol() 运行期操作数。
//
// 入参契约（fail-closed，由 4PE 包装统一校验，本层只做编译期断言）：
//   - 0 < tile_m <= tM、0 < tile_n <= tN（strip 为运行期有效块宽）；
//   - gIM/gIN > 0；归约源物理容量 <= 2048B；状态 Tile >= 128B。
// ============================================================================

#ifndef ARGMAXV2_DETAIL_DYNAMIC_KERNEL_HPP
#define ARGMAXV2_DETAIL_DYNAMIC_KERNEL_HPP

#pragma once

#include "basic_op/argmaxv2/detail/argmaxv2.hpp"

namespace argmaxv2_dynamic_detail {

// ---------------------------------------------------------------------------
// sortable key 位变换（浮点，动态 Valid 载体版）：
//   与 argmaxv2_detail::make_sortable_key 逐指令一致；区别仅在于内部
//   常量/临时 Tile 必须以运行期 (vr, vc) 构造（动态载体默认构造的
//   valid mask 为 0，会静默退化为空区域）。
// ---------------------------------------------------------------------------
template <typename DType, typename TileData, typename TileKey>
inline __attribute__((always_inline)) void make_sortable_key_dynamic(
    TileData &raw, TileKey &key, size_t vr, size_t vc) {
    using Traits = argmaxv2_detail::ArgMaxTraits<DType>;
    using KeyT = typename Traits::KeyT;
    TileData scr(vr, vc), prd(vr, vc), msk(vr, vc);
    TileData nanConst(vr, vc), zeroConst(vr, vc);
    TEXPANDS(nanConst, static_cast<KeyT>(Traits::kNanBits));
    TEXPANDS(zeroConst, static_cast<KeyT>(0));
    // NaN 归一化：mag > +inf 的 mag ⟺ NaN（正/负位模式一并命中）
    TANDS(scr, raw, static_cast<KeyT>(Traits::kMagMask));
    TCMPS<CmpMode::GT>(prd, scr, static_cast<KeyT>(Traits::kInfMag));
    TSEL(raw, prd, nanConst);
    // ±0.0 归一化：-0.0 → +0.0（IEEE 相等，平局取先出现）
    TCMPS<CmpMode::EQ>(prd, raw, static_cast<KeyT>(Traits::kNegZero));
    TSEL(raw, prd, zeroConst);
    // 符号单调化：key = bits ^ (负数 ? 全 1 : 符号位)
    TSHRS(scr, raw, static_cast<KeyT>(Traits::kSignShift));
    TMULS(msk, scr, static_cast<KeyT>(Traits::kInvMag));
    TADDS(msk, msk, static_cast<KeyT>(Traits::kPosFlip));
    TXOR(key, raw, msk);
}

}  // namespace argmaxv2_dynamic_detail

// ----------------------------------------------------------------------------
// argmaxv2_row_dynamic：dim=1（尾轴）argmax，动态 shape（fp16/fp32/bf16/int32），
// 输入 [gIM, gIN] -> 输出 [gIM]。out[r] = 第 r 行首个最大值所在列索引。
//
//   物理Tile [tM, tN] 编译期固定（模板参数，包装层按 dtype traits 或显式
//   覆盖选择）；tile_m/tile_n 为运行期有效块宽（strip，<= 物理容量）。
//   尾数处理：vr = min(gIM - ib, tile_m)、vc = min(gIN - jb, tile_n)，
//   与 normalization 动态版同风格，任意行/列尾数均可。
// ----------------------------------------------------------------------------
template <typename dtype, typename idtype, const int tM, const int tN>
void argmaxv2_row_dynamic(dtype *in_ptr, idtype *out_ptr, int64_t gIM,
                          int64_t gIN, int64_t tile_m, int64_t tile_n) {
    using Traits = argmaxv2_detail::ArgMaxTraits<dtype>;
    using KeyT = typename Traits::KeyT;
    constexpr bool kIsFloat = argmaxv2_detail::kIsFloatKey<dtype>;

    static_assert(
        !std::is_same_v<dtype, int64_t>,
        "int64 input is not supported: the tile-op layer has no 64-bit "
        "reduce/compare instructions (TROWARGMAX/TCOLARGMAX contracts and "
        "the model reduce/compare dtype whitelist cap at 32-bit)");
    // 归约源物理容量 <= 2048B（TROWMAX/TROWARGMAX 契约）。
    static_assert(tM * tN * static_cast<int>(sizeof(KeyT)) <= 2048,
                  "reduction source tile exceeds the 2048B capacity limit");
    // 单列状态 Tile 物理容量 >= 128B（tile 尺寸下限经验）。
    static_assert(tM * static_cast<int>(sizeof(KeyT)) >= 128,
                  "row key-state tile must be at least 128B");
    static_assert(tM * static_cast<int>(sizeof(uint32_t)) >= 128,
                  "row index-state tile must be at least 128B");
    static_assert(std::is_same_v<idtype, int32_t> ||
                      std::is_same_v<idtype, int64_t>,
                  "output dtype must be int32_t or int64_t");

    // 动态 Valid 载体：物理 [tM,tN]/[tM,1]，valid 运行期收缩。
    // TROWARGMAX 目的契约：dtype 必须为 uint32_t、物理单列。
    using tileData = Tile<Location::Vec, KeyT, tM, tN, BLayout::RowMajor, -1, -1>;
    using tileKey = Tile<Location::Vec, KeyT, tM, 1, BLayout::RowMajor, -1, 1>;
    using tileIdx =
        Tile<Location::Vec, uint32_t, tM, 1, BLayout::RowMajor, -1, 1>;
    using tileI64 =
        Tile<Location::Vec, int64_t, tM, 1, BLayout::RowMajor, -1, 1>;
    // 谓词载体与比较源同形（TCMP 产出 packed predicate）
    using tilePred = tileKey;

    for (int64_t ib = 0; ib < gIM; ib += tile_m) {
        const int64_t vr = gIM - ib < tile_m ? gIM - ib : tile_m;
        const size_t vrS = static_cast<size_t>(vr);

        tileKey bestKey(vrS), chunkMax(vrS);
        tilePred predCmp(vrS);
        tileIdx bestIdx(vrS), chunkArg(vrS), gIdxTile(vrS);
        TEXPANDS(bestKey, static_cast<KeyT>(Traits::kMinKey));
        TEXPANDS(bestIdx, static_cast<uint32_t>(0));

        for (int64_t jb = 0; jb < gIN; jb += tile_n) {
            const int64_t vc = gIN - jb < tile_n ? gIN - jb : tile_n;
            const size_t vcS = static_cast<size_t>(vc);
            // GM 视图：浮点按 KeyT 位重解释（qli_radix 同源）；行 stride = gIN。
            global_tensor<KeyT, RowMajor<-1, -1>> gi(
                reinterpret_cast<KeyT *>(in_ptr) + ib * gIN + jb,
                static_cast<int>(vr), static_cast<int>(gIN));
            tileData dataTile(vrS, vcS), keyTile(vrS, vcS);
            TLOAD(dataTile, gi);
            if constexpr (kIsFloat) {
                argmaxv2_dynamic_detail::make_sortable_key_dynamic<dtype>(
                    dataTile, keyTile, vrS, vcS);
                TROWMAX(chunkMax, keyTile);
                TROWARGMAX(chunkArg, keyTile);
            } else {
                TROWMAX(chunkMax, dataTile);
                TROWARGMAX(chunkArg, dataTile);
            }
            // 严格大于更新：跨块平局保留先出现块（首个最大值索引）
            TCMP<CmpMode::GT>(predCmp, chunkMax, bestKey);
            TADDS(gIdxTile, chunkArg, static_cast<uint32_t>(jb));
            TSEL(bestKey, predCmp, chunkMax);
            TSEL(bestIdx, predCmp, gIdxTile);
        }

        // 输出：int32 位恒等直存（u32 视图），int64 走 TCVT。
        if constexpr (std::is_same_v<idtype, int32_t>) {
            global_tensor<uint32_t, RowMajor<-1, -1>> go(
                reinterpret_cast<uint32_t *>(out_ptr) + ib,
                static_cast<int>(vr), 1);
            TSTORE(go, bestIdx);
        } else {
            tileI64 outI64Tile(vrS);
            TCVT(outI64Tile, bestIdx);
            global_tensor<int64_t, RowMajor<-1, -1>> go(
                reinterpret_cast<int64_t *>(out_ptr) + ib,
                static_cast<int>(vr), 1);
            TSTORE(go, outI64Tile);
        }
    }
}

// ----------------------------------------------------------------------------
// argmaxv2_row_segmerge_dynamic：argmaxv2_row_dynamic 的段内归约变体
// （切 R 模板 argmaxv2_rsplit.hpp 专用；视图宽恒 16，全段一个调用）。
//
// 与原版的差异（全部为 gfsim tile 域约束让路，见 argmaxv2_rsplit.hpp 尾注）：
//   1. idx 全局化：idx = base + 行号*gIN + 视图内列号。行折叠在内核内
//      完成：TCI 生成 [1,tM] 行 ramp（预乘 gIN）→ GM 往返转 [tM,1] 列
//      （[C4] 同款列/行互转，group_token minScratch 实证）。
//   2. 逐行块 winner 落盘：每行块 (bestKey,bestIdx)[vr,1] 就地归并为
//      (kmax, idxAt)[1,1] —— TCOLMAX + TCOLEXPAND 广播 + TCMP<EQ> 掩码 +
//      TSEL（非命中填 0xFFFFFFFF 哨兵）+ TCOLMIN（key 平局取最小 idx =
//      块内首现）—— 经 [1,1] cell TSTORE 写入调用方 stage 区（per-PE
//      64B 独占行）。winner 不经任何回读路径，[1,1] 状态 tile 不存在
//      （tile 寄存器压力与基线内核持平，规避后端 S64 溢出断言）。
//   3. 跨块/跨 PE 合并由调用方在屏障后低压区完成（详见 rsplit）。
//
// 归约/平局语义与 argmaxv2_row_dynamic 逐指令一致（严格大于 + 平局取
// 最小索引 ⇒ 视图行内首现）。
// ----------------------------------------------------------------------------
template <typename dtype, const int tM, const int tN>
void argmaxv2_row_segmerge_dynamic(
    dtype *in_ptr, const uint32_t *ramp_scratch, int64_t gIM, int64_t gIN,
    int64_t tile_m, int64_t tile_n, uint32_t base,
    typename argmaxv2_detail::ArgMaxTraits<dtype>::KeyT *stage_key,
    uint32_t *stage_idx) {
    using Traits = argmaxv2_detail::ArgMaxTraits<dtype>;
    using KeyT = typename Traits::KeyT;
    constexpr bool kIsFloat = argmaxv2_detail::kIsFloatKey<dtype>;

    static_assert(
        !std::is_same_v<dtype, int64_t>,
        "int64 input is not supported: the tile-op layer has no "
        "64-bit reduce/compare instructions (TROWARGMAX/TCOLARGMAX contracts "
        "and the model reduce/compare dtype whitelist cap at 32-bit)");
    static_assert(tM * tN * static_cast<int>(sizeof(KeyT)) <= 2048,
                  "reduction source tile exceeds the 2048B capacity limit");
    static_assert(tM * static_cast<int>(sizeof(KeyT)) >= 128,
                  "row key-state tile must be at least 128B");
    static_assert(tM * static_cast<int>(sizeof(uint32_t)) >= 128,
                  "row index-state tile must be at least 128B");

    using tileData = Tile<Location::Vec, KeyT, tM, tN, BLayout::RowMajor, -1, -1>;
    using tileKey = Tile<Location::Vec, KeyT, tM, 1, BLayout::RowMajor, -1, 1>;
    using tileIdx =
        Tile<Location::Vec, uint32_t, tM, 1, BLayout::RowMajor, -1, 1>;
    using tilePred = tileKey;
    using tileKeyOne =
        Tile<Location::Vec, KeyT, tM, 1, BLayout::RowMajor, 1, 1>;
    using tileIdxOne =
        Tile<Location::Vec, uint32_t, tM, 1, BLayout::RowMajor, 1, 1>;

    // ramp_scratch 为调用方入口处一次性预构建的基础 ramp（0..tM-1，
    // [1,tM] 存 [tM,1] 读的 [C4] 往返）——与首次消费间隔整段行循环，
    // 最大化 TLSU store→load 距离（gfsim tile-store 通路缺口规避）。
    // 逐块按本视图 gIN 加权。
    for (int64_t ib = 0; ib < gIM; ib += tile_m) {
        const int64_t vr = gIM - ib < tile_m ? gIM - ib : tile_m;
        const size_t vrS = static_cast<size_t>(vr);

        tileKey bestKey(vrS), chunkMax(vrS);
        tilePred predCmp(vrS);
        tileIdx bestIdx(vrS), chunkArg(vrS), gIdxTile(vrS);
        TEXPANDS(bestKey, static_cast<KeyT>(Traits::kMinKey));
        TEXPANDS(bestIdx, static_cast<uint32_t>(0));

        for (int64_t jb = 0; jb < gIN; jb += tile_n) {
            const int64_t vc = gIN - jb < tile_n ? gIN - jb : tile_n;
            const size_t vcS = static_cast<size_t>(vc);
            global_tensor<KeyT, RowMajor<-1, -1>> gi(
                reinterpret_cast<KeyT *>(in_ptr) + ib * gIN + jb,
                static_cast<int>(vr), static_cast<int>(gIN));
            tileData dataTile(vrS, vcS), keyTile(vrS, vcS);
            TLOAD(dataTile, gi);
            if constexpr (kIsFloat) {
                argmaxv2_dynamic_detail::make_sortable_key_dynamic<dtype>(
                    dataTile, keyTile, vrS, vcS);
                TROWMAX(chunkMax, keyTile);
                TROWARGMAX(chunkArg, keyTile);
            } else {
                TROWMAX(chunkMax, dataTile);
                TROWARGMAX(chunkArg, dataTile);
            }
            TCMP<CmpMode::GT>(predCmp, chunkMax, bestKey);
            TADDS(gIdxTile, chunkArg, static_cast<uint32_t>(jb));
            TSEL(bestKey, predCmp, chunkMax);
            TSEL(bestIdx, predCmp, gIdxTile);
        }

        // 行折叠（chunkArg 复用为 ramp 列载体）：idx += ramp*gIN + base。
        // ramp 装载按 ib 偏移：多行块视图（gIM > tile_m，如长轴 tailA）
        // 的第 2+ 块须取 ramp[ib..ib+vr)，否则该块 idx 系统性少
        // ib*gIN（多行块视图下的存量缺陷）。
        {
            global_tensor<uint32_t, RowMajor<-1, -1>> grc(
                const_cast<uint32_t *>(ramp_scratch) + ib,
                static_cast<int>(vr), 1);
            TLOAD(chunkArg, grc);
            TMULS(chunkArg, chunkArg, static_cast<uint32_t>(gIN));
            TADD(bestIdx, bestIdx, chunkArg);
            TADDS(bestIdx, bestIdx, base);
        }

        // 逐块 winner 归并（[vr,1] → [1,1]）：块内 key 平局取最小 idx
        // （w 升序 ⇒ 块内首现）。合并链 [vr,1] 载体全部复用 jb 循环变量
        // （chunkMax→kB、predCmp→mask、gIdxTile→sel），新增仅 kmax/idxAt
        // 两个 [1,1]——tile 压力与基线内核持平。
        {
            tileKeyOne kmax;
            tileIdxOne idxAt;
            TCOLMAX(kmax, bestKey);
            TCOLEXPAND(chunkMax, kmax);
            TCMP<CmpMode::EQ>(predCmp, bestKey, chunkMax);
            TEXPANDS(gIdxTile, 0xFFFFFFFFu);
            TSEL(gIdxTile, predCmp, bestIdx);
            TCOLMIN(idxAt, gIdxTile);
             // [1,1] 落盘到 stage 区（调用方保证 per-PE 64B 独占行）。
             global_tensor<KeyT, RowMajor<-1, -1>> gk(
                 stage_key + ib / tile_m, 1, 1);
             TSTORE(gk, kmax);
             global_tensor<uint32_t, RowMajor<-1, -1>> gi32(
                 stage_idx + ib / tile_m, 1, 1);
             TSTORE(gi32, idxAt);
         }
     }
 }

// ----------------------------------------------------------------------------
// argmaxv2_rowpar_reduce：行并行直减（切 R 模板的 A 较大分支）。
//
// 与 segmerge 的分工（主循环形态按块数上界 A/B 定界，见 argmaxv2_rsplit.hpp）：
//   - segmerge（W-big reshape）：A 小——单行 reshape 填满 tile，但每行块
//     winner 需 [1,1] 落盘 + 哨兵（行定价小存储是 TLSU TSTORE 子路径
//     76% occupancy 的根源）；
//   - rowpar（本内核）：A 大行利用率高——原始行直接作 tile 行（跨步
//     kAxisDim），段作列块串。[kOuter,1] 寄存器运行态（严格 GT ⇒ 行内
//     首现），块维在寄存器内消耗：无 reshape / 无 winner 链 / 无哨兵 /
//     无 ramp，stage 每 PE 仅 (key, idx) 两个 [kOuter,1] 列存（行距
//     cell_stride）。
//
// idx 全局化直接 TADDS（segStart + jb + 块内列号），无 ramp 折叠。
// ----------------------------------------------------------------------------
template <typename dtype, const int tM, const int tN>
void argmaxv2_rowpar_reduce(
    dtype *in_ptr, int64_t kOuter, int64_t kAxisDim, int64_t segStart,
    int64_t S,
    typename argmaxv2_detail::ArgMaxTraits<dtype>::KeyT *stage_key,
    uint32_t *stage_idx, int64_t cell_stride) {
    using Traits = argmaxv2_detail::ArgMaxTraits<dtype>;
    using KeyT = typename Traits::KeyT;
    constexpr bool kIsFloat = argmaxv2_detail::kIsFloatKey<dtype>;

    static_assert(tM * tN * static_cast<int>(sizeof(KeyT)) <= 2048,
                  "reduction source tile exceeds the 2048B capacity limit");
    static_assert(tM * static_cast<int>(sizeof(KeyT)) >= 128,
                  "row key-state tile must be at least 128B");
    static_assert(tM * static_cast<int>(sizeof(uint32_t)) >= 128,
                  "row index-state tile must be at least 128B");

    using tileData = Tile<Location::Vec, KeyT, tM, tN, BLayout::RowMajor, -1, -1>;
    using tileKey = Tile<Location::Vec, KeyT, tM, 1, BLayout::RowMajor, -1, 1>;
    using tileIdx =
        Tile<Location::Vec, uint32_t, tM, 1, BLayout::RowMajor, -1, 1>;
    using tilePred = tileKey;

    // 单参构造（validRow）：[tM,1] 窄列 tile 的两参 (rows,1) 构造会触发
    // 后端 [2,16] 打包载体（TCMP 源维度失配断言），segmerge 同款单参
    // 构造为已验证形态。
    const size_t AS = static_cast<size_t>(kOuter);
    tileKey bestKey(AS), chunkMax(AS);
    tilePred predCmp(AS);
    tileIdx bestIdx(AS), chunkArg(AS), gIdxTile(AS);
    TEXPANDS(bestKey, static_cast<KeyT>(Traits::kMinKey));
    TEXPANDS(bestIdx, static_cast<uint32_t>(0));

    for (int64_t jb = 0; jb < S; jb += tN) {
        const int64_t vc = S - jb < tN ? S - jb : tN;
        const size_t vcS = static_cast<size_t>(vc);
        // 视图 [kOuter, S]（行跨步 kAxisDim）的列块 [kOuter, vc]。
        global_tensor<KeyT, RowMajor<-1, -1>> g(
            reinterpret_cast<KeyT *>(in_ptr) + segStart + jb, kOuter,
            kAxisDim);
        tileData dataTile(AS, vcS);
        TLOAD(dataTile, g);
        if constexpr (kIsFloat) {
            tileData keyTile(AS, vcS);
            argmaxv2_dynamic_detail::make_sortable_key_dynamic<dtype>(
                dataTile, keyTile, AS, vcS);
            TROWMAX(chunkMax, keyTile);
            TROWARGMAX(chunkArg, keyTile);
        } else {
            TROWMAX(chunkMax, dataTile);
            TROWARGMAX(chunkArg, dataTile);
        }
        TCMP<CmpMode::GT>(predCmp, chunkMax, bestKey);
        TADDS(gIdxTile, chunkArg, static_cast<uint32_t>(jb));
        TSEL(bestKey, predCmp, chunkMax);
        TSEL(bestIdx, predCmp, gIdxTile);
    }
    TADDS(bestIdx, bestIdx, static_cast<uint32_t>(segStart));

    // stage 列存 [kOuter,1]（行距 cell_stride，写每行 cell 0）。
    global_tensor<KeyT, RowMajor<-1, -1>> gk(stage_key, kOuter, cell_stride);
    TSTORE(gk, bestKey);
    global_tensor<uint32_t, RowMajor<-1, -1>> gi(stage_idx, kOuter,
                                                 cell_stride);
    TSTORE(gi, bestIdx);
}


// ----------------------------------------------------------------------------
// argmaxv2_col_dynamic：dim=0（行轴）argmax，动态 shape（fp16/fp32/bf16/int32），
// 输入 [gIM, gIN] -> 输出 [1, gIN]。out[c] = 第 c 列首个最大值所在行索引。
//
//   列窗口：colBegin/colEnd 为运行期参数（4-PE 包装按列块 ceil 分区，
//   窗口自动块对齐；每列归约线独立，窗口内尾块 vc < tile_n 亦合法）。
//   尾数处理：vr = min(gIM - ib, tile_m)；列方向任意尾数。
// ----------------------------------------------------------------------------
template <typename dtype, typename idtype, const int tM, const int tN>
void argmaxv2_col_dynamic(dtype *in_ptr, idtype *out_ptr, int64_t gIM,
                          int64_t gIN, int64_t colBegin, int64_t colEnd,
                          int64_t tile_m, int64_t tile_n) {
    using Traits = argmaxv2_detail::ArgMaxTraits<dtype>;
    using KeyT = typename Traits::KeyT;
    constexpr bool kIsFloat = argmaxv2_detail::kIsFloatKey<dtype>;

    static_assert(
        !std::is_same_v<dtype, int64_t>,
        "int64 input is not supported: the tile-op layer has no 64-bit "
        "reduce/compare instructions (TROWARGMAX/TCOLARGMAX contracts and "
        "the model reduce/compare dtype whitelist cap at 32-bit)");
    static_assert(tM * tN * static_cast<int>(sizeof(KeyT)) <= 2048,
                  "reduction source tile exceeds the 2048B capacity limit");
    static_assert(tN * static_cast<int>(sizeof(KeyT)) >= 128,
                  "col key-state tile must be at least 128B");
    static_assert(tN * static_cast<int>(sizeof(uint32_t)) >= 128,
                  "col index-state tile must be at least 128B");
    static_assert(std::is_same_v<idtype, int32_t> ||
                      std::is_same_v<idtype, int64_t>,
                  "output dtype must be int32_t or int64_t");

    using tileData = Tile<Location::Vec, KeyT, tM, tN, BLayout::RowMajor, -1, -1>;

    // 状态 Tile（物理 [1, tN]）。TCOLARGMAX 目的契约：dtype 必须为 uint32_t、
    // 恰好一行（1×N），列几何随源（动态 ValidCol 载体）。
    using tileKey = Tile<Location::Vec, KeyT, 1, tN, BLayout::RowMajor, 1, -1>;
    using tileIdx =
        Tile<Location::Vec, uint32_t, 1, tN, BLayout::RowMajor, 1, -1>;
    using tilePred = tileKey;
    using tileI64 = Tile<Location::Vec, int64_t, 1, tN, BLayout::RowMajor, 1, -1>;

    // ---- 逐列块：每个列块独立状态（各列归约线互不影响） ------------------
    for (int64_t jc = colBegin; jc < colEnd; jc += tile_n) {
        const int64_t vc = colEnd - jc < tile_n ? colEnd - jc : tile_n;
        const size_t vcS = static_cast<size_t>(vc);

        tileKey bestKey(vcS), chunkMax(vcS);
        tilePred predCmp(vcS);
        tileIdx bestIdx(vcS), chunkArg(vcS), gIdxTile(vcS);
        TEXPANDS(bestKey, static_cast<KeyT>(Traits::kMinKey));
        TEXPANDS(bestIdx, static_cast<uint32_t>(0));

        for (int64_t ib = 0; ib < gIM; ib += tile_m) {
            const int64_t vr = gIM - ib < tile_m ? gIM - ib : tile_m;
            const size_t vrS = static_cast<size_t>(vr);
            global_tensor<KeyT, RowMajor<-1, -1>> gi(
                reinterpret_cast<KeyT *>(in_ptr) + ib * gIN + jc,
                static_cast<int>(vr), static_cast<int>(gIN));
            tileData dataTile(vrS, vcS), keyTile(vrS, vcS);
            TLOAD(dataTile, gi);
            if constexpr (kIsFloat) {
                argmaxv2_dynamic_detail::make_sortable_key_dynamic<dtype>(
                    dataTile, keyTile, vrS, vcS);
                TCOLMAX(chunkMax, keyTile);
                TCOLARGMAX(chunkArg, keyTile);
            } else {
                TCOLMAX(chunkMax, dataTile);
                TCOLARGMAX(chunkArg, dataTile);
            }
            // 严格大于更新：跨行块平局保留先出现行块（首个最大值索引）
            TCMP<CmpMode::GT>(predCmp, chunkMax, bestKey);
            TADDS(gIdxTile, chunkArg, static_cast<uint32_t>(ib));
            TSEL(bestKey, predCmp, chunkMax);
            TSEL(bestIdx, predCmp, gIdxTile);
        }

        if constexpr (std::is_same_v<idtype, int32_t>) {
            global_tensor<uint32_t, RowMajor<-1, -1>> go(
                reinterpret_cast<uint32_t *>(out_ptr) + jc, 1,
                static_cast<int>(gIN));
            TSTORE(go, bestIdx);
        } else {
            tileI64 outI64Tile(vcS);
            TCVT(outI64Tile, bestIdx);
            global_tensor<int64_t, RowMajor<-1, -1>> go(
                reinterpret_cast<int64_t *>(out_ptr) + jc, 1,
                static_cast<int>(gIN));
            TSTORE(go, outI64Tile);
        }
    }
}

#endif  // ARGMAXV2_DETAIL_DYNAMIC_KERNEL_HPP
