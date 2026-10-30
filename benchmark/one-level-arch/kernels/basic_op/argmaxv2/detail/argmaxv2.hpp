// ============================================================================
// ArgMaxV2 算子 — PTO 一层编程模型（单 PE 基础实现，静态 shape）
//
// 动态 shape 对应实现见 argmaxv2_dynamic.hpp（运行期 tiling、动态 Valid
// 载体）；本静态实现不变。
//
// 功能语义对齐 CANN ArgMaxV2（aclnnArgMax 两段式接口的执行算子）：
//
//   y[i0, ..., i(dim-1), i(dim+1), ..., i(n-1)]
//       = argmax_j x[i0, ..., i(dim-1), j, i(dim+1), ..., i(n-1)]
//
//   - 本实现为 2D 输入：dim=1（尾轴，沿列归约）/ dim=0（行轴，沿行归约）。
//   - 平局语义：多个相等最大值时返回首个（最小）索引 —— 与
//     CANN / PyTorch / ONNX(select_last_index=0) 一致。
//   - NaN 语义（fp16/fp32/bf16，对齐 PyTorch 与 CANN 910B/950）：任意
//     （正/负位模式的）NaN 视为最大值，返回首个 NaN 索引。
//   - ±0.0 视为相等（IEEE ==），平局取先出现者。
//   - 归约维长度为 1 时结果恒为 0（自然成立，无特化分支）。
//   - 输出 dtype 与输入 dtype 解耦：int32 / int64（CANN kernel 内部恒为
//     int32，int64 由出口转换；本实现 int32 位恒等直存、int64 走 TCVT）。
//   - dim=0 路径要求 gIN % tN == 0（列窗口对齐）；dim=1 路径支持任意
//     行/列尾数（ValidRow/ValidCol 载体，与 reduction detail 同风格）。
//
// 实现策略（全部 Tile 级指令，无 __vec__ 块）：
//
//   [fp16/fp32/bf16] 浮点输入先做「sortable key」位变换（qli_radix /
//   topk_tiled 同源技巧）：
//     1) NaN 归一化：mag = bits & 去符号掩码；mag > +inf 的 mag 即 NaN
//        （单次 TCMPS<GT>，正/负 NaN 位模式一并命中），TSEL 替换为
//        canonical 正 NaN 位模式 —— 所有 NaN 在 key 域相等且大于 +inf，
//        平局取首个 NaN。
//     2) ±0.0 归一化：-0.0 位模式替换为 +0.0（IEEE 相等语义）。
//     3) 符号单调化（topk load_bin 同源 4-op）：
//        key = bits ^ (负数 ? 全 1 : 符号位)，浮点全序 → 无符号整数全序。
//     此后 argmax 归约完全落在整数域，无 NaN 比较歧义。
//     块内归约：TROWMAX（行最大 key）+ TROWARGMAX（行最大的最小列索引，
//     契约明确平局取最小索引）；跨块 TCMP<GT> 严格大于 + TSEL 双状态
//     （key/idx）合并 —— 严格大于保证跨块平局取先出现块。
//
//   [int32] 全序整数，key 即原始值，直接归约（同上块内/跨块结构）。
//
//   [int64] 暂不支持输入：tile-op 层未封装 64 位归约/比较指令
//   （TROWARGMAX/TCOLARGMAX 契约无 S64，gfrun 模型 reduce/compare
//   dtype 白名单亦只到 32 位）。输出 idtype=int64 不受影响（u32 索引
//   出口 TCVT 直存）。待 tile-op 层补齐 64 位指令后可按「块内 64 位
//   TROWMAX + EQ 掩码 + TROWMIN 首命中回溯」补齐输入支持。
//
//   TROWARGMAX/TCOLARGMAX 契约：目的 Tile dtype 必须为 uint32_t；归约源
//   Tile 物理容量 ≤ 2048B；平局取最小索引。
//   单列/单行状态 Tile 参考 reduction detail 的既有形状（[16,1]/[32,1]
//   等，TROWMAX 目的先例）。
//
// 指令依赖（均已在本仓其它算子/微基准中验证过同形态调用）：
//   TLOAD/TSTORE/TCVT/TEXPANDS/TANDS/TXORS/TXOR/TSHRS/TMULS/TADDS/
//   TCMPS/TCMP/TSEL/TROWMAX/TROWARGMAX/TCOLMAX/TCOLARGMAX
// ============================================================================

#ifndef ARGMAXV2_DETAIL_KERNEL_HPP
#define ARGMAXV2_DETAIL_KERNEL_HPP

#pragma once

#include <common/pto_tileop.hpp>

#include <cstdint>
#include <type_traits>

using namespace pto;

namespace argmaxv2_detail {

// ---------------------------------------------------------------------------
// 类型特征：输入 dtype → sortable key dtype 与位掩码常数 / 默认 tile 形状
// ---------------------------------------------------------------------------
template <typename DType>
struct ArgMaxTraits;  // 仅特化版本可用

// FP32：32 位 sortable key
template <>
struct ArgMaxTraits<float> {
    using KeyT = uint32_t;
    static constexpr KeyT kMagMask = 0x7FFFFFFFu;   // 去符号位
    static constexpr KeyT kInfMag = 0x7F800000u;    // +inf 的去符号位值
    static constexpr KeyT kNanBits = 0x7FC00000u;   // canonical 正 NaN
    static constexpr KeyT kNegZero = 0x80000000u;   // -0.0 位模式
    static constexpr KeyT kSignShift = 31u;
    static constexpr KeyT kInvMag = 0x7FFFFFFFu;    // 负数掩码乘子
    static constexpr KeyT kPosFlip = 0x80000000u;   // 正数 XOR 掩码
    static constexpr KeyT kMinKey = 0u;             // 无符号 key 域下界
    // 行归约默认 tile：[32,16] u32 = 2048B；状态 [32,1] u32 = 128B
    static constexpr int kRowTileM = 32;
    static constexpr int kRowTileN = 16;
    // 列归约默认 tile：[16,32] u32 = 2048B；状态 [1,32] u32 = 128B
    static constexpr int kColTileM = 16;
    static constexpr int kColTileN = 32;
};

// FP16：16 位 sortable key
template <>
struct ArgMaxTraits<__half> {
    using KeyT = uint16_t;
    static constexpr KeyT kMagMask = 0x7FFFu;
    static constexpr KeyT kInfMag = 0x7C00u;
    static constexpr KeyT kNanBits = 0x7E00u;
    static constexpr KeyT kNegZero = 0x8000u;
    static constexpr KeyT kSignShift = 15u;
    static constexpr KeyT kInvMag = 0x7FFFu;
    static constexpr KeyT kPosFlip = 0x8000u;
    static constexpr KeyT kMinKey = 0u;
    // 行归约默认 tile：[64,16] u16 = 2048B；状态 [64,1] u16 = 128B
    static constexpr int kRowTileM = 64;
    static constexpr int kRowTileN = 16;
    // 列归约默认 tile：[16,64] u16 = 2048B；状态 [1,64] u16 = 128B
    static constexpr int kColTileM = 16;
    static constexpr int kColTileN = 64;
};

// BF16：16 位 sortable key（bf16 = fp32 高 16 位，指数/尾数布局同构）
template <>
struct ArgMaxTraits<__bf16> {
    using KeyT = uint16_t;
    static constexpr KeyT kMagMask = 0x7FFFu;
    static constexpr KeyT kInfMag = 0x7F80u;
    static constexpr KeyT kNanBits = 0x7FC0u;
    static constexpr KeyT kNegZero = 0x8000u;
    static constexpr KeyT kSignShift = 15u;
    static constexpr KeyT kInvMag = 0x7FFFu;
    static constexpr KeyT kPosFlip = 0x8000u;
    static constexpr KeyT kMinKey = 0u;
    static constexpr int kRowTileM = 64;
    static constexpr int kRowTileN = 16;
    static constexpr int kColTileM = 16;
    static constexpr int kColTileN = 64;
};

// INT32：全序整数，key 即原始值（无 NaN、无变换）
template <>
struct ArgMaxTraits<int32_t> {
    using KeyT = int32_t;
    static constexpr KeyT kMinKey = static_cast<KeyT>(-2147483647 - 1);
    static constexpr int kRowTileM = 32;
    static constexpr int kRowTileN = 16;
    static constexpr int kColTileM = 16;
    static constexpr int kColTileN = 32;
};

// 浮点判定（需要 sortable key 位变换）；int32 为恒等路径。
template <typename DType>
inline constexpr bool kIsFloatKey =
    std::is_same_v<DType, float> || std::is_same_v<DType, __half> ||
    std::is_same_v<DType, __bf16>;

// ---------------------------------------------------------------------------
// sortable key 位变换（浮点）：raw（已 TLOAD 的 bits tile）→ key
//   NaN 归一化（3 op）+ ±0 归一化（2 op）+ 符号单调化（4 op）
// int32 无需调用。
// ---------------------------------------------------------------------------
template <typename DType, typename TileData, typename TileKey>
inline __attribute__((always_inline)) void make_sortable_key(TileData &raw,
                                                             TileKey &key) {
    using Traits = ArgMaxTraits<DType>;
    using KeyT = typename Traits::KeyT;
    TileData scr, prd, msk, nanConst, zeroConst;
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

}  // namespace argmaxv2_detail

// ----------------------------------------------------------------------------
// argmaxv2_row_impl：dim=1（尾轴）argmax（fp16/fp32/bf16/int32），
// 输入 [gIM, gIN] -> 输出 [gIM]。out[r] = 第 r 行首个最大值所在列索引。
//
//   尾数处理：rmd_M = gIM%tM（尾行块 ValidRow 载体）、rmd_N = gIN%tN
//   （尾列块 ValidCol 载体），与 reduction detail 同风格。
// ----------------------------------------------------------------------------
template <typename dtype, typename idtype, const int gIM, const int gIN,
          const int tM, const int tN>
void argmaxv2_row_impl(dtype *in_ptr, idtype *out_ptr) {
    using Traits = argmaxv2_detail::ArgMaxTraits<dtype>;
    using KeyT = typename Traits::KeyT;
    constexpr bool kIsFloat = argmaxv2_detail::kIsFloatKey<dtype>;

    constexpr int Mb = gIM / tM;
    constexpr int Nb = gIN / tN;
    constexpr int rmd_M = gIM % tM;
    constexpr int rmd_N = gIN % tN;

    static_assert(gIM > 0 && gIN > 0, "tensor must be non-empty");
    static_assert(
        !std::is_same_v<dtype, int64_t>,
        "int64 input is not supported: the tile-op layer has no 64-bit "
        "reduce/compare instructions (TROWARGMAX/TCOLARGMAX contracts and "
        "the model reduce/compare dtype whitelist cap at 32-bit)");
    // 归约源物理容量 ≤ 2048B（TROWMAX/TROWARGMAX 契约）。
    static_assert(tM * tN * static_cast<int>(sizeof(KeyT)) <= 2048,
                  "reduction source tile exceeds the 2048B capacity limit");
    // 单列状态 Tile 物理容量 ≥ 128B（tile 尺寸下限经验）。
    static_assert(tM * static_cast<int>(sizeof(KeyT)) >= 128,
                  "row key-state tile must be at least 128B");
    static_assert(tM * static_cast<int>(sizeof(uint32_t)) >= 128,
                  "row index-state tile must be at least 128B");

    // GM 视图：浮点按 KeyT 位重解释（qli_radix 同源），int32 直接。
    using gmIn = global_tensor<KeyT, RowMajor<gIM, gIN>>;

    using tileData = Tile<Location::Vec, KeyT, tM, tN, BLayout::RowMajor>;
    // 尾列 / 尾行 / 尾角载体（物理同 tileData，valid 收缩）
    using tileData_rN =
        Tile<Location::Vec, KeyT, tM, tN, BLayout::RowMajor, tM, rmd_N>;
    using tileData_rM =
        Tile<Location::Vec, KeyT, tM, tN, BLayout::RowMajor, rmd_M, tN>;
    using tileData_rMN =
        Tile<Location::Vec, KeyT, tM, tN, BLayout::RowMajor, rmd_M, rmd_N>;

    // 状态 Tile：块内归约结果与跨块合并状态（物理 [tM,1]，valid [tM,1] /
    // 尾行 [rmd_M,1]）。TROWARGMAX 目的契约：dtype 必须为 uint32_t。
    using tileKey = Tile<Location::Vec, KeyT, tM, 1, BLayout::RowMajor, tM, 1>;
    using tileKey_rM =
        Tile<Location::Vec, KeyT, tM, 1, BLayout::RowMajor, rmd_M, 1>;
    using tileIdx =
        Tile<Location::Vec, uint32_t, tM, 1, BLayout::RowMajor, tM, 1>;
    using tileIdx_rM =
        Tile<Location::Vec, uint32_t, tM, 1, BLayout::RowMajor, rmd_M, 1>;
    // 谓词载体与比较源同形（TCMP 产出 packed predicate）
    using tilePred = tileKey;
    using tilePred_rM = tileKey_rM;

    using itIn = global_iterator<gmIn, tileData>;

    // 输出视图：int32 位恒等直存（u32 视图），int64 走 TCVT。
    static_assert(std::is_same_v<idtype, int32_t> ||
                      std::is_same_v<idtype, int64_t>,
                  "output dtype must be int32_t or int64_t");
    using gmOutU32 = global_tensor<uint32_t, RowMajor<gIM, 1>>;
    using gmOutI64 = global_tensor<int64_t, RowMajor<gIM, 1>>;
    using itOutU32 = global_iterator<gmOutU32, tileIdx>;
    using itOutI64 = global_iterator<
        gmOutI64,
        Tile<Location::Vec, int64_t, tM, 1, BLayout::RowMajor, tM, 1>>;
    using itOutI64_rM = global_iterator<
        gmOutI64,
        Tile<Location::Vec, int64_t, tM, 1, BLayout::RowMajor, rmd_M, 1>>;

    itIn gIIter(reinterpret_cast<KeyT *>(in_ptr));
    itOutU32 gOIterU32(reinterpret_cast<uint32_t *>(out_ptr));
    itOutI64 gOIterI64(reinterpret_cast<int64_t *>(out_ptr));
    itOutI64_rM gOIterI64_rM(reinterpret_cast<int64_t *>(out_ptr));

    tileData dataTile, keyTile;
    tileData_rN dataTile_rN, keyTile_rN;
    tileData_rM dataTile_rM, keyTile_rM;
    tileData_rMN dataTile_rMN, keyTile_rMN;
    tileKey chunkMax, bestKey;
    tileKey_rM chunkMax_rM, bestKey_rM;
    tileIdx chunkArg, bestIdx, gIdxTile;
    tileIdx_rM chunkArg_rM, bestIdx_rM, gIdxTile_rM;
    tilePred predCmp;
    tilePred_rM predCmp_rM;
    Tile<Location::Vec, int64_t, tM, 1, BLayout::RowMajor, tM, 1> outI64Tile;
    Tile<Location::Vec, int64_t, tM, 1, BLayout::RowMajor, rmd_M, 1>
        outI64Tile_rM;

    // ---- 主行块扫描：Mb × (Nb 满列块 + 尾列块) ---------------------------
    #pragma clang loop unroll(full)
    for (int i = 0; i < Mb; ++i) {
        TEXPANDS(bestKey, static_cast<KeyT>(Traits::kMinKey));
        TEXPANDS(bestIdx, static_cast<uint32_t>(0));

        #pragma clang loop unroll(full)
        for (int j = 0; j < Nb; ++j) {
            auto gI = gIIter(i, j);
            TLOAD(dataTile, gI);
            if constexpr (kIsFloat) {
                argmaxv2_detail::make_sortable_key<dtype>(dataTile, keyTile);
                TROWMAX(chunkMax, keyTile);
                TROWARGMAX(chunkArg, keyTile);
            } else {
                TROWMAX(chunkMax, dataTile);
                TROWARGMAX(chunkArg, dataTile);
            }
            // 严格大于更新：跨块平局保留先出现块（首个最大值索引）
            TCMP<CmpMode::GT>(predCmp, chunkMax, bestKey);
            TADDS(gIdxTile, chunkArg, static_cast<uint32_t>(j * tN));
            TSEL(bestKey, predCmp, chunkMax);
            TSEL(bestIdx, predCmp, gIdxTile);
        }
        if constexpr (rmd_N > 0) {
            auto gI = gIIter(i, Nb);
            TLOAD(dataTile_rN, gI);
            if constexpr (kIsFloat) {
                argmaxv2_detail::make_sortable_key<dtype>(dataTile_rN,
                                                          keyTile_rN);
                TROWMAX(chunkMax, keyTile_rN);
                TROWARGMAX(chunkArg, keyTile_rN);
            } else {
                TROWMAX(chunkMax, dataTile_rN);
                TROWARGMAX(chunkArg, dataTile_rN);
            }
            TCMP<CmpMode::GT>(predCmp, chunkMax, bestKey);
            TADDS(gIdxTile, chunkArg, static_cast<uint32_t>(Nb * tN));
            TSEL(bestKey, predCmp, chunkMax);
            TSEL(bestIdx, predCmp, gIdxTile);
        }

        if constexpr (std::is_same_v<idtype, int32_t>) {
            auto gO = gOIterU32(i, 0);
            TSTORE(gO, bestIdx);
        } else {
            TCVT(outI64Tile, bestIdx);
            auto gO = gOIterI64(i, 0);
            TSTORE(gO, outI64Tile);
        }
    }

    // ---- 尾行块扫描：valid [rmd_M, tN] 载体 ------------------------------
    if constexpr (rmd_M > 0) {
        TEXPANDS(bestKey_rM, static_cast<KeyT>(Traits::kMinKey));
        TEXPANDS(bestIdx_rM, static_cast<uint32_t>(0));

        #pragma clang loop unroll(full)
        for (int j = 0; j < Nb; ++j) {
            auto gI = gIIter(Mb, j);
            TLOAD(dataTile_rM, gI);
            if constexpr (kIsFloat) {
                argmaxv2_detail::make_sortable_key<dtype>(dataTile_rM,
                                                          keyTile_rM);
                TROWMAX(chunkMax_rM, keyTile_rM);
                TROWARGMAX(chunkArg_rM, keyTile_rM);
            } else {
                TROWMAX(chunkMax_rM, dataTile_rM);
                TROWARGMAX(chunkArg_rM, dataTile_rM);
            }
            TCMP<CmpMode::GT>(predCmp_rM, chunkMax_rM, bestKey_rM);
            TADDS(gIdxTile_rM, chunkArg_rM, static_cast<uint32_t>(j * tN));
            TSEL(bestKey_rM, predCmp_rM, chunkMax_rM);
            TSEL(bestIdx_rM, predCmp_rM, gIdxTile_rM);
        }
        if constexpr (rmd_N > 0) {
            auto gI = gIIter(Mb, Nb);
            TLOAD(dataTile_rMN, gI);
            if constexpr (kIsFloat) {
                argmaxv2_detail::make_sortable_key<dtype>(dataTile_rMN,
                                                          keyTile_rMN);
                TROWMAX(chunkMax_rM, keyTile_rMN);
                TROWARGMAX(chunkArg_rM, keyTile_rMN);
            } else {
                TROWMAX(chunkMax_rM, dataTile_rMN);
                TROWARGMAX(chunkArg_rM, dataTile_rMN);
            }
            TCMP<CmpMode::GT>(predCmp_rM, chunkMax_rM, bestKey_rM);
            TADDS(gIdxTile_rM, chunkArg_rM, static_cast<uint32_t>(Nb * tN));
            TSEL(bestKey_rM, predCmp_rM, chunkMax_rM);
            TSEL(bestIdx_rM, predCmp_rM, gIdxTile_rM);
        }

        if constexpr (std::is_same_v<idtype, int32_t>) {
            auto gO = gOIterU32(Mb, 0);
            TSTORE(gO, bestIdx_rM);
        } else {
            TCVT(outI64Tile_rM, bestIdx_rM);
            auto gO = gOIterI64_rM(Mb, 0);
            TSTORE(gO, outI64Tile_rM);
        }
    }
}

// argmaxv2_col_impl：dim=0（行轴）argmax（fp16/fp32/bf16/int32），
// 输入 [gIM, gIN] -> 输出 [1, gIN]。out[c] = 第 c 列首个最大值所在行索引。
//
//   列窗口：colBegin/colEnd 为运行期参数（4-PE 包装按列 SPMD 切分），
//   要求 colBegin % tN == 0 且 (colEnd - colBegin) % tN == 0。
//   尾数处理：rmd_M（尾行块 ValidRow 载体）；列方向要求 gIN % tN == 0。
// ----------------------------------------------------------------------------
template <typename dtype, typename idtype, const int gIM, const int gIN,
          const int tM, const int tN>
void argmaxv2_col_impl(dtype *in_ptr, idtype *out_ptr, int colBegin,
                       int colEnd) {
    using Traits = argmaxv2_detail::ArgMaxTraits<dtype>;
    using KeyT = typename Traits::KeyT;
    constexpr bool kIsFloat = argmaxv2_detail::kIsFloatKey<dtype>;

    constexpr int Mb = gIM / tM;
    constexpr int rmd_M = gIM % tM;

    static_assert(gIM > 0 && gIN > 0, "tensor must be non-empty");
    static_assert(
        !std::is_same_v<dtype, int64_t>,
        "int64 input is not supported: the tile-op layer has no 64-bit "
        "reduce/compare instructions (TROWARGMAX/TCOLARGMAX contracts and "
        "the model reduce/compare dtype whitelist cap at 32-bit)");
    static_assert(gIN % tN == 0,
                  "argmaxv2_col requires gIN divisible by tN");
    static_assert(tM * tN * static_cast<int>(sizeof(KeyT)) <= 2048,
                  "reduction source tile exceeds the 2048B capacity limit");
    static_assert(tN * static_cast<int>(sizeof(KeyT)) >= 128,
                  "col key-state tile must be at least 128B");
    static_assert(tN * static_cast<int>(sizeof(uint32_t)) >= 128,
                  "col index-state tile must be at least 128B");

    using gmIn = global_tensor<KeyT, RowMajor<gIM, gIN>>;

    using tileData = Tile<Location::Vec, KeyT, tM, tN, BLayout::RowMajor>;
    using tileData_rM =
        Tile<Location::Vec, KeyT, tM, tN, BLayout::RowMajor, rmd_M, tN>;

    // 状态 Tile（物理 [1, tN]）。TCOLARGMAX 目的契约：dtype 必须为 uint32_t。
    using tileKey = Tile<Location::Vec, KeyT, 1, tN, BLayout::RowMajor, 1, tN>;
    using tileIdx =
        Tile<Location::Vec, uint32_t, 1, tN, BLayout::RowMajor, 1, tN>;
    using tilePred = tileKey;

    using itIn = global_iterator<gmIn, tileData>;

    static_assert(std::is_same_v<idtype, int32_t> ||
                      std::is_same_v<idtype, int64_t>,
                  "output dtype must be int32_t or int64_t");
    using gmOutU32 = global_tensor<uint32_t, RowMajor<1, gIN>>;
    using gmOutI64 = global_tensor<int64_t, RowMajor<1, gIN>>;
    using itOutU32 = global_iterator<gmOutU32, tileIdx>;
    using itOutI64 = global_iterator<
        gmOutI64,
        Tile<Location::Vec, int64_t, 1, tN, BLayout::RowMajor, 1, tN>>;

    itIn gIIter(reinterpret_cast<KeyT *>(in_ptr));
    itOutU32 gOIterU32(reinterpret_cast<uint32_t *>(out_ptr));
    itOutI64 gOIterI64(reinterpret_cast<int64_t *>(out_ptr));

    tileData dataTile, keyTile;
    tileData_rM dataTile_rM, keyTile_rM;
    tileKey chunkMax, bestKey;
    tileIdx chunkArg, bestIdx, gIdxTile;
    tilePred predCmp;
    Tile<Location::Vec, int64_t, 1, tN, BLayout::RowMajor, 1, tN> outI64Tile;

    const int jcFirst = colBegin / tN;
    const int jcLast = colEnd / tN;

    // ---- 逐列块：每个列块独立状态（各列归约线互不影响） ------------------
    for (int jc = jcFirst; jc < jcLast; ++jc) {
        TEXPANDS(bestKey, static_cast<KeyT>(Traits::kMinKey));
        TEXPANDS(bestIdx, static_cast<uint32_t>(0));

        #pragma clang loop unroll(full)
        for (int i = 0; i < Mb; ++i) {
            auto gI = gIIter(i, jc);
            TLOAD(dataTile, gI);
            if constexpr (kIsFloat) {
                argmaxv2_detail::make_sortable_key<dtype>(dataTile, keyTile);
                TCOLMAX(chunkMax, keyTile);
                TCOLARGMAX(chunkArg, keyTile);
            } else {
                TCOLMAX(chunkMax, dataTile);
                TCOLARGMAX(chunkArg, dataTile);
            }
            // 严格大于更新：跨行块平局保留先出现行块（首个最大值索引）
            TCMP<CmpMode::GT>(predCmp, chunkMax, bestKey);
            TADDS(gIdxTile, chunkArg, static_cast<uint32_t>(i * tM));
            TSEL(bestKey, predCmp, chunkMax);
            TSEL(bestIdx, predCmp, gIdxTile);
        }
        if constexpr (rmd_M > 0) {
            auto gI = gIIter(Mb, jc);
            TLOAD(dataTile_rM, gI);
            if constexpr (kIsFloat) {
                argmaxv2_detail::make_sortable_key<dtype>(dataTile_rM,
                                                          keyTile_rM);
                TCOLMAX(chunkMax, keyTile_rM);
                TCOLARGMAX(chunkArg, keyTile_rM);
            } else {
                TCOLMAX(chunkMax, dataTile_rM);
                TCOLARGMAX(chunkArg, dataTile_rM);
            }
            TCMP<CmpMode::GT>(predCmp, chunkMax, bestKey);
            TADDS(gIdxTile, chunkArg, static_cast<uint32_t>(Mb * tM));
            TSEL(bestKey, predCmp, chunkMax);
            TSEL(bestIdx, predCmp, gIdxTile);
        }

        if constexpr (std::is_same_v<idtype, int32_t>) {
            auto gO = gOIterU32(0, jc);
            TSTORE(gO, bestIdx);
        } else {
            TCVT(outI64Tile, bestIdx);
            auto gO = gOIterI64(0, jc);
            TSTORE(gO, outI64Tile);
        }
    }
}

template <typename dtype, typename idtype, const int gIM, const int gIN,
          const int tM, const int tN>
void argmaxv2_row(dtype *in_ptr, idtype *out_ptr) {
    // int64 输入在此被 static_assert 拒绝（tile-op 层无 64 位归约/比较）。
    argmaxv2_row_impl<dtype, idtype, gIM, gIN, tM, tN>(in_ptr, out_ptr);
}

template <typename dtype, typename idtype, const int gIM, const int gIN,
          const int tM, const int tN>
void argmaxv2_col(dtype *in_ptr, idtype *out_ptr, int colBegin, int colEnd) {
    argmaxv2_col_impl<dtype, idtype, gIM, gIN, tM, tN>(in_ptr, out_ptr,
                                                       colBegin, colEnd);
}

#endif  // ARGMAXV2_DETAIL_KERNEL_HPP
