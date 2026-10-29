// Static-shape 4PE wrapper: shape/axis are compile-time template parameters.
// Tile shapes are fixed inside from the dtype traits — the entry takes no
// tile parameters (normalization static convention). Shapes the compile-time
// partition cannot express (outer rows / column windows not divisible) are
// auto-delegated to the runtime-tiling counterpart argmaxv2_dynamic.hpp
// (ceil partition), so no PE0-solo fallback remains.
#pragma once

#include "basic_op/utils/spmd_partition.hpp"
#include "basic_op/argmaxv2/detail/argmaxv2.hpp"
#include "basic_op/argmaxv2/argmaxv2_dynamic.hpp"
#include "basic_op/argmaxv2/argmaxv2_rsplit.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace supernpu::multi_thread {

namespace argmaxv2_shape {

constexpr inline int prod_range(const int *arr, int lo, int hi) {
    int p = 1;
    for (int i = lo; i < hi; ++i) p *= arr[i];
    return p;
}

// 编译期 shape 折叠：[d0..dk-1, dk, dk+1..dn-1] →
//   kOuter = prod(d0..dk-1)（非归约前缀）、kAxisDim = dk（归约维）、
//   kInner = prod(dk+1..dn-1)（非归约后缀）。
template <int Axis, int... Dims>
struct ArgMaxNdShape {
    static constexpr int kRank = sizeof...(Dims);
    static_assert(kRank >= 1, "at least one dimension is required");
    static_assert(((Dims > 0) && ...), "dimensions must be positive");
    static constexpr int kAxis = Axis < 0 ? Axis + kRank : Axis;
    static_assert(kAxis >= 0 && kAxis < kRank, "axis out of range");
    static constexpr int kArr[sizeof...(Dims)] = {Dims...};
    static constexpr int kOuter = prod_range(kArr, 0, kAxis);
    static constexpr int kAxisDim = kArr[kAxis];
    static constexpr int kInner = prod_range(kArr, kAxis + 1, kRank);
};

}  // namespace argmaxv2_shape

// ----------------------------------------------------------------------------
// ArgMaxV2 四 PE SPMD 包装（功能对齐 CANN ArgMaxV2 / aclnnArgMax）：
// 唯一入口，任意维度。shape 为编译期模板参数包 Dims...，axis 支持负值
// （-rank .. rank-1）。内部做编译期折叠（CANN ArgOpsTiling 同思路：
// first_dim × axis × last_dim 三段化），复用 detail 的 2D 行/列归约：
//
//   - axis 为最后一维（含 1D、2D dim=1/dim=-1）：非归约维折叠为行 →
//     [kOuter, kAxisDim] 行归约。R 大 A 小（ceil 行/PE 的 4 倍 < tM 且
//     kAxisDim >= PeCount*tM）时走切 R 模板 argmaxv2_rsplit（段内 reshape
//     满载 + 跨 PE 合并，对标 CANN GROUP_REDUCE）；否则 kOuter 整除 PE 数
//     时按行 SPMD 分区（编译期均分 + 静态 detail 特化）；不整除时自动转
//     动态入口 argmaxv2_dynamic（均衡分区：地板除 + 余数给前几个 PE，
//     g_outer >= PeCount 时全部 PE 参与），不再有 PE0 单跑兜底
//     （含 1 维输入等场景）。
//   - axis 为中间/首维（含 2D dim=0/dim=-2）：逐 kOuter 切片做
//     [kAxisDim, kInner] 列归约，每 PE 按列窗口分区（outer 为循环维度
//     不受 PE 整除约束）。kInner % (PeCount*tN) == 0（窗口块对齐）时
//     走静态特化；否则同样自动转动态入口（列块 ceil 分区、任意尾数，
//     原先的 static_assert 编译失败改为自动路由）。
//
//   物理 tile 形状按 dtype traits 固定（行/列归约各一套默认形状），
//   入口不接收 tile 参数（对齐 normalization 静态版约定：Kernel entry
//   points do not accept runtime tiling）。
//
// 语义（与 detail 一致，对齐 CANN/PyTorch）：平局返回首个最大值索引；
// fp16/fp32/bf16 的 NaN 视为最大值（首个 NaN 索引）；±0.0 相等。
// 归约维长度为 1 时输出全 0。输出 layout：去掉 axis 维后的 RowMajor
// 连续张量（keepdim 由上层实现）。
//
// 输入 dtype 支持 fp16/fp32/bf16/int32；输出 dtype 为 int32/int64。
// ----------------------------------------------------------------------------
template <typename DType, typename IdType, int Axis, int... Dims,
          std::uint32_t PeCount = kDefaultPeCount>
void argmaxv2(DType *input, IdType *output) {
    static_assert(!std::is_same_v<DType, int64_t>,
                  "int64 input is not supported: the tile-op layer has no "
                  "64-bit reduce/compare instructions (TROWARGMAX/TCOLARGMAX "
                  "contracts and the model reduce/compare dtype whitelist "
                  "cap at 32-bit)");
    using S = argmaxv2_shape::ArgMaxNdShape<Axis, Dims...>;

    if constexpr (S::kAxis == S::kRank - 1) {
        // 尾轴：非归约维折叠为行 → [kOuter, kAxisDim] 行归约
        constexpr int kTileM = argmaxv2_detail::ArgMaxTraits<DType>::kRowTileM;
        constexpr int kTileN = argmaxv2_detail::ArgMaxTraits<DType>::kRowTileN;
        // 切 R 路由（对标 CANN GROUP_REDUCE，argmaxv2_rsplit.hpp）：R 大
        // A 小。三个条件（A/B 实测定界）：
        //   (1) ceil 行/PE 的 4 倍 < tM——行分区喂不满行 tile；
        //   (2) R >= PeCount*tM——每 PE 分段非空且 3*seg < R；
        //   (3) A < PeCount（行分区必有 PE 空转，切 R 靠并行度取胜，如 1D）
        //       或 R >= PeCount*tM*tN/2（段 >= 半个满载行块——reshape 利用
        //       率优势才摊得平屏障 + 落盘固定开销；小 R 大 A 实测反
        //       退化，一并排除）。
        constexpr int kRowsPerPeCeil =
            (S::kOuter + static_cast<int>(PeCount) - 1) /
            static_cast<int>(PeCount);
        // A/B 验证开关：-DARGMAXV2_DISABLE_RSPLIT 强制回落既有路径
        // （整除 SPMD / 动态委托），供路由回归对比与再定界复测。
#ifdef ARGMAXV2_DISABLE_RSPLIT
        constexpr bool kUseRsplit = false;
#else
        constexpr bool kUseRsplit =
            kRowsPerPeCeil * 4 < kTileM &&
            S::kAxisDim >= static_cast<int>(PeCount) * kTileM &&
            (S::kOuter < static_cast<int>(PeCount) ||
             S::kAxisDim >=
                 static_cast<int>(PeCount) * kTileM * kTileN / 2);
#endif
        if constexpr (kUseRsplit) {
            // 切 R 块形（tile 形扫描定界）：4B key 且最大段 <= 1024 =
            // 16*64 时用 [64,8] 块——W 收缩到 16 ⇒ 主视图单行块、尾档
            // A 消除；2B key（tM 已 64）与长段（尾档结构同构）保持
            // traits 缺省。
            using RKeyT =
                typename argmaxv2_detail::ArgMaxTraits<DType>::KeyT;
            constexpr int64_t kRSeg =
                ((static_cast<int64_t>(S::kAxisDim) +
                  static_cast<int64_t>(PeCount) - 1) /
                 static_cast<int64_t>(PeCount)) &
                ~static_cast<int64_t>(15);
            constexpr int64_t kRSMax =
                kRSeg > static_cast<int64_t>(S::kAxisDim) -
                                (static_cast<int64_t>(PeCount) - 1) * kRSeg
                    ? kRSeg
                    : static_cast<int64_t>(S::kAxisDim) -
                          (static_cast<int64_t>(PeCount) - 1) * kRSeg;
            constexpr int kRTileM =
                sizeof(RKeyT) == 4 && kRSMax <= 1024 ? 64 : 0;
            constexpr int kRTileN = kRTileM > 0 ? 8 : 0;
            argmaxv2_rsplit<DType, IdType, S::kOuter, S::kAxisDim, PeCount,
                            kRTileM, kRTileN>(input, output);
        } else if constexpr (S::kOuter % static_cast<int>(PeCount) == 0) {
            using Partition = ContiguousPartition<
                static_cast<std::size_t>(S::kOuter), PeCount>;
            constexpr int kRowsPerPe = static_cast<int>(Partition::kItemsPerPe);
            const std::size_t row_offset = Partition::item_offset();

            ::argmaxv2_row<DType, IdType, kRowsPerPe, S::kAxisDim, kTileM,
                           kTileN>(input + row_offset * S::kAxisDim,
                                   output + row_offset);
        } else {
            // 外层行数不整除 PE 数（含 1 维输入）：自动转动态入口。
            // 均衡分区（地板除 + 余数给前几个 PE）替代原 PE0 单跑兜底。
            const argmaxv2_dyn::ArgMaxV2TilingData tiling{
                S::kOuter, S::kAxisDim, 1, 0, 0, 1};
            argmaxv2_dynamic<DType, IdType, PeCount>(input, output, &tiling);
        }
    } else {
        // 中间/首轴：逐 outer 切片 [kAxisDim, kInner] 列归约
        constexpr int kTileM = argmaxv2_detail::ArgMaxTraits<DType>::kColTileM;
        constexpr int kTileN = argmaxv2_detail::ArgMaxTraits<DType>::kColTileN;
        // 静态特化前提：每 PE 列窗口与列块对齐（蕴含 kInner % tN == 0）。
        constexpr bool kWindowAligned =
            S::kInner % static_cast<int>(PeCount) == 0 &&
            (S::kInner / static_cast<int>(PeCount)) % kTileN == 0;
        if constexpr (kWindowAligned) {
            using Partition = ContiguousPartition<
                static_cast<std::size_t>(S::kInner), PeCount>;
            constexpr int kColsPerPe = static_cast<int>(Partition::kItemsPerPe);
            static_assert(S::kInner % kTileN == 0,
                          "the inner dimension must be divisible by the column "
                          "tile width");
            static_assert(kColsPerPe % kTileN == 0,
                          "the per-PE column window must be tile aligned");

            const std::size_t col_offset = Partition::item_offset();
            const int colBegin = static_cast<int>(col_offset);
            for (int o = 0; o < S::kOuter; ++o) {
                ::argmaxv2_col<DType, IdType, S::kAxisDim, S::kInner, kTileM,
                               kTileN>(
                    input + static_cast<std::size_t>(o) * S::kAxisDim *
                                S::kInner,
                    output + static_cast<std::size_t>(o) * S::kInner, colBegin,
                    colBegin + kColsPerPe);
            }
        } else {
            // 列窗口无法块对齐（kInner 不满足 % (PeCount*tN)）：自动转
            // 动态入口，列块 ceil 分区、任意尾数（原先为编译失败）。
            const argmaxv2_dyn::ArgMaxV2TilingData tiling{
                S::kOuter, S::kAxisDim, S::kInner, 0, 0, 0};
            argmaxv2_dynamic<DType, IdType, PeCount>(input, output, &tiling);
        }
    }
}

}  // namespace supernpu::multi_thread
