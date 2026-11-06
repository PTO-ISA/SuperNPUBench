#pragma once

/**
 * @file transpose_nd_v3_2_1.hpp
 * @brief v3.2.1：Host-tiling 版（对标 ops-math 的 op_host tiling）
 *
 * 与 v3.2 的差异（v3.2 = 设备端运行时做全部推导）：
 *   - 调用方在 **Host** 侧调用 `MakeV32Tiling()`：一次算好
 *     轴规范化（RemoveAxis+MergeAxis）、场景（TENSOR_MOVE/N_LAST/GATHER）、
 *     每维 dim、输入/输出 stride、N_LAST 的前缀 stride 与行数等；
 *   - kernel 只收这个 `V32Tiling` 结构体，**switch 一次**后按参数搬，
 *     不再做 rank 循环、不再重算 stride。
 *
 * 复用 v3.2 的规范化/路由实现（`v32_detail`），保证语义一致。
 */

#include "solution/transpose/transpose_nd_v3_2.hpp"
#include "solution/transpose/transpose_nd_v3_1.hpp"

#include <array>
#include <cstdint>

namespace supernpu::tile_isa {
namespace v321_detail {

constexpr int kMaxAxis = 8;

struct V32Tiling {
    int scenario = 0;  // 0=TensorMove, 1=NLast, 2=Gather（与 v32_detail::Scenario 对齐）
    std::uint32_t numel = 0;
    int numTiles = 0;
    // Gather（按 reduced 后的 rank/dim/stride）
    int rank = 0;
    std::uint32_t dim[kMaxAxis]{};
    std::uint32_t inStride[kMaxAxis]{};
    std::uint32_t outStride[kMaxAxis]{};
    int invPerm[kMaxAxis]{};
    // N_LAST
    int L = 0, numRows = 0, numPrefix = 0;
    long long inPrefixStride[kMaxAxis]{};
    long long outPrefixStride[kMaxAxis]{};
    int prefixPerm[kMaxAxis]{};
};

/** @brief Host 侧：一次算好场景与全部搬运用参数。 */
inline V32Tiling MakeV32Tiling(int rank, const std::uint32_t *in_shape,
                               const std::uint32_t *out_shape, const std::uint32_t *perm,
                               std::uint32_t numel, int tileElements, int dtypeBytes) {
    V32Tiling t{};
    t.numel = numel;
    t.numTiles = static_cast<int>((numel + static_cast<std::uint32_t>(tileElements) - 1) /
                                 static_cast<std::uint32_t>(tileElements));
    t.scenario = static_cast<int>(v32_detail::SelectScenarioRuntime(
        [&] {
            v32_detail::RShape r{};
            r.dim = rank;
            for (int i = 0; i < rank; ++i) { r.inShape[i] = (int)in_shape[i]; r.perm[i] = (int)perm[i]; }
            return r;
        }(),
        static_cast<unsigned long long>(dtypeBytes)));

    const v32_detail::RShape red2 = v32_detail::MergeAxisRuntime(v32_detail::RemoveAxisRuntime(
        [&] {
            v32_detail::RShape r{};
            r.dim = rank;
            for (int i = 0; i < rank; ++i) { r.inShape[i] = (int)in_shape[i]; r.perm[i] = (int)perm[i]; }
            return r;
        }()));

    if (t.scenario == static_cast<int>(v32_detail::Scenario::NLast)) {
        t.numPrefix = red2.dim - 1;
        t.L = red2.inShape[red2.dim - 1];
        t.numRows = static_cast<int>(numel) / t.L;
        long long acc = 1;
        for (int k = t.numPrefix - 1; k >= 0; --k) { t.inPrefixStride[k] = acc; acc *= red2.inShape[k]; }
        long long acc2 = 1;
        for (int k = t.numPrefix - 1; k >= 0; --k) {
            t.outPrefixStride[k] = acc2;
            acc2 *= red2.inShape[red2.perm[k]];
        }
        for (int i = 0; i < t.numPrefix; ++i) t.prefixPerm[i] = red2.perm[i];
    } else if (t.scenario == static_cast<int>(v32_detail::Scenario::Gather)) {
        t.rank = red2.dim;
        std::uint32_t is[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        std::uint32_t acc = 1;
        for (int d = t.rank - 1; d >= 0; --d) { is[d] = acc; acc *= (std::uint32_t)red2.inShape[d]; }
        std::uint32_t os[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        std::uint32_t acc2 = 1;
        for (int d = t.rank - 1; d >= 0; --d) { os[d] = acc2; acc2 *= (std::uint32_t)red2.inShape[red2.perm[d]]; }
        for (int i = 0; i < t.rank; ++i) {
            t.dim[i] = (std::uint32_t)red2.inShape[red2.perm[i]];
            t.inStride[i] = is[i];
            t.outStride[i] = os[i];
        }
        int inv[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        for (int i = 0; i < t.rank; ++i) inv[red2.perm[i]] = i;
        for (int i = 0; i < 8; ++i) t.invPerm[i] = inv[i];
    }
    return t;
}

}  // namespace v321_detail

/** @brief v3.2.1 kernel：只按 Host 算好的参数搬运。 */
template <typename DType, int CapElements, int TileElements = 512,
          std::uint32_t PeCount = supernpu::multi_thread::kDefaultPeCount>
void transpose_nd_v3_2_1(DType *input, DType *output, const v321_detail::V32Tiling &t, int pe_id) {
    using DataTile = pto::Tile<pto::Location::Vec, DType, 1, TileElements, pto::BLayout::RowMajor>;
    using Global = pto::global_tensor<DType, pto::RowMajor<1, CapElements>>;

    if (t.scenario == 0) {  // TENSOR_MOVE：纯连续拷贝
        Global in_g(input), out_g(output);
        using Iter = global_iterator<Global, DataTile>;
        Iter it_in(input), it_out(output);
        (void)in_g; (void)out_g;
        for (int tt = pe_id; tt < t.numTiles; tt += static_cast<int>(PeCount)) {
            DataTile d;
            auto gi = it_in(0, tt);
            auto go = it_out(0, tt);
            TLOAD(d, gi);
            TSTORE(go, d);
        }
        return;
    }

    if (t.scenario == 1) {  // N_LAST：按行搬（前缀 stride 由 Host 算好）
        for (int r_out = pe_id; r_out < t.numRows; r_out += static_cast<int>(PeCount)) {
            long long rem = r_out, r_in = 0;
            for (int i = 0; i < t.numPrefix; ++i) {
                const int coord = static_cast<int>(rem / t.outPrefixStride[i]);
                rem -= (long long)coord * t.outPrefixStride[i];
                r_in += (long long)coord * t.inPrefixStride[t.prefixPerm[i]];
            }
            using RowTile = pto::Tile<pto::Location::Vec, DType, 1, 128, pto::BLayout::RowMajor>;
            using RowIter = global_iterator<Global, RowTile>;
            RowIter ri(input + r_in * t.L), ro(output + (long long)r_out * t.L);
            for (int c = 0; c < t.L / 128; ++c) {
                RowTile d;
                auto gi = ri(0, c);
                auto go = ro(0, c);
                TLOAD(d, gi);
                TSTORE(go, d);
            }
        }
        return;
    }

    // GATHER：dim/inStride/outStride 由 Host 算好，kernel 只做坐标分解
    using InputGlobal = pto::global_tensor<DType, pto::RowMajor<1, CapElements>>;
    using OutputGlobal = pto::global_tensor<DType, pto::RowMajor<1, CapElements>>;
    using OffsetTile = pto::Tile<pto::Location::Vec, std::uint32_t, 1, TileElements,
                                 pto::BLayout::RowMajor>;
    using OutputIter = global_iterator<OutputGlobal, DataTile>;
    InputGlobal in_g(input);
    OutputIter out_iter(output);
    for (int tt = pe_id; tt < t.numTiles; tt += static_cast<int>(PeCount)) {
        DataTile dst;
        OffsetTile lin, q, cyc, cb, coord, contrib, off;
        TCI(lin, (std::uint32_t)tt * TileElements);
        TEXPANDS(off, 0u);
        #pragma clang loop unroll(full)
        for (int d = t.rank - 1; d >= 0; --d) {
            const int od = t.invPerm[d];
            const std::uint32_t os = t.outStride[od];
            if (os == 1) TCVT(q, lin); else TDIVS(q, lin, os);
            TDIVS(cyc, q, t.dim[od]);
            TMULS(cb, cyc, t.dim[od]);
            TSUB(coord, q, cb);
            TMULS(contrib, coord, t.inStride[d] * static_cast<std::uint32_t>(sizeof(DType)));
            TADD(off, off, contrib);
        }
        auto go = out_iter(0, tt);
        MGATHER(dst, in_g, off);
        TSTORE(go, dst);
    }
}


/**
 * @brief v3.2.1 的**编译期特化路径**（小 shape 专用，对标 v3.1）
 *
 * 与 Host-tiling 版互补：shape/perm 编译期已知时，直接在编译期做
 * 轴规范化 + 场景选择，循环边界全为常量（`kFull`）→ 编译器可完全展开、
 * 多个 TLOAD/TSTORE 背靠背发射；配合**大 tile**（默认 2048）进一步减少 op 数。
 *
 * 用法：小 shape 由调用方选这个入口；大/动态 shape 用 `transpose_nd_v3_2_1`。
 */
template <typename DType, int Rank, int Elements, int TileElements,
          std::uint32_t PeCount, std::array<int, 8> Dims, std::array<int, 8> Perm>
void transpose_nd_v3_2_1_ct(DType *input, DType *output, int pe_id) {
    constexpr v31_detail::ShapeInfo kRaw{Rank, Dims, Perm};
    constexpr auto kRed = v31_detail::MergeAxis(v31_detail::RemoveAxis(kRaw));
    constexpr auto kSc = v31_detail::SelectScenario(kRaw, sizeof(DType));
    if constexpr (kSc == Scenario::TensorMove) {
        v31_tensor_move<DType, Elements, TileElements>(input, output, pe_id,
                                                       static_cast<int>(PeCount), pe_id == 0);
    } else if constexpr (kSc == Scenario::NLast) {
        constexpr int kNumPrefix = kRed.dim - 1;
        constexpr int kL = kRed.inShape[kRed.dim - 1];
        constexpr int kNumRows = Elements / kL;
        v31_n_last<DType, kL, kNumRows, kNumPrefix, kRed.inShape, kRed.perm, TileElements>(
            input, output, pe_id, static_cast<int>(PeCount), pe_id == 0);
    } else {
        v31_gather<DType, Rank, Elements, TileElements, Dims, Perm>(
            input, output, pe_id, static_cast<int>(PeCount), pe_id == 0);
    }
}


/**
 * @brief v3.2.1 统一路由入口（编译期路由）
 *
 * 在**编译期**按条件选择实现：
 *   - Rank >= 5（高维散列）        -> host-tiling 版（`transpose_nd_v3_2_1`）
 *   - Elements <= kSmallElements   -> 编译期特化版（`transpose_nd_v3_2_1_ct`，大 tile）
 *   - 其余（大 shape）             -> host-tiling 版（TE=512）
 *
 * @param t     Host 侧 `MakeV32Tiling()` 的结果（Rank<5 且小 shape 时不会用到）
 */
template <typename DType, int Rank, int Elements, int SmallTileElements = 2048,
          std::uint32_t PeCount,
          std::array<int, 8> Dims, std::array<int, 8> Perm>
void transpose_nd_v3_2_1_dispatch(DType *input, DType *output,
                                  const v321_detail::V32Tiling &t, int pe_id) {
    constexpr int kSmallElements = 1 << 16;  // 64K 元素以内走编译期特化
    if constexpr (Rank >= 5) {
        transpose_nd_v3_2_1<DType, Elements, 512, PeCount>(input, output, t, pe_id);
    } else if constexpr (Elements <= kSmallElements) {
        transpose_nd_v3_2_1_ct<DType, Rank, Elements, SmallTileElements, PeCount, Dims, Perm>(
            input, output, pe_id);
    } else {
        transpose_nd_v3_2_1<DType, Elements, 512, PeCount>(input, output, t, pe_id);
    }
}

}  // namespace supernpu::tile_isa
