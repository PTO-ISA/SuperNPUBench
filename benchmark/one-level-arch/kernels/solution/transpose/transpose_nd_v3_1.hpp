#pragma once

/**
 * @file transpose_nd_v3_1.hpp
 * @brief 场景分派转置 v3.1：实现 ops-math transpose 的 #1/#5/#7 场景
 *
 * 参考 `SuperScalar/ops-math-transpose-reference.md` 与
 * `ops-math/conversion/transpose/op_host/arch35/transpose_tiling_arch35.cpp` 的路由：
 *
 *   1. 轴规范化：RemoveAxisV2（删 size=1 轴）+ MergeAxisV2（合并 perm 连续轴）
 *   2. 路由（与 ops-math 一致，取其中可实现的 3 个）：
 *        dim == 1                                  → TENSOR_MOVE   (#1, 10000)
 *        isLastAxisTranspose                       → GATHER        (#7, 10006)
 *        !isLastAxisTranspose && 尾轴≥32 && bytes≥阈值 → N_LAST    (#5, 10004)
 *        其余（NDDMA 等）                          → 回退 GATHER（通用 MGATHER）
 *
 *   isLastAxisTranspose = reducedPerm[dim-1] != dim-1（尾轴是否参与转置）
 *   N_LAST 阈值 = 4000000 字节（SMALL_SHAPE_BYTES_THRES_HOLD 默认）；尾轴下限 32（MOVEALIGN_LAST_MIN_ELE）
 *
 * 场景在 PTO ISA 的机制：
 *   TENSOR_MOVE : 连续 `TLOAD/TSTORE`（纯拷贝）
 *   N_LAST      : 尾轴连续 → **按行搬移**（每输出行取对应输入行，整行连续拷贝）
 *   GATHER      : 尾轴转置 → `MGATHER` 通用（等价 v3）
 *
 * 接口（shape/perm 均为编译期）：
 *   transpose_nd_v3_1<DType, Rank, TileElements, Spec...>(input, output)
 *   Spec = 前 Rank 个为各维尺寸，后 Rank 个为 perm。
 *
 * 状态：草案。
 */

#include <common/pto_tileop.hpp>
#include "basic_op/utils/spmd_partition.hpp"

#include <array>
#include <cstdint>

namespace supernpu::tile_isa {

enum class Scenario { TensorMove, NLast, Gather };

namespace v31_detail {

constexpr int kMaxAxis = 8;
constexpr int kMoveAlignLastMinEle = 32;        // MOVEALIGN_LAST_MIN_ELE
constexpr unsigned long long kSmallShapeThres = 4000000ULL;  // SMALL_SHAPE_BYTES_THRES_HOLD

template <int... V>
constexpr std::array<int, sizeof...(V)> Arr() {
    return {V...};
}

struct ShapeInfo {
    int dim = 0;
    std::array<int, kMaxAxis> inShape{};
    std::array<int, kMaxAxis> perm{};
};

// RemoveAxisV2：删除 size==1 的轴，重映射 perm
constexpr ShapeInfo RemoveAxis(ShapeInfo s) {
    if (s.dim == 1) return s;
    bool allOne = true;
    for (int i = 0; i < s.dim; ++i) {
        if (s.inShape[i] != 1) { allOne = false; break; }
    }
    ShapeInfo r{};
    if (allOne) {
        r.dim = 1; r.inShape[0] = 1; r.perm[0] = 0;
        return r;
    }
    std::array<int, kMaxAxis> shape{};
    std::array<int, kMaxAxis> delPerm{};
    int shapeSize = 0, delSize = 0;
    for (int i = 0; i < s.dim; ++i) {
        if (s.inShape[i] != 1) {
            shape[shapeSize++] = s.inShape[i];
        } else {
            for (int j = 0; j < s.dim; ++j) {
                if (s.perm[j] == i) delPerm[delSize++] = s.perm[j];
            }
        }
    }
    // 降序插入排序
    for (int i = 1; i < delSize; ++i) {
        int key = delPerm[i], j = i - 1;
        while (j >= 0 && delPerm[j] < key) { delPerm[j + 1] = delPerm[j]; --j; }
        delPerm[j + 1] = key;
    }
    std::array<int, kMaxAxis> newPerm{};
    int newSize = 0;
    for (int i = 0; i < s.dim; ++i) {
        bool del = false;
        for (int j = 0; j < delSize; ++j) if (s.perm[i] == delPerm[j]) del = true;
        if (!del) newPerm[newSize++] = s.perm[i];
    }
    for (int i = 0; i < delSize; ++i) {
        for (int j = 0; j < newSize; ++j) if (newPerm[j] > delPerm[i]) --newPerm[j];
    }
    r.dim = newSize;
    for (int i = 0; i < shapeSize; ++i) r.inShape[i] = shape[i];
    for (int i = 0; i < newSize; ++i) r.perm[i] = newPerm[i];
    return r;
}

// MergeAxisV2：合并 perm 中连续的轴
constexpr ShapeInfo MergeAxis(ShapeInfo s) {
    if (s.dim == 1) return s;
    auto perm = s.perm;
    auto shape = s.inShape;
    std::array<int, kMaxAxis> mergedShape{};
    std::array<int, kMaxAxis> newDimPosition{};
    for (int i = 0; i < kMaxAxis; ++i) newDimPosition[i] = -1;
    int curHead = perm[0];
    newDimPosition[curHead] = 0;
    mergedShape[0] = shape[curHead];
    int dimIndex = 0;
    for (int pi = 1; pi < s.dim; ++pi) {
        if (curHead + 1 == perm[pi]) {
            curHead = perm[pi];
            mergedShape[dimIndex] *= shape[curHead];
        } else {
            curHead = perm[pi];
            ++dimIndex;
            newDimPosition[curHead] = dimIndex;
            mergedShape[dimIndex] = shape[curHead];
        }
    }
    const int newDim = dimIndex + 1;
    // 压缩 newDimPosition（去 -1），保留值
    int di = 0;
    for (int i = 0; i < s.dim; ++i) if (newDimPosition[i] >= 0) newDimPosition[di++] = newDimPosition[i];

    std::array<int, kMaxAxis> newPerm{}, newShape{};
    di = 0;
    for (int i = 0; i < s.dim; ++i) {
        if (newDimPosition[i] >= 0) {
            const int np = newDimPosition[i];
            for (int j = 0; j < s.dim; ++j) if (newDimPosition[j] == i) { newPerm[di] = j; break; }
            newShape[di] = mergedShape[np];
            ++di;
        }
    }
    ShapeInfo r{};
    r.dim = newDim;
    for (int i = 0; i < newDim; ++i) { r.inShape[i] = newShape[i]; r.perm[i] = newPerm[i]; }
    return r;
}

// 规范化 → 路由
constexpr Scenario SelectScenario(ShapeInfo s, unsigned long long eleBytes) {
    s = MergeAxis(RemoveAxis(s));
    if (s.dim == 1) return Scenario::TensorMove;
    const int last = s.dim - 1;
    const bool isLastAxisTranspose = (s.perm[last] != last);
    unsigned long long vol = 1;
    for (int i = 0; i < s.dim; ++i) vol *= static_cast<unsigned long long>(s.inShape[i]);
    const unsigned long long bytes = vol * eleBytes;
    // ops-math: GATHER 先于决策树尝试（尾轴转置场景）
    if (isLastAxisTranspose) return Scenario::Gather;
    if (s.inShape[last] >= kMoveAlignLastMinEle && bytes >= kSmallShapeThres) return Scenario::NLast;
    return Scenario::Gather;  // 其余（NDDMA 等）回退通用 MGATHER
}

// 从 Spec 包中取 [Offset, Offset+Rank) 段
template <int Rank, int Offset, int... Spec>
constexpr std::array<int, kMaxAxis> PickSpec() {
    auto sp = Arr<Spec...>();
    std::array<int, kMaxAxis> a{};
    for (int i = 0; i < Rank; ++i) a[i] = sp[Offset + i];
    return a;
}

}  // namespace v31_detail

// ============================ 策略实现 ============================

// #1 TENSOR_MOVE：连续拷贝（Elements 个元素，按 TileElements 切块）
template <typename DType, int Elements, int TileElements>
void v31_tensor_move(DType *input, DType *output, int tile_start, int tile_stride, bool handle_tail) {
    using Global = pto::global_tensor<DType, pto::RowMajor<1, Elements>>;
    using DataTile = pto::Tile<pto::Location::Vec, DType, 1, TileElements, pto::BLayout::RowMajor>;
    using Iter = global_iterator<Global, DataTile>;
    Iter in_iter(input), out_iter(output);
    constexpr int kFull = Elements / TileElements;
    constexpr int kTail = Elements % TileElements;
    for (int t = tile_start; t < kFull; t += tile_stride) {
        DataTile d;
        auto gi = in_iter(0, t);
        auto go = out_iter(0, t);
        TLOAD(d, gi);
        TSTORE(go, d);
    }
    if constexpr (kTail != 0) {
        if (handle_tail) {
            using TailTile = pto::Tile<pto::Location::Vec, DType, 1, TileElements,
                                       pto::BLayout::RowMajor, 1, kTail>;
            using TailIter = global_iterator<Global, TailTile>;
            TailIter ti(input), to(output);
            TailTile d;
            auto gi = ti(0, kFull);
            auto go = to(0, kFull);
            TLOAD(d, gi);
            TSTORE(go, d);
        }
    }
}

// #5 N_LAST：尾轴连续 → 按行搬移。reduced = [prefix dims..., L]
// 每个输出行取对应输入行，整行 L 元素连续拷贝。
template <typename DType, int L, int NumRows, int NumPrefix,
          std::array<int, v31_detail::kMaxAxis> InPrefixShape,
          std::array<int, v31_detail::kMaxAxis> PrefixPerm,
          int TileElements>
void v31_n_last(DType *input, DType *output, int row_start, int row_stride, bool handle_tail) {
    // 输入前缀行距（行单位）
    std::array<long long, v31_detail::kMaxAxis> inPrefixStride{};
    {
        long long acc = 1;
        for (int k = NumPrefix - 1; k >= 0; --k) { inPrefixStride[k] = acc; acc *= InPrefixShape[k]; }
    }
    // 输出前缀 shape：outPrefix[i] = InPrefixShape[PrefixPerm[i]]
    std::array<int, v31_detail::kMaxAxis> outPrefixShape{};
    std::array<long long, v31_detail::kMaxAxis> outPrefixStride{};
    {
        for (int i = 0; i < NumPrefix; ++i) outPrefixShape[i] = InPrefixShape[PrefixPerm[i]];
        long long acc = 1;
        for (int k = NumPrefix - 1; k >= 0; --k) { outPrefixStride[k] = acc; acc *= outPrefixShape[k]; }
    }
    using RowTile = pto::Tile<pto::Location::Vec, DType, 1, L, pto::BLayout::RowMajor>;
    using Global = pto::global_tensor<DType, pto::RowMajor<1, 1>>;  // 占位，用裸指针寻址

    for (int r_out = row_start; r_out < NumRows; r_out += row_stride) {
        // 输出行 → 前缀坐标 → 输入前缀坐标
        long long rem = r_out;
        long long r_in = 0;
        for (int i = 0; i < NumPrefix; ++i) {
            const int coord = static_cast<int>(rem / outPrefixStride[i]);
            rem -= static_cast<long long>(coord) * outPrefixStride[i];
            // 该输出坐标 i 对应输入轴 PrefixPerm[i]
            r_in += static_cast<long long>(coord) * inPrefixStride[PrefixPerm[i]];
        }
        Global in_g(input + r_in * L);
        Global out_g(output + static_cast<long long>(r_out) * L);
        RowTile d;
        TLOAD(d, in_g);
        TSTORE(out_g, d);
    }
    (void)handle_tail;
}

// #7 GATHER：尾轴转置 → 通用 MGATHER（按输出平坦切块，与 v3 同算法）
template <typename DType, int Rank, int Elements, int TileElements,
          std::array<int, v31_detail::kMaxAxis> Dims,
          std::array<int, v31_detail::kMaxAxis> Perm>
void v31_gather(DType *input, DType *output, int tile_start, int tile_stride, bool handle_tail) {
    constexpr auto kInv = [](std::array<int, v31_detail::kMaxAxis> p) {
        std::array<int, v31_detail::kMaxAxis> inv{};
        for (int i = 0; i < Rank; ++i) inv[p[i]] = i;
        return inv;
    }(Perm);
    constexpr int kFull = Elements / TileElements;
    constexpr int kTail = Elements % TileElements;
    using InputGlobal = pto::global_tensor<DType, pto::RowMajor<1, Elements>>;
    using OutputGlobal = pto::global_tensor<DType, pto::RowMajor<1, Elements>>;
    using DataTile = pto::Tile<pto::Location::Vec, DType, 1, TileElements, pto::BLayout::RowMajor>;
    using OffsetTile = pto::Tile<pto::Location::Vec, std::uint32_t, 1, TileElements,
                                 pto::BLayout::RowMajor>;
    using OutputIter = global_iterator<OutputGlobal, DataTile>;
    InputGlobal in_g(input);
    OutputIter out_iter(output);

    for (int t = tile_start; t < kFull; t += tile_stride) {
        DataTile dst;
        OffsetTile lin, quotient, cycle, cycle_base, coordinate, contribution, off_tile;
        TCI(lin, static_cast<std::uint32_t>(t) * TileElements);
        TEXPANDS(off_tile, static_cast<std::uint32_t>(0));
        std::uint32_t in_stride = 1;
        #pragma clang loop unroll(full)
        for (int input_dim = Rank - 1; input_dim >= 0; --input_dim) {
            const int output_dim = kInv[input_dim];
            std::uint32_t out_stride = 1;
            #pragma clang loop unroll(full)
            for (int dim = output_dim + 1; dim < Rank; ++dim) out_stride *= Dims[Perm[dim]];
            if (out_stride == 1) TCVT(quotient, lin);
            else TDIVS(quotient, lin, out_stride);
            TDIVS(cycle, quotient, Dims[Perm[output_dim]]);
            TMULS(cycle_base, cycle, Dims[Perm[output_dim]]);
            TSUB(coordinate, quotient, cycle_base);
            TMULS(contribution, coordinate, in_stride * static_cast<std::uint32_t>(sizeof(DType)));
            TADD(off_tile, off_tile, contribution);
            in_stride *= Dims[input_dim];
        }
        auto go = out_iter(0, t);
        MGATHER(dst, in_g, off_tile);
        TSTORE(go, dst);
    }
    if constexpr (kTail != 0) {
        if (handle_tail) {
            using TailData = pto::Tile<pto::Location::Vec, DType, 1, TileElements,
                                       pto::BLayout::RowMajor, 1, kTail>;
            using TailOff = pto::Tile<pto::Location::Vec, std::uint32_t, 1, TileElements,
                                      pto::BLayout::RowMajor, 1, kTail>;
            using TailGlobal = pto::global_tensor<DType, pto::RowMajor<1, Elements>>;
            using TailIter = global_iterator<TailGlobal, TailData>;
            TailIter to(output);
            TailData dst;
            TailOff lin, quotient, cycle, cycle_base, coordinate, contribution, off_tile;
            TCI(lin, static_cast<std::uint32_t>(kFull) * TileElements);
            TEXPANDS(off_tile, static_cast<std::uint32_t>(0));
            std::uint32_t in_stride = 1;
            #pragma clang loop unroll(full)
            for (int input_dim = Rank - 1; input_dim >= 0; --input_dim) {
                const int output_dim = kInv[input_dim];
                std::uint32_t out_stride = 1;
                #pragma clang loop unroll(full)
                for (int dim = output_dim + 1; dim < Rank; ++dim) out_stride *= Dims[Perm[dim]];
                if (out_stride == 1) TCVT(quotient, lin);
                else TDIVS(quotient, lin, out_stride);
                TDIVS(cycle, quotient, Dims[Perm[output_dim]]);
                TMULS(cycle_base, cycle, Dims[Perm[output_dim]]);
                TSUB(coordinate, quotient, cycle_base);
                TMULS(contribution, coordinate, in_stride * static_cast<std::uint32_t>(sizeof(DType)));
                TADD(off_tile, off_tile, contribution);
                in_stride *= Dims[input_dim];
            }
            auto go = to(0, kFull);
            MGATHER(dst, in_g, off_tile);
            TSTORE(go, dst);
        }
    }
}

// ============================ 入口 ============================
template <typename DType, int Rank, int Elements, int TileElements = 512,
          std::uint32_t PeCount = supernpu::multi_thread::kDefaultPeCount, int... Spec>
void transpose_nd_v3_1(DType *input, DType *output) {
    static_assert(sizeof...(Spec) == 2 * Rank, "Spec 必须为 [Rank 个 dim 后接 Rank 个 perm]");
    constexpr auto kDims = v31_detail::PickSpec<Rank, 0, Spec...>();
    constexpr auto kPerm = v31_detail::PickSpec<Rank, Rank, Spec...>();
    constexpr v31_detail::ShapeInfo kRaw{Rank, kDims, kPerm};
    constexpr v31_detail::ShapeInfo kRed = v31_detail::MergeAxis(v31_detail::RemoveAxis(kRaw));
    constexpr Scenario kScenario = v31_detail::SelectScenario(kRaw, sizeof(DType));

    const int tid = static_cast<int>(get_thread_idx());

    if constexpr (kScenario == Scenario::TensorMove) {
        v31_tensor_move<DType, Elements, TileElements>(input, output, tid,
                                                       static_cast<int>(PeCount), tid == 0);
    } else if constexpr (kScenario == Scenario::NLast) {
        // reduced = [prefix dims..., L]；行数 = Elements / L
        constexpr int kNumPrefix = kRed.dim - 1;
        constexpr int kL = kRed.inShape[kRed.dim - 1];
        constexpr int kNumRows = Elements / kL;
        v31_n_last<DType, kL, kNumRows, kNumPrefix, kRed.inShape, kRed.perm, TileElements>(
            input, output, tid, static_cast<int>(PeCount), tid == 0);
    } else {
        v31_gather<DType, Rank, Elements, TileElements, kDims, kPerm>(
            input, output, tid, static_cast<int>(PeCount), tid == 0);
    }
}

}  // namespace supernpu::tile_isa
