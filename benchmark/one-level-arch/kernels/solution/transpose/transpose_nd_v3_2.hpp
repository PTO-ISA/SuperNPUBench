#pragma once

/**
 * @file transpose_nd_v3_2.hpp
 * @brief v3.2：v3.1 的场景分派 + 动态 shape
 *
 * 与 v3.1 的差异：`rank / in_shape / out_shape / perm / numel` 全部**运行时**决定，
 * 编译期只固定容量 `CapElements`、`TileElements`、`PeCount`。
 * 轴规范化（RemoveAxisV2 / MergeAxisV2）与场景路由（TENSOR_MOVE / N_LAST / GATHER）
 * 都在**运行时**完成，但三条路径的机制与 v3.1 一致：
 *   TENSOR_MOVE : 连续 TLOAD/TSTORE（纯拷贝）
 *   N_LAST      : 尾轴连续 → 按行整行搬移
 *   GATHER      : 尾轴转置 → MGATHER（运行时 rank 循环算元素下标）
 *
 * 与 v3.1 对照：
 *   v3.1: rank/shape/perm 编译期（Spec 模板包），路由在编译期 select；
 *   v3.2: 同上路由规则，但全部下推到运行时。
 *
 * 数据流：GM -> UB -> GM。
 */

#include <common/pto_tileop.hpp>
#include "basic_op/utils/spmd_partition.hpp"

#include <array>
#include <cstdint>

namespace supernpu::tile_isa {

namespace v32_detail {

constexpr int kMaxAxis = 8;
constexpr int kMoveAlignLastMinEle = 32;                     // MOVEALIGN_LAST_MIN_ELE
constexpr unsigned long long kSmallShapeThres = 4000000ULL;  // SMALL_SHAPE_BYTES_THRES_HOLD

enum class Scenario { TensorMove, NLast, Gather };

struct RShape {
    int dim = 0;
    std::array<int, kMaxAxis> inShape{};
    std::array<int, kMaxAxis> perm{};
};

// RemoveAxisV2：删除 size==1 的轴并重映射 perm（运行时版，与 v3.1 constexpr 版等价）
inline RShape RemoveAxisRuntime(const RShape &s) {
    if (s.dim == 1) return s;
    bool allOne = true;
    for (int i = 0; i < s.dim; ++i) {
        if (s.inShape[i] != 1) { allOne = false; break; }
    }
    RShape r{};
    if (allOne) { r.dim = 1; r.inShape[0] = 1; r.perm[0] = 0; return r; }

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
    for (int i = 1; i < delSize; ++i) {  // 降序插入排序
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

// MergeAxisV2：合并 perm 中连续的轴（运行时版）
inline RShape MergeAxisRuntime(const RShape &s) {
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
    RShape r{};
    r.dim = newDim;
    for (int i = 0; i < newDim; ++i) { r.inShape[i] = newShape[i]; r.perm[i] = newPerm[i]; }
    return r;
}

// 运行时路由（与 v3.1 SelectScenario 同规则）
inline Scenario SelectScenarioRuntime(const RShape &raw, unsigned long long eleBytes) {
    RShape s = MergeAxisRuntime(RemoveAxisRuntime(raw));
    if (s.dim == 1) return Scenario::TensorMove;
    const int last = s.dim - 1;
    const bool isLastAxisTranspose = (s.perm[last] != last);
    if (isLastAxisTranspose) return Scenario::Gather;
    unsigned long long vol = 1;
    for (int i = 0; i < s.dim; ++i) vol *= static_cast<unsigned long long>(s.inShape[i]);
    const unsigned long long bytes = vol * eleBytes;
    if (s.inShape[last] >= kMoveAlignLastMinEle && bytes >= kSmallShapeThres) return Scenario::NLast;
    return Scenario::Gather;  // 其余（NDDMA 等）回退通用 MGATHER
}

}  // namespace v32_detail

// ============================ 策略实现 ============================

// #1 TENSOR_MOVE：连续拷贝（numel 个元素，按 TileElements 切块）
// 尾块用"重叠整块"技巧：最后一个不满块改从 (numel - TileElements) 起整块拷贝，
// 避免需要运行时长度的 TLOAD/TSTORE（重叠部分写入相同值）。
template <typename DType, int CapElements, int TileElements>
void v32_tensor_move(DType *input, DType *output, std::uint32_t numel, int tile_start,
                     int tile_stride) {
    using Global = pto::global_tensor<DType, pto::RowMajor<1, CapElements>>;
    using DataTile = pto::Tile<pto::Location::Vec, DType, 1, TileElements, pto::BLayout::RowMajor>;
    using Iter = global_iterator<Global, DataTile>;
    Iter in_iter(input), out_iter(output);

    const int kNumTiles = static_cast<int>((numel + TileElements - 1) / TileElements);
    for (int t = tile_start; t < kNumTiles; t += tile_stride) {
        int off = t * TileElements;
        if (off + TileElements > static_cast<int>(numel)) {  // 尾块：重叠对齐到末尾
            off = static_cast<int>(numel) - TileElements;
            if (off < 0) off = 0;
        }
        DataTile d;
        auto gi = in_iter(0, off / TileElements);
        auto go = out_iter(0, off / TileElements);
        TLOAD(d, gi);
        TSTORE(go, d);
    }
}

// #5 N_LAST：尾轴连续 → 按行搬移（运行时前缀）。行 = numel / L。
// 行内按编译期 RowTileElements(128) 整块拷贝；要求 L % RowTileElements == 0
// （否则入口回退 GATHER，保证正确性）。
template <typename DType, int CapElements, int TileElements>
void v32_n_last(DType *input, DType *output, const v32_detail::RShape &red, std::uint32_t numel,
                int row_start, int row_stride) {
    constexpr int kRowTE = 128;  // 行内整块宽度（元素）
    const int NumPrefix = red.dim - 1;
    const int L = red.inShape[red.dim - 1];
    const int NumRows = static_cast<int>(numel) / L;

    std::array<long long, v32_detail::kMaxAxis> inPrefixStride{};
    {
        long long acc = 1;
        for (int k = NumPrefix - 1; k >= 0; --k) { inPrefixStride[k] = acc; acc *= red.inShape[k]; }
    }
    std::array<int, v32_detail::kMaxAxis> outPrefixShape{};
    std::array<long long, v32_detail::kMaxAxis> outPrefixStride{};
    {
        for (int i = 0; i < NumPrefix; ++i) outPrefixShape[i] = red.inShape[red.perm[i]];
        long long acc = 1;
        for (int k = NumPrefix - 1; k >= 0; --k) { outPrefixStride[k] = acc; acc *= outPrefixShape[k]; }
    }

    using Global = pto::global_tensor<DType, pto::RowMajor<1, CapElements>>;
    using RowTile = pto::Tile<pto::Location::Vec, DType, 1, kRowTE, pto::BLayout::RowMajor>;
    using RowIter = global_iterator<Global, RowTile>;
    const int kChunks = L / kRowTE;

    for (int r_out = row_start; r_out < NumRows; r_out += row_stride) {
        long long rem = r_out;
        long long r_in = 0;
        for (int i = 0; i < NumPrefix; ++i) {
            const int coord = static_cast<int>(rem / outPrefixStride[i]);
            rem -= static_cast<long long>(coord) * outPrefixStride[i];
            r_in += static_cast<long long>(coord) * inPrefixStride[red.perm[i]];
        }
        RowIter in_iter(input + r_in * L);
        RowIter out_iter(output + static_cast<long long>(r_out) * L);
        for (int c = 0; c < kChunks; ++c) {
            RowTile d;
            auto gi = in_iter(0, c);
            auto go = out_iter(0, c);
            TLOAD(d, gi);
            TSTORE(go, d);
        }
    }
}

// #7 GATHER：尾轴转置 → 运行时 rank 循环算元素下标 + MGATHER
template <typename DType, int CapElements, int TileElements, std::uint32_t PeCount>
void v32_gather(DType *input, DType *output, int rank, const std::uint32_t *in_shape,
                const std::uint32_t *out_shape, const std::uint32_t *inv_perm,
                std::uint32_t numel, int pe_id) {
    using InputGlobal = pto::global_tensor<DType, pto::RowMajor<1, CapElements>>;
    using OutputGlobal = pto::global_tensor<DType, pto::RowMajor<1, CapElements>>;
    using DataTile = pto::Tile<pto::Location::Vec, DType, 1, TileElements, pto::BLayout::RowMajor>;
    using OffsetTile = pto::Tile<pto::Location::Vec, std::uint32_t, 1, TileElements,
                                 pto::BLayout::RowMajor>;
    using OutputIterator = global_iterator<OutputGlobal, DataTile>;

    InputGlobal input_global(input);
    OutputIterator output_iter(output);

    const std::uint32_t kTiles =
        (numel + static_cast<std::uint32_t>(TileElements) - 1) /
        static_cast<std::uint32_t>(TileElements);

    for (std::uint32_t t = static_cast<std::uint32_t>(pe_id); t < kTiles;
         t += static_cast<std::uint32_t>(PeCount)) {
        DataTile dst;
        OffsetTile lin, lim, quotient, cycle, cycle_base, coordinate, contribution, off_tile;
        TCI(lin, t * static_cast<std::uint32_t>(TileElements));
        TEXPANDS(lim, numel - 1u);
        TMIN(lin, lin, lim);  // clamp，避免尾块越界读
        TEXPANDS(off_tile, static_cast<std::uint32_t>(0));

        std::uint32_t in_stride = 1;
        for (int input_dim = rank - 1; input_dim >= 0; --input_dim) {
            const int output_dim = static_cast<int>(inv_perm[input_dim]);
            std::uint32_t out_stride = 1;
            for (int dim = output_dim + 1; dim < rank; ++dim) out_stride *= out_shape[dim];
            if (out_stride == 1) TCVT(quotient, lin);
            else TDIVS(quotient, lin, out_stride);
            TDIVS(cycle, quotient, out_shape[output_dim]);
            TMULS(cycle_base, cycle, out_shape[output_dim]);
            TSUB(coordinate, quotient, cycle_base);
            TMULS(contribution, coordinate,
                  in_stride * static_cast<std::uint32_t>(sizeof(DType)));
            TADD(off_tile, off_tile, contribution);
            in_stride *= in_shape[input_dim];
        }
        auto out_global = output_iter(0, static_cast<int>(t));
        MGATHER(dst, input_global, off_tile);
        TSTORE(out_global, dst);
    }
}

// ============================ 入口 ============================

/**
 * @brief v3.2：动态 shape 的场景分派转置（单执行单元）。
 * rank / in_shape / out_shape / perm / numel 运行时；CapElements 为容量上界。
 */
template <typename DType, int CapElements, int TileElements = 512,
          std::uint32_t PeCount = supernpu::multi_thread::kDefaultPeCount>
void transpose_nd_v3_2_tile(DType *input, DType *output, int rank,
                            const std::uint32_t *in_shape, const std::uint32_t *out_shape,
                            const std::uint32_t *perm, std::uint32_t numel, int pe_id) {
    static_assert(CapElements > 0 && TileElements > 0 &&
                      TileElements * static_cast<int>(sizeof(DType)) % 32 == 0,
                  "Tile 行必须 32 字节对齐");
    static_assert(static_cast<unsigned long long>(CapElements) * sizeof(DType) <= 0x100000000ULL,
                  "offset 使用 uint32");

    // 运行时逆置换：inv_perm[input_dim] = output_dim
    std::uint32_t inv_perm[v32_detail::kMaxAxis] = {0, 1, 2, 3, 4, 5, 6, 7};
    for (int i = 0; i < rank; ++i) inv_perm[perm[i]] = static_cast<std::uint32_t>(i);

    // 运行时轴规范化 + 路由
    v32_detail::RShape raw{};
    raw.dim = rank;
    for (int i = 0; i < rank; ++i) {
        raw.inShape[i] = static_cast<int>(in_shape[i]);
        raw.perm[i] = static_cast<int>(perm[i]);
    }
    const v32_detail::RShape red =
        v32_detail::MergeAxisRuntime(v32_detail::RemoveAxisRuntime(raw));
    const v32_detail::Scenario sc =
        v32_detail::SelectScenarioRuntime(raw, static_cast<unsigned long long>(sizeof(DType)));

    if (sc == v32_detail::Scenario::TensorMove) {
        v32_tensor_move<DType, CapElements, TileElements>(input, output, numel, pe_id,
                                                          static_cast<int>(PeCount));
    } else if (sc == v32_detail::Scenario::NLast && red.dim >= 2 &&
               (red.inShape[red.dim - 1] % 128) == 0) {
        v32_n_last<DType, CapElements, TileElements>(input, output, red, numel, pe_id,
                                                     static_cast<int>(PeCount));
    } else {
        // Gather：用 reduced 后的 (rank/perm/shape) 做索引（与 v3.1 一致，先规范化再 gather）
        std::uint32_t r_in_shape[v32_detail::kMaxAxis]{};
        std::uint32_t r_out_shape[v32_detail::kMaxAxis]{};
        std::uint32_t r_perm[v32_detail::kMaxAxis]{};
        std::uint32_t r_inv[v32_detail::kMaxAxis] = {0, 1, 2, 3, 4, 5, 6, 7};
        for (int i = 0; i < red.dim; ++i) {
            r_in_shape[i] = static_cast<std::uint32_t>(red.inShape[i]);
            r_perm[i] = static_cast<std::uint32_t>(red.perm[i]);
            r_out_shape[i] = static_cast<std::uint32_t>(red.inShape[red.perm[i]]);
        }
        for (int i = 0; i < red.dim; ++i) r_inv[r_perm[i]] = static_cast<std::uint32_t>(i);
        v32_gather<DType, CapElements, TileElements, PeCount>(
            input, output, red.dim, r_in_shape, r_out_shape, r_inv, numel, pe_id);
    }
}

}  // namespace supernpu::tile_isa

namespace supernpu::multi_thread {

/** @brief v3.2 多 PE 包装（输出 tile round-robin 分给各 PE）。 */
template <typename DType, int CapElements, int TileElements = 512,
          std::uint32_t PeCount = kDefaultPeCount>
void transpose_nd_v3_2(DType *input, DType *output, int rank,
                       const std::uint32_t *in_shape, const std::uint32_t *out_shape,
                       const std::uint32_t *perm, std::uint32_t numel) {
    static_assert(PeCount > 0);
    const int tid = static_cast<int>(get_thread_idx());
    supernpu::tile_isa::transpose_nd_v3_2_tile<DType, CapElements, TileElements, PeCount>(
        input, output, rank, in_shape, out_shape, perm, numel, tid);
}

}  // namespace supernpu::multi_thread
