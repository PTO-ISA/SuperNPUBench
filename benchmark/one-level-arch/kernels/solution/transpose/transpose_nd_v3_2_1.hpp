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
 * 为保持单一文件，轴规范化/路由实现（`v32_detail`）已内置到本文件。
 */

#include <common/pto_tileop.hpp>
#include "basic_op/utils/spmd_partition.hpp"

#include <array>
#include <cstdint>

namespace supernpu::tile_isa {

// ---- 轴规范化 / 场景判定（运行时；原 transpose_nd_v3_2.hpp 的 v32_detail） ----
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

    // N_LAST 要求保留轴长度是 128 的整数倍，否则回退 GATHER。
    // 注意：必须在**填字段之前**改 scenario，否则会留在 N_LAST 分支里、
    // 导致 GATHER 所需的 rank/dim/stride/invPerm 全是 0。
    if (t.scenario == static_cast<int>(v32_detail::Scenario::NLast) &&
        (red2.inShape[red2.dim - 1] % 128) != 0) {
        t.scenario = static_cast<int>(v32_detail::Scenario::Gather);
    }

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
    static_assert(static_cast<unsigned long long>(CapElements) * sizeof(DType) <= 0xFFFFFFFFULL,
                  "GATHER/TENSOR_MOVE 的偏移是 uint32：张量字节数必须 < 4 GiB");
    using DataTile = pto::Tile<pto::Location::Vec, DType, 1, TileElements, pto::BLayout::RowMajor>;
    using Global = pto::global_tensor<DType, pto::RowMajor<1, CapElements>>;

    if (t.scenario == 0) {  // TENSOR_MOVE：纯连续拷贝
        Global in_g(input), out_g(output);
        using Iter = global_iterator<Global, DataTile>;
        Iter it_in(input), it_out(output);
        (void)in_g; (void)out_g;
        for (int tt = pe_id; tt < t.numTiles; tt += static_cast<int>(PeCount)) {
            int off = tt * TileElements;
            if (off + TileElements > static_cast<int>(t.numel)) {
                off = static_cast<int>(t.numel) - TileElements;
                if (off < 0) off = 0;
            }
            DataTile d;
            Global gi_in(input + off), gi_out(output + off);
            TLOAD(d, gi_in);
            TSTORE(gi_out, d);
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
        // 尾块用"重叠整块"：最后一块从 (numel - TileElements) 起，避免写到 numel 之外
        int base = tt * TileElements;
        if (base + TileElements > static_cast<int>(t.numel)) {
            base = static_cast<int>(t.numel) - TileElements;
            if (base < 0) base = 0;
        }
        TCI(lin, (std::uint32_t)base);
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
        MGATHER(dst, in_g, off);
        OutputGlobal og(output + base);
        TSTORE(og, dst);
    }
}


/**
 * @brief v3.2.1 入口：Host-tiling（参数由 `MakeV32Tiling()` 预先算好）
 *
 * 与 ops-math 原算子对齐：shape 在运行时确定，Host 侧算 tiling，
 * kernel 只按参数搬运，不使用编译期 shape 特化。
 */
template <typename DType, int CapElements, int TileElements = 512,
          std::uint32_t PeCount = supernpu::multi_thread::kDefaultPeCount>
void transpose_nd_v3_2_1_dispatch(DType *input, DType *output,
                                  const v321_detail::V32Tiling &t, int pe_id) {
    transpose_nd_v3_2_1<DType, CapElements, TileElements, PeCount>(input, output, t, pe_id);
}

}  // namespace supernpu::tile_isa
