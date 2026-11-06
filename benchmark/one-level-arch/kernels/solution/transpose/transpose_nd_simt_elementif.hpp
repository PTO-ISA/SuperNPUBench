#pragma once

/**
 * @file transpose_nd_simt_elementif.hpp
 * @brief 对标 ops-math transpose 的 SMALL_SHAPE(#10001, SIMT) —— 用 element-if 实现
 *
 * ops-math 的 SMALL_SHAPE：总字节 < 阈值时走 **SIMT**（逐"线程"算输出元素对应的输入
 * 地址，直读直写）。本 ISA 无 SIMT 线程模型，这里用 **element-if**
 * （`#pragma pto element for` + `ElementTile/TPARTVIEW/TPARTELEMENT`）表达同一语义：
 * 在 Tile 的逻辑元素上写普通 C++（逐元素算输入下标 + 索引 gather + runtime tail 谓词），
 * 由 Linx clang/LLVM 整体降低为 Tile 指令 + execution mask。
 *
 * 写法对标上游 SuperNPUBench PR#202（已 merged）`basic_op/transpose/detail/transpose.hpp`
 * 的 element-if 版实现（该版本已在上游验收过）。
 *
 * 编译要求（关键）：
 *   - Linx LLVM 需含 element-if（#118/#131，dev-llvm15_56 @ >=084bbc31）；
 *   - `-mllvm -linxv5-enable-generic-element=true` 且 `-O2`；
 *   - logical32 profile（S32/U32/F32）按 32 元素分片。
 */

#include <common/pto_tileop.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {

/**
 * @brief element-if 版通用 permute 转置（等价 ops-math SMALL_SHAPE / SIMT）。
 *
 * 输出 tile（TileElements）拆成 32 元素 logical32 分片；分片内逐元素普通 C++ 计算输入
 * 下标（按输出 stride/shape 坐标分解）后索引 gather 到 ElementTile；尾块用 runtime tail
 * 谓词 `element < valid`。最后 TASSEMBLY + TSTORE。
 */
template <typename DType, int Rank, int Elements, int TileElements,
          std::array<int, 8> Dims, std::array<int, 8> Perm>
void transpose_nd_simt_elementif_tile(DType *input, DType *output, int tile_start,
                                      int tile_stride, bool handle_tail) {
    constexpr int kLeaf = 32;  // logical32 分片元素数
    static_assert(TileElements % kLeaf == 0, "TileElements 须为 32 的整数倍");
    constexpr int kParts = TileElements / kLeaf;
    constexpr int kFull = Elements / TileElements;
    constexpr int kTail = Elements % TileElements;

    // 逆置换 + 编译期输入/输出 stride
    constexpr auto kInv = [](std::array<int, 8> p) {
        std::array<int, 8> inv{};
        for (int i = 0; i < Rank; ++i) inv[p[i]] = i;
        return inv;
    }(Perm);
    constexpr std::array<std::uint32_t, Rank> kInStride = [] {
        std::array<std::uint32_t, Rank> s{};
        std::uint32_t acc = 1;
        for (int d = Rank - 1; d >= 0; --d) { s[d] = acc; acc *= static_cast<std::uint32_t>(Dims[d]); }
        return s;
    }();
    constexpr std::array<std::uint32_t, Rank> kOutStride = [] {
        std::array<std::uint32_t, Rank> s{};
        std::uint32_t acc = 1;
        for (int d = Rank - 1; d >= 0; --d) { s[d] = acc; acc *= static_cast<std::uint32_t>(Dims[Perm[d]]); }
        return s;
    }();

    for (int t = tile_start; t < kFull; t += tile_stride) {
        pto::ElementTile<DType, TileElements> layout;
        auto parts = pto::TPARTVIEW<kLeaf>(layout, TileElements);
        pto::TileArray<pto::ElementTile<DType, kLeaf>, 1, kParts> gathered;
        const std::uint32_t base = static_cast<std::uint32_t>(t) * TileElements;
        for (int part = 0; part < kParts; ++part) {
            pto::ElementTile<DType, kLeaf> leaf;
            auto &vals = pto::TPARTELEMENT(leaf);
#pragma pto element for
            for (std::uint32_t e = 0; e < kLeaf; ++e) {
                const std::uint32_t o = base + parts.logical_index(part, e);
                std::uint32_t in_idx = 0;
                #pragma clang loop unroll(full)
                for (int d = 0; d < Rank; ++d) {
                    const std::uint32_t coord =
                        (o / kOutStride[d]) % static_cast<std::uint32_t>(Dims[Perm[d]]);
                    in_idx += coord * kInStride[kInv[d]];
                }
                vals[e] = input[in_idx];
            }
            TMOV(gathered[0][part], leaf);
        }
        auto snapshot =
            pto::TASSEMBLY<pto::ElementTile<DType, TileElements>>(std::move(gathered));
        TSTORE(output + base, snapshot, TileElements);
    }

    if constexpr (kTail != 0) {
        if (handle_tail) {
            pto::ElementTile<DType, TileElements> layout;
            auto parts = pto::TPARTVIEW<kLeaf>(layout, kTail);
            pto::TileArray<pto::ElementTile<DType, kLeaf>, 1, kParts> gathered;
            const std::uint32_t base = static_cast<std::uint32_t>(kFull) * TileElements;
            for (int part = 0; part < kParts; ++part) {
                pto::ElementTile<DType, kLeaf> leaf;
                auto &vals = pto::TPARTELEMENT(leaf);
                const std::uint32_t valid = static_cast<std::uint32_t>(parts.valid_size(part));
#pragma pto element for
                for (std::uint32_t e = 0; e < kLeaf; ++e) {
                    if (e < valid) {
                        const std::uint32_t o = base + parts.logical_index(part, e);
                        std::uint32_t in_idx = 0;
                        #pragma clang loop unroll(full)
                        for (int d = 0; d < Rank; ++d) {
                            const std::uint32_t coord =
                                (o / kOutStride[d]) % static_cast<std::uint32_t>(Dims[Perm[d]]);
                            in_idx += coord * kInStride[kInv[d]];
                        }
                        vals[e] = input[in_idx];
                    } else {
                        vals[e] = static_cast<DType>(0);
                    }
                }
                TMOV(gathered[0][part], leaf);
            }
            auto snapshot =
                pto::TASSEMBLY<pto::ElementTile<DType, TileElements>>(std::move(gathered));
            auto out_parts = pto::TPARTVIEW<kLeaf>(snapshot, kTail);
            for (int part = 0; part < kParts; ++part) {
                pto::ElementTile<DType, kLeaf> stored;
                TMOV(stored, out_parts.part(part));
                TSTORE(output + base, stored, out_parts, part, out_parts.valid_size(part));
            }
        }
    }
}

}  // namespace supernpu::tile_isa
