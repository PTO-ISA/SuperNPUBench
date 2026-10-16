#ifndef QLI_ELEMENT_HPP
#define QLI_ELEMENT_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <utility>

namespace qli_element {

// 将 float 的原始 U32 位模式变为可排序 key。整个原始 chunk 完成快照和
// 组装后才写回，因此 dst 与 src 完全或部分重叠时仍保持逐 chunk 语义。
template <uint32_t Capacity, uint32_t Valid = Capacity>
void make_key(uint32_t *dst, const uint32_t *src) {
    using namespace pto;
    static_assert(Capacity == 128 || Capacity == 2048);
    static_assert(Valid <= Capacity);
    if constexpr (Valid == 0)
        return;
    using Leaf = ElementTile<uint32_t, 32>;
    using Child = ElementTile<uint32_t, 128>;
    using Parent = ElementTile<uint32_t, Capacity>;
    constexpr uint32_t Parts = Capacity / 32;
    constexpr uint32_t Children = Capacity / 128;
    TileArray<Child, 1, Children> children;

    for (uint32_t child = 0; child < Children; ++child) {
        TileArray<Leaf, 1, 4> leaves;
        for (uint32_t part = 0; part < 4; ++part) {
            const uint32_t column = child * 4 + part;
            Leaf keys;
            auto &key = TPARTELEMENT(keys);
#pragma pto element for
            for (uint32_t element = 0; element < 32; ++element) {
                const uint32_t index = element * Parts + column;
                if (index < Valid) {
                    const uint32_t bits = src[index];
                    key[element] = (bits & 0x80000000u)
                        ? ~bits : (bits | 0x80000000u);
                }
            }
            TMOV(leaves[0][part], keys);
        }
        auto closed = TASSEMBLY<Child>(std::move(leaves));
        TMOV(children[0][child], closed);
    }
    auto snapshot = TASSEMBLY<Parent>(std::move(children));

    // physical128/valid32 等尾块只访问有效元素，不读取或写入 GM padding。
    auto parts = TPARTVIEW<32>(snapshot, Valid);
    for (uint32_t part = 0; part < Parts; ++part) {
        auto view = parts.part(part);
        Leaf keys;
        TMOV(keys, view);
        auto &key = TPARTELEMENT(keys);
#pragma pto element for
        for (uint32_t element = 0; element < 32; ++element) {
            const uint32_t index = element * Parts + part;
            if (index < Valid)
                dst[index] = key[element];
        }
    }
}

// 构造独立的 bucket 计数，再按原 QLI 接口输出累积直方图。
// 高位前缀由调用方的标量 radix 状态提供；重复 key 的冲突由原子加法处理。
template <uint32_t Capacity, uint32_t Valid = Capacity>
void histogram(uint32_t *hist, const uint32_t *src,
               uint32_t byte_id, uint32_t high_prefix) {
    using namespace pto;
    static_assert(Capacity == 128 || Capacity == 2048);
    static_assert(Valid <= Capacity);
    if constexpr (Valid == 0) {
        for (uint32_t base = 0; base < 256; base += 32) {
#pragma pto element for
            for (uint32_t element = 0; element < 32; ++element)
                hist[base + element] = 0;
        }
        return;
    }
    using Leaf = ElementTile<uint32_t, 32>;
    using Child = ElementTile<uint32_t, 128>;
    using Parent = ElementTile<uint32_t, Capacity>;
    constexpr uint32_t Parts = Capacity / 32;
    constexpr uint32_t Children = Capacity / 128;
    TileArray<Child, 1, Children> children;

    // 保留整个原始 chunk 的读快照，再修改直方图的 GM 存储。
    for (uint32_t child = 0; child < Children; ++child) {
        TileArray<Leaf, 1, 4> leaves;
        for (uint32_t part = 0; part < 4; ++part) {
            const uint32_t column = child * 4 + part;
            Leaf values;
            auto &value = TPARTELEMENT(values);
#pragma pto element for
            for (uint32_t element = 0; element < 32; ++element) {
                const uint32_t index = element * Parts + column;
                if (index < Valid)
                    value[element] = src[index];
            }
            TMOV(leaves[0][part], values);
        }
        auto closed = TASSEMBLY<Child>(std::move(leaves));
        TMOV(children[0][child], closed);
    }
    auto snapshot = TASSEMBLY<Parent>(std::move(children));

    for (uint32_t base = 0; base < 256; base += 32) {
#pragma pto element for
        for (uint32_t element = 0; element < 32; ++element)
            hist[base + element] = 0;
    }
    const uint32_t shift = byte_id * 8;
    const uint32_t prefix_mask = byte_id == 3 ? 0 : 0xffffffffu << (shift + 8);
    const uint32_t wanted = high_prefix & prefix_mask;
    auto parts = TPARTVIEW<32>(snapshot, Valid);
    for (uint32_t part = 0; part < Parts; ++part) {
        auto view = parts.part(part);
        Leaf values;
        TMOV(values, view);
        auto &value = TPARTELEMENT(values);
#pragma pto element for
        for (uint32_t element = 0; element < 32; ++element) {
            const uint32_t index = element * Parts + part;
            if (index < Valid && (value[element] & prefix_mask) == wanted)
                __atomic_fetch_add(&hist[(value[element] >> shift) & 255u],
                                   1u, __ATOMIC_RELAXED);
        }
    }
    // 此循环具有前缀和依赖，保持标量顺序。
    uint32_t cumulative = 0;
    for (uint32_t bucket = 0; bucket < 256; ++bucket) {
        cumulative += hist[bucket];
        hist[bucket] = cumulative;
    }
}

} // namespace qli_element

#endif
