#ifndef HASHTABLE_LOOKUP_SIMD_HPP
#define HASHTABLE_LOOKUP_SIMD_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <utility>

struct TableEntry {
    int64_t key;
    int32_t value;
    int32_t padding;
};
static_assert(sizeof(TableEntry) == 16);
constexpr int32_t kNotFound = -1;

// MurmurHash3_x86_32：按低、高两个 32-bit 字依次混合，所有算术按 U32 回绕。
inline uint32_t hashTableKey(int64_t key) {
    const uint64_t bits = static_cast<uint64_t>(key);
    uint32_t low = static_cast<uint32_t>(bits);
    uint32_t high = static_cast<uint32_t>(bits >> 32);
    low *= 0xcc9e2d51u;
    low = (low << 15) | (low >> 17);
    low *= 0x1b873593u;
    uint32_t hash = (low << 13) | (low >> 19);
    hash = hash * 5u + 0xe6546b64u;
    high *= 0xcc9e2d51u;
    high = (high << 15) | (high >> 17);
    hash ^= high * 0x1b873593u;
    hash = (hash << 13) | (hash >> 19);
    hash = hash * 5u + 0xe6546b64u;
    hash ^= 8u;
    hash ^= hash >> 16;
    hash *= 0x85ebca6bu;
    hash ^= hash >> 13;
    hash *= 0xc2b2ae35u;
    return hash ^ (hash >> 16);
}

template <int kTileRows, int kTileCols, int kCap, int kMaxProbe>
void runHashFind(int32_t *out, TableEntry *table, int64_t *queries) {
    using namespace pto;
    constexpr uint32_t Count = kTileRows * kTileCols;
    static_assert(Count == 256 || Count == 512 || Count == 1024,
                  "hashtable original batches contain 256, 512 or 1024 keys");
    static_assert(kCap > 0 && kCap <= INT32_MAX / sizeof(TableEntry));
    static_assert(kMaxProbe > 0);
    using KeyParent = ElementTile<int64_t, Count>;
    using ResultParent = ElementTile<int32_t, Count>;
    using ResultLeaf = ElementTile<int32_t, 32>;

    // 整批查询先快照；所有结果组装完成后再写回，保留输入/输出别名语义。
    KeyParent query_snapshot;
    TLOAD(query_snapshot, queries, Count);
    auto query_parts = TPARTVIEW<32>(query_snapshot, Count);
    constexpr uint32_t ChildCount = Count < 512 ? Count : 512;
    constexpr uint32_t Groups = Count / ChildCount;
    constexpr uint32_t Parts = ChildCount / 32;
    using ResultChild = ElementTile<int32_t, ChildCount>;
    TileArray<ResultChild, 1, Groups> result_groups;

    for (uint32_t group = 0; group < Groups; ++group) {
        TileArray<ResultLeaf, 1, Parts> result_parts;
        for (uint32_t part = 0; part < Parts; ++part) {
            auto query_part = query_parts.part(group * Parts + part);
            ElementTile<int64_t, 32> keys;
            TMOV(keys, query_part);
            auto &key = TPARTELEMENT(keys);
            ElementTile<uint32_t, 32> slots;
            auto &slot = TPARTELEMENT(slots);
            ResultLeaf result;
            auto &value = TPARTELEMENT(result);
#pragma pto element for
            for (uint32_t element = 0; element < 32; ++element) {
                slot[element] = hashTableKey(key[element]) % uint32_t(kCap);
                value[element] = kNotFound;
            }

            // 固定探测次数；重复键时保留最后一次匹配，未匹配保持 -1。
            for (uint32_t probe = 0; probe < uint32_t(kMaxProbe); ++probe) {
#pragma pto element for
                for (uint32_t element = 0; element < 32; ++element) {
                    const uint32_t position = slot[element];
                    const int64_t candidate = table[position].key;
                    const int32_t candidate_value = table[position].value;
                    if (candidate == key[element])
                        value[element] = candidate_value;
                    const uint32_t next = position + 1;
                    slot[element] = next == uint32_t(kCap) ? 0u : next;
                }
            }
            TMOV(result_parts[0][part], result);
        }
        auto closed = TASSEMBLY<ResultChild>(std::move(result_parts));
        if constexpr (Groups == 1)
            TSTORE(out, closed, Count);
        else
            TMOV(result_groups[0][group], closed);
    }
    if constexpr (Groups > 1) {
        // 1024 元素先关闭两个 512 元素子块，每代不超过 16 个 writer。
        auto result = TASSEMBLY<ResultParent>(std::move(result_groups));
        TSTORE(out, result, Count);
    }
}

template <int kTileRows, int kTileCols, int kCap, int kMaxProbe>
void LaunchHashFind(int32_t *out, TableEntry *table, int64_t *queries) {
    runHashFind<kTileRows, kTileCols, kCap, kMaxProbe>(out, table, queries);
}

#endif // HASHTABLE_LOOKUP_SIMD_HPP
