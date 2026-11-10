#include <cstdint>
#ifdef RES_CHECK
#include <cstdio>
#endif
#include "multi_thread_res_check.h"
#include "basic_op/hashtable_lookup/hashtable_lookup.hpp"

#ifndef HASH_PE_COUNT
#define HASH_PE_COUNT 4
#endif
static_assert(HASH_PE_COUNT == 1 || HASH_PE_COUNT == 4);

#ifndef HASH_CAPACITY
#define HASH_CAPACITY 37
#endif
#ifndef HASH_FULL_TABLE
#define HASH_FULL_TABLE 0
#endif
namespace {
using supernpu::hash_lookup::TableEntry;
constexpr unsigned capacity = HASH_CAPACITY, count = 139;
#ifndef HASH_TABLE_GUARD
#define HASH_TABLE_GUARD 112
#endif
constexpr unsigned kGuardEntries = HASH_TABLE_GUARD / sizeof(TableEntry);
alignas(4096) TableEntry table[capacity + kGuardEntries];
alignas(4096) int64_t queries[count];
alignas(4096) int32_t output[count + 2];
#ifdef RES_CHECK
int32_t golden[count];
#endif
MultiThreadResCheckSync sync{};
uint32_t rot(uint32_t x, unsigned r) { return (x << r) | (x >> (32-r)); }
uint32_t hash(int64_t key) {
    uint64_t bits = static_cast<uint64_t>(key);
    uint32_t h = 0;
    for (unsigned j = 0; j < 2; ++j) {
        uint32_t k = static_cast<uint32_t>(bits >> (j*32));
        k *= 0xcc9e2d51u; k = rot(k, 15); k *= 0x1b873593u;
        h ^= k; h = rot(h, 13); h = h*5 + 0xe6546b64u;
    }
    h ^= 8; h ^= h>>16; h *= 0x85ebca6bu;
    h ^= h>>13; h *= 0xc2b2ae35u; h ^= h>>16;
    return h;
}
// Mixed signs and identical low halves exercise full 64-bit equality/hash.
int64_t make_key(unsigned n) {
    return static_cast<int64_t>((uint64_t(n ^ 0x80000000u) << 32) | (n % 3));
}
void prepare() {
    for (auto &entry : table) entry = {-1, -1, 0};
    constexpr unsigned inserted = HASH_FULL_TABLE ? capacity : capacity*4/5;
    for (unsigned n = 0; n < inserted; ++n) {
        auto key = make_key(n);
        unsigned slot = hash(key) % capacity;
        while (table[slot].key != -1) slot = (slot + 1) % capacity;
        table[slot] = {key, static_cast<int32_t>(n*17 + 3), 0};
    }
    // Guard tail required by the default kernel: mirror slots 0..2.
    for (unsigned i = 0; i < kGuardEntries; ++i) table[capacity + i] = table[i];
    for (unsigned i = 0; i < count; ++i) {
        queries[i] = i % 7 == 0 ? -1 : make_key(i % (capacity + 7));
#ifdef RES_CHECK
        // Independent oracle: scan entries, not the device probing algorithm.
        golden[i] = -1;
        for (unsigned s = 0; s < capacity; ++s)
            if (table[s].key != -1 && table[s].key == queries[i]) golden[i] = table[s].value;
#endif
    }
    for (auto &x : output) x = 0x12345678;
}
}
extern "C" int main() {
    unsigned tid = get_thread_idx();
    if (tid == 0) prepare();
    res_check_publish_inputs(sync, tid);
    supernpu::multi_thread::hashtable_lookup<capacity, HASH_PE_COUNT>(table, queries, output+1, count);
#if HASH_PE_COUNT == 4
    res_check_wait_for_all(sync, tid);
#endif
#ifdef RES_CHECK
    if (tid == 0) {
        unsigned errors = 0;
        for (unsigned i = 0; i < count; ++i) {
            if (output[i+1] != golden[i]) {
                if (errors < 8) printf("mismatch %u: got %d expected %d\n", i, output[i+1], golden[i]);
                ++errors;
            }
        }
        if (output[0] != 0x12345678 || output[count+1] != 0x12345678) ++errors;
        printf("hash lookup %s: capacity=%u full=%u queries=%u errors=%u\n",
               errors ? "FAIL" : "PASS", capacity, HASH_FULL_TABLE, count, errors);
        return errors ? 1 : 0;
    }
#endif
    return 0;
}
