#include <cstdint>

#include "benchmark/one-level-arch/test/common/multi_thread_res_check.h"
#include "kernel.hpp"

#ifndef HASH_WIDE_LANES
#define HASH_WIDE_LANES 32
#endif

#ifndef HASH_DATA_ELF_PATH
#define HASH_DATA_ELF_PATH "eval/perf_data/hashtable_lookup/hash_6144_PE4.elf"
#endif

#ifndef HASH_QUERY_REPEATS
#define HASH_QUERY_REPEATS 1
#endif
#define HASH_STRINGIFY_IMPL(value) #value
#define HASH_STRINGIFY(value) HASH_STRINGIFY_IMPL(value)
static_assert(HASH_QUERY_REPEATS == 1 || HASH_QUERY_REPEATS == 4);
inline constexpr unsigned kQueryCount = 6144 * HASH_QUERY_REPEATS;

using supernpu::hash_lookup::wide_mask::TableEntry;

extern "C" const unsigned hash_wide_lane_count = HASH_WIDE_LANES;
extern "C" const unsigned hash_wide_column_count = HASH_WIDE_LANES / 32;
extern "C" const unsigned hash_query_count = kQueryCount;
extern "C" const unsigned hash_table_capacity = 2550000;

extern "C" const TableEntry table[2550000];
extern "C" const int64_t queries[kQueryCount];
extern "C" const int32_t golden[kQueryCount];

// Reuse the byte-identical datasets embedded in the archived fixture ELF
// (SHA256 960ba161...). Offsets are file offsets in its first PT_LOAD.
asm(".pushsection .rodata\n"
    ".balign 4096\n"
    ".global table\n"
    ".global table_end\n"
    "table:\n"
    ".incbin \"" HASH_DATA_ELF_PATH "\", 0x1000, 0x26e8f00\n"
    // Guard: 128 entries mirroring slots 0..127, readable after slot
    // Capacity-1.  The fixture's longest probe is 63 slots, so a probe never
    // leaves this window.
    ".incbin \"" HASH_DATA_ELF_PATH "\", 0x1000, 2048\n"
    "table_end:\n"
    ".balign 4096\n"
    ".global queries\n"
    ".global queries_end\n"
    "queries:\n"
    ".rept " HASH_STRINGIFY(HASH_QUERY_REPEATS) "\n"
    ".incbin \"" HASH_DATA_ELF_PATH "\", 0x26ea000, 0xc000\n"
    ".endr\n"
    "queries_end:\n"
    ".balign 4096\n"
    ".global golden\n"
    ".global golden_end\n"
    "golden:\n"
    ".rept " HASH_STRINGIFY(HASH_QUERY_REPEATS) "\n"
    ".incbin \"" HASH_DATA_ELF_PATH "\", 0x26f6000, 0x6000\n"
    ".endr\n"
    "golden_end:\n"
    ".popsection\n");

extern "C" alignas(4096) int32_t output[kQueryCount + 2];
extern "C" MultiThreadResCheckSync sync;
extern "C" volatile unsigned errors;
extern "C" volatile unsigned checked;
alignas(4096) int32_t output[kQueryCount + 2];
MultiThreadResCheckSync sync{};
volatile unsigned errors;
volatile unsigned checked;

extern "C" __attribute__((noinline)) void wide_mask_run() {
    supernpu::hash_lookup::wide_mask::run<2550000, HASH_WIDE_LANES, 4>(
        table, queries, output + 1, kQueryCount);
}

extern "C" __attribute__((noinline)) unsigned wide_mask_check_results() {
    asm volatile(".global wide_mask_checker_load_begin\n"
                 "wide_mask_checker_load_begin:\n" ::: "memory");
    unsigned mismatchCount = 0;
    for (unsigned index = 0; index < kQueryCount; ++index) {
        if (output[index + 1] != golden[index])
            ++mismatchCount;
    }
    if (output[0] != 0x12345678 || output[kQueryCount + 1] != 0x12345678)
        ++mismatchCount;
    asm volatile(".global wide_mask_checker_load_end\n"
                 "wide_mask_checker_load_end:\n" ::: "memory");
    return mismatchCount;
}

extern "C" int main() {
    const unsigned tid = get_thread_idx();
    if (tid == 0) {
        output[0] = 0x12345678;
        output[kQueryCount + 1] = 0x12345678;
    }
    res_check_publish_inputs(sync, tid);
    wide_mask_run();
    res_check_wait_for_all(sync, tid);
    if (tid == 0) {
        errors = wide_mask_check_results();
        __asm__ volatile("" ::: "memory");
        checked = 1;
    } else {
        while (checked == 0) {
        }
    }
    __asm__ volatile("" ::: "memory");
    const unsigned mismatchCount = errors;
    asm volatile(".global wide_mask_finisher_store\n"
                 "wide_mask_finisher_store:\n" ::: "memory");
    *reinterpret_cast<volatile uint32_t *>(0x10009000ULL) =
        mismatchCount ? ((mismatchCount << 16) | 0x3333) : 0x5555;
    return 0;
}

extern "C" [[noreturn]] void _start() {
    (void)main();
    for (;;) {
        asm volatile("" ::: "memory");
    }
}
