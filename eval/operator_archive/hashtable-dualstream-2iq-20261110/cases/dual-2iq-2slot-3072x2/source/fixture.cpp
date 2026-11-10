#include <cstdint>

#include "benchmark/one-level-arch/test/common/multi_thread_res_check.h"
#include "kernel.hpp"

#ifndef HASH_WIDE_LANES
#define HASH_WIDE_LANES 32
#endif

#ifndef HASH_DATA_ELF_PATH
#define HASH_DATA_ELF_PATH "eval/perf_data/hashtable_lookup/hash_6144_PE4.elf"
#endif

// Half-fixture split for the dual-stream experiment.
//   HASH_QUERY_OFFSET   : 0 or 3072, which half of the 6144 queries this ELF owns
//   HASH_QUERY_FILE_OFF : .incbin offset of that half inside the archived data ELF
//   HASH_GOLDEN_FILE_OFF: same for the golden array
//   HASH_LOAD_BIAS      : 0 for stream A; 0x50000000 for stream B. The dual-stream
//                         loader shifts the second image's segments but performs no
//                         relocation, so every absolute data pointer below has to be
//                         biased by hand.
#ifndef HASH_QUERY_OFFSET
#define HASH_QUERY_OFFSET 0
#endif
#ifndef HASH_QUERY_FILE_OFF
#define HASH_QUERY_FILE_OFF 0x26ea000
#endif
#ifndef HASH_GOLDEN_FILE_OFF
#define HASH_GOLDEN_FILE_OFF 0x26f6000
#endif
#ifndef HASH_LOAD_BIAS
#define HASH_LOAD_BIAS 0
#endif

#define HASH_STRINGIFY_IMPL(value) #value
#define HASH_STRINGIFY(value) HASH_STRINGIFY_IMPL(value)

inline constexpr unsigned kQueryCount = 3072;
// Numeric macros so .incbin can stringify the byte counts.
#define kQuerySliceBytes 24576   /* 3072 * 8 */
#define kGoldenSliceBytes 12288  /* 3072 * 4 */

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
    ".incbin \"" HASH_DATA_ELF_PATH "\", " HASH_STRINGIFY(HASH_QUERY_FILE_OFF) ", " HASH_STRINGIFY(kQuerySliceBytes) "\n"
    "queries_end:\n"
    ".balign 4096\n"
    ".global golden\n"
    ".global golden_end\n"
    "golden:\n"
    ".incbin \"" HASH_DATA_ELF_PATH "\", " HASH_STRINGIFY(HASH_GOLDEN_FILE_OFF) ", " HASH_STRINGIFY(kGoldenSliceBytes) "\n"
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

template <typename T>
static inline T *hash_ptr(T *p) {
    return reinterpret_cast<T *>(reinterpret_cast<uintptr_t>(p) + HASH_LOAD_BIAS);
}

extern "C" __attribute__((noinline)) void wide_mask_run() {
    supernpu::hash_lookup::wide_mask::run<2550000, HASH_WIDE_LANES, 4>(
        hash_ptr(table), hash_ptr(queries), hash_ptr(output) + 1, kQueryCount);
}

extern "C" __attribute__((noinline)) unsigned wide_mask_check_results() {
    asm volatile(".global wide_mask_checker_load_begin\n"
                 "wide_mask_checker_load_begin:\n" ::: "memory");
    const int32_t *out = hash_ptr(output);
    const int32_t *ref = hash_ptr(golden);
    unsigned mismatchCount = 0;
    for (unsigned index = 0; index < kQueryCount; ++index) {
        if (out[index + 1] != ref[index])
            ++mismatchCount;
    }
    if (out[0] != 0x12345678 || out[kQueryCount + 1] != 0x12345678)
        ++mismatchCount;
    asm volatile(".global wide_mask_checker_load_end\n"
                 "wide_mask_checker_load_end:\n" ::: "memory");
    return mismatchCount;
}

extern "C" int main() {
    const unsigned tid = get_thread_idx();
    int32_t *out = hash_ptr(output);
    auto *syncp = hash_ptr(&sync);
    volatile unsigned *errp = hash_ptr(&errors);
    volatile unsigned *chkp = hash_ptr(&checked);
    if (tid == 0) {
        out[0] = 0x12345678;
        out[kQueryCount + 1] = 0x12345678;
    }
    res_check_publish_inputs(*syncp, tid);
    wide_mask_run();
    res_check_wait_for_all(*syncp, tid);
    if (tid == 0) {
        *errp = wide_mask_check_results();
        __asm__ volatile("" ::: "memory");
        *chkp = 1;
    } else {
        while (*chkp == 0) {
        }
    }
    __asm__ volatile("" ::: "memory");
    const unsigned mismatchCount = *errp;
    asm volatile(".global wide_mask_finisher_store\n"
                 "wide_mask_finisher_store:\n" ::: "memory");
    // The test finisher is a fixed address the model watches globally; both
    // streams write it on purpose (the run ends after all 8 contexts commit).
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
