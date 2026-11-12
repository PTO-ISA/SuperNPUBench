#include <cstdint>
#ifndef HASH_KERNEL_HEADER
#define HASH_KERNEL_HEADER "basic_op/hashtable_lookup/hashtable_lookup.hpp"
#endif
#include HASH_KERNEL_HEADER
#if HASH_VERIFY
#include "multi_thread_res_check.h"
#endif

#ifndef HASH_VERIFY
#error "Select hash_mode=perf or hash_mode=verify"
#endif
#ifndef HASH_DATA_DIR
#define HASH_DATA_DIR "data"
#endif

using supernpu::hash_lookup::TableEntry;
extern "C" const TableEntry table[2550000];
extern "C" const int64_t queries[6144];
// Adaptive 2+6 default: the B phase reads up to slot o+7, so the guard tail
// must provide 7 entries (112 bytes).  Use 48 with HASH_ADAPTIVE_TAIL=2 or
// the stride4 implementation.
#ifndef HASH_TABLE_GUARD
#define HASH_TABLE_GUARD 112
#endif
#if HASH_TABLE_GUARD > 0
#define HASH_STR_(x) #x
#define HASH_STR(x) HASH_STR_(x)
#endif
asm(".pushsection .rodata\n.balign 4096\n.global table\ntable:\n"
    ".incbin \"" HASH_DATA_DIR "/table_data_70pct.bin\"\n"
#if HASH_TABLE_GUARD > 0
    // Guard entries mirror slots 0.. so a chunk probe may read past slot
    // Capacity-1 instead of wrapping every derived offset.
    ".incbin \"" HASH_DATA_DIR "/table_data_70pct.bin\", 0, " HASH_STR(HASH_TABLE_GUARD) "\n"
#endif
    ".popsection\n");
asm(".pushsection .rodata\n.balign 4096\n.global queries\nqueries:\n"
    ".incbin \"" HASH_DATA_DIR "/input_keys_data.bin\"\n.popsection\n");
alignas(4096) int32_t output[6146];
#if !HASH_VERIFY && HASH_PERF_BARRIER
alignas(64) volatile unsigned perf_done[4];
#endif

#if HASH_VERIFY
extern "C" const int32_t golden[6144];
asm(".pushsection .rodata\n.balign 4096\n.global golden\ngolden:\n"
    ".incbin \"" HASH_DATA_DIR "/lookup_golden.bin\"\n.popsection\n");
MultiThreadResCheckSync sync{};
volatile unsigned errors;
volatile unsigned checked;
#endif

extern "C" int main() {
#if HASH_VERIFY
    const unsigned tid = get_thread_idx();
    if (tid == 0) {
        output[0] = 0x12345678;
        output[6145] = 0x12345678;
    }
    res_check_publish_inputs(sync, tid);
#endif
    supernpu::multi_thread::hashtable_lookup<2550000, 4>(table, queries, output + 1, 6144);
#ifdef HASH_NEG_CONTROL
    // Negative control: corrupt one result so the in-ELF comparator must
    // report a mismatch and the finisher must not see status 0x5555.
    if (get_thread_idx() == 0) output[1] ^= 1;
#endif
#if HASH_VERIFY
    res_check_wait_for_all(sync, tid);
    if (tid == 0) {
        unsigned n = 0;
        for (unsigned i = 0; i < 6144; ++i)
            if (output[i + 1] != golden[i]) ++n;
        if (output[0] != 0x12345678 || output[6145] != 0x12345678) ++n;
        errors = n;
        asm volatile("" ::: "memory");
        checked = 1;
    } else {
        while (checked == 0) {}
    }
    asm volatile("" ::: "memory");
    const unsigned n = errors;
    const uint32_t status = n ? ((n << 16) | 0x3333) : 0x5555;
#else
#if HASH_PERF_BARRIER
    const unsigned tid = get_thread_idx();
    asm volatile("" ::: "memory");
    perf_done[tid] = 1;
    for (unsigned pe=0; pe<4; ++pe) while (perf_done[pe]==0) {}
    asm volatile("" ::: "memory");
#endif
    // Completion marker only. perf mode does not claim numerical correctness.
    const uint32_t status = 0x5555;
#endif
    *reinterpret_cast<volatile uint32_t *>(0x10009000ULL) = status;
    return 0;
}
