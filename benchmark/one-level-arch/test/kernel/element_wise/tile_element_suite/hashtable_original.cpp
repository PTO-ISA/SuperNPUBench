#include "basic_op/control/hashtable_lookup_simd.hpp"
#include "benchmark.h"

#ifndef HASHTABLE_COLUMNS
#define HASHTABLE_COLUMNS 256
#endif
#ifndef HASHTABLE_QUERY_ALIAS
#define HASHTABLE_QUERY_ALIAS 0
#endif
constexpr unsigned kCapacity = 2550000;
constexpr unsigned kQueries = 6144;
constexpr unsigned kProbes = 512;
static_assert(HASHTABLE_COLUMNS == 256 || HASHTABLE_COLUMNS == 512 ||
              HASHTABLE_COLUMNS == 1024);
static_assert(kQueries % HASHTABLE_COLUMNS == 0);

extern "C" {
extern TableEntry hashtable_table[kCapacity];
extern int64_t hashtable_queries[kQueries];
extern int32_t hashtable_output[kQueries];
}
// The fixture generator writes exact original allocations and adjacent guards.
#include "hashtable_data.inc"

int main() {
    int32_t *output = HASHTABLE_QUERY_ALIAS
        ? reinterpret_cast<int32_t *>(hashtable_queries) + 1
        : hashtable_output;
    BENCHSTART;
    for (unsigned base = 0; base < kQueries; base += HASHTABLE_COLUMNS)
        LaunchHashFind<1, HASHTABLE_COLUMNS, kCapacity, kProbes>(
            output + base, hashtable_table, hashtable_queries + base);
    BENCHEND;
    return 0;
}
