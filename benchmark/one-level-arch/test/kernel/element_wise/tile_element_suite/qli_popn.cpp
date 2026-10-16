#include "solution/qli/qli_pto_opt_simple.hpp"
#include "benchmark.h"
#include <cstdint>
using namespace pto;

template <uint32_t Capacity, uint32_t Valid>
void qli_popn_stage(const uint32_t *input, uint32_t *remaining,
                    int32_t *indices, uint32_t count, uint32_t chunk_base) {
    using Row = qli_radix::TKey<Capacity, Valid>;
    using GlobalRow = global_tensor<uint32_t, RowMajor<1, Valid>>;
    Row candidates;
    GlobalRow source(const_cast<uint32_t *>(input));
    TLOAD(candidates, source);
    qli_radix::RadixPopN<Capacity, Valid>(candidates, chunk_base, indices, count);
    GlobalRow output(remaining);
    TSTORE(output, candidates);
}

#include "qli_popn_data.inc"
int main() {
    BENCHSTART;
#include "qli_popn_calls.inc"
    BENCHEND;
    return 0;
}
