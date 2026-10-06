#include "hashtable_lookup_driver.hpp"

int main() {
    using namespace hashtable_lookup_test;
    const std::uint32_t tid = get_thread_idx();
    prepare(tid);
    BENCHSTART;
    supernpu::multi_thread::hashtable_lookup_simd<
        kCapacity, kMaxProbe, kQueryCount>(table, queries, output);
    BENCHEND;
    finish(tid);
    return 0;
}
