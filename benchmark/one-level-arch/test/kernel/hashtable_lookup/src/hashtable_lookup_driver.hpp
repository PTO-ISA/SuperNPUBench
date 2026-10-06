#pragma once

#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#include "basic_op/control/hashtable_lookup.hpp"

namespace hashtable_lookup_test {
constexpr std::size_t kCapacity = 65536;
constexpr std::size_t kQueryCount = 6144;
constexpr std::size_t kMaxProbe = 8;
alignas(4096) supernpu::multi_thread::HashTableEntry table[kCapacity];
alignas(4096) std::int64_t queries[kQueryCount];
alignas(4096) std::int32_t output[kQueryCount];
#ifdef RES_CHECK
MultiThreadResCheckSync res_check_sync{};
#endif

inline void prepare(std::uint32_t tid) {
#ifdef RES_CHECK
    if (tid == 0) {
        readBinaryFile(CHK_DIR "/table.bin", reinterpret_cast<uint8_t *>(table),
                       sizeof(table));
        readBinaryFile(CHK_DIR "/queries.bin",
                       reinterpret_cast<uint8_t *>(queries), sizeof(queries));
    }
    res_check_publish_inputs(res_check_sync, tid);
#else
    (void)tid;
#endif
}

inline void finish(std::uint32_t tid) {
#ifdef RES_CHECK
    res_check_wait_for_all(res_check_sync, tid);
    if (tid == 0) {
        writeBinaryFile(CHK_DIR "/output.bin",
                        reinterpret_cast<uint8_t *>(output), sizeof(output));
    }
#else
    (void)tid;
#endif
}
}  // namespace hashtable_lookup_test
