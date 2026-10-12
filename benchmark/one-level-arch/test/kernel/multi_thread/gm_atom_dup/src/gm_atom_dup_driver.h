#pragma once

#include <cstdint>

#include "fileop.h"
#ifdef RES_CHECK
#include "multi_thread_res_check.h"
#endif
#include "multi_thread/topk/topk_tiled.hpp"

// Same-address GM atomic add serialization microbenchmark.
//
// Each of the four PEs owns one int32 counter word and issues `chunks`
// back-to-back full [32,1] I32 MGATHER.ADD events whose index tile is 0, so
// every lane of every chunk RMWs that one word with value 1.  A correct GM
// atom serializes duplicate effective addresses, so the observed old values
// must be a permutation of 0..chunks*32-1 and the final counter must be
// chunks*32.  topk_tiled's H[bin+1]++ slot allocator relies on exactly this.
//
// The host owns the shape: src/run_gm_atom_dup_check.py writes chunks.bin and
// compares the exported old-value stream (out.bin) and counters (counter.bin),
// so a stale ELF running the wrong chunk count cannot pass vacuously.

namespace gm_atom_dup {

using topk_tiled::I32Tile;
using topk_tiled::mgather_add_s32_m32;

constexpr int kLane = 32;
constexpr int kThreads = 4;
constexpr int kMaxChunks = 64;

struct Buffers {
    // One 32-word region per PE; the atom always addresses word 0 of it.
    alignas(4096) int32_t base[kThreads * kLane];
    alignas(4096) int32_t chunks;
    alignas(4096) int32_t out[kThreads * kMaxChunks * kLane];
};

#ifdef RES_CHECK

inline void prepare(Buffers &b, MultiThreadResCheckSync &sync, uint32_t tid) {
    if (tid == 0) {
        readBinaryFile(CHK_DIR "/chunks.bin",
                       reinterpret_cast<uint8_t *>(&b.chunks), sizeof(b.chunks));
    }
    res_check_publish_inputs(sync, tid);
}

inline void run(Buffers &b, uint32_t tid) {
    if (tid >= kThreads) return;
    const int32_t chunks = b.chunks;
    if (chunks <= 0 || chunks > kMaxChunks) return;
    int32_t *base = b.base + tid * kLane;
    int32_t *out = b.out + tid * kMaxChunks * kLane;
    base[0] = 0;
    for (int32_t c = 0; c < chunks; ++c) {
        I32Tile idx, val, old;
        TEXPANDS(idx, 0);  // all 32 lanes -> base[0]
        TEXPANDS(val, 1);
        mgather_add_s32_m32(old, base, idx, val);
        global_tensor<int32_t, RowMajor<kLane, 1>> g(out + c * kLane);
        TSTORE(g, old);
    }
}

inline int finish(Buffers &b, MultiThreadResCheckSync &sync, uint32_t tid) {
    res_check_wait_for_all(sync, tid);
    if (tid != 0) return 0;
    writeBinaryFile(CHK_DIR "/out.bin",
                    reinterpret_cast<const uint8_t *>(b.out), sizeof(b.out));
    writeBinaryFile(CHK_DIR "/counter.bin",
                    reinterpret_cast<const uint8_t *>(b.base), sizeof(b.base));
    return 0;
}

#else

struct NoSync {};

inline void prepare(Buffers &, NoSync &, uint32_t) {}
inline int finish(Buffers &, NoSync &, uint32_t) { return 0; }

#endif

}  // namespace gm_atom_dup

#ifndef RES_CHECK
using MultiThreadResCheckSync = gm_atom_dup::NoSync;
#endif
