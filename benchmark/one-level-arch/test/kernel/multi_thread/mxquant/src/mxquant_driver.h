#pragma once

// Shared driver for the three mxquant test entries.
//
// The kernels are SPMD: main() runs on all four PEs and each kernel slices the
// [kTotalRows, kCols] tensor by M via get_thread_idx().  Always run with
//   gfrun -s softcore.multiThreadNum=4
//
// Two build modes, matching the library convention:
//
//   default        Run the tile kernel only.  Nothing is read or written, so
//                  this is the build to hand a cycle simulator.  The tile
//                  pipeline has no data-dependent control flow, so the
//                  zero-filled input still yields representative cycle counts.
//
//   res_check=on   PE0 reads input.bin from CHK_DIR, the scalar reference
//                  computes the golden result from that same input, and PE0
//                  exports the kernel output, the golden, and a read-back copy
//                  of the input it actually saw.  run_mxquant_check.py does the
//                  byte comparison on the host.
//
// Why the input must come from the host: device-side scalar generation of the
// input tensor was measured to be unreliable here -- stores issued by a scalar
// fill loop did not become visible to the buffer the kernel and the syscall
// path read, which silently degraded the whole test to an all-zero input where
// the golden is also all zero and any comparison passes vacuously.  Reading
// input.bin through the syscall path is coherent, so the host owns the input.
// input_readback.bin exists to keep that failure mode from ever going quiet
// again: the host asserts it matches the input.bin it generated.
//
// Buffers live in shared .bss because all four PEs address the same tensor;
// only PE0 touches files, per the 4-PE protocol in multi_thread_res_check.h.

#include <common/pto_tileop.hpp>
#include <cstdint>

#include "benchmark.h"

#ifdef RES_CHECK
#include "fileop.h"
#include "multi_thread_res_check.h"
#include "mxquant_scalar_ref.h"
#endif

namespace mxquant_test {

#ifdef RES_CHECK

template <int kTotalRows, int kCols, int kBlocksPerRow>
struct Buffers {
    alignas(4096) __bf16 input[kTotalRows * kCols];
    alignas(4096) __fp8_e4m3 output[kTotalRows * kCols];
    alignas(4096) __fp8_e8m0 scales[kTotalRows * kBlocksPerRow];
    alignas(4096) uint8_t golden_output[kTotalRows * kCols];
    alignas(4096) uint8_t golden_scales[kTotalRows * kBlocksPerRow];
};

// PE0 loads the host input and derives the golden from it; PE1..PE3 wait.
template <int kTotalRows, int kCols, int kBlocksPerRow>
inline void prepare(Buffers<kTotalRows, kCols, kBlocksPerRow> &buffers,
                    MultiThreadResCheckSync &sync, std::uint32_t tid,
                    int block) {
    if (tid == 0) {
        readBinaryFile(CHK_DIR "/input.bin",
                       reinterpret_cast<uint8_t *>(buffers.input),
                       sizeof(buffers.input));
        mxquant_ref::golden(buffers.input, buffers.golden_output,
                            buffers.golden_scales, kTotalRows, kCols, block);
    }
    res_check_publish_inputs(sync, tid);
}

// PE0 waits for every slice, exports the buffers, and runs the byte-exact
// in-simulation comparison.  Returns the failure count on PE0, 0 elsewhere.
template <int kTotalRows, int kCols, int kBlocksPerRow>
inline int finish(Buffers<kTotalRows, kCols, kBlocksPerRow> &buffers,
                  MultiThreadResCheckSync &sync, std::uint32_t tid) {
    res_check_wait_for_all(sync, tid);
    if (tid != 0) {
        return 0;
    }
    // Read-back guard: proves the kernel and the golden saw the host input.
    writeBinaryFile(CHK_DIR "/input_readback.bin",
                    reinterpret_cast<uint8_t *>(buffers.input),
                    sizeof(buffers.input));
    writeBinaryFile(CHK_DIR "/output.bin",
                    reinterpret_cast<uint8_t *>(buffers.output),
                    sizeof(buffers.output));
    writeBinaryFile(CHK_DIR "/scale_output.bin",
                    reinterpret_cast<uint8_t *>(buffers.scales),
                    sizeof(buffers.scales));
    writeBinaryFile(CHK_DIR "/golden_output.bin", buffers.golden_output,
                    sizeof(buffers.golden_output));
    writeBinaryFile(CHK_DIR "/golden_scales.bin", buffers.golden_scales,
                    sizeof(buffers.golden_scales));
    return mxquant_ref::check_result(tid, buffers.output, buffers.scales,
                                     buffers.golden_output,
                                     buffers.golden_scales, kTotalRows, kCols,
                                     kBlocksPerRow);
}

#else  // !RES_CHECK

// Kernel-only build: just the tensors the kernel itself touches.
template <int kTotalRows, int kCols, int kBlocksPerRow>
struct Buffers {
    alignas(4096) __bf16 input[kTotalRows * kCols];
    alignas(4096) __fp8_e4m3 output[kTotalRows * kCols];
    alignas(4096) __fp8_e8m0 scales[kTotalRows * kBlocksPerRow];
};

struct NoSync {};

template <int kTotalRows, int kCols, int kBlocksPerRow>
inline void prepare(Buffers<kTotalRows, kCols, kBlocksPerRow> &, NoSync &,
                    std::uint32_t, int) {}

template <int kTotalRows, int kCols, int kBlocksPerRow>
inline int finish(Buffers<kTotalRows, kCols, kBlocksPerRow> &, NoSync &,
                  std::uint32_t) {
    return 0;
}

#endif  // RES_CHECK

}  // namespace mxquant_test

#ifndef RES_CHECK
using MultiThreadResCheckSync = mxquant_test::NoSync;
#endif
