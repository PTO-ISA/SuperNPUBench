#include "basic_op/transpose/transpose_tload.hpp"
#include "benchmark.h"
#ifdef RES_CHECK
#include "fileop.h"
#endif

#ifndef TRANSPOSE_GENERIC_CASE
#define TRANSPOSE_GENERIC_CASE 0
#endif
#ifndef TRANSPOSE_TILE_ROWS
#define TRANSPOSE_TILE_ROWS 0
#endif
#ifndef TRANSPOSE_TILE_COLS
#define TRANSPOSE_TILE_COLS 64
#endif
namespace {
#if defined(TRANSPOSE_SHAPE) && defined(TRANSPOSE_PERM)
constexpr auto shape=std::array{TRANSPOSE_SHAPE}, perm=std::array{TRANSPOSE_PERM};
#elif TRANSPOSE_GENERIC_CASE == 1
constexpr auto shape=std::array{3u,11u,5u,70u}, perm=std::array{0u,2u,1u,3u};
#elif TRANSPOSE_GENERIC_CASE == 2
constexpr auto shape=std::array{3u,5u,7u}, perm=std::array{2u,1u,0u};
#elif TRANSPOSE_GENERIC_CASE == 3
constexpr auto shape=std::array{2u,11u,3u,5u,7u}, perm=std::array{0u,3u,2u,1u,4u};
#elif TRANSPOSE_GENERIC_CASE == 4
constexpr auto shape=std::array{17u}, perm=std::array{0u};
#elif TRANSPOSE_GENERIC_CASE == 5
constexpr auto shape=std::array{19u,13u}, perm=std::array{1u,0u};
#else
constexpr auto shape=std::array{8u,8u,64u,64u}, perm=std::array{0u,2u,1u,3u};
#endif
constexpr unsigned count=[] { unsigned n=1; for(auto d:shape) n*=d; return n; }();
alignas(128) std::uint16_t input[count];
alignas(128) std::uint16_t output_storage[count+128];
auto* const output=output_storage+64;

}
int main() {
#ifdef RES_CHECK
    const auto tid = get_thread_idx();
    if (tid == 0 && !readBinaryFile(CHK_DIR "/input.bin",
            reinterpret_cast<uint8_t*>(input), sizeof(input))) return 1;
    __builtin_linx_sync_core();
#endif
    BENCHSTART;
    supernpu::multi_thread::transpose_tload<shape,perm,TRANSPOSE_TILE_ROWS,TRANSPOSE_TILE_COLS>(input,output);
    BENCHEND;
#ifdef RES_CHECK
    __builtin_linx_sync_core();
    if (tid == 0 && !writeBinaryFile(CHK_DIR "/output.bin",
            reinterpret_cast<uint8_t*>(output), sizeof(input))) return 2;
#endif
    return 0;
}
