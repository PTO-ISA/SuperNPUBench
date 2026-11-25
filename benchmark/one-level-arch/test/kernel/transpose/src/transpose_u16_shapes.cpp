#include "basic_op/transpose/transpose_u16_shapes.hpp"
#include "benchmark.h"
#ifdef RES_CHECK
#include "fileop.h"
#endif
#ifndef TRANSPOSE_SHAPE_CASE
#define TRANSPOSE_SHAPE_CASE 5
#endif
#ifndef TRANSPOSE_GATHER
#define TRANSPOSE_GATHER 0
#endif
#ifndef TRANSPOSE_CONTIGUOUS
#define TRANSPOSE_CONTIGUOUS 0
#endif
using S=supernpu::multi_thread::TransposeU16Shape<TRANSPOSE_SHAPE_CASE>;
namespace {
alignas(128) std::uint16_t input[S::count];
alignas(128) std::uint16_t output_storage[S::count+128];
auto* const output=output_storage+64;
#if TRANSPOSE_SHAPE_CASE == 1
constexpr unsigned shape[]={129,22,120},perm[]={2,1,0};
#elif TRANSPOSE_SHAPE_CASE == 2
constexpr unsigned shape[]={8,4096,3},perm[]={0,2,1};
#elif TRANSPOSE_SHAPE_CASE == 3
constexpr unsigned shape[]={1476,32},perm[]={1,0};
#elif TRANSPOSE_SHAPE_CASE == 4
constexpr unsigned shape[]={8,64,4,16,7},perm[]={0,3,2,1,4};
#else
constexpr unsigned shape[]={8,8,64,64},perm[]={0,2,1,3};
#endif
}
int main() {
#ifdef RES_CHECK
    const auto tid = get_thread_idx();
    if (tid == 0 && !readBinaryFile(CHK_DIR "/input.bin",
            reinterpret_cast<uint8_t*>(input), sizeof(input))) return 1;
    __builtin_linx_sync_core();
#endif
    BENCHSTART;
    supernpu::multi_thread::transpose_u16_shapes<TRANSPOSE_SHAPE_CASE,TRANSPOSE_GATHER,TRANSPOSE_CONTIGUOUS>(input,output);
    BENCHEND;
#ifdef RES_CHECK
    __builtin_linx_sync_core();
    if (tid == 0 && !writeBinaryFile(CHK_DIR "/output.bin",
            reinterpret_cast<uint8_t*>(output), sizeof(input))) return 2;
#endif
    return 0;
}
