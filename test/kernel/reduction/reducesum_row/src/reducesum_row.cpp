#include <common/pto_tileop.hpp>

#include <cstdint>
#include <cstdio>

#include "fileop.h"
//#include "reduction/reducesum_rowvec.hpp"
#include "reduction/reducesum_rowvec_single_tree_opt.hpp"



#ifndef DType
#define DType int32_t
#endif

#ifndef tMs
#define tMs 128
#endif

#ifndef tNs
#define tNs 64
#endif

#ifndef gIMs
#define gIMs 1024
#endif

#ifndef gINs
#define gINs 8192
#endif    
// ============================================================================
// main
// ============================================================================
int main() {
    using dtype = DType;

    dtype input[gIMs*gINs];
    dtype output[gIMs*1];

//    readBinaryFile("/remote/lms01/q50057645/jcore_project/JanusCoreBench/test/ascpp/reducesum_trowsum/src/data_256x256.bin", (uint8_t*)input, gIM * gIN * sizeof(dtype));
//    readBinaryFile("/remote/lms01/q50057645/jcore_project/JanusCoreBench/test/ascpp/reducesum_trowsum/src/data_1024x8192.bin", (uint8_t*)input, gIM * gIN * sizeof(dtype));
    reducesum_trowsum_rand<dtype, gIMs, gINs, tMs, tNs>(input, output);
//    writeBinaryFile("/remote/lms01/q50057645/jcore_project/JanusCoreBench/test/ascpp/reducesum_trowsum/src/result_rowsum.bin", (uint8_t*)output, gIM * sizeof(dtype));
//每个tile只有前两个位置有数。
}
