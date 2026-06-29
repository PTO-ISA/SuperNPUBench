#include <common/pto_tileop.hpp>

#include <cstdint>

#include "fileop.h"
#include "reduction/reducesum_colvec.hpp"


#ifndef DType
#define DType int32_t
#endif

#ifndef gIMs
#define gIMs 8192
#endif

#ifndef gINs
#define gINs 1024
#endif


#ifndef tMs
#define tMs 32
#endif

#ifndef tNs
#define tNs 128
#endif




// ============================================================================
// main
// ============================================================================
int main() {
    using dtype = DType;

    dtype input_buf[gIMs*gINs];
//    dtype zero_buf[1*gIN];    
    dtype output_buf[1*gINs];

    dtype* input=input_buf;
//    dtype* zero=zero_buf;    
    dtype* output=output_buf;    

//    readBinaryFile("/remote/lms01/q50057645/jcore_project/JanusCoreBench/test/ascpp/reducesum_colsum/src/data_256x256.bin", (uint8_t*)input, gIM * gIN * sizeof(dtype));
//    readBinaryFile("/remote/lms01/q50057645/jcore_project/JanusCoreBench/test/ascpp/reducesum_colsum/src/data_8192x1024.bin", (uint8_t*)input, gIM * gIN * sizeof(dtype));
//    readBinaryFile("/remote/lms01/q50057645/jcore_project/JanusCoreBench/test/ascpp/reducesum_colsum/src/data1x256_zero.bin", (uint8_t*)zero, gIN * sizeof(dtype));    
//    reducesum_colsum_rand<dtype, gIM, gIN, tN>(input, output);
    reducesum_colsum_rand<dtype, gIMs, gINs, tMs, tNs>(input, output);
//    writeBinaryFile("/remote/lms01/q50057645/jcore_project/JanusCoreBench/test/ascpp/reducesum_colsum/src/result.bin", (uint8_t*)output, gIN * sizeof(dtype));

}

